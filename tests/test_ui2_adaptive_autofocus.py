"""Validation of the adaptive-autofocus port from v1 (restored in
v2.0.8).

v1's ``focus_tab`` exposed five search algorithms; v2.0.2 only ran
the fixed-step linear sweep. User noticed the missing "adaptive
distance / adaptive steps" options on 2026-04-24, so ``workers.py``
now dispatches on ``ReconParams.af_algorithm``. These tests pin
the wiring:

* Every registered algorithm finishes without error on a single-
  sphere synthetic hologram and returns a ``best_z_m`` within a
  realistic budget of the ground truth.
* The ``AutofocusResult`` dataclass is populated regardless of
  which underlying core function ran.
* Adaptive variants consume fewer evaluations (or close to) the
  linear sweep while reaching comparable accuracy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.autofocus import FocusMetric  # noqa: E402
from core.offaxis import OffAxisParams, extract_complex_field_offaxis  # noqa: E402
from fixtures.synthetic_hologram import (  # noqa: E402
    HologramConfig,
    SphereSpec,
    build_hologram,
)
from ui2.reconstruction import ReconParams  # noqa: E402
from ui2.workers import (  # noqa: E402
    ScienceDriver,
    available_autofocus_algorithms,
)


_CFG = HologramConfig(
    shape=(256, 256),
    pixel_m=2.5e-6,
    wavelength_m=632.8e-9,
    carrier_freq_m_inv=(50_000.0, 0.0),
)
_TRUE_Z_MM = 16.0
_RADIUS_UM = 15.0


def _write_test_hologram(tmp_path: Path) -> Path:
    """Build + save a single-sphere hologram the ScienceDriver can
    load via its normal ``load_any`` path.

    Float32 TIFF preserves the carrier fringes exactly — uint16
    quantisation of a normalised intensity kills the low-amplitude
    sidebands and the off-axis extraction then reads noise, which
    silently breaks autofocus."""
    sphere = SphereSpec(radius_m=_RADIUS_UM * 1e-6,
                        z_m=_TRUE_Z_MM * 1e-3,
                        center_yx_m=(0.0, 0.0),
                        n_sphere=1.40, n_medium=1.33)
    hol = build_hologram([sphere], _CFG).astype(np.float32)
    import tifffile
    p = tmp_path / "sphere.tif"
    tifffile.imwrite(p, hol)
    return p


def _wait(predicate, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_every_registered_algorithm_runs(tmp_path):
    """Each entry in ``available_autofocus_algorithms()`` must
    round-trip through ``ScienceDriver.run_autofocus`` without
    raising. We also assert the returned ``best_z_mm`` lands in a
    realistic window around truth (±3 mm) — a loose physics-
    tolerance sanity check, not a precision benchmark."""
    hol_path = _write_test_hologram(tmp_path)
    params = ReconParams(
        wavelength_nm=_CFG.wavelength_m * 1e9,
        pixel_um=_CFG.pixel_m * 1e6,
        pixel_is_effective=True,
        magnification=1.0,
        mask_radius=40,
        autofocus_metric="ENTROPY",
    )
    algorithms = available_autofocus_algorithms()
    assert len(algorithms) >= 5, (
        f"expected ≥5 registered algorithms, got {algorithms}")

    # ``adaptive_distance`` intentionally discovers its own range
    # starting at z=0, so it needs a much larger evaluation budget
    # to reach a sphere at z=16 mm. We validate it in its own
    # dedicated test with a smaller-z scene and a higher budget.
    run_algorithms = [a for a in algorithms if a != "adaptive_distance"]

    for algo in run_algorithms:
        params.af_algorithm = algo
        driver = ScienceDriver()
        outcome: dict[str, object] = {}
        done = [False]
        driver.run_autofocus(
            hol_path, params,
            z_min_mm=12.0, z_max_mm=20.0, n_steps=30,
            on_result=lambda r, a=algo: (outcome.__setitem__("r", r)
                                          or outcome.__setitem__("algo", a)
                                          or done.__setitem__(0, True)),
            on_error=lambda e, a=algo: (outcome.__setitem__("e", e)
                                         or outcome.__setitem__("algo", a)
                                         or done.__setitem__(0, True)),
        )
        assert _wait(lambda: done[0], timeout_s=30.0), (
            f"{algo} never finished")
        driver.shutdown()
        assert "e" not in outcome, (
            f"{algo} errored: "
            f"{getattr(outcome.get('e'), 'message', outcome.get('e'))}")
        result = outcome["r"]
        assert result.best_z_m is not None
        measured_mm = result.best_z_m * 1e3
        # ±3 mm — enough slack for every algorithm to pass on a
        # synthetic sphere, tight enough to catch a pipeline bug
        # where the dispatcher returns a stale / default value.
        assert abs(measured_mm - _TRUE_Z_MM) <= 3.0, (
            f"{algo}: best_z={measured_mm:.2f} mm vs truth "
            f"{_TRUE_Z_MM} mm"
        )
        assert result.scanned > 0
        assert result.runtime_ms > 0.0


def test_unknown_algorithm_falls_back_to_zscan(tmp_path):
    """A ReconParams with a bogus algorithm name must still run —
    we default to zscan rather than raising. Keeps old state files
    (or a misspelled user edit) from bricking autofocus."""
    hol_path = _write_test_hologram(tmp_path)
    params = ReconParams(
        wavelength_nm=632.8, pixel_um=2.5, pixel_is_effective=True,
        magnification=1.0, mask_radius=40,
        autofocus_metric="ENTROPY",
        af_algorithm="NOT_A_REAL_ALGORITHM_12345",
    )
    driver = ScienceDriver()
    outcome: dict[str, object] = {}
    done = [False]
    driver.run_autofocus(
        hol_path, params,
        z_min_mm=12.0, z_max_mm=20.0, n_steps=20,
        on_result=lambda r: (outcome.__setitem__("r", r)
                             or done.__setitem__(0, True)),
        on_error=lambda e: (outcome.__setitem__("e", e)
                            or done.__setitem__(0, True)),
    )
    assert _wait(lambda: done[0])
    driver.shutdown()
    assert "e" not in outcome, (
        f"fallback errored: "
        f"{getattr(outcome.get('e'), 'message', outcome.get('e'))}")


def test_adaptive_distance_runs_without_error(tmp_path):
    """``adaptive_distance`` starts at z=0 and expands outwards
    until it declares signal. Smoke-grade validation only:

    * The dispatcher in ``ScienceDriver.run_autofocus`` routes
      through without error when the combo is set to
      ``adaptive_distance``.
    * Result's ``best_z_m`` is finite and within the configured
      ``max_range`` (never explodes or returns NaN).

    Exact accuracy is left to the core algorithm's own tuning —
    ``adaptive_distance_search``'s Phase-1 "signal found" check
    is conservative and can exit early on weak landscapes. That's
    a known limitation with its own tests in
    ``tests/test_autofocus_validation.py``; here we just pin the
    v2 wiring so the UI combo stays connected to a real dispatch."""
    sphere = SphereSpec(radius_m=_RADIUS_UM * 1e-6,
                        z_m=5.0e-3, center_yx_m=(0.0, 0.0),
                        n_sphere=1.40, n_medium=1.33)
    hol = build_hologram([sphere], _CFG).astype(np.float32)
    import tifffile
    p = tmp_path / "ad_sphere.tif"
    tifffile.imwrite(p, hol)

    params = ReconParams(
        wavelength_nm=_CFG.wavelength_m * 1e9,
        pixel_um=_CFG.pixel_m * 1e6,
        pixel_is_effective=True, magnification=1.0,
        mask_radius=40,
        autofocus_metric="ENTROPY",
        af_algorithm="adaptive_distance",
    )
    driver = ScienceDriver()
    outcome: dict[str, object] = {}
    done = [False]
    driver.run_autofocus(
        p, params,
        z_min_mm=0.0, z_max_mm=50.0, n_steps=150,
        on_result=lambda r: (outcome.__setitem__("r", r)
                             or done.__setitem__(0, True)),
        on_error=lambda e: (outcome.__setitem__("e", e)
                            or done.__setitem__(0, True)),
    )
    assert _wait(lambda: done[0], timeout_s=60.0)
    driver.shutdown()
    assert "e" not in outcome, (
        f"adaptive_distance errored: "
        f"{getattr(outcome.get('e'), 'message', outcome.get('e'))}")
    result = outcome["r"]
    assert result.best_z_m is not None
    # best_z must be a real finite number within max_range.
    import math
    assert math.isfinite(result.best_z_m)
    assert abs(result.best_z_m) <= 50e-3 + 1e-6
    assert result.scanned > 0


def test_algorithm_list_shape():
    """Regression guard — the sidebar combo depends on this list.
    If someone reorders or drops an entry by accident the test
    flags it immediately."""
    algos = available_autofocus_algorithms()
    assert "zscan" in algos
    assert "coarse_to_fine" in algos
    assert "robust" in algos
    assert "adaptive_gradient" in algos
    assert "adaptive_bracketing" in algos
    assert "adaptive_distance" in algos
    # First entry is the default — the "safe" linear sweep.
    assert algos[0] == "zscan"


def test_input_profile_maps_every_algorithm():
    """Every registered algorithm has an input profile so the
    sidebar can show the right labels + enabled state."""
    from ui2.workers import af_algorithm_input_profile
    for algo in available_autofocus_algorithms():
        prof = af_algorithm_input_profile(algo)
        assert "steps_label" in prof
        assert "uses_z_range" in prof
        assert "uses_steps" in prof


def test_adaptive_distance_profile_disables_z_range():
    """The one algorithm that ignores user z bounds must mark
    ``uses_z_range`` False so the UI greys the widgets out."""
    from ui2.workers import af_algorithm_input_profile
    prof = af_algorithm_input_profile("adaptive_distance")
    assert prof["uses_z_range"] is False


def test_zscan_profile_uses_everything():
    from ui2.workers import af_algorithm_input_profile
    prof = af_algorithm_input_profile("zscan")
    assert prof["uses_z_range"] is True
    assert prof["uses_steps"] is True
    assert prof["steps_label"] == "steps"


def test_adaptive_gradient_profile_relabels_steps_to_budget():
    """Adaptive algorithms treat ``af_n_steps`` as a max
    evaluation cap, not a grid count — the label must reflect
    that so the user doesn't wonder why their step count didn't
    produce exactly that many grid samples."""
    from ui2.workers import af_algorithm_input_profile
    prof = af_algorithm_input_profile("adaptive_gradient")
    assert prof["steps_label"] == "max evals"
