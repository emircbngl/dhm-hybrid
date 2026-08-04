"""Tests for Phase 3 (AI_VISION_MCP_PLAN.md): the ui2 reference-mode UI
+ reffree pipeline path.

Covers:

* :class:`ReconParams` new fields default correctly and
  ``effective_reference_mode()`` reconciles the legacy
  ``subtract_reference`` checkbox with the new three-way combo.
* :class:`ReconstructionDriver` actually runs the reffree background
  correction on a synthetic off-axis hologram and the result differs
  from the "off" mode run (the bg fit does something).
* The Track C CNN toggle degrades gracefully (no torch in this venv)
  instead of crashing or silently no-op'ing — a status note comes
  back on the result.
* ``Ui2State``/state_store round-trips the new fields, and a v12
  payload migrates to v13 with defaults that reproduce old behaviour.

DPG/DhmApp tests removed 2026-07-06 with the ui2 frontend retirement; driver/state tests kept.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings_schema import SCHEMA_VERSION, AppSettings  # noqa: E402
from ui2 import state_store  # noqa: E402
from ui2.reconstruction import ReconParams, ReconstructionDriver  # noqa: E402
from ui2.workers import reffree_cnn_available  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic off-axis hologram — same pattern as test_reffree_pipeline.py's
# _carrier_hologram, kept local so this file runs standalone.
# ---------------------------------------------------------------------------

def _carrier_hologram(n: int = 256, fc: float = 0.12, seed: int = 5,
                      tilt_rad: float = 3.0) -> np.ndarray:
    """A tilted-reference off-axis hologram with a smooth curvature term
    baked in (mimics uncorrected wavefront curvature a reffree background
    fit should remove) plus a weak Gaussian "object" bump."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.4 * np.exp(
        -(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2))
    )
    # Slow quadratic curvature — this is exactly what a Zernike/poly
    # background fit is supposed to remove; without a reference hologram
    # to divide out, the reffree path has to catch it numerically.
    xn = (X - n / 2) / (n / 2)
    yn = (Y - n / 2) / (n / 2)
    curvature = tilt_rad * (xn ** 2 + yn ** 2)
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * fc * X + bump + curvature))
    holo = np.abs(1.0 + obj) ** 2
    holo = holo + 0.01 * rng.standard_normal((n, n))
    return (holo * 1000).astype(np.uint16)


def _write_hologram(tmp_path: Path, name: str = "holo.tif", **kwargs) -> Path:
    import tifffile
    path = tmp_path / name
    tifffile.imwrite(path, _carrier_hologram(**kwargs))
    return path


def _run_sync(driver: ReconstructionDriver, path: Path, params: ReconParams,
             timeout_s: float = 10.0):
    """Blocking wrapper around ``ReconstructionDriver.submit`` for tests."""
    done = [False]
    box: dict = {}

    def _ok(r):
        box["result"] = r
        done[0] = True

    def _err(e):
        box["error"] = e
        done[0] = True

    driver.submit(path, params, on_result=_ok, on_error=_err)
    deadline = time.monotonic() + timeout_s
    while not done[0] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert done[0], "reconstruction never completed"
    if "error" in box:
        raise AssertionError(f"reconstruction failed: {box['error'].message}")
    return box["result"]


# ---------------------------------------------------------------------------
# (a) ReconParams defaults + effective_reference_mode() back-compat
# ---------------------------------------------------------------------------

def test_recon_params_reffree_field_defaults():
    p = ReconParams()
    assert p.reference_mode == "off"
    assert p.reffree_bg_method == "zernike"
    assert p.reffree_bg_order == 4
    assert p.reffree_n_terms == 15
    assert p.reffree_cnn is False


def test_effective_reference_mode_defers_to_legacy_checkbox_when_off():
    """A ReconParams built the old way (only subtract_reference set,
    reference_mode never touched) must behave exactly like "reference"."""
    p = ReconParams(subtract_reference=True, reference_path=Path("/tmp/r.tif"))
    assert p.reference_mode == "off"          # never explicitly set
    assert p.effective_reference_mode() == "reference"


def test_effective_reference_mode_off_with_no_legacy_flag():
    p = ReconParams()
    assert p.effective_reference_mode() == "off"


def test_effective_reference_mode_explicit_reference_free_wins():
    """Explicit reference_free must NOT be reinterpreted as "reference"
    even if a reference path happens to be loaded/armed."""
    p = ReconParams(reference_mode="reference_free",
                    subtract_reference=True,
                    reference_path=Path("/tmp/r.tif"))
    assert p.effective_reference_mode() == "reference_free"


def test_effective_reference_mode_explicit_reference():
    p = ReconParams(reference_mode="reference")
    assert p.effective_reference_mode() == "reference"


# ---------------------------------------------------------------------------
# (b) ScienceDriver/ReconstructionDriver reffree run vs. off run
# ---------------------------------------------------------------------------

def test_reffree_reconstruction_runs_and_produces_finite_unwrapped_phase(
    tmp_path,
):
    path = _write_hologram(tmp_path)
    driver = ReconstructionDriver()
    try:
        params = ReconParams(
            mask_radius=48, z_mm=0.0, reference_mode="reference_free",
        )
        result = _run_sync(driver, path, params)
    finally:
        driver.shutdown()

    assert result.unwrapped_phase is not None
    assert result.unwrapped_phase.shape == result.phase.shape
    assert np.all(np.isfinite(result.unwrapped_phase))
    assert result.reffree_note is None


def test_off_mode_does_not_populate_unwrapped_phase(tmp_path):
    path = _write_hologram(tmp_path)
    driver = ReconstructionDriver()
    try:
        params = ReconParams(mask_radius=48, z_mm=0.0, reference_mode="off")
        result = _run_sync(driver, path, params)
    finally:
        driver.shutdown()
    assert result.unwrapped_phase is None
    assert result.reffree_note is None


def test_reffree_background_fit_changes_the_result(tmp_path):
    """The whole point of reference-free mode is a numerical background
    correction — the corrected phase must differ from the plain
    unwrap-with-no-correction result on a hologram with baked-in
    curvature (see ``_carrier_hologram``'s tilt term)."""
    from core.phase_unwrap import UnwrapConfig, UnwrapMethod, unwrap_phase_advanced

    path = _write_hologram(tmp_path, tilt_rad=3.0)
    driver = ReconstructionDriver()
    try:
        params = ReconParams(
            mask_radius=48, z_mm=0.0, reference_mode="reference_free",
            reffree_bg_method="polynomial", reffree_bg_order=2,
        )
        result = _run_sync(driver, path, params)
    finally:
        driver.shutdown()

    # Re-derive the plain unwrap (no bg fit) from the same wrapped phase
    # + amplitude the driver computed, using the same unwrap method, and
    # assert the bg-fit result is not the same array — the fit actually
    # subtracted something.
    plain_unwrapped = unwrap_phase_advanced(
        result.phase,
        UnwrapConfig(method=UnwrapMethod[params.unwrap_method]),
        complex_field=None,
    )
    assert not np.allclose(result.unwrapped_phase, plain_unwrapped, atol=1e-6)


def test_reference_free_mode_ignores_a_loaded_reference_path(tmp_path):
    """Phase 3 scope rule: reference_free must NOT divide by a reference
    hologram even if one happens to be loaded + subtract_reference is
    still armed from a previous "reference" session."""
    from ui2.workers import _extract_field_with_reference
    from core.offaxis import OffAxisParams

    holo = _carrier_hologram(seed=1)
    ref = _carrier_hologram(seed=2)
    raw_path = tmp_path / "raw.tif"
    ref_path = tmp_path / "ref.tif"
    import tifffile
    tifffile.imwrite(raw_path, holo)
    tifffile.imwrite(ref_path, ref)

    params_reffree = ReconParams(
        mask_radius=48, reference_mode="reference_free",
        subtract_reference=True, reference_path=ref_path,
    )
    params_off = ReconParams(mask_radius=48, reference_mode="off")

    offaxis = OffAxisParams(radius=48)
    field_reffree, _, note_reffree = _extract_field_with_reference(
        holo.astype(np.float32), params_reffree, offaxis,
    )
    field_off, _, _ = _extract_field_with_reference(
        holo.astype(np.float32), params_off, offaxis,
    )
    # Same field as "off" — the reference division never happened.
    assert np.allclose(field_reffree, field_off)
    # reference_free mode must NOT attempt (or fail) a reference division here.
    assert note_reffree is None


# ---------------------------------------------------------------------------
# (c) CNN gate: torch not installed in this venv -> graceful skip
# ---------------------------------------------------------------------------

def test_reffree_cnn_unavailable_without_torch():
    """This venv ships without torch — the guard must say unavailable,
    never raise."""
    assert reffree_cnn_available() is False


def test_reffree_cnn_toggle_degrades_gracefully(tmp_path):
    """Arming reffree_cnn with no torch/checkpoint must still produce a
    usable result (classical bg fit) plus a non-None status note —
    never a silent no-op, never a crash."""
    path = _write_hologram(tmp_path)
    driver = ReconstructionDriver()
    try:
        params = ReconParams(
            mask_radius=48, z_mm=0.0, reference_mode="reference_free",
            reffree_cnn=True,
        )
        result = _run_sync(driver, path, params)
    finally:
        driver.shutdown()

    assert result.unwrapped_phase is not None
    assert np.all(np.isfinite(result.unwrapped_phase))
    assert result.reffree_note is not None
    assert "unavailable" in result.reffree_note.lower()


def test_reffree_cnn_available_checks_checkpoint_path(tmp_path):
    """Both conditions are required: a checkpoint on disk AND torch.

    A missing checkpoint is False regardless of torch. With the
    checkpoint present the answer follows torch's importability, so
    assert against that rather than against whichever box happens to
    be running the suite — the original version hard-coded False and
    only passed without torch installed (it broke the moment CI
    installed it).
    """
    missing_ckpt = tmp_path / "absent.pt"
    assert reffree_cnn_available(missing_ckpt) is False

    fake_ckpt = tmp_path / "model.pt"
    fake_ckpt.write_bytes(b"not a real checkpoint")

    try:
        import torch  # noqa: F401
        torch_present = True
    except ImportError:
        torch_present = False

    assert reffree_cnn_available(fake_ckpt) is torch_present


# ---------------------------------------------------------------------------
# (d) State round-trip + v12 -> v13 migration
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "ui2_state.json"


def test_state_roundtrip_preserves_reffree_fields(tmp_state):
    s = AppSettings.defaults().with_ui2(
        reference_mode="reference_free",
        reffree_bg_method="polynomial",
        reffree_bg_order=3,
        reffree_n_terms=21,
        reffree_cnn=True,
    )
    state_store.save(s, tmp_state)
    loaded = state_store.load(tmp_state)
    assert loaded.ui2.reference_mode == "reference_free"
    assert loaded.ui2.reffree_bg_method == "polynomial"
    assert loaded.ui2.reffree_bg_order == 3
    assert loaded.ui2.reffree_n_terms == 21
    assert loaded.ui2.reffree_cnn is True


def test_migration_v12_payload_backfills_v13_defaults(tmp_state):
    tmp_state.write_text(json.dumps({
        "schema_version": 12,
        "ui2": {
            "theme": "dark",
            "sample_id": "legacy-v12",
            "subtract_reference": True,
        },
        "ai": {},
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.sample_id == "legacy-v12"
    # Backfilled defaults — no behaviour change for old dumps.
    assert loaded.ui2.reference_mode == "off"
    assert loaded.ui2.reffree_bg_method == "zernike"
    assert loaded.ui2.reffree_bg_order == 4
    assert loaded.ui2.reffree_n_terms == 15
    assert loaded.ui2.reffree_cnn is False
    # And the legacy subtract_reference flag still resolves through
    # ReconParams.effective_reference_mode() to "reference" — this is
    # the exact contract the migration promises.
    params = ReconParams(
        subtract_reference=bool(loaded.ui2.subtract_reference),
        reference_mode=str(loaded.ui2.reference_mode),
    )
    assert params.effective_reference_mode() == "reference"


def test_migration_v1_payload_reaches_v13_with_reffree_defaults(tmp_state):
    """Even a pre-ui2 v1 dump should migrate all the way to v13."""
    tmp_state.write_text(json.dumps({
        "schema_version": 1,
        "recon": {"wavelength_nm": 532.0},
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.reference_mode == "off"
    assert loaded.ui2.reffree_bg_method == "zernike"
    assert loaded.ui2.reffree_n_terms == 15


# ---------------------------------------------------------------------------
# 2026-07-05 review fixes — QPI correction + audit record
# ---------------------------------------------------------------------------

def _run_qpi_sync(path: Path, params: ReconParams, timeout_s: float = 15.0):
    """Blocking wrapper around ``ScienceDriver.run_qpi`` for tests."""
    from ui2.workers import ScienceDriver

    driver = ScienceDriver()
    done = [False]
    box: dict = {}

    def _ok(r):
        box["result"] = r
        done[0] = True

    def _err(e):
        box["error"] = e
        done[0] = True

    try:
        driver.run_qpi(path, params, z_mm=0.0,
                       on_result=_ok, on_error=_err)
        deadline = time.monotonic() + timeout_s
        while not done[0] and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        driver.shutdown()
    assert done[0], "qpi never completed"
    if "error" in box:
        raise AssertionError(f"qpi failed: {box['error'].message}")
    return box["result"]


def test_qpi_applies_reffree_correction(tmp_path):
    """Review MED: QPI numbers under reference_free must be background-
    corrected too — pre-fix run_qpi skipped the fit entirely, so OPD /
    dry mass silently carried the aberration the mode exists to remove."""
    holo = _write_hologram(tmp_path, tilt_rad=3.0)
    results = {}
    for mode in ("off", "reference_free"):
        params = ReconParams(
            mask_radius=48, reference_mode=mode,
            reffree_bg_method="polynomial", reffree_bg_order=2,
        )
        results[mode] = _run_qpi_sync(holo, params)

    off_range = results["off"].qpi.phase_stats.range_nm
    reffree_range = results["reference_free"].qpi.phase_stats.range_nm
    # The tilt-baked fixture guarantees a nonzero background fit — the
    # corrected OPD range must differ measurably from the uncorrected.
    assert reffree_range != pytest.approx(off_range, rel=1e-3)


def test_params_for_audit_records_resolved_mode():
    """Review LOW: the audit record must carry the RESOLVED reference
    mode (+ reffree knobs when active), not just the raw legacy flag."""
    from ui2.workers import _params_for_audit
    d = _params_for_audit(ReconParams(reference_mode="reference_free",
                                      reffree_bg_method="polynomial",
                                      reffree_bg_order=3))
    assert d["reference_mode"] == "reference_free"
    assert d["reffree_bg_method"] == "polynomial"
    assert d["reffree_bg_order"] == 3
    d2 = _params_for_audit(ReconParams(subtract_reference=True))
    assert d2["reference_mode"] == "reference"
    assert "reffree_bg_method" not in d2
