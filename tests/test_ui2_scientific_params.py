"""Tests for the scientific parameters the v2 port dropped.

CEO caught magnification missing in pilot test — these tests pin the
fix so we don't regress:

* :class:`ReconParams` carries M, pixel_is_effective, n_sample,
  n_medium, autofocus_metric.
* ``effective_pixel_um()`` divides only when appropriate.
* Presets ship sane magnifications (Cell @ 40×, USAF @ 10×, Film @ 1×).
* TIFF metadata auto-detect populates the sidebar.
* State roundtrip + v3→v4 migration preserves the new fields.
* Info-text composer shows the pixel arithmetic.

DPG/DhmApp tests removed 2026-07-06 with the ui2 frontend retirement; driver/state tests kept.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings_schema import SCHEMA_VERSION, AppSettings  # noqa: E402
from ui2 import state_store  # noqa: E402
from ui2.reconstruction import ReconParams  # noqa: E402


# ---------------------------------------------------------------------------
# ReconParams field surface
# ---------------------------------------------------------------------------

def test_recon_params_defaults_match_v1():
    """Defaults should mirror v1 so un-touched state behaves identically."""
    p = ReconParams()
    assert p.magnification == 1.0
    assert p.pixel_is_effective is True
    assert p.n_sample == pytest.approx(1.38)
    assert p.n_medium == pytest.approx(1.337)
    assert p.autofocus_metric == "LAPLACIAN_VARIANCE"


@pytest.mark.parametrize("px,mag,effective,expected", [
    (3.45, 40.0, False, 3.45 / 40.0),     # camera pixel on 40× objective
    (3.45, 40.0, True,  3.45),            # user pre-divided
    (5.5,  1.0,  True,  5.5),             # macroscopic imaging, no objective
    (5.5,  1.0,  False, 5.5),             # M=1 and flag False → no-op divide
    (2.2,  10.0, False, 0.22),            # USAF at 10×
])
def test_effective_pixel_um(px, mag, effective, expected):
    p = ReconParams(pixel_um=px, magnification=mag,
                    pixel_is_effective=effective)
    assert p.effective_pixel_um() == pytest.approx(expected, rel=1e-6)


def test_effective_pixel_falls_back_when_magnification_is_zero():
    """Guard: don't ever divide by zero, even if the user types nonsense."""
    p = ReconParams(pixel_um=3.45, magnification=0.0,
                    pixel_is_effective=False)
    assert p.effective_pixel_um() == pytest.approx(3.45)


# ---------------------------------------------------------------------------
# State roundtrip + migration v3 → v4
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "ui2_state.json"


def test_state_roundtrip_preserves_magnification(tmp_state):
    s = AppSettings.defaults().with_ui2(
        magnification=60.0, pixel_is_effective=False,
        n_sample=1.42, n_medium=1.515,
        autofocus_metric="TENENGRAD",
    )
    state_store.save(s, tmp_state)
    loaded = state_store.load(tmp_state)
    assert loaded.ui2.magnification == 60.0
    assert loaded.ui2.pixel_is_effective is False
    assert loaded.ui2.n_sample == pytest.approx(1.42)
    assert loaded.ui2.n_medium == pytest.approx(1.515)
    assert loaded.ui2.autofocus_metric == "TENENGRAD"


def test_migration_v3_payload_hydrates_with_v1_defaults(tmp_state):
    """A state file that predates v4 should load cleanly with
    M=1, pixel_is_effective=True (v2.0.2 behaviour)."""
    tmp_state.write_text(json.dumps({
        "schema_version": 3,
        "ui2": {
            "theme": "dark",
            "sample_id": "legacy",
            "pixel_um": 3.45,
            # No magnification, no n_sample, no metric — simulate
            # a state file written before v4.
        },
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.sample_id == "legacy"
    assert loaded.ui2.pixel_um == pytest.approx(3.45)
    # Backfilled defaults:
    assert loaded.ui2.magnification == 1.0
    assert loaded.ui2.pixel_is_effective is True
    assert loaded.ui2.n_sample == pytest.approx(1.38)
    assert loaded.ui2.autofocus_metric == "LAPLACIAN_VARIANCE"


def test_migration_v1_payload_reaches_v4_with_ui2_defaults(tmp_state):
    """Even a v1 Qt-settings-only dump should reach v4 cleanly."""
    tmp_state.write_text(json.dumps({
        "schema_version": 1,
        "recon": {"wavelength_nm": 532.0},
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.recon.wavelength_nm == pytest.approx(532.0)
    assert loaded.ui2.magnification == 1.0
    assert loaded.ui2.autofocus_metric == "LAPLACIAN_VARIANCE"


# ---------------------------------------------------------------------------
# Metric listing stays in lockstep with the enum
# ---------------------------------------------------------------------------

def test_available_focus_metrics_nonempty():
    from ui2.workers import available_focus_metrics
    names = available_focus_metrics()
    assert len(names) >= 5
    assert "LAPLACIAN_VARIANCE" in names
    # All names are uppercase so the sidebar combo's default lookup
    # with the string form never fails on a case mismatch.
    for n in names:
        assert n == n.upper()


def test_metric_name_round_trips_to_enum():
    from core.autofocus import FocusMetric
    from ui2.workers import _metric_from_params
    p = ReconParams(autofocus_metric="TENENGRAD")
    assert _metric_from_params(p) == FocusMetric.TENENGRAD


def test_unknown_metric_falls_back_to_default():
    from core.autofocus import FocusMetric
    from ui2.workers import _metric_from_params
    p = ReconParams(autofocus_metric="NOT_A_REAL_METRIC")
    assert _metric_from_params(p) == FocusMetric.LAPLACIAN_VARIANCE
