"""NIST bead calibration workflow tests (v2.0.8, D3).

Coverage:

* Diameter recovery from a synthetic single-sphere phase image
  (round-trip vs known input).
* Drift / classify thresholds (green / yellow / red).
* History append + load.
* Operator + path resolution defaults via :mod:`user_profile`.
* Edge cases — blank input, malformed history, zero nominal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core import user_profile  # noqa: E402
from core.calibration import (  # noqa: E402
    CalibrationCheck,
    CalibrationStatus,
    classify,
    load_history,
    measure_bead_diameter,
    record_check,
)


# ---------------------------------------------------------------------------
# Sandbox the user-profile root so writes don't escape tmp_path.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.delenv(user_profile.ENV_VAR, raising=False)
    monkeypatch.delenv(user_profile.ENV_ROOT, raising=False)
    user_profile.set_root_dir(tmp_path)
    monkeypatch.setenv(user_profile.ENV_VAR, "karin")
    yield tmp_path
    user_profile.set_root_dir(None)


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("drift,expected", [
    (0.0,   CalibrationStatus.GREEN),
    (1.5,   CalibrationStatus.GREEN),
    (-1.5,  CalibrationStatus.GREEN),
    (2.5,   CalibrationStatus.YELLOW),
    (-3.7,  CalibrationStatus.YELLOW),
    (4.99,  CalibrationStatus.YELLOW),
    (5.0,   CalibrationStatus.RED),
    (-12.0, CalibrationStatus.RED),
])
def test_classify_thresholds(drift, expected):
    assert classify(drift) == expected


def test_classify_custom_thresholds():
    """Lab can pick stricter cutoffs for publication-grade data."""
    assert classify(0.5, yellow_threshold=0.3, red_threshold=1.0) \
        == CalibrationStatus.YELLOW
    assert classify(2.0, yellow_threshold=0.3, red_threshold=1.0) \
        == CalibrationStatus.RED


# ---------------------------------------------------------------------------
# measure_bead_diameter
# ---------------------------------------------------------------------------

def _synthetic_bead_phase(diameter_um: float = 10.0,
                          pixel_um: float = 0.5,
                          shape=(128, 128)) -> np.ndarray:
    """Synthetic phase image of a sphere — disk of constant phase
    over a flat zero baseline. Resembles a focused single-bead
    reconstruction enough for this test."""
    ny, nx = shape
    y, x = np.indices(shape, dtype=np.float32)
    cy, cx = ny / 2.0, nx / 2.0
    radius_px = (diameter_um / 2.0) / pixel_um
    rho = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    phase = np.zeros(shape, dtype=np.float32)
    phase[rho <= radius_px] = 2.0  # arbitrary uniform phase
    return phase


def test_measure_bead_diameter_recovers_known_size():
    """10 µm bead at 0.5 µm / px → equivalent-disk diameter
    should land within 5 % of 10 µm. Threshold-based metric so
    pixelation bias makes this slightly less than exact, but the
    drift number stays well inside lab acceptance."""
    phase = _synthetic_bead_phase(diameter_um=10.0, pixel_um=0.5)
    measured = measure_bead_diameter(phase, pixel_size_um=0.5)
    assert abs(measured - 10.0) / 10.0 < 0.05


def test_measure_bead_diameter_zero_when_blank():
    """No bead in the frame → diameter 0.0, not a hallucinated peak."""
    blank = np.zeros((64, 64), dtype=np.float32)
    assert measure_bead_diameter(blank, pixel_size_um=0.5) == 0.0


def test_measure_bead_diameter_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        measure_bead_diameter(np.zeros((4, 4, 4), dtype=np.float32),
                              pixel_size_um=0.5)


def test_measure_bead_diameter_off_centre():
    """Bead away from the frame centre — passing centre_yx should
    still recover the size."""
    shape = (128, 128)
    pixel_um = 0.5
    diameter_um = 10.0
    radius_px = (diameter_um / 2.0) / pixel_um
    phase = np.zeros(shape, dtype=np.float32)
    cy, cx = 30, 90
    y, x = np.indices(shape, dtype=np.float32)
    phase[((y - cy) ** 2 + (x - cx) ** 2) <= radius_px ** 2] = 2.0
    measured = measure_bead_diameter(
        phase, pixel_size_um=pixel_um, centre_yx=(cy, cx),
    )
    assert abs(measured - 10.0) / 10.0 < 0.1


# ---------------------------------------------------------------------------
# record_check + history
# ---------------------------------------------------------------------------

def test_record_check_writes_history_jsonl(sandbox):
    """A fresh check populates the per-user JSONL with one line."""
    check = record_check(
        nominal_diameter_um=10.0,
        measured_diameter_um=10.05,
        pixel_size_um=0.5,
    )
    assert check.status == CalibrationStatus.GREEN.value
    history_path = (
        user_profile.user_state_dir("karin") / "calibration_history.jsonl"
    )
    assert history_path.exists()
    rows = [json.loads(l) for l in
            history_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["operator"] == "karin"
    assert abs(rows[0]["drift_percent"] - 0.5) < 1e-6


def test_record_check_classifies_yellow_at_3_percent(sandbox):
    check = record_check(
        nominal_diameter_um=10.0,
        measured_diameter_um=10.3,
    )
    assert check.status == CalibrationStatus.YELLOW.value


def test_record_check_classifies_red_at_8_percent(sandbox):
    check = record_check(
        nominal_diameter_um=10.0,
        measured_diameter_um=10.8,
    )
    assert check.status == CalibrationStatus.RED.value


def test_record_check_appends_history(sandbox):
    """Two checks → two history lines, in append order."""
    record_check(nominal_diameter_um=10.0,
                 measured_diameter_um=10.05)
    record_check(nominal_diameter_um=10.0,
                 measured_diameter_um=10.30)
    history = load_history(
        user_profile.user_state_dir("karin")
        / "calibration_history.jsonl",
    )
    assert len(history) == 2
    assert history[0].status == "green"
    assert history[1].status == "yellow"


def test_record_check_explicit_history_path(sandbox, tmp_path):
    """Caller can override the destination for tests / scripted
    multi-lab setups."""
    custom = tmp_path / "out" / "weekly.jsonl"
    record_check(
        nominal_diameter_um=10.0,
        measured_diameter_um=10.05,
        history_path=custom,
    )
    assert custom.exists()


def test_record_check_explicit_operator(sandbox):
    record_check(
        nominal_diameter_um=10.0,
        measured_diameter_um=10.05,
        operator="erik",
    )
    erik_path = (
        user_profile.user_state_dir("erik") / "calibration_history.jsonl"
    )
    assert erik_path.exists()


def test_record_check_zero_nominal_raises():
    with pytest.raises(ValueError, match="nominal_diameter_um"):
        record_check(nominal_diameter_um=0.0,
                     measured_diameter_um=10.0)


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------

def test_load_history_missing_file_returns_empty(tmp_path):
    assert load_history(tmp_path / "no_such.jsonl") == []


def test_load_history_skips_malformed_lines(tmp_path):
    p = tmp_path / "h.jsonl"
    good = {
        "timestamp": "2026-04-27T12:00:00",
        "operator": "karin",
        "nominal_diameter_um": 10.0,
        "measured_diameter_um": 10.1,
        "drift_percent": 1.0,
        "status": "green",
        "pixel_size_um": 0.5,
        "notes": "",
    }
    p.write_text("\n".join([
        json.dumps(good),
        "not valid json",
        "",
        json.dumps(good),
    ]), encoding="utf-8")
    rows = load_history(p)
    assert len(rows) == 2


def test_load_history_partial_record_filled_with_defaults(tmp_path):
    """Older records may be missing ``pixel_size_um`` etc. The loader
    reconstructs them with safe defaults rather than crashing."""
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps({
        "timestamp": "2026-04-27T12:00:00",
        "operator": "karin",
        "nominal_diameter_um": 10.0,
        "measured_diameter_um": 10.1,
        "drift_percent": 1.0,
        "status": "green",
        # pixel_size_um + notes missing
    }) + "\n", encoding="utf-8")
    rows = load_history(p)
    assert len(rows) == 1
    assert rows[0].pixel_size_um == 0.0
    assert rows[0].notes == ""


def test_load_history_returns_calibration_check_instances(sandbox):
    record_check(nominal_diameter_um=10.0, measured_diameter_um=10.05)
    history = load_history(
        user_profile.user_state_dir("karin")
        / "calibration_history.jsonl",
    )
    assert all(isinstance(c, CalibrationCheck) for c in history)
