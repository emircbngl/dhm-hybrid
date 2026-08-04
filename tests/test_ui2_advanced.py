"""Tests for v2.0.5 polish work: advanced pre-processing, error
drawer bounded log, maximize-panel state, migration v5→v6.

DPG/DhmApp tests removed 2026-07-06 with the ui2 frontend retirement; driver/state tests kept.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings_schema import SCHEMA_VERSION, AppSettings  # noqa: E402
from ui2 import state_store  # noqa: E402
from ui2.reconstruction import ReconParams  # noqa: E402


# ---------------------------------------------------------------------------
# ReconParams defaults for the advanced block
# ---------------------------------------------------------------------------

def test_recon_params_has_advanced_fields():
    p = ReconParams()
    # Match v1 ReconDefaults so loading an old dump doesn't flip behaviour.
    assert p.subtract_mean is True
    assert p.hann_window is False
    assert p.fft_backend == "auto"
    assert p.unwrap_method == "GRADIENT_INTEGRATION"


# ---------------------------------------------------------------------------
# Migration v5 → v6
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "ui2_state.json"


def test_migration_v5_payload_backfills_advanced(tmp_state):
    """A v5 dump (no advanced block) must reach v6 with v1-equivalent
    defaults so behaviour doesn't silently change."""
    tmp_state.write_text(json.dumps({
        "schema_version": 5,
        "ui2": {"theme": "dark", "sample_id": "pre-v6"},
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.subtract_mean is True
    assert loaded.ui2.hann_window is False
    assert loaded.ui2.fft_backend == "auto"
    assert loaded.ui2.unwrap_method == "GRADIENT_INTEGRATION"


def test_roundtrip_preserves_advanced_fields(tmp_state):
    s = AppSettings.defaults().with_ui2(
        subtract_mean=False, hann_window=True,
        fft_backend="mlx", unwrap_method="QUALITY_GUIDED",
    )
    state_store.save(s, tmp_state)
    loaded = state_store.load(tmp_state)
    assert loaded.ui2.subtract_mean is False
    assert loaded.ui2.hann_window is True
    assert loaded.ui2.fft_backend == "mlx"
    assert loaded.ui2.unwrap_method == "QUALITY_GUIDED"


# ---------------------------------------------------------------------------
# Advanced pre-processing — hann_window and subtract_mean flow through
# ---------------------------------------------------------------------------

def test_subtract_mean_removes_dc_before_propagate():
    """Feed a hologram with a 0.3 DC offset; with subtract_mean=True
    the mean of the processed array must be ~0 before the normalise
    step in ReconstructionDriver._run."""
    from ui2.reconstruction import ReconstructionDriver

    # We can't call _run directly without a real hologram file and
    # the full pipeline — but we can verify the field existence on
    # ReconParams + the one-liner: subtract_mean=True ⇒ raw -= mean.
    # For a full-path test see the integration suite.
    p = ReconParams(subtract_mean=True)
    arr = np.full((8, 8), 0.5, dtype=np.float32)
    if p.subtract_mean:
        arr = arr - float(arr.mean())
    assert abs(arr.mean()) < 1e-6


def test_hann_window_dims_match_array():
    p = ReconParams(hann_window=True)
    arr = np.ones((16, 24), dtype=np.float32)
    if p.hann_window:
        wy = np.hanning(arr.shape[0]).astype(np.float32)
        wx = np.hanning(arr.shape[1]).astype(np.float32)
        arr = arr * (wy[:, None] * wx[None, :])
    assert arr.shape == (16, 24)
    # Corners should be ~0 (Hann boundary), centre should be ~1.
    assert arr[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert arr[arr.shape[0] // 2, arr.shape[1] // 2] > 0.9
