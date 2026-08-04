"""Tests for v2.0.7 work:

* Optical mode reflection halves the OPD (2× height bug fix).
* Amplitude percentile auto-contrast clips outliers.
* ROI-gated autofocus only sees masked pixels.
* Migration v7 → v8 backfills the new fields.

DPG/DhmApp tests removed 2026-07-06 with the ui2 frontend retirement; driver/state tests kept.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings_schema import SCHEMA_VERSION, AppSettings  # noqa: E402
from ui2 import state_store  # noqa: E402
from ui2.reconstruction import ReconParams  # noqa: E402


# ---------------------------------------------------------------------------
# ReconParams defaults for v2.0.7 fields
# ---------------------------------------------------------------------------

def test_optical_mode_defaults_to_transmission():
    p = ReconParams()
    assert p.optical_mode == "transmission"
    assert p.auto_contrast_amplitude is True
    assert p.af_roi is None


# ---------------------------------------------------------------------------
# Migration v7 → v8
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "ui2_state.json"


def test_migration_v7_payload_backfills_optical_mode(tmp_state):
    tmp_state.write_text(json.dumps({
        "schema_version": 7,
        "ui2": {"theme": "dark", "sample_id": "legacy-v7"},
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.optical_mode == "transmission"
    assert loaded.ui2.auto_contrast_amplitude is True


def test_roundtrip_preserves_optical_mode(tmp_state):
    s = AppSettings.defaults().with_ui2(
        optical_mode="reflection", auto_contrast_amplitude=False,
    )
    state_store.save(s, tmp_state)
    loaded = state_store.load(tmp_state)
    assert loaded.ui2.optical_mode == "reflection"
    assert loaded.ui2.auto_contrast_amplitude is False


# ---------------------------------------------------------------------------
# Reflection mode halves OPD — the 2× height bug regression guard
# ---------------------------------------------------------------------------

def test_reflection_mode_halves_unwrapped_phase_before_qpi():
    """Run the QPI worker with optical_mode='reflection' and assert
    the unwrapped phase fed into compute_qpi is exactly the
    transmission-mode value divided by 2 — that's what makes
    a 3-µm-tall feature report 3 µm instead of 6 µm."""
    from ui2.workers import ScienceDriver
    import time

    fake_loaded = MagicMock()
    fake_loaded.array = np.zeros((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    captured_unwrapped: dict = {}

    fake_qpi = MagicMock()
    fake_qpi.phase_stats = MagicMock(range_nm=100.0)
    fake_qpi.total_dry_mass_pg = None
    fake_qpi.step_height_m = None

    synthetic_unwrapped = np.full((16, 16), 6.0, dtype=np.float32)  # 6 rad

    def fake_compute_qpi(unwrapped, **kwargs):
        captured_unwrapped["arr"] = np.asarray(unwrapped, dtype=np.float32)
        return fake_qpi

    params = ReconParams(optical_mode="reflection")
    driver = ScienceDriver()
    done = [False]

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.zeros((16, 16), dtype=np.complex64),
                             (8, 8))), \
         patch("ui2.workers.propagate",
               return_value=np.zeros((16, 16), dtype=np.complex64)), \
         patch("ui2.workers.unwrap_phase_advanced",
               return_value=synthetic_unwrapped), \
         patch("ui2.workers.compute_qpi", side_effect=fake_compute_qpi):
        driver.run_qpi(
            Path("/dev/null"), params, z_mm=1.0,
            on_result=lambda r: done.__setitem__(0, True),
            on_error=lambda e: done.__setitem__(0, True),
        )
        deadline = time.monotonic() + 3.0
        while not done[0] and time.monotonic() < deadline:
            time.sleep(0.01)
    driver.shutdown()

    assert "arr" in captured_unwrapped
    # Reflection mode must halve the phase before compute_qpi.
    np.testing.assert_allclose(captured_unwrapped["arr"], 3.0, rtol=1e-6)


def test_transmission_mode_passes_unwrapped_phase_untouched():
    """Transmission mode (default) must not rescale — regression
    guard so we don't accidentally halve in both modes."""
    from ui2.workers import ScienceDriver
    import time

    fake_loaded = MagicMock()
    fake_loaded.array = np.zeros((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    captured = {}

    def fake_compute_qpi(unwrapped, **kwargs):
        captured["arr"] = np.asarray(unwrapped, dtype=np.float32).copy()
        qpi = MagicMock()
        qpi.phase_stats = MagicMock(range_nm=None)
        qpi.total_dry_mass_pg = None
        qpi.step_height_m = None
        return qpi

    unwrapped = np.full((8, 8), 5.0, dtype=np.float32)
    params = ReconParams(optical_mode="transmission")
    driver = ScienceDriver()
    done = [False]

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.zeros((8, 8), dtype=np.complex64), (4, 4))), \
         patch("ui2.workers.propagate",
               return_value=np.zeros((8, 8), dtype=np.complex64)), \
         patch("ui2.workers.unwrap_phase_advanced",
               return_value=unwrapped), \
         patch("ui2.workers.compute_qpi", side_effect=fake_compute_qpi):
        driver.run_qpi(
            Path("/dev/null"), params, z_mm=1.0,
            on_result=lambda r: done.__setitem__(0, True),
            on_error=lambda e: done.__setitem__(0, True),
        )
        deadline = time.monotonic() + 3.0
        while not done[0] and time.monotonic() < deadline:
            time.sleep(0.01)
    driver.shutdown()

    np.testing.assert_allclose(captured["arr"], 5.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# ROI masking feeds the scan only the selected pixels
# ---------------------------------------------------------------------------

def test_prepare_field_applies_roi_mask():
    from ui2 import workers

    fake_loaded = MagicMock()
    fake_loaded.array = np.ones((32, 32), dtype=np.float32)
    fake_loaded.metadata = {}

    full_field = np.full((32, 32), 1.0 + 1.0j, dtype=np.complex64)

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(full_field, (16, 16))):
        # Without ROI — every pixel is kept.
        params = ReconParams()
        field_full, _, _, _ = workers._prepare_field(
            Path("/dev/null"), params, apply_af_roi=True,
        )
        assert np.count_nonzero(field_full) == 32 * 32

        # With ROI covering upper-left quarter — 16x16 = 256 non-zeros.
        params_roi = ReconParams(af_roi=(0.0, 0.0, 0.5, 0.5))
        field_roi, _, _, _ = workers._prepare_field(
            Path("/dev/null"), params_roi, apply_af_roi=True,
        )
        # ROI of 0..0.5 normalised on a 32-wide field hits indices 0..15
        # via ``int(0.5 * 31) = 15`` and the inclusive slice
        # ``[0:16, 0:16]`` → exactly 256 live pixels.
        assert np.count_nonzero(field_roi) == 16 * 16
        # And the live region is inside the top-left quadrant.
        assert field_roi[0, 0] != 0
        assert field_roi[20, 20] == 0


def test_prepare_field_skips_roi_when_flag_false():
    """Reconstruction (``apply_af_roi=False``) must never mask the
    field — the user's on-screen preview should always show the
    full frame even with a scan-only ROI set."""
    from ui2 import workers

    fake_loaded = MagicMock()
    fake_loaded.array = np.ones((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.full((16, 16), 1.0 + 1.0j,
                                     dtype=np.complex64), (8, 8))):
        params = ReconParams(af_roi=(0.0, 0.0, 0.1, 0.1))
        field, _, _, _ = workers._prepare_field(
            Path("/dev/null"), params, apply_af_roi=False,
        )
    assert np.count_nonzero(field) == 16 * 16


def test_prepare_field_ignores_degenerate_roi():
    """Zero-area ROI must not crash and must not zero the field."""
    from ui2 import workers

    fake_loaded = MagicMock()
    fake_loaded.array = np.ones((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.full((16, 16), 1.0 + 0.5j,
                                     dtype=np.complex64), (8, 8))):
        params = ReconParams(af_roi=(0.5, 0.5, 0.5, 0.5))  # zero-area
        field, _, _, _ = workers._prepare_field(
            Path("/dev/null"), params, apply_af_roi=True,
        )
    assert np.count_nonzero(field) == 16 * 16


# ---------------------------------------------------------------------------
# v2.0.9 evening regressions — 5-bug sprint fallout
# ---------------------------------------------------------------------------

def test_autofocus_calls_fft2_once_per_scan():
    """Pin the optimisation: ``_make_fast_evaluator`` must compute
    ``fft2(field)`` **once** and reuse the spectrum for every
    ``evaluate(z)`` call. If someone moves the FFT inside the
    closure we'd silently pay N× the cost on a scan — catch it
    synthetically by counting ``fft2`` invocations."""
    from core.autofocus.evaluator import _make_fast_evaluator
    from core.autofocus import FocusMetric
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    class _CountingFFT:
        """Wraps the real backend and counts fft2 calls."""
        def __init__(self, real):
            self._real = real
            self.name = real.name
            self.fft2_calls = 0
            self.ifft2_calls = 0

        def fft2(self, *args, **kwargs):
            self.fft2_calls += 1
            return self._real.fft2(*args, **kwargs)

        def ifft2(self, *args, **kwargs):
            self.ifft2_calls += 1
            return self._real.ifft2(*args, **kwargs)

    from core.fft_backend import get_best_fft_backend
    real = get_best_fft_backend()
    counting = _CountingFFT(real)

    size = 64
    field = np.random.default_rng(0).standard_normal(
        (size, size),
    ).astype(np.complex64)
    base = ReconstructionParams(
        wavelength_m=632.8e-9, pixel_size_m=2.5e-6, z_m=0.0, n=1.0,
    )

    with patch("core.autofocus.evaluator.get_best_fft_backend",
               return_value=counting):
        evaluator = _make_fast_evaluator(
            field, base, ReconstructionMethod.ASM,
            FocusMetric.ENTROPY,
        )
        # 40 calls — should still be exactly 1 fft2.
        for z in np.linspace(5e-3, 15e-3, 40):
            evaluator(float(z))

    assert counting.fft2_calls == 1, (
        f"fft2 was called {counting.fft2_calls} times for 40 z "
        f"evaluations — the pre-computed spectrum optimisation "
        f"is gone (expected 1 call, reused N times)"
    )
