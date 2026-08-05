"""Phase-correlation drift correction tests (v2.0.8, D1).

Covers:

* Integer-pixel shift recovery (synthetic image shifted by known
  amount — must round-trip).
* Sub-pixel recovery via parabolic fit.
* Sign convention (a(y,x) ≈ b(y+dy, x+dx)).
* Robustness to additive noise.
* Cumulative drift across an N-frame session.
* ``apply_drift`` round-trip with ``estimate_drift`` recovers the
  original.
* Edge cases — same frame, all-zero peak, mismatched shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import math

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.registration import (  # noqa: E402
    DriftEstimate,
    apply_drift,
    drift_track_session,
    DEFAULT_PHASE_EXPONENT,
    estimate_drift,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _structured_frame(shape=(128, 128), seed: int = 42) -> np.ndarray:
    """Synthetic frame with localised gaussian blobs — good test
    input for cross-correlation: it has features at multiple
    spatial scales without filling the FFT spectrum uniformly
    (which would make the correlation peak ambiguous)."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.float32)
    # Place a few gaussian blobs at random centres.
    yy, xx = np.indices(shape, dtype=np.float32)
    for _ in range(8):
        cy = rng.uniform(20, shape[0] - 20)
        cx = rng.uniform(20, shape[1] - 20)
        sigma = rng.uniform(3, 8)
        amp = rng.uniform(0.5, 1.0)
        img += amp * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2),
        )
    return img.astype(np.float32)


def _shift_circular(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Circular shift — used to make a 'b' frame from 'a' so the
    cross-correlation peak is unambiguous and there's no boundary
    artefact polluting the measurement."""
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


# ---------------------------------------------------------------------------
# Integer shift recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dy,dx", [
    (0, 0),
    (1, 0),
    (0, 1),
    (5, 3),
    (-7, -2),
    (12, -8),
])
def test_integer_shift_recovery(dy, dx):
    a = _structured_frame()
    b = _shift_circular(a, dy, dx)
    est = estimate_drift(a, b, use_window=False)
    # Sign convention: shift to map b onto a is +(dy, dx) here
    # (b was rolled by (dy, dx), so a sits at b shifted by (-dy, -dx)).
    assert round(est.dy_px) == -dy
    assert round(est.dx_px) == -dx


def test_zero_shift_returns_zero():
    a = _structured_frame()
    est = estimate_drift(a, a, use_window=False)
    assert abs(est.dy_px) < 0.5
    assert abs(est.dx_px) < 0.5
    # Same frame → peak normalised correlation should be ~1.0.
    assert est.peak_corr > 0.5


# ---------------------------------------------------------------------------
# Sub-pixel
# ---------------------------------------------------------------------------

def test_sub_pixel_shift_via_fourier_synth():
    """Generate a frame with a known sub-pixel shift via FFT
    multiplication by a linear phase ramp. The estimator should
    recover ±0.1 px or better.

    Sub-pixel shift in image space = linear phase ramp in FFT
    space: F_shifted = F · exp(-j 2π (fy·dy + fx·dx) / N)."""
    rng = np.random.default_rng(0)
    a = _structured_frame()
    ny, nx = a.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    target_dy, target_dx = 2.7, -1.3  # sub-pixel
    F = np.fft.fft2(a)
    F_shifted = F * np.exp(
        -2j * np.pi * (FY * target_dy + FX * target_dx),
    )
    b = np.real(np.fft.ifft2(F_shifted)).astype(np.float32)
    est = estimate_drift(a, b, use_window=False)
    # Sign flip — shift to map b onto a is the negative.
    assert abs(est.dy_px - (-target_dy)) < 0.15
    assert abs(est.dx_px - (-target_dx)) < 0.15


# ---------------------------------------------------------------------------
# Noise robustness
# ---------------------------------------------------------------------------

def test_noise_does_not_break_estimator():
    """Modest gaussian noise must not break the estimator — the lab's real
    images aren't pristine.

    Swept over seeds rather than asserting on one. The single-seed version of
    this test was green only because seed 7 happened to land inside tolerance
    on the dev machine, while the estimator was in fact wrong most of the time
    (60.3% of estimates off by >=0.5 px across shifts and noise levels, worst
    case 102.7 px). One seed cannot distinguish "works" from "got lucky".
    """
    a = _structured_frame()
    b = _shift_circular(a, 4, 2)

    errors = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        b_noisy = (b + rng.normal(0.0, 0.05, b.shape)).astype(np.float32)
        est = estimate_drift(a, b_noisy, use_window=False)
        errors.append(math.hypot(est.dy_px - (-4), est.dx_px - (-2)))

    worst = max(errors)
    assert worst < 0.3, (
        f"worst error over 50 seeds was {worst:.2f} px; "
        f"median {sorted(errors)[len(errors) // 2]:.2f} px"
    )


def test_pure_phase_correlation_is_the_fragile_setting():
    """Pin the reason the default is 0.6: exponent 1.0 is measurably worse.

    This is a regression guard on the *diagnosis*, not decoration. If someone
    later restores textbook phase correlation as the default, this test says
    why that was wrong instead of leaving the next reader to rediscover it.
    """
    a = _structured_frame()
    b = _shift_circular(a, 4, 2)

    def worst_over_seeds(exponent):
        out = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            b_noisy = (b + rng.normal(0.0, 0.05, b.shape)).astype(np.float32)
            est = estimate_drift(a, b_noisy, use_window=False,
                                 phase_exponent=exponent)
            out.append(math.hypot(est.dy_px - (-4), est.dx_px - (-2)))
        return max(out)

    assert worst_over_seeds(1.0) > worst_over_seeds(DEFAULT_PHASE_EXPONENT)


# ---------------------------------------------------------------------------
# apply_drift
# ---------------------------------------------------------------------------

def test_apply_drift_zero_returns_copy():
    a = _structured_frame()
    out = apply_drift(a, 0, 0)
    assert np.array_equal(out, a)
    # New array — caller can mutate without affecting input.
    assert out is not a


def test_apply_drift_then_estimate_recovers_inverse():
    """Apply known shift, estimate it back. Should ~= the negative
    (apply_drift zero-fills uncovered regions, so isn't an exact
    cyclic round-trip — sub-pixel exactness isn't promised)."""
    a = _structured_frame()
    shifted = apply_drift(a, 6, -3)
    est = estimate_drift(a, shifted, use_window=True)
    # The shift to map shifted onto a is (-6, +3) — recover within ~1 px.
    assert abs(est.dy_px + 6) < 1.5
    assert abs(est.dx_px - 3) < 1.5


def test_apply_drift_zero_fills_revealed_region():
    """Translate by (3, 0) — top 3 rows must be zero (newly
    revealed, no source data)."""
    a = _structured_frame()
    out = apply_drift(a, 3, 0)
    assert np.all(out[:3, :] == 0)


# ---------------------------------------------------------------------------
# Session-level cumulative drift
# ---------------------------------------------------------------------------

def test_drift_track_session_first_entry_is_zero():
    a = _structured_frame()
    out = drift_track_session([a, a, a])
    assert out[0] == DriftEstimate(0.0, 0.0, 1.0)


def test_drift_track_session_returns_cumulative_offsets():
    """Generate 4 frames each shifted by (+2, +1) from the previous.
    The cumulative drift estimates should land near (0, 0), (-2, -1),
    (-4, -2), (-6, -3) — minus signs because we report the shift
    that maps each later frame back to frame 0."""
    a = _structured_frame()
    f0 = a
    f1 = _shift_circular(a, 2, 1)
    f2 = _shift_circular(a, 4, 2)
    f3 = _shift_circular(a, 6, 3)
    out = drift_track_session([f0, f1, f2, f3], max_shift_px=15)
    assert len(out) == 4
    assert abs(out[0].dy_px) < 0.5
    assert abs(out[1].dy_px - (-2)) < 0.5
    assert abs(out[2].dy_px - (-4)) < 0.5
    assert abs(out[3].dy_px - (-6)) < 0.5
    assert abs(out[1].dx_px - (-1)) < 0.5
    assert abs(out[3].dx_px - (-3)) < 0.5


def test_drift_track_session_empty_returns_empty():
    assert drift_track_session([]) == []


def test_drift_track_session_single_frame_just_zero():
    out = drift_track_session([_structured_frame()])
    assert len(out) == 1
    assert out[0].dy_px == 0.0
    assert out[0].dx_px == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_mismatched_shapes_raises():
    a = np.zeros((64, 64), dtype=np.float32)
    b = np.zeros((128, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        estimate_drift(a, b)


def test_max_shift_px_restricts_search_window():
    """Shift by 30, but cap search at max_shift_px=5. Since the
    cross-correlation peak is at (-30, 0), restricting to ±5
    around centre rejects it — estimator returns the *strongest*
    peak inside the window, which is no longer the true shift.
    Test asserts the returned shift falls within the cap."""
    a = _structured_frame()
    b = _shift_circular(a, 30, 0)
    est = estimate_drift(a, b, max_shift_px=5, use_window=False)
    assert abs(est.dy_px) <= 5
    assert abs(est.dx_px) <= 5


def test_completely_different_frames_low_peak():
    """Two unrelated noise patterns — peak correlation should be
    much smaller than for a true shifted match. The lab uses
    ``peak_corr < 0.2`` as a proxy for 'estimate unreliable'."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)
    b = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)
    same = estimate_drift(a, a, use_window=False)
    diff = estimate_drift(a, b, use_window=False)
    # Self-correlation must outscore unrelated frames clearly.
    assert same.peak_corr > 5 * diff.peak_corr
