"""Multi-line profile sampler tests (v2.0.8, D4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.line_profile import (  # noqa: E402
    LineProfile,
    SampledProfile,
    sample_line,
    sample_profiles,
    stats_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ramp_image(shape=(64, 128)) -> np.ndarray:
    """Image whose value equals its column index — easy to verify
    sampling along a horizontal line returns 0..N."""
    ny, nx = shape
    img = np.tile(np.arange(nx, dtype=np.float32), (ny, 1))
    return img


# ---------------------------------------------------------------------------
# sample_line — basic
# ---------------------------------------------------------------------------

def test_horizontal_line_returns_column_indices():
    """Image where I(y, x) = x. A horizontal line at constant y
    should sample 0, 1, 2, ... up to the line's length."""
    img = _ramp_image()
    line = LineProfile(y0=10, x0=0, y1=10, x1=10, n_samples=11)
    sp = sample_line(img, line)
    assert sp.values.shape == (11,)
    assert sp.values[0] == pytest.approx(0.0)
    assert sp.values[-1] == pytest.approx(10.0)
    assert np.allclose(sp.values, np.linspace(0, 10, 11))


def test_distance_axis_matches_line_length():
    """SampledProfile.distance_px[-1] should equal the line's
    geometric length."""
    img = np.zeros((50, 50), dtype=np.float32)
    line = LineProfile(y0=10, x0=10, y1=40, x1=50, n_samples=20)
    sp = sample_line(img, line)
    assert sp.distance_px[0] == pytest.approx(0.0)
    assert sp.distance_px[-1] == pytest.approx(line.length_px)


def test_diagonal_line_sub_pixel_bilinear():
    """A 45° diagonal samples both x and y by sub-pixel offsets.
    Bilinear interpolation must produce smooth values, not
    integer-quantised stairsteps."""
    # Image: I(y, x) = y + x, so a diagonal line returns
    # values that increase linearly.
    ny, nx = 50, 50
    y, x = np.indices((ny, nx), dtype=np.float32)
    img = (y + x).astype(np.float32)
    line = LineProfile(y0=0, x0=0, y1=20, x1=20, n_samples=21)
    sp = sample_line(img, line)
    # Endpoints exact (on-grid).
    assert sp.values[0] == pytest.approx(0.0)
    assert sp.values[-1] == pytest.approx(40.0)
    # Mid-points should be ~half — bilinear delivers smooth ramps.
    assert sp.values[10] == pytest.approx(20.0, abs=0.5)


def test_auto_n_samples_picks_one_per_pixel():
    """``n_samples=0`` triggers auto = round(length) + 1."""
    img = np.zeros((50, 50), dtype=np.float32)
    line = LineProfile(y0=0, x0=0, y1=0, x1=10, n_samples=0)
    sp = sample_line(img, line)
    assert sp.values.size == 11


def test_zero_length_line_returns_two_samples():
    """Same start + end. Auto-sample should fall back to the
    minimum 2 and both values must be in-bounds (not NaN)."""
    img = np.ones((20, 20), dtype=np.float32)
    line = LineProfile(y0=5, x0=5, y1=5, x1=5, n_samples=0)
    sp = sample_line(img, line)
    assert sp.values.size == 2
    assert np.all(np.isfinite(sp.values))
    assert sp.values[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Out-of-bounds handling
# ---------------------------------------------------------------------------

def test_out_of_bounds_samples_are_nan():
    img = np.ones((20, 20), dtype=np.float32)
    line = LineProfile(y0=10, x0=15, y1=10, x1=30, n_samples=16)
    sp = sample_line(img, line)
    # First few samples in bounds, later ones (x > 19) → NaN.
    assert np.isfinite(sp.values[0])
    assert np.isnan(sp.values[-1])


def test_custom_fill_value():
    img = np.ones((20, 20), dtype=np.float32)
    line = LineProfile(y0=10, x0=15, y1=10, x1=30, n_samples=16)
    sp = sample_line(img, line, fill_value=-1.0)
    # No NaNs anywhere because we picked a sentinel.
    assert not np.any(np.isnan(sp.values))
    # Far end of the line (out-of-bounds) → -1.
    assert sp.values[-1] == -1.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_non_2d_image_raises():
    line = LineProfile(y0=0, x0=0, y1=1, x1=1)
    with pytest.raises(ValueError, match="2-D"):
        sample_line(np.zeros((4, 4, 4), dtype=np.float32), line)


# ---------------------------------------------------------------------------
# sample_profiles
# ---------------------------------------------------------------------------

def test_sample_profiles_preserves_order():
    img = _ramp_image()
    profiles = [
        LineProfile(y0=10, x0=0, y1=10, x1=10, label="A"),
        LineProfile(y0=20, x0=0, y1=20, x1=20, label="B"),
        LineProfile(y0=30, x0=10, y1=30, x1=30, label="C"),
    ]
    out = sample_profiles(img, profiles)
    assert len(out) == 3
    assert out[0].profile.label == "A"
    assert out[1].profile.label == "B"
    assert out[2].profile.label == "C"


def test_sample_profiles_each_returns_sampled_profile():
    img = _ramp_image()
    profiles = [LineProfile(y0=0, x0=0, y1=0, x1=5)]
    out = sample_profiles(img, profiles)
    assert isinstance(out[0], SampledProfile)


# ---------------------------------------------------------------------------
# stats_for
# ---------------------------------------------------------------------------

def test_stats_for_handles_nan_cleanly():
    """Out-of-bounds samples (NaN) must not poison min/max/mean/std."""
    profile = LineProfile(y0=0, x0=0, y1=0, x1=10, n_samples=11)
    distances = np.linspace(0, 10, 11)
    values = np.array([1.0, 2.0, 3.0, np.nan, np.nan, 5.0,
                       6.0, 7.0, 8.0, 9.0, 10.0])
    sp = SampledProfile(profile=profile, distance_px=distances,
                        values=values)
    stats = stats_for(sp)
    assert stats["min"] == 1.0
    assert stats["max"] == 10.0
    assert stats["n"] == 9
    assert stats["mean"] == pytest.approx(np.nanmean(values))


def test_stats_for_all_nan_returns_none_keys():
    profile = LineProfile(y0=0, x0=0, y1=0, x1=5)
    sp = SampledProfile(
        profile=profile,
        distance_px=np.linspace(0, 5, 5),
        values=np.full(5, np.nan),
    )
    stats = stats_for(sp)
    assert stats["min"] is None
    assert stats["mean"] is None
    assert stats["n"] == 0


def test_stats_for_empty_returns_zero_count():
    profile = LineProfile(y0=0, x0=0, y1=0, x1=0)
    sp = SampledProfile(
        profile=profile,
        distance_px=np.array([]),
        values=np.array([]),
    )
    stats = stats_for(sp)
    assert stats["n"] == 0


# ---------------------------------------------------------------------------
# LineProfile geometry helper
# ---------------------------------------------------------------------------

def test_line_length_diagonal():
    line = LineProfile(y0=0, x0=0, y1=3, x1=4)
    assert line.length_px == pytest.approx(5.0)


def test_line_length_zero_when_endpoints_match():
    line = LineProfile(y0=10, x0=10, y1=10, x1=10)
    assert line.length_px == 0.0
