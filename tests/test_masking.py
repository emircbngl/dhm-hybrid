"""circular_mask: hard/tukey/hann/gaussian apodizations."""
from __future__ import annotations

import numpy as np
import pytest

from core.masking import circular_mask


SHAPE = (128, 128)
CENTER = (64, 64)
RADIUS = 20


def test_hard_mask_is_binary():
    m = circular_mask(SHAPE, CENTER, RADIUS, apodization="none")
    uniq = np.unique(m)
    # Only 0 and 1 should appear.
    assert set(uniq.tolist()).issubset({0.0, 1.0})
    assert 1.0 in uniq and 0.0 in uniq


def test_tukey_rolloff_zero_falls_back_to_hard():
    m = circular_mask(SHAPE, CENTER, RADIUS, apodization="tukey", rolloff=0.0)
    assert float(m.max()) == 1.0
    assert float(m.min()) == 0.0
    # Boundary should still be binary (hard fallback), no intermediate values.
    assert set(np.unique(m).tolist()).issubset({0.0, 1.0})


def test_hann_mask_is_smooth():
    m = circular_mask(SHAPE, CENTER, RADIUS, apodization="hann")
    gy, gx = np.gradient(m.astype(np.float64))
    # Smooth taper: max |gradient| stays well below 1 per pixel.
    max_grad = float(max(np.abs(gy).max(), np.abs(gx).max()))
    assert max_grad < 0.5, f"hann mask not smooth: max grad = {max_grad}"


def test_gaussian_mask_normalized():
    m = circular_mask(SHAPE, CENTER, RADIUS, apodization="gaussian", rolloff=0.5)
    # Peak at the centre should be ~1.0.
    assert abs(float(m[CENTER]) - 1.0) < 1e-3
    assert float(m.max()) == pytest.approx(1.0, abs=1e-3)
