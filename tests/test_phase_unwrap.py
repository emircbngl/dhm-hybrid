"""Phase-unwrap validation — wrap-diff bounds, smooth-ramp continuity."""
from __future__ import annotations

import numpy as np
import pytest

from core.autofocus.metrics import _wrap_diff
from core.phase_unwrap import UnwrapConfig, UnwrapMethod, unwrap_phase_advanced


def test_wrap_diff_is_in_minus_pi_pi():
    rng = np.random.default_rng(3)
    a = rng.uniform(-10 * np.pi, 10 * np.pi, size=(64, 64))
    b = rng.uniform(-10 * np.pi, 10 * np.pi, size=(64, 64))
    d = _wrap_diff(a, b)
    assert np.all(d >= -np.pi - 1e-9)
    assert np.all(d <= np.pi + 1e-9)


def test_unwrap_smooth_ramp():
    # Build a smooth phase ramp that spans > 2π along x, then wrap it.
    ny, nx = 32, 64
    x = np.linspace(0, 6 * np.pi, nx, dtype=np.float64)
    ramp = np.broadcast_to(x, (ny, nx)).copy()
    wrapped = np.angle(np.exp(1j * ramp)).astype(np.float32)

    cfg = UnwrapConfig(
        method=UnwrapMethod.LEAST_SQUARES,
        post_bg_remove=False,
        pre_median=False,
        pre_gaussian=False,
    )
    out = unwrap_phase_advanced(wrapped, cfg)

    # A smoothly unwrapped ramp must be monotonic along x up to small numerical jitter.
    mean_row = out.mean(axis=0)
    diffs = np.diff(mean_row)
    # Allow direction to be either ascending or descending; check monotonicity.
    assert np.all(diffs >= -1e-3) or np.all(diffs <= 1e-3), (
        f"unwrapped ramp not monotonic: diffs min={diffs.min():.4g} max={diffs.max():.4g}"
    )


def test_wrap_diff_near_boundary():
    # Two values either side of ±π wrap — difference should be a small number,
    # not ≈ 2π as plain subtraction would give.
    a = np.array([np.pi - 0.05])
    b = np.array([-np.pi + 0.05])
    d = _wrap_diff(a, b)
    assert abs(abs(d[0]) - 0.1) < 1e-6 or abs(d[0]) < 0.2
