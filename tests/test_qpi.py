"""QPI derived-quantity validation — OPD↔height round-trip, roughness."""
from __future__ import annotations

import numpy as np
import pytest

from core.qpi import compute_roughness, opd_to_height, phase_to_opd


_WL = 632.8e-9


def test_opd_height_round_trip():
    """height = OPD / Δn  then  OPD = height × Δn  returns the original OPD."""
    rng = np.random.default_rng(0)
    opd = rng.normal(0.0, 100e-9, size=(32, 32)).astype(np.float64)
    n_sample, n_medium = 1.52, 1.0
    height = opd_to_height(opd, n_sample=n_sample, n_medium=n_medium)
    # height_to_opd isn't exposed — invert manually.
    opd_back = height * (n_sample - n_medium)
    assert np.allclose(opd, opd_back, rtol=1e-6, atol=1e-12)


def test_roughness_on_flat_returns_zeros():
    h = np.zeros((64, 64), dtype=np.float64)
    r = compute_roughness(h)
    assert r.Ra == pytest.approx(0.0, abs=1e-15)
    assert r.Rq == pytest.approx(0.0, abs=1e-15)
    assert r.Rz == pytest.approx(0.0, abs=1e-15)
    # The `rq > 1e-10` guard in compute_roughness returns Gaussian defaults for flat input.
    assert r.Rsk == pytest.approx(0.0, abs=1e-12)
    assert r.Rku == pytest.approx(3.0, abs=1e-12)


def test_roughness_no_nan_on_degenerate():
    # Tiny (not flat) input — rq is well under the 1e-10 threshold.
    h = np.full((16, 16), 1e-20, dtype=np.float64)
    r = compute_roughness(h)
    assert np.isfinite(r.Rsk)
    assert np.isfinite(r.Rku)
