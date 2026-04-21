"""ASM / Fresnel propagation sanity checks — energy, identity, shape."""
from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)


_WL = 632.8e-9
_DX = 5.0e-6


def _fresh_params(z_m: float) -> ReconstructionParams:
    return ReconstructionParams(wavelength_m=_WL, pixel_size_m=_DX, z_m=z_m, n=1.0)


@pytest.mark.slow
def test_asm_round_trip_preserves_energy(synthetic_complex_field):
    field = synthetic_complex_field((128, 128))
    forward = propagate(field, _fresh_params(1e-4), ReconstructionMethod.ASM)
    back = propagate(forward, _fresh_params(-1e-4), ReconstructionMethod.ASM)

    e_in = float(np.sum(np.abs(field) ** 2))
    e_out = float(np.sum(np.abs(back) ** 2))
    assert e_in > 0
    assert e_out / e_in >= 0.99, f"energy ratio {e_out / e_in:.4f} < 0.99"


def test_asm_zero_z_is_identity(synthetic_complex_field):
    field = synthetic_complex_field((64, 64))
    out = propagate(field, _fresh_params(0.0), ReconstructionMethod.ASM)
    assert out.shape == field.shape
    assert np.allclose(out, field, rtol=1e-3, atol=1e-4)


def test_fresnel_output_shape_matches_input(synthetic_complex_field):
    field = synthetic_complex_field((64, 64))
    out = propagate(field, _fresh_params(5e-5), ReconstructionMethod.FRESNEL)
    assert out.shape == field.shape
    assert np.all(np.isfinite(out))
