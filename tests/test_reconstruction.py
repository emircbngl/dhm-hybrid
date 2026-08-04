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


def test_spectrum_cache_survives_id_reuse():
    """Regression (2026-07-05): the global propagate() spectrum cache keyed
    on a bare id(field). Python reuses ids after GC, so a same-shape array
    allocated after the cached field died could collide and be served the
    PREVIOUS field's spectrum — silently reconstructing the wrong frame
    (batch/timelapse loops free+realloc frames constantly). The cache now
    holds a weakref, which can never match a new object.
    """
    from core.reconstruction import (
        propagate, ReconstructionParams, ReconstructionMethod,
    )

    params = ReconstructionParams(wavelength_m=633e-9, pixel_size_m=1e-6,
                                  z_m=1e-4)
    a = np.ones((64, 64), dtype=np.complex64)
    out_a = np.abs(propagate(a, params, ReconstructionMethod.ASM)).copy()
    stale_id = id(a)
    del a

    # Try to land a new same-shape array on the freed object slot; CPython
    # usually reuses it on the first allocation. If no collision happens
    # the assertion below still holds — correctness must not depend on it.
    b = None
    for _ in range(64):
        b = np.full((64, 64), 2.0, dtype=np.complex64)
        if id(b) == stale_id:
            break

    out_b = np.abs(propagate(b, params, ReconstructionMethod.ASM))
    # Linear propagator: doubling the field must double the output. A
    # poisoned cache would return out_a's magnitude instead.
    assert np.allclose(out_b, 2.0 * out_a, rtol=1e-4, atol=1e-6), (
        "propagate() served a stale spectrum for a different field "
        "(id-reuse cache poisoning)"
    )
