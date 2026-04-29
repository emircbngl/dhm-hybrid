"""Reconstruction accuracy validation — LAB-4 / Lindqvist Physicist.

The lab physicist wanted a documented check that ASM round-trip does not
silently accumulate error beyond a physically-defensible tolerance. This
suite does three things:

1. Round-trip phase RMSE: propagate a synthetic complex field forward,
   then back the same distance, and compare against the original.
2. Defocus-refocus recovery: check that amplitude pattern survives a
   ±z pair across a realistic 40 mm working distance.
3. Pixel-size invariance: accuracy must not fall off a cliff across the
   typical sensor pixel-size range (2–6 µm).

Thresholds are deliberately loose enough to survive numerical noise on
32-bit float grids — the point is to catch *regressions*, not to publish
a metrology claim. NIH benchmark hologram validation is tracked
separately (see :doc:`ACCURACY`).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)


_WL = 632.8e-9       # HeNe laser
_SHAPE = (128, 128)


def _params(z_m: float, *, dx: float = 5.0e-6, n: float = 1.0):
    return ReconstructionParams(
        wavelength_m=_WL, pixel_size_m=dx, z_m=z_m, n=n,
    )


# ---- 1. round-trip RMSE -----------------------------------------------------

@pytest.mark.parametrize("z_m", [1e-4, 1e-3, 1e-2, 4.4e-2])
def test_asm_round_trip_phase_rmse(synthetic_complex_field, z_m):
    """Forward ``+z`` then back ``-z`` must recover the field to within a
    small phase RMSE. The tolerance loosens with distance because
    boundary wrap-around eats a ring of pixels at larger ``z``."""
    field = synthetic_complex_field(_SHAPE)

    fwd = propagate(field, _params(z_m), ReconstructionMethod.ASM)
    back = propagate(fwd, _params(-z_m), ReconstructionMethod.ASM)

    # Work on the centre third of the frame to avoid edge ringing.
    ny, nx = _SHAPE
    y0, y1 = ny // 3, 2 * ny // 3
    x0, x1 = nx // 3, 2 * nx // 3
    phi_ref = np.angle(field[y0:y1, x0:x1])
    phi_back = np.angle(back[y0:y1, x0:x1])

    # Compare via circular distance: wrap-safe.
    d = np.angle(np.exp(1j * (phi_back - phi_ref)))
    rmse = float(np.sqrt(np.mean(d ** 2)))

    # Tight near focus, looser at long z. Empirical bounds from this
    # repo's baseline — any regression that doubles the error will trip.
    tolerance = 0.02 if z_m <= 1e-3 else 0.15
    assert rmse <= tolerance, (
        f"z={z_m*1e3:.2f} mm: phase RMSE {rmse:.4f} rad > {tolerance:.4f}"
    )


# ---- 2. defocus-refocus amplitude recovery ---------------------------------

def test_amplitude_survives_defocus_refocus(synthetic_complex_field):
    """A 40 mm working distance is the common DHM regime. The amplitude
    pattern must return within a few percent after ``+z``/``-z``."""
    field = synthetic_complex_field(_SHAPE)
    z = 40e-3

    fwd = propagate(field, _params(z), ReconstructionMethod.ASM)
    back = propagate(fwd, _params(-z), ReconstructionMethod.ASM)

    amp_ref = np.abs(field)
    amp_back = np.abs(back)

    # Centre-crop (same reason as above).
    ny, nx = _SHAPE
    c = (slice(ny // 4, 3 * ny // 4), slice(nx // 4, 3 * nx // 4))
    num = float(np.sum((amp_back[c] - amp_ref[c]) ** 2))
    den = float(np.sum(amp_ref[c] ** 2))
    assert den > 0
    rel_err = np.sqrt(num / den)
    assert rel_err <= 0.10, f"relative amplitude error {rel_err:.3f} > 0.10"


# ---- 3. pixel-size sweep ---------------------------------------------------

@pytest.mark.parametrize("dx_um", [2.0, 3.45, 5.5])
def test_round_trip_stable_across_pixel_sizes(synthetic_complex_field, dx_um):
    """Sensor pixel sizes in the 2–6 µm range should all round-trip
    cleanly at typical ``z``. Catches ``float32`` grid-precision
    regressions in the frequency-grid cache."""
    field = synthetic_complex_field(_SHAPE)
    dx = dx_um * 1e-6
    z = 5e-3

    fwd = propagate(field, _params(z, dx=dx), ReconstructionMethod.ASM)
    back = propagate(fwd, _params(-z, dx=dx), ReconstructionMethod.ASM)

    e_in = float(np.sum(np.abs(field) ** 2))
    e_out = float(np.sum(np.abs(back) ** 2))
    assert e_in > 0
    energy_ratio = e_out / e_in
    assert 0.98 <= energy_ratio <= 1.02, (
        f"dx={dx_um} µm: energy ratio {energy_ratio:.4f} drifted"
    )


# ---- 4. z = 0 is a perfect identity ---------------------------------------

def test_asm_zero_z_reproduces_field_exactly(synthetic_complex_field):
    """At ``z = 0`` the transfer function is 1 everywhere — the output
    must equal the input within round-off."""
    field = synthetic_complex_field((64, 64))
    out = propagate(field, _params(0.0), ReconstructionMethod.ASM)

    max_abs_err = float(np.max(np.abs(out - field)))
    assert max_abs_err < 1e-4, f"z=0 residual {max_abs_err:.2e}"
