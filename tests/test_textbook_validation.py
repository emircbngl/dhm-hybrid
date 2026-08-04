"""
Textbook-consistency validation for the DHM reconstruction engine.

Each test pins a CLOSED-FORM analytical result from an optics/holography
textbook and asserts that the numpy engine (src/core) reproduces it. The
load-bearing formulas were machine-verified (units + symbolic + numeric)
with the physicist Docker oracle on 2026-07-05:

    * OPD = phi*lambda/(2*pi)          -> VERIFIED (4/4)
    * h   = OPD/(n_s - n_m)            -> VERIFIED (4/4)
    * Lambda = lambda/(2 sin theta)    -> VERIFIED (3/3), f_c = 2 sin theta/lambda
    * z_T = 2 p^2 / lambda  (Talbot)   -> DIMENSIONAL green + numeric 202.212 um
    * ASM obliquity sqrt(1-(lambda f/n)^2) -> LIMIT ->1 as f->0 (green);
      evanescent cutoff |f| = n/lambda -> DIMENSIONAL green

Sources (real PDFs read 2026-07-05):
    [Kim]    M. K. Kim, *Digital Holographic Microscopy*, Springer 2011.
    [Kreis]  T. Kreis, *Handbook of Holographic Interferometry*, Wiley-VCH 2005.
    [Hecht]  E. Hecht, *Optics*, Ch. 9-10.
    [BW]     Born & Wolf, *Principles of Optics*, 7th ed. (canonical, cross-ref).

Two supplied "book" PDFs were NOT the claimed texts (Born&Wolf file = a 3-page
review; the Schnars&Jüptner "Full-Download" file = a marketing placeholder), so
their formulas are cross-cited from the genuine sources above.

Engine scope boundary (see test_scope_map_out_of_scope): the engine is a
NEAR-FIELD angular-spectrum / convolution-Fresnel propagator. Far-field
Fraunhofer results (Airy 1.22 lambda f/D, Rayleigh/Abbe resolution) and the
single-FFT Fresnel-transform pixel size (dxi = lambda d/(N dx)) are out of
scope for `propagate()` and are documented, not asserted against it.
"""
import numpy as np
import pytest

from core.reconstruction import (
    propagate,
    ReconstructionParams,
    ReconstructionMethod,
    CachedReconstructor,
)
from core.fft_backend import get_best_fft_backend
from core import qpi
from core.offaxis import extract_complex_field_offaxis, OffAxisParams
from core.masking import detect_plus_one_order
from core.phase_unwrap import unwrap_phase_advanced, UnwrapConfig, UnwrapMethod


# --------------------------------------------------------------------------
# QPI algebra (Kim Ch.11; Hecht Ch.9; BW) — machine-verified formulas
# --------------------------------------------------------------------------

def test_opd_from_phase_one_wave():
    """[Kim/Hecht/BW] OPD = phi*lambda/(2pi); phi=2pi -> OPD = lambda."""
    lam = 633e-9
    phase = np.full((16, 16), 2 * np.pi, dtype=np.float64)
    opd = qpi.phase_to_opd(phase, lam)
    assert np.allclose(opd, lam, rtol=1e-9)  # one full wave -> one wavelength


def test_height_from_opd_index_contrast():
    """[Kim Ch.11] h = OPD/(n_s - n_m). phi=2pi, dn=0.03 -> h = 21.1 um."""
    lam = 633e-9
    opd = qpi.phase_to_opd(np.full((8, 8), 2 * np.pi), lam)
    h = qpi.opd_to_height(opd, n_sample=1.36, n_medium=1.33)
    assert np.allclose(h, lam / 0.03, rtol=1e-6)
    assert np.allclose(h, 21.1e-6, rtol=1e-3)


def test_height_reflection_double_pass():
    """[Kim/Hecht] reflection DHM: h = OPD/2 (light traverses height twice)."""
    lam = 550e-9
    opd = qpi.phase_to_opd(np.full((8, 8), np.pi), lam)  # OPD = 275 nm
    h = qpi.opd_to_height_reflection(opd)
    assert np.allclose(h, 275e-9 / 2, rtol=1e-9)


def test_forward_model_roundtrip():
    """[Kreis/BW forward model] phi=(2pi/lambda) dn h ; OPD->height recovers h."""
    lam = 632.8e-9
    dn = 1.38 - 1.335
    h_true = 6.0e-6
    phi = (2 * np.pi / lam) * dn * h_true
    opd = qpi.phase_to_opd(np.full((8, 8), phi), lam)
    h = qpi.opd_to_height(opd, n_sample=1.38, n_medium=1.335)
    assert np.allclose(h, h_true, rtol=1e-6)


# --------------------------------------------------------------------------
# Off-axis carrier: fringe spacing Lambda = lambda/(2 sin theta), +1 order
# [Kim Ch.7; Kreis FTM; Hecht/BW two-beam] — verified formula
# --------------------------------------------------------------------------

def test_offaxis_carrier_localization_and_amplitude():
    """
    Synthesize I = |1 + o*exp(i 2pi f_c x)|^2 (commensurate carrier).
    The +1 order sits exactly N*f_c bins from DC, and the demodulator
    recovers the object amplitude o.
    """
    N = 256
    fc_cyc_per_px = 0.30            # commensurate: 0.30*256 = 76.8 -> ~77
    o = 0.30
    x = np.arange(N)
    X, _ = np.meshgrid(x, x)
    ref = 1.0
    obj = o * np.exp(1j * 2 * np.pi * fc_cyc_per_px * X)
    holo = np.abs(ref + obj) ** 2

    py, px = detect_plus_one_order(holo.astype(np.float32), exclusion_radius=10)
    offset = abs(px - N // 2)
    assert offset == round(fc_cyc_per_px * N)   # 77 bins from DC (exact)

    field, _ = extract_complex_field_offaxis(
        holo, OffAxisParams(radius=40, dc_exclusion_radius=10)
    )
    # central region avoids mask edge ringing
    amp = np.abs(field)[N // 4 : 3 * N // 4, N // 4 : 3 * N // 4].mean()
    assert amp == pytest.approx(o, rel=0.10)


# --------------------------------------------------------------------------
# ASM transfer function: evanescent cutoff |f| = n/lambda, |H| <= 1
# [Kim Ch.4 Eq.4.22; BW angular spectrum] — verified cutoff + obliquity
# --------------------------------------------------------------------------

def test_asm_evanescent_cutoff_and_no_amplification():
    lam, n, dx, N = 500e-9, 1.0, 0.2e-6, 128
    params = ReconstructionParams(wavelength_m=lam, pixel_size_m=dx, z_m=5e-6, n=n)
    rec = CachedReconstructor((N, N), ReconstructionMethod.ASM, get_best_fft_backend())
    rec._ensure_freq_grid(params)

    fy = np.fft.fftfreq(N, d=dx)
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    fsq = FX ** 2 + FY ** 2
    fc = n / lam                                  # cutoff radius (verified)

    propagating = fsq <= (0.98 * fc) ** 2
    evanescent = fsq >= (1.02 * fc) ** 2
    # sqrt_term is real & >0 inside the light cone, exactly 0 outside
    assert np.all(rec._sqrt_term[propagating] > 0)
    assert np.all(rec._sqrt_term[evanescent] == 0.0)

    # No evanescent amplification for either sign of z (engine clamps |H|<=1)
    for z in (+5e-6, -5e-6):
        H = rec._compute_H(ReconstructionParams(lam, dx, z, n))
        assert np.all(np.abs(H) <= 1.0 + 1e-5)


def test_asm_planewave_onaxis_phase():
    """[BW] on-axis (f=0) plane wave accrues phase k z = 2 pi n z / lambda."""
    lam, n, dx, N, z = 500e-9, 1.0, 0.5e-6, 64, 10e-6
    field = np.ones((N, N), dtype=np.complex64)      # DC-only (f=0)
    out = propagate(field, ReconstructionParams(lam, dx, z, n),
                    ReconstructionMethod.ASM)
    got = np.angle(out.mean())
    expected = np.angle(np.exp(1j * (2 * np.pi * n * z / lam)))
    assert got == pytest.approx(expected, abs=2e-3)


# --------------------------------------------------------------------------
# Propagator unitarity: round-trip +z then -z returns the field
# [BW Fresnel/ASM operator is unitary on propagating waves]
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", [ReconstructionMethod.ASM,
                                     ReconstructionMethod.FRESNEL])
def test_propagation_roundtrip_identity(method):
    lam, dx, N, z = 632.8e-9, 0.5e-6, 128, 200e-6
    rng = np.random.default_rng(0)
    # band-limited field (smooth) so no energy sits in the evanescent band
    base = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    spec = np.fft.fft2(base)
    fy = np.fft.fftfreq(N, d=dx); fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    spec[(FX ** 2 + FY ** 2) > (0.4 / lam) ** 2] = 0
    field = np.fft.ifft2(spec).astype(np.complex64)

    fwd = propagate(field, ReconstructionParams(lam, dx, z), method)
    back = propagate(fwd, ReconstructionParams(lam, dx, -z), method)
    rel = np.linalg.norm(back - field) / np.linalg.norm(field)
    assert rel < 1e-3


# --------------------------------------------------------------------------
# Talbot self-imaging: z_T = 2 p^2 / lambda  [Kim 8.4.5; BW 8.6.3; verified]
# --------------------------------------------------------------------------

def _grating_field(N, period_px):
    x = np.arange(N)
    g = 0.5 + 0.5 * np.cos(2 * np.pi * x / period_px)
    return np.tile(g, (N, 1)).astype(np.complex64)


@pytest.mark.parametrize("method", [ReconstructionMethod.FRESNEL,
                                     ReconstructionMethod.ASM])
def test_talbot_self_image(method):
    """At z = z_T the grating self-images (intensity correlates ~1)."""
    lam, dx, N, period_px = 633e-9, 1.0e-6, 512, 8
    p = period_px * dx
    z_T = 2 * p ** 2 / lam                     # 202.2 um (verified)
    field = _grating_field(N, period_px)
    out = propagate(field, ReconstructionParams(lam, dx, z_T), method)

    row_in = np.abs(field[N // 2]) ** 2
    row_out = np.abs(out[N // 2]) ** 2
    corr = np.corrcoef(row_in - row_in.mean(), row_out - row_out.mean())[0, 1]
    assert corr > 0.98


def test_talbot_half_distance_shift():
    """At z_T/2 the self-image is laterally shifted by p/2 (anti-correlated)."""
    lam, dx, N, period_px = 633e-9, 1.0e-6, 512, 8
    p = period_px * dx
    z_half = p ** 2 / lam                       # z_T / 2
    field = _grating_field(N, period_px)
    out = propagate(field, ReconstructionParams(lam, dx, z_half),
                    ReconstructionMethod.FRESNEL)
    row_in = np.abs(field[N // 2]) ** 2
    row_out = np.abs(out[N // 2]) ** 2
    corr = np.corrcoef(row_in - row_in.mean(), row_out - row_out.mean())[0, 1]
    assert corr < -0.85                         # p/2 shift flips the cosine (strong anti-corr)


# --------------------------------------------------------------------------
# Phase unwrapping: wrapped linear ramp -> recovered slope  [Kreis 5.9.2]
# --------------------------------------------------------------------------

def test_phase_unwrap_linear_ramp():
    """A multi-cycle phase ramp survives wrapping and is unwrapped linearly.

    NOTE: the default pipeline enables post_bg_remove (order-2 plane fit), which
    by design subtracts exactly this linear tilt -> flat output. That is correct
    engine behavior (background flattening); to test the *unwrapping* proper
    (removal of 2pi discontinuities) we disable background removal here.
    """
    N = 128
    slope = 0.4                                  # rad/px -> ~8 wraps across N
    x = np.arange(N)
    phi_true = slope * x
    phi_true2d = np.tile(phi_true, (N, 1))
    E = np.exp(1j * phi_true2d).astype(np.complex64)
    wrapped = np.angle(E)
    assert np.ptp(wrapped) > 6.0                 # input really is wrapped (~2pi jumps)
    unwrapped = unwrap_phase_advanced(
        wrapped,
        UnwrapConfig(method=UnwrapMethod.GRADIENT_INTEGRATION, post_bg_remove=False),
        complex_field=E,
    )
    # recovered mean slope along x matches the true ramp (piston offset ignored)
    recovered_slope = np.mean(np.gradient(unwrapped, axis=1))
    assert recovered_slope == pytest.approx(slope, rel=0.02)
    # and the output is genuinely unwrapped (monotone, no 2pi jumps)
    assert np.ptp(unwrapped) == pytest.approx(slope * (N - 1), rel=0.05)


# --------------------------------------------------------------------------
# Scope map: closed forms that are OUT OF SCOPE for the reconstruction engine
# --------------------------------------------------------------------------

@pytest.mark.skip(reason=(
    "Out of engine scope. reconstruction.propagate is a NEAR-FIELD "
    "angular-spectrum/convolution-Fresnel propagator; it does not compute "
    "far-field Fraunhofer intensity (Airy 1.22 lambda f/D, Rayleigh/Abbe "
    "resolution) nor single-FFT Fresnel-transform reconstruction pixel size "
    "(dxi = lambda d/(N dx) — the convolution Fresnel preserves pixel pitch). "
    "These textbook results were extracted for the scope map but are NOT "
    "asserted against this engine."
))
def test_scope_map_out_of_scope():  # pragma: no cover
    pass
