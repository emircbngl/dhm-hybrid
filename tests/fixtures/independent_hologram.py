"""An independent forward model, written to NOT be the reconstruction.

``tests/fixtures/synthetic_hologram.py`` calls ``core.reconstruction.propagate``
to build its scenes, and the autofocus pipeline calls the same function to read
them back. Every score measured on that fixture therefore describes ASM
inverting ASM — the classic inverse crime. Numbers from it bound the software,
not the optics.

This module exists to break that. It shares no propagation code with the
reconstruction path and imports nothing from ``core``:

* **Direct Rayleigh-Sommerfeld summation, no FFT at all.** Each object-plane
  sample is a Huygens secondary source and the field at every sensor pixel is
  the explicit sum over sources. The reconstruction multiplies an angular
  spectrum by a transfer function; this evaluates a diffraction integral
  pixel by pixel. Different mathematics, different discretisation, different
  error behaviour — and no shared transfer function to cancel out.
* **A different object-plane pitch from the sensor pitch.** The reconstruction
  grid can never be the generator grid, so grid-aligned artefacts cannot
  round-trip.
* **A detector chain the ASM fixture has none of**: Poisson shot noise (the
  dominant real term, where the old fixture had only Gaussian read noise),
  sensor MTF, Gaussian illumination falloff, a curved rather than perfectly
  planar reference, a smooth low-order phase error, optional speckle, full-well
  saturation and quantisation.

What it does NOT claim
----------------------
This is a better forward model, not a metrology reference. It keeps the thin
projected phase-screen object approximation (no refraction inside the sphere,
no multiple scattering, scalar fields only), and the aberration term is a
phenomenological low-order phase, not a traced lens. It is honest about being a
*second, independent* approximation whose errors are uncorrelated with the
reconstruction's — which is exactly what is needed to tell how much of a score
came from the inverse crime.

Cost: O(sensor_pixels x source_points). A 256x256 sensor with a 10 um sphere
sampled at 1 um is ~65k x 400 complex exponentials, a few seconds. That is the
price of not sharing an FFT with the thing under test.

Provenance of the physics — UNVERIFIED (2026-08-17)
---------------------------------------------------
The expressions below were written from memory; no symbolic or literature
verification service was available in the session that produced this file.
They are flagged here rather than presented as settled:

* first Rayleigh-Sommerfeld integral,
  ``U = (1/(i*lambda)) * sum a_i * (z/r_i) * exp(i*k*r_i) / r_i * dA``
* wavelength in medium ``lambda_med = lambda_0 / n_medium``, ``k = 2*pi/lambda_med``
* sphere chord ``2*sqrt(R^2 - r^2)`` and phase ``(2*pi/lambda_0)*(n_s - n_m)*chord``
* paraxial curved reference ``k*(x^2 + y^2)/(2*R)``
* scattered field as ``t - 1`` (thin-screen/Born-style weak-object form)

What DOES substantiate them is a cross-check, not an authority: this forward
model and ``core.reconstruction``'s angular-spectrum propagator are independent
implementations of the same diffraction physics, and RS-forward followed by
ASM-backward recovers the true plane to a median 0.375 mm across nine scenes
(3 radii x 3 distances). Two independent implementations agreeing is real
evidence and is why the fixture is usable; it is not proof that either formula
is written correctly. Anyone with a verification path should check the five
expressions above before quoting absolute accuracy from this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Deliberately no `from core...` import anywhere in this module.
# tests/test_independent_hologram.py asserts that structurally.


@dataclass(frozen=True)
class IndependentSphere:
    """One dielectric sphere, described independently of SphereSpec."""

    radius_m: float
    z_m: float
    center_yx_m: tuple[float, float] = (0.0, 0.0)
    n_sphere: float = 1.40
    n_medium: float = 1.33


@dataclass(frozen=True)
class IndependentConfig:
    """Sensor, illumination and detector-chain parameters.

    Defaults are the same optics as the ASM fixture (632.8 nm, 2.5 um pitch,
    n=1.33, four-pixel diagonal carrier) so results are comparable; everything
    below ``object_amplitude`` is a degradation the ASM fixture does not model.
    """

    shape: tuple[int, int] = (256, 256)
    pixel_m: float = 2.5e-6
    wavelength_m: float = 632.8e-9
    n_medium: float = 1.33
    # Object-plane sampling. Intentionally NOT equal to pixel_m.
    source_pitch_m: float = 1.0e-6
    # Support radius as a multiple of the sphere radius.
    source_pad: float = 1.25
    carrier_period_px: float = 4.0
    # 1.0 = let the physics set the contrast. The Rayleigh-Sommerfeld sum
    # already yields the PHYSICAL scattered amplitude (~0.013 of the incident
    # wave for a 10 um sphere at 12 mm), unlike the ASM fixture whose
    # propagated transmittance is order 1 and needs an explicit 1/50 to
    # represent a weak object. Applying that 1/50 here too buries the object
    # at ~2 counts of 4095 against ~30 counts of shot noise. Keep the knob for
    # deliberately sweeping object strength; do not use it to "match" the ASM
    # fixture's number.
    object_amplitude: float = 1.0
    reference_amplitude: float = 1.0

    # --- detector chain / degradations (absent from the ASM fixture) ---
    photons_full_well: float = 18_000.0
    shot_noise: bool = True
    read_noise_e: float = 6.0
    # Gaussian MTF sigma in pixels; 0 disables. Emulates pixel crosstalk and
    # charge diffusion on top of the geometric pixel aperture.
    mtf_sigma_px: float = 0.45
    # Gaussian illumination falloff, as a fraction of the frame half-width.
    illumination_sigma_frac: float = 0.8
    # Reference wave curvature. inf = perfectly planar (the ASM fixture's
    # assumption); a finite radius gives the spherical reference real rigs have.
    reference_radius_m: float = 0.35
    # Smooth low-order phase error on the object wave, radians peak.
    aberration_rad: float = 0.4
    # Random phase roughness on the object, radians RMS. 0 = smooth.
    speckle_rad: float = 0.0
    saturate: bool = True
    bit_depth: Optional[int] = 12


@dataclass(frozen=True)
class IndependentResult:
    hologram: np.ndarray
    peak_photons: float
    saturated_fraction: float
    source_count: int


def _sphere_sources(
    sphere: IndependentSphere,
    config: IndependentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Object-plane Huygens sources for one sphere.

    Returns ``(x, y, amplitude)``. The amplitude is the *scattered* field
    ``t - 1``: the incident wave is carried separately, so the object
    contributes only its deviation from free propagation. Summing ``t`` itself
    would add one copy of the incident background per sphere, which is the bug
    the ASM fixture's ``legacy_sum`` mode had.
    """
    pitch = config.source_pitch_m
    half = sphere.radius_m * config.source_pad
    n = max(3, int(np.ceil(2.0 * half / pitch)))
    axis = (np.arange(n) - (n - 1) / 2.0) * pitch
    yy, xx = np.meshgrid(axis, axis, indexing="ij")

    rr_sq = yy ** 2 + xx ** 2
    inside = rr_sq < sphere.radius_m ** 2
    # Chord length through a sphere at lateral distance r: 2*sqrt(R^2 - r^2).
    chord = np.zeros_like(rr_sq)
    chord[inside] = 2.0 * np.sqrt(sphere.radius_m ** 2 - rr_sq[inside])
    dn = sphere.n_sphere - sphere.n_medium
    phase = (2.0 * np.pi / config.wavelength_m) * dn * chord

    if config.speckle_rad > 0.0:
        phase = phase + rng.normal(0.0, config.speckle_rad, size=phase.shape)

    scattered = np.exp(1j * phase) - 1.0
    keep = np.abs(scattered) > 1e-9
    cy, cx = sphere.center_yx_m
    return (
        (xx[keep] + cx).astype(np.float64),
        (yy[keep] + cy).astype(np.float64),
        scattered[keep].astype(np.complex128),
    )


def rayleigh_sommerfeld_sum(
    src_x: np.ndarray,
    src_y: np.ndarray,
    src_amp: np.ndarray,
    z_m: float,
    config: IndependentConfig,
    *,
    chunk: int = 64,
) -> np.ndarray:
    """First Rayleigh-Sommerfeld integral, evaluated as an explicit sum.

    ``U(X,Y) = (1/(i*lambda)) * sum_i a_i * (z/r_i) * exp(i*k*r_i)/r_i * dA``

    No FFT, no transfer function, no padding rules — nothing this shares with
    ``core.reconstruction.propagate``. Chunked over sources to bound memory:
    the full outer product would be ``sensor_pixels x sources`` complex128.
    """
    ny, nx = config.shape
    ax_y = (np.arange(ny) - (ny - 1) / 2.0) * config.pixel_m
    ax_x = (np.arange(nx) - (nx - 1) / 2.0) * config.pixel_m
    grid_y, grid_x = np.meshgrid(ax_y, ax_x, indexing="ij")
    flat_y = grid_y.reshape(-1, 1)
    flat_x = grid_x.reshape(-1, 1)

    lambda_med = config.wavelength_m / config.n_medium
    k = 2.0 * np.pi / lambda_med
    d_area = config.source_pitch_m ** 2

    total = np.zeros(flat_y.shape[0], dtype=np.complex128)
    for start in range(0, src_x.size, chunk):
        stop = start + chunk
        dx = flat_x - src_x[start:stop][None, :]
        dy = flat_y - src_y[start:stop][None, :]
        # r >= |z_m| > 0 by construction, so the divisions below are safe.
        r = np.sqrt(dx * dx + dy * dy + z_m * z_m)
        kernel = (z_m / r) * np.exp(1j * k * r) / r
        # macOS Accelerate raises divide/overflow/invalid FPU flags from inside
        # complex matmul even though matmul performs no division and the output
        # is exact: verified 2026-08-17 that r.min()=0.012 (no zeros), the
        # kernel is finite, and matmul agrees with einsum to 6.4e-12 on values
        # of order 6e3 (~1e-15 relative). Suppress the spurious flags, then
        # assert finiteness below so a genuine blow-up still fails loudly.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            total += kernel @ src_amp[start:stop]

    if not np.all(np.isfinite(total)):
        raise FloatingPointError(
            "Rayleigh-Sommerfeld sum produced non-finite values; "
            f"z_m={z_m!r}, sources={src_x.size}"
        )

    total *= d_area / (1j * lambda_med)
    return total.reshape(ny, nx)


def _reference_wave(config: IndependentConfig) -> np.ndarray:
    """Tilted reference with curvature and Gaussian illumination falloff."""
    ny, nx = config.shape
    ax_y = (np.arange(ny) - (ny - 1) / 2.0) * config.pixel_m
    ax_x = (np.arange(nx) - (nx - 1) / 2.0) * config.pixel_m
    grid_y, grid_x = np.meshgrid(ax_y, ax_x, indexing="ij")

    # Diagonal carrier: one fringe every `carrier_period_px` pixels.
    freq = 1.0 / (config.carrier_period_px * config.pixel_m * np.sqrt(2.0))
    phase = 2.0 * np.pi * freq * (grid_x + grid_y)

    if np.isfinite(config.reference_radius_m) and config.reference_radius_m > 0:
        lambda_med = config.wavelength_m / config.n_medium
        k = 2.0 * np.pi / lambda_med
        phase = phase + k * (grid_x ** 2 + grid_y ** 2) / (
            2.0 * config.reference_radius_m)

    envelope = np.ones_like(grid_x)
    if config.illumination_sigma_frac > 0:
        sigma = config.illumination_sigma_frac * (nx * config.pixel_m / 2.0)
        envelope = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))

    return (config.reference_amplitude * envelope
            * np.exp(1j * phase)).astype(np.complex128)


def _aberration_phase(config: IndependentConfig) -> np.ndarray:
    """Smooth low-order phase error (defocus + astigmatism + coma-like).

    Phenomenological: this rig is lensless, so there is no pupil to trace.
    The point is a slowly varying phase the reconstruction does not know about.
    """
    if config.aberration_rad == 0.0:
        return np.zeros(config.shape)
    ny, nx = config.shape
    ax_y = np.linspace(-1.0, 1.0, ny)
    ax_x = np.linspace(-1.0, 1.0, nx)
    yy, xx = np.meshgrid(ax_y, ax_x, indexing="ij")
    rho_sq = xx ** 2 + yy ** 2
    shape_term = (0.6 * (2.0 * rho_sq - 1.0)          # defocus
                  + 0.3 * (xx ** 2 - yy ** 2)          # astigmatism
                  + 0.1 * xx * (3.0 * rho_sq))         # coma-like
    return config.aberration_rad * shape_term


def _apply_mtf(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur via direct convolution (no FFT, no scipy)."""
    if sigma_px <= 0.0:
        return image
    radius = max(1, int(np.ceil(3.0 * sigma_px)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma_px ** 2))
    kernel /= kernel.sum()

    padded = np.pad(image, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(image)
    for weight, shift in zip(kernel, offsets):
        out += weight * padded[radius + shift: radius + shift + image.shape[0]]
    padded = np.pad(out, ((0, 0), (radius, radius)), mode="edge")
    blurred = np.zeros_like(image)
    for weight, shift in zip(kernel, offsets):
        blurred += weight * padded[
            :, radius + shift: radius + shift + image.shape[1]]
    return blurred


def build_independent_hologram(
    spheres: Sequence[IndependentSphere],
    config: Optional[IndependentConfig] = None,
    *,
    rng: Optional[np.random.Generator] = None,
) -> IndependentResult:
    """Scene -> object wave -> interference -> detector -> digital counts."""
    cfg = config or IndependentConfig()
    rng = rng or np.random.default_rng(0)
    if not spheres:
        raise ValueError("at least one sphere is required")

    object_wave = np.zeros(cfg.shape, dtype=np.complex128)
    source_count = 0
    for sphere in spheres:
        src_x, src_y, src_amp = _sphere_sources(sphere, cfg, rng)
        source_count += int(src_amp.size)
        if src_amp.size == 0:
            continue
        object_wave += rayleigh_sommerfeld_sum(
            src_x, src_y, src_amp, sphere.z_m, cfg)

    object_wave *= cfg.object_amplitude
    object_wave *= np.exp(1j * _aberration_phase(cfg))

    total = object_wave + _reference_wave(cfg)
    intensity = np.abs(total) ** 2

    intensity = _apply_mtf(intensity, cfg.mtf_sigma_px)

    # --- detector: photons -> shot noise -> read noise -> saturation -> ADC ---
    peak = float(intensity.max())
    if peak <= 0.0:
        raise ValueError("degenerate scene produced zero intensity")
    photons = intensity * (cfg.photons_full_well / peak)

    if cfg.shot_noise:
        photons = rng.poisson(np.clip(photons, 0.0, None)).astype(np.float64)
    if cfg.read_noise_e > 0.0:
        photons = photons + rng.normal(0.0, cfg.read_noise_e, photons.shape)
    photons = np.clip(photons, 0.0, None)

    saturated_fraction = 0.0
    if cfg.saturate:
        over = photons > cfg.photons_full_well
        saturated_fraction = float(over.mean())
        photons = np.minimum(photons, cfg.photons_full_well)

    if cfg.bit_depth:
        levels = (1 << cfg.bit_depth) - 1
        counts = np.round(photons / cfg.photons_full_well * levels)
        counts = np.clip(counts, 0, levels)
    else:
        counts = photons

    return IndependentResult(
        hologram=counts.astype(np.float32),
        peak_photons=float(cfg.photons_full_well),
        saturated_fraction=saturated_fraction,
        source_count=source_count,
    )
