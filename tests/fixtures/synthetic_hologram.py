"""Synthetic hologram generator for pipeline stress tests.

Models each dielectric sphere as a **projected thin phase screen**, matching
the companion MATLAB simulator's projection approximation. ``n_slices`` is
only a numerical integration of the sphere chord/optical path; there is no
propagation between slices, so this must not be described as split-step or a
volumetric multiple-scattering model.

Pipeline composition:

* Each sphere is represented as a projected thin phase screen.
* Screens at the same z are multiplied; distinct z planes are ordered from
  farthest to nearest and separated by ASM propagation. This avoids the old
  fixture bug that summed one copy of the incident plane wave per object.
* Off-axis reference plane wave added, magnitude squared gives intensity.
* Optional Gaussian camera read-noise.

Intentionally *not* a rigorous simulator. Multi-scattering, vector
effects, aberrations, and 4f magnification (MATLAB reference uses
M=50×) are out of scope — we need ground-truth inputs for the
reconstruction / autofocus / QPI pipeline, not a metrology tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from core.reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)


# ---------------------------------------------------------------------------
# Dataclasses for scene specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SphereSpec:
    """One sphere in a synthetic scene.

    Attributes
    ----------
    radius_m
        Physical sphere radius in metres.
    z_m
        Distance from sphere to camera plane (positive = sphere is in
        front of the camera).
    center_yx_m
        Lateral offset of the sphere from the frame centre, in metres.
        ``(0, 0)`` = dead centre.
    n_sphere
        Refractive index inside the sphere (default 1.40 — typical
        mammalian-cell refractive index).
    n_medium
        Surrounding medium refractive index (default 1.33 — water /
        standard culture buffer).
    n_slices
        Number of axial samples used to integrate the projected chord.
    """
    radius_m: float
    z_m: float
    center_yx_m: Tuple[float, float] = (0.0, 0.0)
    n_sphere: float = 1.40
    n_medium: float = 1.33
    n_slices: int = 32


@dataclass(frozen=True)
class HologramConfig:
    """Camera + laser + off-axis geometry."""
    shape: Tuple[int, int] = (128, 128)
    pixel_m: float = 5.0e-6
    wavelength_m: float = 632.8e-9
    # Carrier frequency of the reference tilt. Set well away from DC
    # so +1 order masking has breathing room; keep below Nyquist/2.
    carrier_freq_m_inv: Tuple[float, float] = (25_000.0, 25_000.0)  # cycles/m
    # Reference amplitude relative to object — large reference is
    # standard (holographic linearisation); tune if SNR testing needed.
    reference_amplitude: float = 1.0
    # Gaussian read noise sigma as a fraction of peak intensity.
    noise_sigma: float = 0.0
    # Object/reference balance. ``1/50`` mirrors the companion MATLAB
    # simulator's 50x 4f amplitude scaling; legacy tests may keep 1.0.
    object_amplitude: float = 1.0
    # Optional camera ADC quantisation. None preserves continuous float data.
    bit_depth: Optional[int] = None
    # ``multiplane`` is the corrected phase-screen model. ``legacy_sum`` is
    # retained only to reproduce historical fixture outputs when diagnosing a
    # regression; it duplicates the incident background for every object.
    composition_mode: str = "multiplane"


# ---------------------------------------------------------------------------
# Primitives — thin-phase sphere at z = 0
# ---------------------------------------------------------------------------

def sphere_phase_field(
    shape: Tuple[int, int],
    pixel_m: float,
    radius_m: float,
    wavelength_m: float,
    *,
    center_yx_m: Tuple[float, float] = (0.0, 0.0),
    n_sphere: float = 1.40,
    n_medium: float = 1.33,
    n_slices: int = 32,
) -> np.ndarray:
    """Return the projected thin-phase field for one sphere at ``z=0``.

    The sphere is discretised into ``n_slices`` axial slabs from ``-r``
    to ``+r``. Each slab multiplies the running field by
    ``exp(i · k0 · Δn · dz · inside(x, y, z_slice))`` — the paraxial
    phase-screen picture. Integrating across slabs recovers the closed-form
    OPL ``2r · (n_s − n_m)`` at the centre. Because the slabs are not
    separated by propagation, this remains a 2-D projection approximation.

    The loop is short (``n_slices`` iterations of one boolean mask +
    complex multiply per slab), so even 32 slices run in under 10 ms
    for a 256×256 grid.
    """
    ny, nx = shape
    y_idx = np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0
    x_idx = np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0
    cy_m, cx_m = center_yx_m
    y_m = y_idx * pixel_m - cy_m
    x_m = x_idx * pixel_m - cx_m
    Y, X = np.meshgrid(y_m, x_m, indexing="ij")
    rho2 = X * X + Y * Y

    k0 = 2.0 * np.pi / wavelength_m
    dn = float(n_sphere - n_medium)
    n_slices = max(int(n_slices), 1)

    if n_slices == 1:
        # Degenerate case — fall back to the 2D thin-phase integral.
        chord = 2.0 * np.sqrt(np.maximum(radius_m * radius_m - rho2, 0.0))
        phase = k0 * dn * chord
        return np.exp(1j * phase).astype(np.complex64)

    z_arr = np.linspace(-radius_m, radius_m, n_slices, dtype=np.float32)
    dz = float(z_arr[1] - z_arr[0])

    field = np.ones(shape, dtype=np.complex64)
    step_phase_scale = k0 * dn * dz
    for z_slice in z_arr:
        inside = rho2 + z_slice * z_slice <= radius_m * radius_m
        # One complex multiply per slab; no intermediate arrays kept.
        field *= np.exp(1j * step_phase_scale * inside).astype(np.complex64)
    return field


def propagate_sphere_to_camera(
    sphere_field: np.ndarray,
    *,
    z_m: float,
    pixel_m: float,
    wavelength_m: float,
    n_medium: float = 1.33,
) -> np.ndarray:
    """Forward-propagate a z=0 sphere field to the camera plane via ASM."""
    params = ReconstructionParams(
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_m,
        z_m=z_m,
        n=n_medium,
    )
    return propagate(sphere_field, params, ReconstructionMethod.ASM)


def reference_plane_wave(
    shape: Tuple[int, int],
    pixel_m: float,
    carrier_freq_m_inv: Tuple[float, float],
    *,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Tilted plane-wave reference: ``A * exp(2*pi*i*(fx*x + fy*y))``.

    Carrier frequency in cycles / metre — picking 25 000 cycles/m (25
    cycles/mm) puts the +1 order at a sensible location for a
    5-µm-pixel, 128-px frame.
    """
    ny, nx = shape
    y_idx = np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0
    x_idx = np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0
    y_m = y_idx * pixel_m
    x_m = x_idx * pixel_m
    Y, X = np.meshgrid(y_m, x_m, indexing="ij")
    fy, fx = carrier_freq_m_inv
    phase = 2.0 * np.pi * (fy * Y + fx * X)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------

def compose_scene_field(
    spheres: Sequence[SphereSpec],
    config: HologramConfig,
) -> np.ndarray:
    """Compose projected phase screens and propagate them to the camera.

    ``multiplane`` groups objects by axial plane, multiplies their transmission
    functions at that plane, and propagates the running field between planes.
    It remains a scalar thin-screen approximation: refraction and multiple
    scattering inside each sphere are out of scope.
    """
    if config.composition_mode not in {"multiplane", "legacy_sum"}:
        raise ValueError(
            "composition_mode must be 'multiplane' or 'legacy_sum'"
        )

    def transmission(s: SphereSpec) -> np.ndarray:
        return sphere_phase_field(
            config.shape, config.pixel_m, s.radius_m,
            config.wavelength_m,
            center_yx_m=s.center_yx_m,
            n_sphere=s.n_sphere, n_medium=s.n_medium,
            n_slices=s.n_slices,
        )

    if config.composition_mode == "legacy_sum":
        acc = np.zeros(config.shape, dtype=np.complex64)
        for s in spheres:
            acc += propagate_sphere_to_camera(
                transmission(s), z_m=s.z_m,
                pixel_m=config.pixel_m,
                wavelength_m=config.wavelength_m,
                n_medium=s.n_medium,
            )
        return acc

    if not spheres:
        return np.ones(config.shape, dtype=np.complex64)

    planes: dict[float, list[SphereSpec]] = {}
    for sphere in spheres:
        planes.setdefault(float(sphere.z_m), []).append(sphere)

    field = np.ones(config.shape, dtype=np.complex64)
    ordered_z = sorted(planes, reverse=True)
    current_z = ordered_z[0]
    medium_n = float(spheres[0].n_medium)
    for plane_z in ordered_z:
        if current_z > plane_z:
            field = propagate_sphere_to_camera(
                field, z_m=current_z - plane_z,
                pixel_m=config.pixel_m,
                wavelength_m=config.wavelength_m,
                n_medium=medium_n,
            )
        plane_transmission = np.ones(config.shape, dtype=np.complex64)
        for sphere in planes[plane_z]:
            plane_transmission *= transmission(sphere)
        field *= plane_transmission
        current_z = plane_z

    if current_z != 0.0:
        field = propagate_sphere_to_camera(
            field, z_m=current_z,
            pixel_m=config.pixel_m,
            wavelength_m=config.wavelength_m,
            n_medium=medium_n,
        )
    return field


def carrier_offset_px(config: HologramConfig) -> float:
    """Radial FFT-bin distance between DC and the configured carrier."""
    ny, nx = config.shape
    fy, fx = config.carrier_freq_m_inv
    return float(np.hypot(fy * ny * config.pixel_m,
                          fx * nx * config.pixel_m))


def validate_offaxis_sampling(
    config: HologramConfig,
    mask_radius_px: float,
    *,
    guard_px: float = 0.0,
) -> None:
    """Reject carrier/mask geometry that overlaps the DC order."""
    nyquist = 1.0 / (2.0 * config.pixel_m)
    if any(abs(freq) >= nyquist for freq in config.carrier_freq_m_inv):
        raise ValueError("off-axis carrier component reaches Nyquist")
    offset = carrier_offset_px(config)
    if offset <= float(mask_radius_px) + float(guard_px):
        raise ValueError(
            "off-axis +1-order mask overlaps DC: "
            f"carrier offset={offset:.2f}px, radius+guard="
            f"{float(mask_radius_px) + float(guard_px):.2f}px"
        )


def build_hologram(
    spheres: Sequence[SphereSpec],
    config: Optional[HologramConfig] = None,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Full synthesis pipeline: scene → + off-axis reference → |·|² + noise.

    Returns a ``float32`` intensity array ready to feed
    :func:`core.offaxis.extract_complex_field_offaxis`.
    """
    cfg = config or HologramConfig()
    obj = cfg.object_amplitude * compose_scene_field(spheres, cfg)
    ref = reference_plane_wave(
        cfg.shape, cfg.pixel_m, cfg.carrier_freq_m_inv,
        amplitude=cfg.reference_amplitude,
    )
    total = obj + ref
    intensity = (total.real ** 2 + total.imag ** 2).astype(np.float32)

    if cfg.noise_sigma > 0.0:
        rng = rng or np.random.default_rng(0)
        peak = float(intensity.max()) or 1.0
        noise = rng.normal(
            0.0, cfg.noise_sigma * peak, size=intensity.shape,
        ).astype(np.float32)
        intensity = np.clip(intensity + noise, 0.0, None)

    if cfg.bit_depth is not None:
        if int(cfg.bit_depth) < 1:
            raise ValueError("bit_depth must be at least 1")
        levels = float(2 ** int(cfg.bit_depth) - 1)
        lo = float(np.min(intensity))
        span = float(np.max(intensity) - lo)
        if span > 0.0:
            normalized = (intensity - lo) / span
            intensity = (np.rint(normalized * levels) / levels).astype(np.float32)

    return intensity


# ---------------------------------------------------------------------------
# Convenience — single-sphere scenes
# ---------------------------------------------------------------------------

def single_sphere_hologram(
    *,
    radius_m: float,
    z_m: float,
    center_yx_m: Tuple[float, float] = (0.0, 0.0),
    config: Optional[HologramConfig] = None,
    n_sphere: float = 1.40,
    n_medium: float = 1.33,
) -> np.ndarray:
    """Shorthand for the most common stress-test input: one sphere."""
    return build_hologram(
        [SphereSpec(radius_m=radius_m, z_m=z_m,
                    center_yx_m=center_yx_m,
                    n_sphere=n_sphere, n_medium=n_medium)],
        config,
    )


__all__ = [
    "SphereSpec",
    "HologramConfig",
    "sphere_phase_field",
    "propagate_sphere_to_camera",
    "reference_plane_wave",
    "compose_scene_field",
    "carrier_offset_px",
    "validate_offaxis_sampling",
    "build_hologram",
    "single_sphere_hologram",
]
