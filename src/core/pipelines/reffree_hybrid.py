"""Reference-free DHM reconstruction pipeline (classical stage).

The in-memory chain that turns a raw off-axis hologram into an unwrapped
phase map WITHOUT a reference hologram at runtime:

    raw → preprocess → Fourier demodulate → autofocus (or fixed z)
        → propagate at z (ASM) → unwrap → polynomial/Zernike background fit
        → ϕ_classical

This is the classical half of Track C; the CNN residual corrector
(``src.recon_dl.inference.ReffreeReconstructor``) consumes ϕ_classical.
The reference-based variant (sample/ref complex division at each trial z)
is also here (``propagate_referenced``), built on ``safe_reference_divide``
from ``core.reconstruction`` (re-exported below) so the benchmark and the
depth/batch paths share one implementation of the per-z reference division.

Optical parameters are carried in :class:`OpticalConfig` — they used to be
module-level globals in ``scripts/run_rapor_data_batch.py`` that both the
benchmark and the shipped DL inference imported, which (a) hardcoded one
microscope and (b) made ``src/`` depend on ``scripts/``. Pass a config
instead; the defaults match the Lindqvist-lab setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.autofocus import FocusMetric
from core.autofocus.search_adaptive import adaptive_gradient_search
from core.background_phase import subtract_background
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.phase_unwrap import unwrap_phase_advanced
from core.reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
    safe_reference_divide,
)


@dataclass(frozen=True)
class OpticalConfig:
    """Optical setup for a reference-free reconstruction.

    Defaults match the lab set on 2026-05-04 (HeNe, 4.4 µm camera pixel at
    50× → 0.088 µm at the sample, aqueous medium).
    """
    wavelength_m: float = 632.8e-9
    effective_pixel_m: float = 0.088e-6   # camera_pixel_um / magnification
    n_medium: float = 1.337
    mask_radius: int = 80
    z_range_mm: float = 20.0              # ± autofocus search half-range
    z_steps: int = 200                    # adaptive_gradient max_evaluations

    @classmethod
    def from_camera(
        cls,
        *,
        wavelength_m: float = 632.8e-9,
        camera_pixel_um: float = 4.4,
        magnification: float = 50.0,
        n_medium: float = 1.337,
        mask_radius: int = 80,
        z_range_mm: float = 20.0,
        z_steps: int = 200,
    ) -> "OpticalConfig":
        """Build from camera pixel + magnification instead of an effective
        pixel size (the form the lab specifies its microscope in)."""
        return cls(
            wavelength_m=wavelength_m,
            effective_pixel_m=(camera_pixel_um / magnification) * 1e-6,
            n_medium=n_medium,
            mask_radius=mask_radius,
            z_range_mm=z_range_mm,
            z_steps=z_steps,
        )


# Lab default — the single canonical instance the CLI scripts share.
LAB_OPTICS = OpticalConfig()


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def preprocess_raw(arr: np.ndarray) -> np.ndarray:
    """v2-style prep: take channel 0, subtract DC, peak-normalise to [-1, 1]."""
    img = arr.astype(np.float32, copy=True)
    if img.ndim == 3:
        img = img[..., 0]
    img = img - float(np.mean(img))
    peak = float(np.max(np.abs(img)))
    if peak > 0:
        img = img / peak
    return img


def demodulate(raw: np.ndarray, cfg: OpticalConfig = LAB_OPTICS,
               *, return_spectrum: bool = False):
    """Pre-propagation Fourier demodulation of a raw hologram.

    Returns the +1-order complex field ``fc``; with ``return_spectrum`` also
    returns the shifted spectrum magnitude (for a diagnostic render).
    """
    pre = preprocess_raw(raw)
    offaxis = OffAxisParams(radius=cfg.mask_radius)
    fc, _center, spectrum_mag, _mask = extract_complex_field_offaxis_debug(
        pre, offaxis,
    )
    if return_spectrum:
        return fc, spectrum_mag
    return fc


# ---------------------------------------------------------------------------
# Propagation + reference division
# ---------------------------------------------------------------------------

def _params_at(z_m: float, cfg: OpticalConfig) -> ReconstructionParams:
    return ReconstructionParams(
        wavelength_m=cfg.wavelength_m, pixel_size_m=cfg.effective_pixel_m,
        z_m=float(z_m), n=cfg.n_medium,
    )


def propagate_field(fc: np.ndarray, z_m: float,
                    cfg: OpticalConfig = LAB_OPTICS) -> np.ndarray:
    """Propagate a demodulated field to plane ``z_m`` with ASM."""
    return propagate(fc, _params_at(z_m, cfg), ReconstructionMethod.ASM,
                     force_python=True)


def propagate_referenced(fc: np.ndarray, ref_fc: np.ndarray, z_m: float,
                         cfg: OpticalConfig = LAB_OPTICS) -> np.ndarray:
    """Propagate sample and reference to ``z_m`` separately, then divide.

    Propagation does not commute with complex division, so the reference
    must be re-propagated at each trial z (B-053 audit, 2026-04-30).
    """
    sample = propagate_field(fc, z_m, cfg)
    ref = propagate_field(ref_fc, z_m, cfg)
    return safe_reference_divide(sample, ref)


# ---------------------------------------------------------------------------
# Autofocus
# ---------------------------------------------------------------------------

def autofocus_z(fc: np.ndarray, cfg: OpticalConfig = LAB_OPTICS,
                ref_fc: Optional[np.ndarray] = None) -> float:
    """Adaptive-gradient autofocus (Laplacian variance, ASM). Returns z in m."""
    base = _params_at(0.0, cfg)
    res = adaptive_gradient_search(
        field=fc, base_params=base, method=ReconstructionMethod.ASM,
        metric=FocusMetric.LAPLACIAN_VARIANCE,
        z_min_m=-cfg.z_range_mm * 1e-3, z_max_m=cfg.z_range_mm * 1e-3,
        max_evaluations=cfg.z_steps,
        ref_field=ref_fc,
    )
    return float(res.best_z_m)


# ---------------------------------------------------------------------------
# Full in-memory reconstruction
# ---------------------------------------------------------------------------

def reconstruct_in_memory(
    holo_fc: np.ndarray,
    ref_fc: Optional[np.ndarray],
    cfg: OpticalConfig = LAB_OPTICS,
    *,
    bg_method: str = "zernike",
    bg_n_terms: int = 15,
    bg_polynomial_order: int = 4,
    fixed_z_m: Optional[float] = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Full unwrapping chain. Returns ``(unwrapped, best_z_m, amplitude)``.

    When ``ref_fc`` is None the reference division is skipped and the
    aberration is removed by a post-unwrap background fit
    (``bg_method`` = "zernike" | "polynomial"). When ``fixed_z_m`` is given
    the autofocus step is skipped (both variants reconstruct at the same z,
    so a comparison measures only the background step, not autofocus drift).
    """
    best_z = (fixed_z_m if fixed_z_m is not None
              else autofocus_z(holo_fc, cfg, ref_fc))
    sample = propagate_field(holo_fc, best_z, cfg)
    if ref_fc is not None:
        ref_at_z = propagate_field(ref_fc, best_z, cfg)
        field = safe_reference_divide(sample, ref_at_z)
    else:
        field = sample

    amp = np.abs(field).astype(np.float32)
    phase = np.angle(field).astype(np.float32)
    unwrapped = unwrap_phase_advanced(
        phase, complex_field=field,
        wavelength_m=cfg.wavelength_m, pixel_size_m=cfg.effective_pixel_m,
    ).astype(np.float32)

    if ref_fc is None:
        unwrapped, _ = subtract_background(
            unwrapped, amplitude=amp,
            method=bg_method,
            n_terms=bg_n_terms,
            polynomial_order=bg_polynomial_order,
        )
        unwrapped = unwrapped.astype(np.float32)

    return unwrapped, best_z, amp


# ---------------------------------------------------------------------------
# Comparison helper (used by the benchmark)
# ---------------------------------------------------------------------------

def piston_align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift ``a`` by a constant so it shares ``b``'s median.

    A global piston is meaningless in QPI; comparing RMSE without aligning
    would inflate the metric by a constant offset.
    """
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return a, 0.0
    delta = float(np.median(b[finite]) - np.median(a[finite]))
    return a + delta, delta
