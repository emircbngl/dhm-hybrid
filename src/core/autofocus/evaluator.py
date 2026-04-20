"""Fast focus-metric evaluator and complex-field downsampling utilities.

`_make_fast_evaluator` builds an `evaluate(z)` closure that reuses the pre-computed
field spectrum and a local `CachedReconstructor`, so each evaluation only needs
one IFFT. `downsample_complex_field` performs bandlimited Fourier-domain cropping
and returns an adjusted `ReconstructionParams` so propagation distances stay correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from ..fft_backend import get_best_fft_backend
from ..reconstruction import (
    CachedReconstructor,
    ReconstructionMethod,
    ReconstructionParams,
)
from .metrics import FocusMetric, _calc_metric


class AutofocusCancelled(Exception):
    """Raised when an autofocus run is cancelled by the user."""
    pass


@dataclass(frozen=True)
class AutoFocusResult:
    best_z_m: float
    scores: Dict[float, float]


def downsample_complex_field(
    field: np.ndarray,
    base_params: ReconstructionParams,
    factor: int = 2,
) -> tuple:
    """
    Downsample a complex field by *factor* for faster autofocus.

    Uses Fourier-domain cropping (correct for bandlimited signals).
    Returns (downsampled_field, adjusted_params) where pixel_size is scaled
    so propagation distances remain correct.
    """
    if factor <= 1:
        return field, base_params

    ny, nx = field.shape
    new_ny, new_nx = ny // factor, nx // factor

    # Fourier-domain crop: FFT → keep central new_ny×new_nx → IFFT
    F = np.fft.fftshift(np.fft.fft2(field))
    cy, cx = ny // 2, nx // 2
    hy, hx = new_ny // 2, new_nx // 2
    F_crop = F[cy - hy: cy - hy + new_ny, cx - hx: cx - hx + new_nx]
    ds_field = np.fft.ifft2(np.fft.ifftshift(F_crop)).astype(np.complex64)

    # Scale pixel size (effective pixel becomes factor× larger)
    ds_params = ReconstructionParams(
        wavelength_m=base_params.wavelength_m,
        pixel_size_m=base_params.pixel_size_m * factor,
        z_m=base_params.z_m,
        n=base_params.n,
    )
    return ds_field, ds_params


def _make_fast_evaluator(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> Callable[[float], float]:
    """
    Build a fast evaluate(z) closure.

    Pre-computes FFT(field) once and creates a local CachedReconstructor
    so each eval only needs: H computation + element-wise multiply + IFFT.
    Avoids per-call overhead of propagate() global cache, id() checks, and .copy().

    If roi_bounds is provided (ROIBounds-like with x0,y0,x1,y1), the metric
    is computed only within that region — much faster and focuses on the object.

    If ref_field is provided, both sample and reference are propagated to each z
    and divided, giving a reference-subtracted complex field for accurate phase
    measurement (all metrics now operate on phase).
    """
    fft = get_best_fft_backend()
    recon = CachedReconstructor(field.shape, method, fft)

    # Pre-compute field spectrum once
    inp = field.astype(np.complex64, copy=False)
    field_spectrum = fft.fft2(inp)
    if field_spectrum.dtype != np.complex64:
        field_spectrum = field_spectrum.astype(np.complex64)

    # Pre-compute reference spectrum if provided (phase metrics always benefit)
    use_ref = ref_field is not None
    ref_spectrum = None
    if use_ref:
        ref_inp = ref_field.astype(np.complex64, copy=False)
        ref_spectrum = fft.fft2(ref_inp)
        if ref_spectrum.dtype != np.complex64:
            ref_spectrum = ref_spectrum.astype(np.complex64)

    # Pre-compute clamped ROI bounds
    ny, nx = field.shape
    if roi_bounds is not None:
        ry0 = max(0, int(roi_bounds.y0))
        rx0 = max(0, int(roi_bounds.x0))
        ry1 = min(ny, int(roi_bounds.y1))
        rx1 = min(nx, int(roi_bounds.x1))
    else:
        ry0, rx0, ry1, rx1 = 0, 0, ny, nx

    def evaluate(z: float) -> float:
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=z, n=base_params.n,
        )
        result = recon.reconstruct_from_spectrum(field_spectrum, params)

        if use_ref and ref_spectrum is not None:
            ref_result = recon.reconstruct_from_spectrum(ref_spectrum, params)
            ref_abs = np.abs(ref_result)
            safe = np.where(ref_abs > 1e-10, ref_result,
                            np.ones_like(ref_result))
            result = result / safe

        roi = result[ry0:ry1, rx0:rx1]
        return _calc_metric(roi, metric)

    return evaluate
