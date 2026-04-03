from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .fft_backend import FFTBackend, get_best_fft_backend
from .masking import detect_plus_one_order, circular_mask


@dataclass(frozen=True)
class OffAxisParams:
    center_yx: Optional[Tuple[int, int]] = None
    radius: int = 60
    dc_exclusion_radius: int = 20
    apodization: str = "tukey"     # "none", "hann", "tukey", "gaussian"
    rolloff: float = 0.25          # fraction of radius for soft transition


def extract_complex_field_offaxis(
    hologram_intensity: Any,
    params: OffAxisParams,
    fft: Optional[FFTBackend] = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    field, center, _, _ = extract_complex_field_offaxis_debug(
        hologram_intensity=hologram_intensity,
        params=params,
        fft=fft,
    )
    return field, center


def extract_complex_field_offaxis_debug(
    hologram_intensity: Any,
    params: OffAxisParams,
    fft: Optional[FFTBackend] = None,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray, np.ndarray]:
    if fft is None:
        fft = get_best_fft_backend()

    holo = np.array(hologram_intensity, copy=False).astype(np.float32)
    if holo.ndim == 3:
        holo = holo[..., 0]

    # Forward FFT and shift
    f = fft.fft2(holo)
    f_shift = np.fft.fftshift(f)
    
    spectrum_mag = np.abs(f_shift)
    
    center = params.center_yx
    if center is None:
        center = detect_plus_one_order(
            holo,
            exclusion_radius=params.dc_exclusion_radius
        )
        
    mask = circular_mask(spectrum_mag.shape, center, params.radius,
                          apodization=params.apodization, rolloff=params.rolloff)
    f_filt = f_shift * mask
    
    # Recenter +1 order to DC
    ny, nx = spectrum_mag.shape
    cy, cx = ny // 2, nx // 2 # following NumPy 0-index standard for DC
    
    sy = cy - center[0]
    sx = cx - center[1]
    
    # circshift using np.roll
    f_recent = np.roll(f_filt, shift=(sy, sx), axis=(0, 1))
    
    # Inverse FFT
    f_unshift = np.fft.ifftshift(f_recent)
    field = fft.ifft2(f_unshift).astype(np.complex64)
    
    return field, center, spectrum_mag, mask
