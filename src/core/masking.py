from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from skimage.feature import peak_local_max


@dataclass(frozen=True)
class OrderMaskResult:
    center_yx: Tuple[int, int]
    mask: np.ndarray
    spectrum_magnitude: np.ndarray


def spectrum_magnitude(field: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(field))
    return np.abs(f)


def detect_plus_one_order(
    field: np.ndarray,
    exclusion_radius: int = 20,
    min_distance: int = 10,
    num_peaks: int = 5,
) -> Tuple[int, int]:
    mag = spectrum_magnitude(field)
    ny, nx = mag.shape
    cy, cx = ny // 2, nx // 2

    y, x = np.ogrid[:ny, :nx]
    r2 = (y - cy) ** 2 + (x - cx) ** 2
    mag = mag.copy()
    mag[r2 <= exclusion_radius**2] = 0.0

    peaks = peak_local_max(mag, min_distance=min_distance, num_peaks=num_peaks)
    if peaks.size == 0:
        raise RuntimeError("No diffraction order peak detected")

    py, px = peaks[0]
    return int(py), int(px)


def circular_mask(shape: Tuple[int, int], center_yx: Tuple[int, int], radius: int,
                  apodization: str = "none", rolloff: float = 0.25) -> np.ndarray:
    """
    Generate a circular mask in 2D with optional soft-edge apodization.

    Parameters
    ----------
    shape : (ny, nx) of the spectrum
    center_yx : center of the +1 order peak
    radius : mask radius in pixels
    apodization : "none" (hard), "hann", "tukey", or "gaussian"
    rolloff : fraction of the radius used for the transition zone (0-1).
              Only used by "tukey". "hann" always uses a full half-cosine rolloff.
              "gaussian" uses rolloff as the sigma fraction relative to radius.

    Returns
    -------
    mask : float32 array in [0, 1]

    Notes
    -----
    Hard masks cause Gibbs ringing in the spatial domain (oscillating artifacts
    in the reconstructed phase). Soft-edge masks suppress this by smoothly
    tapering the spectral energy at the boundary, dramatically improving
    phase image quality in off-axis DHM.
    """
    ny, nx = shape
    cy, cx = center_yx
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.float32)

    if apodization == "none":
        return (r <= radius).astype(np.float32)

    elif apodization == "hann":
        # Hann (raised-cosine): smooth taper from 1 at center to 0 at edge
        # mask(r) = 0.5 * (1 + cos(pi * r / R))  for r <= R, else 0
        mask = np.where(r <= radius,
                        0.5 * (1.0 + np.cos(np.pi * r / radius)),
                        0.0)
        return mask.astype(np.float32)

    elif apodization == "tukey":
        # Tukey (tapered cosine): flat top with cosine rolloff at edges
        # flat region: r <= R * (1 - rolloff)
        # taper region: R * (1 - rolloff) < r <= R
        r_flat = radius * (1.0 - rolloff)
        mask = np.where(
            r <= r_flat, 1.0,
            np.where(
                r <= radius,
                0.5 * (1.0 + np.cos(np.pi * (r - r_flat) / (radius * rolloff + 1e-12))),
                0.0
            )
        )
        return mask.astype(np.float32)

    elif apodization == "gaussian":
        # Gaussian: smooth bell curve, sigma = rolloff * radius
        sigma = max(rolloff * radius, 1.0)
        mask = np.exp(-0.5 * (r / sigma) ** 2)
        # Zero out far tails (beyond 3*radius) to keep it bounded
        mask[r > 3 * radius] = 0.0
        return mask.astype(np.float32)

    else:
        # Fallback: hard mask
        return (r <= radius).astype(np.float32)


def build_plus_one_order_mask(
    field: np.ndarray,
    center_yx: Optional[Tuple[int, int]] = None,
    radius: int = 50,
) -> OrderMaskResult:
    mag = spectrum_magnitude(field)
    if center_yx is None:
        center_yx = detect_plus_one_order(field)
    mask = circular_mask(mag.shape, center_yx=center_yx, radius=radius)
    return OrderMaskResult(center_yx=center_yx, mask=mask, spectrum_magnitude=mag)
