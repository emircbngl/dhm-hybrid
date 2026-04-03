"""
Display-only contrast enhancement for DHM amplitude images.

Pure post-processing — does NOT affect reconstruction, autofocus, or any
numerical computation. Only changes how the amplitude is displayed.

Methods:
  - percentile_stretch : Clips outliers at p_low/p_high percentiles, then
                         linearly stretches to [0, 1]. Safest default.
  - clahe              : Contrast-Limited Adaptive Histogram Equalization.
                         Enhances local detail (edges, fringes).
  - histogram_eq       : Global histogram equalization. Simple, aggressive.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np


class ContrastMethod(str, Enum):
    NONE = "none"
    PERCENTILE = "percentile_stretch"
    CLAHE = "clahe"
    HISTOGRAM_EQ = "histogram_eq"


def apply_contrast(
    image: np.ndarray,
    method: ContrastMethod,
    *,
    p_low: float = 1.0,
    p_high: float = 99.0,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
) -> np.ndarray:
    """
    Apply display contrast enhancement to a 2-D amplitude image.

    Parameters
    ----------
    image : 2-D float array (amplitude, phase, etc.)
    method : which enhancement to use
    p_low, p_high : percentile bounds for percentile_stretch (default 1–99)
    clahe_clip : clip limit for CLAHE (higher = more contrast)
    clahe_grid : tile grid size for CLAHE (NxN tiles)

    Returns
    -------
    Enhanced image as float32 in [0, 1] range.
    """
    if method == ContrastMethod.NONE or image is None or image.size == 0:
        return _to_float01(image)

    if method == ContrastMethod.PERCENTILE:
        return percentile_stretch(image, p_low, p_high)
    elif method == ContrastMethod.CLAHE:
        return clahe(image, clahe_clip, clahe_grid)
    elif method == ContrastMethod.HISTOGRAM_EQ:
        return histogram_eq(image)
    else:
        return _to_float01(image)


# ─────────────────────────────────────────────────────────────
# Individual methods
# ─────────────────────────────────────────────────────────────

def _to_float01(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] float32."""
    if img is None or img.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    out = img.astype(np.float32)
    mn, mx = out.min(), out.max()
    if mx - mn > 1e-15:
        out = (out - mn) / (mx - mn)
    else:
        out = out * 0.0
    return out


def percentile_stretch(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Clip outlier intensities at the given percentiles and linearly
    stretch the remaining range to [0, 1].

    Default 1–99 % removes bright/dark speckle while preserving structure.
    More aggressive: 2–98 or 5–95.
    """
    out = img.astype(np.float32)
    lo = float(np.percentile(out, p_low))
    hi = float(np.percentile(out, p_high))
    if hi - lo < 1e-15:
        return _to_float01(out)
    out = np.clip(out, lo, hi)
    out = (out - lo) / (hi - lo)
    return out


def clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalization.
    Uses skimage if available, falls back to a pure-numpy approximation.

    clip_limit : limits amplification (2.0 = moderate, 4.0 = strong)
    grid_size  : image divided into NxN tiles for local equalization
    """
    norm = _to_float01(img)

    try:
        from skimage.exposure import equalize_adapthist
        # equalize_adapthist expects [0, 1] input and returns [0, 1]
        out = equalize_adapthist(
            norm,
            kernel_size=max(norm.shape[0] // grid_size, 8),
            clip_limit=clip_limit / 100.0,  # skimage uses 0–1 scale
        )
        return out.astype(np.float32)
    except ImportError:
        pass

    # Fallback: OpenCV
    try:
        import cv2
        u8 = (norm * 255).astype(np.uint8)
        cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        result = cl.apply(u8)
        return result.astype(np.float32) / 255.0
    except ImportError:
        pass

    # Last resort: just do global histogram equalization
    return histogram_eq(img)


def histogram_eq(img: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization (pure numpy, no dependencies).
    Maps intensities so the CDF becomes approximately linear.
    """
    norm = _to_float01(img)
    # Flatten → histogram → CDF → remap
    flat = (norm * 255).astype(np.uint8).ravel()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    cdf = hist.cumsum().astype(np.float64)
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    total = flat.size
    if total - cdf_min > 0:
        lut = ((cdf - cdf_min) / (total - cdf_min) * 255).clip(0, 255).astype(np.uint8)
    else:
        lut = np.arange(256, dtype=np.uint8)

    out = lut[(norm * 255).astype(np.uint8)]
    return out.astype(np.float32) / 255.0
