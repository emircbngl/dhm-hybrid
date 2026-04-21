"""Focus metrics and direction helpers — all operate on the phase of the reconstructed field.

`_calc_metric` extracts the wrapped phase (or accepts real phase) and computes one of the
11 phase-based focus metrics. All spatial-difference metrics use wrap-safe phase subtraction.
`_is_minimize` tells the search algorithms whether a metric is optimized by minimization.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict

import numpy as np
import scipy.ndimage as ndimage


class FocusMetric(str, Enum):
    TOTAL_VARIATION = "total_variation"
    GRADIENT = "gradient"
    TENENGRAD = "tenengrad"
    LAPLACIAN_VARIANCE = "laplacian_variance"
    BRENNER = "brenner"
    ENTROPY = "entropy"
    VOLLATH_F4 = "vollath_f4"
    VOLLATH_F5 = "vollath_f5"
    NORMALIZED_VARIANCE = "normalized_variance"
    SPECTRAL_ENERGY = "spectral_energy"
    PHASE_VARIANCE = "phase_variance"


_hf_mask_cache: Dict[tuple, np.ndarray] = {}


def _get_hf_mask(shape: tuple) -> np.ndarray:
    """Return cached high-frequency boolean mask for given image shape."""
    if shape not in _hf_mask_cache:
        ny, nx = shape
        cy, cx = ny // 2, nx // 2
        Y, X = np.ogrid[:ny, :nx]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        r_max = min(cy, cx)
        _hf_mask_cache[shape] = r > (r_max * 0.1)
    return _hf_mask_cache[shape]


def _phase_of(data: np.ndarray) -> np.ndarray:
    """Extract wrapped phase in [-π, π] from complex field, or pass through real phase."""
    if np.iscomplexobj(data):
        return np.angle(data).astype(np.float64)
    return data.astype(np.float64, copy=False)


def has_sufficient_contrast(data: np.ndarray, min_circular_std: float = 0.15) -> bool:
    """True if the phase has enough wrapped-phase structure for ENTROPY to be reliable.

    Uses circular standard deviation (sqrt of circular variance) as the contrast proxy.
    Below 0.15 rad the histogram collapses near a single bin and ENTROPY becomes noise-dominated.
    """
    phase = _phase_of(data)
    mu_x = float(np.mean(np.cos(phase)))
    mu_y = float(np.mean(np.sin(phase)))
    R = np.sqrt(mu_x * mu_x + mu_y * mu_y)
    circ_var = 1.0 - R
    return bool(circ_var >= min_circular_std)


def _wrap_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrap-safe phase difference in [-π, π]. Equivalent to angle(exp(i·(a-b)))."""
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def _calc_metric(data: np.ndarray, metric: FocusMetric) -> float:
    """Compute focus metric on the phase of *data*.

    Accepts either a complex field (phase extracted via np.angle) or a real phase
    array already in radians. All metrics use wrap-safe phase operations so they
    behave correctly across 2π discontinuities.
    """
    phase = _phase_of(data)

    if metric == FocusMetric.PHASE_VARIANCE:
        # Circular variance of the phase — wrap-safe. `np.var(phase)` on
        # wrapped phase explodes near the ±π boundary because values split
        # across the wrap look bimodal. The circular variance
        # `1 − |mean(exp(iφ))|` is well-defined on the torus and rises with
        # in-focus phase structure. Empirically peaks near the lab focus
        # whereas naïve variance peaks at defocus artifacts.
        mu_x = float(np.mean(np.cos(phase)))
        mu_y = float(np.mean(np.sin(phase)))
        result = float(1.0 - np.sqrt(mu_x * mu_x + mu_y * mu_y))
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.TOTAL_VARIATION:
        dy = _wrap_diff(phase[1:, :], phase[:-1, :])
        dx = _wrap_diff(phase[:, 1:], phase[:, :-1])
        result = float(np.sum(np.abs(dy)) + np.sum(np.abs(dx)))
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.GRADIENT:
        dy = _wrap_diff(phase[1:, :], phase[:-1, :])
        dx = _wrap_diff(phase[:, 1:], phase[:, :-1])
        result = float(np.sum(dy * dy) + np.sum(dx * dx))
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.TENENGRAD:
        # Apply Sobel to sin(φ) and cos(φ) separately — wrap-invariant gradient magnitude.
        # |∇φ|² on the unit circle equals |∇sin(φ)|² + |∇cos(φ)|².
        s, c = np.sin(phase), np.cos(phase)
        sx = ndimage.sobel(s, axis=1, mode='reflect')
        sy = ndimage.sobel(s, axis=0, mode='reflect')
        cx = ndimage.sobel(c, axis=1, mode='reflect')
        cy = ndimage.sobel(c, axis=0, mode='reflect')
        result = float(np.sum(sx * sx + sy * sy + cx * cx + cy * cy))
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.LAPLACIAN_VARIANCE:
        # Laplacian on sin/cos — wrap-invariant second-order phase structure.
        s, c = np.sin(phase), np.cos(phase)
        lap_s = ndimage.laplace(s)
        lap_c = ndimage.laplace(c)
        mag = lap_s * lap_s + lap_c * lap_c
        result = float(np.var(mag))
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.BRENNER:
        dy = _wrap_diff(phase[2:, :], phase[:-2, :])
        dx = _wrap_diff(phase[:, 2:], phase[:, :-2])
        result = float(np.sum(dy * dy) + np.sum(dx * dx))
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.ENTROPY:
        hist, _ = np.histogram(phase, bins=256, range=(-np.pi, np.pi), density=False)
        hist = hist[hist > 0].astype(np.float64)
        if hist.size == 0:
            return 0.0
        p = hist / hist.sum()
        result = -float(np.sum(p * np.log(p)))
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.VOLLATH_F4:
        # Circular Vollath F4: cos(Δφ) autocorrelation difference (shift 1 vs 2).
        # At focus, neighbouring phases are coherent (cos Δφ high); shift-2 decays faster.
        n = phase.shape[0] * phase.shape[1]
        g1 = float(np.sum(np.cos(_wrap_diff(phase[:, :-1], phase[:, 1:]))))
        g2 = float(np.sum(np.cos(_wrap_diff(phase[:, :-2], phase[:, 2:]))))
        result = (g1 - g2) / n if n > 0 else 0.0
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.VOLLATH_F5:
        # Circular Vollath F5: cos-autocorrelation minus squared circular mean.
        n = phase.shape[0] * phase.shape[1]
        g1 = float(np.sum(np.cos(_wrap_diff(phase[:, :-1], phase[:, 1:]))))
        mu_x = float(np.mean(np.cos(phase)))
        mu_y = float(np.mean(np.sin(phase)))
        result = (g1 / n - (mu_x * mu_x + mu_y * mu_y)) if n > 0 else 0.0
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.NORMALIZED_VARIANCE:
        # Circular variance: 1 − |mean(exp(iφ))|. At focus, phase gains structure
        # and circular variance rises; fully random phase also has high values,
        # but coherent flat phase (out-of-focus background) pushes it toward 0.
        mu_x = float(np.mean(np.cos(phase)))
        mu_y = float(np.mean(np.sin(phase)))
        R = np.sqrt(mu_x * mu_x + mu_y * mu_y)
        result = float(1.0 - R)
        return result if np.isfinite(result) else 0.0

    if metric == FocusMetric.SPECTRAL_ENERGY:
        # HF energy of exp(iφ) — wrap-safe phase spectrum. Focused images have
        # sharper phase transitions → more high-frequency content.
        F = np.fft.fft2(np.exp(1j * phase))
        F_shift = np.fft.fftshift(F)
        mag = np.abs(F_shift)
        hf_mask = _get_hf_mask(phase.shape)
        result = float(np.sum(mag[hf_mask] ** 2))
        return result if np.isfinite(result) else 0.0

    raise ValueError(f"Unknown metric: {metric}")


def _is_minimize(metric: FocusMetric) -> bool:
    """Returns True if the given metric should be minimized at focus.

    Only ENTROPY minimizes — all other phase metrics MAXIMIZE at focus because
    in-focus reconstructions carry coherent phase structure (sharp cell edges,
    distinct values) that boosts gradient/variance/energy scores. Heavy
    defocus smears the field and drops these quantities.
    """
    return metric == FocusMetric.ENTROPY
