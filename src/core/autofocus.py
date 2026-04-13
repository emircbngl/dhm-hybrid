from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .reconstruction import ReconstructionMethod, ReconstructionParams, propagate, CachedReconstructor
from .fft_backend import get_best_fft_backend

log = logging.getLogger(__name__)


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
    metric: 'FocusMetric',
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
):
    """
    Build a fast evaluate(z) closure.

    Pre-computes FFT(field) once and creates a local CachedReconstructor
    so each eval only needs: H computation + element-wise multiply + IFFT.
    Avoids per-call overhead of propagate() global cache, id() checks, and .copy().

    If roi_bounds is provided (ROIBounds-like with x0,y0,x1,y1), the metric
    is computed only within that region — much faster and focuses on the object.

    If ref_field is provided and the metric is phase-based, both sample and
    reference are propagated to each z and divided, giving a reference-
    subtracted complex field for accurate phase measurement.
    """
    fft = get_best_fft_backend()
    recon = CachedReconstructor(field.shape, method, fft)

    # Pre-compute field spectrum once
    inp = field.astype(np.complex64, copy=False)
    field_spectrum = fft.fft2(inp)
    if field_spectrum.dtype != np.complex64:
        field_spectrum = field_spectrum.astype(np.complex64)

    # Pre-compute reference spectrum if provided
    use_ref = ref_field is not None and _is_phase_metric(metric)
    ref_spectrum = None
    if use_ref:
        ref_inp = ref_field.astype(np.complex64, copy=False)
        ref_spectrum = fft.fft2(ref_inp)
        if ref_spectrum.dtype != np.complex64:
            ref_spectrum = ref_spectrum.astype(np.complex64)

    phase_metric = _is_phase_metric(metric)

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
        if phase_metric:
            return _calc_metric(roi, metric)
        else:
            return _calc_metric(np.abs(roi), metric)

    return evaluate


class AutofocusCancelled(Exception):
    """Raised when an autofocus run is cancelled by the user."""
    pass


# Cache for high-frequency mask used by Spectral Energy metric
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
    # Phase-based metrics — require complex field (not just amplitude)
    PHASE_VARIANCE = "phase_variance"
    AMPLITUDE_FLATNESS = "amplitude_flatness"


def _is_phase_metric(metric: FocusMetric) -> bool:
    """True if the metric requires the complex field (not just amplitude)."""
    return metric == FocusMetric.PHASE_VARIANCE

@dataclass(frozen=True)
class AutoFocusResult:
    best_z_m: float
    scores: Dict[float, float]

def _calc_metric(data: np.ndarray, metric: FocusMetric) -> float:
    """Compute focus metric.  *data* is amplitude (real) for classic metrics,
    or complex field for phase-based metrics."""
    # Phase-based metrics — operate on the complex field directly
    if metric == FocusMetric.PHASE_VARIANCE:
        phase = np.angle(data).astype(np.float64)
        return float(np.var(phase))

    if metric == FocusMetric.AMPLITUDE_FLATNESS:
        amp = np.abs(data).astype(np.float64)
        return float(np.var(amp))

    # Classic amplitude-based metrics — ensure real array
    amp = np.abs(data).astype(np.float64) if np.iscomplexobj(data) else data

    if metric == FocusMetric.TOTAL_VARIATION:
        dy = amp[1:, :] - amp[:-1, :]
        dx = amp[:, 1:] - amp[:, :-1]
        return float(np.sum(np.abs(dy)) + np.sum(np.abs(dx)))
        
    elif metric == FocusMetric.GRADIENT:
        dy = np.gradient(amp, axis=0)
        dx = np.gradient(amp, axis=1)
        return float(np.sum(dy**2 + dx**2))
        
    elif metric == FocusMetric.TENENGRAD:
        sx = ndimage.sobel(amp, axis=1, mode='reflect')
        sy = ndimage.sobel(amp, axis=0, mode='reflect')
        return float(np.sum(sx**2 + sy**2))
        
    elif metric == FocusMetric.LAPLACIAN_VARIANCE:
        lap = ndimage.laplace(amp)
        return float(np.var(lap))
        
    elif metric == FocusMetric.BRENNER:
        dy = (amp[2:, :] - amp[:-2, :])**2
        dx = (amp[:, 2:] - amp[:, :-2])**2
        return float(np.sum(dy) + np.sum(dx))
        
    elif metric == FocusMetric.ENTROPY:
        hist, _ = np.histogram(amp, bins=256, density=False)
        hist = hist[hist > 0].astype(np.float64)
        p = hist / hist.sum()
        return -float(np.sum(p * np.log(p)))

    elif metric == FocusMetric.VOLLATH_F4:
        # Vollath's F4: autocorrelation difference (shift 1 vs shift 2)
        # Good at detecting defocus in both bright-field and phase images
        n = amp.shape[0] * amp.shape[1]
        g1 = float(np.sum(amp[:, :-1] * amp[:, 1:]))
        g2 = float(np.sum(amp[:, :-2] * amp[:, 2:]))
        return (g1 - g2) / n

    elif metric == FocusMetric.VOLLATH_F5:
        # Vollath's F5: autocorrelation minus mean²
        # More robust than F4 for noisy images
        n = amp.shape[0] * amp.shape[1]
        g1 = float(np.sum(amp[:, :-1] * amp[:, 1:]))
        mu = float(np.mean(amp))
        return g1 / n - mu * mu

    elif metric == FocusMetric.NORMALIZED_VARIANCE:
        # Normalized variance: var / mean — intensity-independent
        mu = float(np.mean(amp))
        if mu < 1e-15:
            return 0.0
        return float(np.var(amp)) / mu

    elif metric == FocusMetric.SPECTRAL_ENERGY:
        # High-frequency spectral energy from FFT magnitude
        # Focused images have more high-frequency content
        F = np.fft.fft2(amp)
        F_shift = np.fft.fftshift(F)
        mag = np.abs(F_shift)
        hf_mask = _get_hf_mask(amp.shape)
        return float(np.sum(mag[hf_mask] ** 2))

    raise ValueError(f"Unknown metric: {metric}")


def _is_minimize(metric: FocusMetric) -> bool:
    """Returns True if the given metric should be minimized (e.g. entropy)."""
    return metric in (FocusMetric.ENTROPY, FocusMetric.AMPLITUDE_FLATNESS)

def autofocus_zscan(
    field: np.ndarray,
    base_params: ReconstructionParams,
    z_values_m: List[float],
    method: ReconstructionMethod,
    metric: FocusMetric,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> AutoFocusResult:
    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    scores = {}
    best_z = z_values_m[0]
    minimize = _is_minimize(metric)
    best_score = float('inf') if minimize else -float('inf')
    total = len(z_values_m)
    
    for i, z in enumerate(z_values_m):
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        if on_progress:
            on_progress(i, total)
        score = _eval(z)
        scores[z] = score
        
        if minimize:
            if score < best_score:
                best_score = score
                best_z = z
        else:
            if score > best_score:
                best_score = score
                best_z = z
                
    return AutoFocusResult(best_z_m=best_z, scores=scores)

@dataclass(frozen=True)
class GoldenSearchResult:
    best_z_m: float
    best_score: float
    iterations: int
    evaluations: int

def golden_section_search(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    z_min_m: float,
    z_max_m: float,
    tolerance_m: float = 1e-6,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    eval_offset: int = 0,
    est_total: int = 0,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> GoldenSearchResult:
    """O(log N) search for the best focus plane using Golden Section."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    
    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    state = {"evaluations": 0}
    
    def evaluate(z: float) -> float:
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        state["evaluations"] += 1
        if on_progress and est_total > 0:
            on_progress(eval_offset + state["evaluations"], est_total)
        return _eval(z)
        
    maximize = not _is_minimize(metric)
    
    a = float(z_min_m)
    b = float(z_max_m)
    
    x1 = float(a + resphi * (b - a))
    x2 = float(b - resphi * (b - a))
    
    f1 = evaluate(x1)
    f2 = evaluate(x2)
    
    iterations = 0
    while abs(b - a) > tolerance_m:
        iterations += 1
        
        pick_left = (f1 > f2) if maximize else (f1 < f2)
        
        if pick_left:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = evaluate(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = evaluate(x2)
            
    best_z = float((a + b) / 2.0)
    best_score = float(evaluate(best_z))
    
    return GoldenSearchResult(best_z_m=best_z, best_score=best_score, iterations=iterations, evaluations=state["evaluations"])

def coarse_to_fine_search(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    z_min_m: float,
    z_max_m: float,
    coarse_steps: int = 21,
    fine_tolerance_m: float = 1e-6,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> GoldenSearchResult:
    """Hybrid approach: N-step sweep followed by Golden Section Search."""
    import math
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    margin_est = (z_max_m - z_min_m) / max(1, coarse_steps - 1) * 1.5 * 2
    golden_iters = max(1, int(math.log(margin_est / max(fine_tolerance_m, 1e-15)) / math.log(phi))) + 3
    est_total = coarse_steps + golden_iters

    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    state = {"evaluations": 0}
    
    def evaluate(z: float) -> float:
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        state["evaluations"] += 1
        if on_progress:
            on_progress(state["evaluations"], est_total)
        return _eval(z)
        
    maximize = not _is_minimize(metric)

    # Phase 1: Coarse Scan
    step = (z_max_m - z_min_m) / max(1, coarse_steps - 1)
    
    best_z_coarse = z_min_m
    best_f_coarse = evaluate(z_min_m)
    
    for i in range(1, coarse_steps):
        z = z_min_m + i * step
        f = evaluate(z)
        better = (f > best_f_coarse) if maximize else (f < best_f_coarse)
        if better:
            best_f_coarse = f
            best_z_coarse = z
            
    # Phase 2: Golden Section in Neighborhood
    margin = step * 1.5
    fine_min = max(z_min_m, best_z_coarse - margin)
    fine_max = min(z_max_m, best_z_coarse + margin)
    
    res = golden_section_search(
        field=field,
        base_params=base_params,
        method=method,
        metric=metric,
        z_min_m=fine_min,
        z_max_m=fine_max,
        tolerance_m=fine_tolerance_m,
        cancel_check=cancel_check,
        on_progress=on_progress,
        eval_offset=state["evaluations"],
        est_total=est_total,
        ref_field=ref_field,
    )
    
    return GoldenSearchResult(
        best_z_m=res.best_z_m,
        best_score=res.best_score,
        iterations=res.iterations + coarse_steps,
        evaluations=res.evaluations + state["evaluations"]
    )

# ─────────────────────────────────────────────────────────────
# Robust Coarse-to-Fine Search (smoothing-based, noise-resistant)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RobustSearchResult:
    best_z_m: float
    best_score: float
    total_evaluations: int
    coarse_best_z_m: float
    fine_range: Tuple[float, float]
    coarse_z: Optional[np.ndarray] = None
    coarse_scores: Optional[np.ndarray] = None


def robust_coarse_to_fine_search(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    z_min_m: float,
    z_max_m: float,
    n_coarse: int = 40,
    refine_factor: int = 8,
    smooth_sigma: float = 1.5,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> RobustSearchResult:
    """
    Robust Multi-Scale Autofocus — works even with noisy / multimodal metrics.

    Strategy:
      1. Coarse scan over the full range (n_coarse steps)
      2. Gaussian-smooth the metric curve to suppress noise
      3. Find the global peak on the smoothed curve
      4. Fine scan around the peak neighbourhood (refine_factor × denser)
      5. Smooth + peak again for sub-step precision
    """
    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    evaluations = 0
    minimize = _is_minimize(metric)

    n_fine = max(refine_factor * 4, 20)
    est_total = n_coarse + n_fine

    def evaluate(z: float) -> float:
        nonlocal evaluations
        evaluations += 1
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        if on_progress:
            on_progress(evaluations, est_total)
        return _eval(z)

    # ── Phase 1: Coarse ──
    z_coarse = np.linspace(z_min_m, z_max_m, n_coarse)
    f_coarse = np.array([evaluate(z) for z in z_coarse])

    work = -f_coarse if minimize else f_coarse.copy()
    work_smooth = gaussian_filter1d(work, sigma=smooth_sigma)
    peak_idx = int(np.argmax(work_smooth))
    coarse_best_z = float(z_coarse[peak_idx])

    # ── Phase 2: Fine ──
    coarse_step = (z_max_m - z_min_m) / max(1, n_coarse - 1)
    margin = coarse_step * 2.0
    fine_min = max(z_min_m, coarse_best_z - margin)
    fine_max = min(z_max_m, coarse_best_z + margin)
    z_fine = np.linspace(fine_min, fine_max, n_fine)
    f_fine = np.array([evaluate(z) for z in z_fine])

    work_fine = -f_fine if minimize else f_fine.copy()
    work_fine_smooth = gaussian_filter1d(work_fine, sigma=smooth_sigma)
    fine_peak_idx = int(np.argmax(work_fine_smooth))
    best_z = float(z_fine[fine_peak_idx])
    best_score = float(f_fine[fine_peak_idx])

    return RobustSearchResult(
        best_z_m=best_z,
        best_score=best_score,
        total_evaluations=evaluations,
        coarse_best_z_m=coarse_best_z,
        fine_range=(fine_min, fine_max),
        coarse_z=z_coarse,
        coarse_scores=f_coarse,
    )


# ─────────────────────────────────────────────────────────────
# Adaptive Step Autofocus Algorithms
# ─────────────────────────────────────────────────────────────

@dataclass
class AdaptiveStepResult:
    best_z_m: float
    best_score: float
    evaluations: int
    z_history: np.ndarray
    score_history: np.ndarray
    step_history: np.ndarray
    method_name: str


def adaptive_gradient_search(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    z_min_m: float,
    z_max_m: float,
    step_init: Optional[float] = None,
    step_min: Optional[float] = None,
    step_max: Optional[float] = None,
    grow_factor: float = 1.5,
    shrink_factor: float = 0.4,
    max_evaluations: int = 100,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> AdaptiveStepResult:
    """
    Gradient-Based Adaptive Step autofocus.
    Phase 1 (forward scan, ~40 % budget): Walks z_min→z_max with adaptive steps.
      - |derivative| small → flat region → enlarge step (speed up)
      - |derivative| large → near peak → shrink step (refine)
      - sign change → passed peak → shrink aggressively
    Phase 2 (refinement, remaining budget): Iteratively narrows a window
      around the best z found so far, achieving sub-step resolution.
    """
    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    full_range = z_max_m - z_min_m
    if step_init is None:
        # Gradient algo needs large initial steps to traverse range quickly;
        # adaptive logic shrinks near peaks automatically.
        step_init = full_range / 10.0
    if step_min is None:
        step_min = full_range / max(max_evaluations * 2, 100)
    if step_max is None:
        step_max = step_init * 5.0  # allow big jumps in flat regions

    maximize = not _is_minimize(metric)
    evaluations = 0

    def evaluate(z: float) -> float:
        nonlocal evaluations
        evaluations += 1
        if on_progress:
            on_progress(evaluations, max_evaluations)
        return _eval(z)

    # ── Phase 1: Forward adaptive scan (≤50 % of budget) ──
    forward_budget = min(max(int(max_evaluations * 0.5), 8), max_evaluations)

    z = z_min_m
    step = step_init
    f_prev = evaluate(z)

    best_z, best_f = z, f_prev
    z_hist, f_hist, s_hist = [z], [f_prev], [step]
    prev_deriv = 0.0

    while z < z_max_m and evaluations < forward_budget:
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        z_next = min(z + step, z_max_m)
        f_curr = evaluate(z_next)

        dz = z_next - z
        deriv = (f_curr - f_prev) / dz if dz > 1e-15 else 0.0

        better = (f_curr > best_f) if maximize else (f_curr < best_f)
        if better:
            best_f = f_curr
            best_z = z_next

        z_hist.append(z_next)
        f_hist.append(f_curr)
        s_hist.append(step)

        abs_d = abs(deriv)
        # Detect sign change → passed peak (direction-aware)
        if maximize:
            sign_change = prev_deriv > 0 and deriv < 0
        else:
            sign_change = prev_deriv < 0 and deriv > 0
        if sign_change:
            step = max(step * shrink_factor ** 2, step_min)
        elif abs_d > abs(prev_deriv) * 1.5 and abs_d > 1e-8:
            step = max(step * shrink_factor, step_min)
        elif abs_d < abs(prev_deriv) * 0.5 or abs_d < 1e-8:
            step = min(step * grow_factor, step_max)

        prev_deriv = deriv
        f_prev = f_curr
        z = z_next

    # ── Phase 2: Iterative refinement around best_z ──
    search_half = step_init
    while evaluations < max_evaluations:
        remaining = max_evaluations - evaluations
        if remaining < 3:
            break
        n_pts = min(remaining, max(5, remaining // 2))
        lo = max(z_min_m, best_z - search_half)
        hi = min(z_max_m, best_z + search_half)
        if hi - lo < 1e-15:
            break
        z_fine = np.linspace(lo, hi, n_pts)
        for zp in z_fine:
            if cancel_check and cancel_check():
                raise AutofocusCancelled()
            f = evaluate(float(zp))
            b = (f > best_f) if maximize else (f < best_f)
            if b:
                best_f = f
                best_z = float(zp)
            z_hist.append(float(zp))
            f_hist.append(f)
            s_hist.append((hi - lo) / max(n_pts - 1, 1))
        fine_step = (hi - lo) / max(n_pts - 1, 1)
        search_half = fine_step * 1.5

    return AdaptiveStepResult(
        best_z_m=best_z, best_score=best_f, evaluations=evaluations,
        z_history=np.array(z_hist), score_history=np.array(f_hist),
        step_history=np.array(s_hist), method_name="Gradient-Based Adaptive",
    )


def adaptive_ratio_search(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    z_min_m: float,
    z_max_m: float,
    step_init: Optional[float] = None,
    step_min: Optional[float] = None,
    step_max: Optional[float] = None,
    stable_threshold: float = 0.02,
    grow_factor: float = 1.5,
    shrink_factor: float = 0.4,
    max_evaluations: int = 100,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> AdaptiveStepResult:
    """
    Ratio-Based Adaptive Step autofocus.
    Phase 1 (forward scan, ~40 % budget): Monitors f_new / f_old ratio.
      - ratio ≈ 1 → stable region → enlarge step
      - ratio far from 1 → rapid change → shrink step
      - ratio < 1 after peak → passed peak → shrink aggressively
    Phase 2 (refinement, remaining budget): Iteratively narrows a window
      around the best z found so far.
    """
    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    full_range = z_max_m - z_min_m
    if step_init is None:
        # Ratio algo: moderate step — not as aggressive as Gradient,
        # but still needs to cover the full range in forward scan.
        step_init = full_range / 15.0
    if step_min is None:
        step_min = full_range / max(max_evaluations * 2, 100)
    if step_max is None:
        step_max = step_init * 4.0

    maximize = not _is_minimize(metric)
    evaluations = 0

    def evaluate(z: float) -> float:
        nonlocal evaluations
        evaluations += 1
        if on_progress:
            on_progress(evaluations, max_evaluations)
        return _eval(z)

    # ── Phase 1: Forward scan (≤50 % of budget) ──
    forward_budget = min(max(int(max_evaluations * 0.5), 8), max_evaluations)

    z = z_min_m
    step = step_init
    f_prev = evaluate(z)

    best_z, best_f = z, f_prev
    z_hist, f_hist, s_hist = [z], [f_prev], [step]

    while z < z_max_m and evaluations < forward_budget:
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        z_next = min(z + step, z_max_m)
        f_curr = evaluate(z_next)

        better = (f_curr > best_f) if maximize else (f_curr < best_f)
        if better:
            best_f = f_curr
            best_z = z_next

        z_hist.append(z_next)
        f_hist.append(f_curr)
        s_hist.append(step)

        ratio = f_curr / f_prev if abs(f_prev) > 1e-15 else 1.0
        change = abs(ratio - 1.0)

        if change < stable_threshold:
            step = min(step * grow_factor, step_max)
        elif change > stable_threshold * 5:
            step = max(step * shrink_factor, step_min)

        is_declining = (ratio < 1.0) if maximize else (ratio > 1.0)
        was_best = (f_prev == best_f)
        if is_declining and was_best:
            step = max(step * shrink_factor ** 2, step_min)

        f_prev = f_curr
        z = z_next

    # ── Phase 2: Iterative refinement around best_z ──
    search_half = step_init
    while evaluations < max_evaluations:
        remaining = max_evaluations - evaluations
        if remaining < 3:
            break
        n_pts = min(remaining, max(5, remaining // 2))
        lo = max(z_min_m, best_z - search_half)
        hi = min(z_max_m, best_z + search_half)
        if hi - lo < 1e-15:
            break
        z_fine = np.linspace(lo, hi, n_pts)
        for zp in z_fine:
            if cancel_check and cancel_check():
                raise AutofocusCancelled()
            f = evaluate(float(zp))
            b = (f > best_f) if maximize else (f < best_f)
            if b:
                best_f = f
                best_z = float(zp)
            z_hist.append(float(zp))
            f_hist.append(f)
            s_hist.append((hi - lo) / max(n_pts - 1, 1))
        fine_step = (hi - lo) / max(n_pts - 1, 1)
        search_half = fine_step * 1.5

    return AdaptiveStepResult(
        best_z_m=best_z, best_score=best_f, evaluations=evaluations,
        z_history=np.array(z_hist), score_history=np.array(f_hist),
        step_history=np.array(s_hist), method_name="Ratio-Based Adaptive",
    )


def adaptive_bracketing_search(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    z_min_m: float,
    z_max_m: float,
    step_init: Optional[float] = None,
    n_refine_levels: int = 2,
    refine_divisions: int = 5,
    smooth_sigma: float = 1.0,
    max_evaluations: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> AdaptiveStepResult:
    """
    Bracketing + Refinement autofocus with noise-robust peak detection.

    Phase 1: Coarse sweep across full range, then use Gaussian-smoothed
             metric curve to locate the global peak and bracket around it.
             This avoids false brackets from noisy single-sample dips.
    Phase 2: Iteratively subdivide the bracket for increasing precision.
             Each level uses *both* smoothed peak (for bracket center) and
             raw best value (for the final reported z), so the output is
             always an actually-evaluated z — smoothing only guides search.
             Runs at least *n_refine_levels*; auto-extends if *max_evaluations*
             budget allows more levels.
    """
    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    full_range = z_max_m - z_min_m
    if step_init is None:
        # Allocate ~50% of budget to coarse sweep for reliable peak detection
        eff_budget = max_evaluations if max_evaluations else (n_refine_levels * (refine_divisions + 1) + 30)
        n_coarse_target = max(25, int(eff_budget * 0.5))
        step_init = full_range / n_coarse_target

    import math
    n_coarse_est = max(1, math.ceil(full_range / step_init)) + 1
    if max_evaluations is not None:
        est_total = max_evaluations
    else:
        est_total = n_coarse_est + n_refine_levels * (refine_divisions + 1)

    maximize = not _is_minimize(metric)
    evaluations = 0
    z_hist: List[float] = []
    f_hist: List[float] = []
    s_hist: List[float] = []

    def evaluate(z: float, step_val: float) -> float:
        nonlocal evaluations
        evaluations += 1
        if on_progress:
            on_progress(evaluations, est_total)
        f = _eval(z)
        z_hist.append(z)
        f_hist.append(f)
        s_hist.append(step_val)
        return f

    # ── Phase 1: Coarse sweep (collect ALL points, then smooth) ──
    coarse_zs: List[float] = []
    coarse_fs: List[float] = []

    z = z_min_m
    while z <= z_max_m:
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        f = evaluate(z, step_init)
        coarse_zs.append(z)
        coarse_fs.append(f)
        z = min(z + step_init, z_max_m + 1e-15)
        if z > z_max_m and coarse_zs[-1] < z_max_m:
            f = evaluate(z_max_m, step_init)
            coarse_zs.append(z_max_m)
            coarse_fs.append(f)
            break

    coarse_z_arr = np.array(coarse_zs)
    coarse_f_arr = np.array(coarse_fs)

    # Smooth to suppress noise — find robust peak location
    work = -coarse_f_arr if not maximize else coarse_f_arr.copy()
    effective_sigma = max(smooth_sigma, 1.0) if len(work) >= 5 else 0.5
    work_smooth = gaussian_filter1d(work, sigma=effective_sigma)
    smooth_peak_idx = int(np.argmax(work_smooth))

    # Raw best (actually evaluated, not smoothed)
    raw_peak_idx = int(np.argmax(work))
    best_z = float(coarse_z_arr[raw_peak_idx])
    best_f = float(coarse_f_arr[raw_peak_idx])

    # Bracket: centered on smoothed peak, width = 2 × coarse step on each side
    smooth_peak_z = float(coarse_z_arr[smooth_peak_idx])
    bracket_half = step_init * 2.0
    bracket_left = max(z_min_m, smooth_peak_z - bracket_half)
    bracket_right = min(z_max_m, smooth_peak_z + bracket_half)

    # If raw peak is far from smoothed peak, widen bracket to cover both
    if abs(best_z - smooth_peak_z) > bracket_half:
        bracket_left = max(z_min_m, min(best_z, smooth_peak_z) - step_init)
        bracket_right = min(z_max_m, max(best_z, smooth_peak_z) + step_init)

    # ── Phase 2: Iterative refinement ──
    level = 0
    while True:
        if level >= n_refine_levels:
            if max_evaluations is None or evaluations + refine_divisions + 1 > max_evaluations:
                break

        step = (bracket_right - bracket_left) / refine_divisions
        if step < 1e-15:
            break
        z_points = np.linspace(bracket_left, bracket_right, refine_divisions + 1)

        level_zs: List[float] = []
        level_fs: List[float] = []
        for zp in z_points:
            if cancel_check and cancel_check():
                raise AutofocusCancelled()
            f = evaluate(float(zp), step)
            level_zs.append(float(zp))
            level_fs.append(f)

        level_f_arr = np.array(level_fs)
        level_z_arr = np.array(level_zs)

        # Use smoothing on refinement levels only if enough points
        if len(level_fs) >= 5 and smooth_sigma > 0:
            level_work = -level_f_arr if not maximize else level_f_arr.copy()
            level_smooth = gaussian_filter1d(level_work, sigma=min(smooth_sigma, len(level_fs) / 3))
            smooth_idx = int(np.argmax(level_smooth))
        else:
            smooth_idx = None

        # Raw best in this level
        raw_work = -level_f_arr if not maximize else level_f_arr.copy()
        raw_idx = int(np.argmax(raw_work))
        level_best_z = float(level_z_arr[raw_idx])
        level_best_f = float(level_f_arr[raw_idx])

        b2 = (level_best_f > best_f) if maximize else (level_best_f < best_f)
        if b2:
            best_f = level_best_f
            best_z = level_best_z

        # Re-center bracket: prefer smoothed center if available
        center = level_best_z
        if smooth_idx is not None:
            smooth_center = float(level_z_arr[smooth_idx])
            center = (level_best_z + smooth_center) / 2.0

        margin = step * 1.5
        bracket_left = max(z_min_m, center - margin)
        bracket_right = min(z_max_m, center + margin)

        # Ensure bracket covers both raw and smoothed peaks
        bracket_left = min(bracket_left, max(z_min_m, level_best_z - margin * 0.5))
        bracket_right = max(bracket_right, min(z_max_m, level_best_z + margin * 0.5))

        level += 1

    return AdaptiveStepResult(
        best_z_m=best_z, best_score=best_f, evaluations=evaluations,
        z_history=np.array(z_hist), score_history=np.array(f_hist),
        step_history=np.array(s_hist), method_name="Bracketing + Refinement",
    )


# ─────────────────────────────────────────────────────────────
# Auto-Select Best Metric
# ─────────────────────────────────────────────────────────────

def auto_select_metric(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    z_min_m: float,
    z_max_m: float,
    n_steps: int = 30,
    smooth_sigma: float = 2.0,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> FocusMetric:
    """
    Scans all metrics over a coarse z-range and picks the most reliable one.
    Reliability = high prominence of global peak with fewest secondary peaks.
    """
    fft = get_best_fft_backend()
    recon = CachedReconstructor(field.shape, method, fft)
    inp = field.astype(np.complex64, copy=False)
    field_spectrum = fft.fft2(inp)
    if field_spectrum.dtype != np.complex64:
        field_spectrum = field_spectrum.astype(np.complex64)

    z_values = np.linspace(z_min_m, z_max_m, n_steps)

    # Pre-compute all reconstructions once (store complex for phase metrics)
    complex_fields: List[np.ndarray] = []
    for z in z_values:
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=z,
            n=base_params.n,
        )
        result = recon.reconstruct_from_spectrum(field_spectrum, params)
        complex_fields.append(result)

    # Exclude entropy — unreliable for holographic samples (false peaks)
    candidates = [fm for fm in FocusMetric if fm != FocusMetric.ENTROPY]

    scores: Dict[FocusMetric, float] = {}
    for fm in candidates:
        if _is_phase_metric(fm):
            values = np.array([_calc_metric(c, fm) for c in complex_fields])
        else:
            values = np.array([_calc_metric(np.abs(c), fm) for c in complex_fields])
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-15:
            scores[fm] = -999.0
            continue
        values_norm = (values - vmin) / (vmax - vmin)
        if _is_minimize(fm):
            values_norm = 1.0 - values_norm
        values_smooth = gaussian_filter1d(values_norm, sigma=smooth_sigma)

        peaks_idx, props = find_peaks(values_smooth, prominence=0.05)
        n_peaks = len(peaks_idx)
        max_prominence = float(np.max(props["prominences"])) if n_peaks > 0 else float(np.max(values_smooth))

        # Score: high prominence, few secondary peaks
        scores[fm] = max_prominence - 0.2 * max(0, n_peaks - 1)

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    log.info("auto_select_metric scores: %s  →  %s", scores, best)
    return best


# ─────────────────────────────────────────────────────────────
# Metric Landscape Diagnostic
# ─────────────────────────────────────────────────────────────

@dataclass
class MetricLandscapeResult:
    z_values: np.ndarray
    raw_scores: Dict[FocusMetric, np.ndarray]
    smoothed_scores: Dict[FocusMetric, np.ndarray]
    peak_z: Dict[FocusMetric, float]
    n_peaks: Dict[FocusMetric, int]


def scan_metric_landscape(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    z_min_m: float,
    z_max_m: float,
    n_steps: int = 80,
    smooth_sigma: float = 2.0,
    metrics: Optional[List[FocusMetric]] = None,
) -> MetricLandscapeResult:
    """
    Scans all (or selected) metrics across z and returns raw + smoothed curves.
    Useful for diagnostics: which metric is unimodal, which is noisy?
    """
    if metrics is None:
        metrics = list(FocusMetric)

    fft = get_best_fft_backend()
    z_values = np.linspace(z_min_m, z_max_m, n_steps)

    complex_fields: List[np.ndarray] = []
    for z in z_values:
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=z,
            n=base_params.n,
        )
        recon = propagate(field, params, method, fft=fft, force_python=True)
        complex_fields.append(recon)

    raw_scores: Dict[FocusMetric, np.ndarray] = {}
    smoothed_scores: Dict[FocusMetric, np.ndarray] = {}
    peak_z: Dict[FocusMetric, float] = {}
    n_peaks_map: Dict[FocusMetric, int] = {}

    for fm in metrics:
        if _is_phase_metric(fm):
            vals = np.array([_calc_metric(c, fm) for c in complex_fields])
        else:
            vals = np.array([_calc_metric(np.abs(c), fm) for c in complex_fields])
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-15:
            vals_norm = np.zeros_like(vals)
        else:
            vals_norm = (vals - vmin) / (vmax - vmin)
        if _is_minimize(fm):
            vals_norm = 1.0 - vals_norm

        vals_smooth = gaussian_filter1d(vals_norm, sigma=smooth_sigma)
        peaks_idx, _ = find_peaks(vals_smooth, prominence=0.05)

        raw_scores[fm] = vals
        smoothed_scores[fm] = vals_smooth
        peak_z[fm] = float(z_values[int(np.argmax(vals_smooth))])
        n_peaks_map[fm] = len(peaks_idx)

    return MetricLandscapeResult(
        z_values=z_values,
        raw_scores=raw_scores,
        smoothed_scores=smoothed_scores,
        peak_z=peak_z,
        n_peaks=n_peaks_map,
    )


# ─────────────────────────────────────────────────────────────
# Autofocus Benchmark
# ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkEntry:
    algorithm: str
    metric: FocusMetric
    best_z_m: float
    best_score: float
    elapsed_s: float
    evaluations: int
    error_um: float = 0.0        # error vs reference in µm
    per_eval_ms: float = 0.0     # milliseconds per evaluation
    downsampled: int = 1         # DS factor (1 = full resolution)


@dataclass
class AutofocusBenchmarkResult:
    entries: List[BenchmarkEntry]
    field_shape: Tuple[int, int]
    z_range: Tuple[float, float]
    reference_z_m: Optional[float] = None
    reference_metric: Optional[str] = None
    reference_elapsed_s: float = 0.0


def autofocus_benchmark(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    z_min_m: float,
    z_max_m: float,
    steps: int = 51,
    metrics: Optional[List[FocusMetric]] = None,
    include_downsampled: bool = True,
) -> AutofocusBenchmarkResult:
    """
    Comprehensive benchmark: every algorithm × metric combination.

    1. Finds a reference z via high-resolution linear sweep (Laplacian Var.)
    2. Benchmarks all algorithm × metric combos at full resolution
    3. Optionally benchmarks DS×2 variants for speed comparison
    4. Computes error vs reference, per-eval timing

    Algorithms tested: Linear Sweep, Coarse-to-Fine (Golden), Robust C2F,
                       Adaptive Gradient, Adaptive Ratio, Adaptive Bracketing.
    """
    if metrics is None:
        metrics = list(FocusMetric)

    # ── Step 0: Find reference z (high-res linear, Laplacian Variance) ──
    ref_steps = max(steps * 4, 200)
    ref_metric = FocusMetric.LAPLACIAN_VARIANCE
    z_ref_arr = list(np.linspace(z_min_m, z_max_m, ref_steps))
    t0 = time.perf_counter()
    ref_res = autofocus_zscan(field, base_params, z_ref_arr, method, ref_metric)
    ref_elapsed = time.perf_counter() - t0
    ref_z = ref_res.best_z_m

    entries: List[BenchmarkEntry] = []

    def _run_and_record(algo_name, func, fm, ds_factor=1):
        t0 = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - t0
        best_z = result.best_z_m
        best_score = getattr(result, 'best_score', 0.0)
        if best_score == 0.0 and hasattr(result, 'scores'):
            best_score = result.scores.get(best_z, 0.0)
        evals = getattr(result, 'evaluations',
                        getattr(result, 'total_evaluations',
                               len(getattr(result, 'scores', {}))))
        error_um = abs(best_z - ref_z) * 1e6
        per_eval = (elapsed / max(evals, 1)) * 1000.0
        entries.append(BenchmarkEntry(
            algorithm=algo_name, metric=fm, best_z_m=best_z,
            best_score=best_score, elapsed_s=elapsed, evaluations=evals,
            error_um=error_um, per_eval_ms=per_eval, downsampled=ds_factor,
        ))

    # ── Step 1: Full-resolution benchmarks ──
    for fm in metrics:
        z_arr = list(np.linspace(z_min_m, z_max_m, steps))
        _run_and_record("Linear Sweep",
            lambda f=fm, z=z_arr: autofocus_zscan(field, base_params, z, method, f), fm)

        _run_and_record("Coarse-to-Fine (Golden)",
            lambda f=fm: coarse_to_fine_search(
                field, base_params, method, f, z_min_m, z_max_m,
                coarse_steps=steps, fine_tolerance_m=1e-6), fm)

        _run_and_record("Robust Coarse-to-Fine",
            lambda f=fm: robust_coarse_to_fine_search(
                field, base_params, method, f, z_min_m, z_max_m,
                n_coarse=steps, refine_factor=8, smooth_sigma=1.5), fm)

        _run_and_record("Adaptive Gradient",
            lambda f=fm: adaptive_gradient_search(
                field, base_params, method, f, z_min_m, z_max_m,
                max_evaluations=steps), fm)

        _run_and_record("Adaptive Ratio",
            lambda f=fm: adaptive_ratio_search(
                field, base_params, method, f, z_min_m, z_max_m,
                max_evaluations=steps), fm)

        _run_and_record("Adaptive Bracketing",
            lambda f=fm: adaptive_bracketing_search(
                field, base_params, method, f, z_min_m, z_max_m,
                n_refine_levels=3, refine_divisions=6, smooth_sigma=1.0,
                max_evaluations=steps), fm)

    # ── Step 2: Downsampled (DS×2) benchmarks ──
    if include_downsampled:
        ds_field, ds_params = downsample_complex_field(field, base_params, factor=2)
        for fm in metrics:
            z_arr = list(np.linspace(z_min_m, z_max_m, steps))
            _run_and_record("DS×2 Linear Sweep",
                lambda f=fm, z=z_arr: autofocus_zscan(
                    ds_field, ds_params, z, method, f), fm, ds_factor=2)

            _run_and_record("DS×2 Robust C2F",
                lambda f=fm: robust_coarse_to_fine_search(
                    ds_field, ds_params, method, f, z_min_m, z_max_m,
                    n_coarse=steps, refine_factor=8, smooth_sigma=1.5), fm, ds_factor=2)

            _run_and_record("DS×2 Bracketing",
                lambda f=fm: adaptive_bracketing_search(
                    ds_field, ds_params, method, f, z_min_m, z_max_m,
                    n_refine_levels=3, refine_divisions=6, smooth_sigma=1.0,
                    max_evaluations=steps), fm, ds_factor=2)

    return AutofocusBenchmarkResult(
        entries=entries,
        field_shape=field.shape,
        z_range=(z_min_m, z_max_m),
        reference_z_m=ref_z,
        reference_metric=ref_metric.value,
        reference_elapsed_s=ref_elapsed,
    )


# ─────────────────────────────────────────────────────────────
# Adaptive Distance Search (auto z-range determination)
# ─────────────────────────────────────────────────────────────

@dataclass
class AdaptiveDistanceResult:
    """Result of adaptive distance search."""
    best_z_m: float
    best_score: float
    evaluations: int
    detected_range_m: Tuple[float, float]  # the narrowed z-range where signal was found
    z_history: np.ndarray
    score_history: np.ndarray
    method_name: str = "Adaptive Distance"


def adaptive_distance_search(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    initial_range_m: float = 0.5e-3,
    max_range_m: float = 50e-3,
    expand_factor: float = 2.0,
    signal_threshold: float = 0.3,
    max_evaluations: int = 150,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> AdaptiveDistanceResult:
    """
    Adaptive Distance autofocus — automatically discovers the z-range with signal.

    Strategy:
      1. Start with a small range around z=0 (± initial_range_m)
      2. Coarse scan → check if peak is near the edges
      3. If peak is at an edge → expand the range in that direction by expand_factor
      4. Repeat until peak is interior (not at edge) or max_range_m is reached
      5. Return the detected active zone for downstream refinement

    This eliminates the need for users to manually guess the z-range.
    The algorithm discovers the signal on its own.

    Parameters
    ----------
    field : complex field
    base_params : reconstruction parameters
    method : ASM or Fresnel
    metric : focus metric to use
    initial_range_m : starting half-range in metres (e.g. 0.5e-3 = ±0.5 mm)
    max_range_m : maximum half-range to expand to (safety limit)
    expand_factor : multiply the range by this when signal is at the edge
    signal_threshold : fraction of peak to define "active zone" (0-1)
    max_evaluations : total evaluation budget
    cancel_check, on_progress : callbacks
    """
    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    maximize = not _is_minimize(metric)
    evaluations = 0
    z_hist: List[float] = []
    f_hist: List[float] = []

    def evaluate(z: float) -> float:
        nonlocal evaluations
        evaluations += 1
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        if on_progress:
            on_progress(evaluations, max_evaluations)
        f = _eval(z)
        z_hist.append(z)
        f_hist.append(f)
        return f

    # ── Phase 1: Expanding discovery ──
    current_lo = -initial_range_m
    current_hi = initial_range_m
    all_z = np.array([], dtype=np.float64)
    all_f = np.array([], dtype=np.float64)
    edge_margin = 2  # peak within this many indices of edge → expand

    while evaluations < max_evaluations:
        remaining = max_evaluations - evaluations
        if remaining < 5:
            break

        n_pts = min(remaining, max(10, 15))
        z_scan = np.linspace(current_lo, current_hi, n_pts)
        f_scan = np.array([evaluate(float(z)) for z in z_scan])

        all_z = np.concatenate([all_z, z_scan])
        all_f = np.concatenate([all_f, f_scan])

        # Find peak in this scan
        work = -f_scan if not maximize else f_scan.copy()
        peak_idx = int(np.argmax(work))

        at_low_edge = peak_idx < edge_margin
        at_high_edge = peak_idx >= n_pts - edge_margin

        if not at_low_edge and not at_high_edge:
            # Peak is interior — signal found, stop expanding
            break

        # Expand in the direction of the peak
        span = current_hi - current_lo
        if at_low_edge and at_high_edge:
            # Ambiguous (very flat) — expand both sides
            current_lo = max(-max_range_m, current_lo - span * (expand_factor - 1) / 2)
            current_hi = min(max_range_m, current_hi + span * (expand_factor - 1) / 2)
        elif at_low_edge:
            current_lo = max(-max_range_m, current_lo - span * (expand_factor - 1))
        else:
            current_hi = min(max_range_m, current_hi + span * (expand_factor - 1))

        # Check if we've hit the max range
        if current_lo <= -max_range_m and current_hi >= max_range_m:
            break

    # ── Phase 2: Determine active zone from all collected data ──
    work_all = -all_f if not maximize else all_f.copy()
    w_min, w_max = work_all.min(), work_all.max()
    if w_max - w_min > 1e-15:
        # Sort by z for proper analysis
        sort_idx = np.argsort(all_z)
        sorted_z = all_z[sort_idx]
        sorted_work = work_all[sort_idx]
        w_norm = (sorted_work - w_min) / (w_max - w_min)
        w_smooth = gaussian_filter1d(w_norm, sigma=max(1.0, len(w_norm) / 20.0))

        above = w_smooth >= signal_threshold * np.max(w_smooth)
        active_indices = np.where(above)[0]

        if len(active_indices) == 0:
            peak_idx = int(np.argmax(w_smooth))
            margin = max(2, len(w_norm) // 10)
            active_indices = np.arange(
                max(0, peak_idx - margin),
                min(len(w_norm), peak_idx + margin + 1)
            )

        idx_lo = max(0, int(active_indices[0]) - 1)
        idx_hi = min(len(sorted_z) - 1, int(active_indices[-1]) + 1)
        active_z_min = float(sorted_z[idx_lo])
        active_z_max = float(sorted_z[idx_hi])

        raw_peak = int(np.argmax(sorted_work))
        best_z = float(sorted_z[raw_peak])
        best_f = float(all_f[sort_idx[raw_peak]])
    else:
        # No variation at all
        active_z_min = float(current_lo)
        active_z_max = float(current_hi)
        best_z = 0.0
        best_f = float(all_f[0]) if len(all_f) > 0 else 0.0

    return AdaptiveDistanceResult(
        best_z_m=best_z,
        best_score=best_f,
        evaluations=evaluations,
        detected_range_m=(active_z_min, active_z_max),
        z_history=np.array(z_hist),
        score_history=np.array(f_hist),
    )


# ─────────────────────────────────────────────────────────────
# Adaptive Focus State
# ─────────────────────────────────────────────────────────────

@dataclass
class AdaptiveFocusState:
    current_z_m: float
    search_range_m: float
    steps: int = 5
    metric: FocusMetric = FocusMetric.TOTAL_VARIATION
    algorithm: str = "Linear Sweep (Exhaustive)"

    def step(
        self,
        field: np.ndarray,
        base_params: ReconstructionParams,
        method: ReconstructionMethod,
    ) -> float:
        zmin = self.current_z_m - self.search_range_m / 2.0
        zmax = self.current_z_m + self.search_range_m / 2.0
        step_init = self.search_range_m / max(self.steps, 5)

        if "Robust" in self.algorithm:
            res = robust_coarse_to_fine_search(
                field=field,
                base_params=base_params,
                method=method,
                metric=self.metric,
                z_min_m=zmin,
                z_max_m=zmax,
                n_coarse=self.steps,
                refine_factor=4,
                smooth_sigma=1.5,
            )
            self.current_z_m = res.best_z_m

        elif "Golden" in self.algorithm or "Coarse-to-Fine" in self.algorithm:
            res = coarse_to_fine_search(
                field=field,
                base_params=base_params,
                method=method,
                metric=self.metric,
                z_min_m=zmin,
                z_max_m=zmax,
                coarse_steps=self.steps,
                fine_tolerance_m=1e-6,
            )
            self.current_z_m = res.best_z_m

        elif "Gradient" in self.algorithm:
            res = adaptive_gradient_search(
                field=field,
                base_params=base_params,
                method=method,
                metric=self.metric,
                z_min_m=zmin,
                z_max_m=zmax,
                step_init=step_init,
                max_evaluations=self.steps,
            )
            self.current_z_m = res.best_z_m

        elif "Ratio" in self.algorithm:
            res = adaptive_ratio_search(
                field=field,
                base_params=base_params,
                method=method,
                metric=self.metric,
                z_min_m=zmin,
                z_max_m=zmax,
                step_init=step_init,
                max_evaluations=self.steps,
            )
            self.current_z_m = res.best_z_m

        elif "Distance" in self.algorithm:
            half_range = (zmax - zmin) / 2.0
            res = adaptive_distance_search(
                field=field,
                base_params=base_params,
                method=method,
                metric=self.metric,
                initial_range_m=half_range,
                max_range_m=half_range * 5.0,
                max_evaluations=self.steps,
            )
            self.current_z_m = res.best_z_m

        elif "Bracket" in self.algorithm:
            res = adaptive_bracketing_search(
                field=field,
                base_params=base_params,
                method=method,
                metric=self.metric,
                z_min_m=zmin,
                z_max_m=zmax,
                step_init=step_init,
                n_refine_levels=2,
                refine_divisions=5,
                smooth_sigma=1.0,
                max_evaluations=self.steps,
            )
            self.current_z_m = res.best_z_m

        else:  # Linear Sweep (default)
            z_arr = list(np.linspace(zmin, zmax, self.steps))
            res = autofocus_zscan(
                field=field,
                base_params=base_params,
                z_values_m=z_arr,
                method=method,
                metric=self.metric,
            )
            self.current_z_m = res.best_z_m

        return self.current_z_m
