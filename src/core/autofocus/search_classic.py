"""Classic autofocus search algorithms: linear sweep, Golden Section, coarse-to-fine, and the smoothing-based robust variant."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..reconstruction import ReconstructionMethod, ReconstructionParams
from ..fft_backend import FFTBackend
from .evaluator import (
    AutofocusCancelled,
    AutoFocusResult,
    focus_landscape_warning,
    _make_batch_evaluator,
    _make_fast_evaluator,
)
from .metrics import FocusMetric, _is_minimize


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
    batch_backend: Optional[FFTBackend] = None,
) -> AutoFocusResult:
    """Linear z-scan autofocus.

    v2.1.0: when ``batch_backend`` is supplied AND its
    ``supports_batched`` is True (i.e. a real GPU backend, not the
    default-loop fallback), the scan runs through
    :func:`_make_batch_evaluator` — one batched IFFT for every
    requested z. Falls back to the serial path otherwise so
    cancel + progress callbacks keep working unchanged.
    """
    minimize = _is_minimize(metric)
    total = len(z_values_m)

    # Batch path — only when backend reports a real batched op.
    # Cancellation is coarser: we check before the batched call.
    # Progress fires once at start + once at end so the UI bar
    # still moves.
    if batch_backend is not None and batch_backend.supports_batched:
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        if on_progress:
            on_progress(0, total)
        evaluate_many = _make_batch_evaluator(
            field, base_params, method, metric,
            backend=batch_backend,
            roi_bounds=roi_bounds, ref_field=ref_field,
        )
        all_scores = evaluate_many(z_values_m)
        scores = {z: float(s) for z, s in zip(z_values_m, all_scores)}
        if minimize:
            best_idx = int(np.argmin(all_scores))
        else:
            best_idx = int(np.argmax(all_scores))
        if on_progress:
            on_progress(total, total)
        return AutoFocusResult(
            best_z_m=z_values_m[best_idx], scores=scores,
            warning=focus_landscape_warning(scores),
        )

    _eval = _make_fast_evaluator(field, base_params, method, metric, roi_bounds=roi_bounds, ref_field=ref_field)
    scores = {}
    best_z = z_values_m[0]
    best_score = float('inf') if minimize else -float('inf')

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

    return AutoFocusResult(
        best_z_m=best_z, scores=scores,
        warning=focus_landscape_warning(scores),
    )


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
        roi_bounds=roi_bounds,   # 2026-07-08: was dropped → fine phase
        ref_field=ref_field,     # optimised the FULL frame, not the ROI
    )

    return GoldenSearchResult(
        best_z_m=res.best_z_m,
        best_score=res.best_score,
        iterations=res.iterations + coarse_steps,
        evaluations=res.evaluations + state["evaluations"]
    )


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
