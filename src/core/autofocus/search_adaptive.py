"""Adaptive autofocus search algorithms: gradient-based, ratio-based, bracketing, and auto-distance discovery.

Also exposes `AdaptiveFocusState` — a convenience state machine used by the live
camera mode to re-run a chosen algorithm on each new frame around a moving best-z.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..reconstruction import ReconstructionMethod, ReconstructionParams
from .evaluator import AutofocusCancelled, _make_fast_evaluator
from .metrics import FocusMetric, _is_minimize
from .search_classic import (
    autofocus_zscan,
    coarse_to_fine_search,
    robust_coarse_to_fine_search,
)


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
    prev_deriv: Optional[float] = None  # None = no previous derivative yet

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
        # Skip all step adjustment until we have two derivatives to compare —
        # otherwise the very first iteration always triggers "abs_d > 0*1.5"
        # and shrinks the step before the walker leaves the flat start region.
        if prev_deriv is not None:
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

    # ── Phase 1b: Coverage safety net ──
    # If the greedy walker got trapped at an early local max (e.g. noisy shoulder
    # before the true peak), it may leave a large portion of [z_min, z_max]
    # unexplored. Without this sweep, Phase 2 refinement only narrows around
    # best_z — so a missed peak stays missed. A short uniform sweep across the
    # range is cheap and reliably escapes the trap.
    covered = max(z_hist) - min(z_hist)
    if covered < (z_max_m - z_min_m) * 0.75:
        sweep_budget = min(max(int(max_evaluations * 0.25), 6), 12,
                           max_evaluations - evaluations)
        if sweep_budget >= 3:
            z_sweep = np.linspace(z_min_m, z_max_m, sweep_budget)
            sweep_step = (z_max_m - z_min_m) / max(sweep_budget - 1, 1)
            for zp in z_sweep:
                if cancel_check and cancel_check():
                    raise AutofocusCancelled()
                f = evaluate(float(zp))
                b = (f > best_f) if maximize else (f < best_f)
                if b:
                    best_f = f
                    best_z = float(zp)
                z_hist.append(float(zp))
                f_hist.append(f)
                s_hist.append(sweep_step)

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

        # Peak significance check: an interior peak is only trustworthy if it
        # rises clearly above the scan's edges and overall dynamic range. On a
        # monotonic flank or flat plateau, noise picks a random interior point
        # that is not actually the peak — we must keep expanding instead.
        left_val = float(f_scan[0])
        right_val = float(f_scan[-1])
        peak_val = float(f_scan[peak_idx])
        f_range = float(np.ptp(f_scan))
        baseline_abs = max(abs(float(np.median(f_scan))), 1e-15)

        if maximize:
            rise_from_edges = peak_val - max(left_val, right_val)
        else:
            rise_from_edges = min(left_val, right_val) - peak_val

        # Require ≥3% relative dynamic range AND ≥30% of that range between
        # the peak and the worst edge. Both guards are cheap noise rejectors.
        peak_real = (
            f_range / baseline_abs > 0.03
            and rise_from_edges > 0.3 * f_range
        )

        if not at_low_edge and not at_high_edge and peak_real:
            # Peak is interior and clearly rises above edges — signal found.
            break

        # If the peak is interior but not significant, the scan is dominated by
        # noise on a flat/monotonic surface — expand both sides to keep hunting.
        # If the peak sits at an edge we trust that direction regardless of
        # significance (monotonic flank rising toward the true focus).
        if not peak_real and not at_low_edge and not at_high_edge:
            at_low_edge = at_high_edge = True

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
