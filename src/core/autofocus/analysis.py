"""Diagnostic / research-grade autofocus utilities.

- `auto_select_metric`: scores every phase metric for reliability (peak prominence,
  few secondary peaks) over a coarse z-sweep and picks the most reliable one.
- `scan_metric_landscape`: returns raw + smoothed curves for each metric to inspect
  where they peak and how noisy they are.
- `autofocus_benchmark`: exhaustive algorithm × metric benchmark with reference z
  and timing, used by the GUI "Benchmark" action and the CLI bench scripts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..fft_backend import get_best_fft_backend
from ..reconstruction import (
    CachedReconstructor,
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)
from .evaluator import (
    AutofocusCancelled,
    _make_fast_evaluator,
    downsample_complex_field,
)
from .metrics import FocusMetric, _calc_metric, _is_minimize
from .search_adaptive import (
    adaptive_bracketing_search,
    adaptive_gradient_search,
    adaptive_ratio_search,
)
from .search_classic import (
    autofocus_zscan,
    coarse_to_fine_search,
    robust_coarse_to_fine_search,
)

log = logging.getLogger(__name__)


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
        values = np.array([_calc_metric(c, fm) for c in complex_fields])
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
        # No INTERIOR peak → the metric's optimum sits at/outside the scanned
        # range (monotonic / edge-focus curve): it is the LEAST reliable, so it
        # scores 0. The old fallback used np.max(values_smooth) ≈ 1.0 on the
        # [0,1]-normalised curve, which let a peakless edge metric beat a
        # genuinely-peaked one (prominence < 1) — selecting the worst metric
        # (2026-07-08 review). Reliability == peak prominence, and a peakless
        # curve has none.
        max_prominence = float(np.max(props["prominences"])) if n_peaks > 0 else 0.0

        # Score: high prominence, few secondary peaks
        scores[fm] = max_prominence - 0.2 * max(0, n_peaks - 1)

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    log.info("auto_select_metric scores: %s  →  %s", scores, best)
    return best


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
        vals = np.array([_calc_metric(c, fm) for c in complex_fields])
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


# ───────────────────────────────────────────────────────────────────────────
# Multi-focus discovery (v1.0.1-ux prototype, full UI in v1.1-sci)
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FocusCandidate:
    """One plausible focus plane in a multi-object scene.

    ``z_m`` is the propagation distance at which a local extremum of the
    focus metric sits. ``score`` is the metric value (higher = stronger
    focus signal, post-normalisation). ``prominence`` is ``scipy.signal``
    peak prominence, useful for ranking candidates and filtering out
    diffraction-ring ghosts. ``rank`` orders candidates by decreasing
    prominence (0 = strongest).
    """
    z_m: float
    score: float
    prominence: float
    rank: int


def find_focus_candidates(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    z_min_m: float,
    z_max_m: float,
    *,
    n_steps: int = 60,
    metric: FocusMetric = FocusMetric.ENTROPY,
    smooth_sigma: float = 2.0,
    min_prominence: float = 0.05,
    min_distance_frac: float = 0.05,
    max_candidates: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[FocusCandidate]:
    """Return every plausible focus plane in ``[z_min, z_max]``.

    Runs a single metric sweep over ``n_steps`` equally-spaced z values,
    normalises + flips + smooths the landscape (so ``find_peaks`` always
    looks for *maxima* regardless of whether the metric minimises or
    maximises at focus), then returns every peak whose prominence
    exceeds ``min_prominence`` and whose separation from other peaks is
    at least ``min_distance_frac × n_steps`` samples.

    Compared with :func:`autofocus_zscan` this is the multi-object
    generalisation: autofocus_zscan collapses the landscape to a single
    best_z, while this helper keeps every significant extremum. Useful
    when a scene contains several objects at different depths (cell
    clusters, stacked microstructures, multi-layer samples).

    Performance note: the landscape scan is ``n_steps`` propagations +
    metric evaluations, same cost as ``autofocus_zscan``. The peak
    finder itself is microseconds — free on top of the scan.
    """
    if n_steps < 3:
        raise ValueError(f"n_steps must be >= 3 (got {n_steps})")
    if z_max_m <= z_min_m:
        raise ValueError(f"z_max_m ({z_max_m}) must be > z_min_m ({z_min_m})")

    # v2.0.9: align with the single-z autofocus path — every search
    # variant (autofocus_zscan + coarse_to_fine + robust + all three
    # adaptive algorithms) uses _make_fast_evaluator. Multi-focus
    # was the odd one out; it ran propagate(force_python=True) in a
    # loop, which re-checks ``_GLOBAL_RECON_CACHE`` + ``id(field)``
    # and does a defensive ``spectrum.copy()`` each call.
    #
    # Perf-wise this refactor is a wash on our bench (1594 ms vs
    # 1598 ms @ 1024² × 60 steps) — the copy is ~0.6 % of runtime,
    # dominated by the FFT + metric calc. What it does earn is a
    # single code path for every search variant; a future perf win
    # in _make_fast_evaluator will flow through to multi-focus
    # automatically instead of needing a separate patch.
    _eval = _make_fast_evaluator(field, base_params, method, metric)
    z_values = np.linspace(z_min_m, z_max_m, n_steps)
    scores = np.empty(n_steps, dtype=np.float64)
    for i, z in enumerate(z_values):
        # Cooperative cancellation — scan can be Esc-aborted by a UI
        # layer. Raises at the boundary between propagations so FFTs
        # already in flight still finish cleanly.
        if cancel_check and cancel_check():
            raise AutofocusCancelled()
        scores[i] = _eval(float(z))

    # Normalise into [0, 1] so prominence threshold has a fixed meaning
    # across metrics.
    vmin, vmax = float(scores.min()), float(scores.max())
    if vmax - vmin < 1e-15:
        return []
    norm = (scores - vmin) / (vmax - vmin)
    # Flip minimise-at-focus metrics so we always look for maxima.
    if _is_minimize(metric):
        norm = 1.0 - norm

    smoothed = gaussian_filter1d(norm, sigma=smooth_sigma)
    distance = max(1, int(round(n_steps * min_distance_frac)))
    peaks_idx, props = find_peaks(
        smoothed,
        prominence=min_prominence,
        distance=distance,
    )
    if peaks_idx.size == 0:
        return []

    prominences = props.get("prominences", np.zeros(peaks_idx.shape))
    order = np.argsort(-prominences)  # descending prominence
    if max_candidates is not None:
        order = order[: max_candidates]

    candidates: List[FocusCandidate] = []
    for rank, idx_into_peaks in enumerate(order):
        i = int(peaks_idx[idx_into_peaks])
        candidates.append(FocusCandidate(
            z_m=float(z_values[i]),
            score=float(smoothed[i]),
            prominence=float(prominences[idx_into_peaks]),
            rank=rank,
        ))
    return candidates


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
