"""Thread-based workers for autofocus / QPI / depth map / report
export. Mirrors ``src/ui2/reconstruction.py``'s pattern — each worker
takes a job + two callbacks and runs on a dedicated executor so the
DPG render loop stays responsive.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from core.audit import get_audit_log
from core.autofocus import (
    AutofocusCancelled,
    FocusMetric,
    autofocus_zscan,
    find_focus_candidates,
)
from core.autofocus.search_adaptive import (
    adaptive_bracketing_search,
    adaptive_distance_search,
    adaptive_gradient_search,
)
from core.autofocus.search_classic import (
    coarse_to_fine_search,
    robust_coarse_to_fine_search,
)
from core.depth_map import (
    ClusterHeight,
    DepthMapResult,
    compute_depth_map,
    segment_depth_clusters,
    write_tomography_bundle,
)
from core.ingestion import load_any
from core.offaxis import OffAxisParams, extract_complex_field_offaxis
from core.phase_unwrap import UnwrapConfig, UnwrapMethod, unwrap_phase_advanced
from core.qpi import QPIMode, QPIResult, compute_qpi
from core.qpi_batch import QPIBatchEntry, run_qpi_for_candidates, write_qpi_batch_csv
from core.qpi_export import write_qpi_csv
from core.reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)
from core.report import generate_html_report

from .reconstruction import ReconError, ReconParams

_LOG = logging.getLogger(__name__)


def _metric_from_params(params: ReconParams) -> FocusMetric:
    """Look up the FocusMetric enum from the string stored on ReconParams.

    Falls back to LAPLACIAN_VARIANCE (the v2.0.2 default) if the name
    doesn't match — keeps old state files working."""
    name = str(getattr(params, "autofocus_metric",
                       "LAPLACIAN_VARIANCE") or "").upper()
    try:
        return FocusMetric[name]
    except KeyError:
        return FocusMetric.LAPLACIAN_VARIANCE


def available_focus_metrics() -> list[str]:
    """Every FocusMetric name, in enum declaration order. Used by the
    sidebar combo so the UI stays in sync with core automatically."""
    return [m.name for m in FocusMetric]


# Canonical list of autofocus search algorithms the sidebar exposes.
# Keys are user-visible strings; the values are the legacy v1 names
# used internally (kept here to avoid another round of string refactor
# when porting further v1 algorithms).
_AUTOFOCUS_ALGORITHMS: list[str] = [
    "zscan",               # fixed-step linear sweep — simplest, baseline
    "coarse_to_fine",      # coarse grid → Golden-Section refinement
    "robust",              # coarse scan + Gaussian-smoothed fine refine
    "adaptive_gradient",   # variable step driven by metric derivative
    "adaptive_bracketing", # nested bracketing + divisions refinement
    "adaptive_distance",   # auto-discovers the z-range from scratch
]

# Which UI fields each algorithm actually consumes. The sidebar
# greys out / relabels anything an algorithm ignores so the user
# can see at a glance what the input actually does. Used by the
# sidebar layer via :func:`af_algorithm_input_profile` — keeps the
# UI / dispatch contract in one place so drift is impossible.
_AF_ALGO_PROFILES: dict[str, dict] = {
    "zscan": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "steps",
        "steps_tip": "Number of z samples on the uniform grid.",
    },
    "coarse_to_fine": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "coarse steps",
        "steps_tip": (
            "Coarse-grid sample count. After the coarse pass the "
            "algorithm refines with Golden-Section search — no "
            "extra budget needed."
        ),
    },
    "robust": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "coarse steps",
        "steps_tip": (
            "Coarse-grid sample count before Gaussian smoothing "
            "+ 8× refine pass."
        ),
    },
    "adaptive_gradient": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "max evals",
        "steps_tip": (
            "Evaluation budget — the algorithm adapts step size "
            "itself (big in flat regions, tiny near peak). Unlike "
            "the fixed grid, this is a cap, not a sample count."
        ),
    },
    "adaptive_bracketing": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "max evals",
        "steps_tip": (
            "Evaluation budget across the nested bracketing + "
            "refinement levels (3 levels × 8 divisions + refine). "
            "Cap, not fixed sample count."
        ),
    },
    "adaptive_distance": {
        "uses_z_range": False,
        "uses_steps": True,
        "steps_label": "max evals",
        "steps_tip": (
            "Evaluation budget. Adaptive-distance starts with a "
            "small range around z = 0 and expands until the peak "
            "is interior — user-supplied z_min / z_max are "
            "ignored."
        ),
    },
}


def af_algorithm_input_profile(algo: str) -> dict:
    """Return which sidebar inputs a given algorithm actually uses.

    Keys:

    * ``uses_z_range`` — whether ``af_z_min_mm`` / ``af_z_max_mm``
      influence the run (False for ``adaptive_distance``).
    * ``uses_steps`` — whether ``af_n_steps`` matters at all.
    * ``steps_label`` — the label the sidebar should display
      (e.g. "max evals" for adaptive variants, "steps" for grid
      methods).
    * ``steps_tip`` — per-algorithm tooltip copy.
    """
    return dict(_AF_ALGO_PROFILES.get(
        str(algo or "").lower(),
        _AF_ALGO_PROFILES["zscan"],
    ))


def available_autofocus_algorithms() -> list[str]:
    """Names of autofocus search algorithms the sidebar can pick
    from. Kept as a free function so the UI layer can build the
    combo without importing the dispatch details."""
    return list(_AUTOFOCUS_ALGORITHMS)


def _extract_best_z(result: Any) -> float:
    """Pull ``best_z_m`` out of any of the five result dataclasses
    core returns — autofocus_zscan returns ``AutoFocusResult``,
    coarse/robust return their own, adaptive variants return
    ``AdaptiveStepResult`` / ``RobustResult``. They all carry
    ``best_z_m`` as the first field by convention."""
    return float(getattr(result, "best_z_m"))


def _extract_best_score(result: Any) -> float:
    """Best-score extractor with graceful fallbacks — some result
    dataclasses expose it as ``best_score`` directly, autofocus_zscan
    carries it in its ``scores`` dict keyed by z."""
    if hasattr(result, "best_score"):
        return float(result.best_score)
    if hasattr(result, "scores"):
        scores = result.scores
        best_z = float(getattr(result, "best_z_m"))
        return float(scores.get(best_z, 0.0))
    return 0.0


def _extract_evaluations(result: Any, fallback: int) -> int:
    """Count the number of z evaluations consumed. Adaptive results
    expose it as ``evaluations``; zscan we compute from the supplied
    z list length upstream and pass via ``fallback``."""
    if hasattr(result, "evaluations"):
        return int(result.evaluations)
    return int(fallback)


def _params_for_audit(params: ReconParams, **extra: Any) -> dict:
    """Flatten ReconParams into a JSON-friendly dict for audit logging.

    Mirrors the v1 ``main_window._audit`` spirit: every field that
    influenced the result lands in the record so a future operator can
    reproduce the run from the log alone. Extra kwargs (z_range,
    sample_id, hologram path, …) get merged on top."""
    d: dict[str, Any] = {
        "wavelength_nm": float(params.wavelength_nm),
        "pixel_um": float(params.pixel_um),
        "z_mm": float(params.z_mm),
        "mask_radius": int(params.mask_radius),
        "method": str(params.method),
        "magnification": float(params.magnification),
        "pixel_is_effective": bool(params.pixel_is_effective),
        "effective_pixel_um": float(params.effective_pixel_um()),
        "n_sample": float(params.n_sample),
        "n_medium": float(params.n_medium),
        "autofocus_metric": str(params.autofocus_metric),
        "af_z_min_mm": float(params.af_z_min_mm),
        "af_z_max_mm": float(params.af_z_max_mm),
        "af_n_steps": int(params.af_n_steps),
        "reference_path": (str(params.reference_path)
                           if params.reference_path else ""),
        "subtract_reference": bool(params.subtract_reference),
    }
    for k, v in extra.items():
        if v is None:
            continue
        d[k] = str(v) if isinstance(v, Path) else v
    return d


def _emit_audit(action: str, params: dict,
                result_summary: Optional[dict] = None) -> None:
    """Best-effort audit write — never raises, never blocks science."""
    try:
        get_audit_log().record(
            action=action, params=params, result_summary=result_summary,
        )
    except Exception:
        _LOG.warning("audit write failed for %s", action, exc_info=True)


# ---------------------------------------------------------------------------
# Shared helpers — a demodulated complex field any worker starts from.
# ---------------------------------------------------------------------------

def _preprocess_raw(raw: np.ndarray, params: ReconParams) -> np.ndarray:
    """Pre-offaxis preprocessing shared by every pipeline path.

    Mirrors ``core.batch_renderer``'s branch order:

    1. Subtract the image mean when the user has armed DC removal
       (reduces the 0-order bleed into the +1 order after FFT).
    2. Apply a 2-D Hann taper when requested (suppresses edge
       discontinuities that leak into the FFT as wide sinc wings).
    3. Normalise by the absolute peak so downstream math stays in
       [-1, 1].

    ``ReconstructionDriver._run`` and ``ScienceDriver._prepare_field``
    used to duplicate this logic — they drifted in v2.0.5 (only
    the Reconstruct button got subtract_mean / hann wiring, every
    other pipeline ran raw normalisation). Bug report 2026-04-24:
    "reference subtract isn't working" was half-caused by that
    skew, so it's extracted here as a single source of truth.
    """
    raw = np.asarray(raw, dtype=np.float32, copy=False)
    if getattr(params, "subtract_mean", True):
        raw = raw - float(np.mean(raw))
    if getattr(params, "hann_window", False):
        wy = np.hanning(raw.shape[0]).astype(np.float32)
        wx = np.hanning(raw.shape[1]).astype(np.float32)
        raw = raw * (wy[:, None] * wx[None, :])
    peak = float(np.max(np.abs(raw)))
    if peak > 0:
        raw = raw / peak
    return raw


def _extract_field_with_reference(
    raw: np.ndarray,
    params: ReconParams,
    offaxis: OffAxisParams,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Off-axis demodulation + optional reference-division.

    Returns ``(field, +1_order_centre)``. ``+1_order_centre`` is
    the ``(y, x)`` pixel the demodulator masked out of the sample
    hologram — downstream diagnostics still want to see it even
    when reference division is on.

    ``subtract_reference=True`` loads a second hologram (the
    "reference" — sample-free background), demodulates *it* with
    the same mask, and divides the two complex fields. Division
    (not subtraction) because the reference carries the
    illumination profile multiplicatively; dividing cancels
    illumination non-uniformity and leaves the sample-induced
    perturbation behind.

    Up to v2.0.8 this logic existed only in
    ``ReconstructionDriver._run`` — autofocus, multi-focus, QPI
    and depth map all ignored ``subtract_reference`` because
    ``_prepare_field`` skipped it. User noticed with a real lab
    hologram. Now sits on the shared prep path so every pipeline
    respects the reference toggle identically.
    """
    field, center = extract_complex_field_offaxis(raw, offaxis)
    if not (params.subtract_reference and params.reference_path):
        return field, tuple(int(v) for v in center)
    try:
        ref_loaded = load_any(params.reference_path)
        ref_raw = np.asarray(ref_loaded.array, dtype=np.float32)
        if ref_raw.ndim == 3:
            ref_raw = ref_raw[..., 0]
        ref_raw = _preprocess_raw(ref_raw, params)
        ref_field, _ = extract_complex_field_offaxis(ref_raw, offaxis)
        eps = 1e-9
        return field / (ref_field + eps), tuple(int(v) for v in center)
    except Exception as exc:
        _LOG.warning("reference subtract failed (%s) — "
                     "falling back to unreferenced field", exc)
        return field, tuple(int(v) for v in center)


def _prepare_field(
    hologram_path: Path, params: ReconParams,
    *,
    apply_af_roi: bool = False,
) -> Tuple[np.ndarray, ReconstructionParams, ReconstructionMethod]:
    loaded = load_any(hologram_path)
    raw = np.asarray(loaded.array, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = _preprocess_raw(raw, params)
    offaxis = OffAxisParams(radius=int(params.mask_radius))
    field, _center = _extract_field_with_reference(raw, params, offaxis)
    # v2.0.7: optional ROI mask for local autofocus. The full complex
    # field still feeds reconstruction (so the user's display doesn't
    # change), but scan metrics see only pixels inside the rectangle —
    # lets the lab focus on a specific cell when the scene has several
    # at different depths.
    if apply_af_roi and params.af_roi is not None:
        y0, x0, y1, x1 = params.af_roi
        h, w = field.shape
        yy0 = int(max(0, min(1, y0)) * (h - 1))
        xx0 = int(max(0, min(1, x0)) * (w - 1))
        yy1 = int(max(0, min(1, y1)) * (h - 1))
        xx1 = int(max(0, min(1, x1)) * (w - 1))
        yy0, yy1 = sorted((yy0, yy1))
        xx0, xx1 = sorted((xx0, xx1))
        if yy1 > yy0 and xx1 > xx0:
            mask = np.zeros(field.shape, dtype=np.float32)
            mask[yy0:yy1 + 1, xx0:xx1 + 1] = 1.0
            field = field * mask
    # v2.0.3: hand the effective pixel (camera_pixel / M) to the kernel.
    # v1 did this via ReconTab; v2 was missing the magnification widget
    # altogether, so autofocus/QPI/depth were silently running on the
    # raw camera-pixel value — off by M² in z for every microscope setup.
    # v2.0.8: pass ``n_medium`` into the propagation kernel too. v2.0.2
    # hardcoded ``n=1.0`` (vacuum/air), which silently made every
    # aqueous-culture autofocus converge to the wrong z because the
    # effective wavelength in water is λ/n_medium, not λ.
    base = ReconstructionParams(
        wavelength_m=float(params.wavelength_nm) * 1e-9,
        pixel_size_m=float(params.effective_pixel_um()) * 1e-6,
        z_m=0.0,
        n=float(params.n_medium),
    )
    method = (ReconstructionMethod.ASM
              if params.method.upper() == "ASM"
              else ReconstructionMethod.FRESNEL)
    return field, base, method


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AutofocusResult:
    best_z_m: float
    score: float
    scanned: int
    runtime_ms: float


@dataclass
class MultiFocusResult:
    candidates: List[Any]  # FocusCandidate from core.autofocus
    runtime_ms: float


@dataclass
class QPIOneShotResult:
    qpi: QPIResult
    runtime_ms: float


@dataclass
class QPIBatchResultWrap:
    entries: List[QPIBatchEntry]
    runtime_ms: float


@dataclass
class DepthMapResultWrap:
    result: DepthMapResult
    clusters: List[ClusterHeight]
    runtime_ms: float


@dataclass
class ReportExportResult:
    path: Path
    runtime_ms: float


@dataclass
class BundleExportResult:
    files: List[Path]
    runtime_ms: float


# ---------------------------------------------------------------------------
# Driver — one thread pool, lots of job types.
# ---------------------------------------------------------------------------

class ScienceDriver:
    """Single-threaded job runner for all non-recon pipeline calls.

    Carries a :class:`threading.Event` so the UI can ``cancel()`` a
    running scan — core scan loops poll ``cancel_check`` between
    propagations and raise :class:`AutofocusCancelled` when the event
    is set. Jobs that finish naturally (not cancelled) emit audit
    records via :func:`_emit_audit`."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="science",
        )
        self._inflight: Optional[Future] = None
        self._lock = threading.Lock()
        # Shared cancellation signal. Each ``_submit`` clears it so a
        # cancel on a completed job doesn't poison the next one.
        self._cancel_event = threading.Event()

    def _cancel_check(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> bool:
        """Request a cancel. Returns True if an in-flight job was
        flagged. Core scan loops observe the flag on their next step
        boundary and raise AutofocusCancelled. Single-call work like
        QPI-at-one-z can't be interrupted mid-flight but its result
        is discarded in ``_dispatch`` below."""
        with self._lock:
            if self._inflight and not self._inflight.done():
                self._cancel_event.set()
                return True
        return False

    def _submit(
        self,
        fn: Callable[[], Any],
        on_result: Callable[[Any], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        with self._lock:
            if self._inflight and not self._inflight.done():
                on_error(ReconError(
                    "Another analysis is running — wait for it to finish."))
                return
            self._cancel_event.clear()
            self._inflight = self._pool.submit(fn)
            self._inflight.add_done_callback(
                lambda f: self._dispatch(f, on_result, on_error),
            )

    def _dispatch(self, fut, on_result, on_error):
        cancelled = self._cancel_event.is_set()
        try:
            result = fut.result()
        except AutofocusCancelled:
            on_error(ReconError("Cancelled."))
            return
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("science worker failed")
            on_error(ReconError(str(exc) or repr(exc)))
            return
        if cancelled:
            # Result arrived after Esc — discard so the UI doesn't
            # paint a "done" state on top of a "cancelled" status.
            on_error(ReconError("Cancelled."))
            return
        on_result(result)

    def shutdown(self) -> None:
        self._cancel_event.set()  # unblock any pending check
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- Autofocus (single best z) ------------------------------------

    def run_autofocus(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_min_mm: float, z_max_mm: float, n_steps: int = 40,
        metric: Optional[FocusMetric] = None,
        sample_id: str = "",
        on_result: Callable[[AutofocusResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        """Dispatches to one of five search algorithms based on
        ``params.af_algorithm`` — v1 had six; v2.0.8 restores the
        five most useful (``zscan``, ``coarse_to_fine``, ``robust``,
        ``adaptive_gradient``, ``adaptive_bracketing``). Each branch
        adapts the core result dataclass into :class:`AutofocusResult`
        so the UI stays algorithm-agnostic. Cancel and audit
        semantics are identical across algorithms."""
        chosen_metric = metric or _metric_from_params(params)
        cancel = self._cancel_check
        algo = str(
            getattr(params, "af_algorithm", "zscan") or "zscan",
        ).lower()

        def run() -> AutofocusResult:
            import time
            t0 = time.monotonic()
            field, base, method = _prepare_field(
                hologram_path, params, apply_af_roi=True,
            )
            z_min_m = float(z_min_mm) * 1e-3
            z_max_m = float(z_max_mm) * 1e-3
            scanned = int(n_steps)
            if algo == "coarse_to_fine":
                core_result = coarse_to_fine_search(
                    field, base, method, chosen_metric,
                    z_min_m=z_min_m, z_max_m=z_max_m,
                    coarse_steps=n_steps,
                    fine_tolerance_m=max(1e-9,
                                         (z_max_m - z_min_m) / 1e4),
                    cancel_check=cancel,
                )
            elif algo == "robust":
                core_result = robust_coarse_to_fine_search(
                    field, base, method, chosen_metric,
                    z_min_m=z_min_m, z_max_m=z_max_m,
                    n_coarse=n_steps, refine_factor=8,
                    smooth_sigma=1.5,
                    cancel_check=cancel,
                )
            elif algo == "adaptive_gradient":
                core_result = adaptive_gradient_search(
                    field, base, method, chosen_metric,
                    z_min_m=z_min_m, z_max_m=z_max_m,
                    max_evaluations=n_steps,
                    cancel_check=cancel,
                )
            elif algo == "adaptive_bracketing":
                core_result = adaptive_bracketing_search(
                    field, base, method, chosen_metric,
                    z_min_m=z_min_m, z_max_m=z_max_m,
                    n_refine_levels=3, refine_divisions=8,
                    smooth_sigma=1.0,
                    max_evaluations=n_steps,
                    cancel_check=cancel,
                )
            elif algo == "adaptive_distance":
                # Auto-range discovery. The core helper starts
                # with a tiny window around z = 0 and expands
                # until the peak lands inside the bracket. User's
                # z_min / z_max widgets are ignored *as bounds*
                # but still used as a ceiling hint — adaptive
                # distance reaches at least as far as whichever
                # endpoint is largest in magnitude, with a 50 mm
                # floor so the algorithm doesn't stall out on a
                # zeroed widget.
                max_range = max(
                    abs(z_max_m), abs(z_min_m), 50e-3,
                )
                core_result = adaptive_distance_search(
                    field, base, method, chosen_metric,
                    initial_range_m=0.5e-3,
                    max_range_m=max_range,
                    expand_factor=2.0,
                    signal_threshold=0.3,
                    max_evaluations=n_steps,
                    cancel_check=cancel,
                )
            else:
                # Linear sweep — v2.0.2 behaviour, still the default.
                zs = list(np.linspace(z_min_m, z_max_m, n_steps))
                core_result = autofocus_zscan(
                    field, base, zs, method, chosen_metric,
                    cancel_check=cancel,
                )
                scanned = len(zs)
            best_z_m = _extract_best_z(core_result)
            best_score = _extract_best_score(core_result)
            evaluations = _extract_evaluations(core_result, scanned)
            runtime_ms = (time.monotonic() - t0) * 1000.0
            _emit_audit(
                "autofocus",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_min_mm=z_min_mm, z_max_mm=z_max_mm,
                    n_steps=n_steps, metric=chosen_metric.name,
                    algorithm=algo,
                ),
                result_summary={
                    "best_z_mm": best_z_m * 1e3,
                    "scanned": evaluations,
                    "runtime_ms": runtime_ms,
                },
            )
            return AutofocusResult(
                best_z_m=best_z_m,
                score=best_score,
                scanned=evaluations,
                runtime_ms=runtime_ms,
            )
        self._submit(run, on_result, on_error)

    # ---- Multi-focus candidates ---------------------------------------

    def find_focus_candidates(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_min_mm: float, z_max_mm: float, n_steps: int = 60,
        metric: Optional[FocusMetric] = None,
        min_prominence: float = 0.05,
        sample_id: str = "",
        on_result: Callable[[MultiFocusResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        chosen = metric or _metric_from_params(params)
        cancel = self._cancel_check

        def run() -> MultiFocusResult:
            import time
            t0 = time.monotonic()
            field, base, method = _prepare_field(
                hologram_path, params, apply_af_roi=True,
            )
            cands = find_focus_candidates(
                field, base, method,
                z_min_m=z_min_mm * 1e-3, z_max_m=z_max_mm * 1e-3,
                n_steps=n_steps, metric=chosen,
                min_prominence=min_prominence,
                cancel_check=cancel,
            )
            runtime_ms = (time.monotonic() - t0) * 1000.0
            _emit_audit(
                "multi_focus",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_min_mm=z_min_mm, z_max_mm=z_max_mm, n_steps=n_steps,
                    metric=chosen.name, min_prominence=min_prominence,
                ),
                result_summary={
                    "candidates_mm": [c.z_m * 1e3 for c in cands],
                    "runtime_ms": runtime_ms,
                },
            )
            return MultiFocusResult(
                candidates=list(cands),
                runtime_ms=runtime_ms,
            )
        self._submit(run, on_result, on_error)

    # ---- One-shot QPI --------------------------------------------------

    def run_qpi(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_mm: float,
        n_sample: Optional[float] = None,
        n_medium: Optional[float] = None,
        sample_id: str = "",
        on_result: Callable[[QPIOneShotResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        # Honour explicit args when caller passes them (tests), else
        # pick up whatever ReconParams carries — v2 sidebar writes
        # user-edited values there in v2.0.3.
        ns = float(n_sample) if n_sample is not None else float(params.n_sample)
        nm = float(n_medium) if n_medium is not None else float(params.n_medium)

        def run() -> QPIOneShotResult:
            import time
            t0 = time.monotonic()
            field, base, method = _prepare_field(hologram_path, params)
            p = ReconstructionParams(
                wavelength_m=base.wavelength_m,
                pixel_size_m=base.pixel_size_m,
                z_m=z_mm * 1e-3, n=base.n,
            )
            recon = propagate(field, p, method)
            wrapped = np.angle(recon).astype(np.float32, copy=False)
            # v2.0.5: honour the user-selected unwrap method. Falls
            # back to gradient integration if the user typed a value
            # the enum doesn't recognise (e.g. state file from a newer
            # version whose method we don't implement yet).
            method_name = str(params.unwrap_method or "").upper()
            try:
                unwrap_cfg = UnwrapConfig(method=UnwrapMethod[method_name])
            except KeyError:
                unwrap_cfg = UnwrapConfig(
                    method=UnwrapMethod.GRADIENT_INTEGRATION,
                )
            try:
                unwrapped = unwrap_phase_advanced(
                    wrapped,
                    unwrap_cfg,
                    complex_field=recon,
                    wavelength_m=base.wavelength_m,
                    pixel_size_m=base.pixel_size_m,
                )
            except Exception:
                _LOG.warning("qpi: unwrap fallback to wrapped", exc_info=True)
                unwrapped = wrapped
            # v2.0.7: reflection DHM sees the OPL twice (light
            # traverses the surface, reflects, traverses again).
            # Halve the unwrapped phase *before* compute_qpi so every
            # downstream derived quantity (OPD, height, dry mass,
            # step height) reflects the real single-pass value —
            # this is the "3 µm object measured as 6 µm" fix.
            if str(getattr(params, "optical_mode",
                           "transmission")).lower() == "reflection":
                unwrapped = unwrapped * 0.5
            qpi = compute_qpi(
                unwrapped,
                wavelength_m=base.wavelength_m,
                pixel_size_m=base.pixel_size_m,
                mode=QPIMode.BOTH,
                n_sample=ns, n_medium=nm,
                compute_psd=False,
            )
            runtime_ms = (time.monotonic() - t0) * 1000.0
            opd_nm = None
            if qpi.phase_stats is not None:
                opd_nm = qpi.phase_stats.range_nm
            _emit_audit(
                "qpi",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_mm=z_mm, n_sample=ns, n_medium=nm,
                ),
                result_summary={
                    "opd_range_nm": opd_nm,
                    "total_dry_mass_pg": qpi.total_dry_mass_pg,
                    "step_height_m": qpi.step_height_m,
                    "runtime_ms": runtime_ms,
                },
            )
            return QPIOneShotResult(
                qpi=qpi,
                runtime_ms=runtime_ms,
            )
        self._submit(run, on_result, on_error)

    # ---- QPI batch (one entry per focus candidate) --------------------

    def run_qpi_batch(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_min_mm: float, z_max_mm: float, n_steps: int = 60,
        n_sample: Optional[float] = None, n_medium: Optional[float] = None,
        sample_id: str = "",
        on_result: Callable[[QPIBatchResultWrap], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        ns = float(n_sample) if n_sample is not None else float(params.n_sample)
        nm = float(n_medium) if n_medium is not None else float(params.n_medium)
        af_metric = _metric_from_params(params)
        cancel = self._cancel_check

        def run() -> QPIBatchResultWrap:
            import time
            t0 = time.monotonic()
            field, base, method = _prepare_field(hologram_path, params)
            cands = find_focus_candidates(
                field, base, method,
                z_min_m=z_min_mm * 1e-3, z_max_m=z_max_mm * 1e-3,
                n_steps=n_steps, metric=af_metric,
                min_prominence=0.05,
                cancel_check=cancel,
            )
            entries = run_qpi_for_candidates(
                field, base, method, cands,
                n_sample=ns, n_medium=nm, compute_psd=False,
            )
            runtime_ms = (time.monotonic() - t0) * 1000.0
            _emit_audit(
                "qpi_batch",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_min_mm=z_min_mm, z_max_mm=z_max_mm, n_steps=n_steps,
                    n_sample=ns, n_medium=nm,
                ),
                result_summary={
                    "candidates": len(entries),
                    "runtime_ms": runtime_ms,
                },
            )
            return QPIBatchResultWrap(
                entries=list(entries),
                runtime_ms=runtime_ms,
            )
        self._submit(run, on_result, on_error)

    # ---- Depth map + clusters -----------------------------------------

    def run_depth_map(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_min_mm: float, z_max_mm: float, n_steps: int = 40,
        window_size: int = 5,
        metric: Optional[FocusMetric] = None,
        sample_id: str = "",
        on_result: Callable[[DepthMapResultWrap], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        chosen = metric or _metric_from_params(params)
        cancel = self._cancel_check

        def run() -> DepthMapResultWrap:
            import time
            t0 = time.monotonic()
            field, base, method = _prepare_field(
                hologram_path, params, apply_af_roi=True,
            )
            result = compute_depth_map(
                field, base, method,
                z_min_m=z_min_mm * 1e-3, z_max_m=z_max_mm * 1e-3,
                n_steps=n_steps, window_size=window_size, metric=chosen,
                cancel_check=cancel,
            )
            try:
                clusters = segment_depth_clusters(
                    result,
                    confidence_threshold_frac=0.30,
                    min_area_px=40,
                )
            except Exception:
                clusters = []
            runtime_ms = (time.monotonic() - t0) * 1000.0
            _emit_audit(
                "depth_map",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_min_mm=z_min_mm, z_max_mm=z_max_mm, n_steps=n_steps,
                    window_size=window_size, metric=chosen.name,
                ),
                result_summary={
                    "z_min_mm_detected": float(result.z_map.min()) * 1e3,
                    "z_max_mm_detected": float(result.z_map.max()) * 1e3,
                    "clusters": len(clusters),
                    "runtime_ms": runtime_ms,
                },
            )
            return DepthMapResultWrap(
                result=result,
                clusters=list(clusters),
                runtime_ms=runtime_ms,
            )
        self._submit(run, on_result, on_error)

    # ---- Report export ------------------------------------------------

    def export_report(
        self,
        out_path: Path,
        *,
        last_recon_params: dict,
        phase_image: Optional[np.ndarray],
        amplitude_image: Optional[np.ndarray],
        sample_id: str = "",
        on_result: Callable[[ReportExportResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        def run() -> ReportExportResult:
            import time
            t0 = time.monotonic()
            path = generate_html_report(
                output_path=out_path,
                title="DHM Reconstruction Report",
                operator=sample_id or None,
                recon_params=last_recon_params,
                autofocus_summary=None,
                qpi_result=None,
                phase_image=phase_image,
                amplitude_image=amplitude_image,
                height_image=None,
                notes="",
            )
            return ReportExportResult(
                path=Path(path),
                runtime_ms=(time.monotonic() - t0) * 1000.0,
            )
        self._submit(run, on_result, on_error)

    # ---- Tomography bundle export -------------------------------------

    def export_bundle(
        self,
        out_dir: Path,
        *,
        depth_result: DepthMapResult,
        clusters: Sequence,
        qpi_entries: Sequence,
        sample_id: str = "",
        pixel_size_m: Optional[float] = None,
        on_result: Callable[[BundleExportResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        from datetime import datetime

        def run() -> BundleExportResult:
            import time
            t0 = time.monotonic()
            base_name = f"tomography_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            paths = write_tomography_bundle(
                out_dir, depth_result, clusters, qpi_entries,
                base_name=base_name,
                sample_id=sample_id,
                pixel_size_m=pixel_size_m,
            )
            return BundleExportResult(
                files=list(paths),
                runtime_ms=(time.monotonic() - t0) * 1000.0,
            )
        self._submit(run, on_result, on_error)


__all__ = [
    "ScienceDriver",
    "AutofocusResult",
    "MultiFocusResult",
    "QPIOneShotResult",
    "QPIBatchResultWrap",
    "DepthMapResultWrap",
    "ReportExportResult",
    "BundleExportResult",
    "write_qpi_csv",
    "write_qpi_batch_csv",
    "available_focus_metrics",
    "available_autofocus_algorithms",
    "af_algorithm_input_profile",
]
