"""Background worker for autofocus — prevents UI freezing.

v1.0.1: emits :class:`core.errors.ErrorEvent` on failure alongside the legacy
``error`` string signal, so the UI can bind to one structured error protocol
across reconstruction / autofocus / QPI workers. The algorithm already has
cooperative cancellation and progress reporting, so no deeper refactor is
needed here.
"""
import logging
import time
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.errors import ErrorEvent, Severity
from core.reconstruction import ReconstructionParams, ReconstructionMethod, propagate
from core.autofocus import (
    FocusMetric,
    AutofocusCancelled,
    autofocus_zscan,
    coarse_to_fine_search,
    robust_coarse_to_fine_search,
    auto_select_metric,
    adaptive_gradient_search,
    adaptive_ratio_search,
    adaptive_bracketing_search,
    adaptive_distance_search,
    downsample_complex_field,
    has_sufficient_contrast,
)
from gui.sidebar.focus_tab import FocusTab

_LOG = logging.getLogger(__name__)


class AutofocusWorker(QThread):
    """Runs any autofocus algorithm off the main thread."""

    # Emits (best_z_m, z_array, score_array, metric_value, elapsed_s, evals)
    finished = Signal(float, object, object, str, float, int)
    # Emits metric enum value string when auto-select picks one
    metric_selected = Signal(str)
    # Emits (from_metric, to_metric) when ENTROPY is auto-fallen back due to low contrast
    metric_auto_fallback = Signal(str, str)
    error = Signal(str)
    # Structured error (ErrorEvent) — new in v1.0.1. Emitted alongside `error`.
    error_event = Signal(object)
    cancelled = Signal()
    progress = Signal(str)
    # Emits (percent 0-100, elapsed_s, eta_s)  — eta_s = -1 if unknown
    progress_pct = Signal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job = None
        self._t0 = 0.0

    def cancel(self):
        """Request cancellation — checked each iteration by the algorithm."""
        self.requestInterruption()

    def _is_cancelled(self) -> bool:
        return self.isInterruptionRequested()

    def _on_progress(self, current: int, total: int) -> None:
        """Called from inside algorithm loops on the worker thread."""
        if total <= 0:
            return
        pct = min(int(current * 100 / total), 99)
        elapsed = time.time() - self._t0
        if current > 0 and elapsed > 0.3:
            eta = elapsed / current * (total - current)
        else:
            eta = -1.0
        self.progress_pct.emit(pct, elapsed, eta)

    def configure(
        self,
        field: np.ndarray,
        base_params: ReconstructionParams,
        method: ReconstructionMethod,
        metric: FocusMetric,
        algo: str,
        z_min_m: float,
        z_max_m: float,
        steps: int,
        auto_select: bool = False,
        step_init: float | None = None,
        grow_factor: float = 1.5,
        shrink_factor: float = 0.4,
        refine_levels: int = 3,
        refine_divisions: int = 6,
        use_adaptive_distance: bool = False,
        ad_initial_range_m: float = 0.5e-3,
        ad_max_range_m: float = 50e-3,
        ad_expand_factor: float = 2.0,
        ad_signal_threshold: float = 0.3,
        roi_bounds=None,
        ref_field: np.ndarray | None = None,
    ):
        self._job = dict(
            field=field,
            base_params=base_params,
            method=method,
            metric=metric,
            algo=algo,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            steps=steps,
            auto_select=auto_select,
            step_init=step_init,
            grow_factor=grow_factor,
            shrink_factor=shrink_factor,
            refine_levels=refine_levels,
            refine_divisions=refine_divisions,
            use_adaptive_distance=use_adaptive_distance,
            ad_initial_range_m=ad_initial_range_m,
            ad_max_range_m=ad_max_range_m,
            ad_expand_factor=ad_expand_factor,
            ad_signal_threshold=ad_signal_threshold,
            roi_bounds=roi_bounds,
            ref_field=ref_field,
        )

    def run(self):
        job = self._job
        if job is None:
            self._emit_error(
                ValueError("No autofocus job configured"),
                job=None,
                title="Autofocus failed",
                action="Configure z-range and metric, then retry.",
            )
            return

        try:
            fc_orig = job["field"]
            params_orig = job["base_params"]
            method = job["method"]
            metric = job["metric"]
            algo = job["algo"]
            zmin = job["z_min_m"]
            zmax = job["z_max_m"]
            steps = job["steps"]

            # Smart downsampling: skip 2× if ROI is small (preserves small features)
            rb = job["roi_bounds"]
            ds_factor = 2
            if rb is not None:
                roi_w = abs(rb.x1 - rb.x0)
                roi_h = abs(rb.y1 - rb.y0)
                if max(roi_w, roi_h) < 128:
                    ds_factor = 1  # small ROI → keep full resolution

            if ds_factor > 1:
                fc, params = downsample_complex_field(fc_orig, params_orig, factor=ds_factor)
            else:
                fc, params = fc_orig, params_orig

            # Downsample reference field if provided (for phase metrics)
            ref_fc = job.get("ref_field")
            if ref_fc is not None and ds_factor > 1:
                ref_fc, _ = downsample_complex_field(ref_fc, params_orig, factor=ds_factor)
            elif ref_fc is None:
                pass  # no ref field

            # Scale ROI bounds for downsampled field
            if rb is not None and ds_factor > 1:
                from core.phase_tracker import ROIBounds
                rb = ROIBounds(
                    x0=rb.x0 // ds_factor, y0=rb.y0 // ds_factor,
                    x1=rb.x1 // ds_factor, y1=rb.y1 // ds_factor,
                )

            # Auto-select metric
            if job["auto_select"]:
                if self._is_cancelled():
                    self.cancelled.emit(); return
                self.progress.emit("Auto-selecting best metric...")
                metric = auto_select_metric(fc, params, method, zmin, zmax, n_steps=min(steps, 30), cancel_check=self._is_cancelled)
                self.metric_selected.emit(metric.value)
                self.progress.emit(f"Selected metric: {metric.value}")

            # ENTROPY low-contrast pre-flight check — reconstruct at mid-z and
            # verify phase has enough structure. If not, fall back to PHASE_VARIANCE.
            # ENTROPY collapses into a single-bin histogram on flat/saturated holograms
            # and the search converges to a random z.
            if metric == FocusMetric.ENTROPY:
                z_mid = 0.5 * (zmin + zmax)
                mid_params = ReconstructionParams(
                    wavelength_m=params.wavelength_m,
                    pixel_size_m=params.pixel_size_m,
                    z_m=z_mid,
                    n=params.n,
                )
                field_mid = propagate(fc, mid_params, method)
                if not has_sufficient_contrast(field_mid):
                    # Re-compute circular variance locally for the log message
                    phase_mid = np.angle(field_mid) if np.iscomplexobj(field_mid) else field_mid
                    mu_x = float(np.mean(np.cos(phase_mid)))
                    mu_y = float(np.mean(np.sin(phase_mid)))
                    measured_circ_var = float(1.0 - np.sqrt(mu_x * mu_x + mu_y * mu_y))
                    logging.getLogger(__name__).warning(
                        "ENTROPY low-contrast fallback → PHASE_VARIANCE "
                        "(measured circular variance = %.4f, threshold = 0.15)",
                        measured_circ_var,
                    )
                    self.metric_auto_fallback.emit(
                        FocusMetric.ENTROPY.value, FocusMetric.PHASE_VARIANCE.value
                    )
                    try:
                        from core.audit import get_audit_log
                        get_audit_log().record(
                            action="autofocus_metric_fallback",
                            params={
                                "from": "entropy",
                                "to": "phase_variance",
                                "reason": "low_contrast",
                                "measured_circ_var": measured_circ_var,
                            },
                        )
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "audit record failed", exc_info=True
                        )
                    metric = FocusMetric.PHASE_VARIANCE

            self._t0 = time.time()
            step_init = job["step_init"]
            grow = job["grow_factor"]
            shrink = job["shrink_factor"]
            cc = self._is_cancelled
            prog = self._on_progress

            # ── Phase 0: Adaptive Distance as range finder ──
            ad_z_hist = np.array([])
            ad_f_hist = np.array([])
            if job["use_adaptive_distance"]:
                self.progress.emit("Adaptive Distance: discovering signal range...")
                # Budget must cover log2(max_range/init_range) expansion iterations
                # at 15 pts per iter, plus margin. With init=0.5mm, max=50mm the
                # expansion needs ~7 doublings; 30 evals (old default) stalled at ±1mm.
                import math
                expand_depth = max(1, math.ceil(math.log2(
                    job["ad_max_range_m"] / max(job["ad_initial_range_m"], 1e-15)
                )))
                ad_budget = max(60, expand_depth * 15 + 15, int(steps * 0.5))
                ad_res = adaptive_distance_search(
                    field=fc, base_params=params, method=method, metric=metric,
                    initial_range_m=job["ad_initial_range_m"],
                    max_range_m=job["ad_max_range_m"],
                    expand_factor=job["ad_expand_factor"],
                    signal_threshold=job["ad_signal_threshold"],
                    max_evaluations=ad_budget,
                    cancel_check=cc, on_progress=prog,
                    roi_bounds=rb, ref_field=ref_fc,
                )
                # Narrow the search range to detected active zone
                zmin, zmax = ad_res.detected_range_m
                ad_z_hist = ad_res.z_history
                ad_f_hist = ad_res.score_history
                # Keep a healthy refinement budget — don't subtract AdDist cost,
                # user's `steps` is meant for refinement density, not a global cap.
                steps = max(steps, 20)
                self.progress.emit(
                    f"Range found: {zmin*1e3:.3f} \u2013 {zmax*1e3:.3f} mm \u2192 refining with {algo}..."
                )
                # After AdDist narrows the range, let the refinement algorithm pick
                # its own default step_init — passing (range/20) forced tiny steps
                # that couldn't traverse a wide detected range in the forward scan.
                if step_init is not None and not job["use_adaptive_distance"]:
                    pass  # keep user-supplied step_init
                else:
                    step_init = None  # let algorithm choose (full_range/10 for AdGrad)

            if algo == FocusTab.ALGO_ROBUST:
                self.progress.emit("Running Robust Coarse-to-Fine...")
                res = robust_coarse_to_fine_search(
                    field=fc, base_params=params, method=method, metric=metric,
                    z_min_m=zmin, z_max_m=zmax,
                    n_coarse=steps, refine_factor=8, smooth_sigma=1.5,
                    on_progress=prog, cancel_check=cc,
                    roi_bounds=rb, ref_field=ref_fc,
                )
                best_z = res.best_z_m
                z_m_arr = res.coarse_z if res.coarse_z is not None else np.array([best_z])
                scores = res.coarse_scores if res.coarse_scores is not None else np.array([res.best_score])

            elif algo == FocusTab.ALGO_GOLDEN:
                self.progress.emit("Running Golden Section...")
                res = coarse_to_fine_search(
                    field=fc, base_params=params, method=method, metric=metric,
                    z_min_m=zmin, z_max_m=zmax,
                    coarse_steps=steps, fine_tolerance_m=1e-6,
                    cancel_check=cc, on_progress=prog,
                    roi_bounds=rb, ref_field=ref_fc,
                )
                best_z = res.best_z_m
                z_m_arr = np.array([best_z])
                scores = np.array([res.best_score])

            elif algo == FocusTab.ALGO_ADAPT_GRADIENT:
                self.progress.emit("Running Adaptive Gradient...")
                res = adaptive_gradient_search(
                    field=fc, base_params=params, method=method, metric=metric,
                    z_min_m=zmin, z_max_m=zmax,
                    step_init=step_init, grow_factor=grow, shrink_factor=shrink,
                    max_evaluations=steps,
                    cancel_check=cc, on_progress=prog,
                    roi_bounds=rb, ref_field=ref_fc,
                )
                best_z = res.best_z_m
                z_m_arr = res.z_history
                scores = res.score_history

            elif algo == FocusTab.ALGO_ADAPT_RATIO:
                self.progress.emit("Running Adaptive Ratio...")
                res = adaptive_ratio_search(
                    field=fc, base_params=params, method=method, metric=metric,
                    z_min_m=zmin, z_max_m=zmax,
                    step_init=step_init, grow_factor=grow, shrink_factor=shrink,
                    max_evaluations=steps,
                    cancel_check=cc, on_progress=prog,
                    roi_bounds=rb, ref_field=ref_fc,
                )
                best_z = res.best_z_m
                z_m_arr = res.z_history
                scores = res.score_history

            elif algo == FocusTab.ALGO_ADAPT_BRACKET:
                self.progress.emit("Running Adaptive Bracketing...")
                res = adaptive_bracketing_search(
                    field=fc, base_params=params, method=method, metric=metric,
                    z_min_m=zmin, z_max_m=zmax,
                    step_init=step_init,
                    n_refine_levels=job["refine_levels"],
                    refine_divisions=job["refine_divisions"],
                    smooth_sigma=1.0,
                    max_evaluations=steps,
                    cancel_check=cc, on_progress=prog,
                    roi_bounds=rb, ref_field=ref_fc,
                )
                best_z = res.best_z_m
                z_m_arr = res.z_history
                scores = res.score_history

            else:  # Linear Sweep
                self.progress.emit("Running Linear Sweep...")
                z_arr = list(np.linspace(zmin, zmax, steps))
                res = autofocus_zscan(
                    field=fc, base_params=params, z_values_m=z_arr,
                    method=method, metric=metric,
                    on_progress=prog, cancel_check=cc,
                    roi_bounds=rb, ref_field=ref_fc,
                )
                best_z = res.best_z_m
                z_m_arr = np.array(list(res.scores.keys()))
                scores = np.array(list(res.scores.values()))

            # Merge adaptive distance history with refinement history
            if len(ad_z_hist) > 0:
                z_m_arr = np.concatenate([ad_z_hist, z_m_arr])
                scores = np.concatenate([ad_f_hist, scores])

            elapsed = time.time() - self._t0
            evals = getattr(res, 'evaluations', getattr(res, 'total_evaluations', len(z_m_arr)))
            if len(ad_z_hist) > 0:
                evals += len(ad_z_hist)

            if not self._is_cancelled():
                self.progress_pct.emit(100, elapsed, 0.0)
                self.finished.emit(best_z, z_m_arr, scores, metric.value, elapsed, evals)
            else:
                self.cancelled.emit()

        except AutofocusCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            if self._is_cancelled():
                # User hit Cancel mid-flight; treat as cooperative cancellation
                # rather than a surprise error.
                self.cancelled.emit()
                return
            self._emit_error(e, job=job)

    # ---- helpers -----------------------------------------------------------

    def _emit_error(self, exc: BaseException, *,
                    job: dict | None,
                    title: str = "Autofocus failed",
                    action: str = "") -> None:
        """Log, build an :class:`ErrorEvent`, and fan it out on both signals.

        The structured event carries the z-range / algo / metric context so
        the UI can render a useful one-line copy without re-parsing the
        exception string. The legacy ``error(str)`` signal is retained for
        v1.0.0 call-sites that still listen for it.
        """
        _LOG.exception("[AF] worker failed")
        context: dict = {}
        if job is not None:
            metric_val = job.get("metric")
            algo_val = job.get("algo", "")
            context = {
                "z_min_mm": float(job.get("z_min_m", 0.0)) * 1e3,
                "z_max_mm": float(job.get("z_max_m", 0.0)) * 1e3,
                "steps": int(job.get("steps", 0)),
                "metric": getattr(metric_val, "value", str(metric_val or "")),
                "algo": str(algo_val),
            }
        event = ErrorEvent.from_exception(
            exc,
            title=title,
            action=action or _suggest_action(exc, job or {}),
            context=context,
            severity=Severity.ERROR,
        )
        self.error_event.emit(event)
        self.error.emit(f"{type(exc).__name__}: {exc}")


def _suggest_action(exc: BaseException, job: dict) -> str:
    """One-sentence, user-correctable hint — Rams #4 (understandable)."""
    msg = str(exc).lower()
    if isinstance(exc, MemoryError) or "memory" in msg:
        return "Reduce step count or narrow the z-range and retry."
    zmin = float(job.get("z_min_m", 0.0) or 0.0)
    zmax = float(job.get("z_max_m", 0.0) or 0.0)
    if zmax <= zmin:
        return "Set z-max greater than z-min in the Focus panel."
    if "wavelength" in msg:
        return "Set a positive wavelength in the Reconstruction panel."
    if "metric" in msg:
        return "Pick a different focus metric or enable Auto-select."
    return ""
