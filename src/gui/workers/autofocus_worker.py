"""Background worker for autofocus — prevents UI freezing."""
import time
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.reconstruction import ReconstructionParams, ReconstructionMethod
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
)
from gui.sidebar.focus_tab import FocusTab


class AutofocusWorker(QThread):
    """Runs any autofocus algorithm off the main thread."""

    # Emits (best_z_m, z_array, score_array, metric_value, elapsed_s, evals)
    finished = Signal(float, object, object, str, float, int)
    # Emits metric enum value string when auto-select picks one
    metric_selected = Signal(str)
    error = Signal(str)
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
        )

    def run(self):
        job = self._job
        if job is None:
            self.error.emit("No autofocus job configured")
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

            # Auto-downsample ×2 for faster autofocus (~4× speedup)
            ds_factor = 2
            fc, params = downsample_complex_field(fc_orig, params_orig, factor=ds_factor)

            # Scale ROI bounds for downsampled field
            rb = job["roi_bounds"]
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
                ad_budget = max(30, int(steps * 0.5))
                ad_res = adaptive_distance_search(
                    field=fc, base_params=params, method=method, metric=metric,
                    initial_range_m=job["ad_initial_range_m"],
                    max_range_m=job["ad_max_range_m"],
                    expand_factor=job["ad_expand_factor"],
                    signal_threshold=job["ad_signal_threshold"],
                    max_evaluations=ad_budget,
                    cancel_check=cc, on_progress=prog,
                    roi_bounds=rb,
                )
                # Narrow the search range to detected active zone
                zmin, zmax = ad_res.detected_range_m
                ad_z_hist = ad_res.z_history
                ad_f_hist = ad_res.score_history
                steps = max(steps - ad_res.evaluations, 10)
                self.progress.emit(
                    f"Range found: {zmin*1e3:.3f} \u2013 {zmax*1e3:.3f} mm \u2192 refining with {algo}..."
                )
                # Recalculate step_init for adaptive algorithms within new range
                if step_init is None or job["use_adaptive_distance"]:
                    step_init = (zmax - zmin) / 20.0

            if algo == FocusTab.ALGO_ROBUST:
                self.progress.emit("Running Robust Coarse-to-Fine...")
                res = robust_coarse_to_fine_search(
                    field=fc, base_params=params, method=method, metric=metric,
                    z_min_m=zmin, z_max_m=zmax,
                    n_coarse=steps, refine_factor=8, smooth_sigma=1.5,
                    on_progress=prog, cancel_check=cc,
                    roi_bounds=rb,
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
                    roi_bounds=rb,
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
                    roi_bounds=rb,
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
                    roi_bounds=rb,
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
                    roi_bounds=rb,
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
                    roi_bounds=rb,
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
        except Exception as e:
            import traceback
            traceback.print_exc()
            if not self._is_cancelled():
                self.error.emit(str(e))
            else:
                self.cancelled.emit()
