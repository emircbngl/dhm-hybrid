"""FocusPanel — autofocus feature panel (ui3).

Mirrors ``ui2``'s focus/autofocus sidebar section (``src/ui2/app.py``
``_on_af_algorithm_changed`` / ``_apply_af_algorithm_ux``) and the
compute contract in ``src/ui2/workers.py`` (``ScienceDriver.run_autofocus``
/ ``find_focus_candidates``) — same fields on ``ReconParams``, same
six search algorithms, same "adaptive hides z-range" UX rule. Only the
presentation is new: Qt widgets instead of Dear PyGui, wired through
``PanelContext`` instead of direct dpg item tags.

Controls:

* Base algorithm combo — ``ReconParams.af_algorithm``. Options come
  straight from ``ui2.workers.available_autofocus_algorithms()`` so
  the two UIs can never drift on what's supported.
* Focus metric combo — ``ReconParams.autofocus_metric``, options from
  ``core.autofocus.FocusMetric`` (default ``PHASE_VARIANCE`` per the
  panel spec — v1/v2 default to ``LAPLACIAN_VARIANCE``, but bio
  samples with weak amplitude contrast want a phase-based metric, so
  this panel's *initial selection* differs from the field default
  while still writing whatever the user picks back onto the same
  ``autofocus_metric`` field).
* z-min / z-max (mm) — ``af_z_min_mm`` / ``af_z_max_mm``.
* n-steps — ``af_n_steps``. Relabelled per-algorithm exactly like
  ``af_algorithm_input_profile`` describes (grid "steps" vs adaptive
  "max evals"), and the z-range spinboxes are disabled for
  ``adaptive_distance`` (it auto-discovers its own range — see
  ``ScienceDriver.run_autofocus``).
* "Adaptive" sub-section — visible only when an adaptive-family
  algorithm is selected. The adaptive algorithms' internal knobs
  (initial range / expand factor / threshold) stay hardcoded in
  ``ScienceDriver.run_autofocus`` — SETTLED 2026-07-06: the 9-scene
  real-lab benchmark (scripts/benchmark_af_real.py, results in
  docs/AUTOFOCUS_ADAPTIVE.md) showed the auto-derived internals are
  not the accuracy bottleneck (algorithm choice + metric pairing is),
  so no new ``ReconParams`` fields are warranted. The benchmark also
  pinned the default ``af_algorithm`` to "robust" and stamped the
  per-algorithm verdicts into ``af_algorithm_input_profile``'s tips,
  which this panel surfaces as read-only explanatory copy.

Buttons:

* "Autofocus" → ``ctx.bridge.autofocus``; on ``autofocus_done`` writes
  ``best_z_mm`` onto ``ctx.get_params().z_mm``, calls
  ``ctx.params_changed()`` and reports via ``ctx.set_status``.
* "Find focus candidates" → ``ctx.bridge.find_focus_candidates``; on
  ``focus_candidates_done`` populates a ``QListWidget`` (one row per
  candidate: z_mm + score). Double-click jumps ``z_mm`` to that
  candidate and triggers a reconstruction at the new z.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.autofocus import FocusMetric
from core.drivers.workers import (
    af_algorithm_input_profile,
    available_autofocus_algorithms,
)
from ui3.context import PanelContext
from ui3.design import Space

_LOG = logging.getLogger(__name__)

# Algorithm names (from core.drivers.workers._AUTOFOCUS_ALGORITHMS) whose search
# adapts its own step size rather than walking a fixed grid. Used only
# to decide whether the "Adaptive" sub-section is shown — the actual
# dispatch lives entirely in ScienceDriver.run_autofocus.
_ADAPTIVE_ALGORITHMS = {
    "adaptive_gradient", "adaptive_bracketing", "adaptive_distance",
}

# Default focus metric for this panel — PHASE_VARIANCE per the ui3
# focus_panel spec (ReconParams itself still defaults to
# LAPLACIAN_VARIANCE for back-compat with existing state files; this
# only governs what the combo shows on a fresh ReconParams instance
# where autofocus_metric wasn't already overridden by loaded state).
_PANEL_DEFAULT_METRIC = FocusMetric.PHASE_VARIANCE.name


def _fmt_z(z_mm: float) -> str:
    return f"{z_mm:+.4f} mm"


class FocusPanel(QWidget):
    """Autofocus controls: single best-z search + multi-candidate scan."""

    # Emits the full candidate list every time a scan completes, so a
    # host window can also show it elsewhere (e.g. overlay on a depth
    # panel) without re-querying the bridge.
    candidates_found = Signal(list)

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._candidates: List[Any] = []
        self._build_ui()
        self._wire_bridge()
        self._sync_from_params()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.md)
        root.setSpacing(Space.md)

        heading = QLabel("Autofocus")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        # -- Search parameters card -------------------------------------
        card = QFrame(self)
        card.setProperty("role", "card")
        form = QFormLayout(card)

        self.cmb_algorithm = QComboBox(card)
        self.cmb_algorithm.addItems(available_autofocus_algorithms())
        self.cmb_algorithm.currentTextChanged.connect(self._on_algorithm_changed)
        form.addRow("Algorithm", self.cmb_algorithm)

        self.cmb_metric = QComboBox(card)
        self.cmb_metric.addItems([m.name for m in FocusMetric])
        self.cmb_metric.currentTextChanged.connect(self._on_metric_changed)
        form.addRow("Focus metric", self.cmb_metric)

        self.spin_z_min = QDoubleSpinBox(card)
        self.spin_z_min.setRange(-1000.0, 1000.0)
        self.spin_z_min.setDecimals(4)
        self.spin_z_min.setSingleStep(0.1)
        self.spin_z_min.setSuffix(" mm")
        self.spin_z_min.valueChanged.connect(self._on_z_min_changed)
        self.lbl_z_min = QLabel("z min")
        form.addRow(self.lbl_z_min, self.spin_z_min)

        self.spin_z_max = QDoubleSpinBox(card)
        self.spin_z_max.setRange(-1000.0, 1000.0)
        self.spin_z_max.setDecimals(4)
        self.spin_z_max.setSingleStep(0.1)
        self.spin_z_max.setSuffix(" mm")
        self.spin_z_max.valueChanged.connect(self._on_z_max_changed)
        self.lbl_z_max = QLabel("z max")
        form.addRow(self.lbl_z_max, self.spin_z_max)

        self.spin_n_steps = QSpinBox(card)
        self.spin_n_steps.setRange(2, 5000)
        self.spin_n_steps.valueChanged.connect(self._on_n_steps_changed)
        self.lbl_n_steps = QLabel("steps")
        form.addRow(self.lbl_n_steps, self.spin_n_steps)

        root.addWidget(card)

        # -- Adaptive sub-section (visible only for adaptive algos) -----
        self.grp_adaptive = QGroupBox("Adaptive", self)
        adaptive_lay = QVBoxLayout(self.grp_adaptive)
        self.lbl_adaptive_info = QLabel(self.grp_adaptive)
        self.lbl_adaptive_info.setProperty("role", "muted")
        self.lbl_adaptive_info.setWordWrap(True)
        adaptive_lay.addWidget(self.lbl_adaptive_info)
        root.addWidget(self.grp_adaptive)
        self.grp_adaptive.setVisible(False)

        # -- Action buttons -----------------------------------------------
        btn_row = QHBoxLayout()
        self.btn_autofocus = QPushButton("Autofocus", self)
        self.btn_autofocus.setProperty("role", "primary")
        self.btn_autofocus.clicked.connect(self._on_autofocus_clicked)
        btn_row.addWidget(self.btn_autofocus)

        self.btn_find_candidates = QPushButton("Find focus candidates", self)
        self.btn_find_candidates.clicked.connect(self._on_find_candidates_clicked)
        btn_row.addWidget(self.btn_find_candidates)
        root.addLayout(btn_row)

        # -- Result readout -------------------------------------------------
        self.lbl_result = QLabel("", self)
        self.lbl_result.setProperty("role", "readout")
        self.lbl_result.setWordWrap(True)
        root.addWidget(self.lbl_result)

        # -- Candidate list ---------------------------------------------
        cand_label = QLabel("Focus candidates", self)
        cand_label.setProperty("role", "muted")
        root.addWidget(cand_label)

        self.list_candidates = QListWidget(self)
        self.list_candidates.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_candidates.itemDoubleClicked.connect(
            self._on_candidate_double_clicked)
        root.addWidget(self.list_candidates, stretch=1)

    # ------------------------------------------------------------------
    # Param sync (panel <-> ReconParams)
    # ------------------------------------------------------------------
    def _sync_from_params(self) -> None:
        params = self._ctx.get_params()
        algo = str(getattr(params, "af_algorithm", "robust") or "robust")
        idx = self.cmb_algorithm.findText(algo)
        self.cmb_algorithm.blockSignals(True)
        self.cmb_algorithm.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_algorithm.blockSignals(False)

        metric = str(getattr(params, "autofocus_metric", "") or "")
        if not metric:
            metric = _PANEL_DEFAULT_METRIC
        midx = self.cmb_metric.findText(metric)
        self.cmb_metric.blockSignals(True)
        self.cmb_metric.setCurrentIndex(
            midx if midx >= 0 else self.cmb_metric.findText(_PANEL_DEFAULT_METRIC))
        self.cmb_metric.blockSignals(False)
        # Panel default differs from the ReconParams dataclass default
        # (LAPLACIAN_VARIANCE) — write it back only if the field was
        # genuinely unset, so we never clobber a value loaded from state.
        if not getattr(params, "autofocus_metric", ""):
            params.autofocus_metric = metric

        self.spin_z_min.blockSignals(True)
        self.spin_z_min.setValue(float(getattr(params, "af_z_min_mm", -25.0)))
        self.spin_z_min.blockSignals(False)

        self.spin_z_max.blockSignals(True)
        self.spin_z_max.setValue(float(getattr(params, "af_z_max_mm", 25.0)))
        self.spin_z_max.blockSignals(False)

        self.spin_n_steps.blockSignals(True)
        self.spin_n_steps.setValue(int(getattr(params, "af_n_steps", 40)))
        self.spin_n_steps.blockSignals(False)

        self._apply_algorithm_ux(algo)

    def _on_algorithm_changed(self, text: str) -> None:
        params = self._ctx.get_params()
        params.af_algorithm = text
        self._ctx.params_changed()
        self._apply_algorithm_ux(text)

    def _on_metric_changed(self, text: str) -> None:
        params = self._ctx.get_params()
        params.autofocus_metric = text
        self._ctx.params_changed()

    def _on_z_min_changed(self, value: float) -> None:
        params = self._ctx.get_params()
        params.af_z_min_mm = float(value)
        self._ctx.params_changed()

    def _on_z_max_changed(self, value: float) -> None:
        params = self._ctx.get_params()
        params.af_z_max_mm = float(value)
        self._ctx.params_changed()

    def _on_n_steps_changed(self, value: int) -> None:
        params = self._ctx.get_params()
        params.af_n_steps = int(value)
        self._ctx.params_changed()

    def _apply_algorithm_ux(self, algo: str) -> None:
        """Relabel/enable widgets per algorithm — mirrors ui2's
        ``_apply_af_algorithm_ux`` (``af_algorithm_input_profile``),
        and shows/hides the Adaptive sub-section."""
        profile = af_algorithm_input_profile(algo)
        steps_label = profile.get("steps_label", "steps")
        steps_tip = profile.get("steps_tip", "")
        uses_range = bool(profile.get("uses_z_range", True))

        self.lbl_n_steps.setText(steps_label)
        self.spin_n_steps.setToolTip(steps_tip)

        suffix = "" if uses_range else "  (auto)"
        self.lbl_z_min.setText("z min" + suffix)
        self.lbl_z_max.setText("z max" + suffix)
        self.spin_z_min.setEnabled(uses_range)
        self.spin_z_max.setEnabled(uses_range)

        is_adaptive = str(algo or "").lower() in _ADAPTIVE_ALGORITHMS
        self.grp_adaptive.setVisible(is_adaptive)
        if is_adaptive:
            self.lbl_adaptive_info.setText(self._adaptive_info_text(algo))

    def _adaptive_info_text(self, algo: str) -> str:
        """Explanatory copy for the adaptive sub-section.

        No dedicated ``ReconParams`` fields exist yet for the adaptive
        algorithms' internal knobs (initial range, expand factor,
        signal threshold — see ``ScienceDriver.run_autofocus`` in
        ``src/ui2/workers.py``, which hardcodes them). Rather than add
        UI controls that would silently do nothing, this panel names
        the actual behaviour tied to the fields that DO exist
        (``af_algorithm``, ``af_n_steps``)."""
        algo = str(algo or "").lower()
        if algo == "adaptive_gradient":
            return (
                "Gradient-driven adaptive step size: large steps in "
                "flat regions, small steps near the peak. \"steps\" "
                "above is a max-evaluations budget, not a fixed grid "
                "count."
            )
        if algo == "adaptive_bracketing":
            return (
                "Nested bracketing (3 levels x 8 divisions) + a final "
                "refine pass. \"steps\" above caps total evaluations "
                "across all levels."
            )
        if algo == "adaptive_distance":
            return (
                "Auto-discovers its own z-range: starts near z=0 and "
                "expands until the focus peak is interior, ignoring "
                "the z min/max fields above (kept only as a ceiling "
                "hint). \"steps\" caps total evaluations."
            )
        return ""

    # ------------------------------------------------------------------
    # Bridge wiring
    # ------------------------------------------------------------------
    def _wire_bridge(self) -> None:
        bridge = self._ctx.bridge
        bridge.autofocus_done.connect(self._on_autofocus_done)
        bridge.autofocus_error.connect(self._on_autofocus_error)
        bridge.focus_candidates_done.connect(self._on_candidates_done)
        bridge.focus_candidates_error.connect(self._on_candidates_error)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_autofocus_clicked(self) -> None:
        path = self._ctx.hologram_path()
        if path is None:
            self._ctx.set_status("Load a hologram before autofocusing.", "warn")
            return
        params = self._ctx.get_params()
        self._ctx.set_status("Autofocusing…", "info")
        self._ctx.bridge.autofocus(
            path, params,
            z_min_mm=float(params.af_z_min_mm),
            z_max_mm=float(params.af_z_max_mm),
            n_steps=int(params.af_n_steps),
        )

    def _on_autofocus_done(self, result: Any) -> None:
        best_z_mm = float(result.best_z_m) * 1e3
        params = self._ctx.get_params()
        params.z_mm = best_z_mm
        self._ctx.params_changed()
        self.lbl_result.setText(
            f"Best z = {_fmt_z(best_z_mm)}  score={result.score:.4g}  "
            f"({result.scanned} evals, {result.runtime_ms:.0f} ms)")
        # A degenerate/flat focus landscape means best_z is a guess, not a
        # real focus — surface the diagnostic instead of a confident "ok"
        # (2026-07-08 version audit, B-098).
        warning = getattr(result, "warning", None)
        if warning:
            self.lbl_result.setText(self.lbl_result.text() + "  ⚠")
            self._ctx.set_status(warning, "warn")
            self._ctx.toast("Autofocus: focus landscape unreliable", "warn")
        else:
            self._ctx.set_status(
                f"Autofocus done: z={_fmt_z(best_z_mm)}", "ok")
            self._ctx.toast(f"Autofocus: z={_fmt_z(best_z_mm)}", "ok")

    def _on_autofocus_error(self, message: str) -> None:
        self.lbl_result.setText("")
        self._ctx.set_status(f"Autofocus failed: {message}", "error")
        self._ctx.toast(f"Autofocus failed: {message}", "error")

    def _on_find_candidates_clicked(self) -> None:
        path = self._ctx.hologram_path()
        if path is None:
            self._ctx.set_status(
                "Load a hologram before scanning for focus candidates.", "warn")
            return
        params = self._ctx.get_params()
        self._ctx.set_status("Scanning focus landscape…", "info")
        self._ctx.bridge.find_focus_candidates(
            path, params,
            z_min_mm=float(params.af_z_min_mm),
            z_max_mm=float(params.af_z_max_mm),
            n_steps=max(int(params.af_n_steps), 2),
        )

    def _on_candidates_done(self, result: Any) -> None:
        self._candidates = list(getattr(result, "candidates", []) or [])
        self.list_candidates.clear()
        for cand in self._candidates:
            z_mm = float(cand.z_m) * 1e3
            item = QListWidgetItem(
                f"z={_fmt_z(z_mm)}   score={cand.score:.4g}   "
                f"rank={cand.rank}")
            item.setData(Qt.UserRole, z_mm)
            self.list_candidates.addItem(item)
        # No routine status/toast: the focus-candidates dialog owns the
        # "Found N" status (B-083). But a reference-division fallback (B-110)
        # is exceptional — surface it as a one-off toast so the scan isn't
        # silently unreferenced.
        warn = getattr(result, "warning", None)
        if warn:
            self._ctx.toast(warn, "warn")
        self.candidates_found.emit(self._candidates)

    def _on_candidates_error(self, message: str) -> None:
        self._ctx.set_status(f"Focus scan failed: {message}", "error")
        self._ctx.toast(f"Focus scan failed: {message}", "error")

    def _on_candidate_double_clicked(self, item: QListWidgetItem) -> None:
        z_mm = item.data(Qt.UserRole)
        if z_mm is None:
            return
        z_mm = float(z_mm)
        params = self._ctx.get_params()
        params.z_mm = z_mm
        self._ctx.params_changed()
        path = self._ctx.hologram_path()
        if path is None:
            self._ctx.set_status(
                f"z set to {_fmt_z(z_mm)} (load a hologram to reconstruct).",
                "info")
            return
        self._ctx.set_status(f"Reconstructing at z={_fmt_z(z_mm)}…", "info")
        self._ctx.bridge.reconstruct(path, params)


__all__ = ["FocusPanel"]
