from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QGroupBox,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
)
from PySide6.QtCore import Qt
from core.autofocus import FocusMetric


class _Separator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(2)


class FocusTab(QWidget):
    """Sidebar tab — autofocus configuration (one-shot, adaptive step, live tracking)."""

    # Algorithm keys used for dispatch in autofocus_worker
    ALGO_ROBUST = "Robust Coarse-to-Fine"
    ALGO_GOLDEN = "Coarse-to-Fine (Golden Section)"
    ALGO_LINEAR = "Linear Sweep (Exhaustive)"
    ALGO_ADAPT_GRADIENT = "Adaptive Step: Gradient"
    ALGO_ADAPT_RATIO = "Adaptive Step: Ratio"
    ALGO_ADAPT_BRACKET = "Adaptive Step: Bracketing"
    ALGO_ADAPT_DISTANCE = "Adaptive Distance"

    # Sub-mode keys for Adaptive Steps
    _ASTEP_GRADIENT = "Gradient"
    _ASTEP_RATIO = "Ratio"
    _ASTEP_BRACKET = "Bracketing"

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ═══════════════════════════════════════════
        #  BASE AUTOFOCUS
        # ═══════════════════════════════════════════
        focus_group = QGroupBox("Autofocus")
        fg = QGridLayout(focus_group)
        fg.setSpacing(6)
        row = 0

        # Base algorithm (non-adaptive)
        self.algo_combo = QComboBox()
        self.algo_combo.setToolTip(
            "Base search algorithm (used when neither adaptive mode is enabled)"
        )
        self.algo_combo.addItem(self.ALGO_ROBUST)
        self.algo_combo.addItem(self.ALGO_GOLDEN)
        self.algo_combo.addItem(self.ALGO_LINEAR)
        fg.addWidget(QLabel("Algorithm"), row, 0)
        fg.addWidget(self.algo_combo, row, 1); row += 1

        # Metric
        self.metric_combo = QComboBox()
        self.metric_combo.setToolTip("Focus quality metric")
        self.metric_combo.addItem("Tenengrad", FocusMetric.TENENGRAD.value)
        self.metric_combo.addItem("Gradient", FocusMetric.GRADIENT.value)
        self.metric_combo.addItem("Brenner", FocusMetric.BRENNER.value)
        self.metric_combo.addItem("Total Variation", FocusMetric.TOTAL_VARIATION.value)
        self.metric_combo.addItem("Laplacian Var.", FocusMetric.LAPLACIAN_VARIANCE.value)
        self.metric_combo.addItem("Entropy", FocusMetric.ENTROPY.value)
        self.metric_combo.addItem("Vollath F4", FocusMetric.VOLLATH_F4.value)
        self.metric_combo.addItem("Vollath F5", FocusMetric.VOLLATH_F5.value)
        self.metric_combo.addItem("Norm. Variance", FocusMetric.NORMALIZED_VARIANCE.value)
        self.metric_combo.addItem("Spectral Energy", FocusMetric.SPECTRAL_ENERGY.value)
        fg.addWidget(QLabel("Metric"), row, 0)
        fg.addWidget(self.metric_combo, row, 1); row += 1

        self.auto_metric_cb = QCheckBox("Auto-select best metric")
        self.auto_metric_cb.setToolTip("Scans all metrics first and picks the most reliable one")
        fg.addWidget(self.auto_metric_cb, row, 0, 1, 2); row += 1

        # Z range
        fg.addWidget(_Separator(), row, 0, 1, 2); row += 1

        self.zscan_min_mm = QDoubleSpinBox()
        self.zscan_min_mm.setRange(-1e6, 1e6)
        self.zscan_min_mm.setDecimals(4)
        self.zscan_min_mm.setValue(-1.0)
        self.zscan_min_mm.setSuffix(" mm")
        fg.addWidget(QLabel("Z min"), row, 0)
        fg.addWidget(self.zscan_min_mm, row, 1); row += 1

        self.zscan_max_mm = QDoubleSpinBox()
        self.zscan_max_mm.setRange(-1e6, 1e6)
        self.zscan_max_mm.setDecimals(4)
        self.zscan_max_mm.setValue(1.0)
        self.zscan_max_mm.setSuffix(" mm")
        fg.addWidget(QLabel("Z max"), row, 0)
        fg.addWidget(self.zscan_max_mm, row, 1); row += 1

        self.zscan_steps = QSpinBox()
        self.zscan_steps.setRange(3, 500_000)
        self.zscan_steps.setValue(51)
        self.zscan_steps.setToolTip("Number of z-positions to evaluate")
        fg.addWidget(QLabel("Steps / Budget"), row, 0)
        fg.addWidget(self.zscan_steps, row, 1); row += 1

        # Action buttons
        fg.addWidget(_Separator(), row, 0, 1, 2); row += 1

        self.autofocus_btn = QPushButton("Auto-Focus")
        self.autofocus_btn.setEnabled(False)
        self.autofocus_btn.setMinimumHeight(32)
        self.autofocus_btn.setToolTip("Run one-shot autofocus with selected algorithm")
        fg.addWidget(self.autofocus_btn, row, 0, 1, 2); row += 1

        btn_row = QHBoxLayout()
        self.benchmark_btn = QPushButton("Benchmark")
        self.benchmark_btn.setEnabled(False)
        self.benchmark_btn.setToolTip("Compare all algorithms x metrics")
        self.diagnostic_btn = QPushButton("Landscape")
        self.diagnostic_btn.setEnabled(False)
        self.diagnostic_btn.setToolTip("Plot focus metric curves across z range")
        btn_row.addWidget(self.benchmark_btn)
        btn_row.addWidget(self.diagnostic_btn)
        fg.addLayout(btn_row, row, 0, 1, 2); row += 1

        layout.addWidget(focus_group)

        # ═══════════════════════════════════════════
        #  ADAPTIVE DISTANCE (toggleable)
        # ═══════════════════════════════════════════
        self.adapt_dist_group = QGroupBox("Adaptive Distance")
        self.adapt_dist_group.setCheckable(True)
        self.adapt_dist_group.setChecked(False)
        self.adapt_dist_group.setToolTip(
            "Automatically determine the active z-range where focus signal is strong.\n"
            "Performs a coarse scan → detects signal zone → narrows the search range.\n"
            "Can be combined with Adaptive Steps for range detection + smart refinement."
        )
        adg = QGridLayout(self.adapt_dist_group)
        adg.setSpacing(5)
        adr = 0

        self.ad_initial_range_mm = QDoubleSpinBox()
        self.ad_initial_range_mm.setRange(0.01, 50.0)
        self.ad_initial_range_mm.setDecimals(2)
        self.ad_initial_range_mm.setValue(0.50)
        self.ad_initial_range_mm.setSuffix(" mm")
        self.ad_initial_range_mm.setToolTip(
            "Starting half-range for distance discovery (±value).\n"
            "Algorithm starts here and expands outward until it finds signal."
        )
        adg.addWidget(QLabel("Initial range ±"), adr, 0)
        adg.addWidget(self.ad_initial_range_mm, adr, 1); adr += 1

        self.ad_max_range_mm = QDoubleSpinBox()
        self.ad_max_range_mm.setRange(1.0, 500.0)
        self.ad_max_range_mm.setDecimals(1)
        self.ad_max_range_mm.setValue(50.0)
        self.ad_max_range_mm.setSuffix(" mm")
        self.ad_max_range_mm.setToolTip(
            "Maximum half-range the algorithm will expand to (safety limit).\n"
            "Search stops expanding beyond ±this value."
        )
        adg.addWidget(QLabel("Max range ±"), adr, 0)
        adg.addWidget(self.ad_max_range_mm, adr, 1); adr += 1

        self.ad_expand_factor = QDoubleSpinBox()
        self.ad_expand_factor.setRange(1.2, 5.0)
        self.ad_expand_factor.setDecimals(1)
        self.ad_expand_factor.setValue(2.0)
        self.ad_expand_factor.setToolTip(
            "Range expansion multiplier when signal is at the edge.\n"
            "Higher → faster discovery but coarser, lower → more precise expansion."
        )
        adg.addWidget(QLabel("Expand factor"), adr, 0)
        adg.addWidget(self.ad_expand_factor, adr, 1); adr += 1

        self.ad_signal_threshold = QDoubleSpinBox()
        self.ad_signal_threshold.setRange(0.05, 0.9)
        self.ad_signal_threshold.setDecimals(2)
        self.ad_signal_threshold.setValue(0.30)
        self.ad_signal_threshold.setToolTip(
            "Threshold (fraction of peak) for detecting active signal zone.\n"
            "Lower → wider active zone, higher → more conservative detection."
        )
        adg.addWidget(QLabel("Signal threshold"), adr, 0)
        adg.addWidget(self.ad_signal_threshold, adr, 1); adr += 1

        layout.addWidget(self.adapt_dist_group)

        # ═══════════════════════════════════════════
        #  ADAPTIVE STEPS (toggleable)
        # ═══════════════════════════════════════════
        self.adapt_step_group = QGroupBox("Adaptive Steps")
        self.adapt_step_group.setCheckable(True)
        self.adapt_step_group.setChecked(False)
        self.adapt_step_group.setToolTip(
            "Variable step-size search: step grows in flat regions, shrinks near peaks.\n"
            "Choose a sub-mode and tune sensitivity.\n"
            "Can be combined with Adaptive Distance for automatic range detection first."
        )
        as_lay = QVBoxLayout(self.adapt_step_group)
        as_lay.setSpacing(5)

        # Sub-mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.astep_mode_combo = QComboBox()
        self.astep_mode_combo.addItem(self._ASTEP_GRADIENT)
        self.astep_mode_combo.addItem(self._ASTEP_RATIO)
        self.astep_mode_combo.addItem(self._ASTEP_BRACKET)
        self.astep_mode_combo.setToolTip(
            "Gradient: step ∝ metric gradient magnitude\n"
            "Ratio: step ∝ consecutive score ratio\n"
            "Bracketing: coarse sweep + multi-level narrowing"
        )
        mode_row.addWidget(self.astep_mode_combo)
        as_lay.addLayout(mode_row)

        # Common params
        common_grid = QGridLayout()
        common_grid.setSpacing(4)
        cr = 0

        self.step_init_mm = QDoubleSpinBox()
        self.step_init_mm.setRange(0.0, 100.0)
        self.step_init_mm.setDecimals(4)
        self.step_init_mm.setValue(0.0)
        self.step_init_mm.setSpecialValueText("Auto")
        self.step_init_mm.setSuffix(" mm")
        self.step_init_mm.setToolTip("Initial step size (Auto = range / 10)")
        common_grid.addWidget(QLabel("Initial step"), cr, 0)
        common_grid.addWidget(self.step_init_mm, cr, 1); cr += 1
        as_lay.addLayout(common_grid)

        as_lay.addWidget(_Separator())

        # Stacked widget: page 0 = Gradient/Ratio params, page 1 = Bracketing params
        self._astep_stack = QStackedWidget()

        # Page 0: Gradient / Ratio sensitivity
        grad_ratio_page = QWidget()
        gr_grid = QGridLayout(grad_ratio_page)
        gr_grid.setContentsMargins(0, 0, 0, 0)
        gr_grid.setSpacing(4)
        grr = 0

        self.grow_factor = QDoubleSpinBox()
        self.grow_factor.setRange(1.01, 5.0)
        self.grow_factor.setDecimals(2)
        self.grow_factor.setValue(1.50)
        self.grow_factor.setToolTip("Step enlarge factor in flat regions")
        gr_grid.addWidget(QLabel("Grow factor"), grr, 0)
        gr_grid.addWidget(self.grow_factor, grr, 1); grr += 1

        self.shrink_factor = QDoubleSpinBox()
        self.shrink_factor.setRange(0.05, 0.99)
        self.shrink_factor.setDecimals(2)
        self.shrink_factor.setValue(0.40)
        self.shrink_factor.setToolTip("Step shrink factor near peak")
        gr_grid.addWidget(QLabel("Shrink factor"), grr, 0)
        gr_grid.addWidget(self.shrink_factor, grr, 1); grr += 1

        self._astep_stack.addWidget(grad_ratio_page)  # index 0

        # Page 1: Bracketing sensitivity
        bracket_page = QWidget()
        br_grid = QGridLayout(bracket_page)
        br_grid.setContentsMargins(0, 0, 0, 0)
        br_grid.setSpacing(4)
        brr = 0

        self.refine_levels = QSpinBox()
        self.refine_levels.setRange(1, 10)
        self.refine_levels.setValue(3)
        self.refine_levels.setToolTip(
            "Number of refinement levels.\n"
            "More levels → higher precision, more evaluations."
        )
        br_grid.addWidget(QLabel("Refine levels"), brr, 0)
        br_grid.addWidget(self.refine_levels, brr, 1); brr += 1

        self.refine_divisions = QSpinBox()
        self.refine_divisions.setRange(3, 20)
        self.refine_divisions.setValue(6)
        self.refine_divisions.setToolTip(
            "Divisions per refinement level.\n"
            "More divisions → finer search at each level."
        )
        br_grid.addWidget(QLabel("Refine divisions"), brr, 0)
        br_grid.addWidget(self.refine_divisions, brr, 1); brr += 1

        self._astep_stack.addWidget(bracket_page)  # index 1

        as_lay.addWidget(self._astep_stack)
        layout.addWidget(self.adapt_step_group)

        # Wire sub-mode combo to stacked widget
        self.astep_mode_combo.currentTextChanged.connect(self._on_astep_mode_changed)
        self._on_astep_mode_changed(self.astep_mode_combo.currentText())


        # ═══════════════════════════════════════════
        #  ROI TRACKER AUTOFOCUS
        # ═══════════════════════════════════════════
        self.roi_tracker_group = QGroupBox("ROI Tracker Autofocus")
        self.roi_tracker_group.setCheckable(True)
        self.roi_tracker_group.setChecked(False)
        self.roi_tracker_group.setToolTip(
            "Detect objects from PHASE data (cells, microspheres, etc.),\n"
            "select one, then autofocus uses that ROI as a mask.\n"
            "Phase-based detection is more reliable for transparent objects.\n"
            "The ROI bounds are passed to autofocus algorithms for focused evaluation."
        )
        rtg = QGridLayout(self.roi_tracker_group)
        rtg.setSpacing(5)
        rr = 0

        # ── Action buttons row ──
        btn_row = QHBoxLayout()
        self.detect_objects_btn = QPushButton("Detect Objects (Phase)")
        self.detect_objects_btn.setToolTip(
            "Detect cells/microspheres from unwrapped phase data.\n"
            "Results shown on Amplitude panel as autofocus mask regions.\n"
            "Click on a detected object to select it as ROI for autofocus."
        )
        self.roi_clear_btn = QPushButton("Clear ROI")
        self.roi_clear_btn.setToolTip("Clear the current ROI selection and overlays")
        self.roi_clear_btn.setEnabled(False)
        btn_row.addWidget(self.detect_objects_btn)
        btn_row.addWidget(self.roi_clear_btn)
        rtg.addLayout(btn_row, rr, 0, 1, 2); rr += 1

        # ── Detection parameters ──
        self.detect_method_combo = QComboBox()
        self.detect_method_combo.addItem("Adaptive Threshold", "adaptive")
        self.detect_method_combo.addItem("Otsu Threshold", "otsu")
        self.detect_method_combo.addItem("LoG Blob Detection", "log")
        self.detect_method_combo.setToolTip(
            "Adaptive: best for uneven illumination (default)\n"
            "Otsu: best for high-contrast images\n"
            "LoG: best for round particles/cells"
        )
        rtg.addWidget(QLabel("Detection"), rr, 0)
        rtg.addWidget(self.detect_method_combo, rr, 1); rr += 1

        self.detect_sensitivity = QDoubleSpinBox()
        self.detect_sensitivity.setRange(0.1, 5.0)
        self.detect_sensitivity.setValue(1.0)
        self.detect_sensitivity.setSingleStep(0.1)
        self.detect_sensitivity.setDecimals(1)
        self.detect_sensitivity.setToolTip("Higher = more objects detected (lower threshold)")
        rtg.addWidget(QLabel("Sensitivity"), rr, 0)
        rtg.addWidget(self.detect_sensitivity, rr, 1); rr += 1

        self.detect_min_area = QSpinBox()
        self.detect_min_area.setRange(10, 10000)
        self.detect_min_area.setValue(50)
        self.detect_min_area.setSingleStep(10)
        self.detect_min_area.setSuffix(" px")
        self.detect_min_area.setToolTip("Minimum object area to detect")
        rtg.addWidget(QLabel("Min area"), rr, 0)
        rtg.addWidget(self.detect_min_area, rr, 1); rr += 1

        rtg.addWidget(_Separator(), rr, 0, 1, 2); rr += 1

        # ── Manual ROI ──
        manual_row = QHBoxLayout()
        self.roi_select_btn = QPushButton("Manual ROI Select")
        self.roi_select_btn.setToolTip("Click on the Amplitude panel to manually place ROI as autofocus mask")
        self.roi_select_btn.setCheckable(True)
        manual_row.addWidget(self.roi_select_btn)

        self.roi_size_spin = QSpinBox()
        self.roi_size_spin.setRange(16, 512)
        self.roi_size_spin.setValue(64)
        self.roi_size_spin.setSingleStep(16)
        self.roi_size_spin.setSuffix(" px")
        self.roi_size_spin.setToolTip("ROI size in pixels (square)")
        manual_row.addWidget(self.roi_size_spin)
        rtg.addLayout(manual_row, rr, 0, 1, 2); rr += 1

        rtg.addWidget(_Separator(), rr, 0, 1, 2); rr += 1

        # ── Tracking ──
        self.roi_track_live_cb = QCheckBox("Track ROI across frames")
        self.roi_track_live_cb.setToolTip(
            "Use phase-correlation to follow the ROI\n"
            "as the object moves between frames (live mode)"
        )
        rtg.addWidget(self.roi_track_live_cb, rr, 0, 1, 2); rr += 1

        self.roi_status_label = QLabel("No ROI selected")
        self.roi_status_label.setStyleSheet("color: gray; font-style: italic;")
        rtg.addWidget(self.roi_status_label, rr, 0, 1, 2); rr += 1

        layout.addWidget(self.roi_tracker_group)

        # ═══════════════════════════════════════════
        #  LIVE ADAPTIVE TRACKING
        # ═══════════════════════════════════════════
        adapt_group = QGroupBox("Live Tracking")
        adapt_group.setToolTip("Continuous autofocus during live camera acquisition")
        al = QGridLayout(adapt_group)
        al.setSpacing(5)
        ar = 0

        self.adapt_enable_cb = QCheckBox("Enable adaptive tracking")
        al.addWidget(self.adapt_enable_cb, ar, 0, 1, 2); ar += 1

        self.adapt_range_mm = QDoubleSpinBox()
        self.adapt_range_mm.setRange(0.001, 100.0)
        self.adapt_range_mm.setDecimals(3)
        self.adapt_range_mm.setValue(0.2)
        self.adapt_range_mm.setSuffix(" mm")
        self.adapt_range_mm.setToolTip("Search window around current z")
        al.addWidget(QLabel("Search range ±"), ar, 0)
        al.addWidget(self.adapt_range_mm, ar, 1); ar += 1

        self.adapt_steps = QSpinBox()
        self.adapt_steps.setRange(3, 50)
        self.adapt_steps.setValue(11)
        al.addWidget(QLabel("Steps"), ar, 0)
        al.addWidget(self.adapt_steps, ar, 1); ar += 1

        self.adapt_n_interval = QSpinBox()
        self.adapt_n_interval.setRange(1, 1000)
        self.adapt_n_interval.setValue(5)
        self.adapt_n_interval.setToolTip("Run autofocus every N acquired frames")
        al.addWidget(QLabel("Every N frames"), ar, 0)
        al.addWidget(self.adapt_n_interval, ar, 1); ar += 1

        layout.addWidget(adapt_group)
        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ─── Internal slots ───

    def _on_astep_mode_changed(self, text: str):
        """Switch stacked widget page based on adaptive step sub-mode."""
        if text == self._ASTEP_BRACKET:
            self._astep_stack.setCurrentIndex(1)
        else:
            self._astep_stack.setCurrentIndex(0)

    # ─── Public API ───

    def get_effective_algorithm(self) -> str:
        """
        Return the dispatch key for the refinement algorithm.
        If Adaptive Steps is enabled, use that; otherwise use the base combo.
        (Adaptive Distance is handled separately as a range-finder.)
        """
        if self.adapt_step_group.isChecked():
            mode = self.astep_mode_combo.currentText()
            if mode == self._ASTEP_GRADIENT:
                return self.ALGO_ADAPT_GRADIENT
            elif mode == self._ASTEP_RATIO:
                return self.ALGO_ADAPT_RATIO
            elif mode == self._ASTEP_BRACKET:
                return self.ALGO_ADAPT_BRACKET
        return self.algo_combo.currentText()

    def is_adaptive_distance_enabled(self) -> bool:
        """Return True if Adaptive Distance (range finder) is enabled."""
        return self.adapt_dist_group.isChecked()

    def get_state(self) -> dict:
        return {
            "metric": self.metric_combo.currentText(),
            "algorithm": self.algo_combo.currentText(),
            "auto_metric": self.auto_metric_cb.isChecked(),
            "zscan_min_mm": self.zscan_min_mm.value(),
            "zscan_max_mm": self.zscan_max_mm.value(),
            "zscan_steps": self.zscan_steps.value(),
            # Adaptive Distance
            "adapt_dist_enabled": self.adapt_dist_group.isChecked(),
            "ad_initial_range_mm": self.ad_initial_range_mm.value(),
            "ad_max_range_mm": self.ad_max_range_mm.value(),
            "ad_expand_factor": self.ad_expand_factor.value(),
            "ad_signal_threshold": self.ad_signal_threshold.value(),
            # Adaptive Steps
            "adapt_step_enabled": self.adapt_step_group.isChecked(),
            "astep_mode": self.astep_mode_combo.currentText(),
            "step_init_mm": self.step_init_mm.value(),
            "grow_factor": self.grow_factor.value(),
            "shrink_factor": self.shrink_factor.value(),
            "refine_levels": self.refine_levels.value(),
            "refine_divisions": self.refine_divisions.value(),
            # Live tracking
            "adapt_enable": self.adapt_enable_cb.isChecked(),
            "adapt_range_mm": self.adapt_range_mm.value(),
            "adapt_steps": self.adapt_steps.value(),
            "adapt_n_interval": self.adapt_n_interval.value(),
            # ROI Tracker
            "roi_tracker_enabled": self.roi_tracker_group.isChecked(),
            "roi_size": self.roi_size_spin.value(),
            "roi_track_live": self.roi_track_live_cb.isChecked(),
            "detect_method": self.detect_method_combo.currentData(),
            "detect_sensitivity": self.detect_sensitivity.value(),
            "detect_min_area": self.detect_min_area.value(),
        }

    def set_state(self, state: dict):
        if "metric" in state:
            idx = self.metric_combo.findText(state["metric"])
            if idx >= 0: self.metric_combo.setCurrentIndex(idx)
        if "algorithm" in state:
            idx = self.algo_combo.findText(state["algorithm"])
            if idx >= 0: self.algo_combo.setCurrentIndex(idx)
        if "auto_metric" in state:
            self.auto_metric_cb.setChecked(state["auto_metric"])
        for key, widget in [
            ("zscan_min_mm", self.zscan_min_mm), ("zscan_max_mm", self.zscan_max_mm),
            ("ad_initial_range_mm", self.ad_initial_range_mm),
            ("ad_max_range_mm", self.ad_max_range_mm),
            ("ad_expand_factor", self.ad_expand_factor),
            ("ad_signal_threshold", self.ad_signal_threshold),
            ("step_init_mm", self.step_init_mm),
            ("grow_factor", self.grow_factor), ("shrink_factor", self.shrink_factor),
            ("adapt_range_mm", self.adapt_range_mm),
        ]:
            if key in state:
                widget.setValue(state[key])
        for key, widget in [
            ("zscan_steps", self.zscan_steps),
            ("refine_levels", self.refine_levels),
            ("refine_divisions", self.refine_divisions),
            ("adapt_steps", self.adapt_steps),
            ("adapt_n_interval", self.adapt_n_interval),
        ]:
            if key in state:
                widget.setValue(state[key])
        for key, widget in [
            ("adapt_enable", self.adapt_enable_cb),
        ]:
            if key in state:
                widget.setChecked(state[key])
        if "adapt_dist_enabled" in state:
            self.adapt_dist_group.setChecked(state["adapt_dist_enabled"])
        if "adapt_step_enabled" in state:
            self.adapt_step_group.setChecked(state["adapt_step_enabled"])
        if "astep_mode" in state:
            idx = self.astep_mode_combo.findText(state["astep_mode"])
            if idx >= 0: self.astep_mode_combo.setCurrentIndex(idx)
        # ROI Tracker
        if "roi_tracker_enabled" in state:
            self.roi_tracker_group.setChecked(state["roi_tracker_enabled"])
        if "roi_size" in state:
            self.roi_size_spin.setValue(state["roi_size"])
        if "roi_track_live" in state:
            self.roi_track_live_cb.setChecked(state["roi_track_live"])
        if "detect_method" in state:
            idx = self.detect_method_combo.findData(state["detect_method"])
            if idx >= 0: self.detect_method_combo.setCurrentIndex(idx)
        if "detect_sensitivity" in state:
            self.detect_sensitivity.setValue(state["detect_sensitivity"])
        if "detect_min_area" in state:
            self.detect_min_area.setValue(state["detect_min_area"])
