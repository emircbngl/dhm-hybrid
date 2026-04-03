"""
QPI (Quantitative Phase Imaging) sidebar tab.

Controls:
  - Mode selection (bio / micro / both)
  - Optical parameters (n_sample, n_medium, alpha)
  - Cell segmentation threshold
  - Display selectors (which maps to show)
  - Compute button → populates display panels
  - Stats panel (read-only, shows computed metrics)

Add to SidebarTabs:
    from gui.sidebar.qpi_tab import QPITab
    self.qpi_tab = QPITab()
    self.tabs.addTab(self.qpi_tab, "QPI")
"""
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
    QTextEdit,
    QTabWidget,
)
from PySide6.QtCore import Qt, Signal


class _Separator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(2)


class QPITab(QWidget):
    """Sidebar tab — Quantitative Phase Imaging parameters and display controls."""

    # Signals
    compute_requested = Signal()
    display_changed = Signal()

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
        #  MODE & OPTICAL PARAMETERS
        # ═══════════════════════════════════════════
        params_group = QGroupBox("QPI Parameters")
        pg = QGridLayout(params_group)
        pg.setSpacing(5)
        row = 0

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Biological (cells)", "biological")
        self.mode_combo.addItem("Micro-structure / thin film", "micro_structure")
        self.mode_combo.addItem("Both", "both")
        self.mode_combo.setToolTip(
            "Biological: dry mass, cell morphology, growth rate\n"
            "Micro-structure: height, roughness, PSD\n"
            "Both: all parameters"
        )
        pg.addWidget(QLabel("Mode"), row, 0)
        pg.addWidget(self.mode_combo, row, 1); row += 1

        pg.addWidget(_Separator(), row, 0, 1, 2); row += 1

        # n_sample
        self.n_sample = QDoubleSpinBox()
        self.n_sample.setRange(1.0, 3.0)
        self.n_sample.setDecimals(4)
        self.n_sample.setValue(1.3800)
        self.n_sample.setToolTip(
            "Sample refractive index\n"
            "Mammalian cell: ~1.36-1.40\n"
            "Bacteria: ~1.38-1.42\n"
            "Glass: ~1.52\n"
            "Photoresist: ~1.58-1.64"
        )
        pg.addWidget(QLabel("n (sample)"), row, 0)
        pg.addWidget(self.n_sample, row, 1); row += 1

        # n_medium
        self.n_medium = QDoubleSpinBox()
        self.n_medium.setRange(1.0, 2.0)
        self.n_medium.setDecimals(4)
        self.n_medium.setValue(1.3370)
        self.n_medium.setToolTip(
            "Medium refractive index\n"
            "Air: 1.000\n"
            "Water: 1.333\n"
            "Culture medium: ~1.337\n"
            "Immersion oil: ~1.515"
        )
        pg.addWidget(QLabel("n (medium)"), row, 0)
        pg.addWidget(self.n_medium, row, 1); row += 1

        # Alpha (specific refractive increment)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.01, 1.0)
        self.alpha.setDecimals(3)
        self.alpha.setValue(0.180)
        self.alpha.setSuffix(" mL/g")
        self.alpha.setToolTip(
            "Specific refractive increment (dn/dc)\n"
            "Protein: 0.18-0.19\n"
            "DNA: 0.16-0.18\n"
            "Lipid: 0.14-0.16\n"
            "General cell: 0.18"
        )
        pg.addWidget(QLabel("α (dn/dc)"), row, 0)
        pg.addWidget(self.alpha, row, 1); row += 1

        # Known thickness (optional, for RI mapping)
        self.known_thickness_cb = QCheckBox("Known thickness")
        self.known_thickness_cb.setToolTip("If checked, compute refractive index map instead of height")
        pg.addWidget(self.known_thickness_cb, row, 0, 1, 2); row += 1

        self.known_thickness_nm = QDoubleSpinBox()
        self.known_thickness_nm.setRange(0.1, 1e6)
        self.known_thickness_nm.setDecimals(1)
        self.known_thickness_nm.setValue(100.0)
        self.known_thickness_nm.setSuffix(" nm")
        self.known_thickness_nm.setEnabled(False)
        pg.addWidget(QLabel("Thickness"), row, 0)
        pg.addWidget(self.known_thickness_nm, row, 1); row += 1

        self.known_thickness_cb.toggled.connect(self.known_thickness_nm.setEnabled)

        # Cell segmentation threshold
        self.cell_threshold = QDoubleSpinBox()
        self.cell_threshold.setRange(0.01, 0.90)
        self.cell_threshold.setDecimals(2)
        self.cell_threshold.setValue(0.15)
        self.cell_threshold.setToolTip(
            "OPD threshold for cell segmentation (fraction of range)\n"
            "Lower = more pixels included, higher = stricter"
        )
        pg.addWidget(QLabel("Cell threshold"), row, 0)
        pg.addWidget(self.cell_threshold, row, 1); row += 1

        layout.addWidget(params_group)

        # ═══════════════════════════════════════════
        #  DISPLAY PANELS SELECTION
        # ═══════════════════════════════════════════
        display_group = QGroupBox("Display Panels")
        dg = QGridLayout(display_group)
        dg.setSpacing(4)
        dr = 0

        self.show_opd_cb = QCheckBox("OPD map (nm)")
        self.show_opd_cb.setChecked(True)
        dg.addWidget(self.show_opd_cb, dr, 0); dr += 1

        self.show_height_cb = QCheckBox("Height / topography (nm)")
        self.show_height_cb.setChecked(True)
        dg.addWidget(self.show_height_cb, dr, 0); dr += 1

        self.show_drymass_cb = QCheckBox("Dry mass density (pg/μm²)")
        self.show_drymass_cb.setChecked(True)
        dg.addWidget(self.show_drymass_cb, dr, 0); dr += 1

        self.show_ri_cb = QCheckBox("Refractive index map")
        self.show_ri_cb.setChecked(False)
        self.show_ri_cb.setToolTip("Only available when 'Known thickness' is set")
        dg.addWidget(self.show_ri_cb, dr, 0); dr += 1

        self.show_3d_cb = QCheckBox("3D surface view")
        self.show_3d_cb.setChecked(False)
        self.show_3d_cb.setToolTip("3D rendering of height map (slower)")
        dg.addWidget(self.show_3d_cb, dr, 0); dr += 1

        self.show_psd_cb = QCheckBox("Power spectral density")
        self.show_psd_cb.setChecked(False)
        dg.addWidget(self.show_psd_cb, dr, 0); dr += 1

        self.show_histogram_cb = QCheckBox("Phase histogram")
        self.show_histogram_cb.setChecked(True)
        dg.addWidget(self.show_histogram_cb, dr, 0); dr += 1

        # Connect checkboxes to display_changed signal
        for cb in [self.show_opd_cb, self.show_height_cb, self.show_drymass_cb,
                    self.show_ri_cb, self.show_3d_cb, self.show_psd_cb, self.show_histogram_cb]:
            cb.toggled.connect(self.display_changed.emit)

        layout.addWidget(display_group)

        # ═══════════════════════════════════════════
        #  COMPUTE BUTTON
        # ═══════════════════════════════════════════
        self.compute_btn = QPushButton("Compute QPI")
        self.compute_btn.setMinimumHeight(32)
        self.compute_btn.setEnabled(False)
        self.compute_btn.setToolTip("Run full QPI analysis on current reconstruction")
        self.compute_btn.clicked.connect(self.compute_requested.emit)
        layout.addWidget(self.compute_btn)

        # ═══════════════════════════════════════════
        #  RESULTS — READ-ONLY STATS DISPLAY
        # ═══════════════════════════════════════════
        self.stats_tabs = QTabWidget()
        self.stats_tabs.setMaximumHeight(260)

        # Phase stats
        self.phase_stats_text = QTextEdit()
        self.phase_stats_text.setReadOnly(True)
        self.phase_stats_text.setPlaceholderText("Phase statistics will appear here...")
        self.stats_tabs.addTab(self.phase_stats_text, "Phase")

        # Morphology stats (bio)
        self.morph_stats_text = QTextEdit()
        self.morph_stats_text.setReadOnly(True)
        self.morph_stats_text.setPlaceholderText("Cell morphology will appear here...")
        self.stats_tabs.addTab(self.morph_stats_text, "Cell")

        # Roughness stats (micro)
        self.roughness_stats_text = QTextEdit()
        self.roughness_stats_text.setReadOnly(True)
        self.roughness_stats_text.setPlaceholderText("Surface roughness will appear here...")
        self.stats_tabs.addTab(self.roughness_stats_text, "Surface")

        layout.addWidget(self.stats_tabs)

        layout.addStretch()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Connect mode to visibility
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed(0)

    def _on_mode_changed(self, idx: int):
        mode = self.mode_combo.currentData()
        is_bio = mode in ("biological", "both")
        is_micro = mode in ("micro_structure", "both")

        self.cell_threshold.setVisible(is_bio)
        # Find and toggle the label too
        for i in range(self.layout().count()):
            pass  # labels are inside group grid, handled by visibility of spinbox

        self.show_drymass_cb.setVisible(is_bio)
        self.show_psd_cb.setVisible(is_micro)

    # ─────────────────────────────────────────
    # Public: populate stats from QPIResult
    # ─────────────────────────────────────────

    def set_phase_stats(self, stats):
        """Accepts a PhaseStatistics dataclass."""
        lines = [
            f"Mean:  {stats.mean_rad:.4f} rad  ({stats.mean_nm:.1f} nm OPD)",
            f"Std:   {stats.std_rad:.4f} rad  ({stats.std_nm:.1f} nm OPD)",
            f"Range: {stats.range_rad:.4f} rad  ({stats.range_nm:.1f} nm OPD)",
        ]
        self.phase_stats_text.setPlainText("\n".join(lines))

    def set_cell_stats(self, morph):
        """Accepts a CellMorphology dataclass."""
        lines = [
            f"Area:       {morph.area_um2:.1f} μm²",
            f"Volume:     {morph.volume_fL:.1f} fL",
            f"Dry mass:   {morph.dry_mass_pg:.2f} pg",
            f"Density:    {morph.dry_mass_density_pg_um2:.4f} pg/μm²",
            f"Max height: {morph.max_height_nm:.1f} nm",
            f"Mean height:{morph.mean_height_nm:.1f} nm",
            f"─────────────────────",
            f"Perimeter:  {morph.perimeter_um:.1f} μm",
            f"Circularity:{morph.circularity:.3f}",
            f"Aspect:     {morph.aspect_ratio:.2f}",
            f"Eccentr.:   {morph.eccentricity:.3f}",
        ]
        self.morph_stats_text.setPlainText("\n".join(lines))

    def set_roughness_stats(self, r):
        """Accepts a RoughnessResult dataclass."""
        lines = [
            f"Ra (mean):    {r.Ra*1e9:.2f} nm",
            f"Rq (RMS):     {r.Rq*1e9:.2f} nm",
            f"Rz (peak-val):{r.Rz*1e9:.2f} nm",
            f"Rsk (skew):   {r.Rsk:.3f}",
            f"Rku (kurt):   {r.Rku:.3f}",
            f"",
            f"Rsk > 0 → peaks dominant",
            f"Rsk < 0 → valleys dominant",
            f"Rku > 3 → sharp features",
        ]
        self.roughness_stats_text.setPlainText("\n".join(lines))

    def clear_stats(self):
        self.phase_stats_text.clear()
        self.morph_stats_text.clear()
        self.roughness_stats_text.clear()

    # ─────────────────────────────────────────
    # State save/load
    # ─────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "mode": self.mode_combo.currentData(),
            "n_sample": self.n_sample.value(),
            "n_medium": self.n_medium.value(),
            "alpha": self.alpha.value(),
            "known_thickness": self.known_thickness_cb.isChecked(),
            "known_thickness_nm": self.known_thickness_nm.value(),
            "cell_threshold": self.cell_threshold.value(),
            "show_opd": self.show_opd_cb.isChecked(),
            "show_height": self.show_height_cb.isChecked(),
            "show_drymass": self.show_drymass_cb.isChecked(),
            "show_ri": self.show_ri_cb.isChecked(),
            "show_3d": self.show_3d_cb.isChecked(),
            "show_psd": self.show_psd_cb.isChecked(),
            "show_histogram": self.show_histogram_cb.isChecked(),
        }

    def set_state(self, state: dict):
        if "mode" in state:
            for i in range(self.mode_combo.count()):
                if self.mode_combo.itemData(i) == state["mode"]:
                    self.mode_combo.setCurrentIndex(i); break
        for key, widget in [
            ("n_sample", self.n_sample), ("n_medium", self.n_medium),
            ("alpha", self.alpha), ("known_thickness_nm", self.known_thickness_nm),
            ("cell_threshold", self.cell_threshold),
        ]:
            if key in state:
                widget.setValue(state[key])
        for key, widget in [
            ("known_thickness", self.known_thickness_cb),
            ("show_opd", self.show_opd_cb), ("show_height", self.show_height_cb),
            ("show_drymass", self.show_drymass_cb), ("show_ri", self.show_ri_cb),
            ("show_3d", self.show_3d_cb), ("show_psd", self.show_psd_cb),
            ("show_histogram", self.show_histogram_cb),
        ]:
            if key in state:
                widget.setChecked(state[key])
