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
    QGroupBox,
    QFrame,
    QScrollArea,
    QPushButton,
)
from PySide6.QtCore import Signal
from ..widgets.contrast_controls import ContrastControls


class ProcessTab(QWidget):
    """Sidebar tab — preprocessing, reference hologram, and frequency-domain filtering."""

    ref_load_requested = Signal()       # user wants to load a reference from file
    ref_capture_requested = Signal()    # user wants to capture current frame as reference
    ref_clear_requested = Signal()      # user wants to clear the active reference

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ─── Preprocessing ───
        pre_group = QGroupBox("Preprocessing")
        pg = QGridLayout(pre_group)
        pg.setSpacing(6)

        self.subtract_mean_cb = QCheckBox("Subtract mean")
        self.subtract_mean_cb.setChecked(True)
        self.subtract_mean_cb.setToolTip("Remove DC offset before reconstruction")
        pg.addWidget(self.subtract_mean_cb, 0, 0)

        self.hann_cb = QCheckBox("Hann window")
        self.hann_cb.setChecked(False)
        self.hann_cb.setToolTip("Apply Hann window to reduce edge artefacts")
        pg.addWidget(self.hann_cb, 1, 0)

        pg.addWidget(QLabel("Spectral mask"), 2, 0)
        self.mask_apod_combo = QComboBox()
        self.mask_apod_combo.addItem("Tukey (recommended)", "tukey")
        self.mask_apod_combo.addItem("Hann", "hann")
        self.mask_apod_combo.addItem("Gaussian", "gaussian")
        self.mask_apod_combo.addItem("Hard (no apodization)", "none")
        self.mask_apod_combo.setToolTip(
            "Apodization window for the spectral circular mask.\n"
            "Hard masks cause Gibbs ringing artifacts in the phase image.\n"
            "Tukey/Hann/Gaussian provide smooth edges that eliminate ringing."
        )
        pg.addWidget(self.mask_apod_combo, 2, 1)

        pg.addWidget(QLabel("Mask rolloff"), 3, 0)
        self.mask_rolloff = QDoubleSpinBox()
        self.mask_rolloff.setRange(0.05, 0.80)
        self.mask_rolloff.setDecimals(2)
        self.mask_rolloff.setValue(0.25)
        self.mask_rolloff.setSingleStep(0.05)
        self.mask_rolloff.setToolTip(
            "Fraction of the mask radius used for the soft transition zone.\n"
            "Larger values = smoother but narrower effective bandwidth."
        )
        pg.addWidget(self.mask_rolloff, 3, 1)

        layout.addWidget(pre_group)

        # ─── Reference Hologram ───
        ref_group = QGroupBox("Reference Hologram")
        rg = QVBoxLayout(ref_group)
        rg.setSpacing(6)

        ref_desc = QLabel(
            "Load or capture a reference hologram (without sample).\n"
            "The reference phase is subtracted from every reconstruction\n"
            "to remove systematic aberrations (tilt, curvature, lens errors)."
        )
        ref_desc.setWordWrap(True)
        ref_desc.setStyleSheet("color: #888; font-size: 11px;")
        rg.addWidget(ref_desc)

        self.ref_enable_cb = QCheckBox("Enable reference subtraction")
        self.ref_enable_cb.setChecked(False)
        self.ref_enable_cb.setToolTip(
            "When enabled, the reference complex field is divided out\n"
            "from each reconstruction before phase unwrapping."
        )
        rg.addWidget(self.ref_enable_cb)

        btn_row = QHBoxLayout()
        self.ref_load_btn = QPushButton("Load from File…")
        self.ref_load_btn.setToolTip("Load a reference hologram image from disk")
        self.ref_load_btn.clicked.connect(self.ref_load_requested.emit)
        btn_row.addWidget(self.ref_load_btn)

        self.ref_capture_btn = QPushButton("Capture Current")
        self.ref_capture_btn.setToolTip(
            "Use the currently loaded/live image as the reference hologram.\n"
            "Make sure no sample is present when capturing."
        )
        self.ref_capture_btn.clicked.connect(self.ref_capture_requested.emit)
        btn_row.addWidget(self.ref_capture_btn)

        self.ref_clear_btn = QPushButton("Clear")
        self.ref_clear_btn.setToolTip("Remove the loaded reference hologram")
        self.ref_clear_btn.clicked.connect(self.ref_clear_requested.emit)
        btn_row.addWidget(self.ref_clear_btn)
        rg.addLayout(btn_row)

        self.ref_status_label = QLabel("No reference loaded")
        self.ref_status_label.setStyleSheet("color: #999; font-style: italic;")
        rg.addWidget(self.ref_status_label)

        layout.addWidget(ref_group)

        # ─── Frequency Filter ───
        filter_group = QGroupBox("Frequency Filter")
        fg = QGridLayout(filter_group)
        fg.setSpacing(6)

        self.filter_enable_cb = QCheckBox("Enable filter")
        self.filter_enable_cb.setChecked(False)
        self.filter_enable_cb.setToolTip("Apply frequency-domain filter after off-axis extraction")
        fg.addWidget(self.filter_enable_cb, 0, 0, 1, 2)

        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItem("Low-pass", "lowpass")
        self.filter_type_combo.addItem("High-pass", "highpass")
        self.filter_type_combo.setToolTip("Filter type")
        fg.addWidget(QLabel("Type"), 1, 0)
        fg.addWidget(self.filter_type_combo, 1, 1)

        self.filter_cutoff = QDoubleSpinBox()
        self.filter_cutoff.setRange(0.001, 0.5)
        self.filter_cutoff.setDecimals(3)
        self.filter_cutoff.setValue(0.15)
        self.filter_cutoff.setToolTip("Normalised cutoff frequency (0 to 0.5)")
        fg.addWidget(QLabel("Cutoff"), 2, 0)
        fg.addWidget(self.filter_cutoff, 2, 1)

        self.filter_rolloff = QDoubleSpinBox()
        self.filter_rolloff.setRange(0.001, 0.2)
        self.filter_rolloff.setDecimals(3)
        self.filter_rolloff.setValue(0.02)
        self.filter_rolloff.setToolTip("Tukey rolloff width")
        fg.addWidget(QLabel("Rolloff"), 3, 0)
        fg.addWidget(self.filter_rolloff, 3, 1)
        layout.addWidget(filter_group)

        # ─── Phase Unwrapping ───
        unwrap_group = QGroupBox("Phase Unwrapping")
        ug = QGridLayout(unwrap_group)
        ug.setSpacing(6)
        r = 0

        self.unwrap_method_combo = QComboBox()
        self.unwrap_method_combo.addItem("Gradient Integration (recommended)", "gradient_integration")
        self.unwrap_method_combo.addItem("TIE (Transport of Intensity)", "tie")
        self.unwrap_method_combo.addItem("Thin Sample (Born approx.)", "thin_sample")
        self.unwrap_method_combo.addItem("Optical (Synthetic λ)", "optical")
        self.unwrap_method_combo.addItem("Least-Squares (skimage)", "least_squares")
        self.unwrap_method_combo.addItem("Quality-Guided", "quality_guided")
        self.unwrap_method_combo.addItem("Goldstein Branch-Cut", "goldstein")
        self.unwrap_method_combo.setToolTip(
            "Gradient Integration: Recovers phase from complex field via DCT Poisson. (recommended)\n"
            "TIE: Propagates to z±δz, solves Transport of Intensity Equation.\n"
            "Thin Sample: Born approximation — fastest, best for |OPD| < λ.\n"
            "Optical: Synthetic-wavelength unwrapping for extended range.\n"
            "Least-Squares: skimage C-compiled baseline.\n"
            "Quality-Guided: Heap-based reliability unwrap (slower).\n"
            "Goldstein: Branch-cut based (slower)."
        )
        ug.addWidget(QLabel("Method"), r, 0)
        ug.addWidget(self.unwrap_method_combo, r, 1); r += 1

        # TIE-specific parameters
        self.tie_dz_label = QLabel("  TIE δz (µm)")
        self.tie_dz = QDoubleSpinBox()
        self.tie_dz.setRange(0.1, 50.0)
        self.tie_dz.setDecimals(1)
        self.tie_dz.setValue(1.0)
        self.tie_dz.setToolTip("Axial propagation step for TIE (micrometres)")
        ug.addWidget(self.tie_dz_label, r, 0)
        ug.addWidget(self.tie_dz, r, 1); r += 1

        # Optical-specific parameters
        self.optical_dlam_label = QLabel("  Synth. Δλ (nm)")
        self.optical_dlam = QDoubleSpinBox()
        self.optical_dlam.setRange(1.0, 100.0)
        self.optical_dlam.setDecimals(1)
        self.optical_dlam.setValue(10.0)
        self.optical_dlam.setToolTip("Synthetic wavelength offset for optical unwrapping (nm)")
        ug.addWidget(self.optical_dlam_label, r, 0)
        ug.addWidget(self.optical_dlam, r, 1); r += 1

        # Pre-processing
        self.unwrap_pre_median = QCheckBox("Pre: Median filter")
        self.unwrap_pre_median.setToolTip(
            "Apply median filter on wrapped phase before unwrapping.\n"
            "Reduces salt-and-pepper noise without destroying wraps."
        )
        ug.addWidget(self.unwrap_pre_median, r, 0, 1, 2); r += 1

        self.unwrap_pre_median_size = QSpinBox()
        self.unwrap_pre_median_size.setRange(3, 9)
        self.unwrap_pre_median_size.setSingleStep(2)
        self.unwrap_pre_median_size.setValue(3)
        self.unwrap_pre_median_size.setToolTip("Median kernel size (odd number)")
        ug.addWidget(QLabel("  Kernel"), r, 0)
        ug.addWidget(self.unwrap_pre_median_size, r, 1); r += 1

        self.unwrap_pre_gaussian = QCheckBox("Pre: Gaussian smooth")
        self.unwrap_pre_gaussian.setToolTip(
            "Gentle Gaussian blur on wrapped phase (in cos/sin domain).\n"
            "Smooths noise while preserving phase wraps."
        )
        ug.addWidget(self.unwrap_pre_gaussian, r, 0, 1, 2); r += 1

        self.unwrap_pre_gauss_sigma = QDoubleSpinBox()
        self.unwrap_pre_gauss_sigma.setRange(0.1, 5.0)
        self.unwrap_pre_gauss_sigma.setDecimals(1)
        self.unwrap_pre_gauss_sigma.setValue(0.8)
        self.unwrap_pre_gauss_sigma.setToolTip("Gaussian sigma (pixels)")
        ug.addWidget(QLabel("  Sigma"), r, 0)
        ug.addWidget(self.unwrap_pre_gauss_sigma, r, 1); r += 1

        # Post-processing
        self.unwrap_post_outlier = QCheckBox("Post: Outlier clip")
        self.unwrap_post_outlier.setChecked(False)
        self.unwrap_post_outlier.setToolTip(
            "Clip outlier values in unwrapped phase (mean ± N×sigma)."
        )
        ug.addWidget(self.unwrap_post_outlier, r, 0, 1, 2); r += 1

        self.unwrap_post_outlier_sigma = QDoubleSpinBox()
        self.unwrap_post_outlier_sigma.setRange(1.0, 10.0)
        self.unwrap_post_outlier_sigma.setDecimals(1)
        self.unwrap_post_outlier_sigma.setValue(3.0)
        self.unwrap_post_outlier_sigma.setToolTip("Outlier clipping threshold (N × std)")
        ug.addWidget(QLabel("  Sigma"), r, 0)
        ug.addWidget(self.unwrap_post_outlier_sigma, r, 1); r += 1

        self.unwrap_post_bg = QCheckBox("Post: Remove background")
        self.unwrap_post_bg.setChecked(True)
        self.unwrap_post_bg.setToolTip(
            "Fit and subtract polynomial background from unwrapped phase.\n"
            "Removes tilt and curvature artefacts. Essential for clean phase images."
        )
        ug.addWidget(self.unwrap_post_bg, r, 0, 1, 2); r += 1

        self.unwrap_post_bg_order = QSpinBox()
        self.unwrap_post_bg_order.setRange(1, 4)
        self.unwrap_post_bg_order.setValue(2)
        self.unwrap_post_bg_order.setToolTip("Polynomial order (1=tilt, 2=curvature, 3+=higher)")
        ug.addWidget(QLabel("  Order"), r, 0)
        ug.addWidget(self.unwrap_post_bg_order, r, 1); r += 1

        layout.addWidget(unwrap_group)

        # ─── Display Contrast ───
        self.contrast = ContrastControls()
        layout.addWidget(self.contrast)

        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def get_unwrap_config(self):
        """Build an UnwrapConfig from current UI state."""
        from core.phase_unwrap import UnwrapConfig, UnwrapMethod
        return UnwrapConfig(
            method=UnwrapMethod(self.unwrap_method_combo.currentData()),
            pre_median=self.unwrap_pre_median.isChecked(),
            pre_median_size=self.unwrap_pre_median_size.value(),
            pre_gaussian=self.unwrap_pre_gaussian.isChecked(),
            pre_gaussian_sigma=self.unwrap_pre_gauss_sigma.value(),
            post_outlier_clip=self.unwrap_post_outlier.isChecked(),
            post_outlier_sigma=self.unwrap_post_outlier_sigma.value(),
            post_bg_remove=self.unwrap_post_bg.isChecked(),
            post_bg_order=self.unwrap_post_bg_order.value(),
            tie_dz_um=self.tie_dz.value(),
            optical_synth_nm=self.optical_dlam.value(),
        )

    def get_state(self) -> dict:
        return {
            "subtract_mean": self.subtract_mean_cb.isChecked(),
            "hann_window": self.hann_cb.isChecked(),
            "mask_apodization": self.mask_apod_combo.currentData(),
            "mask_rolloff": self.mask_rolloff.value(),
            "filter_enable": self.filter_enable_cb.isChecked(),
            "filter_type": self.filter_type_combo.currentText(),
            "filter_cutoff": self.filter_cutoff.value(),
            "filter_rolloff": self.filter_rolloff.value(),
            "unwrap_method": self.unwrap_method_combo.currentData(),
            "unwrap_pre_median": self.unwrap_pre_median.isChecked(),
            "unwrap_pre_median_size": self.unwrap_pre_median_size.value(),
            "unwrap_pre_gaussian": self.unwrap_pre_gaussian.isChecked(),
            "unwrap_pre_gauss_sigma": self.unwrap_pre_gauss_sigma.value(),
            "unwrap_post_outlier": self.unwrap_post_outlier.isChecked(),
            "unwrap_post_outlier_sigma": self.unwrap_post_outlier_sigma.value(),
            "unwrap_post_bg": self.unwrap_post_bg.isChecked(),
            "unwrap_post_bg_order": self.unwrap_post_bg_order.value(),
            "tie_dz_um": self.tie_dz.value(),
            "optical_synth_nm": self.optical_dlam.value(),
            "contrast": self.contrast.get_state(),
        }

    def set_state(self, state: dict):
        if "subtract_mean" in state:
            self.subtract_mean_cb.setChecked(state["subtract_mean"])
        if "hann_window" in state:
            self.hann_cb.setChecked(state["hann_window"])
        if "mask_apodization" in state:
            idx = self.mask_apod_combo.findData(state["mask_apodization"])
            if idx >= 0: self.mask_apod_combo.setCurrentIndex(idx)
        if "mask_rolloff" in state:
            self.mask_rolloff.setValue(state["mask_rolloff"])
        if "filter_enable" in state:
            self.filter_enable_cb.setChecked(state["filter_enable"])
        if "filter_type" in state:
            idx = self.filter_type_combo.findText(state["filter_type"])
            if idx >= 0: self.filter_type_combo.setCurrentIndex(idx)
        if "filter_cutoff" in state:
            self.filter_cutoff.setValue(state["filter_cutoff"])
        if "filter_rolloff" in state:
            self.filter_rolloff.setValue(state["filter_rolloff"])
        if "unwrap_method" in state:
            idx = self.unwrap_method_combo.findData(state["unwrap_method"])
            if idx >= 0: self.unwrap_method_combo.setCurrentIndex(idx)
        for key, widget in [
            ("unwrap_pre_median", self.unwrap_pre_median),
            ("unwrap_pre_gaussian", self.unwrap_pre_gaussian),
            ("unwrap_post_outlier", self.unwrap_post_outlier),
            ("unwrap_post_bg", self.unwrap_post_bg),
        ]:
            if key in state:
                widget.setChecked(state[key])
        for key, widget in [
            ("unwrap_pre_median_size", self.unwrap_pre_median_size),
            ("unwrap_post_bg_order", self.unwrap_post_bg_order),
        ]:
            if key in state:
                widget.setValue(state[key])
        for key, widget in [
            ("unwrap_pre_gauss_sigma", self.unwrap_pre_gauss_sigma),
            ("unwrap_post_outlier_sigma", self.unwrap_post_outlier_sigma),
            ("tie_dz_um", self.tie_dz),
            ("optical_synth_nm", self.optical_dlam),
        ]:
            if key in state:
                widget.setValue(state[key])
        if "contrast" in state:
            self.contrast.set_state(state["contrast"])
