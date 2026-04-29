from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
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
)
from core.reconstruction import ReconstructionMethod
from gui.widgets.collapsible_box import CollapsibleBox
from gui.widgets.preset_chips import PresetChipRow
from gui.widgets.validation_dot import ValidationDot


class ReconTab(QWidget):
    """Sidebar tab — reconstruction parameters, propagation, and masking."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ─── Sample Preset ───
        preset_group = QGroupBox("Sample Preset")
        preset_lay = QGridLayout(preset_group)
        preset_lay.setSpacing(6)
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip(
            "Quick-configure reconstruction for your sample type.\n"
            "Biological: ASM, 632.8 nm, small z-range, suited for cells in medium.\n"
            "Thin Film: Fresnel, 532 nm, larger z-range, suited for reflective samples.\n"
            "Custom: Keep current settings."
        )
        self.preset_combo.addItem("Custom")
        self.preset_combo.addItem("Biological Cell")
        self.preset_combo.addItem("Thin Film / Reflective")
        self.preset_combo.addItem("USAF Target")
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_lay.addWidget(QLabel("Preset"), 0, 0)
        preset_lay.addWidget(self.preset_combo, 0, 1)

        # v1.4 chip row — see _on_preset_chip handler.
        self.preset_chips = PresetChipRow(
            [self.preset_combo.itemText(i)
             for i in range(self.preset_combo.count())],
            active=self.preset_combo.currentText(),
        )
        self.preset_chips.preset_selected.connect(self._on_preset_chip)
        preset_lay.addWidget(self.preset_chips, 1, 0, 1, 2)
        layout.addWidget(preset_group)

        # ─── Reconstruction — essentials only (v1.4 progressive disclosure) ───
        #
        # Every knob the operator touches on a typical run lives here;
        # propagation algorithm / FFT backend / magnification options
        # moved into the collapsible Advanced section below so the
        # default view stays scannable. Widget *instances* are still
        # attached to ``self`` so every downstream consumer
        # (``get_state``, ``set_state``, persistence, main_window
        # reads) keeps working unchanged.
        recon_group = QGroupBox("Reconstruction")
        rg = QGridLayout(recon_group)
        rg.setSpacing(6)
        row = 0

        self.wavelength_nm = QDoubleSpinBox()
        self.wavelength_nm.setRange(100.0, 2000.0)
        self.wavelength_nm.setDecimals(2)
        self.wavelength_nm.setValue(632.8)
        self.wavelength_nm.setSuffix(" nm")
        self.wavelength_nm.setToolTip("Laser wavelength")
        self.wavelength_nm.setAccessibleName("Laser wavelength (nm)")
        self.wavelength_dot = ValidationDot()
        rg.addWidget(QLabel("Wavelength"), row, 0)
        rg.addWidget(self.wavelength_nm, row, 1)
        rg.addWidget(self.wavelength_dot, row, 2); row += 1

        self.pixel_um = QDoubleSpinBox()
        self.pixel_um.setRange(0.01, 1000.0)
        self.pixel_um.setDecimals(4)
        self.pixel_um.setValue(1.0)
        self.pixel_um.setSuffix(" \u00b5m")
        self.pixel_um.setToolTip("Camera pixel pitch")
        self.pixel_um.setAccessibleName("Camera pixel pitch (microns)")
        self.pixel_dot = ValidationDot()
        rg.addWidget(QLabel("Pixel size"), row, 0)
        rg.addWidget(self.pixel_um, row, 1)
        rg.addWidget(self.pixel_dot, row, 2); row += 1

        # Live inline validation — each field's validity pushes through
        # to the adjacent dot.
        self.wavelength_nm.valueChanged.connect(self._revalidate_numeric_fields)
        self.pixel_um.valueChanged.connect(self._revalidate_numeric_fields)
        self._revalidate_numeric_fields()

        layout.addWidget(recon_group)

        # ─── Advanced (collapsed by default) ───
        # Method, FFT backend, optical magnification, and the
        # "pixel-is-effective" toggle sit here. These are set once per
        # setup and rarely touched during a session.
        self.advanced_box = CollapsibleBox("Advanced")
        advanced_layout = QGridLayout()
        advanced_layout.setSpacing(6)
        arow = 0

        self.method_combo = QComboBox()
        self.method_combo.setToolTip("Propagation algorithm")
        self.method_combo.addItem("ASM", ReconstructionMethod.ASM)
        self.method_combo.addItem("Fresnel", ReconstructionMethod.FRESNEL)
        advanced_layout.addWidget(QLabel("Method"), arow, 0)
        advanced_layout.addWidget(self.method_combo, arow, 1); arow += 1

        self.fft_backend_combo = QComboBox()
        self.fft_backend_combo.setToolTip("FFT computation backend")
        self.fft_backend_combo.addItem("PyFFTW", "pyfftw")
        self.fft_backend_combo.addItem("MLX (Metal)", "mlx")
        self.fft_backend_combo.addItem("Scipy", "scipy")
        self.fft_backend_combo.addItem("NumPy", "numpy")
        advanced_layout.addWidget(QLabel("FFT Backend"), arow, 0)
        advanced_layout.addWidget(self.fft_backend_combo, arow, 1); arow += 1

        self.magnification = QDoubleSpinBox()
        self.magnification.setRange(0.01, 1000.0)
        self.magnification.setDecimals(2)
        self.magnification.setValue(1.0)
        self.magnification.setPrefix("\u00d7")
        self.magnification.setToolTip("Objective magnification")
        advanced_layout.addWidget(QLabel("Magnification"), arow, 0)
        advanced_layout.addWidget(self.magnification, arow, 1); arow += 1

        self.pixel_is_effective_cb = QCheckBox("Pixel is already effective")
        self.pixel_is_effective_cb.setChecked(True)
        self.pixel_is_effective_cb.setToolTip(
            "If checked, pixel size = effective pixel size (no magnification division)"
        )
        advanced_layout.addWidget(self.pixel_is_effective_cb, arow, 0, 1, 2)
        arow += 1

        self.advanced_box.setContentLayout(advanced_layout)
        layout.addWidget(self.advanced_box)

        # ─── Propagation ───
        prop_group = QGroupBox("Propagation")
        pg = QGridLayout(prop_group)
        pg.setSpacing(6)

        self.z_mm = QDoubleSpinBox()
        self.z_mm.setRange(-1e6, 1e6)
        self.z_mm.setDecimals(4)
        self.z_mm.setValue(0.0)
        self.z_mm.setSingleStep(0.1)
        self.z_mm.setSuffix(" mm")
        self.z_mm.setToolTip("Reconstruction distance")
        pg.addWidget(QLabel("z"), 0, 0)
        pg.addWidget(self.z_mm, 0, 1)

        self.recon_btn = QPushButton("Reconstruct")
        self.recon_btn.setEnabled(False)
        self.recon_btn.setMinimumHeight(32)
        self.recon_btn.setToolTip("Run reconstruction with current parameters")
        pg.addWidget(self.recon_btn, 1, 0, 1, 2)
        layout.addWidget(prop_group)

        # ─── Masking ───
        mask_group = QGroupBox("Masking")
        mg = QGridLayout(mask_group)
        mg.setSpacing(6)
        self.mask_radius = QSpinBox()
        self.mask_radius.setRange(5, 500)
        self.mask_radius.setValue(80)
        self.mask_radius.setSuffix(" px")
        self.mask_radius.setToolTip("+1 diffraction order mask radius")
        mg.addWidget(QLabel("Mask radius"), 0, 0)
        mg.addWidget(self.mask_radius, 0, 1)
        layout.addWidget(mask_group)

        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _apply_preset(self, text: str):
        """Auto-configure reconstruction parameters for the selected sample type."""
        if text == "Custom":
            return

        PRESETS = {
            "Biological Cell": {
                "method": "ASM",
                "wavelength_nm": 632.8,
                "pixel_um": 3.45,
                "magnification": 20.0,
                "pixel_is_effective": False,
                "mask_radius": 80,
            },
            "Thin Film / Reflective": {
                "method": "Fresnel",
                "wavelength_nm": 532.0,
                "pixel_um": 3.45,
                "magnification": 10.0,
                "pixel_is_effective": False,
                "mask_radius": 100,
            },
            "USAF Target": {
                "method": "ASM",
                "wavelength_nm": 632.8,
                "pixel_um": 3.45,
                "magnification": 40.0,
                "pixel_is_effective": False,
                "mask_radius": 80,
            },
        }

        preset = PRESETS.get(text)
        if preset is None:
            return

        idx = self.method_combo.findText(preset["method"])
        if idx >= 0:
            self.method_combo.setCurrentIndex(idx)
        self.wavelength_nm.setValue(preset["wavelength_nm"])
        self.pixel_um.setValue(preset["pixel_um"])
        self.magnification.setValue(preset["magnification"])
        self.pixel_is_effective_cb.setChecked(preset["pixel_is_effective"])
        self.mask_radius.setValue(preset["mask_radius"])

    def get_state(self) -> dict:
        return {
            "method": self.method_combo.currentText(),
            "fft_backend": self.fft_backend_combo.currentText(),
            "wavelength_nm": self.wavelength_nm.value(),
            "magnification": self.magnification.value(),
            "pixel_um": self.pixel_um.value(),
            "pixel_is_effective": self.pixel_is_effective_cb.isChecked(),
            "z_mm": self.z_mm.value(),
            "mask_radius": self.mask_radius.value()
        }

    def set_state(self, state: dict):
        if "method" in state:
            idx = self.method_combo.findText(state["method"])
            if idx >= 0: self.method_combo.setCurrentIndex(idx)
        if "fft_backend" in state:
            idx = self.fft_backend_combo.findText(state["fft_backend"])
            if idx >= 0: self.fft_backend_combo.setCurrentIndex(idx)
        if "wavelength_nm" in state:
            self.wavelength_nm.setValue(state["wavelength_nm"])
        if "magnification" in state:
            self.magnification.setValue(state["magnification"])
        if "pixel_um" in state:
            self.pixel_um.setValue(state["pixel_um"])
        if "pixel_is_effective" in state:
            self.pixel_is_effective_cb.setChecked(state["pixel_is_effective"])
        if "z_mm" in state:
            self.z_mm.setValue(state["z_mm"])
        if "mask_radius" in state:
            self.mask_radius.setValue(state["mask_radius"])
        # State change may have rendered fields valid/invalid — refresh dots.
        self._revalidate_numeric_fields()

    # ─── Preset chip row (v1.4 UI Redesign) ─────────────────────────
    def _on_preset_chip(self, label: str) -> None:
        """Chip clicked → push through the combo so the existing
        ``_apply_preset`` handler runs once, exactly as before."""
        idx = self.preset_combo.findText(label)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    # ─── Validation dots (v1.4 inline validation) ─────────────────────
    def _revalidate_numeric_fields(self) -> None:
        """Feed the current wavelength / pixel size through
        :func:`core.settings_schema.validate` and flip the inline
        dots accordingly.

        The validator returns a list of human-readable problem strings;
        we map them to the field whose name appears in the message.
        Anything not matched stays green — the dot is a *local* signal,
        not a blanket form status.
        """
        try:
            from core.settings_schema import (
                AppSettings,
                ReconDefaults,
                validate,
            )
        except Exception:
            return

        try:
            recon = ReconDefaults(
                wavelength_nm=float(self.wavelength_nm.value()),
                pixel_um=float(self.pixel_um.value()),
                z_mm=float(self.z_mm.value()) if hasattr(self, "z_mm") else 0.0,
                mask_radius=int(self.mask_radius.value()) if hasattr(self, "mask_radius") else 80,
            )
            settings = AppSettings(recon=recon)
            problems = validate(settings)
        except Exception:
            return

        def _matching(field: str) -> str:
            for p in problems:
                if field in p:
                    return p
            return ""

        wv_reason = _matching("wavelength_nm")
        px_reason = _matching("pixel_um")

        self.wavelength_dot.setState(not wv_reason, reason=wv_reason)
        self.pixel_dot.setState(not px_reason, reason=px_reason)
