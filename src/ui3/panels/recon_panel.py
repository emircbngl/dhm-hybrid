"""ReconPanel — the rich reconstruct panel (ui3).

The full, dockable counterpart to the spine's inline "Reconstruct" control
dock (``MainWindow._build_control_dock`` in ``ui3/main_window.py``). Where the
inline dock exposes six fields (wavelength / pixel / z / mask / method /
reference), this panel surfaces **every** field on
``ui2.reconstruction.ReconParams`` — basic physics stays inline, everything
else lives in a collapsible "Advanced" group, mirroring ui2's sidebar split
(``src/ui2/app.py`` ``section_params`` vs. the ``adv_header`` collapsing
block).

Behaviour parity notes (ui2 -> ui3, unchanged physics/logic):

* Reference mode is the 3-way Off / Reference / Reference-free combo from
  ``ui2.app.DhmApp._on_reference_mode_changed``. The back-compat rule lives on
  ``ReconParams.effective_reference_mode()`` itself (not duplicated here): an
  explicit "Off" pick must also disarm the legacy ``subtract_reference`` flag,
  because that flag is ``effective_reference_mode()``'s back-compat input —
  leaving it armed would silently re-resolve "Off" back to "reference" (the
  exact bug fixed in the 2026-07-05 review, see ``ui2/app.py`` around
  ``_on_reference_mode_changed``). This panel reproduces that same disarm.
* Reference-free sub-section (bg method / bg order / n-terms / CNN) only
  shows when reference_mode == "reference_free" — same visibility rule as
  ``section_reffree`` in ``ui2/app.py``. The CNN checkbox is gated on
  ``ui2.workers.reffree_cnn_available()`` (torch importable + trained
  checkpoint present) exactly like the ui2 sidebar; disabled with the same
  "requires torch + trained model" tooltip when unavailable.
* Load/Clear reference use this panel's own ``QFileDialog`` (ui2's
  ``_on_load_reference`` / ``_on_clear_reference``) — picking a file sets
  ``params.reference_path`` and switches mode to "reference"; clearing drops
  ``reference_path`` and, if the mode was "reference", falls back to "off"
  (a reference_free selection is unrelated to the file and survives, same as
  ui2's ``_on_clear_reference``).
* Presets (Cell/Film/USAF/Custom) mirror ``ui2.app.DhmApp._builtin_presets``
  exactly (same field values). Save/Delete only manage a panel-local dict —
  persistence is explicitly out of scope per the panel spec ("kalici depolama
  ctx disi ... asma").
* ``unwrap_method`` options come from ``core.phase_unwrap.UnwrapMethod`` so
  the combo can never drift from the enum the compute layer accepts.

Every control writes into the SAME live ``ReconParams`` instance
(``ctx.get_params()``) via a single ``_sync_params_to_ctx`` — mutate + call
``ctx.params_changed()`` — so this panel and any other mounted panel (or the
spine's own inline dock, when both are present) stay coherent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.phase_unwrap import UnwrapMethod
from core.drivers.workers import reffree_cnn_available
from ui3.context import PanelContext
from ui3.design import Space

_LOG = logging.getLogger("ui3.panels.recon")

# Display label <-> ReconParams.reference_mode value — same mapping as
# ui2.app.DhmApp._REFERENCE_MODE_LABELS / _REFERENCE_MODE_VALUES.
_REFERENCE_MODE_LABELS: dict = {
    "off": "Off", "reference": "Reference", "reference_free": "Reference-free",
}
_REFERENCE_MODE_VALUES: dict = {v: k for k, v in _REFERENCE_MODE_LABELS.items()}

_FFT_BACKENDS = ("auto", "pyfftw", "mlx", "scipy", "numpy")

_BUILTIN_PRESET_NAMES = ("Cell", "Film", "USAF", "Custom")


def _builtin_presets() -> dict:
    """Same values as ``ui2.app.DhmApp._builtin_presets`` — kept identical
    so a preset picked here behaves exactly as it would in ui2."""
    return {
        "Cell": dict(wavelength_nm=632.8, pixel_um=3.45,
                     z_mm=10.0, mask_radius=80, method="ASM",
                     magnification=40.0, pixel_is_effective=False,
                     n_sample=1.38, n_medium=1.337,
                     autofocus_metric="LAPLACIAN_VARIANCE"),
        "Film": dict(wavelength_nm=532.0, pixel_um=5.5,
                     z_mm=50.0, mask_radius=100, method="Fresnel",
                     magnification=1.0, pixel_is_effective=True,
                     n_sample=1.50, n_medium=1.000,
                     autofocus_metric="LAPLACIAN_VARIANCE"),
        "USAF": dict(wavelength_nm=632.8, pixel_um=2.2,
                     z_mm=5.0, mask_radius=60, method="ASM",
                     magnification=10.0, pixel_is_effective=False,
                     n_sample=1.50, n_medium=1.000,
                     autofocus_metric="TENENGRAD"),
        "Custom": dict(wavelength_nm=632.8, pixel_um=5.0,
                       z_mm=10.0, mask_radius=40, method="ASM",
                       magnification=1.0, pixel_is_effective=True,
                       n_sample=1.38, n_medium=1.337,
                       autofocus_metric="LAPLACIAN_VARIANCE"),
    }


class ReconPanel(QWidget):
    """Full reconstruct controls: every ``ReconParams`` field, presets,
    reference-mode handling, Reconstruct/Autofocus actions."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        # Panel-local preset store — built-ins plus any user-saved ones for
        # this session. Persistence is intentionally out of scope (spec:
        # "kalici depolama ctx disi ... asma").
        self._user_presets: dict = {}
        self._suspend_sync = False

        root = QVBoxLayout(self)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.md)
        root.setSpacing(Space.md)

        heading = QLabel("Reconstruct")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        root.addWidget(self._build_file_group())
        root.addWidget(self._build_preset_group())
        root.addWidget(self._build_basic_group())
        root.addWidget(self._build_reference_group())
        root.addWidget(self._build_advanced_group())
        root.addLayout(self._build_action_row())
        root.addStretch(1)

        self._load_params_into_controls()
        self._wire_bridge()

    # ------------------------------------------------------------------
    # File / hologram
    # ------------------------------------------------------------------
    def _build_file_group(self) -> QGroupBox:
        box = QGroupBox("Hologram")
        lay = QVBoxLayout(box)
        self.lbl_file = QLabel("(no hologram loaded)")
        self.lbl_file.setProperty("role", "muted")
        self.lbl_file.setWordWrap(True)
        lay.addWidget(self.lbl_file)
        self._refresh_file_label()
        return box

    def _refresh_file_label(self) -> None:
        path = self._ctx.hologram_path()
        if path:
            self.lbl_file.setText(str(Path(path).name))
        else:
            self.lbl_file.setText("(no hologram loaded)")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _presets(self) -> dict:
        built_in = _builtin_presets()
        user = {n: c for n, c in self._user_presets.items() if n not in built_in}
        return {**built_in, **user}

    def _build_preset_group(self) -> QGroupBox:
        box = QGroupBox("Preset")
        outer = QVBoxLayout(box)

        chip_row = QHBoxLayout()
        self.cmb_preset = QComboBox()
        self._reload_preset_combo()
        self.cmb_preset.currentTextChanged.connect(self._on_preset_selected)
        chip_row.addWidget(self.cmb_preset, 1)
        outer.addLayout(chip_row)

        btn_row = QHBoxLayout()
        self.btn_save_preset = QPushButton("Save preset…")
        self.btn_save_preset.clicked.connect(self._on_save_preset_clicked)
        self.btn_delete_preset = QPushButton("Delete…")
        self.btn_delete_preset.clicked.connect(self._on_delete_preset_clicked)
        btn_row.addWidget(self.btn_save_preset)
        btn_row.addWidget(self.btn_delete_preset)
        outer.addLayout(btn_row)

        return box

    def _reload_preset_combo(self) -> None:
        current = self.cmb_preset.currentText() if hasattr(self, "cmb_preset") else ""
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        self.cmb_preset.addItems(list(self._presets().keys()))
        if current:
            idx = self.cmb_preset.findText(current)
            if idx >= 0:
                self.cmb_preset.setCurrentIndex(idx)
        self.cmb_preset.blockSignals(False)

    def _on_preset_selected(self, name: str) -> None:
        cfg = self._presets().get(name)
        if not cfg:
            return
        params = self._ctx.get_params()
        for key, value in cfg.items():
            if hasattr(params, key):
                setattr(params, key, value)
        self._ctx.params_changed()
        self._load_params_into_controls()
        self._ctx.set_status(f"Preset applied: {name}", "info")

    def _on_save_preset_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        name = str(name or "").strip()
        if not ok or not name:
            return
        if name in _BUILTIN_PRESET_NAMES:
            QMessageBox.warning(
                self, "Save preset",
                f"'{name}' is a built-in name — pick something else.")
            return
        if name in self._user_presets:
            reply = QMessageBox.question(
                self, "Replace preset",
                f"A preset named '{name}' already exists. Replace it?")
            if reply != QMessageBox.Yes:
                return
        params = self._ctx.get_params()
        self._user_presets[name] = dict(
            wavelength_nm=float(params.wavelength_nm),
            pixel_um=float(params.pixel_um),
            z_mm=float(params.z_mm),
            mask_radius=int(params.mask_radius),
            method=str(params.method),
            magnification=float(params.magnification),
            pixel_is_effective=bool(params.pixel_is_effective),
            n_sample=float(params.n_sample),
            n_medium=float(params.n_medium),
            autofocus_metric=str(params.autofocus_metric),
        )
        self._reload_preset_combo()
        idx = self.cmb_preset.findText(name)
        if idx >= 0:
            self.cmb_preset.setCurrentIndex(idx)
        self._ctx.set_status(f"Preset saved: {name}", "ok")
        self._ctx.toast(f"Preset saved: {name}", "ok")

    def _on_delete_preset_clicked(self) -> None:
        names = [n for n in self._user_presets if n not in _BUILTIN_PRESET_NAMES]
        if not names:
            self._ctx.set_status("No user presets to delete.", "info")
            return
        name, ok = QInputDialog.getItem(
            self, "Delete preset", "Preset:", names, 0, False)
        if not ok or not name:
            return
        self._user_presets.pop(name, None)
        self._reload_preset_combo()
        self._ctx.set_status(f"Preset deleted: {name}", "info")

    # ------------------------------------------------------------------
    # Basic parameters
    # ------------------------------------------------------------------
    def _build_basic_group(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        form = QFormLayout(box)

        self.sp_wavelength = QDoubleSpinBox()
        self.sp_wavelength.setRange(100.0, 2000.0)
        self.sp_wavelength.setDecimals(2)
        self.sp_wavelength.setSingleStep(10.0)
        self.sp_wavelength.setSuffix(" nm")
        self.sp_wavelength.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("Wavelength", self.sp_wavelength)

        self.sp_pixel = QDoubleSpinBox()
        self.sp_pixel.setRange(0.01, 100.0)
        self.sp_pixel.setDecimals(3)
        self.sp_pixel.setSingleStep(0.1)
        self.sp_pixel.setSuffix(" µm")
        self.sp_pixel.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("Pixel size", self.sp_pixel)

        self.sp_magnification = QDoubleSpinBox()
        self.sp_magnification.setRange(0.01, 1000.0)
        self.sp_magnification.setDecimals(2)
        self.sp_magnification.setSingleStep(1.0)
        self.sp_magnification.setSuffix(" ×")
        self.sp_magnification.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("Magnification", self.sp_magnification)

        self.cb_pixel_is_effective = QCheckBox("Pixel is already effective")
        self.cb_pixel_is_effective.setToolTip(
            "Check when the pixel size above is already divided by "
            "magnification. Unchecked: effective pixel = pixel / M.")
        self.cb_pixel_is_effective.toggled.connect(self._sync_params_to_ctx)
        form.addRow("", self.cb_pixel_is_effective)

        self.sp_z = QDoubleSpinBox()
        self.sp_z.setRange(-1000.0, 1000.0)
        self.sp_z.setDecimals(4)
        self.sp_z.setSingleStep(0.5)
        self.sp_z.setSuffix(" mm")
        self.sp_z.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("Distance z", self.sp_z)

        self.sp_mask = QSpinBox()
        self.sp_mask.setRange(5, 2048)
        self.sp_mask.setSingleStep(5)
        self.sp_mask.setSuffix(" px")
        self.sp_mask.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("Mask radius", self.sp_mask)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["ASM", "Fresnel"])
        self.cmb_method.currentTextChanged.connect(self._sync_params_to_ctx)
        form.addRow("Method", self.cmb_method)

        return box

    # ------------------------------------------------------------------
    # Reference mode
    # ------------------------------------------------------------------
    def _build_reference_group(self) -> QGroupBox:
        box = QGroupBox("Reference")
        outer = QVBoxLayout(box)

        self.lbl_reference_path = QLabel("(none)")
        self.lbl_reference_path.setProperty("role", "muted")
        self.lbl_reference_path.setWordWrap(True)
        outer.addWidget(self.lbl_reference_path)

        btn_row = QHBoxLayout()
        self.btn_load_reference = QPushButton("Load reference…")
        self.btn_load_reference.clicked.connect(self._on_load_reference_clicked)
        self.btn_clear_reference = QPushButton("Clear reference")
        self.btn_clear_reference.clicked.connect(self._on_clear_reference_clicked)
        btn_row.addWidget(self.btn_load_reference)
        btn_row.addWidget(self.btn_clear_reference)
        outer.addLayout(btn_row)

        mode_form = QFormLayout()
        self.cmb_reference_mode = QComboBox()
        self.cmb_reference_mode.addItems(["Off", "Reference", "Reference-free"])
        self.cmb_reference_mode.currentTextChanged.connect(
            self._on_reference_mode_changed)
        self.cmb_reference_mode.setToolTip(
            "Off: no background handling. Reference: divide by a loaded "
            "reference hologram. Reference-free: fit and subtract a "
            "numerical background (no reference needed).")
        mode_form.addRow("Mode", self.cmb_reference_mode)
        outer.addLayout(mode_form)

        # Reference-free sub-section — only visible in that mode.
        self.grp_reffree = QGroupBox("Reference-free background fit")
        rf_form = QFormLayout(self.grp_reffree)

        self.cmb_reffree_bg_method = QComboBox()
        self.cmb_reffree_bg_method.addItems(["zernike", "polynomial"])
        self.cmb_reffree_bg_method.currentTextChanged.connect(
            self._sync_params_to_ctx)
        rf_form.addRow("BG method", self.cmb_reffree_bg_method)

        self.sp_reffree_bg_order = QSpinBox()
        self.sp_reffree_bg_order.setRange(1, 6)
        self.sp_reffree_bg_order.setToolTip(
            "Polynomial total order (only used when bg method=polynomial).")
        self.sp_reffree_bg_order.valueChanged.connect(self._sync_params_to_ctx)
        rf_form.addRow("BG order", self.sp_reffree_bg_order)

        self.sp_reffree_n_terms = QSpinBox()
        self.sp_reffree_n_terms.setRange(1, 100)
        self.sp_reffree_n_terms.setToolTip(
            "Zernike Noll term count (only used when bg method=zernike).")
        self.sp_reffree_n_terms.valueChanged.connect(self._sync_params_to_ctx)
        rf_form.addRow("N terms", self.sp_reffree_n_terms)

        cnn_available = reffree_cnn_available()
        self.cb_reffree_cnn = QCheckBox("CNN residual correction")
        self.cb_reffree_cnn.setEnabled(cnn_available)
        self.cb_reffree_cnn.setToolTip(
            "requires torch + trained model" if not cnn_available else
            "Runs the Track C CNN residual corrector on top of the "
            "classical background fit.")
        self.cb_reffree_cnn.toggled.connect(self._sync_params_to_ctx)
        rf_form.addRow("", self.cb_reffree_cnn)

        outer.addWidget(self.grp_reffree)

        return box

    def _on_load_reference_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference hologram", "",
            "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;All files (*)")
        if not path:
            return
        params = self._ctx.get_params()
        params.reference_path = Path(path)
        params.reference_mode = "reference"
        params.subtract_reference = True
        self._ctx.params_changed()
        self._load_params_into_controls()
        self._ctx.set_status(f"Reference loaded: {Path(path).name}", "ok")
        self._ctx.toast("Reference set — next Reconstruct will divide it out.", "info")

    def _on_clear_reference_clicked(self) -> None:
        params = self._ctx.get_params()
        # Clearing the file drops "reference" mode back to Off; an active
        # reference_free choice is unrelated to the file and survives —
        # mirrors ui2.app.DhmApp._on_clear_reference exactly.
        mode = params.reference_mode
        if params.effective_reference_mode() == "reference":
            mode = "off"
        params.reference_path = None
        params.subtract_reference = False
        params.reference_mode = mode
        self._ctx.params_changed()
        self._load_params_into_controls()
        self._ctx.set_status("Reference cleared.", "info")

    def _on_reference_mode_changed(self, label: str) -> None:
        if self._suspend_sync:
            return
        mode = _REFERENCE_MODE_VALUES.get(label, "off")
        params = self._ctx.get_params()
        # Explicit "Off" must also disarm the legacy subtract_reference
        # flag — it's effective_reference_mode()'s back-compat input, so a
        # stale True would silently re-resolve "Off" back to "reference"
        # (2026-07-05 review fix, mirrored here).
        params.reference_mode = mode
        params.subtract_reference = (mode == "reference")
        self._ctx.params_changed()
        self.grp_reffree.setVisible(mode == "reference_free")
        if mode == "reference" and params.reference_path is None:
            # Must NOT be overwritten by the info line below — otherwise the
            # operator never sees that reference mode is armed with no
            # reference, and the reconstruction silently comes out
            # unreferenced (2026-07-08 review).
            self._ctx.set_status(
                "Reference mode selected but no reference is loaded "
                "(Load reference…).", "warn")
        else:
            self._ctx.set_status(f"Reference mode: {label}", "info")

    # ------------------------------------------------------------------
    # Advanced
    # ------------------------------------------------------------------
    def _build_advanced_group(self) -> QGroupBox:
        box = QGroupBox("Advanced")
        box.setCheckable(True)
        box.setChecked(False)
        outer = QVBoxLayout(box)
        inner = QWidget()
        form = QFormLayout(inner)
        outer.addWidget(inner)
        inner.setVisible(False)
        # Collapsed by default: drop the card border/body so it reads as just a
        # disclosure row, not an orphaned empty frame (the [collapsed] property
        # is styled in design.build_qss). Repolish on toggle to swap the look.
        box.setProperty("collapsed", True)

        def _toggle_advanced(checked: bool) -> None:
            inner.setVisible(checked)
            box.setProperty("collapsed", not checked)
            box.style().unpolish(box)
            box.style().polish(box)

        box.toggled.connect(_toggle_advanced)

        # Optical path.
        self.cmb_optical_mode = QComboBox()
        self.cmb_optical_mode.addItems(["transmission", "reflection"])
        self.cmb_optical_mode.setToolTip(
            "Transmission: light passes through the sample once. "
            "Reflection: light traverses the surface height twice — "
            "OPD is halved before height conversion.")
        self.cmb_optical_mode.currentTextChanged.connect(self._sync_params_to_ctx)
        form.addRow("Optical path", self.cmb_optical_mode)

        # QPI refractive indices.
        self.sp_n_sample = QDoubleSpinBox()
        self.sp_n_sample.setRange(1.0, 3.0)
        self.sp_n_sample.setDecimals(3)
        self.sp_n_sample.setSingleStep(0.01)
        self.sp_n_sample.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("n sample", self.sp_n_sample)

        self.sp_n_medium = QDoubleSpinBox()
        self.sp_n_medium.setRange(1.0, 3.0)
        self.sp_n_medium.setDecimals(3)
        self.sp_n_medium.setSingleStep(0.01)
        self.sp_n_medium.setToolTip(
            "Refractive index of the surrounding medium (water ≈ 1.337, "
            "air ≈ 1.000). Drives QPI cell-height and dry-mass calculations.")
        self.sp_n_medium.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("n medium", self.sp_n_medium)

        # Autofocus metric + range (also editable from the Focus panel;
        # kept here since it's a ReconParams field like everything else).
        self.cmb_af_metric = QComboBox()
        self.cmb_af_metric.addItems(_focus_metric_names())
        self.cmb_af_metric.currentTextChanged.connect(self._sync_params_to_ctx)
        form.addRow("Autofocus metric", self.cmb_af_metric)

        self.sp_af_z_min = QDoubleSpinBox()
        self.sp_af_z_min.setRange(-1000.0, 1000.0)
        self.sp_af_z_min.setDecimals(3)
        self.sp_af_z_min.setSingleStep(0.5)
        self.sp_af_z_min.setSuffix(" mm")
        self.sp_af_z_min.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("AF z min", self.sp_af_z_min)

        self.sp_af_z_max = QDoubleSpinBox()
        self.sp_af_z_max.setRange(-1000.0, 1000.0)
        self.sp_af_z_max.setDecimals(3)
        self.sp_af_z_max.setSingleStep(0.5)
        self.sp_af_z_max.setSuffix(" mm")
        self.sp_af_z_max.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("AF z max", self.sp_af_z_max)

        self.sp_af_n_steps = QSpinBox()
        self.sp_af_n_steps.setRange(5, 5000)
        self.sp_af_n_steps.setSingleStep(5)
        self.sp_af_n_steps.valueChanged.connect(self._sync_params_to_ctx)
        form.addRow("AF steps", self.sp_af_n_steps)

        # Pre/post-processing + engine selection.
        self.cb_subtract_mean = QCheckBox("Subtract mean")
        self.cb_subtract_mean.setToolTip(
            "Removes DC bias from the hologram before the off-axis mask.")
        self.cb_subtract_mean.toggled.connect(self._sync_params_to_ctx)
        form.addRow("", self.cb_subtract_mean)

        self.cb_hann_window = QCheckBox("Hann window")
        self.cb_hann_window.setToolTip(
            "Applies a 2D Hann envelope to the raw frame — suppresses "
            "edge discontinuities that leak into the FFT.")
        self.cb_hann_window.toggled.connect(self._sync_params_to_ctx)
        form.addRow("", self.cb_hann_window)

        self.cmb_fft_backend = QComboBox()
        self.cmb_fft_backend.addItems(list(_FFT_BACKENDS))
        self.cmb_fft_backend.currentTextChanged.connect(self._sync_params_to_ctx)
        form.addRow("FFT backend", self.cmb_fft_backend)

        self.cmb_unwrap_method = QComboBox()
        self.cmb_unwrap_method.addItems([m.name for m in UnwrapMethod])
        self.cmb_unwrap_method.currentTextChanged.connect(self._sync_params_to_ctx)
        form.addRow("Unwrap method", self.cmb_unwrap_method)

        self.cb_auto_contrast_amp = QCheckBox("Auto-contrast amplitude preview")
        self.cb_auto_contrast_amp.setToolTip(
            "Percentile-based contrast stretch on the amplitude preview "
            "(display only — scientific values untouched).")
        self.cb_auto_contrast_amp.toggled.connect(self._sync_params_to_ctx)
        form.addRow("", self.cb_auto_contrast_amp)

        return box

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_reconstruct = QPushButton("Reconstruct")
        self.btn_reconstruct.setProperty("role", "primary")
        self.btn_reconstruct.clicked.connect(self._on_reconstruct_clicked)
        self.btn_autofocus = QPushButton("Autofocus")
        self.btn_autofocus.clicked.connect(self._on_autofocus_clicked)
        row.addWidget(self.btn_reconstruct)
        row.addWidget(self.btn_autofocus)
        return row

    def _on_reconstruct_clicked(self) -> None:
        path = self._ctx.hologram_path()
        if not path:
            self._ctx.set_status("Load a hologram before reconstructing.", "warn")
            self._ctx.toast("Load a hologram first.", "warn")
            return
        params = self._ctx.get_params()
        self._ctx.set_status("Reconstructing…", "info")
        self._ctx.bridge.reconstruct(path, params)

    def _on_autofocus_clicked(self) -> None:
        path = self._ctx.hologram_path()
        if not path:
            self._ctx.set_status("Load a hologram before autofocusing.", "warn")
            self._ctx.toast("Load a hologram first.", "warn")
            return
        params = self._ctx.get_params()
        self._ctx.set_status("Autofocusing…", "info")
        self._ctx.bridge.autofocus(
            path, params,
            z_min_mm=float(params.af_z_min_mm),
            z_max_mm=float(params.af_z_max_mm),
            n_steps=int(params.af_n_steps),
        )

    # ------------------------------------------------------------------
    # Bridge wiring
    # ------------------------------------------------------------------
    def _wire_bridge(self) -> None:
        # Status/toast ownership (2026-07-06, B-083): recon status is the
        # shell's (it paints the viewports and carries the reffree note);
        # this panel keeps the completion toast. Autofocus status/toast/
        # readout belong to FocusPanel — here we only keep the control
        # refresh so the panel stays correct standalone. Recon errors are
        # owned here (the shell no longer duplicates them); autofocus errors
        # are FocusPanel's.
        bridge = self._ctx.bridge
        bridge.recon_done.connect(self._on_recon_done)
        bridge.recon_error.connect(self._on_recon_error)
        bridge.autofocus_done.connect(self._on_autofocus_done)
        bridge.busy_changed.connect(self._on_busy_changed)

    def _on_recon_done(self, result) -> None:
        self._ctx.toast("Reconstruction done.", "ok")

    def _on_recon_error(self, message: str) -> None:
        self._ctx.set_status(f"Reconstruction failed: {message}", "error")
        self._ctx.toast(f"Reconstruction failed: {message}", "error")

    def _on_autofocus_done(self, result) -> None:
        best_z_mm = float(result.best_z_m) * 1e3
        params = self._ctx.get_params()
        params.z_mm = best_z_mm
        self._ctx.params_changed()
        self._load_params_into_controls()

    def _on_busy_changed(self, busy: bool, _label: str) -> None:
        self.btn_reconstruct.setEnabled(not busy)
        self.btn_autofocus.setEnabled(not busy)

    # ------------------------------------------------------------------
    # Param sync (panel <-> ReconParams), single direction each way
    # ------------------------------------------------------------------
    def _load_params_into_controls(self) -> None:
        """Push ``ctx.get_params()`` onto every widget. Signals are
        suspended so this never bounces back into ``_sync_params_to_ctx``."""
        self._suspend_sync = True
        try:
            params = self._ctx.get_params()

            self._refresh_file_label()

            self.sp_wavelength.setValue(float(params.wavelength_nm))
            self.sp_pixel.setValue(float(params.pixel_um))
            self.sp_magnification.setValue(float(params.magnification))
            self.cb_pixel_is_effective.setChecked(bool(params.pixel_is_effective))
            self.sp_z.setValue(float(params.z_mm))
            self.sp_mask.setValue(int(params.mask_radius))
            idx = self.cmb_method.findText(str(params.method))
            self.cmb_method.setCurrentIndex(idx if idx >= 0 else 0)

            # Reference.
            if params.reference_path is not None:
                self.lbl_reference_path.setText(Path(params.reference_path).name)
            else:
                self.lbl_reference_path.setText("(none)")
            mode = params.effective_reference_mode()
            label = _REFERENCE_MODE_LABELS.get(mode, "Off")
            midx = self.cmb_reference_mode.findText(label)
            self.cmb_reference_mode.setCurrentIndex(midx if midx >= 0 else 0)
            self.grp_reffree.setVisible(mode == "reference_free")

            bg_idx = self.cmb_reffree_bg_method.findText(str(params.reffree_bg_method))
            self.cmb_reffree_bg_method.setCurrentIndex(bg_idx if bg_idx >= 0 else 0)
            self.sp_reffree_bg_order.setValue(int(params.reffree_bg_order))
            self.sp_reffree_n_terms.setValue(int(params.reffree_n_terms))
            self.cb_reffree_cnn.setChecked(
                bool(params.reffree_cnn) and reffree_cnn_available())

            # Advanced.
            om_idx = self.cmb_optical_mode.findText(str(params.optical_mode))
            self.cmb_optical_mode.setCurrentIndex(om_idx if om_idx >= 0 else 0)
            self.sp_n_sample.setValue(float(params.n_sample))
            self.sp_n_medium.setValue(float(params.n_medium))
            am_idx = self.cmb_af_metric.findText(str(params.autofocus_metric))
            self.cmb_af_metric.setCurrentIndex(am_idx if am_idx >= 0 else 0)
            self.sp_af_z_min.setValue(float(params.af_z_min_mm))
            self.sp_af_z_max.setValue(float(params.af_z_max_mm))
            self.sp_af_n_steps.setValue(int(params.af_n_steps))
            self.cb_subtract_mean.setChecked(bool(params.subtract_mean))
            self.cb_hann_window.setChecked(bool(params.hann_window))
            fb_idx = self.cmb_fft_backend.findText(str(params.fft_backend))
            self.cmb_fft_backend.setCurrentIndex(fb_idx if fb_idx >= 0 else 0)
            um_idx = self.cmb_unwrap_method.findText(str(params.unwrap_method))
            self.cmb_unwrap_method.setCurrentIndex(um_idx if um_idx >= 0 else 0)
            self.cb_auto_contrast_amp.setChecked(bool(params.auto_contrast_amplitude))
        finally:
            self._suspend_sync = False

    def _sync_params_to_ctx(self, *_args) -> None:
        """Single write path: every control callback lands here. Mutates
        the live ``ReconParams`` then calls ``ctx.params_changed()``."""
        if self._suspend_sync:
            return
        params = self._ctx.get_params()

        params.wavelength_nm = float(self.sp_wavelength.value())
        params.pixel_um = float(self.sp_pixel.value())
        params.magnification = float(self.sp_magnification.value())
        params.pixel_is_effective = bool(self.cb_pixel_is_effective.isChecked())
        params.z_mm = float(self.sp_z.value())
        params.mask_radius = int(self.sp_mask.value())
        params.method = str(self.cmb_method.currentText())

        params.reffree_bg_method = str(self.cmb_reffree_bg_method.currentText())
        params.reffree_bg_order = int(self.sp_reffree_bg_order.value())
        params.reffree_n_terms = int(self.sp_reffree_n_terms.value())
        params.reffree_cnn = bool(
            self.cb_reffree_cnn.isChecked() and reffree_cnn_available())

        params.optical_mode = str(self.cmb_optical_mode.currentText())
        params.n_sample = float(self.sp_n_sample.value())
        params.n_medium = float(self.sp_n_medium.value())
        params.autofocus_metric = str(self.cmb_af_metric.currentText())
        params.af_z_min_mm = float(self.sp_af_z_min.value())
        params.af_z_max_mm = float(self.sp_af_z_max.value())
        params.af_n_steps = int(self.sp_af_n_steps.value())
        params.subtract_mean = bool(self.cb_subtract_mean.isChecked())
        params.hann_window = bool(self.cb_hann_window.isChecked())
        params.fft_backend = str(self.cmb_fft_backend.currentText())
        params.unwrap_method = str(self.cmb_unwrap_method.currentText())
        params.auto_contrast_amplitude = bool(self.cb_auto_contrast_amp.isChecked())

        self._ctx.params_changed()


def _focus_metric_names() -> list[str]:
    """Same indirection idea as ``ui2.app._available_focus_metric_names`` —
    isolates the sidebar from the enum so it stays buildable even if
    ``core.autofocus`` import fails in a stub test environment."""
    try:
        from core.autofocus import FocusMetric
        return [m.name for m in FocusMetric]
    except Exception:
        return ["LAPLACIAN_VARIANCE"]


__all__ = ["ReconPanel"]
