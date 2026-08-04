"""QPIPanel — quantitative phase imaging controls + readouts.

Mirrors ``ui2``'s QPI flow (``ui2.workers.ScienceDriver.run_qpi`` /
``run_qpi_batch``, both already wrapped by ``ui3.bridge.WorkerBridge``):

* One-shot QPI at the current ``params.z_mm`` — computes
  ``core.qpi.compute_qpi`` (via the shared driver) and displays the
  ``QPIResult`` bundled inside ``QPIOneShotResult`` as a mono readout
  grid: OPD range (nm), total dry mass (pg), step height (microstructure
  mode), and cell morphology summary (bio mode).
* QPI batch across the autofocus landscape — computes one QPI entry per
  focus candidate (``QPIBatchResultWrap.entries: List[QPIBatchEntry]``)
  and lists them in a table (z_mm, dry_mass_pg, area_um2, circularity).

No new physics here — this panel only calls the bridge and formats
whatever ``core.qpi.QPIResult`` already computed. Fields not present on
a given result (e.g. ``cell_morph`` is None in pure micro_structure mode)
render as "—" rather than raising.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui3.context import PanelContext


def _fmt(value: Optional[float], unit: str = "", digits: int = 3) -> str:
    """None-guarded numeric formatter — readouts show '—' rather than
    raising/crashing when a QPIResult field wasn't computed (e.g.
    ``step_height_m`` is None when no bimodal histogram peak was found,
    ``cell_morph`` is None outside bio mode)."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}g}{unit}"
    except (TypeError, ValueError):
        return "—"


class QPIPanel(QWidget):
    """QPI controls + readouts. Built against the frozen ``PanelContext``."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._last_batch_entries: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        heading = QLabel("Quantitative Phase Imaging")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        root.addWidget(self._build_controls_group())
        root.addWidget(self._build_oneshot_group())
        root.addWidget(self._build_batch_group(), 1)
        root.addStretch(0)

        self._load_params_into_controls()

        self.ctx.bridge.qpi_done.connect(self._on_qpi_done)
        self.ctx.bridge.qpi_error.connect(self._on_qpi_error)
        self.ctx.bridge.qpi_batch_done.connect(self._on_batch_done)
        self.ctx.bridge.qpi_batch_error.connect(self._on_batch_error)
        self.ctx.bridge.busy_changed.connect(self._on_busy_changed)

    # -- controls ---------------------------------------------------------

    def _build_controls_group(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("Mode"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Biological", "biological")
        self.mode_combo.addItem("Micro-structure", "micro_structure")
        self.mode_combo.addItem("Both", "both")
        self.mode_combo.setCurrentIndex(2)  # "both" — matches compute_qpi default
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        grid.addWidget(self.mode_combo, 0, 1)

        grid.addWidget(QLabel("n_sample"), 1, 0)
        self.n_sample_spin = QDoubleSpinBox()
        self.n_sample_spin.setRange(1.0, 2.5)
        self.n_sample_spin.setDecimals(4)
        self.n_sample_spin.setSingleStep(0.001)
        self.n_sample_spin.valueChanged.connect(self._on_n_sample_changed)
        grid.addWidget(self.n_sample_spin, 1, 1)

        grid.addWidget(QLabel("n_medium"), 2, 0)
        self.n_medium_spin = QDoubleSpinBox()
        self.n_medium_spin.setRange(1.0, 2.5)
        self.n_medium_spin.setDecimals(4)
        self.n_medium_spin.setSingleStep(0.001)
        self.n_medium_spin.valueChanged.connect(self._on_n_medium_changed)
        grid.addWidget(self.n_medium_spin, 2, 1)

        # Dry-mass specific refractive increment (alpha) — optional knob.
        # core.qpi.compute_qpi defaults to DEFAULT_ALPHA (0.18 mL/g,
        # protein) when the caller doesn't pass one; ui2's run_qpi driver
        # doesn't expose alpha as a ReconParams field, so this widget is
        # a display-only note rather than plumbed into the bridge call —
        # keeps this panel from inventing a param the driver doesn't
        # accept yet.
        grid.addWidget(QLabel("alpha (dry-mass, mL/g)"), 3, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 0.50)
        self.alpha_spin.setDecimals(3)
        self.alpha_spin.setSingleStep(0.005)
        self.alpha_spin.setValue(0.180)
        self.alpha_spin.setEnabled(False)
        self.alpha_spin.setToolTip(
            "Specific refractive increment used by core.qpi.compute_qpi "
            "(default 0.18 mL/g, protein). Not yet exposed on ReconParams "
            "— shown for reference; batch/one-shot runs use the core default."
        )
        grid.addWidget(self.alpha_spin, 3, 1)

        return box

    def _load_params_into_controls(self) -> None:
        params = self.ctx.get_params()
        self.n_sample_spin.blockSignals(True)
        self.n_medium_spin.blockSignals(True)
        self.n_sample_spin.setValue(float(getattr(params, "n_sample", 1.38)))
        self.n_medium_spin.setValue(float(getattr(params, "n_medium", 1.337)))
        self.n_sample_spin.blockSignals(False)
        self.n_medium_spin.blockSignals(False)

    def _on_mode_changed(self, _index: int) -> None:
        # Mode only affects how *this panel* reads the result (which
        # readouts it shows) — core.qpi.compute_qpi is always invoked
        # with QPIMode.BOTH by ui2.workers.run_qpi/run_qpi_batch, so
        # every field is already computed; we just choose what to show.
        self._render_last_result()

    def _on_n_sample_changed(self, value: float) -> None:
        params = self.ctx.get_params()
        params.n_sample = float(value)
        self.ctx.params_changed()

    def _on_n_medium_changed(self, value: float) -> None:
        params = self.ctx.get_params()
        params.n_medium = float(value)
        self.ctx.params_changed()

    # -- one-shot -----------------------------------------------------------

    def _build_oneshot_group(self) -> QGroupBox:
        box = QGroupBox("One-shot QPI")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute QPI")
        self.compute_btn.setProperty("role", "primary")
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        row.addWidget(self.compute_btn)
        row.addStretch(1)
        outer.addLayout(row)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self._readout_labels: dict[str, QLabel] = {}
        rows = [
            ("opd_range", "OPD range (nm)"),
            ("dry_mass", "Total dry mass (pg)"),
            ("step_height", "Step height (nm)"),
            ("cell_area", "Cell area (µm²)"),
            ("cell_volume", "Cell volume (fL)"),
            ("cell_circularity", "Cell circularity"),
            ("cell_mean_height", "Cell mean height (nm)"),
            ("runtime", "Runtime (ms)"),
        ]
        for r, (key, label_text) in enumerate(rows):
            label = QLabel(label_text)
            label.setProperty("role", "muted")
            grid.addWidget(label, r, 0)
            value = QLabel("—")
            value.setProperty("role", "readout")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(value, r, 1)
            self._readout_labels[key] = value
        outer.addLayout(grid)

        self._oneshot_result: Any = None
        return box

    def _on_compute_clicked(self) -> None:
        path = self.ctx.hologram_path()
        if not path:
            self.ctx.set_status("Load a hologram before computing QPI.", "warn")
            self.ctx.toast("Load a hologram first.", "warn")
            return
        params = self.ctx.get_params()
        self.ctx.set_status("Computing QPI…", "info")
        self.ctx.bridge.qpi(
            path, params,
            z_mm=float(params.z_mm),
            n_sample=float(params.n_sample),
            n_medium=float(params.n_medium),
        )

    def _on_qpi_done(self, result: Any) -> None:
        self._oneshot_result = result
        self._render_last_result()
        # This panel is the single status owner for one-shot QPI (the shell
        # no longer connects qpi_done — B-083), so carry the headline number.
        # A reference-division fallback (B-110) makes the dry mass / OPD
        # quantitatively unreliable — surface it as a warning instead of a
        # clean "ok", so the operator isn't misled by a normal-looking number.
        warn = getattr(result, "warning", None)
        if warn:
            self.ctx.set_status(f"QPI computed (UNREFERENCED) — {warn}", "warn")
            self.ctx.toast(warn, "warn")
            return
        q = getattr(result, "qpi", None)
        dm = getattr(q, "total_dry_mass_pg", None) if q is not None else None
        msg = "QPI computed."
        if dm is not None:
            msg = f"QPI computed — dry mass {float(dm):.1f} pg."
        self.ctx.set_status(msg, "ok")
        self.ctx.toast(msg, "ok")

    def _on_qpi_error(self, message: str) -> None:
        self.ctx.set_status(f"QPI failed: {message}", "error")
        self.ctx.toast(f"QPI failed: {message}", "error")

    def _render_last_result(self) -> None:
        labels = self._readout_labels
        result = self._oneshot_result
        if result is None:
            for lbl in labels.values():
                lbl.setText("—")
            return

        qpi = getattr(result, "qpi", None)
        runtime_ms = getattr(result, "runtime_ms", None)

        opd_range_nm = None
        if qpi is not None and getattr(qpi, "phase_stats", None) is not None:
            opd_range_nm = qpi.phase_stats.range_nm

        mode = self.mode_combo.currentData()
        show_bio = mode in ("biological", "both")
        show_micro = mode in ("micro_structure", "both")

        labels["opd_range"].setText(_fmt(opd_range_nm))
        labels["dry_mass"].setText(
            _fmt(getattr(qpi, "total_dry_mass_pg", None)) if show_bio and qpi else "—"
        )
        step_height_nm = None
        if show_micro and qpi is not None and qpi.step_height_m is not None:
            step_height_nm = qpi.step_height_m * 1e9
        labels["step_height"].setText(_fmt(step_height_nm))

        morph = getattr(qpi, "cell_morph", None) if qpi is not None else None
        if show_bio and morph is not None:
            labels["cell_area"].setText(_fmt(morph.area_um2))
            labels["cell_volume"].setText(_fmt(morph.volume_fL))
            labels["cell_circularity"].setText(_fmt(morph.circularity, digits=3))
            labels["cell_mean_height"].setText(_fmt(morph.mean_height_nm))
        else:
            for key in ("cell_area", "cell_volume", "cell_circularity", "cell_mean_height"):
                labels[key].setText("—")

        labels["runtime"].setText(_fmt(runtime_ms, " ms", digits=4))

    # -- batch ----------------------------------------------------------

    def _build_batch_group(self) -> QGroupBox:
        box = QGroupBox("QPI batch (across focus candidates)")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self.batch_btn = QPushButton("QPI batch")
        self.batch_btn.clicked.connect(self._on_batch_clicked)
        row.addWidget(self.batch_btn)
        row.addStretch(1)
        outer.addLayout(row)

        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(
            ["z (mm)", "dry mass (pg)", "area (µm²)",
             "circularity", "OPD range (nm)"]
        )
        # Fit columns to their headers (equal Stretch clipped "dry mass (pg)"
        # etc. mid-word in the narrow dock); last column takes leftover width.
        _bhdr = self.batch_table.horizontalHeader()
        _bhdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        _bhdr.setStretchLastSection(True)
        self.batch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_table.setAlternatingRowColors(True)
        outer.addWidget(self.batch_table)

        return box

    def _on_batch_clicked(self) -> None:
        path = self.ctx.hologram_path()
        if not path:
            self.ctx.set_status("Load a hologram before running QPI batch.", "warn")
            self.ctx.toast("Load a hologram first.", "warn")
            return
        params = self.ctx.get_params()
        z_min_mm = float(getattr(params, "af_z_min_mm", -1.0))
        z_max_mm = float(getattr(params, "af_z_max_mm", 1.0))
        n_steps = int(getattr(params, "af_n_steps", 60) or 60)
        self.ctx.set_status("Running QPI batch…", "info")
        self.ctx.bridge.qpi_batch(
            path, params,
            z_min_mm=z_min_mm, z_max_mm=z_max_mm, n_steps=n_steps,
            n_sample=float(params.n_sample), n_medium=float(params.n_medium),
        )

    def _on_batch_done(self, result: Any) -> None:
        entries = list(getattr(result, "entries", []) or [])
        self._last_batch_entries = entries
        self._populate_batch_table(entries)
        # No status/toast: the QPI-batch dialog owns the batch status (it is
        # presented by the shell on this same signal); writing it here too
        # produced duplicate status lines per batch (2026-07-06, B-083).

    def _on_batch_error(self, message: str) -> None:
        self.ctx.set_status(f"QPI batch failed: {message}", "error")
        self.ctx.toast(f"QPI batch failed: {message}", "error")

    def _populate_batch_table(self, entries: list) -> None:
        table = self.batch_table
        table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            cand = getattr(entry, "candidate", None)
            qpi = getattr(entry, "qpi_result", None)

            z_mm = None
            if cand is not None:
                z_mm = float(cand.z_m) * 1e3

            dry_mass_pg = getattr(qpi, "total_dry_mass_pg", None) if qpi else None

            morph = getattr(qpi, "cell_morph", None) if qpi else None
            area_um2 = morph.area_um2 if morph is not None else None
            circularity = morph.circularity if morph is not None else None

            opd_range_nm = None
            if qpi is not None and getattr(qpi, "phase_stats", None) is not None:
                opd_range_nm = qpi.phase_stats.range_nm

            values = [
                _fmt(z_mm, digits=4),
                _fmt(dry_mass_pg),
                _fmt(area_um2),
                _fmt(circularity, digits=3),
                _fmt(opd_range_nm),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, col, item)

    # -- busy state -------------------------------------------------------

    def _on_busy_changed(self, busy: bool, _label: str) -> None:
        self.compute_btn.setEnabled(not busy)
        self.batch_btn.setEnabled(not busy)


__all__ = ["QPIPanel"]
