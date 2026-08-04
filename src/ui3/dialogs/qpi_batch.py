"""QPIBatchDialog — QPI batch review (one row per focus candidate).

Mirrors ``ui2.dialogs.show_qpi_batch_review`` (the Dear PyGui table:
rank / z / prominence / dry mass / area / OPD / step / circularity /
"Focus here") but built as a ``QTableWidget`` against the frozen
``PanelContext``, wired to ``ctx.bridge.qpi_batch`` /
``qpi_batch_done`` / ``qpi_batch_error`` (``ui3.bridge.WorkerBridge``,
which already carries the ``QPIBatchResultWrap`` produced by
``ui2.workers.ScienceDriver.run_qpi_batch``).

Behaviour parity:

* One row per ``QPIBatchEntry`` (``core.qpi_batch.QPIBatchEntry`` —
  ``candidate: FocusCandidate`` + ``qpi_result: QPIResult``).
* Selecting a row and clicking "Reconstruct here" writes that
  candidate's ``z_mm`` onto ``ctx.get_params().z_mm``, calls
  ``ctx.params_changed()``, then triggers
  ``ctx.bridge.reconstruct(path, params)`` — same "commit" semantics
  as ui2's ``on_focus`` callback, just renamed to match the ui3 phrase
  ("reconstruct here" rather than "focus here", since this dialog's
  action is a full reconstruction, not a bridge-level focus-only op).
* Missing/None fields (``cell_morph`` outside bio mode, ``step_height_m``
  with no bimodal histogram peak, …) render as "—" rather than raising.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
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

_COLUMNS = (
    "#", "z (mm)", "prominence", "dry mass (pg)", "area (µm²)",
    "OPD range (nm)", "step height (nm)", "circularity",
)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    import math
    if not math.isfinite(v):
        return "—"
    return f"{v:.{digits}g}"


class QPIBatchDialog(QDialog):
    """Review table for a ``QPIBatchResultWrap`` — pick a candidate plane
    and jump the live reconstruction there."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._entries: list = []
        self.setWindowTitle("QPI batch — candidate comparison")
        self.resize(880, 480)

        root = QVBoxLayout(self)

        heading = QLabel("QPI batch — candidate comparison")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        self.hint_label = QLabel(
            "Run a QPI batch (via the QPI panel) — results land here. "
            "Select a row and click \"Reconstruct here\" to jump the live "
            "reconstruction to that focus plane."
        )
        self.hint_label.setProperty("role", "muted")
        self.hint_label.setWordWrap(True)
        root.addWidget(self.hint_label)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        # Size each column to fit its (uppercase) header/content instead of
        # equal Stretch, which squeezed headers like "OPD RANGE (NM)" so tight
        # they clipped mid-word; the last column absorbs any leftover width.
        _hdr = self.table.horizontalHeader()
        _hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        _hdr.setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemDoubleClicked.connect(lambda _item: self._on_reconstruct_here())
        root.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.reconstruct_btn = QPushButton("Reconstruct here")
        self.reconstruct_btn.setProperty("role", "primary")
        self.reconstruct_btn.clicked.connect(self._on_reconstruct_here)
        btn_row.addWidget(self.reconstruct_btn)

        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self._on_export_csv)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch(1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self.ctx.bridge.qpi_batch_done.connect(self._on_batch_done)
        self.ctx.bridge.qpi_batch_error.connect(self._on_batch_error)

    # ---- public API -----------------------------------------------------

    def set_result(self, result: Any) -> None:
        """Populate the table from a ``QPIBatchResultWrap``-shaped object
        (or any object exposing ``.entries``)."""
        entries = list(getattr(result, "entries", []) or [])
        self._populate(entries)

    def open_with(self, result: Any) -> None:
        self.set_result(result)
        self.present()

    def present(self) -> None:
        """Show/raise without repopulating — for callers reacting to the
        same bridge signal this dialog already consumes (the shell), so the
        table isn't populated twice per result (2026-07-06 review, B-083)."""
        from ui3.dialogs._present import present_centered
        present_centered(self)

    # ---- bridge callbacks -------------------------------------------------

    def _on_batch_done(self, result: Any) -> None:
        self.set_result(result)
        self.ctx.set_status(
            f"QPI batch: {len(self._entries)} candidate(s).", "ok")

    def _on_batch_error(self, message: str) -> None:
        self.ctx.set_status(f"QPI batch failed: {message}", "danger")

    # ---- table ------------------------------------------------------------

    def _populate(self, entries: list) -> None:
        self._entries = entries
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            cand = getattr(entry, "candidate", None)
            qpi = getattr(entry, "qpi_result", None)
            morph = getattr(qpi, "cell_morph", None) if qpi is not None else None
            phase_stats = getattr(qpi, "phase_stats", None) if qpi is not None else None

            rank = getattr(cand, "rank", row)
            z_mm = float(cand.z_m) * 1e3 if cand is not None else None
            prominence = getattr(cand, "prominence", None) if cand is not None else None
            dry_mass_pg = getattr(qpi, "total_dry_mass_pg", None) if qpi is not None else None
            area_um2 = getattr(morph, "area_um2", None) if morph is not None else None
            opd_range_nm = getattr(phase_stats, "range_nm", None) if phase_stats is not None else None
            step_h_m = getattr(qpi, "step_height_m", None) if qpi is not None else None
            step_h_nm = step_h_m * 1e9 if step_h_m is not None else None
            circularity = getattr(morph, "circularity", None) if morph is not None else None

            values = [
                str(int(rank) + 1) if isinstance(rank, int) else str(rank),
                _fmt(z_mm, digits=4),
                _fmt(prominence),
                _fmt(dry_mass_pg),
                _fmt(area_um2),
                _fmt(opd_range_nm),
                _fmt(step_h_nm),
                _fmt(circularity),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 0:
                    item.setData(Qt.UserRole, z_mm)
                self.table.setItem(row, col, item)

    def _selected_z_mm(self) -> Optional[float]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        row = rows[0].row() if rows else self.table.currentRow()
        if row is None or row < 0 or row >= len(self._entries):
            return None
        cand = getattr(self._entries[row], "candidate", None)
        if cand is None:
            return None
        return float(cand.z_m) * 1e3

    # ---- actions ------------------------------------------------------

    def _on_reconstruct_here(self) -> None:
        z_mm = self._selected_z_mm()
        if z_mm is None:
            self.ctx.set_status("Select a candidate row first.", "warn")
            return
        params = self.ctx.get_params()
        params.z_mm = float(z_mm)
        self.ctx.params_changed()
        path = self.ctx.hologram_path()
        if path is None:
            self.ctx.set_status(
                f"z set to {z_mm:+.4f} mm (load a hologram to reconstruct).",
                "info")
            return
        self.ctx.set_status(f"Reconstructing at z={z_mm:+.4f} mm…", "info")
        self.ctx.bridge.reconstruct(path, params)

    def _on_export_csv(self) -> None:
        if not self._entries:
            self.ctx.set_status("No QPI batch entries to export.", "warn")
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Export QPI batch CSV", "qpi_batch.csv", "CSV (*.csv)")
        if not path:
            return
        self.export_csv(path)

    def export_csv(self, path: str) -> None:
        """Write the current entries to ``path`` via
        ``core.qpi_batch.write_qpi_batch_csv``. Split out of the
        file-dialog handler so tests can call it directly."""
        from core.qpi_batch import write_qpi_batch_csv

        write_qpi_batch_csv(path, self._entries)
        self.ctx.set_status(f"QPI batch exported to {path}", "ok")


__all__ = ["QPIBatchDialog"]
