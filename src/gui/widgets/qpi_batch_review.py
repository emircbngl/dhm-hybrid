"""QPI batch review dialog — compare candidate focus planes in-app.

The CSV-only flow (v1.2-tomo beta) works for the R/MATLAB crowd but
leaves the bench operator scrolling through a file to decide which
focus plane to trust. This dialog puts the comparison in the app:
one row per candidate with the quantitative fields that actually
drive a biology decision — dry mass, projected area, OPD range, step
height, circularity — and a *Focus here* button that commits to the
row's z with one click.

Signals
-------
``focus_requested(z_m)`` — user clicked *Focus here* on a row.
``export_csv_requested()`` — user clicked *Export CSV…*; the host
opens a :class:`QFileDialog` and writes the batch via
:func:`core.qpi_batch.write_qpi_batch_csv`.

Keeping the file-dialog outside this widget is deliberate — the
dialog stays pure presentation and is trivially testable.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
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

from core.qpi_batch import QPIBatchEntry


_COLUMNS: tuple[str, ...] = (
    "#",             # rank
    "z (mm)",
    "prominence",
    "dry mass (pg)",
    "area (µm²)",
    "OPD range (nm)",
    "step (nm)",
    "circularity",
    "",              # Focus here button
)


def _fmt_cell(value, digits: int = 3, *, missing: str = "—") -> str:
    """Render a possibly-missing float cell — empty QPI fields land as
    None/NaN, render as em-dash so the column aligns."""
    if value is None:
        return missing
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return missing
    import math
    if not math.isfinite(fv):
        return missing
    return f"{fv:.{digits}g}"


class QPIBatchReviewDialog(QDialog):
    """Tabular picker over a :class:`~core.qpi_batch.QPIBatchEntry` list."""

    focus_requested = Signal(float)  # z_m
    export_csv_requested = Signal()  # user clicked Export CSV…

    def __init__(
        self,
        entries: Sequence[QPIBatchEntry],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("QPI batch — candidate focus comparison")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setMinimumSize(720, 320)
        self._entries: List[QPIBatchEntry] = list(entries)
        self._build_ui()
        self._populate()

    # ---- UI construction -----------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._header = QLabel()
        self._header.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(self._header)

        self._table = QTableWidget(self)
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        # Rank narrow, numeric columns stretch, button narrow.
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, len(_COLUMNS) - 1):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            len(_COLUMNS) - 1, QHeaderView.ResizeMode.ResizeToContents,
        )
        layout.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("Export CSV…", self)
        self._export_btn.clicked.connect(self.export_csv_requested.emit)
        btn_row.addWidget(self._export_btn)
        btn_row.addStretch(1)
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ---- population ----------------------------------------------------

    def _populate(self) -> None:
        n = len(self._entries)
        if n == 0:
            self._header.setText(
                "No QPI batch entries — widen the z range or drop the "
                "prominence threshold."
            )
            self._table.setRowCount(0)
            self._export_btn.setEnabled(False)
            return

        self._header.setText(
            f"{n} candidate focus plane{'s' if n != 1 else ''} — "
            f"click *Focus here* to reconstruct at that z, "
            f"*Export CSV…* to dump the whole table."
        )
        self._export_btn.setEnabled(True)

        self._table.setRowCount(n)
        for row, entry in enumerate(self._entries):
            self._render_row(row, entry)

    def _render_row(self, row: int, entry: QPIBatchEntry) -> None:
        cand = entry.candidate
        q = entry.qpi_result
        morph = getattr(q, "cell_morph", None)
        phase_stats = getattr(q, "phase_stats", None)

        values: list[str] = [
            str(cand.rank + 1),                                    # # (1-based)
            f"{cand.z_m * 1e3:+.3f}",                             # z (mm)
            f"{cand.prominence:.3f}",                              # prominence
            _fmt_cell(getattr(q, "total_dry_mass_pg", None)),      # dry mass
            _fmt_cell(getattr(morph, "area_um2", None) if morph else None),
            _fmt_cell(getattr(phase_stats, "range_nm", None)
                      if phase_stats else None),
            _fmt_cell(
                (getattr(q, "step_height_m", None) or 0) * 1e9
                if getattr(q, "step_height_m", None) is not None
                else None,
            ),
            _fmt_cell(getattr(morph, "circularity", None) if morph else None),
        ]

        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, col, item)

        btn = QPushButton("Focus here", self)
        z_m = float(cand.z_m)
        btn.clicked.connect(
            lambda _=False, z=z_m: self._on_focus_clicked(z)
        )
        self._table.setCellWidget(row, len(_COLUMNS) - 1, btn)

    # ---- actions -------------------------------------------------------

    def _on_focus_clicked(self, z_m: float) -> None:
        self.focus_requested.emit(z_m)
        self.accept()

    # ---- public API ----------------------------------------------------

    def entries(self) -> List[QPIBatchEntry]:
        """Defensive copy for tests."""
        return list(self._entries)


__all__ = ["QPIBatchReviewDialog"]
