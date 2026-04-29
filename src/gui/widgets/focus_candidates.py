"""Focus candidates dialog — picker for multi-focus autofocus results.

The single-best-z autofocus lands on one plane. When a scene has
several objects at different depths (cell clusters, stacked
micro-structures), that one value discards the rest. This dialog lists
every plausible focus plane the landscape scan turned up and lets the
lab operator commit to one with a click.

Design notes
------------

* Non-modal — the user can keep scrolling the image while they compare
  candidates.
* One row per candidate, sorted by prominence descending so the
  "strongest" focus sits at top.
* ``focus_requested(z_m)`` signal fires when a row's *Focus here*
  button is clicked; the host wires this to the reconstruction trigger.
* Empty-list case renders a muted placeholder — we never want an
  empty grid staring back at the operator.

UI-side integration (command registration, reconstruction wiring) lives
in ``main_window``; this widget is pure presentation.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

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

from core.autofocus import FocusCandidate


class FocusCandidatesDialog(QDialog):
    """Pickable list of focus candidates from ``find_focus_candidates()``.

    Emits :attr:`focus_requested` with the chosen ``z_m`` when the user
    clicks *Focus here*.
    """

    focus_requested = Signal(float)  # z_m in metres

    def __init__(
        self,
        candidates: Sequence[FocusCandidate],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Focus candidates")
        self.setWindowFlag(Qt.WindowType.Tool, True)  # floats over main window
        self.setMinimumWidth(420)
        self._candidates: List[FocusCandidate] = list(candidates)
        self._build_ui()
        self._populate()

    # ---- UI construction ------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._header = QLabel()
        self._header.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(self._header)

        self._table = QTableWidget(self)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["#", "z (mm)", "Prominence", ""]
        )
        # Column sizing: narrow rank/button, flexible middle.
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self._table, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    def _populate(self) -> None:
        count = len(self._candidates)
        if count == 0:
            self._header.setText(
                "No candidate focus planes found. "
                "Try a wider z range or a different metric."
            )
            self._table.setRowCount(0)
            return

        self._header.setText(
            f"{count} candidate focus plane{'s' if count != 1 else ''} "
            f"— click *Focus here* to reconstruct at that z."
        )

        self._table.setRowCount(count)
        for row, cand in enumerate(self._candidates):
            self._set_row(row, cand)

    def _set_row(self, row: int, cand: FocusCandidate) -> None:
        # Rank (0-based in data, 1-based in UI)
        rank_item = QTableWidgetItem(str(cand.rank + 1))
        rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 0, rank_item)

        # z in mm, 3 decimals
        z_item = QTableWidgetItem(f"{cand.z_m * 1e3:+.3f}")
        z_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 1, z_item)

        # Prominence
        prom_item = QTableWidgetItem(f"{cand.prominence:.3f}")
        prom_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 2, prom_item)

        # Focus here button — captures z_m by value
        btn = QPushButton("Focus here", self)
        z_m = float(cand.z_m)
        btn.clicked.connect(lambda _=False, z=z_m: self._on_focus_clicked(z))
        self._table.setCellWidget(row, 3, btn)

    # ---- actions --------------------------------------------------------

    def _on_focus_clicked(self, z_m: float) -> None:
        """Emit the chosen z and dismiss. The host wires the reconstruction."""
        self.focus_requested.emit(z_m)
        self.accept()

    # ---- public API -----------------------------------------------------

    def candidates(self) -> List[FocusCandidate]:
        """Copy of the candidate list shown to the user. Handy for tests."""
        return list(self._candidates)


__all__ = ["FocusCandidatesDialog"]
