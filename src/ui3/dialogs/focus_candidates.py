"""FocusCandidatesDialog — multi-focus candidate picker.

Mirrors ``ui2.dialogs.show_focus_candidates`` (rank / z / prominence /
"Focus here" table) but built as a ``QTableWidget`` against the frozen
``PanelContext``, wired to ``ctx.bridge.find_focus_candidates`` /
``focus_candidates_done`` / ``focus_candidates_error``
(``ui3.bridge.WorkerBridge``, wrapping
``ui2.workers.ScienceDriver.find_focus_candidates`` →
``MultiFocusResult(candidates: List[FocusCandidate], runtime_ms)``,
``core.autofocus.FocusCandidate`` carrying ``z_m`` / ``score`` /
``prominence`` / ``rank``).

"Focus here" on a selected row writes that candidate's ``z_mm`` onto
``ctx.get_params().z_mm``, calls ``ctx.params_changed()``, then triggers
``ctx.bridge.reconstruct`` — same commit semantics as
``FocusPanel._on_candidate_double_clicked`` (``ui3/panels/focus_panel.py``,
frozen-adjacent reference, not touched here) and as ui2's ``on_focus``
callback.
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

_COLUMNS = ("#", "z (mm)", "score", "prominence")


def _fmt(value: Optional[float], digits: int = 4) -> str:
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


class FocusCandidatesDialog(QDialog):
    """Review table for a multi-focus candidate scan — pick a candidate
    plane and jump the live reconstruction there."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._candidates: list = []
        self.setWindowTitle("Focus candidates")
        self.resize(560, 420)

        root = QVBoxLayout(self)

        heading = QLabel("Focus candidates")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        self.hint_label = QLabel(
            "Plausible focus planes from the last scan. Select a row and "
            "click \"Focus here\" to reconstruct at that z."
        )
        self.hint_label.setProperty("role", "muted")
        self.hint_label.setWordWrap(True)
        root.addWidget(self.hint_label)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemDoubleClicked.connect(lambda _item: self._on_focus_here())
        root.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        self.focus_btn = QPushButton("Focus here")
        self.focus_btn.setProperty("role", "primary")
        self.focus_btn.clicked.connect(self._on_focus_here)
        btn_row.addWidget(self.focus_btn)
        btn_row.addStretch(1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self.ctx.bridge.focus_candidates_done.connect(self._on_candidates_done)
        self.ctx.bridge.focus_candidates_error.connect(self._on_candidates_error)

    # ---- public API -----------------------------------------------------

    def set_result(self, result: Any) -> None:
        """Populate the table from a ``MultiFocusResult``-shaped object
        (or any object exposing ``.candidates``), or a bare list of
        ``FocusCandidate``."""
        candidates = list(getattr(result, "candidates", result) or [])
        self._populate(candidates)

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

    def _on_candidates_done(self, result: Any) -> None:
        self.set_result(result)
        self.ctx.set_status(
            f"Found {len(self._candidates)} focus candidate(s).", "ok")

    def _on_candidates_error(self, message: str) -> None:
        self.ctx.set_status(f"Focus scan failed: {message}", "danger")

    # ---- table ------------------------------------------------------------

    def _populate(self, candidates: list) -> None:
        self._candidates = candidates
        self.table.setRowCount(len(candidates))
        for row, cand in enumerate(candidates):
            rank = getattr(cand, "rank", row)
            z_mm = float(cand.z_m) * 1e3 if hasattr(cand, "z_m") else None
            score = getattr(cand, "score", None)
            prominence = getattr(cand, "prominence", None)

            values = [
                str(int(rank) + 1) if isinstance(rank, int) else str(rank),
                _fmt(z_mm),
                _fmt(score, digits=3),
                _fmt(prominence, digits=3),
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
        if row is None or row < 0 or row >= len(self._candidates):
            return None
        cand = self._candidates[row]
        if not hasattr(cand, "z_m"):
            return None
        return float(cand.z_m) * 1e3

    # ---- actions ------------------------------------------------------

    def _on_focus_here(self) -> None:
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


__all__ = ["FocusCandidatesDialog"]
