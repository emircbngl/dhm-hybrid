"""AuditViewerDialog — browse + filter the JSONL audit log.

Mirrors ``ui2.dialogs.show_audit_viewer`` (operator / action dropdowns
populated from the log itself, a free-text query box, a newest-first
results table capped at 500 rows) but built as a ``QTableWidget``
against the frozen ``PanelContext``. Pure-data read/filter logic lives
in ``core.audit_viewer`` (``iter_entries``, ``AuditFilter``,
``read_entries``, ``apply_filter``, ``known_operators``,
``known_actions``) — this dialog only renders it.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.audit_viewer import (
    AuditFilter,
    apply_filter,
    known_actions,
    known_operators,
    read_entries,
)
from ui3.context import PanelContext

_ANY = "(any)"
_LIMIT = 500
_COLUMNS = ("When (UTC)", "Operator", "Action", "Params (preview)")


class AuditViewerDialog(QDialog):
    """Filterable, newest-first view of the JSONL audit log."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None,
                 *, audit_dir=None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._audit_dir = audit_dir
        self.setWindowTitle("Audit log")
        self.resize(900, 520)

        root = QVBoxLayout(self)

        heading = QLabel("Audit log")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Operator"))
        self.operator_combo = QComboBox()
        filt_row.addWidget(self.operator_combo)

        filt_row.addWidget(QLabel("Action"))
        self.action_combo = QComboBox()
        filt_row.addWidget(self.action_combo)

        filt_row.addWidget(QLabel("Search"))
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("substring match on full record…")
        self.query_edit.returnPressed.connect(self.refresh)
        filt_row.addWidget(self.query_edit, 1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        filt_row.addWidget(self.refresh_btn)
        root.addLayout(filt_row)

        self.operator_combo.currentTextChanged.connect(self.refresh)
        self.action_combo.currentTextChanged.connect(self.refresh)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table, 1)

        bottom_row = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setProperty("role", "muted")
        bottom_row.addWidget(self.status_label)
        bottom_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        bottom_row.addWidget(close_btn)
        root.addLayout(bottom_row)

        self._populate_filter_choices()
        self.refresh()

    # ---- filter choice population --------------------------------------

    def _populate_filter_choices(self) -> None:
        operators = known_operators(self._audit_dir)
        actions = known_actions(self._audit_dir)

        self.operator_combo.blockSignals(True)
        self.operator_combo.clear()
        self.operator_combo.addItem(_ANY)
        self.operator_combo.addItems(operators)
        self.operator_combo.blockSignals(False)

        self.action_combo.blockSignals(True)
        self.action_combo.clear()
        self.action_combo.addItem(_ANY)
        self.action_combo.addItems(actions)
        self.action_combo.blockSignals(False)

    # ---- refresh ----------------------------------------------------------

    def refresh(self) -> None:
        op_sel = self.operator_combo.currentText()
        ac_sel = self.action_combo.currentText()
        query = (self.query_edit.text() or "").strip() or None

        filt = AuditFilter(
            operators=[op_sel] if op_sel and op_sel != _ANY else None,
            actions=[ac_sel] if ac_sel and ac_sel != _ANY else None,
            query=query,
        )
        all_entries = read_entries(self._audit_dir, limit=_LIMIT)
        rows = apply_filter(all_entries, filt)
        self._populate_table(rows)
        self.status_label.setText(f"{len(rows)} of {len(all_entries)} entries")

    def _populate_table(self, rows) -> None:
        self.table.setRowCount(len(rows))
        for row, entry in enumerate(rows):
            summary = ", ".join(
                f"{k}={v}" for k, v in (entry.params or {}).items())
            values = [
                str(entry.timestamp)[:19],
                str(entry.operator or entry.user or "—"),
                str(entry.action),
                summary[:120],
            ]
            for col, text in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(text))

    # ---- introspection helpers (for tests) -------------------------------

    def row_count(self) -> int:
        return self.table.rowCount()


__all__ = ["AuditViewerDialog"]
