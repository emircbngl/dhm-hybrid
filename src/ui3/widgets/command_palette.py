"""Command palette (⌘K) — fuzzy launcher over registered commands.

A ``Command`` is an id + title + callback (+ optional category/shortcut). The
``CommandRegistry`` is a plain dict (Qt-free, testable); ``CommandPalette`` is a
frameless modal that filters by substring and runs the chosen command. Mirrors
the v1 pattern that consolidated scattered actions into one registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)


@dataclass
class Command:
    id: str
    title: str
    run: Callable[[], None]
    category: str = ""
    shortcut: str = ""


class CommandRegistry:
    """Ordered registry of commands (Qt-free)."""

    def __init__(self) -> None:
        self._by_id: Dict[str, Command] = {}
        self._order: List[str] = []

    def register(self, cmd: Command) -> None:
        if cmd.id not in self._by_id:
            self._order.append(cmd.id)
        self._by_id[cmd.id] = cmd

    def all(self) -> List[Command]:
        return [self._by_id[i] for i in self._order]

    def search(self, query: str) -> List[Command]:
        q = (query or "").strip().lower()
        if not q:
            return self.all()
        hits = []
        for cmd in self.all():
            hay = f"{cmd.title} {cmd.category}".lower()
            if all(tok in hay for tok in q.split()):
                hits.append(cmd)
        # Title-prefix matches first.
        hits.sort(key=lambda c: 0 if c.title.lower().startswith(q) else 1)
        return hits

    def get(self, cmd_id: str) -> Optional[Command]:
        return self._by_id.get(cmd_id)


class CommandPalette(QDialog):
    """Frameless modal launcher."""

    def __init__(self, registry: CommandRegistry,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._registry = registry
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setFixedWidth(560)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("Run a command…")
        self._list = QListWidget()
        self._list.setUniformItemSizes(True)
        lay.addWidget(self._edit)
        lay.addWidget(self._list)

        self._edit.textChanged.connect(self._refilter)
        self._edit.returnPressed.connect(self._run_selected)
        self._list.itemActivated.connect(lambda _i: self._run_selected())
        self._edit.installEventFilter(self)
        self._refilter("")

    def open_palette(self) -> None:
        self._edit.clear()
        self._refilter("")
        self._edit.setFocus()
        self.show()
        self.raise_()

    def _refilter(self, text: str) -> None:
        self._list.clear()
        for cmd in self._registry.search(text):
            label = cmd.title if not cmd.category else f"{cmd.category} · {cmd.title}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, cmd.id)
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)

    def _run_selected(self) -> None:
        item = self._list.currentItem()
        if item is None and self._list.count():
            item = self._list.item(0)
        if item is None:
            return
        cmd = self._registry.get(item.data(Qt.UserRole))
        self.hide()
        if cmd is not None:
            cmd.run()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        # Arrow keys in the line edit move the list selection.
        from PySide6.QtCore import QEvent
        if obj is self._edit and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                row = self._list.currentRow()
                row += 1 if key == Qt.Key_Down else -1
                row = max(0, min(self._list.count() - 1, row))
                self._list.setCurrentRow(row)
                return True
            if key == Qt.Key_Escape:
                self.hide()
                return True
        return super().eventFilter(obj, event)
