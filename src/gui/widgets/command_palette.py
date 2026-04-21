"""⌘K command palette.

A keyboard-first launcher that sits over the main window and lets the
user pick any registered :class:`gui.commands.Command` by typing part of
its title. Modelled on the Raycast / VS Code palette pattern — open,
type, Enter, done.

Design ethos — Ma (間), Rams #10
-------------------------------
* Frameless. No title bar. The palette *is* its search box.
* One visual rhythm: row = title (left) + shortcut mnemonic (right).
* Category shows up as a muted one-word prefix, not a coloured badge.
* Width locked at 560 px — wide enough to never wrap, narrow enough to
  read like a command line, not a dialog.
* No confirm step. Enter runs. Escape cancels. That's the grammar.

Threading / ownership
---------------------
The palette reads from a :class:`~gui.commands.CommandRegistry`. The
``when`` predicate is re-evaluated every time the palette opens and
every keystroke so commands that shouldn't run right now grey out
without the caller having to update anything.

Callbacks fire on the Qt main thread — the same thread that opened the
palette. Any threading the command callback needs it owns itself.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.commands import Command, CommandRegistry, get_registry

_LOG = logging.getLogger(__name__)

# UX constants — spelled here so a designer can tune without spelunking.
_PALETTE_WIDTH = 560
_PALETTE_MARGIN_TOP_RATIO = 0.18  # % of parent height from the top
_MAX_VISIBLE_ROWS = 10
_ROW_HEIGHT = 28


class CommandPalette(QDialog):
    """Keyboard-first command launcher. Open with ⌘K / Ctrl+K."""

    # Emitted after a command successfully runs — the main window uses this
    # for audit logging without having to monkey-patch the registry.
    command_invoked = Signal(str)  # command id

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        registry: Optional[CommandRegistry] = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry or get_registry()
        self._results: list[Command] = []

        # Frameless, above the main window, dismiss on focus loss.
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        # Qt doesn't offer WA_TranslucentBackground without a compositor on
        # some platforms; skip it to keep this widget boring and portable.
        self.setObjectName("commandPalette")
        self.setFixedWidth(_PALETTE_WIDTH)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self._build_ui()
        self._apply_style()

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        frame = QFrame(self)
        frame.setObjectName("commandPaletteFrame")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(frame)

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._input = QLineEdit(frame)
        self._input.setObjectName("commandPaletteInput")
        self._input.setPlaceholderText("Type a command…")
        self._input.setClearButtonEnabled(False)
        self._input.textChanged.connect(self._on_query_changed)
        self._input.returnPressed.connect(self._run_selected)
        outer.addWidget(self._input)

        self._results_view = QListWidget(frame)
        self._results_view.setObjectName("commandPaletteResults")
        self._results_view.setFrameShape(QFrame.Shape.NoFrame)
        self._results_view.setUniformItemSizes(True)
        self._results_view.itemActivated.connect(lambda _: self._run_selected())
        outer.addWidget(self._results_view)

        # Install event filter so arrow keys typed in the QLineEdit move
        # the list selection instead of the caret.
        self._input.installEventFilter(self)

    def _apply_style(self) -> None:
        # Minimal qss — system background, one border, muted separators.
        # Deliberately understated: Rams #10 ("as little design as possible").
        self.setStyleSheet("""
            QDialog#commandPalette {
                background: transparent;
            }
            QFrame#commandPaletteFrame {
                background: palette(base);
                border: 1px solid palette(mid);
                border-radius: 6px;
            }
            QLineEdit#commandPaletteInput {
                border: none;
                border-bottom: 1px solid palette(mid);
                padding: 12px 14px;
                font-size: 15px;
                background: transparent;
            }
            QListWidget#commandPaletteResults {
                border: none;
                padding: 4px 0;
                outline: none;
            }
            QListWidget#commandPaletteResults::item {
                padding: 4px 14px;
            }
            QListWidget#commandPaletteResults::item:selected {
                background: palette(highlight);
                color: palette(highlighted-text);
            }
        """)

    # ---- Behaviour ---------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:  # noqa: D401 — Qt override
        self._input.clear()
        self._input.setFocus(Qt.FocusReason.PopupFocusReason)
        self._populate(self._registry.search(""))
        self._center_on_parent()
        super().showEvent(event)

    def eventFilter(self, obj, event: QEvent) -> bool:  # noqa: D401 — Qt override
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            assert isinstance(event, QKeyEvent)
            if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self._move_selection(+1 if event.key() == Qt.Key.Key_Down else -1)
                return True
            if event.key() == Qt.Key.Key_Escape:
                self.reject()
                return True
        return super().eventFilter(obj, event)

    def _on_query_changed(self, text: str) -> None:
        self._populate(self._registry.search(text))

    def _populate(self, commands: Iterable[Command]) -> None:
        self._results = list(commands)
        self._results_view.clear()
        for cmd in self._results:
            item = QListWidgetItem(self._format_row(cmd))
            # Grey out when-blocked commands so users see them but can't run them.
            if cmd.when is not None and not cmd.when():
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.ItemDataRole.UserRole, cmd.id)
            self._results_view.addItem(item)

        if self._results:
            self._results_view.setCurrentRow(0)

        # Size the list to the content, up to a cap.
        n_visible = min(len(self._results), _MAX_VISIBLE_ROWS)
        self._results_view.setFixedHeight(max(n_visible, 1) * _ROW_HEIGHT + 8)

    def _format_row(self, cmd: Command) -> str:
        # "Category · Title                         Ctrl+R"
        # Single line, monospace shortcut — aligned visually by the fixed
        # width of the palette. We deliberately avoid HTML / rich text so
        # the item stays cheap and selectable.
        shortcut = f"   {cmd.shortcut}" if cmd.shortcut else ""
        return f"{cmd.category} · {cmd.title}{shortcut}"

    def _move_selection(self, delta: int) -> None:
        if not self._results:
            return
        row = self._results_view.currentRow()
        new_row = max(0, min(len(self._results) - 1, row + delta))
        self._results_view.setCurrentRow(new_row)

    def _run_selected(self) -> None:
        item = self._results_view.currentItem()
        if item is None:
            return
        if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        cmd_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(cmd_id, str):
            return
        # Close first so the callback can open another dialog without
        # fighting our modality.
        self.accept()

        def _invoke() -> None:
            try:
                invoked = self._registry.invoke(cmd_id)
            except Exception:  # noqa: BLE001
                _LOG.exception("commands: callback raised for %s", cmd_id)
                return
            if invoked:
                self.command_invoked.emit(cmd_id)

        # Defer so the dialog is fully hidden before the callback runs.
        QTimer.singleShot(0, _invoke)

    # ---- Placement ---------------------------------------------------------

    def _center_on_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            # Fall back to the screen centre — the palette should still be
            # usable when opened without a parent (diagnostics, tests).
            screen = QApplication.primaryScreen()
            if screen is None:
                return
            geo = screen.geometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + int(geo.height() * _PALETTE_MARGIN_TOP_RATIO)
        else:
            geo = parent.geometry()
            top_left = parent.mapToGlobal(geo.topLeft() - parent.pos())
            x = top_left.x() + (geo.width() - self.width()) // 2
            y = top_left.y() + int(geo.height() * _PALETTE_MARGIN_TOP_RATIO)
        self.move(x, y)


__all__ = ["CommandPalette"]
