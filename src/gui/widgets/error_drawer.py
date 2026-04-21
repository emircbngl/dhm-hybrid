"""Session error log — the "Show log ›" destination from a toast.

Structured as a right-docked drawer (``QDockWidget``) so it sits next to
the main view without stealing a window. Reads from an
:class:`~core.errors.ErrorCenter`'s in-memory history and refreshes on
:meth:`refresh` or when opened via :meth:`open_for_event`.

Row layout
----------

    10:42:13  ERROR  Reconstruction failed
                     wavelength produced invalid k
                     Set a positive wavelength…
                     ▸ Traceback (4 frames)        ← click to expand

The traceback is lazy: we only add the expanded text widget when the
user unfolds the row. Session history grows to the ``ErrorCenter``'s
capacity (200 by default); older entries age out automatically.

Keyboard
--------
* **Esc** closes the drawer.
* **Enter** on a selected row toggles its traceback.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.errors import ErrorCenter, ErrorEvent, Severity, get_error_center

_LOG = logging.getLogger(__name__)

_SEVERITY_LABEL = {
    Severity.INFO: "INFO",
    Severity.WARN: "WARN",
    Severity.ERROR: "ERROR",
    Severity.FATAL: "FATAL",
}


def _format_ts(ts: str) -> str:
    """Render an ISO8601 timestamp as local ``HH:MM:SS``.

    Falls back to the raw string when parsing fails — a malformed
    timestamp shouldn't crash the drawer.
    """
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return ts


class _ErrorRow(QFrame):
    """One event's worth of rendered detail."""

    def __init__(self, event: ErrorEvent, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._event = event
        self._trace_widget: Optional[QLabel] = None
        self.setObjectName("errorRow")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(2)

        header = QHBoxLayout()
        header.setSpacing(10)
        header.addWidget(self._muted(_format_ts(self._event.timestamp)))
        header.addWidget(self._muted(
            _SEVERITY_LABEL.get(self._event.severity, "ERROR")
        ))
        title = QLabel(self._event.title or "Error")
        title_font = title.font()
        title_font.setWeight(QFont.Weight.DemiBold)
        title.setFont(title_font)
        title.setWordWrap(True)
        header.addWidget(title, 1)
        outer.addLayout(header)

        if self._event.cause:
            cause = QLabel(self._event.cause)
            cause.setWordWrap(True)
            outer.addWidget(cause)

        if self._event.action:
            action = QLabel(self._event.action)
            action.setObjectName("rowAction")
            action.setWordWrap(True)
            outer.addWidget(action)

        if self._event.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self._event.context.items())
            ctx_label = self._muted(ctx)
            ctx_label.setWordWrap(True)
            outer.addWidget(ctx_label)

        if self._event.trace:
            self._trace_toggle = QPushButton("▸ Traceback")
            self._trace_toggle.setObjectName("rowTraceToggle")
            self._trace_toggle.setFlat(True)
            self._trace_toggle.setStyleSheet(
                "text-align: left; color: palette(mid); border: none;"
            )
            self._trace_toggle.clicked.connect(self._toggle_trace)
            outer.addWidget(self._trace_toggle)

    @staticmethod
    def _muted(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: palette(mid); font-size: 11px;")
        return lbl

    def _toggle_trace(self) -> None:
        if self._trace_widget is None:
            self._trace_widget = QLabel(self._event.trace or "")
            self._trace_widget.setWordWrap(True)
            self._trace_widget.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            # Monospace for tracebacks so frame paths line up.
            mono = QFont("Menlo")
            mono.setStyleHint(QFont.StyleHint.Monospace)
            mono.setPointSize(10)
            self._trace_widget.setFont(mono)
            self._trace_widget.setStyleSheet(
                "color: palette(text); background: palette(alternate-base);"
                " padding: 6px; border-radius: 4px;"
            )
            self.layout().addWidget(self._trace_widget)
            self._trace_toggle.setText("▾ Traceback")
        else:
            visible = self._trace_widget.isVisible()
            self._trace_widget.setVisible(not visible)
            self._trace_toggle.setText(
                "▾ Traceback" if not visible else "▸ Traceback"
            )


class ErrorDrawer(QDockWidget):
    """Session error log, newest-first. Toggle via :meth:`toggle`."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        center: Optional[ErrorCenter] = None,
    ) -> None:
        super().__init__("Error log", parent)
        self._center = center or get_error_center()
        self.setObjectName("errorDrawer")
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea
                             | Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self._build_ui()

        # Esc closes the drawer — matches palette grammar.
        self._esc = QShortcut(QKeySequence("Escape"), self)
        self._esc.activated.connect(self.close)

    # ---- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        container = QWidget(self)
        self.setWidget(container)

        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(12, 8, 8, 8)
        self._header_label = QLabel("Session errors")
        header.addWidget(self._header_label, 1)

        clear_btn = QPushButton("Clear")
        clear_btn.setFlat(True)
        clear_btn.clicked.connect(self._on_clear)
        header.addWidget(clear_btn)
        outer.addLayout(header)

        self._scroll = QScrollArea(container)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._list_host = QWidget(self._scroll)
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch(1)
        self._scroll.setWidget(self._list_host)
        outer.addWidget(self._scroll, 1)

        self.setMinimumWidth(360)

    # ---- public API --------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the list from the center's history, newest-first."""
        # Clear
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()

        events = self._center.recent(limit=200)
        if not events:
            placeholder = QLabel("No errors yet.")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: palette(mid); padding: 24px;")
            self._list_layout.insertWidget(0, placeholder)
            self._header_label.setText("Session errors")
            return

        for i, event in enumerate(events):
            row = _ErrorRow(event, self)
            self._list_layout.insertWidget(i, row)
            # Subtle separator between rows — muted, not a hard rule.
            if i < len(events) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("color: palette(mid); background: palette(mid);")
                sep.setFixedHeight(1)
                self._list_layout.insertWidget(i + 1, sep)

        self._header_label.setText(f"Session errors ({len(events)})")

    def open_for_event(self, event_id: str) -> None:
        """Show the drawer and scroll to a specific event if present."""
        self.refresh()
        self.show()
        self.raise_()
        # Scroll-to-event left as a follow-up — the newest events are at the
        # top already, which covers the most common toast-click path.
        _LOG.debug("error-drawer: opened for event %s", event_id)

    def toggle(self) -> None:
        if self.isVisible():
            self.close()
        else:
            self.refresh()
            self.show()
            self.raise_()

    # ---- actions -----------------------------------------------------------

    def _on_clear(self) -> None:
        self._center.clear_history()
        self.refresh()


__all__ = ["ErrorDrawer"]
