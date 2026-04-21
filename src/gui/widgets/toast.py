"""Error-bus toast — non-blocking, right-anchored, Ma-minimal.

An :class:`~core.errors.ErrorCenter` subscriber that surfaces each
:class:`~core.errors.ErrorEvent` as a small card in the top-right of the
main window. Cards stack downward, max three at a time — a fourth push
dismisses the oldest so the stack never grows forever.

Design
------
Each toast is one visual block:

    Reconstruction failed             ×
    wavelength produced invalid k
    Set a positive wavelength…
                            Show log ›

* Title: the event's ``title`` — a noun phrase, present tense.
* Cause: one line of ``cause`` — the exception message, truncated.
* Action: one line of ``action`` — a Rams-#4 user-correctable hint.
  Hidden if empty.
* "Show log ›": opens the :class:`ErrorDrawer` at this event.
* ×: dismiss this toast only.

There's deliberately no severity colour. A red banner on every warning
trains the user to ignore red; instead the *tone* (title + cause +
action) carries severity. WARN and ERROR look the same; INFO doesn't
toast at all — it lands in the audit log and the drawer.

Ownership
---------
``ToastHost`` parents all toasts. Exactly one ``ToastHost`` per main
window. Subscribe to the singleton ``ErrorCenter`` on attach; detach
when the window closes.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.errors import ErrorCenter, ErrorEvent, Severity, get_error_center


def _severity_rank(s: Severity) -> int:
    """Mirror the private rank in ``core.errors`` — duplicating three
    integers beats exposing an internal helper over the module boundary."""
    return {Severity.INFO: 0, Severity.WARN: 1,
            Severity.ERROR: 2, Severity.FATAL: 3}[s]

_LOG = logging.getLogger(__name__)

_TOAST_WIDTH = 360
_TOAST_MARGIN = 16
_TOAST_SPACING = 8
_MAX_STACK = 3
# Tonally-neutral severity floor. ``INFO`` events never toast — they're
# log-only. Anything ``WARN`` and above surfaces.
_MIN_TOAST_SEVERITY = Severity.WARN


class Toast(QFrame):
    """A single event card. Dismiss via ``×`` or by calling :meth:`dismiss`."""

    dismissed = Signal(object)       # emits self
    show_log_requested = Signal(str)  # emits event_id

    def __init__(self, event: ErrorEvent, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._event = event
        self.setObjectName("toastCard")
        self.setFixedWidth(_TOAST_WIDTH)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build_ui()
        self._apply_style()

    @property
    def event(self) -> ErrorEvent:
        return self._event

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 10, 10)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel(self._event.title or "Error")
        title.setObjectName("toastTitle")
        title_font = title.font()
        title_font.setWeight(QFont.Weight.DemiBold)
        title.setFont(title_font)
        title.setWordWrap(True)
        header.addWidget(title, 1)

        close = QPushButton("×", self)
        close.setObjectName("toastClose")
        close.setFlat(True)
        close.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        close.setFixedSize(22, 22)
        close.clicked.connect(self.dismiss)
        header.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(header)

        if self._event.cause:
            cause = QLabel(self._truncate(self._event.cause, 160))
            cause.setObjectName("toastCause")
            cause.setWordWrap(True)
            outer.addWidget(cause)

        if self._event.action:
            action = QLabel(self._event.action)
            action.setObjectName("toastAction")
            action.setWordWrap(True)
            outer.addWidget(action)

        show_log_row = QHBoxLayout()
        show_log_row.addStretch(1)
        show_log = QPushButton("Show log ›", self)
        show_log.setObjectName("toastShowLog")
        show_log.setFlat(True)
        show_log.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        show_log.clicked.connect(
            lambda: self.show_log_requested.emit(self._event.event_id)
        )
        show_log_row.addWidget(show_log)
        outer.addLayout(show_log_row)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QFrame#toastCard {
                background: palette(base);
                border: 1px solid palette(mid);
                border-radius: 6px;
            }
            QLabel#toastTitle { font-size: 13px; }
            QLabel#toastCause { color: palette(text); font-size: 12px; }
            QLabel#toastAction {
                color: palette(mid);
                font-size: 12px;
                font-style: italic;
            }
            QPushButton#toastClose {
                color: palette(mid);
                font-size: 16px;
                border: none;
                padding: 0;
            }
            QPushButton#toastClose:hover { color: palette(text); }
            QPushButton#toastShowLog {
                color: palette(mid);
                border: none;
                padding: 4px 0 0 0;
                font-size: 12px;
            }
            QPushButton#toastShowLog:hover { color: palette(text); }
        """)

    @staticmethod
    def _truncate(text: str, n: int) -> str:
        text = text.strip()
        return text if len(text) <= n else text[: n - 1] + "…"

    def dismiss(self) -> None:
        self.hide()
        self.dismissed.emit(self)
        self.deleteLater()


class ToastHost(QWidget):
    """Owns a stack of :class:`Toast` widgets anchored to a parent window.

    Subscribes to the given :class:`ErrorCenter` (the singleton by default)
    on :meth:`attach` and unsubscribes on :meth:`detach`. Positioning is
    repositioned on parent resize — install an event filter on the parent.
    """

    show_drawer_requested = Signal(str)  # forwards Toast.show_log_requested

    def __init__(
        self,
        parent: QWidget,
        *,
        center: Optional[ErrorCenter] = None,
    ) -> None:
        super().__init__(parent)
        self._stack: list[Toast] = []
        self._center = center or get_error_center()
        self._unsubscribe: Optional[Callable[[], None]] = None

        # Transparent overlay — we draw nothing, only our Toast children do.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("toastHost")
        if parent is not None:
            parent.installEventFilter(self)
            self._reposition_host()

    # ---- lifecycle ---------------------------------------------------------

    def attach(self) -> None:
        """Start listening for error events."""
        if self._unsubscribe is not None:
            return
        self._unsubscribe = self._center.subscribe(self._on_event)

    def detach(self) -> None:
        """Stop listening and dismiss any active toasts."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        for t in list(self._stack):
            t.dismiss()
        self._stack.clear()

    # ---- event handling ----------------------------------------------------

    def _on_event(self, event: ErrorEvent) -> None:
        # ``ErrorCenter`` delivers inline on the emitter's thread — workers
        # often emit from a QThread. Route the widget work back onto the
        # Qt main thread with a queued single-shot timer.
        if _severity_rank(event.severity) < _severity_rank(_MIN_TOAST_SEVERITY):
            return
        QTimer.singleShot(0, lambda: self._push(event))

    def _push(self, event: ErrorEvent) -> None:
        while len(self._stack) >= _MAX_STACK:
            self._stack[0].dismiss()

        toast = Toast(event, self)
        toast.dismissed.connect(self._on_dismissed)
        toast.show_log_requested.connect(self.show_drawer_requested.emit)
        self._stack.append(toast)
        toast.show()
        self._relayout()

    def _on_dismissed(self, toast: Toast) -> None:
        try:
            self._stack.remove(toast)
        except ValueError:
            pass
        self._relayout()

    # ---- positioning -------------------------------------------------------

    def eventFilter(self, obj, event: QEvent) -> bool:  # noqa: D401 — Qt override
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            self._reposition_host()
            self._relayout()
        return super().eventFilter(obj, event)

    def _reposition_host(self) -> None:
        parent = self.parent()
        if not isinstance(parent, QWidget):
            return
        # Fill the top-right corner; actual toasts are positioned inside us.
        self.setGeometry(parent.rect())
        self.raise_()

    def _relayout(self) -> None:
        # Bottom edge of host minus margin; stack upward-to-down from top.
        y = _TOAST_MARGIN
        for toast in self._stack:
            x = self.width() - toast.width() - _TOAST_MARGIN
            toast.move(x, y)
            y += toast.sizeHint().height() + _TOAST_SPACING
        self.raise_()


__all__ = ["Toast", "ToastHost"]
