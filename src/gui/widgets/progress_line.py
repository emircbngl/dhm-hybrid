"""Silent 1-pixel progress indicator.

The loudest signal a UI sends is the one it *doesn't* send. For
operations that finish in under half a second — the vast majority of
reconstructions — a spinner that pops up and vanishes is worse than
silence. This widget says nothing until the work runs long enough to
matter.

Three states
------------
1. **hidden** (t < 500 ms): the widget is mounted but paints nothing.
   No layout jump when the operation starts and finishes quickly.
2. **line** (500 ms ≤ t < 5 s): a 2-pixel determinate progress bar,
   taking the current operation's ``pct``. No text.
3. **line + caption** (t ≥ 5 s): the bar plus one muted line of text
   — phase + pct + an "Esc to cancel" hint. The user has waited long
   enough to deserve context.

Completion (``done=True``) or cancellation (``cancelled=True``) resets
the state to hidden after a brief fade window that this widget doesn't
animate (animation for a 200 ms fade is more visual noise than value).

The widget does not *execute* cancellation itself; it emits
``cancel_requested`` when the user presses the Esc hint, and the host
wires that to whatever worker is live.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from core.progress import ProgressEvent

_LOG = logging.getLogger(__name__)

# Thresholds tuned for our worker distribution — see Faz 1 bench_af.
_HIDE_MS = 500
_CAPTION_MS = 5000
_TICK_MS = 100  # state-transition poll interval


class ProgressLine(QWidget):
    """Thin, quiet progress line. Connect a worker's ``progress`` signal
    to :meth:`on_progress` and the line takes care of the rest."""

    # Emitted when the user clicks the Esc hint on the caption row. The
    # widget doesn't know which worker to cancel; the host wires this up.
    cancel_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("progressLine")
        self._event: Optional[ProgressEvent] = None
        self._start_time: Optional[float] = None
        self._build_ui()
        self._apply_style()
        self._render_hidden()

        self._tick = QTimer(self)
        self._tick.setInterval(_TICK_MS)
        self._tick.timeout.connect(self._on_tick)

    # ---- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._bar = QProgressBar(self)
        self._bar.setObjectName("progressLineBar")
        self._bar.setTextVisible(False)
        self._bar.setMinimum(0)
        self._bar.setMaximum(1000)  # integer-resolution pct*10
        self._bar.setFixedHeight(2)
        self._bar.setValue(0)
        root.addWidget(self._bar)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._caption = QLabel("", self)
        self._caption.setObjectName("progressLineCaption")
        row.addWidget(self._caption, 1)
        self._cancel_hint = QLabel("Press Esc to cancel", self)
        self._cancel_hint.setObjectName("progressLineCancelHint")
        self._cancel_hint.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_hint.mousePressEvent = (  # type: ignore[assignment]
            lambda _e: self.cancel_requested.emit()
        )
        row.addWidget(self._cancel_hint, 0, Qt.AlignmentFlag.AlignRight)
        root.addLayout(row)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QProgressBar#progressLineBar {
                background: transparent;
                border: none;
            }
            QProgressBar#progressLineBar::chunk {
                background-color: palette(highlight);
            }
            QLabel#progressLineCaption {
                color: palette(mid);
                font-size: 11px;
            }
            QLabel#progressLineCancelHint {
                color: palette(mid);
                font-size: 11px;
                text-decoration: underline;
            }
            QLabel#progressLineCancelHint:hover {
                color: palette(text);
            }
        """)

    # ---- public API --------------------------------------------------------

    def on_progress(self, event: object) -> None:
        """Slot compatible with ``QSignal(object)`` worker progress signals.

        We accept ``object`` rather than :class:`ProgressEvent` directly
        because Qt's signal machinery would otherwise demand a registered
        type. The check below rejects anything that doesn't look like a
        ProgressEvent — defensive in case a stray ``str`` still gets
        connected.
        """
        if not isinstance(event, ProgressEvent):
            return
        self._event = event

        if event.done:
            self._reset()
            return

        if self._start_time is None:
            self._start_time = time.monotonic()
            self._tick.start()

        self._render_for_elapsed(self._elapsed_ms())

    # ---- timing ------------------------------------------------------------

    def _elapsed_ms(self) -> int:
        if self._start_time is None:
            return 0
        return int((time.monotonic() - self._start_time) * 1000)

    def _on_tick(self) -> None:
        if self._event is None or self._start_time is None:
            return
        self._render_for_elapsed(self._elapsed_ms())

    # ---- rendering ---------------------------------------------------------

    def _render_for_elapsed(self, ms: int) -> None:
        if self._event is None:
            return

        # Update the bar regardless — it's cheap and the bar's visibility
        # alone decides whether the user sees it.
        pct = max(0.0, min(1.0, float(self._event.pct or 0.0)))
        self._bar.setValue(int(pct * 1000))

        if ms < _HIDE_MS:
            self._render_hidden()
            return

        if ms < _CAPTION_MS:
            self._render_line_only()
            return

        self._render_line_with_caption(self._event)

    def _render_hidden(self) -> None:
        self._bar.setVisible(False)
        self._caption.setVisible(False)
        self._cancel_hint.setVisible(False)

    def _render_line_only(self) -> None:
        self._bar.setVisible(True)
        self._caption.setVisible(False)
        self._cancel_hint.setVisible(False)

    def _render_line_with_caption(self, event: ProgressEvent) -> None:
        self._bar.setVisible(True)
        self._caption.setVisible(True)
        self._cancel_hint.setVisible(True)

        pct_int = int(round((event.pct or 0.0) * 100))
        kind = event.operation_kind or "Working"
        phase = event.phase or ""
        if phase:
            self._caption.setText(f"{kind.title()} · {phase} · {pct_int}%")
        else:
            self._caption.setText(f"{kind.title()} · {pct_int}%")

    def _reset(self) -> None:
        self._tick.stop()
        self._start_time = None
        self._event = None
        self._bar.setValue(0)
        self._render_hidden()


__all__ = ["ProgressLine"]
