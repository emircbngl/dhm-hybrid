"""Tiny green/red validation indicator (v1.4 UI Redesign).

Rendered as a rounded QLabel with a background colour — no custom
``paintEvent`` override. Custom paint handlers triggered a
``QPainter::begin: Paint device returned engine == 0`` crash on Qt
6.8+ when the widget received its first paint before the backing
surface was ready; going through Qt's style pipeline avoids that
window entirely.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget


class ValidationDot(QLabel):
    """Compact state indicator.

    States:
        * ``None``  — neutral (no check run yet) — muted grey
        * ``True``  — valid — green
        * ``False`` — invalid — red, tooltip carries the reason

    ``setState(valid_or_none, reason="")`` is the only public setter.
    """

    _COLOR_CSS = {
        None:  "#a0a0a0",
        True:  "#2ea043",
        False: "#d73a49",
    }

    def __init__(self, parent: Optional[QWidget] = None, *, diameter: int = 12) -> None:
        super().__init__(parent)
        d = int(diameter)
        self._diameter = d
        # Leave a 2-px halo so the coloured circle reads against dense
        # sidebar backgrounds.
        self.setFixedSize(d + 4, d + 4)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAccessibleName("Input validation indicator")
        self._state: Optional[bool] = None
        self._apply_style()

    def setState(self, valid: Optional[bool], reason: str = "") -> None:
        new_state = None if valid is None else bool(valid)
        if new_state == self._state and reason == self.toolTip():
            return
        self._state = new_state
        if new_state is False and reason:
            self.setToolTip(reason)
        elif new_state is True:
            self.setToolTip("Valid")
        else:
            self.setToolTip(reason or "Not yet validated")
        self._apply_style()

    def state(self) -> Optional[bool]:
        return self._state

    # ---- style ---------------------------------------------------------

    def _apply_style(self) -> None:
        color = self._COLOR_CSS.get(self._state, self._COLOR_CSS[None])
        r = self._diameter // 2 + 1  # rounded pill that hugs the content
        self.setStyleSheet(
            f"QLabel {{"
            f"  background-color: {color};"
            f"  border: 1px solid rgba(0, 0, 0, 25%);"
            f"  border-radius: {r}px;"
            f"  margin: 2px;"
            f"}}"
        )


__all__ = ["ValidationDot"]
