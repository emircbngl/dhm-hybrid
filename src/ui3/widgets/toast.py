"""Transient toast notifications — bottom-right, auto-dismissing.

A single ``ToastHost`` overlays the main window; ``show(text, level)`` stacks a
labelled pill that fades out after a TTL. Levels map to palette roles
(info/ok/warn/error). Qt-native (QLabel + QTimer + QPropertyAnimation), so no
DPG-style manual tick loop.
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import QPropertyAnimation, Qt, QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui3.design import Palette, Radius, Space

_LEVEL_ROLE = {
    "info": "accent",
    "ok": "ok",
    "warn": "warn",
    "error": "danger",
}


class ToastHost(QWidget):
    """Overlay pinned to the bottom-right of its parent."""

    def __init__(self, parent: QWidget, palette: Palette) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Space.sm)
        lay.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self._lay = lay
        self._toasts: List[QLabel] = []
        self._reposition()

    def apply_palette(self, palette: Palette) -> None:
        self._palette = palette

    def show_toast(self, text: str, level: str = "info",
                   ttl_ms: int = 4000) -> None:
        role = _LEVEL_ROLE.get(level, "accent")
        color = getattr(self._palette, role, self._palette.accent)
        pill = QLabel(text, self)
        pill.setStyleSheet(
            f"background: {self._palette.surface_high};"
            f" color: {self._palette.text};"
            f" border: 1px solid {color};"
            f" border-radius: {Radius.card}px;"
            f" padding: {Space.sm}px {Space.md}px;")
        pill.setWordWrap(True)
        pill.setMaximumWidth(360)
        self._lay.addWidget(pill)
        self._toasts.append(pill)
        self._reposition()
        QTimer.singleShot(ttl_ms, lambda: self._dismiss(pill))

    def _dismiss(self, pill: QLabel) -> None:
        if pill not in self._toasts:
            return
        anim = QPropertyAnimation(pill, b"windowOpacity", self)
        anim.setDuration(250)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.finished.connect(lambda: self._remove(pill))
        anim.start()
        pill._fade_anim = anim  # keep a ref alive

    def _remove(self, pill: QLabel) -> None:
        if pill in self._toasts:
            self._toasts.remove(pill)
            self._lay.removeWidget(pill)
            pill.deleteLater()
        self._reposition()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        margin = Space.lg
        w, h = 400, 300
        self.setGeometry(parent.width() - w - margin,
                         parent.height() - h - margin, w, h)

    def resize_to_parent(self) -> None:
        self._reposition()
