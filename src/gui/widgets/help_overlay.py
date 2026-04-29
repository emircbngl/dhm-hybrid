"""Contextual help overlay (v1.4 — keyboard shortcut ``?``).

A single pop-over that surfaces the tooltips of every visible control
inside the window. Useful when the operator doesn't remember what a
button does and wouldn't hover-discover it naturally (e.g. a
freshly-installed build, a collaborator on a shared workstation).

Implementation is intentionally simple: walk the parent's widget
tree, collect every widget's accessible name / tooltip / status tip,
and render them as a scrollable list inside a modeless dialog. No
runtime hook into the event loop, no interception of mouse events —
``?`` opens it, Esc closes it.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


def _walk(widget: QWidget) -> Iterable[QWidget]:
    """Yield ``widget`` + every visible descendant widget, depth-first."""
    yield widget
    for child in widget.findChildren(QWidget):
        yield child


def _collect_hints(root: QWidget) -> List[Tuple[str, str]]:
    """Return ``(name, hint)`` pairs for every visible widget that
    carries an accessible name or a tooltip.

    Accessible-name wins when both are set — the SR-oriented label is
    usually the crisper one. Duplicates (same name + same hint) are
    dropped to keep the list scannable.
    """
    pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for w in _walk(root):
        try:
            if not w.isVisible():
                continue
        except Exception:
            continue
        name = (w.accessibleName() or "").strip()
        tip = (w.toolTip() or "").strip()
        if not name and not tip:
            continue
        display_name = name or w.__class__.__name__
        # Strip rich-text tags minimally for the overlay (keeps the
        # list legible without a HTML renderer).
        if tip.startswith("<") and ">" in tip:
            tip_plain = tip
        else:
            tip_plain = tip
        key = (display_name, tip_plain)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((display_name, tip_plain))
    return pairs


class HelpOverlay(QDialog):
    """Modeless contextual-help popup. Esc closes. ``?`` is the
    keyboard entry point (wired at the host level)."""

    def __init__(
        self,
        target: QWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Help — current controls")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setMinimumSize(480, 360)

        pairs = _collect_hints(target)
        self._build_ui(pairs)

        # Esc closes — redundant with QDialog's default but explicit.
        self._esc = QShortcut(QKeySequence("Escape"), self)
        self._esc.activated.connect(self.accept)

    # ---- UI ---------------------------------------------------------

    def _build_ui(self, pairs: List[Tuple[str, str]]) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        header = QLabel(
            f"{len(pairs)} control(s) with help text. "
            "Press Esc to close."
        )
        header.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(header)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(8)
        if not pairs:
            empty = QLabel(
                "No visible control on the current view carries a "
                "tooltip. Hover over a widget first, or switch "
                "workflow to reveal the relevant tab."
            )
            empty.setWordWrap(True)
            empty.setStyleSheet("color: palette(mid); padding: 12px;")
            host_layout.addWidget(empty)
        else:
            for name, tip in pairs:
                block = self._make_block(name, tip)
                host_layout.addWidget(block)
        host_layout.addStretch(1)
        scroll.setWidget(host)
        layout.addWidget(scroll, 1)

    @staticmethod
    def _make_block(name: str, tip: str) -> QWidget:
        frame = QFrame()
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(2)
        title = QLabel(name)
        f = title.font()
        f.setWeight(QFont.Weight.DemiBold)
        title.setFont(f)
        title.setWordWrap(True)
        fl.addWidget(title)
        if tip:
            body = QLabel(tip)
            body.setWordWrap(True)
            body.setStyleSheet("color: palette(mid);")
            fl.addWidget(body)
        return frame


__all__ = ["HelpOverlay"]
