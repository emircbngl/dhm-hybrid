"""Reusable collapsible section (v1.4 UI Redesign — progressive disclosure).

Qt's :class:`QGroupBox` supports ``setCheckable`` but the "collapse"
behaviour is cosmetic — children become disabled, not hidden. Dense
sidebar tabs need the *hide* variant so the secondary parameters
actually leave the layout when not in use. This widget provides
that:

    Advanced ▸          ← collapsed (default)
    Advanced ▾
    ├─ Method            [ASM ▾]
    ├─ FFT Backend       [pyfftw ▾]
    └─ …

Tiny contract — no external deps, styling confined to the header
button so it re-skins cleanly with the v1.4 theme system.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleBox(QWidget):
    """A header button + a content area that shows/hides on click.

    Use :meth:`setContentLayout` (preferred) with any ``QLayout`` of
    child widgets, or call :meth:`content` and add widgets directly
    to the inner frame.
    """

    toggled = Signal(bool)  # True when the box is expanded

    def __init__(
        self,
        title: str,
        parent: Optional[QWidget] = None,
        *,
        expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._title = title

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toggle = QToolButton(self)
        self._toggle.setObjectName("collapsibleHeader")
        self._toggle.setCheckable(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # Plain text + a rotating arrow is fine and avoids dragging in
        # icon assets.
        self._toggle.setStyleSheet(
            "QToolButton { border: none; padding: 4px 2px; "
            "text-align: left; font-weight: 600; }"
        )
        self._toggle.clicked.connect(self._on_toggle_clicked)
        root.addWidget(self._toggle)

        self._content = QFrame(self)
        self._content.setObjectName("collapsibleContent")
        self._content.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(self._content)

        # Start in the requested state; use the internal helper to keep
        # toggle + arrow + visibility in sync.
        self._apply_state(bool(expanded))

    # ---- public API ----------------------------------------------------

    def setContentLayout(self, layout: QLayout) -> None:
        """Install ``layout`` on the collapsible content frame.

        Qt doesn't support re-parenting a layout that already has a
        parent, so call this exactly once. If you need to rebuild the
        section, swap the child widgets instead of the layout itself.
        """
        if self._content.layout() is not None:
            raise RuntimeError(
                "CollapsibleBox: content layout already set — "
                "swap child widgets instead of replacing the layout."
            )
        self._content.setLayout(layout)

    def content(self) -> QFrame:
        """Return the inner content frame so callers that prefer
        ``addWidget`` over ``setContentLayout`` can reach it directly."""
        return self._content

    def setExpanded(self, expanded: bool) -> None:
        """Force the section into the requested expanded / collapsed
        state. Emits :attr:`toggled` only on actual change."""
        new = bool(expanded)
        if new == self._toggle.isChecked():
            return
        self._toggle.setChecked(new)
        self._apply_state(new)

    def isExpanded(self) -> bool:
        return self._toggle.isChecked()

    def title(self) -> str:
        return self._title

    def setTitle(self, title: str) -> None:
        """Change the header label without re-creating the box."""
        self._title = title
        self._apply_state(self._toggle.isChecked())

    # ---- internals -----------------------------------------------------

    def _on_toggle_clicked(self, checked: bool) -> None:
        self._apply_state(bool(checked))

    def _apply_state(self, expanded: bool) -> None:
        arrow = "\u25be" if expanded else "\u25b8"  # ▾ / ▸
        self._toggle.setChecked(expanded)
        self._toggle.setText(f"{arrow}  {self._title}")
        self._content.setVisible(expanded)
        self.toggled.emit(expanded)


__all__ = ["CollapsibleBox"]
