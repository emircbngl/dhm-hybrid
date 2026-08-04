"""Shared dialog-presentation helper (2026-07-08 UX fix).

Every ui3 dialog used to end its open method with the bare
``show(); raise_(); activateWindow()`` triad. That brings the window to the
front but leaves it at Qt's DEFAULT top-level position — so every dialog
opened at the same spot, stacking on top of one another and partly behind the
main window ("they open but overlap, I can't tell what's where"). Routing all
of them through ``present_centered`` shows each one centred on its parent
window (with a small per-open cascade so several open at once don't perfectly
coincide) and pulled to the front.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget

# Module-level cascade counter so a burst of dialogs fans out instead of
# perfectly overlapping. Wraps to keep the offset small.
_cascade = 0


def present_centered(dialog: QWidget) -> None:
    """Show ``dialog`` centred on its parent window, cascaded, and raised."""
    global _cascade
    dialog.show()  # realise geometry so centring uses the true size

    parent = dialog.parent()
    ref = parent.window() if isinstance(parent, QWidget) else None
    if ref is not None and ref.isVisible():
        pg = ref.frameGeometry()
        offset = (_cascade % 5) * 28
        top_left = pg.center() - dialog.rect().center()
        dialog.move(top_left.x() + offset, top_left.y() + offset)
        _cascade += 1

    dialog.raise_()
    dialog.activateWindow()
