"""ui3 application entry — QApplication bootstrap + theme + crash handler."""
from __future__ import annotations

import logging
import os
import sys

# On macOS Sonoma+ with PySide6 6.8+, some widgets receive their first paint
# event before Cocoa finishes wiring the backing surface — surfacing as
# "QPainter::begin: Paint device returned engine == 0" and a hard SEGFAULT
# (no Python crash dump possible). Qt's software-raster backend bypasses the
# Cocoa layer entirely and is plenty for our 2D scientific rendering. Must be
# set BEFORE QApplication is constructed — same proven guard as v1 src/main.py.
os.environ.setdefault("QT_WIDGETS_RHI", "0")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui3.design import build_qss, get_palette

_LOG = logging.getLogger("ui3")


def apply_theme(app: QApplication, theme_name: str) -> None:
    """Set the global stylesheet for a palette by name."""
    app.setStyleSheet(build_qss(get_palette(theme_name)))


def build_app(argv=None) -> QApplication:
    """Return an existing QApplication or create one (idempotent for tests)."""
    app = QApplication.instance()
    if app is None:
        # Shared GL contexts: the 3D surface viewer (pyqtgraph.opengl
        # GLViewWidget) needs this on macOS; must be set pre-construction.
        QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        app = QApplication(argv or sys.argv)
    app.setApplicationName("DHM Reconstruction")
    app.setOrganizationName("Hybrid Optics")
    return app


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        from core.crash_handler import install_crash_handler
        install_crash_handler()
    except Exception:  # crash handler is best-effort
        _LOG.debug("crash handler unavailable", exc_info=True)

    app = build_app()

    # Import here so the module stays light for headless token tests.
    from ui3.state import load_state, save_state
    from ui3.main_window import MainWindow

    state = load_state()
    apply_theme(app, state.theme)

    win = MainWindow(state)
    win.show()

    rc = app.exec()
    save_state(win.export_state())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
