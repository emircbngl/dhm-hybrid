import sys
import logging
from PySide6.QtWidgets import QApplication

# Import the root Main Window constructed from modular components in Phase 1
from gui import MainWindow
from core.crash_handler import install_crash_handler

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    install_crash_handler()
    # On macOS Sonoma+ with PySide6 6.8+, some widgets can receive their
    # first paint event before Cocoa finishes wiring the backing surface,
    # which surfaces as ``QPainter::begin: Paint device returned engine
    # == 0`` and a segfault. Qt's software-raster backend bypasses the
    # Cocoa layer and is robust enough for our 2D plotting / scientific
    # rendering needs. The setting has to land before QApplication is
    # constructed.
    import os
    os.environ.setdefault("QT_WIDGETS_RHI", "0")
    from PySide6.QtCore import Qt
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
