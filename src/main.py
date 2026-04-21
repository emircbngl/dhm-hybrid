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
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
