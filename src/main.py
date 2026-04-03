import sys
from PySide6.QtWidgets import QApplication

# Import the root Main Window constructed from modular components in Phase 1
from gui import MainWindow

def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
