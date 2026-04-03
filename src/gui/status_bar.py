from PySide6.QtWidgets import QStatusBar, QLabel, QWidget

class MainStatusBar(QStatusBar):
    """Bottom status bar managing generic application state, file status, and badge alerts."""
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("main_status_bar")
        
        # Base UI info
        self.info_label = QLabel("Ready")
        self.addWidget(self.info_label)
        
        self.addPermanentWidget(QLabel("AF: OFF"))
        self.addPermanentWidget(QLabel("REC: OFF"))

    def show_message(self, message: str, timeout: int = 0) -> None:
        """Shows a temporary message taking over the left side of the bar."""
        self.showMessage(message, timeout)
        
    def set_info_text(self, text: str) -> None:
        """Sets the permanent info string describing the loaded image."""
        self.info_label.setText(text)
