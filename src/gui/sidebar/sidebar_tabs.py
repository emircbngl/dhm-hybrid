from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout
from .recon_tab import ReconTab
from .process_tab import ProcessTab
from .focus_tab import FocusTab
from .qpi_tab import QPITab
from .record_tab import RecordTab
from .camera_tab import CameraTab

class SidebarTabs(QWidget):
    """Central container wrapping all configuration settings into a tabbed layout."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("sidebar_tabs")

        # Instantiate our modules
        self.camera_tab = CameraTab()
        self.recon_tab = ReconTab()
        self.process_tab = ProcessTab()
        self.focus_tab = FocusTab()
        self.qpi_tab = QPITab()
        self.record_tab = RecordTab()

        # Add tabs to the QTabWidget in the order specified by the PRD
        self.tabs.addTab(self.camera_tab, "Camera")
        self.tabs.addTab(self.recon_tab, "Recon")
        self.tabs.addTab(self.process_tab, "Process")
        self.tabs.addTab(self.focus_tab, "Focus")
        self.tabs.addTab(self.qpi_tab, "QPI")
        self.tabs.addTab(self.record_tab, "Record")

        layout.addWidget(self.tabs)
