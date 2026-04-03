from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PySide6.QtCore import Qt
import pyqtgraph as pg

class ImageGrid(QWidget):
    """
    Manages a dynamically resizable grid of image panels using nested QSplitters.
    Supports layouts like 2x2, 1x4, and double-click to maximize a specific view.
    """
    def __init__(self, panels: list, parent=None):
        super().__init__(parent)
        self.panels = panels
        self._maximized_panel = None
        self._init_ui()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Default layout: 2x2 grid
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # If we have 4+ panels, put first two on top, next two on bottom
        if len(self.panels) >= 4:
            self.top_splitter.addWidget(self.panels[0])
            self.top_splitter.addWidget(self.panels[1])
            self.bottom_splitter.addWidget(self.panels[2])
            self.bottom_splitter.addWidget(self.panels[3])
        elif len(self.panels) > 0:
            for p in self.panels:
                self.top_splitter.addWidget(p)

        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.bottom_splitter)
        
        self.layout.addWidget(self.main_splitter)
        
        # Setup stretch factors
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)
        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 1)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 1)

        # Hook up double-click events for maximizing
        for panel in self.panels:
            try:
                # We need to reach into the ImagePanel to hook double-clicks
                vb = panel.get_view()
                vb.scene().sigMouseClicked.connect(lambda ev, p=panel: self._on_panel_clicked(ev, p))
            except Exception:
                pass

    def _on_panel_clicked(self, ev, panel):
        if ev.double():
            self.toggle_maximize(panel)

    def toggle_maximize(self, panel):
        if self._maximized_panel == panel:
            # Restore all
            self.restore_grid()
        else:
            # Maximize one
            for p in self.panels:
                if p != panel:
                    p.hide()
            panel.show()
            self._maximized_panel = panel

    def restore_grid(self):
        """Restores all panels to be visible."""
        if self._maximized_panel is not None:
            for p in self.panels:
                p.show()
            self._maximized_panel = None

    def save_state(self) -> bytes:
        return self.main_splitter.saveState()

    def restore_state(self, state: bytes) -> bool:
        return self.main_splitter.restoreState(state)