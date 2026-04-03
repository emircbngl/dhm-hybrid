from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
import numpy as np

from .image_panel import ImagePanel

class PhasePanel(QWidget):
    """A composite panel combining Wrapped and Unwrapped phase visualizations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Header controls
        self.header_layout = QHBoxLayout()
        self.header_layout.setContentsMargins(5, 5, 5, 5)
        
        self.header_label = QLabel("Phase Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Wrapped", "Unwrapped"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        
        self.header_layout.addWidget(self.header_label)
        self.header_layout.addWidget(self.mode_combo)
        self.header_layout.addStretch()
        
        self.layout.addLayout(self.header_layout)
        
        # Inner Image Panel
        self.image_panel = ImagePanel()
        self.layout.addWidget(self.image_panel, stretch=1)
        
        # Caches
        self._wrapped_data = None
        self._unwrapped_data = None
        
    def set_wrapped(self, data: np.ndarray, **kwargs) -> None:
        self._wrapped_data = data
        if self.mode_combo.currentText() == "Wrapped":
            self.image_panel.set_image(data, **kwargs)
            
    def set_unwrapped(self, data: np.ndarray, **kwargs) -> None:
        self._unwrapped_data = data
        if self.mode_combo.currentText() == "Unwrapped":
            self.image_panel.set_image(data, **kwargs)
            
    def _on_mode_changed(self, index: int) -> None:
        if index == 0 and self._wrapped_data is not None:
            self.image_panel.set_image(self._wrapped_data, autoLevels=True)
        elif index == 1 and self._unwrapped_data is not None:
            self.image_panel.set_image(self._unwrapped_data, autoLevels=True)
            
    def get_view(self):
        return self.image_panel.get_view()
        
    def set_colormap(self, cmap_name: str) -> None:
        self.image_panel.set_colormap(cmap_name)
