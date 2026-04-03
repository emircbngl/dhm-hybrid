import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
import numpy as np

class ImagePanel(QWidget):
    """A generic wrapper for pyqtgraph's ImageView to be embedded anywhere."""
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        
        self.view = pg.ImageView()
        self.layout().addWidget(self.view)
        
    def set_image(self, data: np.ndarray, **kwargs) -> None:
        """Helper to pass images securely to the pg viewer."""
        self.view.setImage(data, **kwargs)
        
    def get_view(self) -> pg.ViewBox:
        """Returns the internal ViewBox for event connection."""
        return self.view.getView()
        
    def set_colormap(self, cmap_name: str) -> None:
        """Sets standard colormaps like 'hsv' or 'viridis'."""
        try:
            self.view.setColorMap(pg.colormap.get(cmap_name))
        except Exception:
            pass
