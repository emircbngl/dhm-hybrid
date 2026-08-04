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

        # Scalebar overlay items — created lazily by ``set_scalebar``,
        # torn down before each refresh so the panel never accumulates
        # stale bars across image changes.
        self._scalebar_line: pg.PlotDataItem | None = None
        self._scalebar_text: pg.TextItem | None = None

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

    # ---- scalebar overlay --------------------------------------------

    def set_scalebar(
        self,
        pixel_size_um: float | None,
        *,
        fixed_length_um: float | None = None,
    ) -> None:
        """Paint a horizontal µm scalebar at the bottom-right of the
        image. Pass ``None`` (or ≤ 0) to clear."""
        from core.scalebar import compute_scalebar

        # Always start fresh — old bar stays around if we don't tear it
        # down explicitly.
        self.clear_scalebar()
        if pixel_size_um is None or pixel_size_um <= 0:
            return

        img_item = self.view.getImageItem()
        if img_item is None or img_item.image is None:
            return
        h, w = img_item.image.shape[-2:]
        spec = compute_scalebar(int(w), float(pixel_size_um),
                                fixed_length_um=fixed_length_um)
        if spec is None or spec.length_px <= 0:
            return

        margin = max(8.0, w * 0.04)
        # pyqtgraph ImageItem coords: x = column, y = row. Bottom-right
        # margin (image y axis points down).
        x_end = float(w - margin)
        x_start = float(x_end - spec.length_px)
        y_pos = float(h - margin)

        view = self.get_view()
        line = pg.PlotDataItem(
            [x_start, x_end], [y_pos, y_pos],
            pen=pg.mkPen("w", width=4),
        )
        line.setZValue(50)
        view.addItem(line)

        text = pg.TextItem(spec.label, color=(255, 255, 255),
                           anchor=(0.5, 1.0))
        text.setPos((x_start + x_end) / 2.0, y_pos - margin / 4.0)
        text.setZValue(51)
        view.addItem(text)

        self._scalebar_line = line
        self._scalebar_text = text

    def clear_scalebar(self) -> None:
        view = self.get_view()
        for item in (self._scalebar_line, self._scalebar_text):
            if item is None:
                continue
            try:
                view.removeItem(item)
            except Exception:
                pass
        self._scalebar_line = None
        self._scalebar_text = None
