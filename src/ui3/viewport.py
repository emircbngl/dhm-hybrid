"""ImagePanel — a pyqtgraph-backed scientific image view.

One titled preview with mouse-wheel zoom / drag pan, a colormap, robust
1–99% display scaling, an optional physical scalebar, drag-drop of a hologram
file, and a click-drag line-profile hook. Four of these make the main grid
(input / amplitude / phase / spectrum).

pyqtgraph does the heavy lifting in-process on the GPU-capable Qt scene, so a
1200×1600 float array pans/zooms smoothly without an IPC hop — the whole reason
ui3 is Qt+pyqtgraph rather than a web front.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from core.scalebar import compute_scalebar
from ui3.design import Palette, Space, Type

# Global pyqtgraph niceties — image row/col order matches numpy (row=y).
pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

# Named colormaps we expose (pyqtgraph built-ins / matplotlib names).
COLORMAPS = {
    "gray": "gray",
    "viridis": "viridis",
    "magma": "magma",
    "inferno": "inferno",
    "twilight": "twilight",
    "turbo": "turbo",
    "cividis": "cividis",
}


class ImagePanel(QFrame):
    """A single zoomable image tile."""

    file_dropped = Signal(str)        # a hologram path was dropped here
    line_requested = Signal(object)   # ((y0,x0),(y1,x1)) after a drag (future)
    pixel_clicked = Signal(int, int)  # (row, col) — only while pick-mode armed

    def __init__(self, title: str, palette: Palette, *,
                 default_cmap: str = "gray",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        self._title = title
        self._palette = palette
        self._cmap_name = default_cmap
        self._data: Optional[np.ndarray] = None
        self._pixel_um: Optional[float] = None
        self.setAcceptDrops(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(Space.sm, Space.sm, Space.sm, Space.sm)
        lay.setSpacing(Space.xs)

        header = QLabel(title)
        header.setProperty("role", "muted")
        f = QFont()
        f.setPixelSize(Type.label)   # was setPointSize(10); 10pt == 13px here
        header.setFont(f)
        lay.addWidget(header)
        self._header = header

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(palette.view_bg)
        self._vb = self._glw.addViewBox(lockAspect=True, enableMenu=False)
        self._vb.invertY(True)     # image convention: row 0 at top
        self._img = pg.ImageItem()
        self._vb.addItem(self._img)
        lay.addWidget(self._glw, 1)

        # Drop-zone affordance (shown until an image lands).
        self._drop_hint = QLabel("Drop a hologram here", self._glw)
        self._drop_hint.setAlignment(Qt.AlignCenter)
        self._drop_hint.setStyleSheet(
            f"color: {palette.text_muted}; font-size: {Type.title}px;"
            " background: transparent;")
        self._drop_hint.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # Scalebar overlay (a pg line + text in view coords).
        self._sb_line: Optional[pg.PlotDataItem] = None
        self._sb_text: Optional[pg.TextItem] = None

        # Pick-mode: when armed, a single left-click emits pixel_clicked
        # (row, col) instead of doing nothing. Used to set the off-axis
        # +1-order center by clicking the spectrum. Drag still pans and the
        # wheel still zooms — pyqtgraph's sigMouseClicked only fires on a
        # click that isn't a drag (2026-07-08).
        self._pick_mode = False
        self._marker: Optional[pg.ScatterPlotItem] = None
        self._glw.scene().sigMouseClicked.connect(self._on_scene_click)

        self._apply_cmap()

    # -- public API -----------------------------------------------------
    def set_title(self, title: str) -> None:
        self._title = title
        self._header.setText(title)

    def set_colormap(self, name: str) -> None:
        self._cmap_name = COLORMAPS.get(name, "gray")
        self._apply_cmap()

    def set_image(self, data: np.ndarray, *, contrast: str = "percentile",
                  pixel_size_um: Optional[float] = None,
                  autorange: bool = True) -> None:
        """Display a 2-D real array. ``contrast`` = 'percentile' (1–99, robust)
        or 'minmax'. Full-resolution — pyqtgraph handles the zoom LOD."""
        arr = np.asarray(data)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0)
        self._data = arr
        self._pixel_um = pixel_size_um

        lo, hi = self._levels(arr, contrast)
        self._img.setImage(arr, levels=(lo, hi), autoLevels=False)
        if autorange:
            self._vb.autoRange(padding=0.0)
        self._drop_hint.setVisible(False)
        self._refresh_scalebar()

    def clear(self) -> None:
        self._data = None
        self._img.clear()
        self._drop_hint.setVisible(True)
        self._clear_scalebar()
        self.clear_marker()

    # -- pick mode (click-to-set a pixel; e.g. off-axis order center) ----
    def set_pick_mode(self, enabled: bool) -> None:
        """Arm/disarm click-to-pick. While armed the view shows a crosshair
        cursor and a left-click emits :attr:`pixel_clicked` (row, col)."""
        self._pick_mode = bool(enabled)
        self._glw.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_marker(self, row: int, col: int) -> None:
        """Show a crosshair marker at image pixel (row, col) — constant
        screen size, so it stays visible at any zoom."""
        self.clear_marker()
        self._marker = pg.ScatterPlotItem(
            [float(col)], [float(row)], symbol="x", size=18,
            pen=pg.mkPen("#ff3b30", width=2), brush=None, pxMode=True)
        self._vb.addItem(self._marker)

    def clear_marker(self) -> None:
        if self._marker is not None:
            self._vb.removeItem(self._marker)
            self._marker = None

    def _on_scene_click(self, ev) -> None:
        if not self._pick_mode or self._data is None:
            return
        if ev.button() != Qt.LeftButton:
            return
        pt = self._vb.mapSceneToView(ev.scenePos())
        h, w = self._data.shape[:2]
        col = int(np.clip(round(pt.x()), 0, w - 1))
        row = int(np.clip(round(pt.y()), 0, h - 1))
        ev.accept()
        self.pixel_clicked.emit(row, col)

    def has_image(self) -> bool:
        return self._data is not None

    def apply_palette(self, palette: Palette) -> None:
        self._palette = palette
        self._glw.setBackground(palette.view_bg)
        self._drop_hint.setStyleSheet(
            f"color: {palette.text_muted}; font-size: {Type.title}px;"
            " background: transparent;")
        self._refresh_scalebar()

    # -- internals ------------------------------------------------------
    @staticmethod
    def _levels(arr: np.ndarray, contrast: str) -> tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0, 1.0
        if contrast == "minmax":
            lo, hi = float(finite.min()), float(finite.max())
        else:
            lo = float(np.percentile(finite, 1))
            hi = float(np.percentile(finite, 99))
        if hi - lo < 1e-12:
            hi = lo + 1e-9
        return lo, hi

    def _apply_cmap(self) -> None:
        try:
            cmap = pg.colormap.get(self._cmap_name)
            self._img.setLookupTable(cmap.getLookupTable(nPts=256))
        except Exception:
            pass  # unknown cmap → leave default grayscale

    def _clear_scalebar(self) -> None:
        for item in (self._sb_line, self._sb_text):
            if item is not None:
                self._vb.removeItem(item)
        self._sb_line = None
        self._sb_text = None

    def _refresh_scalebar(self) -> None:
        self._clear_scalebar()
        if self._data is None or not self._pixel_um or self._pixel_um <= 0:
            return
        h, w = self._data.shape[:2]
        spec = compute_scalebar(w, self._pixel_um)
        if spec is None or spec.length_px <= 0:
            return
        margin = max(6, int(w * 0.04))
        y = h - margin
        x1 = w - margin
        x0 = x1 - spec.length_px
        pen = pg.mkPen(color="w", width=3)
        line = pg.PlotDataItem([x0, x1], [y, y], pen=pen)
        self._vb.addItem(line)
        self._sb_line = line
        # Dark backing pill so the white caption stays legible on any theme —
        # it used to render white-on-light and vanish on the light theme's
        # pale panel margin.
        text = pg.TextItem(spec.label, color="w", anchor=(0.5, 0.0),
                           fill=pg.mkBrush(0, 0, 0, 140))
        text.setPos((x0 + x1) / 2.0, y + margin * 0.3)
        self._vb.addItem(text)
        self._sb_text = text

    # -- drag & drop ----------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.file_dropped.emit(path)
                break
        event.acceptProposedAction()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Keep the drop hint centred over the graphics widget.
        self._drop_hint.setGeometry(self._glw.geometry())
