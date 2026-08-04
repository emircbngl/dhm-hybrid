from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
import numpy as np
import pyqtgraph as pg

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
        # Depth-map overlay (v1.2-tomo final+). Lazily instantiated when
        # ``set_depth_overlay`` is first called, removed on clear.
        self._depth_overlay_item: pg.ImageItem | None = None
        # Cluster markers (v1.3-polish). One ScatterPlotItem + one
        # TextItem per cluster centroid. Kept as a list so
        # ``clear_cluster_markers`` can tear them down cleanly.
        self._cluster_marker_items: list = []
        
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

    def set_scalebar(self, pixel_size_um: float | None,
                     *, fixed_length_um: float | None = None) -> None:
        """Proxy to the inner ImagePanel — keeps callers panel-agnostic."""
        self.image_panel.set_scalebar(pixel_size_um,
                                      fixed_length_um=fixed_length_um)

    def clear_scalebar(self) -> None:
        self.image_panel.clear_scalebar()

    # ---- depth-map overlay (v1.2-tomo final+) -----------------------

    def set_depth_overlay(
        self,
        depth_map: np.ndarray,
        *,
        alpha: float = 0.55,
        colormap: str = "viridis",
    ) -> None:
        """Paint a colormapped ``depth_map`` over the phase image.

        ``depth_map`` is ``(H, W)`` float — z values in any unit (the
        overlay only cares about the *ordering* inside the frame). NaN
        pixels (from :func:`core.depth_map.mask_low_confidence`) fall
        back to fully transparent so background doesn't bleed over the
        phase.

        ``alpha`` is the fixed transparency for non-NaN pixels (0 = fully
        transparent, 1 = opaque). Default 0.55 lets the underlying phase
        texture read through the tint.

        Calling this a second time replaces the overlay without
        rebuilding the :class:`pg.ImageItem`.
        """
        if depth_map is None or depth_map.size == 0:
            self.clear_depth_overlay()
            return

        arr = np.asarray(depth_map, dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            self.clear_depth_overlay()
            return

        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi - lo < 1e-15:
            norm = np.zeros_like(arr)
        else:
            norm = (arr - lo) / (hi - lo)
            norm = np.clip(norm, 0.0, 1.0)

        # Build RGBA — colormap table lookup + per-pixel alpha mask.
        try:
            cm = pg.colormap.get(colormap)
        except Exception:
            cm = pg.colormap.get("viridis")
        lut = cm.getLookupTable(nPts=256)  # (256, 3) or (256, 4)
        # Replace NaN before cast so int conversion doesn't emit a
        # ``RuntimeWarning``. The alpha channel (built from ``finite``)
        # will still hide these pixels — the colour itself doesn't
        # matter.
        idx = np.clip(
            np.nan_to_num(norm * 255, nan=0.0).astype(np.int32),
            0, 255,
        )
        rgb = lut[idx][..., :3]  # (H, W, 3)

        alpha_u8 = int(round(max(0.0, min(1.0, float(alpha))) * 255))
        alpha_channel = np.where(finite, alpha_u8, 0).astype(np.uint8)
        rgba = np.dstack([rgb.astype(np.uint8), alpha_channel])

        if self._depth_overlay_item is None:
            self._depth_overlay_item = pg.ImageItem(rgba)
            # Sit above the base image in z-order.
            self._depth_overlay_item.setZValue(10)
            view = self.image_panel.get_view()
            view.addItem(self._depth_overlay_item)
        else:
            self._depth_overlay_item.setImage(rgba)
        self._depth_overlay_item.setVisible(True)

    def clear_depth_overlay(self) -> None:
        """Remove the depth overlay if one is attached."""
        if self._depth_overlay_item is None:
            return
        try:
            view = self.image_panel.get_view()
            view.removeItem(self._depth_overlay_item)
        except Exception:
            pass
        self._depth_overlay_item = None

    def has_depth_overlay(self) -> bool:
        """True when :meth:`set_depth_overlay` has painted something."""
        return self._depth_overlay_item is not None

    # ---- cluster centroid markers (v1.3-polish) ---------------------

    def set_cluster_markers(self, clusters) -> None:
        """Paint a dot + z-label at each cluster centroid.

        ``clusters`` is an iterable of :class:`core.depth_map.ClusterHeight`.
        Each cluster gets a ScatterPlotItem (circle marker) and a
        TextItem with its z value in mm. Existing markers are replaced
        — second call wipes the old set before painting the new one
        so the panel never accumulates stale dots.
        """
        self.clear_cluster_markers()
        clusters = list(clusters) if clusters is not None else []
        if not clusters:
            return

        view = self.image_panel.get_view()
        for cluster in clusters:
            y, x = cluster.centroid_yx
            # ScatterPlotItem: one centroid dot. ImageItem's coordinate
            # convention is (x, y) = (column, row), so we swap.
            dot = pg.ScatterPlotItem(
                x=[float(x)], y=[float(y)],
                size=11,
                pen=pg.mkPen("w", width=1),
                brush=pg.mkBrush(255, 220, 40, 220),
            )
            dot.setZValue(20)  # above the depth overlay
            view.addItem(dot)

            # TextItem: z_mean in mm + cluster id. Offset slightly up
            # so the dot isn't obscured.
            label = pg.TextItem(
                text=f"#{cluster.cluster_id}  {cluster.z_mean_m * 1e3:+.2f} mm",
                color=(255, 220, 40),
                anchor=(0.0, 1.0),  # anchor bottom-left of the text box
            )
            label.setPos(float(x) + 6.0, float(y) - 6.0)
            label.setZValue(21)
            view.addItem(label)

            self._cluster_marker_items.extend([dot, label])

    def clear_cluster_markers(self) -> None:
        """Remove any cluster markers currently painted."""
        if not self._cluster_marker_items:
            return
        view = self.image_panel.get_view()
        for item in self._cluster_marker_items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        self._cluster_marker_items = []

    def has_cluster_markers(self) -> bool:
        """True when at least one cluster marker is currently painted."""
        return bool(self._cluster_marker_items)
