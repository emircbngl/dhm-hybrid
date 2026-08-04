"""SurfaceViewer — 3D surface render of depth z(x, y) or unwrapped phase.

Non-modal ``QDialog`` built against the frozen ``PanelContext``. Mirrors
``gui.widgets.depth_surface_viewer.DepthSurfaceViewer`` (the v1 Qt +
``pyqtgraph.opengl`` implementation — same widget family, same
``GLSurfacePlotItem`` construction, same colour-mapping approach) rather
than ``ui2.surface`` (which spawns a separate matplotlib subprocess — not
applicable here since ui3 already runs a real Qt event loop).

Two entry points:

* ``show_depth(zmap_mm)`` — a 2-D depth map already in millimetres (the
  ui3 display convention; see ``DepthPanel._on_depth_done`` /
  ``MainWindow._on_depth_done``, both multiply the raw metres-valued
  ``z_map`` by ``1e3`` before it reaches a panel).
* ``show_phase(unwrapped)`` — a 2-D unwrapped-phase array in radians.
  Negative values are NOT clipped (same physical-meaning rule the v1
  viewer documents).
* ``open_for(kind)`` — pulls the field straight from
  ``ctx.get_field("depth")`` (metres — converted to mm here) or
  ``ctx.get_field("phase_unwrapped")`` (radians) and renders it. Accepts
  either ``open_for(kind)`` or ``open_for(host, kind)`` — ``MainWindow``
  (frozen; see ``ui3/main_window.py::_open_surface``) calls
  ``self._panels["surface"].open_for(self, kind)``, passing itself as a
  first positional argument that this dialog does not otherwise need
  (it already has ``ctx`` from ``__init__``).

Large arrays are downsampled (stride) above 256 px per side before
building the ``GLSurfacePlotItem`` — matplotlib/OpenGL surface rendering
gets sluggish well before typical hologram resolutions, and the v1
viewer's docstring makes the same call for its own downsampling.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from ui3.context import PanelContext

_COLORMAPS = ("viridis", "plasma", "inferno", "magma", "turbo", "cividis")

# Above this many pixels per side, stride the array down before building
# the GL surface — keeps interaction responsive on full-res holograms.
_MAX_SIDE = 256


class SurfaceViewer(QDialog):
    """3D surface viewer for depth maps and unwrapped phase."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Surface Viewer")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(820, 720)

        layout = QVBoxLayout(self)

        self.gl = gl.GLViewWidget()
        # Match the 2D viewports' theme-aware background instead of a hardcoded
        # dark grey (2026-07-10 audit #8); fall back to dark if no palette.
        _view_bg = getattr(getattr(ctx, "palette", None), "view_bg", None)
        self.gl.setBackgroundColor(_view_bg or "#141414")
        self.gl.opts["distance"] = 200
        layout.addWidget(self.gl, stretch=1)

        grid = gl.GLGridItem()
        grid.setSize(200, 200)
        grid.setSpacing(20, 20)
        self.gl.addItem(grid)
        self._grid = grid

        self._surface: Optional[gl.GLSurfacePlotItem] = None

        # -- Controls ---------------------------------------------------
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(_COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_change)
        ctrl.addWidget(self.cmap_combo)

        ctrl.addWidget(QLabel("Z exaggeration:"))
        self.zscale_spin = QDoubleSpinBox()
        self.zscale_spin.setRange(0.01, 1000.0)
        self.zscale_spin.setDecimals(2)
        self.zscale_spin.setValue(1.0)
        self.zscale_spin.setSingleStep(0.1)
        self.zscale_spin.valueChanged.connect(self._on_zscale_change)
        ctrl.addWidget(self.zscale_spin)

        self.wireframe_cb = QCheckBox("Wireframe")
        self.wireframe_cb.toggled.connect(self._on_wireframe_change)
        ctrl.addWidget(self.wireframe_cb)

        ctrl.addStretch()

        save_btn = QPushButton("Save PNG…")
        save_btn.clicked.connect(self._on_save_png)
        ctrl.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        ctrl.addWidget(close_btn)

        layout.addLayout(ctrl)

        self._last_z: Optional[np.ndarray] = None
        self._last_x: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None

    # ---- public API ---------------------------------------------------

    def show_depth(self, zmap_mm: np.ndarray, *,
                   pixel_size_um: Optional[float] = None,
                   title: Optional[str] = None) -> None:
        """Render a 2-D depth map already expressed in millimetres."""
        z = np.asarray(zmap_mm, dtype=np.float32)
        self.setWindowTitle(title or "Surface — Depth z(x, y)")
        self._render(z, pixel_size_um, z_label="z (mm)")

    def show_phase(self, unwrapped: np.ndarray, *,
                   pixel_size_um: Optional[float] = None,
                   title: Optional[str] = None) -> None:
        """Render a 2-D unwrapped-phase array (radians). Signs preserved."""
        z = np.asarray(unwrapped, dtype=np.float32)
        self.setWindowTitle(title or "Surface — Phase φ(x, y)")
        self._render(z, pixel_size_um, z_label="φ (rad)")

    def open_for(self, host=None, kind: Optional[str] = None) -> None:
        """Pull ``kind`` ("depth" | "phase") straight from ``ctx.get_field``
        and show it. Accepts ``open_for(kind)`` or ``open_for(host, kind)``
        — ``MainWindow._open_surface`` (frozen) calls the latter, passing
        itself as ``host`` even though this dialog only needs ``ctx``."""
        if kind is None:
            kind = host
            host = None
        kind = str(kind or "depth").lower()
        pixel_size_um = None
        try:
            params = self.ctx.get_params()
            pixel_size_um = float(params.effective_pixel_um())
        except Exception:
            pixel_size_um = None

        if kind == "phase":
            field = self.ctx.get_field("phase_unwrapped")
            if field is None:
                self.ctx.set_status(
                    "No unwrapped phase available yet — reconstruct first.",
                    "warn")
                return
            self.show_phase(field, pixel_size_um=pixel_size_um)
        else:
            field = self.ctx.get_field("depth")
            if field is None:
                self.ctx.set_status(
                    "No depth map available yet — compute one first.", "warn")
                return
            # get_field("depth") returns metres; this viewer's display
            # convention (and show_depth's contract) is millimetres.
            zmap_mm = np.asarray(field, dtype=np.float32) * 1e3
            self.show_depth(zmap_mm, pixel_size_um=pixel_size_um)

        from ui3.dialogs._present import present_centered
        present_centered(self)

    # ---- internals ----------------------------------------------------

    def _render(self, z: np.ndarray, pixel_size_um: Optional[float],
                z_label: str) -> None:
        if z.ndim != 2 or z.size == 0:
            return
        z = self._downsample(z)
        h, w = z.shape
        if pixel_size_um and pixel_size_um > 0:
            xs = (np.arange(w, dtype=np.float32) - w / 2) * float(pixel_size_um)
            ys = (np.arange(h, dtype=np.float32) - h / 2) * float(pixel_size_um)
        else:
            xs = np.arange(w, dtype=np.float32) - w / 2
            ys = np.arange(h, dtype=np.float32) - h / 2

        self._last_z = z
        self._last_x = xs
        self._last_y = ys

        try:
            self._grid.setSize(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]))
            self._grid.setSpacing(
                float(xs[-1] - xs[0]) / 10.0,
                float(ys[-1] - ys[0]) / 10.0,
            )
        except Exception:
            pass

        self._rebuild_surface()

    @staticmethod
    def _downsample(z: np.ndarray) -> np.ndarray:
        h, w = z.shape
        longest = max(h, w)
        if longest <= _MAX_SIDE:
            return z
        # Ceiling division so the strided result actually respects
        # _MAX_SIDE (floor division under-strides whenever longest isn't
        # an exact multiple of _MAX_SIDE, e.g. 300 // 256 == 1 leaves 300).
        stride = -(-longest // _MAX_SIDE)
        return z[::stride, ::stride]

    def _rebuild_surface(self) -> None:
        if self._last_z is None:
            return
        z = self._last_z * float(self.zscale_spin.value())
        colors = self._compute_colors(self._last_z)
        if self._surface is not None:
            try:
                self.gl.removeItem(self._surface)
            except Exception:
                pass
        self._surface = gl.GLSurfacePlotItem(
            x=self._last_x, y=self._last_y, z=z.T,
            colors=colors.transpose(1, 0, 2).reshape(-1, 4),
            shader="shaded",
            smooth=True,
            drawEdges=self.wireframe_cb.isChecked(),
        )
        self.gl.addItem(self._surface)
        self._auto_camera()

    def _compute_colors(self, z: np.ndarray) -> np.ndarray:
        try:
            cm = pg.colormap.get(self.cmap_combo.currentText())
        except Exception:
            cm = pg.colormap.get("viridis")
        finite = np.isfinite(z)
        if not finite.any():
            rgba = np.zeros((*z.shape, 4), dtype=np.float32)
            rgba[..., 3] = 1.0
            return rgba
        lo = float(np.nanmin(z))
        hi = float(np.nanmax(z))
        if hi - lo < 1e-15:
            norm = np.zeros_like(z, dtype=np.float32)
        else:
            norm = (z - lo) / (hi - lo)
            norm = np.clip(norm, 0.0, 1.0)
        lut = cm.getLookupTable(nPts=256, alpha=True)
        idx = np.clip((norm * 255).astype(np.int32), 0, 255)
        rgba = lut[idx].astype(np.float32) / 255.0
        rgba[~finite, 3] = 0.0
        return rgba

    def _auto_camera(self) -> None:
        if self._last_x is None or self._last_y is None:
            return
        span = max(float(self._last_x[-1] - self._last_x[0]),
                   float(self._last_y[-1] - self._last_y[0]))
        self.gl.opts["distance"] = max(50.0, span * 1.5)
        self.gl.update()

    # ---- callbacks ------------------------------------------------------

    def _on_cmap_change(self, _name: str) -> None:
        self._rebuild_surface()

    def _on_zscale_change(self, _val: float) -> None:
        self._rebuild_surface()

    def _on_wireframe_change(self, _checked: bool) -> None:
        self._rebuild_surface()

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save surface PNG", "surface.png", "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path = path + ".png"
        self.save_png(path)

    def save_png(self, path: str) -> None:
        """Grab the GL framebuffer and write it to ``path``. Split out of
        the file-dialog callback so tests can call it directly without a
        native save dialog.

        ``grabFramebuffer`` can return a *null* image (no exception —
        e.g. no real GL context, as under the offscreen Qt platform used
        by the test suite) rather than raising, so a bare ``try/except``
        around it doesn't catch that case. Check ``isNull()`` explicitly
        and fall back to a plain widget grab either way.
        """
        img = None
        try:
            img = self.gl.grabFramebuffer()
        except Exception:
            img = None
        if img is None or img.isNull():
            img = self.gl.grab()
        img.save(path)


__all__ = ["SurfaceViewer"]
