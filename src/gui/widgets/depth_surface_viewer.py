"""Standalone 3D surface viewer for depth maps + phase / unwrapped phase.

A separate :class:`QDialog` window that hosts a ``pyqtgraph.opengl``
surface plot. Lab uses this to inspect cell-height topography on
unwrapped phase (rad → height conversion) and on depth-map z(x, y) at
the same time. Modelled on the Julia ``Axis3 + surface!(:Spectral)``
flow in ``o2/myjulia/process_files.jl`` — same vocabulary, different
toolkit.

Usage::

    viewer = DepthSurfaceViewer(parent=main_window)
    viewer.show_depth(depth_result, pixel_size_um=eff_um)
    # or:
    viewer.show_phase(unwrapped_phase, pixel_size_um=eff_um)

Both forms accept a ``title`` and rebuild the surface in-place — no
need to spawn a new dialog per recon. The viewer also exposes a Save
PNG button so the user can drop the rendered scene into a report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

import pyqtgraph as pg
import pyqtgraph.opengl as gl


_COLORMAPS = ("viridis", "plasma", "inferno", "magma", "turbo", "cividis")


class DepthSurfaceViewer(QDialog):
    """3D surface render of a 2-D scalar field (depth z or phase rad).

    The window is non-modal so the user can compare it side-by-side
    with the 2-D panels. Closing it does NOT delete it — the host can
    cache the instance and call ``show_depth`` / ``show_phase`` on
    each new recon to update the same window.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Surface Viewer")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(820, 720)

        layout = QVBoxLayout(self)

        # GL surface widget — anti-aliasing on for smoother edges.
        self.gl = gl.GLViewWidget()
        self.gl.setBackgroundColor((20, 20, 20))
        self.gl.opts["distance"] = 200
        layout.addWidget(self.gl, stretch=1)

        # Add a faint grid at z=0 for visual depth reference.
        grid = gl.GLGridItem()
        grid.setSize(200, 200)
        grid.setSpacing(20, 20)
        self.gl.addItem(grid)
        self._grid = grid

        # Axis lines (tiny — just so the viewer isn't empty before
        # data lands).
        self._surface: Optional[gl.GLSurfacePlotItem] = None

        # ─── Controls ───
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(_COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_change)
        ctrl.addWidget(self.cmap_combo)

        ctrl.addWidget(QLabel("Z exaggeration:"))
        self.zscale_spin = QDoubleSpinBox()
        self.zscale_spin.setRange(0.01, 100.0)
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

        # State so the user's cmap / zscale choice survives a re-show
        # of fresh data.
        self._last_z: Optional[np.ndarray] = None
        self._last_x: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None
        self._last_unit: str = "rad"  # for the title hint

    # ---- public API ---------------------------------------------------

    def show_depth(self, depth_result, *,
                   pixel_size_um: float | None = None,
                   title: str | None = None) -> None:
        """Render a :class:`core.depth_map.DepthMapResult`.

        Z is the per-pixel best-focus depth in mm; lateral axes are
        µm when ``pixel_size_um`` is supplied, otherwise pixels.
        """
        z_mm = np.asarray(depth_result.z_map, dtype=np.float32) * 1e3
        self._last_unit = "mm"
        self.setWindowTitle(title or "Surface — Depth z(x, y)")
        self._render(z_mm, pixel_size_um, z_label="z (mm)")

    def show_phase(self, phase_rad: np.ndarray, *,
                   pixel_size_um: float | None = None,
                   title: str | None = None) -> None:
        """Render a 2-D phase / unwrapped-phase array (radians).

        Negative values are NOT clipped — the lab made it explicit
        on 2026-04-30 that both signs carry physical meaning. The
        Julia post-processing scripts that zero negatives are not
        the model here.
        """
        z = np.asarray(phase_rad, dtype=np.float32)
        self._last_unit = "rad"
        self.setWindowTitle(title or "Surface — Phase φ(x, y)")
        self._render(z, pixel_size_um, z_label="φ (rad)")

    # ---- internals ----------------------------------------------------

    def _render(self, z: np.ndarray, pixel_size_um: float | None,
                z_label: str) -> None:
        if z.ndim != 2 or z.size == 0:
            return
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

        # Resize the grid so it matches the lateral extent — gives the
        # user a sense of physical scale even before the colormap legend
        # comes in.
        try:
            self._grid.setSize(float(xs[-1] - xs[0]),
                               float(ys[-1] - ys[0]))
            self._grid.setSpacing(
                float(xs[-1] - xs[0]) / 10.0,
                float(ys[-1] - ys[0]) / 10.0,
            )
        except Exception:
            pass

        self._rebuild_surface()

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
        # GLSurfacePlotItem expects z transposed relative to (x, y) —
        # x along axis 0, y along axis 1. Our arrays are row-major
        # (y, x), so swap.
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
        """Map ``z`` to RGBA via the selected pyqtgraph colormap."""
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
        # Hide non-finite pixels by zeroing alpha.
        rgba[~finite, 3] = 0.0
        return rgba

    def _auto_camera(self) -> None:
        if self._last_x is None or self._last_y is None:
            return
        span = max(float(self._last_x[-1] - self._last_x[0]),
                   float(self._last_y[-1] - self._last_y[0]))
        self.gl.opts["distance"] = max(50.0, span * 1.5)
        self.gl.update()

    # ---- callbacks ---------------------------------------------------

    def _on_cmap_change(self, _name: str) -> None:
        self._rebuild_surface()

    def _on_zscale_change(self, _val: float) -> None:
        self._rebuild_surface()

    def _on_wireframe_change(self, _checked: bool) -> None:
        self._rebuild_surface()

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save surface PNG", "surface.png",
            "PNG image (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path = path + ".png"
        try:
            img = self.gl.grabFramebuffer()
            img.save(path)
        except Exception:
            # grabFramebuffer can fail on some Qt/GL combos; fall back
            # to a Qt-native pixmap grab of the widget.
            self.gl.grab().save(path)


__all__ = ["DepthSurfaceViewer"]
