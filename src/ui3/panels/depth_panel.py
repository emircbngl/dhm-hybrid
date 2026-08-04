"""DepthPanel — depth-map + 3D surface entry panel (ui3).

Reads back to ``ui2.workers.run_depth_map`` / ``core.depth_map`` via
``ctx.bridge.depth_map`` — the same ``ScienceDriver`` used by the ui3 spine's
inline "Depth map" menu action (``MainWindow._on_depth_map`` /
``_on_depth_done`` in ``ui3/main_window.py``). This panel is the rich,
dockable counterpart: it exposes the full z-range / step / window / metric
controls ui2's sidebar has (``src/ui2/workers.py::run_depth_map``,
``core/depth_map.py::compute_depth_map``) instead of hard-coding
``params.af_z_min_mm`` / ``af_z_max_mm``.

Physics/behaviour parity notes (ui2 -> ui3, unchanged):

* ``z_map`` comes back in **metres**; the display convention across ui3
  (see ``MainWindow._on_depth_done``) is millimetres, so we multiply by
  ``1e3`` before handing the array to ``ctx.show_in_panel``.
* The autofocus metric used by the depth scan is resolved from
  ``ReconParams.autofocus_metric`` (see ``ui2.workers._metric_from_params``)
  — there is no separate ``metric=`` kwarg on ``WorkerBridge.depth_map``, so
  this panel's metric combo writes straight into that shared field (the same
  field the Autofocus panel/controls use) rather than inventing a
  depth-only parameter the compute layer would silently ignore.
* z-min/z-max/n-steps here mirror ``ReconParams.af_z_min_mm`` /
  ``af_z_max_mm`` / ``af_n_steps`` — same fields the inline Reconstruct dock
  and the (future) Focus panel edit, kept in sync via ``ctx.get_params()``
  being the live, shared instance.
* ``window_size`` is depth-map-specific (``compute_depth_map``'s per-pixel
  focus-metric window) and has no ``ReconParams`` home, so it lives as
  local panel state only, passed straight through to
  ``ctx.bridge.depth_map(..., window_size=...)``.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from core.autofocus import FocusMetric
from ui3.context import PanelContext
from ui3.design import Space

_LOG = logging.getLogger("ui3.panels.depth")


class DepthPanel(QWidget):
    """Depth-map controls + 3D-surface launchers.

    Signals
    -------
    surface_requested(str)
        Emitted with ``"depth"`` or ``"phase"`` when the user asks to open
        the 3D surface viewer for that field. The dialog itself
        (``ui3/dialogs/surface_viewer.py::SurfaceViewer``) is wired up by
        the integration step (``MainWindow.mount_panel`` /
        ``_open_surface``); until then this panel falls back to
        ``ctx.set_status`` so the button is never a silent no-op.
    """

    surface_requested = Signal(str)

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._last_z_map: Optional[np.ndarray] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.md)
        root.setSpacing(Space.md)

        title = QLabel("Depth map")
        title.setProperty("role", "heading")
        root.addWidget(title)

        params = self._ctx.get_params()

        form = QFormLayout()
        form.setSpacing(Space.sm)

        self._sp_z_min = QDoubleSpinBox()
        self._sp_z_min.setRange(-500.0, 500.0)
        self._sp_z_min.setDecimals(3)
        self._sp_z_min.setSingleStep(0.1)
        self._sp_z_min.setSuffix(" mm")
        self._sp_z_min.setValue(float(params.af_z_min_mm))

        self._sp_z_max = QDoubleSpinBox()
        self._sp_z_max.setRange(-500.0, 500.0)
        self._sp_z_max.setDecimals(3)
        self._sp_z_max.setSingleStep(0.1)
        self._sp_z_max.setSuffix(" mm")
        self._sp_z_max.setValue(float(params.af_z_max_mm))

        self._sp_n_steps = QSpinBox()
        self._sp_n_steps.setRange(2, 2000)
        self._sp_n_steps.setValue(int(params.af_n_steps))

        self._sp_window = QSpinBox()
        self._sp_window.setRange(1, 99)
        self._sp_window.setSingleStep(2)
        self._sp_window.setValue(5)
        self._sp_window.setSuffix(" px")

        self._cb_metric = QComboBox()
        self._cb_metric.addItems([m.name for m in FocusMetric])
        current_metric = str(getattr(params, "autofocus_metric",
                                     "LAPLACIAN_VARIANCE") or "LAPLACIAN_VARIANCE")
        idx = self._cb_metric.findText(current_metric)
        self._cb_metric.setCurrentIndex(idx if idx >= 0 else 0)

        form.addRow("z min", self._sp_z_min)
        form.addRow("z max", self._sp_z_max)
        form.addRow("n steps", self._sp_n_steps)
        form.addRow("Window size", self._sp_window)
        form.addRow("Focus metric", self._cb_metric)
        root.addLayout(form)

        # Push widget edits straight into the live ReconParams instance so
        # every other panel/dock reading the same object sees them too.
        self._sp_z_min.valueChanged.connect(self._on_controls_changed)
        self._sp_z_max.valueChanged.connect(self._on_controls_changed)
        self._sp_n_steps.valueChanged.connect(self._on_controls_changed)
        self._cb_metric.currentTextChanged.connect(self._on_controls_changed)

        self._compute_btn = QPushButton("Compute depth map")
        self._compute_btn.setProperty("role", "primary")
        self._compute_btn.clicked.connect(self._on_compute)
        root.addWidget(self._compute_btn)

        self._clear_btn = QPushButton("Clear overlay")
        self._clear_btn.clicked.connect(self._on_clear)
        root.addWidget(self._clear_btn)

        self._readout = QLabel("z: — / — / — mm (min/max/mean)")
        self._readout.setProperty("role", "readout")
        self._readout.setWordWrap(True)
        root.addWidget(self._readout)

        surf_row = QHBoxLayout()
        self._surface_depth_btn = QPushButton("Open 3D surface — depth")
        self._surface_depth_btn.clicked.connect(
            lambda: self._request_surface("depth"))
        self._surface_phase_btn = QPushButton("Open 3D surface — phase")
        self._surface_phase_btn.clicked.connect(
            lambda: self._request_surface("phase"))
        surf_row.addWidget(self._surface_depth_btn)
        surf_row.addWidget(self._surface_phase_btn)
        root.addLayout(surf_row)

        root.addStretch(1)

        self._ctx.bridge.depth_done.connect(self._on_depth_done)
        self._ctx.bridge.depth_error.connect(self._on_depth_error)

    # ------------------------------------------------------------------
    # Controls -> live params
    # ------------------------------------------------------------------
    def _on_controls_changed(self, *_args) -> None:
        p = self._ctx.get_params()
        p.af_z_min_mm = float(self._sp_z_min.value())
        p.af_z_max_mm = float(self._sp_z_max.value())
        p.af_n_steps = int(self._sp_n_steps.value())
        p.autofocus_metric = str(self._cb_metric.currentText())
        self._ctx.params_changed()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_compute(self) -> None:
        path = self._ctx.hologram_path()
        if not path:
            self._ctx.set_status("Load a hologram before computing a depth map.",
                                  "warn")
            return
        self._on_controls_changed()
        p = self._ctx.get_params()
        self._ctx.bridge.depth_map(
            path, p,
            z_min_mm=p.af_z_min_mm, z_max_mm=p.af_z_max_mm,
            n_steps=p.af_n_steps, window_size=int(self._sp_window.value()),
        )

    def _on_depth_done(self, result) -> None:
        wrapped = getattr(result, "result", result)
        z_map = getattr(wrapped, "z_map", None)
        if z_map is None:
            self._ctx.set_status("Depth map returned no data.", "warn")
            return
        z_map = np.asarray(z_map)
        self._last_z_map = z_map
        # metres -> mm for display, matching the ui3 spine convention
        # (MainWindow._on_depth_done multiplies by 1e3 too).
        z_mm = z_map * 1e3
        self._ctx.show_in_panel("spectrum", z_mm, contrast="minmax")
        self._readout.setText(
            f"z: {float(z_mm.min()):.3f} / {float(z_mm.max()):.3f} / "
            f"{float(z_mm.mean()):.3f} mm (min/max/mean)"
        )
        # A reference-division fallback (B-110) makes absolute heights
        # unreliable — surface it instead of a clean "computed".
        warn = getattr(result, "warning", None)
        if warn:
            self._ctx.set_status(f"Depth map (UNREFERENCED) — {warn}", "warn")
            self._ctx.toast(warn, "warn")
        else:
            self._ctx.set_status("Depth map computed", "ok")

    def _on_depth_error(self, message: str) -> None:
        self._ctx.set_status(f"Depth map failed: {message}", "error")

    def _on_clear(self) -> None:
        self._last_z_map = None
        self._ctx.show_in_panel("spectrum", np.zeros((2, 2), dtype=np.float32))
        self._readout.setText("z: — / — / — mm (min/max/mean)")
        self._ctx.set_status("Depth overlay cleared")

    def _request_surface(self, kind: str) -> None:
        """Ask the shell to open the 3D surface viewer for ``kind``.

        ``SurfaceViewer`` (``ui3/dialogs/surface_viewer.py``) is being built
        by another agent in parallel; until the shell wires this signal to
        it, fall back to a status message so the button is never silently
        inert.
        """
        self.surface_requested.emit(kind)
        self._ctx.set_status(
            f"3D surface ({kind}) requested — opens once the surface "
            "viewer is mounted.", "info")

    # ------------------------------------------------------------------
    # Introspection helpers (used by tests)
    # ------------------------------------------------------------------
    def last_z_map(self) -> Optional[np.ndarray]:
        return self._last_z_map
