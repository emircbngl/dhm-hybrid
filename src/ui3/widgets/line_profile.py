"""LineProfileTool + LineProfileDialog — click-drag line profiles (ui3).

Behaviour parity: ``ui2.line_profile_state.LineProfileEditor`` is the
click/click state machine ui2's DPG dialog drives (first_click ->
second_click -> commit); ``core.line_profile`` is the Qt-free bilinear
sampler. ui3 replaces both the click/click gesture *and* the state machine
with pyqtgraph's native click-drag ``LineSegmentROI`` — a single drag defines
a line directly (no separate first-click/second-click steps needed, since the
ROI widget already owns "drag to define, drag handles to adjust"). The
sampling math is untouched: every profile still goes through
``core.line_profile.sample_line`` bilinearly, so numbers match ui2 exactly
for the same endpoints.

``ImagePanel`` (``ui3/viewport.py``) is frozen — this tool does not modify
it. It attaches externally: an ``LineSegmentROI`` is added directly to the
panel's already-public ``_vb`` (pyqtgraph ``ViewBox``) and reads the
already-public ``_data`` (the last array drawn via ``set_image``) to sample
against. ``ImagePanel`` already declares a ``line_requested`` signal "for
future" — this is that future; the tool does not require it (it manages its
own ROI + sampling) but emits its own ``profile_added`` for hosts that want
to react without importing pyqtgraph types.

Coordinate convention: ``ImagePanel`` sets ``imageAxisOrder="row-major"`` and
``invertY(True)``, so a pyqtgraph view-coordinate point ``(x, y)`` maps
directly to array coordinates ``(row=y, col=x)`` — no flip needed beyond
that already-established convention.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.line_profile import LineProfile, SampledProfile, sample_line, stats_for
from ui3.design import Space, Type

# Same six-colour cycle as ui2.line_profile_state._DEFAULT_PALETTE, so a
# profile added Nth lands on the same visual colour in either UI.
_DEFAULT_COLOURS: Tuple[Tuple[float, float, float], ...] = (
    (0.27, 0.55, 0.95),
    (0.95, 0.45, 0.30),
    (0.40, 0.80, 0.40),
    (0.80, 0.45, 0.85),
    (0.95, 0.85, 0.30),
    (0.40, 0.85, 0.85),
)


def _rgb_to_qcolor_tuple(rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgb)


class LineProfileTool:
    """Click-drag line-profile capture on top of an ``ImagePanel``.

    Not a ``QWidget`` itself — a controller object that owns zero-or-more
    ``pg.LineSegmentROI`` overlays on ``image_panel``'s viewbox and produces
    :class:`SampledProfile` results by sampling ``image_panel``'s current
    array. Multiple profiles can coexist (each its own ROI + colour),
    matching ui2's "compare three lines on one frame" use case.
    """

    def __init__(self, image_panel) -> None:
        self._panel = image_panel
        self._rois: List[pg.LineSegmentROI] = []
        self._labels: List[str] = []

    # ------------------------------------------------------------------
    def add_line(self, y0: float, x0: float, y1: float, x1: float,
                 *, label: str = "") -> pg.LineSegmentROI:
        """Add a new draggable line ROI at the given array-coordinate
        endpoints (row, col each). Returns the ROI so a caller can further
        customise it if needed."""
        colour = _DEFAULT_COLOURS[len(self._rois) % len(_DEFAULT_COLOURS)]
        pen = pg.mkPen(color=_rgb_to_qcolor_tuple(colour), width=2)
        # ROI positions are (x, y) in view coords == (col, row) in array
        # coords, per the row-major / invertY(True) convention ImagePanel
        # establishes.
        roi = pg.LineSegmentROI([[x0, y0], [x1, y1]], pen=pen)
        self._panel._vb.addItem(roi)
        self._rois.append(roi)
        self._labels.append(label or f"line-{len(self._rois)}")
        return roi

    def remove_line(self, index: int) -> None:
        if not 0 <= index < len(self._rois):
            return
        roi = self._rois.pop(index)
        self._labels.pop(index)
        self._panel._vb.removeItem(roi)

    def clear(self) -> None:
        for roi in self._rois:
            self._panel._vb.removeItem(roi)
        self._rois.clear()
        self._labels.clear()

    def count(self) -> int:
        return len(self._rois)

    def endpoints(self, index: int) -> Optional[Tuple[float, float, float, float]]:
        """Current (y0, x0, y1, x1) for ROI ``index`` — reads live handle
        positions, so this reflects any drag the user made after adding."""
        if not 0 <= index < len(self._rois):
            return None
        pts = self._rois[index].listPoints()
        if len(pts) < 2:
            return None
        (x0, y0), (x1, y1) = (pts[0].x(), pts[0].y()), (pts[1].x(), pts[1].y())
        return (y0, x0, y1, x1)

    def sample(self, index: int) -> Optional[SampledProfile]:
        """Sample the current array in ``image_panel`` along ROI
        ``index`` using ``core.line_profile.sample_line`` (bilinear,
        Qt-free). ``None`` if there's no image or no such ROI."""
        data = getattr(self._panel, "_data", None)
        pts = self.endpoints(index)
        if data is None or pts is None:
            return None
        y0, x0, y1, x1 = pts
        colour = _DEFAULT_COLOURS[index % len(_DEFAULT_COLOURS)]
        profile = LineProfile(
            y0=y0, x0=x0, y1=y1, x1=x1,
            label=self._labels[index], colour_rgb=colour,
        )
        return sample_line(np.asarray(data), profile)

    def sample_all(self) -> List[SampledProfile]:
        out = []
        for i in range(len(self._rois)):
            s = self.sample(i)
            if s is not None:
                out.append(s)
        return out


class LineProfileDialog(QDialog):
    """Wrapper dialog: shows a :class:`LineProfileTool`'s current profiles
    (multi-line overlay + min/max/mean) in a small pyqtgraph
    ``PlotWidget``, refreshed on demand.

    Built against ``PanelContext`` so it can be opened from any panel that
    holds an ``ImagePanel`` reference (e.g. via ``ctx.show_in_panel``'s
    target panels on the shell) — the dialog itself only needs a
    :class:`LineProfileTool` instance, ``ctx`` is used solely for the
    palette + status/toast feedback, kept consistent with other ui3
    dialogs.
    """

    profile_added = Signal(object)  # SampledProfile

    def __init__(self, tool: LineProfileTool, ctx=None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tool = tool
        self._ctx = ctx
        self.setWindowTitle("Line profile")
        self.resize(560, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.md)
        root.setSpacing(Space.sm)

        heading = QLabel("Line profile")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "distance (px)")
        self.plot.setLabel("left", "value")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        root.addWidget(self.plot, 1)

        self.stats_list = QListWidget()
        root.addWidget(self.stats_list)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_clear = QPushButton("Clear lines")
        self.btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-sample every line in the tool and redraw the overlay plot +
        stats list."""
        self.plot.clear()
        self.stats_list.clear()
        sampled = self._tool.sample_all()
        for s in sampled:
            colour = _rgb_to_qcolor_tuple(s.profile.colour_rgb)
            pen = pg.mkPen(color=colour, width=2)
            self.plot.plot(s.distance_px, s.values, pen=pen, name=s.profile.label)
            st = stats_for(s)
            item = QListWidgetItem(
                f"{s.profile.label}: min={_fmt(st['min'])} "
                f"max={_fmt(st['max'])} mean={_fmt(st['mean'])} n={st['n']}")
            self.stats_list.addItem(item)
        if self._ctx is not None:
            self._ctx.set_status(f"Line profile: {len(sampled)} line(s).", "info")

    def _on_clear(self) -> None:
        self._tool.clear()
        self.refresh()


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.4g}"


__all__ = ["LineProfileTool", "LineProfileDialog"]
