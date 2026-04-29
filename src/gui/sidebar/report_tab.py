"""Report-mode sidebar tab (v1.4 UI Redesign).

Hosts one-click entry points for every export the app ships. The
tab is populated at host time — we can't import ``main_window``
from here without a circular dependency, so the host wires each
button to its existing handler via :meth:`bind_report_actions`.

Layout mirrors the Tools menu groupings: HTML / PDF report, QPI
CSV, Depth map, Tomography bundle, QPI batch with focus picker.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QFrame,
)


class ReportTab(QWidget):
    """Report / export action launcher.

    The tab stays inert until :meth:`bind_report_actions` is called
    by the host — safer than reaching into the main window at
    construction time (tests can instantiate the tab without a full
    MainWindow).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        intro = QLabel(
            "<b>Report mode</b><br>"
            "One-click exports for the latest reconstruction / QPI / depth run."
        )
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        intro.setStyleSheet("color: palette(mid);")
        layout.addWidget(intro)

        # Each group = one thematic export family.
        self.btn_report = self._add_button(
            layout,
            "Generate report (HTML / PDF)…",
            "Write a full HTML/PDF summary of the latest reconstruction, "
            "autofocus, and QPI.",
        )
        self.btn_qpi_csv = self._add_button(
            layout,
            "Export QPI CSV…",
            "Flatten the last QPI result into a single-row CSV.",
        )
        self.btn_depth_map = self._add_button(
            layout,
            "Compute + save depth map (NPZ + CSV)…",
            "Run the tomographic depth scan and save NPZ plus flat CSV.",
        )
        self.btn_qpi_batch = self._add_button(
            layout,
            "Run QPI batch for focus candidates…",
            "Scan focus landscape, run QPI per plane, review in-app.",
        )
        self.btn_bundle = self._add_button(
            layout,
            "Export tomography bundle…",
            "Write depth NPZ + depth CSV + cluster CSV + QPI batch CSV "
            "in one directory.",
        )

        layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ---- helpers ------------------------------------------------------

    def _add_button(
        self, layout, label: str, hint: str,
    ) -> QPushButton:
        group = QGroupBox()
        group.setFlat(True)
        gl = QVBoxLayout(group)
        gl.setContentsMargins(2, 2, 2, 6)
        gl.setSpacing(4)

        btn = QPushButton(label)
        btn.setMinimumHeight(32)
        btn.setAccessibleName(label)
        btn.setStatusTip(hint)
        btn.setEnabled(False)  # host enables once bound
        gl.addWidget(btn)

        desc = QLabel(hint)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: palette(mid); font-size: 11px;")
        gl.addWidget(desc)

        layout.addWidget(group)
        return btn

    # ---- public: bind to host actions --------------------------------

    def bind_report_actions(
        self,
        *,
        on_report: Callable[[], None],
        on_qpi_csv: Callable[[], None],
        on_depth_map: Callable[[], None],
        on_qpi_batch: Callable[[], None],
        on_bundle: Callable[[], None],
    ) -> None:
        """Called by the host (main_window) after construction."""
        for btn, handler in (
            (self.btn_report, on_report),
            (self.btn_qpi_csv, on_qpi_csv),
            (self.btn_depth_map, on_depth_map),
            (self.btn_qpi_batch, on_qpi_batch),
            (self.btn_bundle, on_bundle),
        ):
            btn.clicked.connect(handler)
            btn.setEnabled(True)


__all__ = ["ReportTab"]
