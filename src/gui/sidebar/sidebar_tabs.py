"""Sidebar tab container — workflow-mode filter sits on top.

The raw QTabWidget hosts six settings tabs (Camera, Recon, Process,
Focus, QPI, Record). v1.4-UI-redesign adds a **workflow mode** combo
above it: the user chooses *what they're doing right now* (Acquire /
Reconstruct / Analyse) and the sidebar hides the tabs that aren't
relevant for that stage. The underlying run-mode (File vs Live)
independently controls Camera/Record availability; the workflow
filter composes on top of that.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .camera_tab import CameraTab
from .focus_tab import FocusTab
from .process_tab import ProcessTab
from .qpi_tab import QPITab
from .recon_tab import ReconTab
from .record_tab import RecordTab
from .report_tab import ReportTab


class WorkflowMode(str, Enum):
    """Top-level task the operator is performing right now."""
    ACQUIRE = "acquire"
    RECONSTRUCT = "reconstruct"
    ANALYSE = "analyse"
    REPORT = "report"


# Which tab attributes belong to each workflow mode. Union semantics —
# a tab visible in multiple modes (there are none today, but reserved
# for future refactors) would be listed in each set.
_WORKFLOW_TABS: dict[WorkflowMode, frozenset[str]] = {
    WorkflowMode.ACQUIRE:     frozenset({"camera_tab", "record_tab"}),
    WorkflowMode.RECONSTRUCT: frozenset({"recon_tab", "process_tab"}),
    WorkflowMode.ANALYSE:     frozenset({"focus_tab", "qpi_tab"}),
    WorkflowMode.REPORT:      frozenset({"report_tab"}),
}

_ALL_WORKFLOW_TABS: frozenset[str] = frozenset().union(*_WORKFLOW_TABS.values())


_MODE_LABELS: dict[WorkflowMode, str] = {
    WorkflowMode.ACQUIRE:     "Acquire",
    WorkflowMode.RECONSTRUCT: "Reconstruct",
    WorkflowMode.ANALYSE:     "Analyse",
    WorkflowMode.REPORT:      "Report",
}


class SidebarTabs(QWidget):
    """Central container wrapping all configuration settings into a
    tabbed layout with a workflow-mode filter on top."""

    # Emitted after the workflow mode has been applied — host can
    # persist the choice / react in the status bar.
    workflow_mode_changed = Signal(str)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Workflow mode combo (v1.4) ──
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(6, 4, 6, 4)
        mode_row.setSpacing(6)
        mode_row.addWidget(QLabel("Workflow:"))
        self.workflow_combo = QComboBox()
        for mode in WorkflowMode:
            self.workflow_combo.addItem(_MODE_LABELS[mode], mode.value)
        self.workflow_combo.setAccessibleName("Workflow mode selector")
        self.workflow_combo.setToolTip(
            "Filter the sidebar to the tabs relevant for the current task."
        )
        mode_row.addWidget(self.workflow_combo, 1)
        layout.addLayout(mode_row)

        # ── Tab widget ──
        self.tabs = QTabWidget()
        self.tabs.setObjectName("sidebar_tabs")

        self.camera_tab = CameraTab()
        self.recon_tab = ReconTab()
        self.process_tab = ProcessTab()
        self.focus_tab = FocusTab()
        self.qpi_tab = QPITab()
        self.record_tab = RecordTab()
        self.report_tab = ReportTab()

        self.tabs.addTab(self.camera_tab, "Camera")
        self.tabs.addTab(self.recon_tab, "Recon")
        self.tabs.addTab(self.process_tab, "Process")
        self.tabs.addTab(self.focus_tab, "Focus")
        self.tabs.addTab(self.qpi_tab, "QPI")
        self.tabs.addTab(self.record_tab, "Record")
        self.tabs.addTab(self.report_tab, "Report")

        layout.addWidget(self.tabs)

        # Default: Reconstruct. Host may overwrite from persisted state.
        self._current_mode: WorkflowMode = WorkflowMode.RECONSTRUCT
        idx = self.workflow_combo.findData(self._current_mode.value)
        if idx >= 0:
            self.workflow_combo.setCurrentIndex(idx)
        self._apply_mode(self._current_mode)
        self.workflow_combo.currentIndexChanged.connect(
            self._on_combo_changed,
        )

    # ---- public API ----------------------------------------------------

    def workflow_mode(self) -> WorkflowMode:
        """Return the mode currently applied to the tab filter."""
        return self._current_mode

    def set_workflow_mode(self, mode) -> None:
        """Switch the sidebar to the given workflow mode.

        Accepts a :class:`WorkflowMode` or the underlying string value
        (``"acquire"`` / ``"reconstruct"`` / ``"analyse"``). Unknown
        modes raise :class:`ValueError`.
        """
        if isinstance(mode, str):
            try:
                mode_enum = WorkflowMode(mode)
            except ValueError as exc:
                raise ValueError(
                    f"unknown workflow mode {mode!r}; valid: "
                    f"{[m.value for m in WorkflowMode]}"
                ) from exc
        elif isinstance(mode, WorkflowMode):
            mode_enum = mode
        else:
            raise ValueError(f"workflow mode must be str or WorkflowMode (got {type(mode).__name__})")

        if mode_enum is self._current_mode:
            return

        self._current_mode = mode_enum
        # Keep the combo in sync (muted — avoid signal bounce).
        idx = self.workflow_combo.findData(mode_enum.value)
        if idx >= 0:
            blocked = self.workflow_combo.blockSignals(True)
            try:
                self.workflow_combo.setCurrentIndex(idx)
            finally:
                self.workflow_combo.blockSignals(blocked)

        self._apply_mode(mode_enum)
        self.workflow_mode_changed.emit(mode_enum.value)

    # ---- internals -----------------------------------------------------

    def _apply_mode(self, mode: WorkflowMode) -> None:
        """Show tabs in ``_WORKFLOW_TABS[mode]``, hide the rest.

        Only the tabs governed by the workflow filter are touched —
        any tab index that the run-mode (File/Live) already removed
        from the QTabWidget is ignored.
        """
        active = _WORKFLOW_TABS[mode]
        for attr in _ALL_WORKFLOW_TABS:
            tab = getattr(self, attr, None)
            if tab is None:
                continue
            idx = self.tabs.indexOf(tab)
            if idx < 0:
                # Not mounted — e.g. Camera in File run-mode.
                continue
            visible = attr in active
            # Qt 6.2+: setTabVisible hides without detaching the widget.
            if hasattr(self.tabs, "setTabVisible"):
                self.tabs.setTabVisible(idx, visible)
            elif not visible:
                # Fallback for older Qt — just disable the tab.
                self.tabs.setTabEnabled(idx, False)

        # Land on the first visible tab inside the active set so the
        # user isn't looking at a hidden pane after a mode switch.
        for attr in active:
            tab = getattr(self, attr, None)
            if tab is None:
                continue
            idx = self.tabs.indexOf(tab)
            if idx >= 0:
                if hasattr(self.tabs, "isTabVisible") and not self.tabs.isTabVisible(idx):
                    continue
                self.tabs.setCurrentIndex(idx)
                break

    def _on_combo_changed(self, _index: int) -> None:
        data = self.workflow_combo.currentData()
        if not data:
            return
        try:
            mode_enum = WorkflowMode(data)
        except ValueError:
            return
        if mode_enum is self._current_mode:
            return
        self._current_mode = mode_enum
        self._apply_mode(mode_enum)
        self.workflow_mode_changed.emit(mode_enum.value)


__all__ = ["SidebarTabs", "WorkflowMode"]
