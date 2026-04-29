"""Workflow-mode filter in SidebarTabs (v1.4 UI Redesign)."""
from __future__ import annotations

import os

import pytest


def _headless_qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return None
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception:
            return None
    return app


# ---- Enum / default -----------------------------------------------------

def test_workflow_mode_enum_values():
    from gui.sidebar.sidebar_tabs import WorkflowMode
    assert {m.value for m in WorkflowMode} == {
        "acquire", "reconstruct", "analyse", "report",
    }


def test_sidebar_default_mode_is_reconstruct():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.sidebar_tabs import SidebarTabs, WorkflowMode

    sidebar = SidebarTabs()
    try:
        assert sidebar.workflow_mode() is WorkflowMode.RECONSTRUCT
    finally:
        sidebar.deleteLater()


# ---- Filter behaviour ---------------------------------------------------

def _visible_tab_names(sidebar) -> set[str]:
    """Return the attribute names of sidebar tabs whose QTabWidget
    entry is currently visible."""
    out = set()
    names = ("camera_tab", "recon_tab", "process_tab",
             "focus_tab", "qpi_tab", "record_tab")
    for name in names:
        tab = getattr(sidebar, name)
        idx = sidebar.tabs.indexOf(tab)
        if idx < 0:
            continue
        if hasattr(sidebar.tabs, "isTabVisible"):
            if sidebar.tabs.isTabVisible(idx):
                out.add(name)
        else:
            # Fallback — treat enabled == visible for pre-6.2 Qt.
            if sidebar.tabs.isTabEnabled(idx):
                out.add(name)
    return out


def test_reconstruct_mode_shows_recon_and_process():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.sidebar_tabs import SidebarTabs

    sidebar = SidebarTabs()
    try:
        sidebar.set_workflow_mode("reconstruct")
        visible = _visible_tab_names(sidebar)
        # Workflow tabs: recon + process visible; camera/record/focus/qpi hidden.
        assert "recon_tab" in visible
        assert "process_tab" in visible
        assert "focus_tab" not in visible
        assert "qpi_tab" not in visible
        assert "camera_tab" not in visible
        assert "record_tab" not in visible
    finally:
        sidebar.deleteLater()


def test_analyse_mode_shows_focus_and_qpi():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.sidebar_tabs import SidebarTabs

    sidebar = SidebarTabs()
    try:
        sidebar.set_workflow_mode("analyse")
        visible = _visible_tab_names(sidebar)
        assert "focus_tab" in visible
        assert "qpi_tab" in visible
        assert "recon_tab" not in visible
        assert "process_tab" not in visible
    finally:
        sidebar.deleteLater()


def test_acquire_mode_shows_camera_and_record():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.sidebar_tabs import SidebarTabs

    sidebar = SidebarTabs()
    try:
        sidebar.set_workflow_mode("acquire")
        visible = _visible_tab_names(sidebar)
        assert "camera_tab" in visible
        assert "record_tab" in visible
        assert "recon_tab" not in visible
        assert "focus_tab" not in visible
    finally:
        sidebar.deleteLater()


def test_switching_modes_lands_on_first_visible_tab():
    """After a mode switch the current tab must be one that's
    actually visible in the new mode, not left stranded on a
    hidden pane."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.sidebar_tabs import SidebarTabs

    sidebar = SidebarTabs()
    try:
        sidebar.set_workflow_mode("analyse")
        # Should be landing on focus or qpi.
        current = sidebar.tabs.currentWidget()
        assert current in (sidebar.focus_tab, sidebar.qpi_tab)
    finally:
        sidebar.deleteLater()


# ---- Signal + accepts enum or string -----------------------------------

def test_workflow_mode_changed_signal_fires_on_switch():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.sidebar_tabs import SidebarTabs

    sidebar = SidebarTabs()
    try:
        seen: list[str] = []
        sidebar.workflow_mode_changed.connect(lambda m: seen.append(m))
        sidebar.set_workflow_mode("analyse")
        sidebar.set_workflow_mode("acquire")
        sidebar.set_workflow_mode("acquire")  # noop — must not double-fire
        assert seen == ["analyse", "acquire"]
    finally:
        sidebar.deleteLater()


def test_set_workflow_mode_accepts_enum():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.sidebar_tabs import SidebarTabs, WorkflowMode

    sidebar = SidebarTabs()
    try:
        sidebar.set_workflow_mode(WorkflowMode.ANALYSE)
        assert sidebar.workflow_mode() is WorkflowMode.ANALYSE
    finally:
        sidebar.deleteLater()


def test_set_workflow_mode_rejects_unknown_string():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.sidebar_tabs import SidebarTabs

    sidebar = SidebarTabs()
    try:
        with pytest.raises(ValueError):
            sidebar.set_workflow_mode("bogus")
    finally:
        sidebar.deleteLater()


# ---- MainWindow persistence --------------------------------------------

def test_main_window_restores_saved_workflow_mode():
    """Persist a mode, boot MainWindow, assert the sidebar restored it."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow
    from gui.sidebar.sidebar_tabs import WorkflowMode

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="wf-persist-"))

    # Seed the persisted mode BEFORE creating the window so the
    # restore branch fires on construction.
    qs = QSettings("DHM", "Reconstruction")
    qs.setValue("ui/workflow_mode", "analyse")
    qs.sync()

    mw = MainWindow()
    try:
        # Restore is QTimer.singleShot(0, ...) deferred to dodge a
        # macOS first-paint segfault; drain the event loop once so
        # the timer fires inside the test.
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        assert mw.sidebar_tabs.workflow_mode() is WorkflowMode.ANALYSE
    finally:
        mw.close()
