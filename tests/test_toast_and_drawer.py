"""Toast host + error drawer — bus wiring, severity floor, drawer refresh.

Same strategy as ``test_command_palette.py``: skip Qt-dependent tests
when a ``QApplication`` can't be created in offscreen mode, but test
the plumbing (subscription, filtering, refresh) wherever possible.
"""
from __future__ import annotations

import os
import time

import pytest

from core.errors import ErrorCenter, ErrorEvent, Severity


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


def _emit_sync(center: ErrorCenter, **overrides) -> ErrorEvent:
    """Build an ErrorEvent and push it through the center immediately."""
    defaults = dict(
        severity=Severity.ERROR,
        title="Synthetic failure",
        cause="something went wrong",
        action="Try again.",
        context={"k": 1},
    )
    defaults.update(overrides)
    event = ErrorEvent.make(**defaults)
    center.emit(event)
    return event


# ---- imports don't need Qt --------------------------------------------------

def test_toast_module_imports():
    import gui.widgets.toast as t  # noqa: F401
    assert hasattr(t, "ToastHost")
    assert hasattr(t, "Toast")


def test_drawer_module_imports():
    import gui.widgets.error_drawer as d  # noqa: F401
    assert hasattr(d, "ErrorDrawer")


# ---- Qt-dependent: ToastHost ------------------------------------------------

def test_toast_host_subscribes_and_filters_by_severity():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtWidgets import QWidget
    from gui.widgets.toast import ToastHost

    center = ErrorCenter()
    parent = QWidget()
    host = ToastHost(parent, center=center)
    host.attach()

    # INFO is below the WARN floor → no toast scheduled.
    _emit_sync(center, severity=Severity.INFO, title="Just info")
    # WARN and ERROR both surface.
    _emit_sync(center, severity=Severity.WARN, title="Warn one")
    _emit_sync(center, severity=Severity.ERROR, title="Error one")

    # The host uses QTimer.singleShot to marshal onto the Qt thread; run
    # the event loop briefly so those fire.
    app.processEvents()
    time.sleep(0.01)
    app.processEvents()

    titles = [t.event.title for t in host._stack]
    assert "Just info" not in titles
    assert "Warn one" in titles
    assert "Error one" in titles

    host.detach()
    parent.deleteLater()


def test_toast_host_stack_cap_drops_oldest():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtWidgets import QWidget
    from gui.widgets.toast import ToastHost, _MAX_STACK

    center = ErrorCenter()
    parent = QWidget()
    host = ToastHost(parent, center=center)
    host.attach()

    for i in range(_MAX_STACK + 2):
        _emit_sync(center, title=f"E{i}")

    app.processEvents()
    time.sleep(0.01)
    app.processEvents()

    assert len(host._stack) <= _MAX_STACK
    # Oldest ("E0", "E1") should have been evicted.
    titles = [t.event.title for t in host._stack]
    assert "E0" not in titles

    host.detach()
    parent.deleteLater()


def test_toast_host_detach_unsubscribes():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtWidgets import QWidget
    from gui.widgets.toast import ToastHost

    center = ErrorCenter()
    parent = QWidget()
    host = ToastHost(parent, center=center)
    host.attach()
    host.detach()

    _emit_sync(center, title="After detach")
    app.processEvents()
    # The detached host's stack should stay empty.
    assert host._stack == []

    parent.deleteLater()


# ---- Qt-dependent: ErrorDrawer ---------------------------------------------

def test_drawer_refresh_shows_placeholder_when_empty():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.error_drawer import ErrorDrawer

    center = ErrorCenter()
    drawer = ErrorDrawer(center=center)
    drawer.refresh()
    # Placeholder row + stretch = 2 children in the layout.
    assert drawer._list_layout.count() == 2


def test_drawer_refresh_renders_events():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.error_drawer import ErrorDrawer

    center = ErrorCenter()
    _emit_sync(center, title="First")
    _emit_sync(center, title="Second")
    drawer = ErrorDrawer(center=center)
    drawer.refresh()
    # 2 rows + 1 separator + trailing stretch = 4.
    assert drawer._list_layout.count() == 4
    assert "(2)" in drawer._header_label.text()


def test_drawer_clear_wipes_center_history():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.error_drawer import ErrorDrawer

    center = ErrorCenter()
    _emit_sync(center, title="before clear")
    drawer = ErrorDrawer(center=center)
    drawer.refresh()
    assert len(center.recent()) == 1

    drawer._on_clear()
    assert center.recent() == []
    # Placeholder is back.
    assert drawer._list_layout.count() == 2
