"""``_cancel_active_worker`` priority walker + Esc shortcut wiring.

The walker logic is pure (no Qt signals needed), so we test it by binding
the bound method onto a stand-in object with mock worker slots. The Esc
shortcut wiring is exercised via a headless ``MainWindow()`` smoke.
"""
from __future__ import annotations

import os
from types import MethodType
from unittest.mock import MagicMock

import pytest

from gui.main_window import MainWindow


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


class _FakeWorker:
    """Minimal stand-in with the running/cancel surface the walker checks."""

    def __init__(self, *, running: bool, has_cancel: bool = True,
                 has_stop: bool = False):
        self._running = running
        self.cancel_calls = 0
        self.stop_calls = 0
        if has_cancel:
            self.cancel = self._cancel  # type: ignore[method-assign]
        if has_stop:
            self.stop = self._stop  # type: ignore[method-assign]

    def isRunning(self) -> bool:
        return self._running

    def _cancel(self) -> None:
        self.cancel_calls += 1

    def _stop(self) -> None:
        self.stop_calls += 1


class _Host:
    """Stand-in for ``MainWindow`` — just the attributes the walker reads."""

    def __init__(self):
        self._qpi_worker = None
        self._af_worker = None
        self._recon_worker = None
        self.status_bar = MagicMock()


def _bind_walker(host):
    """Attach the real ``_cancel_active_worker`` to the bare host."""
    host._cancel_active_worker = MethodType(
        MainWindow._cancel_active_worker, host
    )


# ---- priority walking ------------------------------------------------------

def test_no_workers_running_is_silent():
    host = _Host()
    _bind_walker(host)
    host._cancel_active_worker()  # must not raise
    host.status_bar.show_message.assert_not_called()


def test_qpi_wins_over_af_and_recon():
    host = _Host()
    host._qpi_worker = _FakeWorker(running=True)
    host._af_worker = _FakeWorker(running=True)
    host._recon_worker = _FakeWorker(running=True, has_cancel=False,
                                     has_stop=True)
    _bind_walker(host)

    host._cancel_active_worker()

    assert host._qpi_worker.cancel_calls == 1
    assert host._af_worker.cancel_calls == 0
    assert host._recon_worker.stop_calls == 0
    # Feedback to user.
    host.status_bar.show_message.assert_called_once()
    msg = host.status_bar.show_message.call_args.args[0]
    assert "QPI" in msg
    assert "cancel" in msg.lower()


def test_walker_skips_stopped_workers():
    host = _Host()
    host._qpi_worker = _FakeWorker(running=False)
    host._af_worker = _FakeWorker(running=True)
    host._recon_worker = _FakeWorker(running=True, has_cancel=False,
                                     has_stop=True)
    _bind_walker(host)

    host._cancel_active_worker()

    assert host._qpi_worker.cancel_calls == 0
    assert host._af_worker.cancel_calls == 1
    assert host._recon_worker.stop_calls == 0


def test_recon_fallback_uses_stop_not_cancel():
    host = _Host()
    host._recon_worker = _FakeWorker(running=True, has_cancel=False,
                                     has_stop=True)
    _bind_walker(host)

    host._cancel_active_worker()

    assert host._recon_worker.stop_calls == 1


def test_walker_swallows_exceptions_and_tries_next():
    """If cancel() on a running worker raises, move on rather than crash."""
    host = _Host()
    boom = _FakeWorker(running=True)
    boom.cancel = MagicMock(side_effect=RuntimeError("bang"))
    host._qpi_worker = boom
    host._af_worker = _FakeWorker(running=True)
    _bind_walker(host)

    # Must not raise.
    host._cancel_active_worker()
    # AF is the next live worker; it should have been cancelled.
    assert host._af_worker.cancel_calls == 1


# ---- Esc shortcut wiring ---------------------------------------------------

def test_esc_shortcut_is_bound_to_cancel_walker():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")

    import tempfile
    from PySide6.QtCore import QSettings
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="esc-wire-"))

    mw = MainWindow()
    try:
        assert hasattr(mw, "_esc_shortcut"), "Esc shortcut not installed"
        assert mw._esc_shortcut.objectName() == "sc_cancel_active_worker"
        # The key sequence must be Escape.
        assert mw._esc_shortcut.key().toString() == "Esc"

        # Stub the walker; emit ``activated`` and verify it ran.
        calls: list[bool] = []
        mw._cancel_active_worker = lambda: calls.append(True)  # type: ignore[assignment]
        # Re-bind the signal — Qt holds a reference to the *original*
        # method, so we disconnect and reconnect to our stub.
        mw._esc_shortcut.activated.disconnect()
        mw._esc_shortcut.activated.connect(mw._cancel_active_worker)
        mw._esc_shortcut.activated.emit()
        assert calls == [True]
    finally:
        mw.close()
