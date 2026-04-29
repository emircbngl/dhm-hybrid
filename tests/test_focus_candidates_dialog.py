"""FocusCandidatesDialog — multi-focus picker UI smoke tests.

Uses offscreen Qt so the suite stays headless. Each test instantiates
the dialog directly with a hand-made candidate list and exercises the
bits that matter: row count, signal emission, empty-list placeholder.
The ``main_window`` wiring (command registration, handler presence) is
covered by a separate smoke below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

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


def _make_candidates(n: int = 3):
    from core.autofocus import FocusCandidate
    return [
        FocusCandidate(
            z_m=-(i + 1) * 5e-3,
            score=0.9 - 0.1 * i,
            prominence=0.5 - 0.1 * i,
            rank=i,
        )
        for i in range(n)
    ]


# ---- import is Qt-free ------------------------------------------------------

def test_module_imports():
    import gui.widgets.focus_candidates as fc  # noqa: F401
    assert hasattr(fc, "FocusCandidatesDialog")


# ---- Qt-dependent ----------------------------------------------------------

def test_dialog_shows_one_row_per_candidate():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.focus_candidates import FocusCandidatesDialog

    candidates = _make_candidates(3)
    dlg = FocusCandidatesDialog(candidates)
    try:
        assert dlg._table.rowCount() == 3
        # Check rank column is 1-based.
        assert dlg._table.item(0, 0).text() == "1"
        assert dlg._table.item(2, 0).text() == "3"
        # z column formatted in mm with sign.
        assert "5.000" in dlg._table.item(0, 1).text()
    finally:
        dlg.deleteLater()


def test_dialog_empty_list_shows_placeholder():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.focus_candidates import FocusCandidatesDialog

    dlg = FocusCandidatesDialog([])
    try:
        assert dlg._table.rowCount() == 0
        assert "No candidate" in dlg._header.text()
    finally:
        dlg.deleteLater()


def test_focus_here_button_emits_z_and_closes():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.focus_candidates import FocusCandidatesDialog

    candidates = _make_candidates(2)
    dlg = FocusCandidatesDialog(candidates)
    try:
        fired: list[float] = []
        dlg.focus_requested.connect(lambda z: fired.append(z))
        # Simulate click on the 2nd row's button.
        btn = dlg._table.cellWidget(1, 3)
        assert btn is not None
        btn.click()
        assert len(fired) == 1
        assert fired[0] == pytest.approx(candidates[1].z_m)
    finally:
        dlg.deleteLater()


def test_candidates_api_returns_copy():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.focus_candidates import FocusCandidatesDialog

    candidates = _make_candidates(2)
    dlg = FocusCandidatesDialog(candidates)
    try:
        snap = dlg.candidates()
        assert snap == candidates
        assert snap is not candidates  # defensive copy
    finally:
        dlg.deleteLater()


# ---- main_window integration -----------------------------------------------

def test_autofocus_find_multiple_command_registered():
    """The ⌘K palette discovery path — ``autofocus.find_multiple`` must
    be installed by ``install_main_window_commands`` and correctly
    gated on a loaded hologram."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.commands import get_registry, reset_registry_for_tests
    from gui.main_window import MainWindow

    reset_registry_for_tests()
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="mf-cmd-"))
    mw = MainWindow()
    try:
        cmd = get_registry().get("autofocus.find_multiple")
        assert cmd is not None, "autofocus.find_multiple not registered"
        assert cmd.title.lower().startswith("find multiple")
        # Without a hologram, the command's ``when`` gate is False.
        assert callable(cmd.when)
        assert cmd.when() is False
    finally:
        mw.close()


def test_main_window_handler_present():
    """MainWindow must expose ``_on_find_focus_candidates_triggered``
    so the command callback resolves. We don't invoke it — it needs
    a loaded hologram and would trigger a full scan."""
    from gui.main_window import MainWindow
    assert hasattr(MainWindow, "_on_find_focus_candidates_triggered")
    assert hasattr(MainWindow, "_apply_focus_candidate")
