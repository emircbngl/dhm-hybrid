"""Onboarding wizard — page count + first-run flag + command reopen."""
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


# ---- Wizard construction ------------------------------------------------

def test_wizard_constructs_with_expected_page_count():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.onboarding import OnboardingWizard

    w = OnboardingWizard()
    try:
        # Intro + 3 workflow steps + outro.
        assert len(w.pageIds()) == 5
        # Every page has a title set.
        for page_id in w.pageIds():
            page = w.page(page_id)
            assert page.title().strip(), f"page {page_id} has no title"
    finally:
        w.deleteLater()


def test_wizard_start_page_has_no_back_button_option():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtWidgets import QWizard
    from gui.widgets.onboarding import OnboardingWizard

    w = OnboardingWizard()
    try:
        assert w.testOption(QWizard.WizardOption.NoBackButtonOnStartPage)
    finally:
        w.deleteLater()


def test_module_imports():
    import gui.widgets.onboarding as m  # noqa: F401
    assert hasattr(m, "OnboardingWizard")


# ---- Command registration + handler presence ---------------------------

def test_show_onboarding_command_registered():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.commands import get_registry, reset_registry_for_tests
    from gui.main_window import MainWindow

    reset_registry_for_tests()
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="onb-cmd-"))
    mw = MainWindow()
    try:
        cmd = get_registry().get("help.show_onboarding")
        assert cmd is not None
        assert "onboarding" in cmd.title.lower()
    finally:
        mw.close()


def test_main_window_handlers_defined():
    from gui.main_window import MainWindow
    assert hasattr(MainWindow, "_show_onboarding")
    assert hasattr(MainWindow, "_maybe_show_onboarding")


# ---- First-run flag behaviour ------------------------------------------

def test_maybe_show_skips_when_flag_is_set():
    """If ``ui/onboarding_seen`` is truthy, the auto-show branch must
    not schedule the wizard."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="onb-seen-"))
    qs = QSettings("DHM", "Reconstruction")
    qs.setValue("ui/onboarding_seen", "1")
    qs.sync()

    mw = MainWindow()
    try:
        called: list[bool] = []
        # Replace the open call so we can detect auto-show without
        # actually spinning a modal inside the test.
        mw._show_onboarding = lambda: called.append(True)  # type: ignore
        mw._maybe_show_onboarding()
        # Timer deferred — still shouldn't have called the opener.
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        assert called == []
    finally:
        mw.close()


def test_show_onboarding_sets_flag():
    """Calling ``_show_onboarding`` must flip the flag even if the
    user closes the dialog immediately."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="onb-flag-"))
    mw = MainWindow()
    try:
        # Replace exec() so the test doesn't actually show the dialog.
        from gui.widgets import onboarding as onb_mod
        from PySide6.QtWidgets import QDialog

        class _StubWizard(QDialog):
            """Non-modal stub — just needs ``finished`` and ``show``
            for the host to hook into."""
            def __init__(self, parent=None):
                super().__init__(parent)

        mw._qt_settings.remove("ui/onboarding_seen")
        orig = onb_mod.OnboardingWizard
        onb_mod.OnboardingWizard = _StubWizard  # type: ignore[assignment]
        try:
            mw._show_onboarding()
            # Wizard is modeless now — emit ``finished`` ourselves to
            # simulate the user closing it.
            mw._onboarding_dialog.finished.emit(0)
        finally:
            onb_mod.OnboardingWizard = orig  # type: ignore[assignment]

        assert mw._qt_settings.value("ui/onboarding_seen") == "1"
    finally:
        mw.close()
