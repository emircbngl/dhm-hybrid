"""Theme system — token shape + apply roundtrip + persistence.

Token dicts are pure data, so they can be tested without Qt. The
Qt-side apply / persistence tests spin an offscreen QApplication and
drive the MainWindow's command handler directly.
"""
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


# ---- Token dicts (Qt-free) -----------------------------------------------

def test_theme_name_enum_values():
    from gui.theme import ThemeName
    assert {m.value for m in ThemeName} == {
        "light", "dark", "system", "high_contrast",
    }


def test_light_and_dark_token_dicts_have_required_roles():
    """Both palettes must cover the QPalette.ColorRole names we read
    from in `apply_theme`. A missing role would leave a default Qt
    colour bleeding through — easy to miss visually, catch it here."""
    from gui.theme import ThemeName, get_theme_tokens

    required = {
        "Window", "WindowText", "Base", "AlternateBase", "Text",
        "Button", "ButtonText", "Highlight", "HighlightedText",
        "Mid", "Dark", "Light", "ToolTipBase", "ToolTipText",
    }
    for theme in (ThemeName.LIGHT, ThemeName.DARK):
        tokens = get_theme_tokens(theme)
        missing = required - tokens.keys()
        assert not missing, f"{theme.value} missing roles: {missing}"
        for key, rgb in tokens.items():
            assert len(rgb) == 3
            for c in rgb:
                assert 0 <= int(c) <= 255, f"{theme.value}/{key}: {c}"


def test_system_theme_has_no_tokens():
    from gui.theme import ThemeName, get_theme_tokens
    with pytest.raises(ValueError):
        get_theme_tokens(ThemeName.SYSTEM)


def test_dark_and_light_differ():
    """Sanity: the two explicit palettes should not be byte-identical."""
    from gui.theme import ThemeName, get_theme_tokens
    assert get_theme_tokens(ThemeName.LIGHT) != get_theme_tokens(ThemeName.DARK)


# ---- apply_theme + Qt ----------------------------------------------------

def test_apply_theme_sets_palette_roles():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtGui import QPalette
    from gui.theme import ThemeName, apply_theme, reset_for_tests

    reset_for_tests()
    apply_theme(app, ThemeName.DARK)
    palette = app.palette()
    # Dark theme Window role is near-black.
    col = palette.color(QPalette.ColorRole.Window)
    assert col.red() < 80 and col.green() < 80 and col.blue() < 80


def test_apply_light_then_dark_changes_palette():
    """Light → Dark cycle must actually swap the Window colour.
    We test this cycle (not Dark → System) because the offscreen
    platform's default palette happens to be dark on some Qt builds,
    which makes Dark / System indistinguishable. Light is always
    bright enough to produce a visible delta."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtGui import QPalette
    from gui.theme import ThemeName, apply_theme, reset_for_tests

    reset_for_tests()
    apply_theme(app, ThemeName.LIGHT)
    light_window = app.palette().color(QPalette.ColorRole.Window).rgb()
    apply_theme(app, ThemeName.DARK)
    dark_window = app.palette().color(QPalette.ColorRole.Window).rgb()
    assert light_window != dark_window


def test_apply_system_does_not_crash():
    """Switching to SYSTEM must succeed whatever the platform default
    is — we don't assert a specific colour because the offscreen
    palette varies by Qt build."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.theme import ThemeName, apply_theme, reset_for_tests

    reset_for_tests()
    apply_theme(app, ThemeName.LIGHT)
    apply_theme(app, ThemeName.SYSTEM)  # must not raise


# ---- main_window command + persistence -----------------------------------

def test_theme_commands_registered():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.commands import get_registry, reset_registry_for_tests
    from gui.main_window import MainWindow

    reset_registry_for_tests()
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="theme-cmd-"))
    mw = MainWindow()
    try:
        reg = get_registry()
        for cmd_id in ("view.theme.light", "view.theme.dark",
                       "view.theme.system"):
            assert reg.get(cmd_id) is not None, f"{cmd_id} not registered"
    finally:
        mw.close()


def test_apply_theme_by_name_persists():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow
    from gui.theme import reset_for_tests

    reset_for_tests()
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="theme-persist-"))
    mw = MainWindow()
    try:
        mw._apply_theme_by_name("dark")
        # Read back directly from the window's QSettings handle.
        saved = mw._qt_settings.value("ui/theme")
        assert str(saved) == "dark"
    finally:
        mw.close()


def test_apply_theme_by_name_invalid_is_silent():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="theme-invalid-"))
    mw = MainWindow()
    try:
        # Must not raise.
        mw._apply_theme_by_name("fuchsia")
        # And must not have written the bogus value.
        saved = mw._qt_settings.value("ui/theme")
        assert saved in (None, "", "dark", "light", "system")
    finally:
        mw.close()
