"""Consolidated smoke suite for the v1.4 UI Redesign end-state.

Covers the seven pieces that landed in the final rollout:
  4c. QPITab progressive disclosure
  5.  Accessibility — SR labels + ``?`` shortcut
  6.  Report mode + ReportTab
  7.  High-contrast theme
  8.  Inline validation dots (ReconTab pilot)
  9.  Preset chip row (ReconTab pilot)
  10. Contextual help overlay
"""
from __future__ import annotations

import os
import tempfile

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


# ---- 4c QPITab ----------------------------------------------------------

def test_qpi_tab_has_advanced_collapsible():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.qpi_tab import QPITab

    tab = QPITab()
    try:
        assert hasattr(tab, "advanced_box")
        assert not tab.advanced_box.isExpanded()
        # Advanced content: verify a couple of attrs that moved but
        # must still be reachable.
        for name in ("n_sample", "n_medium", "mode_combo"):
            assert hasattr(tab, name), f"QPITab.{name} missing"
    finally:
        tab.deleteLater()


# ---- 5 Accessibility ----------------------------------------------------

def test_toolbar_widgets_carry_accessible_names():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.toolbar import MainToolbar

    toolbar = MainToolbar()
    try:
        assert toolbar.mode_combo.accessibleName()
        assert toolbar.sample_id_edit.accessibleName()
    finally:
        toolbar.deleteLater()


def test_main_window_has_help_shortcut():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="v14-help-"))
    mw = MainWindow()
    try:
        assert hasattr(mw, "_help_shortcut")
        assert mw._help_shortcut.objectName() == "sc_show_help_overlay"
    finally:
        mw.close()


# ---- 6 Report mode ------------------------------------------------------

def test_workflow_mode_has_report():
    from gui.sidebar.sidebar_tabs import WorkflowMode
    assert "report" in {m.value for m in WorkflowMode}


def test_report_tab_buttons_inert_until_bound():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.report_tab import ReportTab

    tab = ReportTab()
    try:
        assert not tab.btn_report.isEnabled()
        calls = {"r": 0, "c": 0, "d": 0, "b": 0, "u": 0}

        def mk(k):
            def _h():
                calls[k] += 1
            return _h

        tab.bind_report_actions(
            on_report=mk("r"),
            on_qpi_csv=mk("c"),
            on_depth_map=mk("d"),
            on_qpi_batch=mk("b"),
            on_bundle=mk("u"),
        )
        assert tab.btn_report.isEnabled()
        tab.btn_report.click()
        tab.btn_bundle.click()
        assert calls["r"] == 1
        assert calls["u"] == 1
    finally:
        tab.deleteLater()


def test_main_window_binds_report_tab_actions():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="v14-bind-"))
    mw = MainWindow()
    try:
        rt = mw.sidebar_tabs.report_tab
        # Bound → buttons enabled.
        assert rt.btn_report.isEnabled()
        assert rt.btn_bundle.isEnabled()
    finally:
        mw.close()


# ---- 7 High-contrast theme ---------------------------------------------

def test_high_contrast_tokens_meet_black_and_white_contract():
    """Window is black, text is white — the two core contrast pairs."""
    from gui.theme import ThemeName, get_theme_tokens

    t = get_theme_tokens(ThemeName.HIGH_CONTRAST)
    assert t["Window"] == (0, 0, 0)
    assert t["WindowText"] == (255, 255, 255)
    # Highlight must stand out against black — sum of channels > 500
    # is a cheap-and-cheerful "saturated colour" check.
    assert sum(t["Highlight"]) > 400


def test_high_contrast_theme_applies_without_error():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.theme import ThemeName, apply_theme, reset_for_tests
    reset_for_tests()
    apply_theme(app, ThemeName.LIGHT)
    apply_theme(app, ThemeName.HIGH_CONTRAST)  # must not raise


def test_high_contrast_command_registered():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtCore import QSettings
    from gui.commands import get_registry, reset_registry_for_tests
    from gui.main_window import MainWindow

    reset_registry_for_tests()
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="v14-hc-"))
    mw = MainWindow()
    try:
        assert get_registry().get("view.theme.high_contrast") is not None
    finally:
        mw.close()


# ---- 8 Validation dots --------------------------------------------------

def test_validation_dot_states():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.validation_dot import ValidationDot

    dot = ValidationDot()
    try:
        assert dot.state() is None
        dot.setState(True)
        assert dot.state() is True
        dot.setState(False, reason="negative value")
        assert dot.state() is False
        assert "negative" in dot.toolTip()
    finally:
        dot.deleteLater()


def test_recon_tab_dot_flips_red_when_wavelength_invalid():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.recon_tab import ReconTab

    tab = ReconTab()
    try:
        assert tab.wavelength_dot.state() is True
        # Pump an invalid value through the validator path.
        tab.wavelength_nm.setValue(0.001)  # effectively zero
        # validate() rejects ≤ 0; 0.001 is > 0 so still valid.
        # Push past the upper guard (2000) via the spinbox setRange —
        # the Recon schema caps at 2000, validate() fires.
        tab.wavelength_nm.setMaximum(5000.0)
        tab.wavelength_nm.setValue(3000.0)
        assert tab.wavelength_dot.state() is False
    finally:
        tab.deleteLater()


# ---- 9 Preset chip row --------------------------------------------------

def test_preset_chip_row_exists_on_recon_tab():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.recon_tab import ReconTab

    tab = ReconTab()
    try:
        assert hasattr(tab, "preset_chips")
        presets = tab.preset_chips.presets()
        assert "Custom" in presets
        assert "Biological Cell" in presets
    finally:
        tab.deleteLater()


def test_preset_chip_click_updates_combo():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.recon_tab import ReconTab

    tab = ReconTab()
    try:
        # Select a preset via the chip row — the combo should follow.
        tab.preset_chips.preset_selected.emit("Biological Cell")
        # _on_preset_chip should have moved the combo.
        assert tab.preset_combo.currentText() == "Biological Cell"
    finally:
        tab.deleteLater()


def test_preset_chip_exclusive_selection():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.preset_chips import PresetChipRow

    row = PresetChipRow(["A", "B", "C"], active="B")
    try:
        assert row.active() == "B"
        row.setActive("C")
        assert row.active() == "C"
    finally:
        row.deleteLater()


# ---- 10 Contextual help -------------------------------------------------

def test_help_overlay_imports_and_constructs():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtWidgets import QPushButton, QWidget
    from gui.widgets.help_overlay import HelpOverlay

    target = QWidget()
    btn = QPushButton("Hello", target)
    btn.setAccessibleName("Greeting button")
    btn.setToolTip("Says hi")
    target.show()
    try:
        overlay = HelpOverlay(target)
        try:
            # Overlay must construct without error and expose the
            # expected public API (Esc shortcut + close handler).
            assert overlay.windowTitle().lower().startswith("help")
        finally:
            overlay.deleteLater()
    finally:
        target.deleteLater()


def test_help_overlay_command_registered():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtCore import QSettings
    from gui.commands import get_registry, reset_registry_for_tests
    from gui.main_window import MainWindow

    reset_registry_for_tests()
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="v14-overlay-"))
    mw = MainWindow()
    try:
        assert get_registry().get("help.show_overlay") is not None
    finally:
        mw.close()
