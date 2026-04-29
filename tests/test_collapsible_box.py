"""CollapsibleBox + ReconTab progressive-disclosure smoke (v1.4)."""
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


# ---- Widget basics ------------------------------------------------------

def test_collapsible_box_imports():
    import gui.widgets.collapsible_box as m  # noqa: F401
    assert hasattr(m, "CollapsibleBox")


def test_collapsible_box_defaults_to_collapsed():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.collapsible_box import CollapsibleBox

    box = CollapsibleBox("Advanced")
    try:
        assert not box.isExpanded()
        # Arrow in the header reflects collapsed state.
        assert "\u25b8" in box._toggle.text()  # ▸
        # Content frame hidden.
        assert not box.content().isVisible()
    finally:
        box.deleteLater()


def test_collapsible_box_set_expanded_updates_content_visibility():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.collapsible_box import CollapsibleBox

    box = CollapsibleBox("Advanced")
    # Needs to be realised so isVisible() reports truthfully.
    box.show()
    try:
        box.setExpanded(True)
        assert box.isExpanded()
        assert "\u25be" in box._toggle.text()  # ▾
        assert box.content().isVisible()

        box.setExpanded(False)
        assert not box.isExpanded()
        assert "\u25b8" in box._toggle.text()
        assert not box.content().isVisible()
    finally:
        box.deleteLater()


def test_collapsible_box_toggle_via_header_click():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.collapsible_box import CollapsibleBox

    box = CollapsibleBox("Advanced")
    try:
        box._toggle.click()
        assert box.isExpanded()
        box._toggle.click()
        assert not box.isExpanded()
    finally:
        box.deleteLater()


def test_collapsible_box_emits_toggled_signal_on_change():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.collapsible_box import CollapsibleBox

    box = CollapsibleBox("Advanced")
    try:
        seen: list[bool] = []
        box.toggled.connect(lambda s: seen.append(s))
        box.setExpanded(True)
        box.setExpanded(True)   # noop — must not re-emit
        box.setExpanded(False)
        assert seen == [True, False]
    finally:
        box.deleteLater()


def test_set_content_layout_once_then_raises():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from PySide6.QtWidgets import QLabel, QVBoxLayout
    from gui.widgets.collapsible_box import CollapsibleBox

    box = CollapsibleBox("Section")
    try:
        layout = QVBoxLayout()
        layout.addWidget(QLabel("hi"))
        box.setContentLayout(layout)
        # Second call must raise — Qt doesn't allow layout re-parenting.
        with pytest.raises(RuntimeError):
            box.setContentLayout(QVBoxLayout())
    finally:
        box.deleteLater()


# ---- ReconTab integration ----------------------------------------------

def test_recon_tab_has_advanced_collapsible_box_default_collapsed():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.recon_tab import ReconTab

    tab = ReconTab()
    try:
        assert hasattr(tab, "advanced_box")
        assert not tab.advanced_box.isExpanded()
    finally:
        tab.deleteLater()


def test_recon_tab_advanced_widgets_still_reachable_via_attrs():
    """Moving widgets into the CollapsibleBox must not break the
    attribute paths main_window / persistence rely on."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.recon_tab import ReconTab

    tab = ReconTab()
    try:
        for name in ("method_combo", "fft_backend_combo",
                     "magnification", "pixel_is_effective_cb"):
            widget = getattr(tab, name, None)
            assert widget is not None, f"ReconTab.{name} missing"
    finally:
        tab.deleteLater()


def test_recon_tab_get_state_still_returns_all_keys():
    """The existing profile-manager contract (``get_state`` must return
    wavelength, method, magnification, etc.) must not regress."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.recon_tab import ReconTab

    tab = ReconTab()
    try:
        state = tab.get_state()
        required = {
            "method", "fft_backend", "wavelength_nm", "magnification",
            "pixel_um", "pixel_is_effective", "z_mm", "mask_radius",
        }
        assert required.issubset(state.keys()), (
            f"get_state missing: {required - state.keys()}"
        )
    finally:
        tab.deleteLater()


def test_focus_tab_has_advanced_box_default_collapsed():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.focus_tab import FocusTab

    tab = FocusTab()
    try:
        assert hasattr(tab, "advanced_box")
        assert not tab.advanced_box.isExpanded()
    finally:
        tab.deleteLater()


def test_focus_tab_advanced_subgroups_reachable():
    """The four advanced subgroups (adapt_dist, adapt_step, roi_tracker,
    live adapt) must still be reachable via their existing attribute
    paths so connect_signals, profiles, and get_state keep working."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.focus_tab import FocusTab

    tab = FocusTab()
    try:
        for name in ("adapt_dist_group", "adapt_step_group",
                     "roi_tracker_group", "adapt_enable_cb",
                     "ad_initial_range_mm", "step_init_mm",
                     "roi_size_spin"):
            assert hasattr(tab, name), f"FocusTab.{name} missing after refactor"
    finally:
        tab.deleteLater()


def test_focus_tab_get_state_still_returns_advanced_keys():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.focus_tab import FocusTab

    tab = FocusTab()
    try:
        state = tab.get_state()
        required = {
            "metric", "zscan_min_mm", "zscan_max_mm", "zscan_steps",
            "adapt_dist_enabled", "ad_initial_range_mm",
            "adapt_step_enabled", "step_init_mm",
            "roi_tracker_enabled", "roi_size", "adapt_enable",
        }
        missing = required - state.keys()
        assert not missing, f"get_state missing: {missing}"
    finally:
        tab.deleteLater()


def test_focus_tab_set_state_applies_to_widgets_inside_advanced_box():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.focus_tab import FocusTab

    tab = FocusTab()
    try:
        tab.set_state({
            "zscan_min_mm": -5.0,
            "zscan_max_mm": 5.0,
            "adapt_dist_enabled": True,
            "ad_initial_range_mm": 0.75,
            "step_init_mm": 0.05,
            "roi_tracker_enabled": True,
            "roi_size": 96,
            "adapt_enable": True,
            "adapt_range_mm": 1.25,
        })
        assert tab.zscan_min_mm.value() == pytest.approx(-5.0)
        assert tab.zscan_max_mm.value() == pytest.approx(5.0)
        assert tab.adapt_dist_group.isChecked() is True
        assert tab.ad_initial_range_mm.value() == pytest.approx(0.75)
        assert tab.step_init_mm.value() == pytest.approx(0.05)
        assert tab.roi_tracker_group.isChecked() is True
        assert tab.roi_size_spin.value() == 96
        assert tab.adapt_enable_cb.isChecked() is True
        assert tab.adapt_range_mm.value() == pytest.approx(1.25)
    finally:
        tab.deleteLater()


def test_recon_tab_set_state_roundtrip_after_refactor():
    """``set_state`` must still push values into every widget,
    including the ones that moved into Advanced."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.sidebar.recon_tab import ReconTab

    tab = ReconTab()
    try:
        tab.set_state({
            "method": "Fresnel",
            "fft_backend": "NumPy",
            "wavelength_nm": 488.0,
            "magnification": 40.0,
            "pixel_um": 3.45,
            "pixel_is_effective": False,
            "z_mm": 12.5,
            "mask_radius": 120,
        })
        # Each should land on its widget regardless of which group it lives in.
        assert tab.wavelength_nm.value() == pytest.approx(488.0)
        assert tab.magnification.value() == pytest.approx(40.0)
        assert tab.pixel_um.value() == pytest.approx(3.45)
        assert tab.pixel_is_effective_cb.isChecked() is False
        assert tab.mask_radius.value() == 120
        assert tab.method_combo.currentText() == "Fresnel"
        assert tab.fft_backend_combo.currentText() == "NumPy"
    finally:
        tab.deleteLater()
