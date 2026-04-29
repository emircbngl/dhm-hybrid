"""PhasePanel depth overlay + toggle smoke.

Overlay is pure presentation — we set a synthetic depth array, check
that an overlay ImageItem is added to the underlying ViewBox, and
that ``clear_depth_overlay`` takes it back off. The ``toggle`` command
wiring is exercised by instantiating a MainWindow and checking the
handler reacts to a loaded hologram vs empty state.
"""
from __future__ import annotations

import os

import numpy as np
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


def _sample_depth(shape=(64, 64), *, include_nan: bool = True):
    """Linear z-gradient across x, with a NaN block in the corner."""
    y, x = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    depth = -10e-3 + (x / shape[1]) * (-15e-3 - (-10e-3))
    if include_nan:
        depth[:8, :8] = np.nan
    return depth


# ---- PhasePanel ---------------------------------------------------------

def test_set_depth_overlay_adds_image_item():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        assert not panel.has_depth_overlay()
        panel.set_depth_overlay(_sample_depth(), alpha=0.5)
        assert panel.has_depth_overlay()

        # The overlay item is parented to the ViewBox's additional items.
        view = panel.image_panel.get_view()
        items = view.addedItems
        assert panel._depth_overlay_item in items
        # Z-value puts it above the base image.
        assert panel._depth_overlay_item.zValue() > 0
    finally:
        panel.deleteLater()


def test_clear_depth_overlay_removes_item():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        panel.set_depth_overlay(_sample_depth())
        assert panel.has_depth_overlay()
        panel.clear_depth_overlay()
        assert not panel.has_depth_overlay()
        view = panel.image_panel.get_view()
        assert panel._depth_overlay_item is None
        # No residual overlay item left in the view.
        assert not any(
            type(it).__name__ == "ImageItem" and it.zValue() > 0
            for it in view.addedItems
        )
    finally:
        panel.deleteLater()


def test_set_depth_overlay_second_call_updates_in_place():
    """Second call must not stack a new ImageItem — it reuses the
    existing one. Regression catch: a naive implementation would keep
    adding layers on every redraw."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        panel.set_depth_overlay(_sample_depth())
        first = panel._depth_overlay_item
        panel.set_depth_overlay(_sample_depth() * 0.8)
        second = panel._depth_overlay_item
        assert first is second
    finally:
        panel.deleteLater()


def test_set_depth_overlay_all_nan_clears():
    """An entirely-NaN map is equivalent to 'nothing to show' — clear
    any existing overlay rather than paint a transparent rectangle."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        panel.set_depth_overlay(_sample_depth())
        panel.set_depth_overlay(np.full((32, 32), np.nan, dtype=np.float32))
        assert not panel.has_depth_overlay()
    finally:
        panel.deleteLater()


def test_set_depth_overlay_empty_array_clears():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        panel.set_depth_overlay(_sample_depth())
        panel.set_depth_overlay(np.zeros((0, 0), dtype=np.float32))
        assert not panel.has_depth_overlay()
    finally:
        panel.deleteLater()


# ---- main_window wiring -------------------------------------------------

def test_toggle_depth_overlay_command_registered():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.commands import get_registry, reset_registry_for_tests
    from gui.main_window import MainWindow

    reset_registry_for_tests()
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="overlay-cmd-"))
    mw = MainWindow()
    try:
        cmd = get_registry().get("tools.toggle_depth_overlay")
        assert cmd is not None
        # Without a hologram, the when-predicate should be False.
        assert cmd.when() is False
    finally:
        mw.close()


def test_toggle_handler_is_defined_on_main_window():
    from gui.main_window import MainWindow
    assert hasattr(MainWindow, "_on_toggle_depth_overlay_triggered")


# ---- Cluster centroid markers (v1.3-polish) ------------------------------

def _sample_clusters():
    from core.depth_map import ClusterHeight
    return [
        ClusterHeight(
            cluster_id=1, centroid_yx=(32.0, 48.0),
            area_px=300, z_mean_m=-12.0e-3, z_std_m=1.0e-3,
            mean_confidence=0.85,
        ),
        ClusterHeight(
            cluster_id=2, centroid_yx=(100.0, 120.0),
            area_px=180, z_mean_m=-17.5e-3, z_std_m=0.8e-3,
            mean_confidence=0.70,
        ),
    ]


def test_set_cluster_markers_adds_one_dot_and_one_label_per_cluster():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        assert not panel.has_cluster_markers()
        panel.set_cluster_markers(_sample_clusters())
        assert panel.has_cluster_markers()
        # Two clusters → 2 dots + 2 labels = 4 items.
        assert len(panel._cluster_marker_items) == 4
        # Spot-check: label text contains cluster id + z in mm.
        texts = [
            it.toPlainText() for it in panel._cluster_marker_items
            if hasattr(it, "toPlainText")
        ]
        assert any("#1" in t and "-12" in t for t in texts)
        assert any("#2" in t and "-17" in t for t in texts)
    finally:
        panel.deleteLater()


def test_set_cluster_markers_replaces_previous_set():
    """Second call wipes the first set before painting the new one —
    no stale dots accumulating on repeated recomputes."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        panel.set_cluster_markers(_sample_clusters())
        assert len(panel._cluster_marker_items) == 4
        # Replace with a single-cluster set.
        panel.set_cluster_markers(_sample_clusters()[:1])
        assert len(panel._cluster_marker_items) == 2  # 1 dot + 1 label
    finally:
        panel.deleteLater()


def test_clear_cluster_markers_removes_items():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        panel.set_cluster_markers(_sample_clusters())
        panel.clear_cluster_markers()
        assert not panel.has_cluster_markers()
        assert panel._cluster_marker_items == []
    finally:
        panel.deleteLater()


def test_set_cluster_markers_empty_list_is_noop():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.panels.phase_panel import PhasePanel

    panel = PhasePanel()
    try:
        panel.set_cluster_markers([])
        assert not panel.has_cluster_markers()
    finally:
        panel.deleteLater()


# ---- Live overlay recompute (v1.3-polish) --------------------------------

def test_live_overlay_timer_installed_on_main_window():
    """Observer + debounce timer must exist after MainWindow construction."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings, QTimer
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="live-obs-"))
    mw = MainWindow()
    try:
        timer = getattr(mw, "_overlay_recompute_timer", None)
        assert timer is not None
        assert isinstance(timer, QTimer)
        assert timer.isSingleShot()
    finally:
        mw.close()


def test_focus_params_change_ignored_when_overlay_off():
    """The observer must not schedule a recompute while the overlay
    is inactive — otherwise innocent Focus-tab edits would burn CPU."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="live-off-"))
    mw = MainWindow()
    try:
        assert not mw.panel_phase.has_depth_overlay()
        # Fire the signal manually.
        mw._on_focus_params_changed()
        # Timer must not be running.
        assert not mw._overlay_recompute_timer.isActive()
    finally:
        mw.close()


def test_focus_params_change_starts_timer_when_overlay_on():
    """With a real overlay painted, the observer schedules the
    debounce timer. We don't wait for the timeout — just verify the
    request landed."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    import numpy as np
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="live-on-"))
    mw = MainWindow()
    try:
        # Paint a synthetic overlay directly so we don't need a real
        # hologram.
        mw.panel_phase.set_depth_overlay(
            _sample_depth(shape=(32, 32), include_nan=False),
            alpha=0.5,
        )
        assert mw.panel_phase.has_depth_overlay()

        mw._on_focus_params_changed()
        assert mw._overlay_recompute_timer.isActive()
    finally:
        mw.close()


def test_fire_overlay_recompute_is_noop_when_overlay_off():
    """Even if the timer fires after the user toggled the overlay
    off, the recompute must not crash and must not re-paint."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="live-fire-"))
    mw = MainWindow()
    try:
        assert not mw.panel_phase.has_depth_overlay()
        # Must not raise, must not paint.
        mw._fire_overlay_recompute()
        assert not mw.panel_phase.has_depth_overlay()
    finally:
        mw.close()


def test_toggle_without_hologram_is_silent_no_overlay():
    """Calling the handler with no hologram must not crash or add an
    overlay to an unready panel."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    import tempfile
    from PySide6.QtCore import QSettings
    from gui.main_window import MainWindow

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      tempfile.mkdtemp(prefix="overlay-empty-"))
    mw = MainWindow()
    try:
        # No loaded array.
        mw._on_toggle_depth_overlay_triggered()
        assert not mw.panel_phase.has_depth_overlay()
    finally:
        mw.close()
