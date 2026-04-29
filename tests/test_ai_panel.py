"""AIPanel and AISettingsDialog Qt-level smoke tests.

These run under ``QT_QPA_PLATFORM=offscreen`` (set by conftest in this
project) so no real window appears. We assert construction succeeds,
the public state-cache setters write into the panel, and the settings
dialog round-trips an :class:`AIDefaults`.

Sending a real prompt is skipped — that needs a running LLM endpoint
and is covered by the manual smoke checklist.
"""
from __future__ import annotations

import os

import pytest

# Force offscreen Qt before any Qt module imports happen elsewhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from core.settings_schema import AIDefaults
from gui.dialogs.ai_settings_dialog import AISettingsDialog
from gui.panels.ai_panel import AIPanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _StubMainWindow:
    """Minimal MainWindow stand-in so AIPanel can construct without
    pulling the whole app."""
    sidebar_tabs = None
    _settings = None
    _loaded_path = None
    _loaded_array = None


def test_panel_constructs_with_stub_main_window(qapp):
    panel = AIPanel(_StubMainWindow())
    try:
        assert panel.windowTitle() == "AI Assistant"
        # Send / Stop buttons exist and have the right enabled state.
        send = panel.findChild(type(panel)._dummy(), "ai_send_button") if False else None
        # Look up by object name through the widget tree
        from PySide6.QtWidgets import QPushButton
        send_btn = panel.findChild(QPushButton, "ai_send_button")
        stop_btn = panel.findChild(QPushButton, "ai_stop_button")
        assert send_btn is not None
        assert stop_btn is not None
        assert send_btn.isEnabled() is True
        assert stop_btn.isEnabled() is False
    finally:
        panel.deleteLater()


def test_panel_set_loaded_updates_cache(qapp):
    import numpy as np
    panel = AIPanel(_StubMainWindow())
    try:
        arr = np.zeros((128, 64), dtype=np.float32)
        panel.set_loaded("/tmp/x.tif", arr)
        snap = panel._build_snapshot()
        assert snap.loaded_path == "/tmp/x.tif"
        assert snap.loaded_shape == (128, 64)
        assert snap.loaded_dtype == "float32"
    finally:
        panel.deleteLater()


def test_panel_set_summary_setters(qapp):
    panel = AIPanel(_StubMainWindow())
    try:
        panel.set_recon_summary({"phase_std": 0.5})
        panel.set_af_summary({"best_z_mm": 12.4})
        panel.set_qpi_summary({"total_dry_mass_pg": 64.2})
        panel.set_depth_summary({"valid_pct": 87.0})
        snap = panel._build_snapshot()
        assert snap.last_recon_summary["phase_std"] == 0.5
        assert snap.last_af_summary["best_z_mm"] == 12.4
        assert snap.last_qpi_summary["total_dry_mass_pg"] == 64.2
        assert snap.last_depth_summary["valid_pct"] == 87.0
    finally:
        panel.deleteLater()


def test_panel_apply_ai_settings_swaps_client(qapp):
    panel = AIPanel(_StubMainWindow())
    try:
        original_endpoint = panel._client.config.endpoint
        new = AIDefaults(endpoint_url="http://localhost:1234",
                         model_name="qwen2.5:7b-instruct")
        panel.apply_ai_settings(new)
        assert panel._client.config.endpoint == "http://localhost:1234"
        assert panel._client.config.endpoint != original_endpoint
        assert panel._client.config.model == "qwen2.5:7b-instruct"
    finally:
        panel.deleteLater()


def test_panel_clear_resets_history(qapp):
    panel = AIPanel(_StubMainWindow())
    try:
        # Simulate a few rendered messages.
        panel._history.append(object())
        panel._history.append(object())
        panel._on_clear_clicked()
        assert panel._history == []
    finally:
        panel.deleteLater()


# --- AISettingsDialog --------------------------------------------------------


def test_settings_dialog_round_trip_preserves_values(qapp):
    initial = AIDefaults(
        endpoint_url="http://localhost:1234",
        model_name="qwen2.5:7b-instruct",
        temperature=0.55,
        max_iterations=10,
        restrict_to_home=False,
    )
    dlg = AISettingsDialog(initial)
    try:
        # Simulate Accept by directly invoking the on-accept handler so
        # the dialog doesn't need to actually be exec()'d.
        dlg._on_accept()
        out = dlg.result_settings()
        assert out is not None
        assert out.endpoint_url == "http://localhost:1234"
        assert out.model_name == "qwen2.5:7b-instruct"
        assert out.temperature == pytest.approx(0.55)
        assert out.max_iterations == 10
        assert out.restrict_to_home is False
    finally:
        dlg.deleteLater()


def test_settings_dialog_returns_none_before_accept(qapp):
    dlg = AISettingsDialog(AIDefaults())
    try:
        assert dlg.result_settings() is None
    finally:
        dlg.deleteLater()


# --- Persistence path: per-user dir, not flat home ---------------------------


def test_persist_sample_map_writes_under_user_state_dir(qapp, tmp_path, monkeypatch):
    """``_persist_sample_map`` must use ``core.user_profile.user_state_dir``
    so multi-user installs don't collide on a shared
    ``~/.dhm-reconstruction/sample_maps``.

    We redirect the user_profile root to ``tmp_path`` and verify the
    file lands at ``<tmp>/users/<user>/sample_maps/<id>.json``.
    """
    from core import user_profile
    from core.sample_map import CellLocation

    monkeypatch.setenv("DHM_USER", "alice")
    user_profile.set_root_dir(tmp_path / "dhm")
    try:
        panel = AIPanel(_StubMainWindow())
        try:
            panel._sample_map.add(CellLocation(
                cell_id=1, stage_x_mm=0.0, stage_y_mm=0.0, stage_z_mm=0.0,
                in_frame_y_px=0, in_frame_x_px=0,
            ))
            panel._sample_map.stamp(
                x_min=0, x_max=1, y_min=0, y_max=1, step_mm=0.5,
                sample_id="round_trip",
            )
            result_path = panel._persist_sample_map()
            assert result_path is not None
            from pathlib import Path
            target = Path(result_path)
            assert target.exists()
            # Must sit *inside* the per-user dir, not at the project root.
            assert "users" in target.parts
            assert "alice" in target.parts
            assert target.name == "round_trip.json"
        finally:
            panel.deleteLater()
    finally:
        user_profile.set_root_dir(None)


# --- Race fix: wait_for_signal drains the event queue ------------------------


def test_wait_for_signal_returns_true_when_signal_fires(qapp):
    from PySide6.QtCore import QObject, QTimer, Signal

    class _Emitter(QObject):
        fired = Signal()

    em = _Emitter()
    # Schedule the signal a tick after we enter the wait — the
    # nested event loop must process it.
    QTimer.singleShot(50, em.fired.emit)
    fired = AIPanel._wait_for_signal(em.fired, timeout_ms=5000)
    assert fired is True


def test_wait_for_signal_returns_false_on_timeout(qapp):
    from PySide6.QtCore import QObject, Signal

    class _Emitter(QObject):
        fired = Signal()

    em = _Emitter()
    fired = AIPanel._wait_for_signal(em.fired, timeout_ms=120)
    assert fired is False
