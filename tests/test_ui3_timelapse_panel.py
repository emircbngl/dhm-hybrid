"""ui3 TimelapsePanel tests — offscreen Qt, real PanelContext/WorkerBridge.

Mirrors the ``_make_ctx`` / ``_pump_until`` pattern from
``tests/test_ui3_qpi_panel.py`` / ``tests/test_ui3_focus_panel.py``. This
panel doesn't touch the bridge's recon/QPI/depth compute (no hologram
needed) — it drives ``core.timelapse.TimelapseRunner`` /
``core.multi_position_timelapse.MultiPositionTimelapseRunner`` directly
against the synthetic camera + mock devices, on a background ``QThread``.

Tests run a short synthetic single-position time-lapse (2-3 frames, a small
interval) and a short multi-position run, and assert:

* the panel builds with the expected default widget state;
* mode switching toggles the position-table visibility;
* Start actually produces frames on disk + a session manifest in the chosen
  output directory, and returns the UI to an idle state;
* Stop cancels a run cleanly (no crash, no hang, worker thread joins).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DHM_USER", "ci")

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def _make_ctx(tmp_path):
    """A minimal-but-real PanelContext: a live WorkerBridge (unused by this
    panel, but part of the frozen contract), a mutable ReconParams, and
    no-op display/feedback hooks (recorded for assertions)."""
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.context import PanelContext
    from ui3.design import DARK

    bridge = WorkerBridge()
    params = ReconParams(
        wavelength_nm=632.8, pixel_um=1.0, z_mm=0.0, mask_radius=40,
        n_sample=1.38, n_medium=1.337,
    )
    events: dict = {"status": [], "toast": []}

    ctx = PanelContext(
        bridge=bridge,
        get_params=lambda: params,
        params_changed=lambda: None,
        hologram_path=lambda: None,
        last_recon=lambda: None,
        last_depth=lambda: None,
        get_field=lambda target: None,
        set_status=lambda text, level="info": events["status"].append((text, level)),
        toast=lambda text, level="info": events["toast"].append((text, level)),
        show_in_panel=lambda target, array, **kw: None,
        palette=DARK,
    )
    return ctx, bridge, params, events


def _pump_until(qapp, predicate, deadline_s=20):
    deadline = time.time() + deadline_s
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    return predicate()


def test_timelapse_panel_builds(qapp, tmp_path):
    from ui3.panels.timelapse_panel import TimelapsePanel, _MODE_SINGLE

    ctx, bridge, params, events = _make_ctx(tmp_path)
    panel = TimelapsePanel(ctx)

    assert panel.cmb_mode.currentText() == _MODE_SINGLE
    # isVisibleTo, not isVisible — the panel is never shown/top-level in
    # this offscreen test, so isVisible() would always read False
    # regardless of setVisible(); isVisibleTo(panel) reflects the
    # explicit setVisible() flag this panel actually manages.
    assert not panel.grp_positions.isVisibleTo(panel)
    assert panel.btn_start.isEnabled()
    assert not panel.btn_stop.isEnabled()
    assert panel.lbl_progress.text() == "Idle"

    bridge.shutdown()


def test_timelapse_panel_mode_switch_shows_positions(qapp, tmp_path):
    from ui3.panels.timelapse_panel import TimelapsePanel, _MODE_MULTI

    ctx, bridge, params, events = _make_ctx(tmp_path)
    panel = TimelapsePanel(ctx)

    panel.cmb_mode.setCurrentText(_MODE_MULTI)
    assert panel.grp_positions.isVisibleTo(panel)
    assert panel.lbl_total.text() == "Total ticks"

    bridge.shutdown()


def test_timelapse_panel_requires_output_dir(qapp, tmp_path):
    from ui3.panels.timelapse_panel import TimelapsePanel

    ctx, bridge, params, events = _make_ctx(tmp_path)
    panel = TimelapsePanel(ctx)

    panel._on_start_clicked()

    assert not panel.btn_stop.isEnabled()
    assert events["status"], "expected a status message"
    assert "output directory" in events["status"][-1][0].lower()

    bridge.shutdown()


def test_timelapse_panel_single_position_run_completes(qapp, tmp_path):
    """Short synthetic single-position run: 2-3 frames land in the chosen
    output directory and the UI returns to idle when done."""
    from ui3.panels.timelapse_panel import TimelapsePanel

    ctx, bridge, params, events = _make_ctx(tmp_path)
    panel = TimelapsePanel(ctx)

    out_dir = tmp_path / "tl_single"
    panel._out_dir = out_dir
    panel.lbl_out_dir.setText(str(out_dir))

    panel.spin_interval.setValue(0.05)
    panel.spin_total.setValue(3)
    panel.spin_max_duration.setValue(0)

    panel._on_start_clicked()

    assert panel._worker is not None
    assert not panel.btn_start.isEnabled()
    assert panel.btn_stop.isEnabled()

    finished = _pump_until(
        qapp, lambda: panel.btn_start.isEnabled(), deadline_s=20)
    assert finished, f"run did not finish; last progress={panel.lbl_progress.text()!r}"

    assert not panel.btn_stop.isEnabled()
    assert out_dir.exists()
    tiffs = sorted(out_dir.glob("frame_*.tif"))
    assert len(tiffs) == 3, f"expected 3 frames, found {[p.name for p in tiffs]}"
    assert (out_dir / "session.json").exists()
    assert "complete" in panel.lbl_progress.text()
    assert events["toast"], "expected a completion toast"

    bridge.shutdown()


def test_timelapse_panel_stop_is_clean(qapp, tmp_path):
    """Stop mid-run cancels without crashing/hanging; worker thread joins."""
    from ui3.panels.timelapse_panel import TimelapsePanel

    ctx, bridge, params, events = _make_ctx(tmp_path)
    panel = TimelapsePanel(ctx)

    out_dir = tmp_path / "tl_stop"
    panel._out_dir = out_dir
    panel.lbl_out_dir.setText(str(out_dir))

    # Long interval + high total so the run is still alive when we stop it.
    panel.spin_interval.setValue(5.0)
    panel.spin_total.setValue(50)
    panel.spin_max_duration.setValue(0)

    panel._on_start_clicked()
    assert panel._worker is not None

    # Let the worker thread actually start before stopping it.
    _pump_until(qapp, lambda: panel._worker.isRunning(), deadline_s=5)

    panel._on_stop_clicked()

    finished = _pump_until(
        qapp, lambda: panel.btn_start.isEnabled(), deadline_s=20)
    assert finished, "worker did not stop cleanly within deadline"
    assert "stopped" in panel.lbl_progress.text()

    bridge.shutdown()


def test_timelapse_panel_multi_position_requires_positions(qapp, tmp_path):
    from ui3.panels.timelapse_panel import TimelapsePanel, _MODE_MULTI

    ctx, bridge, params, events = _make_ctx(tmp_path)
    panel = TimelapsePanel(ctx)

    out_dir = tmp_path / "tl_multi_empty"
    panel._out_dir = out_dir
    panel.lbl_out_dir.setText(str(out_dir))
    panel.cmb_mode.setCurrentText(_MODE_MULTI)

    panel._on_start_clicked()

    assert not panel.btn_stop.isEnabled()
    assert events["status"], "expected an error status"
    assert "could not start" in events["status"][-1][0].lower()

    bridge.shutdown()


def test_timelapse_panel_multi_position_run_completes(qapp, tmp_path):
    from ui3.panels.timelapse_panel import TimelapsePanel, _MODE_MULTI

    ctx, bridge, params, events = _make_ctx(tmp_path)
    panel = TimelapsePanel(ctx)

    out_dir = tmp_path / "tl_multi"
    panel._out_dir = out_dir
    panel.lbl_out_dir.setText(str(out_dir))
    panel.cmb_mode.setCurrentText(_MODE_MULTI)

    panel._on_add_position()
    panel._on_add_position()

    panel.spin_interval.setValue(0.05)
    panel.spin_total.setValue(2)
    panel.spin_max_duration.setValue(0)

    panel._on_start_clicked()
    assert panel._worker is not None

    finished = _pump_until(
        qapp, lambda: panel.btn_start.isEnabled(), deadline_s=20)
    assert finished, f"run did not finish; last progress={panel.lbl_progress.text()!r}"

    assert out_dir.exists()
    tiffs = sorted(out_dir.glob("frame_*.tif"))
    # 2 ticks * 2 positions = 4 frames.
    assert len(tiffs) == 4, f"expected 4 frames, found {[p.name for p in tiffs]}"
    assert (out_dir / "session.json").exists()

    bridge.shutdown()


def test_timelapse_panel_link_detections_hook(qapp, tmp_path):
    """Tracking hook is wired to core.tracking.link_detections and doesn't
    crash even with zero recorded detections (the honest default when no
    detector is wired in)."""
    from ui3.panels.timelapse_panel import TimelapsePanel

    ctx, bridge, params, events = _make_ctx(tmp_path)
    panel = TimelapsePanel(ctx)

    panel._on_link_detections_clicked()

    assert "0 track" in panel.lbl_track_result.text()

    bridge.shutdown()
