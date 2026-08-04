"""Tests for ``ui3.panels.device_panel.DevicePanel``.

Offscreen Qt (QT_QPA_PLATFORM=offscreen), real ``core.devices`` mock
backends (``mock_stage`` / ``mock_shutter`` / ``mock_led``) — same
pattern as ``tests/test_ui3_focus_panel.py``: no compute/device logic
is re-implemented or mocked-out, this proves the panel actually drives
the real device Protocols end to end.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

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


class _FakeContext:
    """Minimal stand-in for ``ui3.context.PanelContext``. DevicePanel
    only uses ``set_status`` / ``toast`` for feedback — it never
    touches ``bridge`` or ``get_params``."""

    def __init__(self):
        self.status_log: List[tuple] = []
        self.toast_log: List[tuple] = []
        self.palette = None

    def get_params(self):
        raise AssertionError("DevicePanel must not touch ctx.get_params()")

    def params_changed(self):
        raise AssertionError("DevicePanel must not touch ctx.params_changed()")

    def hologram_path(self):
        return None

    def last_recon(self):
        return None

    def last_depth(self):
        return None

    def get_field(self, target: str):
        return None

    def set_status(self, text: str, level: str = "info") -> None:
        self.status_log.append((text, level))

    def toast(self, text: str, level: str = "info") -> None:
        self.toast_log.append((text, level))

    def show_in_panel(self, target: str, array, **kw) -> None:
        pass


@pytest.fixture()
def ctx():
    return _FakeContext()


# ---------------------------------------------------------------------------
# Construction with mock devices (default resolution path)
# ---------------------------------------------------------------------------

def test_panel_builds_with_mock_devices_by_default(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)

    assert panel.stage is not None
    assert panel.shutter is not None
    assert panel.led is not None
    assert panel.lbl_stage_status.text() == "linked"
    assert panel.lbl_shutter_status.text() == "closed"
    assert panel.lbl_led_status.text() == "off"
    assert panel.lbl_x.text().strip() == "+0.0000"
    assert panel.lbl_y.text().strip() == "+0.0000"
    assert panel.lbl_z.text().strip() == "+0.0000"


def test_step_combo_lists_all_presets(qapp, ctx):
    from ui3.panels.device_panel import (
        DEFAULT_STEP_MM, JOG_STEPS_MM, DevicePanel, _step_label,
    )

    panel = DevicePanel(ctx)
    items = [panel.cmb_step.itemText(i) for i in range(panel.cmb_step.count())]
    assert items == [_step_label(s) for s in JOG_STEPS_MM]
    assert panel.step_mm == pytest.approx(DEFAULT_STEP_MM)


# ---------------------------------------------------------------------------
# Jog — X 100um -> readout 0.100mm
# ---------------------------------------------------------------------------

def test_jog_x_100um_updates_readout(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    panel.cmb_step.setCurrentText("100 um")
    assert panel.step_mm == pytest.approx(0.100)

    panel.btn_xplus.click()

    x_um, y_um, z_um = panel.stage.position_um
    assert x_um == pytest.approx(100.0)
    assert y_um == pytest.approx(0.0)
    assert z_um == pytest.approx(0.0)
    assert panel.lbl_x.text().strip() == "+0.1000"


def test_jog_x_negative_direction(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    panel.cmb_step.setCurrentText("100 um")
    panel.btn_xminus.click()

    x_um, _, _ = panel.stage.position_um
    assert x_um == pytest.approx(-100.0)
    assert panel.lbl_x.text().strip() == "-0.1000"


def test_jog_y_and_z(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    panel.cmb_step.setCurrentText("1 mm")
    panel.btn_yplus.click()
    panel.btn_zplus.click()

    x_um, y_um, z_um = panel.stage.position_um
    assert x_um == pytest.approx(0.0)
    assert y_um == pytest.approx(1000.0)
    assert z_um == pytest.approx(1000.0)


def test_diagonal_jog_moves_both_axes(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    panel.cmb_step.setCurrentText("100 um")
    panel.btn_diag_pxpy.click()

    x_um, y_um, _ = panel.stage.position_um
    assert x_um == pytest.approx(100.0)
    assert y_um == pytest.approx(100.0)


def test_home_returns_to_zero(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    panel.cmb_step.setCurrentText("1 mm")
    panel.btn_xplus.click()
    panel.btn_yplus.click()
    assert panel.stage.position_um != (0.0, 0.0, 0.0)

    panel.btn_home.click()

    assert panel.stage.position_um == (0.0, 0.0, 0.0)
    assert panel.lbl_x.text().strip() == "+0.0000"
    assert any(lvl == "ok" for _, lvl in ctx.status_log)


# ---------------------------------------------------------------------------
# 2-axis stage tolerance — missing axis padded, not a crash.
# ---------------------------------------------------------------------------

class _TwoAxisStage:
    """Stand-in for a real 2-axis stage: position_um only returns
    (x, y). Exercises the jog() axis-padding path."""

    def __init__(self):
        self._x = 0.0
        self._y = 0.0
        self._connected = True

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    @property
    def name(self):
        return "two-axis-stage"

    @property
    def position_um(self):
        return (self._x, self._y)

    def move_to(self, x_um, y_um, z_um=0.0):
        self._x = x_um
        self._y = y_um

    def home(self):
        self._x = 0.0
        self._y = 0.0


def test_jog_against_two_axis_stage_pads_missing_z(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    stage = _TwoAxisStage()
    panel = DevicePanel(ctx, stage=stage)
    panel.cmb_step.setCurrentText("100 um")

    # Z jog against a stage with no z should not raise.
    panel.btn_zplus.click()
    panel.btn_xplus.click()

    assert stage.position_um[0] == pytest.approx(100.0)
    # z readout stays 0 (padded), no crash.
    assert panel.lbl_z.text().strip() == "+0.0000"


# ---------------------------------------------------------------------------
# Shutter / LED toggles
# ---------------------------------------------------------------------------

def test_shutter_toggle(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    assert panel.shutter.is_open is False
    assert panel.btn_shutter.text() == "Open"

    panel.btn_shutter.click()
    assert panel.shutter.is_open is True
    assert panel.btn_shutter.text() == "Close"
    assert panel.lbl_shutter_status.text() == "open"

    panel.btn_shutter.click()
    assert panel.shutter.is_open is False
    assert panel.btn_shutter.text() == "Open"
    assert panel.lbl_shutter_status.text() == "closed"


def test_led_toggle_and_intensity(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    assert panel.led.is_on is False

    panel.btn_led.click()
    assert panel.led.is_on is True
    assert panel.btn_led.text() == "Off"

    panel.sld_intensity.setValue(75)
    assert panel.led.intensity_percent == pytest.approx(75.0)
    assert "75" in panel.lbl_led_status.text()

    panel.btn_led.click()
    assert panel.led.is_on is False
    assert panel.lbl_led_status.text() == "off"


def test_intensity_slider_clamped_0_100(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    panel.sld_intensity.setValue(150)  # QSlider itself clamps to max=100
    assert panel.led.intensity_percent == pytest.approx(100.0)
    panel.sld_intensity.setValue(-10)  # clamps to min=0
    assert panel.led.intensity_percent == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Graceful degradation — missing devices disable, never crash.
# ---------------------------------------------------------------------------

def test_none_stage_disables_card_without_crash(qapp, ctx, monkeypatch):
    import ui3.panels.device_panel as dp

    real_try_make = dp._try_make
    monkeypatch.setattr(dp, "_try_make",
                         lambda name: None if name == "mock_stage" else
                         real_try_make(name))
    panel = dp.DevicePanel(ctx)

    assert panel.stage is None
    assert panel.lbl_stage_status.text() == "not configured"
    assert not panel.btn_xplus.isEnabled()
    assert not panel.btn_home.isEnabled()
    assert not panel.cmb_step.isEnabled()

    # Clicking a disabled-in-spirit jog action must not raise even if
    # invoked programmatically (defensive: Qt won't let a real click
    # reach a disabled button, but the handler itself should no-op).
    panel._jog("x", +1)
    panel._on_home_clicked()


def test_none_shutter_disables_card_without_crash(qapp, ctx, monkeypatch):
    import ui3.panels.device_panel as dp

    real_try_make = dp._try_make
    monkeypatch.setattr(dp, "_try_make",
                         lambda name: None if name == "mock_shutter" else
                         real_try_make(name))
    panel = dp.DevicePanel(ctx)

    assert panel.shutter is None
    assert panel.lbl_shutter_status.text() == "not configured"
    assert not panel.btn_shutter.isEnabled()
    panel._on_shutter_clicked()  # no-op, no raise


def test_none_led_disables_card_without_crash(qapp, ctx, monkeypatch):
    import ui3.panels.device_panel as dp

    real_try_make = dp._try_make
    monkeypatch.setattr(dp, "_try_make",
                         lambda name: None if name == "mock_led" else
                         real_try_make(name))
    panel = dp.DevicePanel(ctx)

    assert panel.led is None
    assert panel.lbl_led_status.text() == "not configured"
    assert not panel.btn_led.isEnabled()
    assert not panel.sld_intensity.isEnabled()
    panel._on_led_clicked()
    panel._on_intensity_changed(50)  # no-op, no raise


def test_all_devices_none_panel_still_builds(qapp, ctx, monkeypatch):
    import ui3.panels.device_panel as dp

    monkeypatch.setattr(dp, "_try_make", lambda name: None)
    panel = dp.DevicePanel(ctx)

    assert panel.stage is None
    assert panel.shutter is None
    assert panel.led is None
    assert "not configured" in panel.lbl_connection.text() or \
        panel.lbl_connection.text().count("—") == 3


# ---------------------------------------------------------------------------
# Polling tick — dirty-check re-render path exercised directly (no
# reliance on real wall-clock timer firing in CI).
# ---------------------------------------------------------------------------

def test_tick_updates_readout_after_external_stage_move(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    # Simulate an external actor moving the stage (e.g. an AI tool)
    # between polling ticks.
    panel.stage.move_to(250.0, 0.0, 0.0)
    assert panel.lbl_x.text().strip() == "+0.0000"  # not yet polled

    panel._on_tick()

    assert panel.lbl_x.text().strip() == "+0.2500"


def test_tick_is_noop_when_snapshot_unchanged(qapp, ctx):
    from ui3.panels.device_panel import DevicePanel

    panel = DevicePanel(ctx)
    panel._on_tick()
    snap_before = panel._last_snapshot
    panel._on_tick()
    assert panel._last_snapshot == snap_before
