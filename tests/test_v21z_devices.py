"""Lab-device control tests (v2.1.z).

Covers:

* :mod:`core.devices` registry (all/available/by_kind, factory).
* Mock stage / shutter / LED behaviours + Protocol shape.
* Generic serial backend SDK gating (skip when pyserial missing).
* :mod:`core.devices.orchestrator` — plan-based multi-position
  acquisition against mock devices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import devices  # noqa: E402
from core.devices import (  # noqa: E402
    DeviceBackendInfo, DeviceKind,
    all_backends, available_backends, backends_by_kind, make_device,
)
from core.devices._base import (  # noqa: E402
    Device, LEDDevice, ShutterDevice, StageDevice,
)
from core.devices.orchestrator import (  # noqa: E402
    AcquisitionPlan, FrameRecord, PlanResult, StagePosition, run_plan,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_advertises_mocks_and_serial():
    names = {b.name for b in all_backends()}
    assert "mock_stage" in names
    assert "mock_shutter" in names
    assert "mock_led" in names
    assert "serial_generic" in names


def test_registry_groups_by_kind():
    stages = backends_by_kind(DeviceKind.STAGE)
    shutters = backends_by_kind(DeviceKind.SHUTTER)
    leds = backends_by_kind(DeviceKind.LED)
    assert "mock_stage" in {b.name for b in stages}
    assert "mock_shutter" in {b.name for b in shutters}
    assert "mock_led" in {b.name for b in leds}


def test_available_excludes_serial_when_pyserial_missing():
    """The dev venv doesn't ship pyserial — serial_generic should
    fall out of ``available_backends`` while staying registered."""
    available = {b.name for b in available_backends()}
    try:
        import serial  # noqa: F401
        has_pyserial = True
    except ImportError:
        has_pyserial = False
    if has_pyserial:
        assert "serial_generic" in available
    else:
        assert "serial_generic" not in available
        assert "serial_generic" in {b.name for b in all_backends()}


def test_make_device_unknown_raises_valueerror():
    with pytest.raises(ValueError, match="unknown device"):
        make_device("not_a_device")


def test_make_device_serial_without_sdk_raises_runtime():
    try:
        import serial  # noqa: F401
        pytest.skip("pyserial available — RuntimeError won't fire")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="requires SDK"):
        make_device("serial_generic")


def test_backend_info_carries_capabilities():
    stage_info = next(b for b in all_backends()
                      if b.name == "mock_stage")
    assert stage_info.kind is DeviceKind.STAGE
    assert "limits_um" in stage_info.capabilities


# ---------------------------------------------------------------------------
# Mock stage
# ---------------------------------------------------------------------------

def test_mock_stage_implements_stage_protocol():
    s = make_device("mock_stage")
    assert isinstance(s, StageDevice)


def test_mock_stage_connect_and_position():
    s = make_device("mock_stage")
    s.connect()
    assert s.is_connected
    assert s.position_um == (0.0, 0.0, 0.0)
    s.move_to(100, 200, 50)
    assert s.position_um == (100.0, 200.0, 50.0)


def test_mock_stage_home_resets_to_origin():
    s = make_device("mock_stage")
    s.connect()
    s.move_to(500, 500, 500)
    s.home()
    assert s.position_um == (0.0, 0.0, 0.0)


def test_mock_stage_rejects_out_of_range():
    s = make_device("mock_stage", limits_um=(-100, 100))
    s.connect()
    with pytest.raises(ValueError, match="outside"):
        s.move_to(200, 0, 0)


def test_mock_stage_move_before_connect_raises():
    s = make_device("mock_stage")
    with pytest.raises(RuntimeError, match="before connect"):
        s.move_to(0, 0, 0)


def test_mock_stage_disconnect_resets_state():
    s = make_device("mock_stage")
    s.connect()
    s.move_to(10, 20, 30)
    s.disconnect()
    assert not s.is_connected


# ---------------------------------------------------------------------------
# Mock shutter
# ---------------------------------------------------------------------------

def test_mock_shutter_implements_shutter_protocol():
    s = make_device("mock_shutter")
    assert isinstance(s, ShutterDevice)


def test_mock_shutter_open_close_cycle():
    s = make_device("mock_shutter")
    s.connect()
    assert not s.is_open
    s.open()
    assert s.is_open
    s.close()
    assert not s.is_open


def test_mock_shutter_open_before_connect_raises():
    s = make_device("mock_shutter")
    with pytest.raises(RuntimeError, match="before connect"):
        s.open()


# ---------------------------------------------------------------------------
# Mock LED
# ---------------------------------------------------------------------------

def test_mock_led_implements_led_protocol():
    led = make_device("mock_led")
    assert isinstance(led, LEDDevice)


def test_mock_led_intensity_round_trip():
    led = make_device("mock_led")
    led.connect()
    led.set_intensity(42.0)
    assert led.intensity_percent == pytest.approx(42.0)


def test_mock_led_intensity_clamped_to_bounds():
    """Slider drag past 100 % shouldn't raise — clamp instead."""
    led = make_device("mock_led")
    led.connect()
    led.set_intensity(150.0)
    assert led.intensity_percent == 100.0
    led.set_intensity(-5.0)
    assert led.intensity_percent == 0.0


def test_mock_led_on_off_toggle():
    led = make_device("mock_led")
    led.connect()
    assert not led.is_on
    led.on()
    assert led.is_on
    led.off()
    assert not led.is_on


# ---------------------------------------------------------------------------
# Orchestrator — multi-position acquisition
# ---------------------------------------------------------------------------

class _StubCamera:
    """Minimal CameraSource stand-in for orchestrator tests."""
    def __init__(self, *, fail_grab_at: list = None):
        self._fail = set(fail_grab_at or [])
        self._n = 0
        self.grabs = []

    @property
    def size(self):
        return (16, 16)

    @property
    def fps(self):
        return 30.0

    def start(self):
        pass

    def stop(self):
        pass

    def grab(self):
        i = self._n
        self._n += 1
        if i in self._fail:
            raise RuntimeError(f"sim grab failure at {i}")
        f = np.full((16, 16), float(i + 1), dtype=np.float32)
        self.grabs.append(f)
        return f


def test_orchestrator_runs_one_grab_when_no_positions():
    """Empty positions list = single capture at current position."""
    cam = _StubCamera()
    res = run_plan(AcquisitionPlan(), camera=cam, sleep=lambda s: None)
    assert len(res.frames) == 1
    assert res.frames[0].frame is not None
    assert res.frames[0].position.label == "current"


def test_orchestrator_visits_each_position_in_order():
    cam = _StubCamera()
    stage = make_device("mock_stage")
    plan = AcquisitionPlan(
        positions=[
            StagePosition(0, 0, label="A"),
            StagePosition(100, 0, label="B"),
            StagePosition(200, 0, label="C"),
        ],
    )
    res = run_plan(plan, camera=cam, stage=stage,
                   sleep=lambda s: None)
    assert len(res.frames) == 3
    assert [r.position.label for r in res.frames] == ["A", "B", "C"]
    # Stage ended at the last waypoint.
    assert stage.position_um[0] == 200.0


def test_orchestrator_per_frame_shutter_cycles():
    """``shutter_per_frame=True`` → shutter open/close around
    every grab."""
    cam = _StubCamera()
    shutter = make_device("mock_shutter")
    open_count = {"n": 0}
    close_count = {"n": 0}
    real_open = shutter.open
    real_close = shutter.close

    def _wrap_open():
        open_count["n"] += 1; real_open()

    def _wrap_close():
        close_count["n"] += 1; real_close()

    shutter.open = _wrap_open
    shutter.close = _wrap_close

    plan = AcquisitionPlan(
        positions=[StagePosition(i * 10, 0) for i in range(3)],
        shutter_per_frame=True,
    )
    run_plan(plan, camera=cam,
             stage=make_device("mock_stage"),
             shutter=shutter, sleep=lambda s: None)
    # 3 frames → 3 opens (per frame) + 3 closes per frame + 1
    # final cleanup close. Lab cares the count is sane, not exact.
    assert open_count["n"] == 3
    assert close_count["n"] >= 3


def test_orchestrator_continuous_shutter_opens_once():
    """``shutter_per_frame=False`` → one open at start,
    one close at end."""
    cam = _StubCamera()
    shutter = make_device("mock_shutter")
    open_count = {"n": 0}
    real_open = shutter.open
    shutter.open = lambda: (open_count.update(n=open_count["n"] + 1),
                            real_open())[1]
    plan = AcquisitionPlan(
        positions=[StagePosition(i * 10, 0) for i in range(4)],
        shutter_per_frame=False,
    )
    run_plan(plan, camera=cam,
             stage=make_device("mock_stage"),
             shutter=shutter, sleep=lambda s: None)
    assert open_count["n"] == 1


def test_orchestrator_led_intensity_applied_once():
    cam = _StubCamera()
    led = make_device("mock_led")
    plan = AcquisitionPlan(
        positions=[StagePosition(0, 0)],
        led_intensity_percent=37.0,
    )
    run_plan(plan, camera=cam, led=led, sleep=lambda s: None)
    # LED ended off (cleanup), but intensity stayed at the
    # configured value.
    assert led.intensity_percent == pytest.approx(37.0)


def test_orchestrator_grab_failure_marked_on_frame():
    cam = _StubCamera(fail_grab_at=[1])
    plan = AcquisitionPlan(
        positions=[StagePosition(i * 10, 0) for i in range(3)],
    )
    res = run_plan(plan, camera=cam,
                   stage=make_device("mock_stage"),
                   sleep=lambda s: None)
    assert res.frames[0].frame is not None
    assert res.frames[1].frame is None
    assert "grab" in (res.frames[1].error or "")
    assert res.frames[2].frame is not None


def test_orchestrator_stage_move_failure_marked_on_frame():
    """Out-of-limit move logs the error per-frame and continues."""
    cam = _StubCamera()
    stage = make_device("mock_stage", limits_um=(-50, 50))
    plan = AcquisitionPlan(
        positions=[
            StagePosition(0, 0),
            StagePosition(100, 0),    # out of range
            StagePosition(20, 20),
        ],
    )
    res = run_plan(plan, camera=cam, stage=stage,
                   sleep=lambda s: None)
    assert res.frames[1].error and "stage_move" in res.frames[1].error
    assert res.frames[1].frame is None
    assert res.frames[2].frame is not None


def test_orchestrator_cancel_check_aborts_remaining():
    cam = _StubCamera()
    cancel = {"flag": False}

    def _on_frame(rec):
        if rec.index == 1:
            cancel["flag"] = True

    plan = AcquisitionPlan(
        positions=[StagePosition(i * 10, 0) for i in range(5)],
    )
    res = run_plan(
        plan, camera=cam,
        stage=make_device("mock_stage"),
        cancel_check=lambda: cancel["flag"],
        on_frame=_on_frame,
        sleep=lambda s: None,
    )
    # 2 frames went through (index 0 + 1), then cancel triggered.
    assert res.cancelled
    assert len(res.frames) == 2


def test_orchestrator_on_frame_callback_fires_in_order():
    cam = _StubCamera()
    seen = []
    plan = AcquisitionPlan(
        positions=[StagePosition(i * 10, 0, label=str(i))
                   for i in range(3)],
    )
    run_plan(plan, camera=cam,
             stage=make_device("mock_stage"),
             on_frame=lambda r: seen.append(r.position.label),
             sleep=lambda s: None)
    assert seen == ["0", "1", "2"]


def test_orchestrator_returns_plan_result_with_elapsed():
    cam = _StubCamera()
    plan = AcquisitionPlan(positions=[StagePosition(0, 0)])
    res = run_plan(plan, camera=cam, sleep=lambda s: None)
    assert isinstance(res, PlanResult)
    assert isinstance(res.frames[0], FrameRecord)
    assert res.elapsed_s >= 0.0
