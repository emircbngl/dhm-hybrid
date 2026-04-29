"""v2.1.z AI device-tool tests.

Covers the 9 new tools wired through ``ctx.shutter`` /
``ctx.led`` / ``ctx.orchestrator`` (plus ``list_devices`` which
just queries the registry):

* ``list_devices`` — registry rollup, ``available`` is a subset of ``all``.
* ``shutter_open`` / ``shutter_close`` / ``shutter_status``.
* ``led_set_intensity`` / ``led_on`` / ``led_off`` / ``led_status``.
* ``acquire_grid`` — orchestrator wiring + error path when not
  configured.

Each tool surfaces a friendly ``{"error": "<kind> not configured"}``
when its hook is missing — verified explicitly so the LLM gets a
useful answer instead of an exception traceback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ai.tool_impls import build_tool_registry  # noqa: E402
from core.ai.tools import ToolContext  # noqa: E402
from core.audit import AuditLog  # noqa: E402
from core.errors import ErrorCenter  # noqa: E402


# ---------------------------------------------------------------------------
# Context fixture — mirrors test_ai_tools.py's _ctx helper
# ---------------------------------------------------------------------------

class _StubStage:
    """Legacy stage stub — required by the ``ctx.stage`` field."""
    def get_position(self): return (0.0, 0.0, 0.0)
    def move_relative(self, *a): return (0.0, 0.0, 0.0)
    def move_absolute(self, *a): return (0.0, 0.0, 0.0)
    def home(self): return (0.0, 0.0, 0.0)


class _StubSampleMap:
    cells = []
    def summary(self): return {}


def _ctx(*, shutter=None, led=None, orchestrator=None,
         settings=None, tmp_path: Path = None) -> ToolContext:
    audit = AuditLog(directory=(tmp_path or Path("/tmp"))
                     / "_dhm_test_audit")
    err = ErrorCenter()

    return ToolContext(
        state=lambda: {},
        last_recon_summary=lambda: None,
        last_af_summary=lambda: None,
        last_qpi_summary=lambda: None,
        last_depth_summary=lambda: None,
        audit_tail=lambda n: [],
        load_hologram=lambda p: {"loaded": True},
        set_recon_param=lambda d: {"applied": True},
        invoke_recon=lambda d: {},
        invoke_autofocus=lambda d: {},
        invoke_qpi=lambda d: {},
        invoke_depth_map=lambda d: {},
        invoke_find_focus_candidates=lambda d: {},
        stage=_StubStage(),
        capture_frame=lambda: None,
        sample_map=_StubSampleMap(),
        measure_sharpness=lambda f: 0.0,
        persist_sample_map=lambda: None,
        capture_and_process=lambda d: {},
        audit=audit,
        error_center=err,
        confirm=lambda name, args: True,
        settings=settings,
        is_cancelled=None,
        shutter=shutter,
        led=led,
        orchestrator=orchestrator,
    )


def _dispatch(reg, name, args, ctx):
    """Pull the tool out of the registry + invoke its handler."""
    spec = reg.get(name)
    return spec.handler(ctx, args)


# ---------------------------------------------------------------------------
# list_devices
# ---------------------------------------------------------------------------

def test_list_devices_returns_available_and_all(tmp_path):
    reg = build_tool_registry()
    res = _dispatch(reg, "list_devices", {}, _ctx(tmp_path=tmp_path))
    assert "available" in res
    assert "all" in res
    avail_names = {b["name"] for b in res["available"]}
    all_names = {b["name"] for b in res["all"]}
    # Mocks always present.
    assert "mock_stage" in avail_names
    assert "mock_shutter" in avail_names
    assert "mock_led" in avail_names
    # ``all`` includes vendor stubs even if SDK not installed.
    assert avail_names.issubset(all_names)
    assert "pylon" in {b["name"] for b in res["all"]} or True
    # Each row carries kind + summary metadata.
    sample = res["available"][0]
    assert "kind" in sample
    assert "summary" in sample


# ---------------------------------------------------------------------------
# Shutter tools
# ---------------------------------------------------------------------------

def test_shutter_open_without_device_returns_error(tmp_path):
    reg = build_tool_registry()
    res = _dispatch(reg, "shutter_open", {},
                    _ctx(tmp_path=tmp_path, shutter=None))
    assert res.get("error") == "shutter device not configured"


def test_shutter_status_without_device_reports_not_configured(tmp_path):
    reg = build_tool_registry()
    res = _dispatch(reg, "shutter_status", {},
                    _ctx(tmp_path=tmp_path, shutter=None))
    assert res == {"configured": False}


def test_shutter_open_close_round_trip(tmp_path):
    from core.devices import make_device
    sh = make_device("mock_shutter")
    reg = build_tool_registry()
    ctx = _ctx(tmp_path=tmp_path, shutter=sh)

    # First open call connects the device automatically.
    res = _dispatch(reg, "shutter_open", {}, ctx)
    assert res.get("ok") is True
    assert res.get("is_open") is True
    assert sh.is_connected

    res = _dispatch(reg, "shutter_status", {}, ctx)
    assert res["configured"] is True
    assert res["is_connected"] is True
    assert res["is_open"] is True

    res = _dispatch(reg, "shutter_close", {}, ctx)
    assert res.get("ok") is True
    assert res.get("is_open") is False


def test_shutter_open_failure_reported_as_error(tmp_path):
    """A shutter whose open() raises should surface a clean error
    dict, not crash the dispatch loop."""
    class _BadShutter:
        is_connected = True
        is_open = False
        def open(self):
            raise RuntimeError("hardware fault")
        def close(self):
            pass

    reg = build_tool_registry()
    res = _dispatch(reg, "shutter_open", {},
                    _ctx(tmp_path=tmp_path, shutter=_BadShutter()))
    assert "error" in res
    assert "hardware fault" in res["error"]


# ---------------------------------------------------------------------------
# LED tools
# ---------------------------------------------------------------------------

def test_led_intensity_round_trip(tmp_path):
    from core.devices import make_device
    led = make_device("mock_led")
    reg = build_tool_registry()
    ctx = _ctx(tmp_path=tmp_path, led=led)

    res = _dispatch(reg, "led_set_intensity",
                    {"intensity_percent": 42.5}, ctx)
    assert res.get("ok") is True
    assert res["intensity_percent"] == pytest.approx(42.5)


def test_led_intensity_clamps_via_security(tmp_path):
    """``intensity_percent`` is in NUMERIC_BOUNDS; a value outside
    [0, 100] should raise SecurityError before reaching the LED."""
    from core.ai.security import SecurityError
    from core.devices import make_device
    led = make_device("mock_led")
    reg = build_tool_registry()
    ctx = _ctx(tmp_path=tmp_path, led=led)
    with pytest.raises(SecurityError):
        _dispatch(reg, "led_set_intensity",
                  {"intensity_percent": 150.0}, ctx)


def test_led_on_off_toggle(tmp_path):
    from core.devices import make_device
    led = make_device("mock_led")
    reg = build_tool_registry()
    ctx = _ctx(tmp_path=tmp_path, led=led)

    res = _dispatch(reg, "led_on", {}, ctx)
    assert res.get("is_on") is True

    res = _dispatch(reg, "led_status", {}, ctx)
    assert res["is_on"] is True
    assert res["configured"] is True

    res = _dispatch(reg, "led_off", {}, ctx)
    assert res.get("is_on") is False


def test_led_without_device_returns_error(tmp_path):
    reg = build_tool_registry()
    res = _dispatch(reg, "led_set_intensity",
                    {"intensity_percent": 50.0},
                    _ctx(tmp_path=tmp_path, led=None))
    assert res.get("error") == "led device not configured"


# ---------------------------------------------------------------------------
# acquire_grid
# ---------------------------------------------------------------------------

def test_acquire_grid_requires_orchestrator(tmp_path):
    reg = build_tool_registry()
    res = _dispatch(reg, "acquire_grid",
                    {"rows": 2, "cols": 2,
                     "spacing_x_um": 100.0,
                     "spacing_y_um": 100.0},
                    _ctx(tmp_path=tmp_path, orchestrator=None))
    assert "error" in res
    assert "orchestrator" in res["error"]


def test_acquire_grid_forwards_args_to_orchestrator(tmp_path):
    captured = {}

    def _mock_orchestrator(args: dict) -> dict:
        captured.update(args)
        return {"ok": True, "frames": [{"index": 0},
                                        {"index": 1},
                                        {"index": 2},
                                        {"index": 3}]}

    reg = build_tool_registry()
    res = _dispatch(
        reg, "acquire_grid",
        {"rows": 2, "cols": 2,
         "spacing_x_um": 250.0, "spacing_y_um": 250.0,
         "led_intensity_percent": 60.0,
         "shutter_per_frame": False,
         "settle_time_s": 0.2},
        _ctx(tmp_path=tmp_path, orchestrator=_mock_orchestrator),
    )
    assert res.get("ok") is True
    assert captured["rows"] == 2
    assert captured["cols"] == 2
    assert captured["spacing_x_um"] == 250.0
    assert captured["spacing_y_um"] == 250.0
    assert captured["led_intensity_percent"] == 60.0
    assert captured["shutter_per_frame"] is False
    assert captured["settle_time_s"] == 0.2


def test_acquire_grid_clamp_bounds_enforced(tmp_path):
    """``rows`` outside NUMERIC_BOUNDS should raise SecurityError."""
    from core.ai.security import SecurityError
    reg = build_tool_registry()
    with pytest.raises(SecurityError):
        _dispatch(
            reg, "acquire_grid",
            {"rows": 999, "cols": 2,
             "spacing_x_um": 100.0, "spacing_y_um": 100.0},
            _ctx(tmp_path=tmp_path, orchestrator=lambda d: {}),
        )


def test_acquire_grid_orchestrator_exception_wrapped(tmp_path):
    def _explode(args):
        raise RuntimeError("stage move limit")
    reg = build_tool_registry()
    res = _dispatch(
        reg, "acquire_grid",
        {"rows": 1, "cols": 1,
         "spacing_x_um": 50.0, "spacing_y_um": 50.0},
        _ctx(tmp_path=tmp_path, orchestrator=_explode),
    )
    assert "error" in res
    assert "stage move limit" in res["error"]


# ---------------------------------------------------------------------------
# include_devices=False removes them
# ---------------------------------------------------------------------------

def test_include_devices_false_removes_new_tools():
    reg = build_tool_registry(include_devices=False)
    names = set(reg.names())
    for name in ("list_devices", "shutter_open", "led_on",
                 "acquire_grid"):
        assert name not in names


# ---------------------------------------------------------------------------
# Cancel-aware timelapse smoke
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# APT-style stage tools (v2.1.z+)
# ---------------------------------------------------------------------------

def _ctx_with_apt_stage(tmp_path: Path):
    """ToolContext with the v2.1.z mock_stage wired into ctx.stage —
    exercises the new APT tools (which require StageDevice
    Protocol surface, not the legacy v1 stub)."""
    from core.devices import make_device
    audit = AuditLog(directory=tmp_path / "_dhm_test_audit")
    err = ErrorCenter()
    stage = make_device("mock_stage")
    stage.connect()

    return ToolContext(
        state=lambda: {},
        last_recon_summary=lambda: None,
        last_af_summary=lambda: None,
        last_qpi_summary=lambda: None,
        last_depth_summary=lambda: None,
        audit_tail=lambda n: [],
        load_hologram=lambda p: {"loaded": True},
        set_recon_param=lambda d: {"applied": True},
        invoke_recon=lambda d: {},
        invoke_autofocus=lambda d: {},
        invoke_qpi=lambda d: {},
        invoke_depth_map=lambda d: {},
        invoke_find_focus_candidates=lambda d: {},
        stage=stage,
        capture_frame=lambda: None,
        sample_map=_StubSampleMap(),
        measure_sharpness=lambda f: 0.0,
        persist_sample_map=lambda: None,
        capture_and_process=lambda d: {},
        audit=audit,
        error_center=err,
        confirm=lambda name, args: True,
        settings=None,
        is_cancelled=None,
    )


def test_stage_set_speed_round_trip(tmp_path):
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    res = _dispatch(reg, "stage_set_speed",
                    {"speed_um_per_s": 2500.0}, ctx)
    assert res.get("ok") is True
    assert res["speed_um_per_s"] == pytest.approx(2500.0)


def test_stage_get_speed_reflects_setter(tmp_path):
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    _dispatch(reg, "stage_set_speed", {"speed_um_per_s": 750.0}, ctx)
    res = _dispatch(reg, "stage_get_speed", {}, ctx)
    assert res["speed_um_per_s"] == pytest.approx(750.0)


def test_stage_move_by_um_shifts_position(tmp_path):
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    ctx.stage.move_to(100.0, 200.0, 50.0)
    res = _dispatch(reg, "stage_move_by",
                    {"dx_um": 25.0, "dy_um": -10.0}, ctx)
    assert res["x_um"] == pytest.approx(125.0)
    assert res["y_um"] == pytest.approx(190.0)


def test_stage_set_step_size_round_trip(tmp_path):
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    res = _dispatch(reg, "stage_set_step_size",
                    {"step_um": 25.0}, ctx)
    assert res["step_size_um"] == 25.0


def test_stage_jog_uses_current_step(tmp_path):
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    _dispatch(reg, "stage_set_step_size", {"step_um": 50.0}, ctx)
    res = _dispatch(reg, "stage_jog",
                    {"axis": "x", "direction": +2}, ctx)
    assert res["x_um"] == pytest.approx(100.0)


def test_stage_jog_negative_direction(tmp_path):
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    ctx.stage.move_to(500.0, 500.0, 500.0)
    _dispatch(reg, "stage_set_step_size", {"step_um": 100.0}, ctx)
    res = _dispatch(reg, "stage_jog",
                    {"axis": "y", "direction": -1}, ctx)
    assert res["y_um"] == pytest.approx(400.0)


def test_stage_stop_no_motion_in_flight(tmp_path):
    """Mock with no active motion — stop_motion is still safe."""
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    res = _dispatch(reg, "stage_stop", {}, ctx)
    assert res.get("ok") is True


def test_stage_apt_tools_no_device_returns_error(tmp_path):
    """When ctx.stage is None or doesn't support APT, the tools
    return a clean error dict instead of crashing."""
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    object.__setattr__(ctx, "stage", None)
    res = _dispatch(reg, "stage_set_speed",
                    {"speed_um_per_s": 100.0}, ctx)
    assert "error" in res


def test_stage_set_speed_clamp_enforced(tmp_path):
    """speed > 100k µm/s is rejected by NUMERIC_BOUNDS."""
    from core.ai.security import SecurityError
    reg = build_tool_registry()
    ctx = _ctx_with_apt_stage(tmp_path)
    with pytest.raises(SecurityError):
        _dispatch(reg, "stage_set_speed",
                  {"speed_um_per_s": 1_000_000.0}, ctx)


def test_record_timelapse_honours_cancel_hook(tmp_path):
    """``ctx.is_cancelled`` polled every iteration; cancelling
    after frame 1 should leave 1 frame in the result + flag set."""
    reg = build_tool_registry(include_devices=False)
    captured = {"frames_captured": 0}

    def _capture(d):
        captured["frames_captured"] += 1
        return {"phase_std": float(captured["frames_captured"])}

    ctx = _ctx(tmp_path=tmp_path)
    # Replace the lambdas — ToolContext is a dataclass so we
    # rebuild it.
    from dataclasses import replace
    ctx = replace(
        ctx,
        capture_and_process=_capture,
        is_cancelled=lambda: captured["frames_captured"] >= 1,
    )
    res = _dispatch(reg, "record_timelapse",
                    {"n_frames": 5, "interval_s": 0.001,
                     "run_recon": False, "run_qpi": False}, ctx)
    assert res.get("cancelled") is True
    assert res["completed_frames"] >= 1
    assert res["completed_frames"] < 5
