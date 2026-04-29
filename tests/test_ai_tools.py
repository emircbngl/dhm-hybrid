"""ToolRegistry: schema export, dispatch, validation.

These tests cover the registry's public contract using a minimal
synthetic ``ToolContext`` — they assert that:

* All 14 tools are registered.
* ``schemas()`` returns OpenAI tools-array entries.
* Unknown tool calls return a friendly error, not a raise.
* JSON-schema validation rejects bad args.
* Numeric clamps inside handlers reject out-of-range values.
* The audit hook fires on both start and done.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from core.ai.protocol import ToolCall
from core.ai.tool_impls import build_tool_registry
from core.ai.tools import ToolContext


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

@dataclass
class _Audit:
    records: list = field(default_factory=list)

    def record(self, action, params, result_summary=None):
        self.records.append((action, params, result_summary))


class _ErrorCenter:
    def __init__(self):
        self.events = []

    def error(self, **kwargs):
        self.events.append(kwargs)


class _MockSettings:
    restrict_to_home = False  # tests don't poke real paths
    audit_redact_for_llm = True


class _Snapshot:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.loaded_path = kw.get("loaded_path")
        self.loaded_shape = kw.get("loaded_shape")
        self.loaded_dtype = kw.get("loaded_dtype")
        self.recon_params = kw.get("recon_params", {})
        self.autofocus_params = kw.get("autofocus_params", {})
        self.qpi_params = kw.get("qpi_params", {})
        self.last_recon_summary = kw.get("last_recon_summary")
        self.last_af_summary = kw.get("last_af_summary")
        self.last_qpi_summary = kw.get("last_qpi_summary")
        self.last_depth_summary = kw.get("last_depth_summary")
        self.stage_position_mm = kw.get("stage_position_mm")
        self.audit_tail = kw.get("audit_tail", ())

    def as_dict(self):
        return {
            "loaded_path": self.loaded_path,
            "loaded_shape": self.loaded_shape,
            "loaded_dtype": self.loaded_dtype,
            "recon_params": self.recon_params,
            "autofocus_params": self.autofocus_params,
            "qpi_params": self.qpi_params,
            "last_recon_summary": self.last_recon_summary,
            "last_af_summary": self.last_af_summary,
            "last_qpi_summary": self.last_qpi_summary,
            "last_depth_summary": self.last_depth_summary,
            "stage_position_mm": self.stage_position_mm,
            "audit_tail": list(self.audit_tail),
        }


def _make_ctx(**overrides) -> ToolContext:
    import numpy as np
    from core.sample_map import SampleMap
    from core.stage import MockStage

    state = overrides.get("state_snapshot", _Snapshot())
    audit = overrides.get("audit", _Audit())
    err = overrides.get("error_center", _ErrorCenter())
    settings = overrides.get("settings", _MockSettings())
    sample_map = overrides.get("sample_map", SampleMap())

    invoked: dict = overrides.get("invoked", {})

    def record_call(name):
        def _fn(args):
            invoked.setdefault(name, []).append(args)
            return {"submitted": True, "echo": dict(args)}
        return _fn

    def load_holo(path):
        invoked.setdefault("load_hologram", []).append(path)
        return {"path": path, "shape": [256, 256], "dtype": "float32"}

    def set_recon(args):
        invoked.setdefault("set_recon_param", []).append(args)
        return {"updated": dict(args), "current": dict(args)}

    def capture():
        return overrides.get("frame", np.zeros((32, 32), dtype=np.float32))

    def cap_proc(args):
        invoked.setdefault("capture_and_process", []).append(args)
        return overrides.get("capture_summary", {"phase_std": 0.5})

    return ToolContext(
        state=lambda: state,
        last_recon_summary=lambda: state.last_recon_summary,
        last_af_summary=lambda: state.last_af_summary,
        last_qpi_summary=lambda: state.last_qpi_summary,
        last_depth_summary=lambda: state.last_depth_summary,
        audit_tail=lambda lim: list(state.audit_tail)[-lim:],
        load_hologram=load_holo,
        set_recon_param=set_recon,
        invoke_recon=record_call("invoke_recon"),
        invoke_autofocus=record_call("invoke_autofocus"),
        invoke_qpi=record_call("invoke_qpi"),
        invoke_depth_map=record_call("invoke_depth_map"),
        invoke_find_focus_candidates=record_call("invoke_find_focus_candidates"),
        stage=MockStage(),
        capture_frame=capture,
        sample_map=sample_map,
        measure_sharpness=lambda f: float(getattr(f, "var", lambda: 1.0)()),
        persist_sample_map=lambda: "/tmp/test_sample_map.json",
        capture_and_process=cap_proc,
        audit=audit,
        error_center=err,
        confirm=lambda name, args: True,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_registry_has_all_19_tools():
    """Canonical 19 (10 pipeline + 4 stage + 5 sprint-2). v2.1.z
    added device tools behind ``include_devices=True``; this test
    pins the legacy surface."""
    r = build_tool_registry(include_devices=False)
    expected = {
        "load_hologram", "set_recon_param", "run_reconstruction",
        "run_autofocus", "find_focus_candidates", "run_qpi",
        "compute_depth_map", "get_state", "get_last_result",
        "get_audit_tail", "stage_get_position", "stage_move_relative",
        "stage_move_absolute", "stage_home",
        # Sprint 2 — focus search, mapping, time-lapse:
        "stage_focus_search", "map_sample_grid", "list_mapped_cells",
        "goto_cell", "record_timelapse",
    }
    assert set(r.names()) == expected


def test_registry_default_includes_v21z_device_tools():
    """v2.1.z added shutter / led / list_devices / acquire_grid.
    Default factory call surfaces them; ``include_devices=False``
    pulls them out cleanly."""
    r = build_tool_registry()
    new_tools = {
        "list_devices", "shutter_open", "shutter_close",
        "shutter_status", "led_set_intensity", "led_on", "led_off",
        "led_status", "acquire_grid",
    }
    assert new_tools.issubset(set(r.names()))
    # Canonical 19 still present.
    assert "run_reconstruction" in r.names()


def test_schemas_are_openai_tools_array_shape():
    r = build_tool_registry(include_devices=False)
    schemas = r.schemas()
    assert len(schemas) == 19
    for s in schemas:
        assert s["type"] == "function"
        assert "function" in s
        fn = s["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_dispatch_unknown_tool_returns_error():
    r = build_tool_registry()
    ctx = _make_ctx()
    call = ToolCall(id="x", name="not_a_tool", arguments_json="{}")
    out = r.dispatch(ctx, call)
    assert out.ok is False
    assert "unknown tool" in out.payload["error"]


def test_dispatch_invalid_json_returns_error():
    r = build_tool_registry()
    ctx = _make_ctx()
    call = ToolCall(id="x", name="get_state", arguments_json="not-json")
    out = r.dispatch(ctx, call)
    assert out.ok is False
    assert "invalid JSON arguments" in out.payload["error"]


def test_dispatch_get_state_returns_snapshot_dict():
    r = build_tool_registry()
    snap = _Snapshot(loaded_path="/tmp/x.tif", loaded_shape=(128, 128))
    ctx = _make_ctx(state_snapshot=snap)
    call = ToolCall(id="x", name="get_state", arguments_json="{}")
    out = r.dispatch(ctx, call)
    assert out.ok is True
    assert out.payload["loaded_path"] == "/tmp/x.tif"
    assert out.payload["loaded_shape"] == [128, 128]


def test_dispatch_run_autofocus_clamps_z_range_swap():
    """If LLM swaps z_min/z_max, the handler returns an error result."""
    r = build_tool_registry()
    ctx = _make_ctx()
    call = ToolCall(
        id="x", name="run_autofocus",
        arguments_json=json.dumps({"z_min_mm": 50.0, "z_max_mm": 30.0}),
    )
    out = r.dispatch(ctx, call)
    assert out.ok is False
    assert "must exceed" in out.payload["error"]


def test_dispatch_run_autofocus_passes_through_to_invoke():
    r = build_tool_registry()
    invoked: dict = {}
    ctx = _make_ctx(invoked=invoked)
    call = ToolCall(
        id="x", name="run_autofocus",
        arguments_json=json.dumps({
            "z_min_mm": -10.0, "z_max_mm": 10.0,
            "metric": "PHASE_VARIANCE", "n_steps": 40,
        }),
    )
    out = r.dispatch(ctx, call)
    assert out.ok is True
    args = invoked["invoke_autofocus"][0]
    assert args["z_min_mm"] == -10.0
    assert args["metric"] == "PHASE_VARIANCE"


def test_dispatch_set_recon_param_rejects_out_of_range_z():
    r = build_tool_registry()
    ctx = _make_ctx()
    call = ToolCall(
        id="x", name="set_recon_param",
        arguments_json=json.dumps({"z_mm": 9999.0}),
    )
    out = r.dispatch(ctx, call)
    # JSON-schema enforces maximum=200, so this is a schema error
    # before clamp ever runs. Either path returns ``ok=False``.
    assert out.ok is False


def test_dispatch_records_audit_start_and_done():
    r = build_tool_registry()
    audit = _Audit()
    ctx = _make_ctx(audit=audit)
    call = ToolCall(id="x", name="get_state", arguments_json="{}")
    r.dispatch(ctx, call)
    actions = [a for (a, _, _) in audit.records]
    assert "ai.tool.get_state.start" in actions
    assert "ai.tool.get_state.done" in actions


def test_dispatch_stage_move_relative_updates_position():
    r = build_tool_registry()
    ctx = _make_ctx()
    call = ToolCall(
        id="x", name="stage_move_relative",
        arguments_json=json.dumps({"dx_mm": 1.5}),
    )
    out = r.dispatch(ctx, call)
    assert out.ok is True
    assert out.payload["x_mm"] == pytest.approx(1.5)


def test_dispatch_stage_move_absolute_clamps_in_security_layer():
    r = build_tool_registry()
    ctx = _make_ctx()
    call = ToolCall(
        id="x", name="stage_move_absolute",
        arguments_json=json.dumps({"x_mm": 9999.0, "y_mm": 0.0, "z_mm": 0.0}),
    )
    out = r.dispatch(ctx, call)
    assert out.ok is False


def test_dispatch_get_last_result_unknown_stage_errors():
    r = build_tool_registry()
    ctx = _make_ctx()
    call = ToolCall(
        id="x", name="get_last_result",
        arguments_json=json.dumps({"stage": "nope"}),
    )
    out = r.dispatch(ctx, call)
    assert out.ok is False


def test_dispatch_get_last_result_returns_summary_when_present():
    r = build_tool_registry()
    snap = _Snapshot(last_recon_summary={"shape": [64, 64], "phase_std": 0.5})
    ctx = _make_ctx(state_snapshot=snap)
    call = ToolCall(
        id="x", name="get_last_result",
        arguments_json=json.dumps({"stage": "recon"}),
    )
    out = r.dispatch(ctx, call)
    assert out.ok is True
    assert out.payload["available"] is True
    assert out.payload["summary"]["phase_std"] == 0.5


def test_dispatch_handler_exception_emits_error_event():
    """Force an exception via a tool whose ctx callable raises."""
    from core.sample_map import SampleMap
    r = build_tool_registry()
    err = _ErrorCenter()

    bad_state = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    ctx = ToolContext(
        state=bad_state,
        last_recon_summary=lambda: None,
        last_af_summary=lambda: None,
        last_qpi_summary=lambda: None,
        last_depth_summary=lambda: None,
        audit_tail=lambda lim: [],
        load_hologram=lambda p: {},
        set_recon_param=lambda a: {},
        invoke_recon=lambda a: {},
        invoke_autofocus=lambda a: {},
        invoke_qpi=lambda a: {},
        invoke_depth_map=lambda a: {},
        invoke_find_focus_candidates=lambda a: {},
        stage=None,
        capture_frame=lambda: None,
        sample_map=SampleMap(),
        measure_sharpness=lambda f: 0.0,
        persist_sample_map=lambda: None,
        capture_and_process=lambda a: {},
        audit=_Audit(),
        error_center=err,
        confirm=lambda n, a: True,
        settings=_MockSettings(),
    )
    call = ToolCall(id="x", name="get_state", arguments_json="{}")
    out = r.dispatch(ctx, call)
    assert out.ok is False
    assert err.events, "ErrorCenter should have received an event"
