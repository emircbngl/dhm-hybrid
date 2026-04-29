"""Agent loop: text-only, tool calls, iteration cap, cancellation.

Uses a scripted ``FakeClient`` that yields a deterministic sequence of
:class:`AssistantTurn`s. No HTTP, no real LLM — the loop's contracts
are entirely about message accounting + tool-call dispatch.
"""
from __future__ import annotations

import json

import pytest

from core.ai.agent import Agent, AgentEvent
from core.ai.protocol import AssistantTurn, ChatMessage, ToolCall
from core.ai.tool_impls import build_tool_registry
from core.ai.tools import ToolContext


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeClient:
    """LLM stand-in. Returns scripted turns one by one."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []  # captured (messages, tools) per chat()

    def chat(self, messages, tools, *, cancel=None):
        self.calls.append((list(messages), list(tools)))
        if not self._turns:
            return AssistantTurn(text="(out of script)")
        return self._turns.pop(0)

    @property
    def config(self):
        # Agent error path inspects this on failure; provide a minimal stub.
        class _C:
            endpoint = "http://fake"
        return _C()


class _Audit:
    def __init__(self):
        self.records = []

    def record(self, action, params, result_summary=None):
        self.records.append((action, params, result_summary))


class _ErrorCenter:
    def __init__(self):
        self.events = []

    def error(self, **kwargs):
        self.events.append(kwargs)


def _make_ctx(snapshot_dict=None) -> ToolContext:
    import numpy as np
    from core.sample_map import SampleMap
    from core.stage import MockStage

    class _Snap:
        def __init__(self):
            self.loaded_path = None
            self.loaded_shape = None
            self.loaded_dtype = None
            self.recon_params = {}
            self.autofocus_params = {}
            self.qpi_params = {}
            self.last_recon_summary = None
            self.last_af_summary = None
            self.last_qpi_summary = None
            self.last_depth_summary = None
            self.stage_position_mm = (0.0, 0.0, 0.0)
            self.audit_tail = ()

        def as_dict(self):
            return snapshot_dict or {"loaded_path": None}

    class _Settings:
        restrict_to_home = False
        audit_redact_for_llm = True

    snap = _Snap()
    return ToolContext(
        state=lambda: snap,
        last_recon_summary=lambda: snap.last_recon_summary,
        last_af_summary=lambda: snap.last_af_summary,
        last_qpi_summary=lambda: snap.last_qpi_summary,
        last_depth_summary=lambda: snap.last_depth_summary,
        audit_tail=lambda lim: [],
        load_hologram=lambda p: {"path": p},
        set_recon_param=lambda a: {"updated": a},
        invoke_recon=lambda a: {"submitted": True},
        invoke_autofocus=lambda a: {"submitted": True, "best_z_mm": 12.4},
        invoke_qpi=lambda a: {"submitted": True},
        invoke_depth_map=lambda a: {"submitted": True},
        invoke_find_focus_candidates=lambda a: {"submitted": True},
        stage=MockStage(),
        capture_frame=lambda: np.zeros((16, 16), dtype=np.float32),
        sample_map=SampleMap(),
        measure_sharpness=lambda f: 1.0,
        persist_sample_map=lambda: None,
        capture_and_process=lambda a: {"phase_std": 0.5},
        audit=_Audit(),
        error_center=_ErrorCenter(),
        confirm=lambda n, a: True,
        settings=_Settings(),
    )


def _events(agent_iter):
    return list(agent_iter)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_max_iterations_must_be_positive():
    r = build_tool_registry()
    ctx = _make_ctx()
    with pytest.raises(ValueError):
        Agent(client=FakeClient([]), registry=r, ctx=ctx, max_iterations=0)


def test_text_only_response_completes_in_one_turn():
    r = build_tool_registry()
    ctx = _make_ctx()
    client = FakeClient([AssistantTurn(text="Hello!")])
    agent = Agent(client=client, registry=r, ctx=ctx, max_iterations=4)
    events = _events(agent.run("Hi", "system", history=[]))
    kinds = [e.kind for e in events]
    assert "assistant_text" in kinds
    assert kinds[-1] == "completed"
    assert len(client.calls) == 1


def test_single_tool_call_then_text_finishes_normally():
    r = build_tool_registry()
    ctx = _make_ctx()
    turns = [
        AssistantTurn(
            text="",
            tool_calls=(ToolCall(id="c1", name="get_state",
                                 arguments_json="{}"),),
        ),
        AssistantTurn(text="State retrieved.")
    ]
    client = FakeClient(turns)
    agent = Agent(client=client, registry=r, ctx=ctx, max_iterations=4)
    events = _events(agent.run("status?", "system", history=[]))
    kinds = [e.kind for e in events]
    assert kinds.count("tool_call_start") == 1
    assert kinds.count("tool_call_done") == 1
    assert kinds[-1] == "completed"


def test_tool_call_chain_runs_in_order():
    r = build_tool_registry()
    ctx = _make_ctx()
    turns = [
        AssistantTurn(text="", tool_calls=(
            ToolCall(id="c1", name="get_state", arguments_json="{}"),
        )),
        AssistantTurn(text="", tool_calls=(
            ToolCall(id="c2", name="run_reconstruction", arguments_json="{}"),
        )),
        AssistantTurn(text="Done."),
    ]
    client = FakeClient(turns)
    agent = Agent(client=client, registry=r, ctx=ctx, max_iterations=8)
    events = _events(agent.run("plan", "system", history=[]))
    started = [e.payload["call"].name
               for e in events if e.kind == "tool_call_start"]
    assert started == ["get_state", "run_reconstruction"]


def test_iteration_cap_emits_event_and_stops():
    r = build_tool_registry()
    ctx = _make_ctx()
    # Always returns a tool call → never finishes → must hit cap.
    turns = [
        AssistantTurn(text="", tool_calls=(
            ToolCall(id=f"c{i}", name="get_state", arguments_json="{}"),
        ))
        for i in range(20)
    ]
    client = FakeClient(turns)
    agent = Agent(client=client, registry=r, ctx=ctx, max_iterations=3)
    events = _events(agent.run("loop", "system", history=[]))
    assert events[-1].kind == "iteration_cap"
    assert len(client.calls) == 3


def test_cancellation_between_iterations():
    r = build_tool_registry()
    ctx = _make_ctx()
    turns = [
        AssistantTurn(text="", tool_calls=(
            ToolCall(id="c1", name="get_state", arguments_json="{}"),
        )),
        AssistantTurn(text="never reached."),
    ]
    client = FakeClient(turns)
    agent = Agent(client=client, registry=r, ctx=ctx, max_iterations=8)

    flips = {"n": 0}
    def cancel():
        flips["n"] += 1
        return flips["n"] >= 3  # cancel after a few checks

    events = _events(agent.run("x", "system", history=[], cancel=cancel))
    assert any(e.kind == "cancelled" for e in events)


def test_unknown_tool_returns_error_to_llm_and_continues():
    r = build_tool_registry()
    ctx = _make_ctx()
    turns = [
        AssistantTurn(text="", tool_calls=(
            ToolCall(id="c1", name="does_not_exist", arguments_json="{}"),
        )),
        AssistantTurn(text="Ah, sorry."),
    ]
    client = FakeClient(turns)
    agent = Agent(client=client, registry=r, ctx=ctx, max_iterations=4)
    events = _events(agent.run("call missing", "system", history=[]))
    done_events = [e for e in events if e.kind == "tool_call_done"]
    assert len(done_events) == 1
    assert done_events[0].payload["ok"] is False
    assert events[-1].kind == "completed"


def test_llm_error_is_surfaced_as_error_event():
    from core.ai.client import LLMClientError

    class BoomClient(FakeClient):
        def chat(self, messages, tools, *, cancel=None):
            raise LLMClientError("endpoint not reachable")

    r = build_tool_registry()
    ctx = _make_ctx()
    agent = Agent(client=BoomClient([]), registry=r, ctx=ctx)
    events = _events(agent.run("hi", "system", history=[]))
    assert events[-1].kind == "error"


def test_tool_result_message_appended_to_history_for_next_turn():
    """Second LLM turn must see the tool result row added by the agent."""
    r = build_tool_registry()
    ctx = _make_ctx()
    turns = [
        AssistantTurn(text="", tool_calls=(
            ToolCall(id="c1", name="get_state", arguments_json="{}"),
        )),
        AssistantTurn(text="ok."),
    ]
    client = FakeClient(turns)
    agent = Agent(client=client, registry=r, ctx=ctx, max_iterations=4)
    _events(agent.run("status?", "system", history=[]))
    second_call_messages = client.calls[1][0]
    roles = [m.role for m in second_call_messages]
    assert "tool" in roles, "tool result row must reach the second LLM call"
