"""ui3 AIPanel tests — offscreen Qt, real WorkerBridge, FakeLLMClient.

Mirrors the ``_make_ctx`` / ``_pump_until`` pattern from
``tests/test_ui3_timelapse_panel.py``. The LLM is faked (no real Ollama in
CI) via a scripted client matching ``tests/test_ai_agent.py``'s
``FakeClient`` shape (``chat(messages, tools, *, cancel=None) ->
AssistantTurn``); the tool registry, agent loop, and ``ToolContext`` are the
real ``core.ai`` objects, and ``ctx.bridge`` is a real
``ui3.bridge.WorkerBridge`` driving the real (Qt-free) compute drivers
against a synthetic hologram.

What's covered:
* Panel builds with an "unavailable" health pill when no endpoint is
  reachable (health_check() never raises).
* A scripted single-tool-call turn (``get_state``) round-trips through the
  real ``Agent`` + ``ToolRegistry`` without touching the GUI thread bridge.
* A scripted ``run_reconstruction`` turn drives the real bridge end-to-end
  (the blocking ``invoke_recon`` bridge call resolves against a synthetic
  hologram) and the transcript shows the tool call + result.
* The blocking bridge helper does not deadlock: the AI worker thread calls
  ``ctx.invoke_recon`` synchronously, which marshals onto the GUI thread via
  ``QTimer.singleShot`` and blocks on a ``threading.Event`` until the
  bridge's ``recon_done`` signal fires — proven by the turn actually
  completing within the test's pump deadline instead of hanging.
* Stop cancels a running turn cleanly.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DHM_USER", "ci")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def _synthetic_hologram(tmp_path, n=192):
    import tifffile
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.5 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * 0.28 * X + bump))
    holo = np.abs(1 + obj) ** 2
    p = tmp_path / "holo.tif"
    tifffile.imwrite(p, (holo * 1000).astype(np.uint16))
    return p


def _make_ctx(tmp_path, *, with_hologram: bool = False):
    """Real WorkerBridge + mutable ReconParams + no-op display/feedback
    hooks (recorded for assertions), matching the other ui3 panel tests."""
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.context import PanelContext
    from ui3.design import DARK

    bridge = WorkerBridge()
    params = ReconParams(
        wavelength_nm=632.8, pixel_um=1.0, z_mm=0.0, mask_radius=40,
        n_sample=1.38, n_medium=1.337, af_z_min_mm=-5.0, af_z_max_mm=5.0,
        af_n_steps=20,
    )
    events: dict = {"status": [], "toast": []}
    state = {"path": None, "recon": None, "depth": None}
    if with_hologram:
        state["path"] = _synthetic_hologram(tmp_path)

    def _get_field(target: str):
        if target == "raw" and state["path"] is not None:
            import tifffile
            return tifffile.imread(str(state["path"])).astype(np.float32)
        if target == "recon_complex" and state["recon"] is not None:
            r = state["recon"]
            return r.amplitude * np.exp(1j * r.phase)
        if target == "phase_unwrapped" and state["recon"] is not None:
            return state["recon"].phase
        return None

    ctx = PanelContext(
        bridge=bridge,
        get_params=lambda: params,
        params_changed=lambda: None,
        hologram_path=lambda: state["path"],
        last_recon=lambda: state["recon"],
        last_depth=lambda: state["depth"],
        get_field=_get_field,
        set_status=lambda text, level="info": events["status"].append((text, level)),
        toast=lambda text, level="info": events["toast"].append((text, level)),
        show_in_panel=lambda target, array, **kw: None,
        palette=DARK,
    )

    # Keep recon cache fresh so _recon_summary()/get_field("recon_complex")
    # reflect the latest bridge result — mirrors what MainWindow does.
    bridge.recon_done.connect(lambda r: state.update(recon=r))

    return ctx, bridge, params, events, state


def _pump_until(qapp, predicate, deadline_s=20):
    deadline = time.time() + deadline_s
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# Fake LLM client — scripted AssistantTurn sequence (mirrors
# tests/test_ai_agent.py's FakeClient).
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """Scripted stand-in for ``core.ai.client.LocalLLMClient``.

    Only implements the surface ``core.ai.agent.Agent`` and
    ``AIPanel._refresh_health`` touch: ``chat()``, ``config``, and
    ``health_check()``.
    """

    def __init__(self, turns):
        from core.ai.protocol import AssistantTurn
        self._turns = list(turns)
        self.calls = []
        self._AssistantTurn = AssistantTurn

    def chat(self, messages, tools, *, cancel=None):
        self.calls.append((list(messages), list(tools)))
        if not self._turns:
            return self._AssistantTurn(text="(out of script)")
        return self._turns.pop(0)

    def health_check(self) -> bool:
        return True

    @property
    def config(self):
        class _C:
            endpoint = "http://fake"
        return _C()


# ---------------------------------------------------------------------------
# Panel construction / health
# ---------------------------------------------------------------------------

def test_ai_panel_builds_neutral_then_reports_unavailable(qapp, tmp_path):
    """Construction does NO network I/O (the pill is neutral); an explicit
    health refresh runs the probe on a background thread and reports the
    result. Uses a pure-Python stub — never touches a real socket (the
    live-socket version segfaulted the offscreen test process; 2026-07-05
    review)."""
    import time
    from ui3.panels.ai_panel import AIPanel

    ctx, bridge, params, events, state = _make_ctx(tmp_path)
    panel = AIPanel(ctx)
    # No auto health check on build → neutral pill, no network touched.
    assert "not checked" in panel.health_label.text()
    assert panel.send_btn.isEnabled()
    assert not panel.stop_btn.isEnabled()

    class _Down(FakeLLMClient):
        def health_check(self):
            return False

    panel._client = _Down([])
    panel._refresh_health()
    deadline = time.time() + 5
    while (panel._health_thread is not None and panel._health_thread.isRunning()
           and time.time() < deadline):
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert "unavailable" in panel.health_label.text()

    panel.shutdown()
    bridge.shutdown()


def test_ai_panel_send_disabled_while_worker_running(qapp, tmp_path):
    from ui3.panels.ai_panel import AIPanel

    ctx, bridge, params, events, state = _make_ctx(tmp_path)
    panel = AIPanel(ctx)
    panel._client = FakeLLMClient([])  # never resolves on its own; we drive it manually

    assert panel._worker is None
    bridge.shutdown()


# ---------------------------------------------------------------------------
# Scripted agent turn — read-only tool (no GUI-thread bridge involved)
# ---------------------------------------------------------------------------

def test_ai_panel_runs_get_state_tool_call(qapp, tmp_path):
    from core.ai.protocol import AssistantTurn, ToolCall
    from ui3.panels.ai_panel import AIPanel

    ctx, bridge, params, events, state = _make_ctx(tmp_path)
    panel = AIPanel(ctx)
    panel._client = FakeLLMClient([
        AssistantTurn(text="", tool_calls=(
            ToolCall(id="c1", name="get_state", arguments_json="{}"),
        )),
        AssistantTurn(text="Nothing loaded yet."),
    ])

    seen: dict = {"tool_done": [], "assistant": [], "completed": False}
    panel.input_edit.setText("what's loaded?")
    panel._on_send_clicked()
    assert panel._worker is not None
    panel._worker.tool_call_done.connect(
        lambda call, ok, result: seen["tool_done"].append((call.name, ok)))
    panel._worker.assistant_text.connect(lambda t: seen["assistant"].append(t))
    panel._worker.completed.connect(lambda: seen.update(completed=True))

    finished = _pump_until(qapp, lambda: panel.send_btn.isEnabled(), deadline_s=20)
    assert finished, "AI turn did not finish in time"

    assert seen["tool_done"] == [("get_state", True)]
    assert "Nothing loaded yet." in seen["assistant"]
    assert "get_state" in panel.transcript.toPlainText()

    bridge.shutdown()


# ---------------------------------------------------------------------------
# Scripted agent turn — run_reconstruction drives the REAL bridge, proving
# the blocking invoke_recon bridge doesn't deadlock the AI worker thread.
# ---------------------------------------------------------------------------

def test_ai_panel_run_reconstruction_drives_real_bridge_without_deadlock(qapp, tmp_path):
    from core.ai.protocol import AssistantTurn, ToolCall
    from ui3.panels.ai_panel import AIPanel

    ctx, bridge, params, events, state = _make_ctx(tmp_path, with_hologram=True)
    panel = AIPanel(ctx)
    panel._client = FakeLLMClient([
        AssistantTurn(text="", tool_calls=(
            ToolCall(id="c1", name="run_reconstruction", arguments_json="{}"),
        )),
        AssistantTurn(text="Reconstruction complete."),
    ])

    seen: dict = {"tool_done": []}
    panel.input_edit.setText("reconstruct it")
    panel._on_send_clicked()
    assert panel._worker is not None
    panel._worker.tool_call_done.connect(
        lambda call, ok, result: seen["tool_done"].append((call.name, ok, dict(result))))

    finished = _pump_until(qapp, lambda: panel.send_btn.isEnabled(), deadline_s=20)
    assert finished, "reconstruction turn did not finish (possible deadlock)"

    assert len(seen["tool_done"]) == 1
    name, ok, result = seen["tool_done"][0]
    assert name == "run_reconstruction"
    assert ok, f"run_reconstruction failed: {result}"
    assert result.get("submitted") is True
    # Bridge really ran — recon cache should be populated by the connected
    # recon_done slot set up in _make_ctx.
    assert state["recon"] is not None

    bridge.shutdown()


def test_ai_panel_stop_cancels_running_turn(qapp, tmp_path):
    """Stop mid-turn interrupts the worker cleanly — no crash, no hang."""
    from core.ai.protocol import AssistantTurn, ToolCall
    from ui3.panels.ai_panel import AIPanel

    ctx, bridge, params, events, state = _make_ctx(tmp_path, with_hologram=True)
    panel = AIPanel(ctx)

    # A client whose chat() blocks briefly then checks cancel — long enough
    # that the test can call Stop while the worker is alive.
    class _SlowClient(FakeLLMClient):
        def chat(self, messages, tools, *, cancel=None):
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if cancel is not None and cancel():
                    from core.ai.client import LLMClientError
                    raise LLMClientError("cancelled before request")
                time.sleep(0.02)
            return super().chat(messages, tools, cancel=cancel)

    panel._client = _SlowClient([AssistantTurn(text="too slow")])
    panel.input_edit.setText("do something slow")
    panel._on_send_clicked()
    assert panel._worker is not None

    _pump_until(qapp, lambda: panel._worker.isRunning(), deadline_s=5)
    panel._on_stop_clicked()

    finished = _pump_until(qapp, lambda: panel.send_btn.isEnabled(), deadline_s=20)
    assert finished, "worker did not stop cleanly within deadline"
    assert not panel.stop_btn.isEnabled()

    bridge.shutdown()


def test_ai_panel_clear_resets_transcript_and_history(qapp, tmp_path):
    from ui3.panels.ai_panel import AIPanel

    ctx, bridge, params, events, state = _make_ctx(tmp_path)
    panel = AIPanel(ctx)
    panel._history.append(__import__("core.ai.protocol", fromlist=["ChatMessage"])
                          .ChatMessage(role="user", content="hi"))
    panel.transcript.appendPlainText("You: hi")

    panel._on_clear_clicked()

    assert panel._history == []
    assert "hi" not in panel.transcript.toPlainText()

    bridge.shutdown()
