"""Dear PyGui AI assistant panel — v2 port of the v1 PySide6 panel.

What it does
------------
Modal-style window with chat history, single-line input, Send /
Stop / Clear buttons, and a settings text input for the LLM
endpoint URL. AI Agent runs on a daemon thread so the DPG render
loop stays responsive; events flow back through ``app._post_mailbox``
the same way camera frames do, so re-render lands on the next tick.

Public API
----------
* :func:`show_ai_panel(app)` — open or focus the panel.
* :class:`AIPanelController` — owns the chat state + worker
  thread + ToolContext bridge. Tests construct one directly.

The DPG widgets are minimal — chat history is a single multi-line
``input_text`` set to read-only with the rendered transcript. That
trades pretty styling for predictable layout: the v2 frontend
doesn't have a rich-text widget, and ``add_text`` per row blows
up the layout once you cross ~50 entries.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import dearpygui.dearpygui as dpg

from .ai_panel_state import AIChatState, AIRunStatus, ChatRole


_LOG = logging.getLogger(__name__)


_DLG_TAG = "ai_assistant_window"
_TRANSCRIPT_TAG = "ai_assistant_transcript"
_INPUT_TAG = "ai_assistant_input"
_STATUS_TAG = "ai_assistant_status"
_ENDPOINT_TAG = "ai_assistant_endpoint"
_MODEL_TAG = "ai_assistant_model"


# Minimal AgentEvent-like type so the state machine has a stable
# attr surface across whatever ``Agent`` yields. The real event
# type lives in ``core.ai.agent``; we duck-type so a future
# rename there doesn't break v2.
class _Evt:
    def __init__(self, kind: str, **kw):
        self.kind = kind
        for k, v in kw.items():
            setattr(self, k, v)


class AIPanelController:
    """Bridges the DPG window to ``core.ai.agent.Agent``.

    Construction is cheap — no LLM client is built until
    :meth:`send` runs. ``app`` is a ``DhmApp``-shaped object
    used to build the :class:`ToolContext` on each turn.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.state = AIChatState()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        # Conversation history fed to the LLM each turn. Kept on
        # the controller so multi-turn context works.
        self._history: list = []  # list of ChatMessage
        # Wire app._ai_cancel_check so the bridge can pass it
        # through to ToolContext.is_cancelled — long-running
        # tools poll this between iterations.
        app._ai_cancel_check = self._cancel.is_set

    # ── Public surface ────────────────────────────────────────
    def send(self, prompt: str) -> None:
        """Kick off one agent turn on the worker thread. No-op if
        a turn is already in flight."""
        if not prompt.strip():
            return
        if self._thread is not None and self._thread.is_alive():
            self.state.push(
                ChatRole.SYSTEM,
                "(busy — wait for the current turn to finish "
                "or click Stop)",
            )
            return
        self._cancel.clear()
        self.state.push(ChatRole.USER, prompt)
        self.state.status = AIRunStatus.RUNNING
        self._thread = threading.Thread(
            target=self._run_turn, args=(prompt,),
            daemon=True, name="ai-agent-worker",
        )
        self._thread.start()

    def cancel(self) -> None:
        """Request the in-flight turn to stop. Latency depends on
        the LLM endpoint's read timeout + tool cancel checks."""
        self._cancel.set()

    def clear(self) -> None:
        """Reset chat + history."""
        self.state.clear()
        self._history.clear()

    # ── Worker — runs on a daemon thread ──────────────────────
    def _run_turn(self, prompt: str) -> None:
        try:
            from core.ai.agent import Agent
            from core.ai.client import LocalLLMClient, LLMConfig
            from core.ai.protocol import ChatMessage
            from core.ai.tool_impls import build_tool_registry
            from .ai_bridge import make_v2_tool_context

            ai_settings = getattr(self.app._settings, "ai", None)
            cfg = LLMConfig(
                endpoint=getattr(ai_settings, "endpoint_url",
                                 "http://localhost:11434"),
                model=getattr(ai_settings, "model_name", "llama3.2"),
                temperature=float(
                    getattr(ai_settings, "temperature", 0.2),
                ),
                max_tokens=int(
                    getattr(ai_settings, "max_tokens", 2048),
                ),
                request_timeout_s=float(
                    getattr(ai_settings, "request_timeout_s", 120.0),
                ),
            )
            client = LocalLLMClient(cfg)
            registry = build_tool_registry()
            ctx = make_v2_tool_context(self.app)

            self._history.append(
                ChatMessage(role="user", content=prompt),
            )
            agent = Agent(
                client=client,
                registry=registry,
                context=ctx,
                max_iterations=int(
                    getattr(ai_settings, "max_iterations", 8),
                ),
            )
            for event in agent.run(
                self._history,
                cancel=self._cancel.is_set,
            ):
                self._dispatch_event(event)
                if self._cancel.is_set():
                    break
            # Mark idle when finished cleanly.
            self._dispatch_event(_Evt("done"))
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("ai-agent-worker crashed")
            self._dispatch_event(
                _Evt("error", message=f"{type(exc).__name__}: {exc}"),
            )

    def _dispatch_event(self, event: Any) -> None:
        """Forward an ``AgentEvent`` (or our ``_Evt`` stand-in)
        to the chat state. Posts on the app mailbox so the next
        DPG render tick refreshes the transcript."""
        self.state.handle_event(event)
        # Best-effort mailbox post — if the app exposes one, the
        # render loop will repaint. Otherwise we rely on the
        # caller polling ``state.transcript()``.
        post = getattr(self.app, "_post_mailbox", None)
        if callable(post):
            try:
                post(("ai_event", event))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# DPG window
# ---------------------------------------------------------------------------

def show_ai_panel(app: Any) -> AIPanelController:
    """Open the AI assistant window. Idempotent — re-opens with
    the same controller so chat history survives close/reopen."""
    ctrl = getattr(app, "_ai_panel_controller", None)
    if ctrl is None:
        ctrl = AIPanelController(app)
        app._ai_panel_controller = ctrl

    if dpg.does_item_exist(_DLG_TAG):
        dpg.configure_item(_DLG_TAG, show=True)
        try:
            dpg.focus_item(_INPUT_TAG)
        except Exception:
            pass
        return ctrl

    ai_settings = getattr(app._settings, "ai", None)
    endpoint_default = getattr(
        ai_settings, "endpoint_url", "http://localhost:11434",
    )
    model_default = getattr(ai_settings, "model_name", "llama3.2")

    def _refresh_transcript() -> None:
        try:
            dpg.set_value(
                _TRANSCRIPT_TAG, ctrl.state.transcript(max_width=72),
            )
            dpg.set_value(
                _STATUS_TAG,
                {
                    AIRunStatus.IDLE: "● idle",
                    AIRunStatus.RUNNING: "● running",
                    AIRunStatus.CANCELLED: "● cancelled",
                    AIRunStatus.ERROR: "● error",
                }[ctrl.state.status],
            )
        except Exception:
            pass

    # Hook the controller's dispatch so every event triggers a
    # refresh. We wrap, not replace, so the mailbox post still
    # fires for any future hooks.
    _orig_dispatch = ctrl._dispatch_event

    def _dispatch_then_refresh(event: Any) -> None:
        _orig_dispatch(event)
        _refresh_transcript()

    ctrl._dispatch_event = _dispatch_then_refresh  # type: ignore

    def _on_send() -> None:
        prompt = (dpg.get_value(_INPUT_TAG) or "").strip()
        if not prompt:
            return
        # Apply latest endpoint / model overrides into the
        # AI settings dataclass so the next turn picks them up.
        try:
            ep = (dpg.get_value(_ENDPOINT_TAG) or "").strip()
            md = (dpg.get_value(_MODEL_TAG) or "").strip()
            if ep:
                app._settings = app._settings.with_ai(endpoint_url=ep)
            if md:
                app._settings = app._settings.with_ai(model_name=md)
        except Exception:
            pass
        ctrl.send(prompt)
        dpg.set_value(_INPUT_TAG, "")
        _refresh_transcript()

    def _on_stop() -> None:
        ctrl.cancel()
        _refresh_transcript()

    def _on_clear() -> None:
        ctrl.clear()
        _refresh_transcript()

    def _on_close() -> None:
        try:
            dpg.configure_item(_DLG_TAG, show=False)
        except Exception:
            pass

    with dpg.window(label="AI Assistant", tag=_DLG_TAG,
                    modal=False, no_collapse=True,
                    width=560, height=620):
        dpg.add_text(
            "Local LLM-driven assistant. Configure endpoint + "
            "model below; first turn warms the connection. Tools: "
            "load_hologram, run_reconstruction, run_autofocus, "
            "run_qpi, depth_map, shutter / led / acquire_grid, …",
            wrap=520,
        )
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text("Endpoint")
            dpg.add_input_text(
                tag=_ENDPOINT_TAG, default_value=endpoint_default,
                width=300,
            )
            dpg.add_text("Model")
            dpg.add_input_text(
                tag=_MODEL_TAG, default_value=model_default,
                width=160,
            )
        dpg.add_separator()
        dpg.add_input_text(
            tag=_TRANSCRIPT_TAG,
            default_value="",
            multiline=True, readonly=True,
            width=-1, height=380,
        )
        dpg.add_separator()
        dpg.add_input_text(
            tag=_INPUT_TAG, hint="Ask the assistant…",
            width=-1, multiline=False,
            on_enter=True, callback=lambda *_: _on_send(),
        )
        with dpg.group(horizontal=True):
            dpg.add_button(label="Send", callback=lambda: _on_send())
            dpg.add_button(label="Stop", callback=lambda: _on_stop())
            dpg.add_button(label="Clear", callback=lambda: _on_clear())
            dpg.add_spacer(width=12)
            dpg.add_text("● idle", tag=_STATUS_TAG)
            dpg.add_spacer(width=12)
            dpg.add_button(label="Close", callback=lambda: _on_close())

    _refresh_transcript()
    return ctrl


__all__ = [
    "AIPanelController",
    "show_ai_panel",
]
