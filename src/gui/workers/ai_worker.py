"""QThread that runs one Agent turn off the GUI thread.

The AI worker exists for one reason: HTTP calls to the local LLM are
synchronous and can take seconds. Running them on the GUI thread would
freeze the window. We hand the whole agent loop to a worker thread and
relay every :class:`AgentEvent` back to the panel via Qt signals — Qt
auto-detects the cross-thread emission and queues the slot call onto
the GUI thread.

The worker owns the :class:`Agent`, the :class:`LocalLLMClient`, the
:class:`ToolRegistry`, and the :class:`ToolContext` for the duration of
one user turn. Everything is constructed before ``start()`` so the GUI
thread can wire signals up. After ``run()`` returns, the worker emits
``finished`` and the panel discards it.

Cancellation: panel's Stop button calls ``requestInterruption()``. The
agent loop checks ``isInterruptionRequested()`` between turns and
between tool calls. The HTTP request itself is not interruptible
mid-flight (no SSE in v1) — we accept that, since the cancel sites
between turns are tight enough for user-perceivable responsiveness.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from PySide6.QtCore import QThread, Signal

from core.ai.agent import Agent, AgentEvent
from core.ai.client import LocalLLMClient
from core.ai.context import StateSnapshot, build_system_prompt
from core.ai.protocol import ChatMessage
from core.ai.tools import ToolContext, ToolRegistry
from core.errors import ErrorEvent, Severity, get_error_center

_LOG = logging.getLogger(__name__)


class AIWorker(QThread):
    """One-shot worker that drives one user prompt to completion."""

    # Forwarded AgentEvents — UI listens to whichever it wants to render.
    # ``object`` types because Qt signals don't introspect dataclasses.
    assistant_text = Signal(str)              # turn.text
    tool_call_start = Signal(object)          # ToolCall
    tool_call_done = Signal(object, bool, dict)   # (ToolCall, ok, result)
    completed = Signal()                      # natural finish
    iteration_cap = Signal(int)               # hit the cap; argument = max
    cancelled_by_user = Signal()
    error_event = Signal(object)              # ErrorEvent

    def __init__(
        self,
        *,
        client: LocalLLMClient,
        registry: ToolRegistry,
        ctx: ToolContext,
        prompt: str,
        history: Iterable[ChatMessage],
        snapshot: StateSnapshot,
        max_iterations: int = 8,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._registry = registry
        self._ctx = ctx
        self._prompt = str(prompt)
        self._history = list(history)
        self._snapshot = snapshot
        self._max = int(max_iterations)
        self.setObjectName("AIWorker")

    # ----- thread entry point ----------------------------------------------

    def run(self) -> None:  # noqa: D401 - QThread API
        try:
            agent = Agent(
                client=self._client,
                registry=self._registry,
                ctx=self._ctx,
                max_iterations=self._max,
            )
            system_prompt = build_system_prompt(self._snapshot, len(self._registry))
            for event in agent.run(
                user_prompt=self._prompt,
                system_prompt=system_prompt,
                history=self._history,
                cancel=self.isInterruptionRequested,
            ):
                self._dispatch(event)
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            _LOG.exception("AIWorker crashed: %s", exc)
            err = ErrorEvent.make(
                severity=Severity.ERROR,
                title="AI worker crashed",
                cause=str(exc),
                action="Check the log; if it persists, report the trace.",
                exc=exc,
            )
            try:
                get_error_center().emit(err)
            except Exception:  # noqa: BLE001
                pass
            self.error_event.emit(err)

    # ----- event dispatch --------------------------------------------------

    def _dispatch(self, event: AgentEvent) -> None:
        kind = event.kind
        if kind == "assistant_text":
            self.assistant_text.emit(event.payload.get("text", ""))
        elif kind == "tool_call_start":
            self.tool_call_start.emit(event.payload.get("call"))
        elif kind == "tool_call_done":
            call = event.payload.get("call")
            ok = bool(event.payload.get("ok", False))
            result = event.payload.get("result", {}) or {}
            self.tool_call_done.emit(call, ok, dict(result))
        elif kind == "completed":
            self.completed.emit()
        elif kind == "iteration_cap":
            self.iteration_cap.emit(self._max)
        elif kind == "cancelled":
            self.cancelled_by_user.emit()
        elif kind == "error":
            err = event.payload.get("event")
            if isinstance(err, ErrorEvent):
                self.error_event.emit(err)
            else:
                # Unknown shape — wrap defensively.
                self.error_event.emit(ErrorEvent.make(
                    severity=Severity.ERROR,
                    title="AI error",
                    cause=str(err) if err is not None else "unknown",
                ))


__all__ = ["AIWorker"]
