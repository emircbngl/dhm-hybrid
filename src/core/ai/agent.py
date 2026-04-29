"""Chat → tool-call → response loop.

Pure Python (no Qt). Wraps a :class:`LocalLLMClient` and a
:class:`ToolRegistry`, runs them in a bounded loop, and yields one
:class:`AgentEvent` per logical step. The panel turns each event into
a row in the chat history; tests drive the same loop with a fake
client.

Iteration cap is the safety net: a misbehaving model can produce an
endless tool-call cascade. Eight iterations is enough for the deep
chains we care about (load → set params → recon → AF → recon → QPI)
while still bailing fast on a regression.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Literal, Optional

from core.ai.client import LLMClientError, LocalLLMClient
from core.ai.protocol import ChatMessage, ToolCall
from core.ai.tools import ToolContext, ToolDispatchResult, ToolRegistry
from core.errors import ErrorEvent, Severity

_LOG = logging.getLogger(__name__)


AgentKind = Literal[
    "assistant_text",
    "tool_call_start",
    "tool_call_done",
    "completed",
    "error",
    "iteration_cap",
    "cancelled",
]


@dataclass(frozen=True)
class AgentEvent:
    """One step the panel needs to render or the test needs to assert.

    ``payload`` carries kind-specific data:
        * assistant_text → ``{"text": "..."}``
        * tool_call_start → ``{"call": ToolCall}``
        * tool_call_done → ``{"call": ToolCall, "ok": bool, "result": dict}``
        * error → ``{"event": ErrorEvent}``
        * iteration_cap / completed / cancelled → ``{}``
    """
    kind: AgentKind
    payload: dict = field(default_factory=dict)


class Agent:
    """One-shot agent runner. Construct, call ``run`` once, then drop."""

    def __init__(
        self,
        client: LocalLLMClient,
        registry: ToolRegistry,
        ctx: ToolContext,
        *,
        max_iterations: int = 8,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._client = client
        self._registry = registry
        self._ctx = ctx
        self._max = int(max_iterations)

    # ----- entry point -----------------------------------------------------

    def run(
        self,
        user_prompt: str,
        system_prompt: str,
        history: Iterable[ChatMessage],
        *,
        cancel: Callable[[], bool] = lambda: False,
    ) -> Iterator[AgentEvent]:
        """Drive one user turn to completion (or cap, or cancel).

        ``history`` is the conversation up to but *not* including the
        new user prompt. The agent prepends the system prompt, appends
        the user prompt, then iterates: call LLM → emit text → run
        tool calls → append tool results → loop.
        """
        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
        messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_prompt))

        tools_schema = self._registry.schemas()

        for iteration in range(self._max):
            if cancel():
                yield AgentEvent("cancelled")
                return

            try:
                turn = self._client.chat(
                    messages, tools_schema, cancel=cancel,
                )
            except LLMClientError as exc:
                event = ErrorEvent.make(
                    severity=Severity.ERROR,
                    title="AI request failed",
                    cause=str(exc),
                    action="Check that the local LLM endpoint is running and reachable.",
                    context={"iteration": iteration, "endpoint": self._client.config.endpoint},
                )
                yield AgentEvent("error", {"event": event})
                return

            # Emit text (if any) before tool calls
            if turn.text:
                yield AgentEvent("assistant_text", {"text": turn.text})

            # Track the assistant turn in the message history regardless
            # of whether tool calls follow — some LLMs emit text then
            # tool calls in the same turn.
            messages.append(ChatMessage(
                role="assistant",
                content=turn.text or "",
                tool_calls=tuple(turn.tool_calls),
            ))

            if not turn.tool_calls:
                yield AgentEvent("completed")
                return

            # Run tool calls in order. Each one's result joins the
            # message history so the next iteration sees it.
            for call in turn.tool_calls:
                if cancel():
                    yield AgentEvent("cancelled")
                    return
                yield AgentEvent("tool_call_start", {"call": call})
                result: ToolDispatchResult = self._registry.dispatch(self._ctx, call)
                yield AgentEvent(
                    "tool_call_done",
                    {"call": call, "ok": result.ok, "result": result.payload},
                )
                messages.append(result.message)

        # Hit the iteration cap.
        yield AgentEvent("iteration_cap")


__all__ = [
    "Agent",
    "AgentEvent",
    "AgentKind",
]
