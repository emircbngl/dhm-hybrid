"""DPG-side AI assistant chat state machine — v2 frontend port.

The v1 PySide6 panel (``gui/panels/ai_panel.py``) couples chat
state, tool-call rendering, threading, and Qt widgets in one
class. Porting that wholesale would drag Qt into v2; instead this
module owns the **pure-data side** (turn list, status flags,
streaming-tool-call rendering rules) and a thin DPG wrapper in
``ui2/ai_panel.py`` calls ``set_value`` against the rendered text.

Design rules
------------
* No DPG, no Qt, no threading primitives — those live in
  ``ui2/ai_panel.py``. This is only state + formatting.
* Same ``LocalLLMClient`` + ``Agent`` core machinery as v1; we
  just produce Markdown-style chat blocks the DPG ``add_text`` /
  ``QTextBrowser`` can render.
* Streaming-aware: as ``Agent.run`` yields ``AgentEvent``s, the
  panel feeds them to :meth:`AIChatState.handle_event` and re-
  renders. Every event lands as an immutable ``ChatEntry``.

Public API
----------
* :class:`ChatRole` — enum: USER, ASSISTANT, TOOL_CALL,
  TOOL_RESULT, SYSTEM, ERROR.
* :class:`ChatEntry` — frozen dataclass: role, text, timestamp,
  optional tool_name + tool_args + tool_result.
* :class:`AIChatState` — owns the chronological entry list +
  derived ``transcript`` text + status flag (idle / running /
  cancelled / error).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, List, Optional


class ChatRole(str, Enum):
    """Source of one chat row. Used for the visual prefix."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    ERROR = "error"


_ROLE_PREFIX = {
    ChatRole.USER: ("You", ""),
    ChatRole.ASSISTANT: ("Assistant", ""),
    ChatRole.TOOL_CALL: ("→ tool call", "  "),
    ChatRole.TOOL_RESULT: ("  ← result", "    "),
    ChatRole.SYSTEM: ("[system]", ""),
    ChatRole.ERROR: ("[error]", ""),
}


class AIRunStatus(str, Enum):
    """Coarse status used by the panel header indicator."""
    IDLE = "idle"
    RUNNING = "running"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class ChatEntry:
    """One chat row.

    For ``TOOL_CALL`` / ``TOOL_RESULT`` rows, ``tool_name`` is
    populated and ``tool_args`` / ``tool_result`` carry JSON-
    serialisable payloads. Other roles leave them ``None``.
    """
    role: ChatRole
    text: str
    timestamp_s: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[dict] = None

    def render_block(self, *, max_width: int = 80) -> str:
        """Markdown-ish single-block rendering. Caller appends to
        the chat transcript."""
        prefix, indent = _ROLE_PREFIX[self.role]
        if self.role is ChatRole.TOOL_CALL:
            args = json.dumps(self.tool_args or {}, ensure_ascii=False)
            return (
                f"{prefix} {self.tool_name}({args})\n"
            )
        if self.role is ChatRole.TOOL_RESULT:
            res = json.dumps(self.tool_result or {}, ensure_ascii=False,
                             default=str)
            # Truncate big results so the chat box doesn't stretch.
            if len(res) > 400:
                res = res[:400] + "…"
            return f"{prefix} {self.tool_name} → {res}\n"
        body = self.text.strip()
        if not body:
            return ""
        # Wrap user / assistant blocks at max_width-ish for
        # readability inside the DPG text widget.
        wrapped_lines: List[str] = []
        for line in body.splitlines() or [""]:
            if not line:
                wrapped_lines.append("")
                continue
            cur = ""
            for word in line.split(" "):
                if not cur:
                    cur = word
                elif len(cur) + 1 + len(word) <= max_width:
                    cur = cur + " " + word
                else:
                    wrapped_lines.append(cur)
                    cur = word
            if cur:
                wrapped_lines.append(cur)
        wrapped = "\n".join(indent + ln for ln in wrapped_lines)
        return f"{prefix}\n{wrapped}\n"


@dataclass
class AIChatState:
    """Chronological chat transcript + derived rendering helpers.

    The DPG panel wraps this; the AI worker thread feeds events
    via :meth:`handle_event`. Every public mutator is thread-
    safe-when-the-caller-serialises — :class:`AIChatState` does
    no locking; the panel mailbox guarantees one writer.
    """
    entries: List[ChatEntry] = field(default_factory=list)
    status: AIRunStatus = AIRunStatus.IDLE
    last_error: Optional[str] = None

    def push(self, role: ChatRole, text: str = "", **kwargs) -> None:
        """Append a new entry."""
        self.entries.append(ChatEntry(role=role, text=text, **kwargs))

    def handle_event(self, event: Any) -> None:
        """Convert an :class:`core.ai.agent.AgentEvent` into one
        or more :class:`ChatEntry`. Defensive — unknown event
        kinds become a SYSTEM row rather than raising."""
        # Late imports — keep this module Qt-free + free of the
        # core.ai dependency at *import* time (so unit tests can
        # exercise the state machine without dragging requests).
        kind = getattr(event, "kind", None)
        if kind == "assistant_text":
            self.push(ChatRole.ASSISTANT, str(getattr(event, "text", "")))
        elif kind == "tool_call":
            self.push(
                ChatRole.TOOL_CALL,
                tool_name=str(getattr(event, "name", "")),
                tool_args=dict(getattr(event, "arguments", {}) or {}),
            )
        elif kind == "tool_result":
            self.push(
                ChatRole.TOOL_RESULT,
                tool_name=str(getattr(event, "name", "")),
                tool_result=dict(getattr(event, "result", {}) or {}),
            )
        elif kind == "error":
            err = str(getattr(event, "message", "unknown error"))
            self.push(ChatRole.ERROR, err)
            self.status = AIRunStatus.ERROR
            self.last_error = err
        elif kind == "done":
            self.status = AIRunStatus.IDLE
        elif kind == "cancelled":
            self.status = AIRunStatus.CANCELLED
            self.push(ChatRole.SYSTEM, "(cancelled)")
        else:
            # Unknown — log textually so debugging is possible.
            self.push(ChatRole.SYSTEM, f"event: {kind!r}")

    def transcript(self, *, max_entries: int = 200,
                   max_width: int = 80) -> str:
        """Render the (capped) transcript as plain text the DPG
        ``set_value`` accepts. Newest entries first is *not* the
        chat convention; we keep chronological order so the UI
        scrolls naturally."""
        recent = self.entries[-max_entries:]
        return "\n".join(e.render_block(max_width=max_width)
                         for e in recent).strip()

    def clear(self) -> None:
        self.entries.clear()
        self.status = AIRunStatus.IDLE
        self.last_error = None


__all__ = [
    "ChatRole",
    "ChatEntry",
    "AIRunStatus",
    "AIChatState",
]
