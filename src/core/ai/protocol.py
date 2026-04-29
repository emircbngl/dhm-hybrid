"""Cross-backend message shape for the AI assistant.

Both Ollama (`/api/chat`) and OpenAI-compatible (`/v1/chat/completions`)
chat APIs converge on the same idea: a list of role-tagged messages, and
optionally a list of tool calls in the assistant turn. Each backend uses
slightly different wire formats; we keep a normalized in-memory shape
here and let the client translate at the boundary.

Pure Python — no Qt, no HTTP, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """One assistant request to invoke a registered tool.

    ``id`` is the tool-call identifier the assistant will reference when
    the matching tool result message is sent back. ``arguments_json`` is
    a raw JSON string — agent-side code parses it inside a try/except so
    a malformed payload becomes a friendly error rather than a crash.
    """
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ChatMessage:
    """One row in the conversation. Compatible with both dialects."""
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_openai(self) -> dict:
        """Serialize to the OpenAI chat-completions message shape."""
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.content,
            }
        out: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for tc in self.tool_calls
            ]
        return out

    def to_ollama(self) -> dict:
        """Serialize to the Ollama `/api/chat` message shape.

        Ollama uses ``tool_calls`` as a list of dicts with a ``function``
        sub-object whose ``arguments`` is a *parsed object*, not a string.
        We round-trip via JSON so a typo in the assistant's payload
        surfaces as a clean error before it reaches the model.
        """
        import json

        if self.role == "tool":
            return {
                "role": "tool",
                "content": self.content,
            }
        out: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            calls = []
            for tc in self.tool_calls:
                try:
                    args_obj = json.loads(tc.arguments_json) if tc.arguments_json else {}
                except json.JSONDecodeError:
                    args_obj = {}
                calls.append({"function": {"name": tc.name, "arguments": args_obj}})
            out["tool_calls"] = calls
        return out


@dataclass(frozen=True)
class AssistantTurn:
    """What we got back from one model call.

    ``text`` may be empty when the assistant only emitted tool calls.
    Both dialects allow a turn to carry text *and* tool calls together —
    the agent handles either case.
    """
    text: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


def make_tool_result_message(tool_call_id: str, name: str, content: str) -> ChatMessage:
    """Helper for the agent: wrap a tool-result payload as a chat message.

    ``content`` is expected to already be JSON-encoded by the caller — keeping
    the encoding decision at the call site lets us emit ``{"error": ...}``
    bodies without surprising the agent loop.
    """
    return ChatMessage(role="tool", content=content, tool_call_id=tool_call_id, name=name)


__all__ = [
    "Role",
    "ToolCall",
    "ChatMessage",
    "AssistantTurn",
    "make_tool_result_message",
]
