"""Local-LLM-driven AI assistant — package marker.

Public surface re-exported here so callers can do
``from core.ai import Agent, LocalLLMClient, ToolRegistry`` without learning
the internal layout. The module is import-cheap: no Qt, no HTTP at import
time, no model load.

Layout:
    * :mod:`core.ai.protocol`  — message shape dataclasses (no I/O).
    * :mod:`core.ai.client`    — HTTP wrapper (Ollama + OpenAI dialect).
    * :mod:`core.ai.tools`     — registry, tool spec, dispatch.
    * :mod:`core.ai.tool_impls`— concrete tool callables for the DHM app.
    * :mod:`core.ai.context`   — state snapshot + system-prompt builder.
    * :mod:`core.ai.agent`     — chat → tool-call loop with iteration cap.
    * :mod:`core.ai.security`  — guardrails (path, clamp, whitelist).
"""
from __future__ import annotations

from core.ai.protocol import (
    AssistantTurn,
    ChatMessage,
    ToolCall,
    make_tool_result_message,
)
from core.ai.client import LLMConfig, LocalLLMClient
from core.ai.tools import ToolContext, ToolRegistry, ToolSpec
from core.ai.context import StateSnapshot, build_system_prompt
from core.ai.agent import Agent, AgentEvent

__all__ = [
    "AssistantTurn",
    "ChatMessage",
    "ToolCall",
    "make_tool_result_message",
    "LLMConfig",
    "LocalLLMClient",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "StateSnapshot",
    "build_system_prompt",
    "Agent",
    "AgentEvent",
]
