"""LocalLLMClient: dialect detection, request body shape, response parsing.

We don't hit a real network. The ``requests`` session is replaced with
a stub that captures the outgoing body and returns a canned response.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

requests = pytest.importorskip("requests")

from core.ai.client import LLMClientError, LLMConfig, LocalLLMClient
from core.ai.protocol import ChatMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_session(response_payload: dict, status: int = 200):
    """Build a MagicMock session whose post returns canned JSON."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = status
    response.json.return_value = response_payload
    response.text = json.dumps(response_payload)
    session.post.return_value = response
    return session


def _new_client(*, endpoint="http://localhost:11434", model="llama3.2",
                dialect="auto"):
    return LocalLLMClient(LLMConfig(endpoint=endpoint, model=model,
                                    dialect=dialect))


def _patch_session(client: LocalLLMClient, session) -> None:
    client._session = session  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------

def test_default_endpoint_resolves_to_ollama():
    cfg = LLMConfig()
    assert cfg.detected_dialect() == "ollama"


def test_lm_studio_port_resolves_to_openai():
    cfg = LLMConfig(endpoint="http://localhost:1234")
    assert cfg.detected_dialect() == "openai"


def test_v1_suffix_resolves_to_openai():
    cfg = LLMConfig(endpoint="http://example.com/v1")
    assert cfg.detected_dialect() == "openai"


def test_explicit_dialect_overrides_heuristic():
    cfg = LLMConfig(endpoint="http://localhost:11434", dialect="openai")
    assert cfg.detected_dialect() == "openai"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

def test_ollama_request_body_includes_model_messages_options():
    client = _new_client()
    canned = {"message": {"content": "ok", "tool_calls": []}}
    session = _stub_session(canned)
    _patch_session(client, session)

    out = client.chat(
        messages=[ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )

    args, kwargs = session.post.call_args
    assert args[0].endswith("/api/chat")
    body = kwargs["json"]
    assert body["model"] == "llama3.2"
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "user"
    assert "tools" in body
    assert out.text == "ok"


def test_openai_request_body_targets_v1_chat_completions():
    client = _new_client(endpoint="http://localhost:1234", model="lmstudio")
    canned = {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}
    session = _stub_session(canned)
    _patch_session(client, session)

    client.chat(
        messages=[ChatMessage(role="user", content="ping")],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )
    args, kwargs = session.post.call_args
    assert args[0].endswith("/v1/chat/completions")
    body = kwargs["json"]
    assert body["model"] == "lmstudio"
    assert body["stream"] is False
    assert body["temperature"] == LLMConfig().temperature


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_ollama_tool_call_parsed_into_tool_calls():
    client = _new_client()
    canned = {
        "message": {
            "content": "running",
            "tool_calls": [{
                "function": {
                    "name": "load_hologram",
                    "arguments": {"path": "/tmp/x.tif"},
                },
            }],
        },
    }
    _patch_session(client, _stub_session(canned))
    out = client.chat([ChatMessage(role="user", content="x")], tools=[])
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.name == "load_hologram"
    assert json.loads(tc.arguments_json) == {"path": "/tmp/x.tif"}


def test_openai_tool_call_parsed_into_tool_calls():
    client = _new_client(endpoint="http://localhost:1234")
    canned = {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "abc",
                    "type": "function",
                    "function": {
                        "name": "get_state",
                        "arguments": "{}",
                    },
                }],
            },
        }],
    }
    _patch_session(client, _stub_session(canned))
    out = client.chat([ChatMessage(role="user", content="x")], tools=[])
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].id == "abc"
    assert out.tool_calls[0].name == "get_state"


def test_empty_choices_returns_empty_turn():
    client = _new_client(endpoint="http://localhost:1234")
    _patch_session(client, _stub_session({"choices": []}))
    out = client.chat([ChatMessage(role="user", content="x")], tools=[])
    assert out.text == ""
    assert out.tool_calls == ()


# ---------------------------------------------------------------------------
# Errors / cancellation
# ---------------------------------------------------------------------------

def test_http_400_raises_client_error():
    client = _new_client()
    session = _stub_session({"error": "bad request"}, status=400)
    _patch_session(client, session)
    with pytest.raises(LLMClientError):
        client.chat([ChatMessage(role="user", content="x")], tools=[])


def test_pre_request_cancel_raises():
    client = _new_client()
    _patch_session(client, _stub_session({"message": {"content": ""}}))
    with pytest.raises(LLMClientError):
        client.chat([ChatMessage(role="user", content="x")], tools=[],
                    cancel=lambda: True)


def test_request_exception_translates_to_client_error():
    client = _new_client()
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError("nope")
    _patch_session(client, session)
    with pytest.raises(LLMClientError):
        client.chat([ChatMessage(role="user", content="x")], tools=[])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check_succeeds_on_ok_response():
    client = _new_client()
    session = MagicMock()
    response = MagicMock(status_code=200)
    session.get.return_value = response
    _patch_session(client, session)
    assert client.health_check() is True


def test_health_check_returns_false_when_get_raises():
    client = _new_client()
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("nope")
    _patch_session(client, session)
    assert client.health_check() is False


# ---------------------------------------------------------------------------
# normalised_endpoint
# ---------------------------------------------------------------------------

def test_normalised_endpoint_strips_trailing_v1():
    cfg = LLMConfig(endpoint="http://localhost:1234/v1")
    assert cfg.normalised_endpoint() == "http://localhost:1234"


def test_normalised_endpoint_strips_trailing_slash():
    cfg = LLMConfig(endpoint="http://localhost:11434/")
    assert cfg.normalised_endpoint() == "http://localhost:11434"
