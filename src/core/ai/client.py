"""Local LLM HTTP client.

Speaks two dialects:

* **Ollama**           — ``POST {endpoint}/api/chat``. Tool calls land in
  ``message.tool_calls`` with parsed-object ``arguments``.
* **OpenAI-compatible** — ``POST {endpoint}/v1/chat/completions``. Tool
  calls land in ``choices[0].message.tool_calls`` with stringified
  ``function.arguments``.

LM Studio, llama.cpp's server, vLLM, and any future runtime that ships an
OpenAI-style API all fall under the second branch. Detection is by URL:
if the endpoint already ends in ``/v1`` we use OpenAI; otherwise we try
Ollama. The user can force a dialect in :class:`LLMConfig`.

No streaming in v1 — see plan §6 for the rationale. ``cancel`` is checked
periodically while the response is in flight by setting a short read
timeout on the underlying ``requests`` session.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Optional

from core.ai.protocol import AssistantTurn, ChatMessage, ToolCall

_LOG = logging.getLogger(__name__)

Dialect = Literal["auto", "ollama", "openai"]


class LLMClientError(RuntimeError):
    """Raised when the remote endpoint returns an error or is unreachable."""


@dataclass(frozen=True)
class LLMConfig:
    """All knobs that the user can turn for the local LLM.

    ``endpoint`` is a base URL (no trailing path). The client appends
    ``/api/chat`` or ``/v1/chat/completions`` itself based on dialect.
    For LM Studio that means the user types ``http://localhost:1234``
    (not ``http://localhost:1234/v1``) — but we tolerate the trailing
    ``/v1`` and strip it.
    """
    endpoint: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.2
    max_tokens: int = 2048
    request_timeout_s: float = 120.0
    dialect: Dialect = "auto"

    def normalised_endpoint(self) -> str:
        ep = self.endpoint.rstrip("/")
        # If user typed `.../v1`, peel it off — the dialect detector
        # decides whether to add it back.
        if ep.endswith("/v1"):
            ep = ep[: -len("/v1")]
        return ep

    def detected_dialect(self) -> str:
        """Resolve ``auto`` to a concrete dialect by URL heuristic."""
        if self.dialect != "auto":
            return self.dialect
        ep = self.endpoint.rstrip("/")
        # LM Studio defaults to port 1234; OpenAI proxies usually mount
        # under ``/v1``. Either signal flips us to OpenAI mode.
        if ep.endswith("/v1") or ":1234" in ep:
            return "openai"
        return "ollama"


class LocalLLMClient:
    """Sync HTTP wrapper around a local chat-completions endpoint.

    The client is stateless apart from a persistent ``requests.Session``
    that reuses TCP connections across turns. Construct one per panel
    or per app — re-using the session is roughly free and reduces tail
    latency for short turn cycles.
    """

    def __init__(self, config: LLMConfig) -> None:
        # v2.1.z: truly-lazy requests — defer import + Session creation
        # until the first chat() / health_check() call. Comment in the
        # original v12-time code claimed "lazy" but imported eagerly,
        # which made AIPanel.__init__ blow up on systems without the
        # dep (the lab's slim Mac dev box).
        self._config = config
        self._session = None
        self._requests = None

    # ----- public API ------------------------------------------------------

    @property
    def config(self) -> LLMConfig:
        return self._config

    def _ensure_session(self) -> None:
        """First-use bootstrap: import requests, build a Session.

        Raises ``LLMClientError`` if requests isn't installed so the
        AI panel can degrade to "AI unavailable" instead of crashing
        the whole window. Idempotent — every public network method
        calls this before touching ``self._session``.

        We guard on *both* ``_session`` and ``_requests`` so a test that
        monkey-patches ``_session`` (e.g. with a MagicMock) still gets
        ``_requests`` populated for the ``except requests.exceptions...``
        clauses downstream. Otherwise the patched session works for the
        success path but blows up on the error path with
        ``AttributeError: NoneType has no 'exceptions'``.
        """
        if self._session is not None and self._requests is not None:
            return
        try:
            import requests  # noqa: WPS433
        except ImportError as exc:
            raise LLMClientError(
                "AI assistant needs the 'requests' package "
                "(pip install requests). Install it or disable AI "
                "in Settings.",
            ) from exc
        if self._requests is None:
            self._requests = requests
        if self._session is None:
            self._session = requests.Session()

    def with_config(self, **kw) -> "LocalLLMClient":
        """Return a copy with overridden config fields. Reuses the session."""
        clone = LocalLLMClient.__new__(LocalLLMClient)
        clone._config = replace(self._config, **kw)
        clone._session = self._session
        clone._requests = self._requests
        return clone

    def chat(
        self,
        messages: Iterable[ChatMessage],
        tools: list[dict],
        *,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> AssistantTurn:
        """Send one message turn; return the assistant's response.

        ``messages`` includes the entire history the model should see —
        the agent loop owns memory. ``tools`` is the OpenAI-style tools
        array; both dialects accept the same shape (Ollama added support
        in v0.3+).

        If ``cancel`` returns truthy before the request fires, the call
        raises a friendly :class:`LLMClientError` rather than blocking
        until the timeout.
        """
        if cancel is not None and cancel():
            raise LLMClientError("cancelled before request")

        dialect = self._config.detected_dialect()
        if dialect == "ollama":
            return self._chat_ollama(list(messages), tools, cancel=cancel)
        return self._chat_openai(list(messages), tools, cancel=cancel)

    def health_check(self) -> bool:
        """Cheap reachability probe. ``False`` on any failure (404 too).

        Used by the panel for the ``● Connected`` indicator. We try the
        dialect-specific lightweight route first, fall back to a generic
        ``GET /``. Exceptions are swallowed: this is just a UX probe.
        """
        try:
            self._ensure_session()
        except LLMClientError:
            return False
        ep = self._config.normalised_endpoint()
        dialect = self._config.detected_dialect()
        urls = [
            f"{ep}/api/tags" if dialect == "ollama" else f"{ep}/v1/models",
            f"{ep}/",
        ]
        for url in urls:
            try:
                r = self._session.get(url, timeout=2.0)
                if r.status_code < 500:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    # ----- dialect-specific posts -----------------------------------------

    def _chat_ollama(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        cancel: Optional[Callable[[], bool]],
    ) -> AssistantTurn:
        url = f"{self._config.normalised_endpoint()}/api/chat"
        body: dict = {
            "model": self._config.model,
            "messages": [m.to_ollama() for m in messages],
            "stream": False,
            "options": {
                "temperature": float(self._config.temperature),
                "num_predict": int(self._config.max_tokens),
            },
        }
        if tools:
            body["tools"] = list(tools)

        data = self._post_json(url, body, cancel=cancel)

        msg = data.get("message", {}) or {}
        text = msg.get("content", "") or ""
        raw_calls = msg.get("tool_calls", []) or []
        calls = []
        for i, c in enumerate(raw_calls):
            fn = (c or {}).get("function", {}) or {}
            args = fn.get("arguments", {})
            args_json = args if isinstance(args, str) else json.dumps(args)
            calls.append(ToolCall(
                id=str(c.get("id") or f"ollama_{i}"),
                name=str(fn.get("name") or ""),
                arguments_json=args_json,
            ))
        return AssistantTurn(text=text, tool_calls=tuple(calls))

    def _chat_openai(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        cancel: Optional[Callable[[], bool]],
    ) -> AssistantTurn:
        url = f"{self._config.normalised_endpoint()}/v1/chat/completions"
        body: dict = {
            "model": self._config.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": float(self._config.temperature),
            "max_tokens": int(self._config.max_tokens),
            "stream": False,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"

        data = self._post_json(url, body, cancel=cancel)

        choices = data.get("choices") or []
        if not choices:
            return AssistantTurn(text="", tool_calls=())
        msg = (choices[0] or {}).get("message", {}) or {}
        text = msg.get("content", "") or ""
        raw_calls = msg.get("tool_calls", []) or []
        calls = []
        for i, c in enumerate(raw_calls):
            fn = (c or {}).get("function", {}) or {}
            calls.append(ToolCall(
                id=str(c.get("id") or f"openai_{i}"),
                name=str(fn.get("name") or ""),
                arguments_json=str(fn.get("arguments") or "{}"),
            ))
        return AssistantTurn(text=text, tool_calls=tuple(calls))

    # ----- transport -------------------------------------------------------

    def _post_json(
        self,
        url: str,
        body: dict,
        *,
        cancel: Optional[Callable[[], bool]],
    ) -> dict:
        self._ensure_session()
        timeout = float(self._config.request_timeout_s)
        try:
            response = self._session.post(
                url,
                json=body,
                timeout=(5.0, timeout),  # (connect, read)
                headers={"Accept": "application/json"},
            )
        except self._requests.exceptions.RequestException as exc:
            raise LLMClientError(f"HTTP request to {url} failed: {exc}") from exc
        # Cancel after the server got the request: rare but worth catching.
        if cancel is not None and cancel():
            raise LLMClientError("cancelled while awaiting response")

        if response.status_code >= 400:
            preview = response.text[:240] if response.text else "<empty>"
            raise LLMClientError(
                f"endpoint {url} returned {response.status_code}: {preview}"
            )
        try:
            return response.json()
        except ValueError as exc:  # JSONDecodeError
            preview = response.text[:240] if response.text else "<empty>"
            raise LLMClientError(
                f"endpoint {url} returned non-JSON body: {preview}"
            ) from exc


__all__ = [
    "LLMClientError",
    "LLMConfig",
    "LocalLLMClient",
    "Dialect",
]
