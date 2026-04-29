"""Tool registry, spec, and per-call context.

A *tool* is a function the LLM can call. We declare each tool with a
JSON Schema for its parameters and a Python handler that does the work.
The registry serializes the schemas to the OpenAI tools array shape
(both Ollama and OpenAI accept it) and dispatches by name when a
``ToolCall`` comes back.

The :class:`ToolContext` is a frozen carrier of dependencies that every
handler needs: a way to read app state, a way to drive the existing
pipeline workers from a non-GUI thread, the audit log, and the stage.
Concrete handlers live in :mod:`core.ai.tool_impls`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.ai.protocol import ToolCall, make_tool_result_message
from core.ai.security import SecurityError
from core.audit import AuditLog
from core.errors import ErrorCenter, Severity

_LOG = logging.getLogger(__name__)


# Result of a single dispatch — keeps wiring clean for the agent loop.
@dataclass(frozen=True)
class ToolDispatchResult:
    """Outcome of one ``ToolRegistry.dispatch`` call.

    ``message`` is the chat-shaped tool-result row to feed back to the
    LLM. ``ok`` is True when the handler returned cleanly, False when
    the handler raised or the registry rejected the call. ``payload``
    is the raw dict the handler returned (or an ``{"error": ...}`` if
    the call failed) — useful for the panel's tool-call inspector.
    """
    message: Any  # ChatMessage; typed loosely to avoid cyclic imports.
    ok: bool
    payload: dict


# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """One registered tool.

    ``parameters`` is JSON Schema (Draft 7-ish) — the registry validates
    incoming args against it before invoking the handler. ``handler`` is
    a callable ``(ctx, args_dict) -> dict``.

    ``requires_gui_thread`` marks tools that mutate sidebar widgets or
    submit jobs to the existing pipeline workers. The AI worker can't
    touch QWidgets directly, so the panel's tool context wraps the
    handler in a ``QMetaObject.invokeMethod(..., BlockingQueuedConnection)``.

    ``irreversible`` triggers a confirmation modal. v1's MockStage
    operations are cheap and reversible so this stays False; a real
    stage driver will mark its move tools True.
    """
    name: str
    description: str
    parameters: dict
    handler: Callable[["ToolContext", dict], dict]
    requires_gui_thread: bool = False
    irreversible: bool = False

    def to_openai_schema(self) -> dict:
        """OpenAI/Ollama tools-array entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


# ---------------------------------------------------------------------------
# Tool context
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Dependencies passed into every tool handler.

    The fields are callables rather than direct objects so the panel
    can decide where each call lands: a ``get_state`` that returns a
    cached :class:`StateSnapshot` runs cheaply on the AI thread, while
    ``invoke_recon`` posts to the GUI thread under the hood.

    ``confirm`` blocks until the user clicks Yes/No on a modal. The
    AI thread is paused for the duration; this is intentional — an
    irreversible stage move shouldn't proceed until the operator says so.
    """
    # State queries — must be thread-safe. Implementations cache the
    # latest snapshot on the GUI thread and let any thread read.
    state: Callable[[], Any]
    last_recon_summary: Callable[[], Optional[dict]]
    last_af_summary: Callable[[], Optional[dict]]
    last_qpi_summary: Callable[[], Optional[dict]]
    last_depth_summary: Callable[[], Optional[dict]]
    audit_tail: Callable[[int], list[dict]]

    # Pipeline drivers — each blocks the AI thread until the underlying
    # GUI-thread worker emits its `*_completed` signal. Returns a dict
    # summary the LLM can read.
    load_hologram: Callable[[str], dict]
    set_recon_param: Callable[[dict], dict]
    invoke_recon: Callable[[dict], dict]
    invoke_autofocus: Callable[[dict], dict]
    invoke_qpi: Callable[[dict], dict]
    invoke_depth_map: Callable[[dict], dict]
    invoke_find_focus_candidates: Callable[[dict], dict]

    # Stage interface (MockStage in v1).
    stage: Any

    # ---- Capture + map + time-lapse (sprint 2) ----
    # ``capture_frame`` returns the most recent raw camera image as a
    # 2-D numpy array. In v1 (no real camera) it falls back to the
    # currently-loaded hologram so stage_focus_search and map_sample
    # can produce meaningful sharpness signals against a synthetic
    # offset. Real-hardware drivers replace this with a live grab.
    capture_frame: Callable[[], Any]
    # The current sample map (a ``core.sample_map.SampleMap`` instance).
    # Tools mutate it in place; the panel persists changes after each
    # mapping run.
    sample_map: Any
    # Run a quick sharpness measurement against the most recent frame.
    # Implementations typically use Laplacian variance on the magnitude;
    # exposed as a callable so we can swap metrics without reaching
    # into core internals from the AI module.
    measure_sharpness: Callable[[Any], float]
    # Persist the current sample map to disk. No-op in tests; the real
    # panel writes ``~/.dhm-reconstruction/sample_maps/<sample_id>.json``.
    persist_sample_map: Callable[[], Optional[str]]
    # Take a single snapshot and (optionally) run pipeline steps.
    # Returns a per-frame summary dict the LLM can compare across
    # time-lapse iterations. Heavy — runs on the GUI thread the same
    # way ``invoke_recon`` does.
    capture_and_process: Callable[[dict], dict]

    # Cross-cutting.
    audit: AuditLog
    error_center: ErrorCenter
    confirm: Callable[[str, dict], bool]
    # Settings snapshot — handlers read e.g. ``restrict_to_home`` here
    # rather than re-reading QSettings every call.
    settings: Any
    # v2.1.z: cancellation hook. Long-running tools poll this between
    # iterations so the AI worker's ``Stop`` button can interrupt
    # mid-loop (e.g. ``record_timelapse`` between frames). Default
    # ``None`` keeps existing tools backward compatible.
    is_cancelled: Optional[Callable[[], bool]] = None
    # v2.1.z: device hooks for the new APT-aligned tools. Optional —
    # ``None`` when the panel didn't supply a backend, in which case
    # the tool returns ``{"error": "<kind> device not configured"}``
    # instead of crashing.
    shutter: Any = None
    led: Any = None
    orchestrator: Optional[Callable[[dict], dict]] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Insertion-ordered name → spec map with JSON-Schema validation.

    Construction is cheap; tests build a fresh registry per case to
    avoid singleton bleed.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        # Lazy import — jsonschema is a dep but we don't want module
        # import to blow up if it's missing in a minimal CI env.
        try:
            import jsonschema as _jsonschema  # noqa: WPS433
            self._jsonschema = _jsonschema
        except ImportError:  # pragma: no cover - the test env installs it
            self._jsonschema = None

    # ----- registration / introspection -----------------------------------

    def register(self, spec: ToolSpec) -> None:
        if not spec.name:
            raise ValueError("tool name must be non-empty")
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs.values())

    def schemas(self) -> list[dict]:
        """OpenAI-style tools array."""
        return [s.to_openai_schema() for s in self._specs.values()]

    # ----- dispatch -------------------------------------------------------

    def dispatch(self, ctx: ToolContext, call: ToolCall) -> ToolDispatchResult:
        """Run one tool call. Always returns a result — never raises.

        The agent loop relies on this contract: every tool call ends with
        a ``tool``-role message in the conversation, even when something
        went wrong. Letting an exception propagate would strand the LLM
        without context for what failed.
        """
        spec = self.get(call.name)
        if spec is None:
            payload = {"error": f"unknown tool {call.name!r}",
                       "available": list(self._specs.keys())}
            return ToolDispatchResult(
                message=make_tool_result_message(
                    call.id, call.name, json.dumps(payload),
                ),
                ok=False,
                payload=payload,
            )

        # Parse args
        try:
            args = json.loads(call.arguments_json) if call.arguments_json else {}
            if not isinstance(args, dict):
                raise ValueError(f"args must be an object, got {type(args).__name__}")
        except (ValueError, json.JSONDecodeError) as exc:
            payload = {"error": "invalid JSON arguments",
                       "details": str(exc),
                       "raw": call.arguments_json[:240]}
            return ToolDispatchResult(
                message=make_tool_result_message(
                    call.id, call.name, json.dumps(payload),
                ),
                ok=False,
                payload=payload,
            )

        # Schema validation
        try:
            self._validate_args(spec, args)
        except SecurityError as exc:
            payload = {"error": "security check failed", "details": str(exc)}
            return ToolDispatchResult(
                message=make_tool_result_message(
                    call.id, call.name, json.dumps(payload),
                ),
                ok=False,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001 — invalid args
            payload = {"error": "invalid arguments", "details": str(exc)}
            return ToolDispatchResult(
                message=make_tool_result_message(
                    call.id, call.name, json.dumps(payload),
                ),
                ok=False,
                payload=payload,
            )

        # Confirmation gate
        if spec.irreversible:
            try:
                ok = bool(ctx.confirm(spec.name, args))
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                payload = {"error": "user declined", "tool": spec.name}
                return ToolDispatchResult(
                    message=make_tool_result_message(
                        call.id, call.name, json.dumps(payload),
                    ),
                    ok=False,
                    payload=payload,
                )

        # Audit start
        try:
            ctx.audit.record(f"ai.tool.{spec.name}.start", {"args": args})
        except Exception:  # noqa: BLE001 - never fail because audit failed
            pass

        # Run handler
        try:
            payload = spec.handler(ctx, args)
        except SecurityError as exc:
            payload = {"error": "security check failed", "details": str(exc)}
            ok_flag = False
        except Exception as exc:  # noqa: BLE001
            ctx.error_center.error(
                title=f"AI tool {spec.name} failed",
                cause=str(exc),
                action="Inspect the AI panel and try again, or rephrase the request.",
                exc=exc,
                context={"tool": spec.name, "args": args},
            )
            payload = {"error": type(exc).__name__, "message": str(exc)}
            ok_flag = False
        else:
            ok_flag = "error" not in payload

        # Coerce to JSON-serializable
        payload = _jsonable(payload)

        # Audit done
        try:
            ctx.audit.record(
                f"ai.tool.{spec.name}.done",
                {"args": args},
                result_summary=_compact(payload),
            )
        except Exception:  # noqa: BLE001
            pass

        return ToolDispatchResult(
            message=make_tool_result_message(
                call.id, call.name, json.dumps(payload, default=str),
            ),
            ok=ok_flag,
            payload=payload,
        )

    # ----- helpers --------------------------------------------------------

    def _validate_args(self, spec: ToolSpec, args: dict) -> None:
        if self._jsonschema is None:
            return
        try:
            self._jsonschema.validate(args, spec.parameters)
        except self._jsonschema.ValidationError as exc:
            # Re-raise as a normal ValueError so the dispatcher can
            # treat it like any other invalid-args failure.
            raise ValueError(exc.message) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Best-effort coercion to a JSON-serializable form.

    Numpy scalars + Path objects sneak into tool returns. Rather than
    audit every call site, normalise at the registry boundary.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # numpy scalars carry .item()
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def _compact(payload: dict) -> dict:
    """Trim arrays to length-only summaries before audit-logging."""
    out: dict = {}
    for k, v in payload.items():
        if isinstance(v, (list, tuple)) and len(v) > 8:
            out[k] = {"len": len(v), "head": list(v[:3])}
        elif isinstance(v, dict):
            out[k] = _compact(v)
        else:
            out[k] = v
    return out


__all__ = [
    "ToolSpec",
    "ToolContext",
    "ToolRegistry",
    "ToolDispatchResult",
]
