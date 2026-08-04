"""``dhm-mcp`` — MCP transport bridging the DHM AI tool registry.

Generic bridge: every :class:`~core.ai.tools.ToolSpec` in
``core.ai.tool_impls.build_tool_registry()`` becomes one MCP tool with
the same name/description/JSON-schema. Calls run through
``ToolRegistry.dispatch`` — never the raw handler — so schema
validation, numeric clamping (``core.ai.security.clamp``), the
confirm-gate for irreversible tools, and audit logging all stay in
effect exactly as they do for the in-app copilot.

``render_view`` gets special treatment: when the dispatch payload is
``{"ok": True, "path": ..., ...}`` we additionally read the PNG off
disk and return it as MCP image content (a vision-capable client, e.g.
Claude, can then literally look at the reconstruction) alongside the
text summary.

Importing this module is always safe (no ``mcp`` dependency at import
time — see ``_load_fastmcp``). Only ``main()`` requires the SDK; it
exits with a clear, actionable message if it's missing rather than
letting an ``ImportError`` traceback leak out.

Install: ``pip install "mcp[cli]"`` (kept out of requirements.txt —
this is an optional, separate entry point; see ``__main__.py``).
"""
from __future__ import annotations

import base64
import inspect
import json
import logging
import sys
from typing import Any, Optional

from core.ai.protocol import ToolCall
from core.ai.tool_impls import build_tool_registry
from core.ai.tools import ToolContext, ToolRegistry
from dhm_mcp.headless import HeadlessSession

_LOG = logging.getLogger(__name__)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_ABOUT_TEXT = """\
dhm-mcp — headless MCP server for the DHM Reconstruction app.

Pipeline (chain tools in this order for a fresh hologram):
  load_hologram -> set_recon_param -> run_reconstruction
    -> run_autofocus / find_focus_candidates -> run_qpi / compute_depth_map

Read-before-write:
  * get_state           - current params, last-result summaries, stage position.
  * get_last_result      - scalars for one stage ('recon' | 'autofocus' | 'qpi' | 'depth').
  * get_audit_tail       - recent actions, for "what did I just do?".

Vision / observation (no picture needed for the first two):
  * inspect_reconstruction - focus/contrast/phase-gradient verdict on the latest recon.
  * inspect_phase          - OPD stats + optional cell segmentation + dry mass.
  * inspect_field          - amplitude/phase histograms, dynamic range.
  * render_view            - PNG of amplitude/phase/spectrum/depth/raw; this server
                              returns it as MCP image content, so a vision-capable
                              client (you) can literally look at the field.

Reference handling:
  * set_reconstruction_mode - 'off' | 'reference' | 'reference_free' (background-fit,
                               no reference hologram needed — this server supports it
                               natively).

Everything else (stage_*, list_devices, shutter_*, led_*, acquire_grid,
map_sample_grid, goto_cell, record_timelapse) mirrors the in-app AI
copilot's tool set 1:1 — same names, same JSON schemas, same
guardrails (core.ai.security: path/numeric validation; irreversible
tools require confirmation, which this headless server denies by
default — set DHM_MCP_ALLOW_IRREVERSIBLE=1 to auto-approve).
"""


def _load_fastmcp():
    """Lazy import of the MCP SDK. Raises ImportError if missing —
    callers (``main``) turn that into a friendly ``SystemExit``."""
    from mcp.server.fastmcp import FastMCP  # noqa: WPS433
    return FastMCP


def _load_image_type():
    """Return ``mcp.server.fastmcp.Image`` if the installed SDK has
    it, else ``None`` (caller falls back to base64 ImageContent)."""
    try:
        from mcp.server.fastmcp import Image  # noqa: WPS433
        return Image
    except ImportError:
        return None


def build_mcp_server(session: Optional[HeadlessSession] = None):
    """Construct the ``FastMCP`` instance with every registry tool
    bridged in, plus the static ``about`` tool.

    Raises ``ImportError`` if ``mcp`` isn't installed — callers that
    want the honest-SystemExit behaviour should catch it (``main``
    does this; tests that only exercise the registry/session never
    call this function at all, so they never need ``mcp`` installed).
    """
    FastMCP = _load_fastmcp()
    Image = _load_image_type()

    sess = session if session is not None else HeadlessSession()
    registry: ToolRegistry = build_tool_registry(include_devices=True)

    mcp = FastMCP("dhm-mcp")

    def _make_handler(tool_name: str, parameters: dict):
        def _handler(**kwargs: Any):
            # Drop optional args the client left at the None sentinel —
            # the DHM schemas are additionalProperties:false and don't
            # accept nulls for absent-means-default parameters.
            required = set(parameters.get("required", []))
            args = {k: v for k, v in kwargs.items()
                    if v is not None or k in required}
            ctx: ToolContext = sess.build_context()
            call = ToolCall(id=tool_name, name=tool_name,
                            arguments_json=json.dumps(args))
            result = registry.dispatch(ctx, call)
            payload = result.payload

            if tool_name == "render_view" and isinstance(payload, dict) \
                    and payload.get("ok") and "path" in payload:
                return _render_view_response(payload, Image)

            return payload

        # FastMCP derives each tool's inputSchema by introspecting the
        # handler's *signature* — a bare ``**kwargs`` collapses every
        # tool to one required "kwargs" field and makes the whole
        # bridge uncallable over the real protocol (2026-07-05 review,
        # CRITICAL). Synthesize an explicit keyword-only signature from
        # the ToolSpec's JSON schema instead; strict validation still
        # happens inside registry.dispatch.
        _handler.__signature__ = _signature_from_schema(parameters)
        _handler.__annotations__ = _annotations_from_schema(parameters)
        _handler.__name__ = f"dhm_tool_{tool_name}"
        return _handler

    for spec in registry:
        mcp.tool(name=spec.name, description=spec.description)(
            _make_handler(spec.name, spec.parameters or {}),
        )

    @mcp.tool(name="about",
              description="Read this first: pipeline order, tool "
                          "categories, and reference-mode notes for "
                          "the dhm-mcp server.")
    def _about() -> str:  # noqa: WPS430 - small closure, mirrors sibling project
        return _ABOUT_TEXT

    return mcp


# JSON-schema "type" → Python annotation for the synthesized handler
# signatures. Loose on purpose: the DHM registry re-validates strictly
# against the full schema (enums, ranges, additionalProperties) — the
# MCP-side schema only needs the right field names and rough types.
_JSON_TO_PY = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _annotation_for(sub_schema: Any) -> Any:
    if isinstance(sub_schema, dict):
        return _JSON_TO_PY.get(sub_schema.get("type"), Any)
    return Any


def _signature_from_schema(parameters: dict) -> inspect.Signature:
    """Keyword-only Signature mirroring a JSON schema's properties.

    Required properties get no default; optional ones default to None
    (the handler strips None before dispatch, so "absent" round-trips
    correctly)."""
    props = parameters.get("properties", {}) if parameters else {}
    required = set(parameters.get("required", [])) if parameters else set()
    params = []
    for name, sub in props.items():
        ann = _annotation_for(sub)
        if name in required:
            params.append(inspect.Parameter(
                name, kind=inspect.Parameter.KEYWORD_ONLY, annotation=ann))
        else:
            params.append(inspect.Parameter(
                name, kind=inspect.Parameter.KEYWORD_ONLY,
                default=None, annotation=Optional[ann]))
    return inspect.Signature(params)


def _annotations_from_schema(parameters: dict) -> dict:
    props = parameters.get("properties", {}) if parameters else {}
    required = set(parameters.get("required", [])) if parameters else set()
    out = {}
    for name, sub in props.items():
        ann = _annotation_for(sub)
        out[name] = ann if name in required else Optional[ann]
    return out


def _render_view_response(payload: dict, image_type):
    """Turn a successful ``render_view`` dispatch payload into MCP
    content: the PNG bytes (as ``Image`` when the SDK has that
    convenience type, else a raw base64 content block) plus the text
    summary so the client sees both the picture and the scalars."""
    try:
        with open(payload["path"], "rb") as fh:
            png_bytes = fh.read()
    except OSError as exc:
        return {**payload, "read_error": str(exc)}

    if png_bytes[:8] != _PNG_MAGIC:
        return {**payload, "read_error": "file at path is not a PNG"}

    summary_text = json.dumps({k: v for k, v in payload.items() if k != "path"})

    if image_type is not None:
        return [image_type(data=png_bytes, format="png"), summary_text]

    # Fallback: base64 image content block (older SDKs without the
    # ``Image`` convenience wrapper).
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return [
        {"type": "image", "data": b64, "mimeType": "image/png"},
        summary_text,
    ]


def main() -> None:
    """Entry point (``python -m dhm_mcp`` / console script).

    Honest failure: if ``mcp`` isn't installed, exit with a clear
    message rather than an ImportError traceback. This check happens
    at call time, not at module import time, so ``import
    dhm_mcp.server`` (and ``import dhm_mcp.headless``) stay safe in a
    minimal test environment without the SDK.
    """
    try:
        server = build_mcp_server()
    except ImportError as exc:
        raise SystemExit(
            'dhm-mcp requires: pip install "mcp[cli]" '
            f"(import failed: {exc})",
        ) from exc

    server.run()


if __name__ == "__main__":  # pragma: no cover - exercised via __main__.py
    main()


__all__ = ["build_mcp_server", "main"]
