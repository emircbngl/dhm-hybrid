"""MCP-SDK-gated tests for the dhm-mcp FastMCP bridge.

These require the ``mcp`` package (importorskip'd — the project venv
deliberately doesn't ship it). Regression for the 2026-07-05 review
CRITICAL: FastMCP derives each tool's inputSchema from the handler's
*signature*, so the original ``**kwargs`` handlers collapsed every
bridged tool to one required "kwargs" field — every tool was uncallable
over the real protocol. The bridge now synthesizes an explicit
keyword-only signature from each ToolSpec's JSON schema.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

mcp_sdk = pytest.importorskip("mcp")

from dhm_mcp.headless import HeadlessSession  # noqa: E402
from dhm_mcp.server import build_mcp_server  # noqa: E402


def _write_carrier_hologram(path: Path, n: int = 192) -> None:
    import tifffile

    rng = np.random.default_rng(3)
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.4 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2)
                          / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * 0.25 * X + bump))
    holo = np.abs(1.0 + obj) ** 2 + 0.01 * rng.standard_normal((n, n))
    tifffile.imwrite(path, (holo * 1000).astype(np.uint16))


def _run(coro):
    return asyncio.run(coro)


def _open_session() -> HeadlessSession:
    """Session with the home-dir path guard relaxed — pytest tmp_path
    lives under /private/var, which the (correct) default rejects."""
    sess = HeadlessSession()
    sess.settings.restrict_to_home = False
    return sess


def test_bridged_tools_expose_real_schemas():
    """No tool may expose the degenerate {'kwargs': ...} schema; the
    real property names from the ToolSpec JSON schema must survive."""
    server = build_mcp_server(HeadlessSession())
    tools = _run(server.list_tools())
    by_name = {t.name: t for t in tools}

    assert "set_recon_param" in by_name
    schema = by_name["set_recon_param"].inputSchema
    props = schema.get("properties", {})
    assert "kwargs" not in props, "degenerate **kwargs schema leaked through"
    assert "z_mm" in props, f"expected real params, got {sorted(props)}"

    # Every bridged tool must be kwargs-free.
    offenders = [name for name, t in by_name.items()
                 if "kwargs" in t.inputSchema.get("properties", {})]
    assert not offenders, f"degenerate schemas: {offenders}"


def test_bridged_tool_callable_end_to_end(tmp_path):
    """The reviewer's exact repro: call a bridged tool the normal MCP
    way with plain arguments — pre-fix this raised
    'kwargs Field required'."""
    holo = tmp_path / "holo.tif"
    _write_carrier_hologram(holo)

    session = _open_session()
    server = build_mcp_server(session)

    out = _run(server.call_tool("load_hologram", {"path": str(holo)}))
    text = str(out)
    assert "error" not in text.lower() or "ok" in text.lower(), text

    out = _run(server.call_tool("set_recon_param", {"z_mm": 0.0,
                                                    "mask_radius": 40}))
    assert "z_mm" in str(out)

    out = _run(server.call_tool("run_reconstruction", {}))
    assert "error" not in str(out).lower(), str(out)

    # Zero-argument call must work too (pre-fix even {} failed).
    out = _run(server.call_tool("get_state", {}))
    assert out is not None


def test_render_view_returns_image_content(tmp_path, monkeypatch):
    """render_view over the bridge returns actual MCP image content."""
    import core.ai.tool_impls as ti
    monkeypatch.setattr(ti, "_RENDER_DIR", tmp_path / "renders")

    holo = tmp_path / "holo.tif"
    _write_carrier_hologram(holo)
    session = _open_session()
    server = build_mcp_server(session)
    _run(server.call_tool("load_hologram", {"path": str(holo)}))
    _run(server.call_tool("set_recon_param", {"z_mm": 0.0,
                                              "mask_radius": 40}))
    _run(server.call_tool("run_reconstruction", {}))

    out = _run(server.call_tool("render_view", {"kind": "phase"}))
    blocks = out[0] if isinstance(out, tuple) else out
    kinds = [getattr(b, "type", None) for b in blocks]
    assert "image" in kinds, f"no image content block, got {kinds}"
