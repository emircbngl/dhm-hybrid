"""``dhm_mcp`` headless session + registry e2e (Phase 2, no ``mcp`` needed).

Mirrors ``blender-optics-mcp``'s test shape: the MCP transport (needs
the ``mcp`` SDK) is exercised nowhere here — only the Qt-free
``HeadlessSession`` + the existing ``core.ai.tool_impls`` registry via
``ToolRegistry.dispatch``, exactly the surface ``server.py`` bridges.

Covers:
  * ``import dhm_mcp.headless`` with no ``mcp`` installed.
  * end-to-end: load_hologram -> set_recon_param -> run_reconstruction
    -> inspect_reconstruction (ok) -> render_view (PNG on disk)
    -> run_qpi.
  * set_reconstruction_mode('reference_free') then run_reconstruction
    still works (headless natively supports reffree, unlike the v1 GUI
    panel).
  * confirm-gate: a synthetic irreversible tool is rejected by
    dispatch's default-deny confirm.
  * ``server.main()`` exits cleanly (SystemExit) when ``mcp`` isn't
    installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ai.protocol import ToolCall  # noqa: E402
from core.ai.tool_impls import build_tool_registry  # noqa: E402
from core.ai.tools import ToolSpec  # noqa: E402
from core.audit import AuditLog  # noqa: E402
from core.errors import ErrorCenter  # noqa: E402
from dhm_mcp.headless import HeadlessSession  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic off-axis hologram (adapted from tests/test_reffree_pipeline.py's
# _carrier_hologram — same idiom: tilted reference + weak object bump).
# ---------------------------------------------------------------------------

def _carrier_hologram(n: int = 256, fc: float = 0.3, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.4 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * fc * X + bump))
    holo = np.abs(1.0 + obj) ** 2
    holo = holo + 0.01 * rng.standard_normal((n, n))
    return (holo * 1000).astype(np.uint16)


@pytest.fixture()
def holo_path(tmp_path: Path) -> Path:
    p = tmp_path / "holo.tif"
    tifffile.imwrite(str(p), _carrier_hologram())
    return p


def _session(tmp_path: Path) -> HeadlessSession:
    """A HeadlessSession with an isolated audit dir + a small mask
    radius (matches test_reffree_pipeline's 256-px synthetic image)
    and a tight autofocus search so the test stays fast.

    ``restrict_to_home`` is turned off — pytest's ``tmp_path`` lives
    under the system temp dir, outside ``$HOME``, and the security
    default (matching the in-app AI panel's default) would otherwise
    reject every synthetic hologram this test writes.
    """
    sess = HeadlessSession(audit=AuditLog(directory=tmp_path / "_audit"),
                           error_center=ErrorCenter())
    sess.recon_params.update({"mask_radius": 30})
    sess.settings.restrict_to_home = False
    return sess


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id="x", name=name, arguments_json=json.dumps(args))


# ---------------------------------------------------------------------------
# (a) import dhm_mcp.headless works with no mcp installed
# ---------------------------------------------------------------------------

def test_import_headless_without_mcp_installed():
    with pytest.raises(ModuleNotFoundError):
        import mcp  # noqa: F401
    import dhm_mcp.headless  # noqa: F401 - the point is this import succeeds


# ---------------------------------------------------------------------------
# (b) end-to-end pipeline through the real registry dispatch
# ---------------------------------------------------------------------------

def test_e2e_load_recon_inspect_render_qpi(tmp_path, holo_path):
    sess = _session(tmp_path)
    reg = build_tool_registry(include_devices=False)
    ctx = sess.build_context()

    out = reg.dispatch(ctx, _call("load_hologram", path=str(holo_path)))
    assert out.ok is True
    assert out.payload["shape"] == [256, 256]

    out = reg.dispatch(ctx, _call("set_recon_param", z_mm=0.0,
                                  wavelength_nm=632.8, pixel_um=0.7))
    assert out.ok is True

    out = reg.dispatch(ctx, _call("run_reconstruction"))
    assert out.ok is True, out.payload
    assert out.payload["submitted"] is True
    assert sess.recon_complex is not None
    assert sess.phase_unwrapped is not None

    out = reg.dispatch(ctx, _call("inspect_reconstruction"))
    assert out.ok is True, out.payload
    assert out.payload["ok"] is True
    assert out.payload["shape"] == [256, 256]
    # JSON-serializable, no numpy leaking through the registry boundary.
    json.dumps(out.payload)

    out = reg.dispatch(ctx, _call("render_view", kind="amplitude"))
    assert out.ok is True, out.payload
    png_path = out.payload["path"]
    with open(png_path, "rb") as fh:
        magic = fh.read(8)
    assert magic == b"\x89PNG\r\n\x1a\n"

    out = reg.dispatch(ctx, _call("run_qpi"))
    assert out.ok is True, out.payload
    assert "summary" in out.payload


def test_e2e_reference_free_mode_runs(tmp_path, holo_path):
    """Headless natively supports reference_free (unlike the v1 GUI
    panel, which reports it unsupported) — set_reconstruction_mode
    then run_reconstruction must both succeed."""
    sess = _session(tmp_path)
    reg = build_tool_registry(include_devices=False)
    ctx = sess.build_context()

    reg.dispatch(ctx, _call("load_hologram", path=str(holo_path)))

    out = reg.dispatch(ctx, _call("set_reconstruction_mode",
                                  mode="reference_free",
                                  bg_method="polynomial", bg_order=4))
    assert out.ok is True, out.payload
    assert out.payload["mode"] == "reference_free"

    out = reg.dispatch(ctx, _call("run_reconstruction"))
    assert out.ok is True, out.payload
    assert out.payload["summary"]["reference_mode"] == "reference_free"
    assert np.all(np.isfinite(sess.phase_unwrapped))


# ---------------------------------------------------------------------------
# (c) confirm-gate: irreversible tool is rejected by default
# ---------------------------------------------------------------------------

def test_irreversible_tool_rejected_by_default_confirm(tmp_path):
    sess = _session(tmp_path)
    reg = build_tool_registry(include_devices=False)

    reg.register(ToolSpec(
        name="_test_irreversible_noop",
        description="test-only irreversible tool",
        parameters={"type": "object", "properties": {},
                    "additionalProperties": False},
        handler=lambda ctx, args: {"ok": True, "did_it": True},
        irreversible=True,
    ))

    ctx = sess.build_context()
    out = reg.dispatch(ctx, _call("_test_irreversible_noop"))
    assert out.ok is False
    assert out.payload.get("error") == "user declined"


def test_irreversible_tool_allowed_via_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DHM_MCP_ALLOW_IRREVERSIBLE", "1")
    sess = _session(tmp_path)
    reg = build_tool_registry(include_devices=False)

    reg.register(ToolSpec(
        name="_test_irreversible_noop2",
        description="test-only irreversible tool",
        parameters={"type": "object", "properties": {},
                    "additionalProperties": False},
        handler=lambda ctx, args: {"ok": True, "did_it": True},
        irreversible=True,
    ))

    ctx = sess.build_context()
    out = reg.dispatch(ctx, _call("_test_irreversible_noop2"))
    assert out.ok is True
    assert out.payload["did_it"] is True


# ---------------------------------------------------------------------------
# (d) server.main() SystemExits cleanly when mcp isn't installed
# ---------------------------------------------------------------------------

def test_server_main_exits_without_mcp():
    from dhm_mcp import server

    with pytest.raises(SystemExit) as excinfo:
        server.main()
    assert "mcp" in str(excinfo.value).lower()


def test_build_mcp_server_raises_import_error_without_mcp():
    from dhm_mcp import server

    with pytest.raises(ImportError):
        server.build_mcp_server()
