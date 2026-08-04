"""Tests for the Phase 1b observation/vision tool wiring.

Covers the 5 new tools in ``core.ai.tool_impls`` that delegate to
``core.observe``: ``inspect_reconstruction``, ``inspect_phase``,
``inspect_field``, ``render_view``, ``set_reconstruction_mode``.

Uses the same fake-``ToolContext`` scaffolding pattern as
``test_ai_tools.py`` — a minimal synthetic context, no Qt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pytest

from core.ai.protocol import ToolCall
from core.ai.tool_impls import build_tool_registry
from core.ai.tools import ToolContext
from core.sample_map import SampleMap
from core.stage import MockStage


# ---------------------------------------------------------------------------
# Test doubles (slimmer copy of test_ai_tools._make_ctx, extended with the
# two new optional hooks)
# ---------------------------------------------------------------------------

@dataclass
class _Audit:
    records: list = field(default_factory=list)

    def record(self, action, params, result_summary=None):
        self.records.append((action, params, result_summary))


class _ErrorCenter:
    def __init__(self):
        self.events = []

    def error(self, **kwargs):
        self.events.append(kwargs)


class _Settings:
    restrict_to_home = False
    audit_redact_for_llm = True


class _Snapshot:
    def __init__(self, **kw):
        self.loaded_path = kw.get("loaded_path")
        self.loaded_shape = kw.get("loaded_shape")
        self.loaded_dtype = kw.get("loaded_dtype")
        self.recon_params = kw.get("recon_params", {})
        self.autofocus_params = kw.get("autofocus_params", {})
        self.qpi_params = kw.get("qpi_params", {})
        self.last_recon_summary = kw.get("last_recon_summary")
        self.last_af_summary = kw.get("last_af_summary")
        self.last_qpi_summary = kw.get("last_qpi_summary")
        self.last_depth_summary = kw.get("last_depth_summary")
        self.stage_position_mm = kw.get("stage_position_mm", (0.0, 0.0, 0.0))
        self.audit_tail = kw.get("audit_tail", ())

    def as_dict(self):
        return {"loaded_path": self.loaded_path}


def _synthetic_field(n: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n]
    r2 = ((x - n / 2) ** 2 + (y - n / 2) ** 2) / (2 * (n / 6) ** 2)
    amp = 0.3 + 0.7 * np.exp(-r2)
    phase = 2.5 * np.exp(-r2)
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _make_ctx(
    *,
    get_last_field=None,
    set_reference_mode=None,
    recon_params: Optional[dict] = None,
) -> ToolContext:
    snap = _Snapshot(recon_params=recon_params or {})

    return ToolContext(
        state=lambda: snap,
        last_recon_summary=lambda: None,
        last_af_summary=lambda: None,
        last_qpi_summary=lambda: None,
        last_depth_summary=lambda: None,
        audit_tail=lambda lim: [],
        load_hologram=lambda p: {"path": p},
        set_recon_param=lambda a: {"updated": a},
        invoke_recon=lambda a: {"submitted": True},
        invoke_autofocus=lambda a: {"submitted": True},
        invoke_qpi=lambda a: {"submitted": True},
        invoke_depth_map=lambda a: {"submitted": True},
        invoke_find_focus_candidates=lambda a: {"submitted": True},
        stage=MockStage(),
        capture_frame=lambda: np.ones((32, 32), dtype=np.float32),
        sample_map=SampleMap(),
        measure_sharpness=lambda f: 1.0,
        persist_sample_map=lambda: None,
        capture_and_process=lambda a: {"phase_std": 0.0},
        audit=_Audit(),
        error_center=_ErrorCenter(),
        confirm=lambda n, a: True,
        settings=_Settings(),
        get_last_field=get_last_field,
        set_reference_mode=set_reference_mode,
    )


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id="x", name=name, arguments_json=json.dumps(args))


# ---------------------------------------------------------------------------
# (e) registry has the 5 new tool names
# ---------------------------------------------------------------------------

def test_registry_has_observation_tools():
    r = build_tool_registry(include_devices=False)
    expected = {
        "inspect_reconstruction", "inspect_phase", "inspect_field",
        "render_view", "set_reconstruction_mode",
    }
    assert expected.issubset(set(r.names()))


# ---------------------------------------------------------------------------
# (a) get_last_field is None (hook not wired) -> error dict, no crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name,args", [
    ("inspect_reconstruction", {}),
    ("inspect_phase", {}),
    ("inspect_field", {}),
    ("render_view", {"kind": "amplitude"}),
])
def test_observation_tools_error_when_hook_not_wired(tool_name, args):
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=None)
    out = r.dispatch(ctx, _call(tool_name, **args))
    assert out.ok is False
    assert "error" in out.payload
    assert "not wired" in out.payload["error"]


def test_observation_tools_error_when_field_missing():
    """Hook is wired but returns None for the requested target."""
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: None)
    out = r.dispatch(ctx, _call("inspect_reconstruction"))
    assert out.ok is False
    assert "error" in out.payload


# ---------------------------------------------------------------------------
# (b) synthetic complex field -> inspect_reconstruction dispatch -> ok
# ---------------------------------------------------------------------------

def test_inspect_reconstruction_dispatch_ok_and_json_serializable():
    field = _synthetic_field()
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: field
                     if target == "recon_complex" else None)
    out = r.dispatch(ctx, _call("inspect_reconstruction"))
    assert out.ok is True
    assert out.payload["ok"] is True
    assert out.payload["shape"] == [64, 64]
    # JSON-friendly — no numpy scalars leak through the registry.
    json.dumps(out.payload)


def test_inspect_reconstruction_explicit_target():
    field = _synthetic_field()
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: field
                     if target == "recon_complex" else None)
    out = r.dispatch(ctx, _call("inspect_reconstruction", target="recon_complex"))
    assert out.ok is True


def test_inspect_phase_uses_recon_params_snapshot():
    phase = np.angle(_synthetic_field())
    r = build_tool_registry()
    ctx = _make_ctx(
        get_last_field=lambda target: phase if target == "phase_unwrapped" else None,
        recon_params={"wavelength_nm": 632.8, "pixel_um": 0.088},
    )
    out = r.dispatch(ctx, _call("inspect_phase", segment=False))
    assert out.ok is True
    assert out.payload["ok"] is True
    # OPD peak ~ phase_peak * wavelength / (2*pi); sanity check it's a
    # plausible nm-scale number, not something wildly off due to a
    # unit-conversion bug (nm vs m).
    assert 0 < out.payload["opd_nm"]["max"] < 1000


def test_inspect_phase_args_override_n_sample_n_medium():
    phase = np.angle(_synthetic_field())
    r = build_tool_registry()
    ctx = _make_ctx(
        get_last_field=lambda target: phase if target == "phase_unwrapped" else None,
    )
    out = r.dispatch(ctx, _call("inspect_phase", segment=False,
                                n_sample=1.35, n_medium=1.33))
    assert out.ok is True


def test_inspect_field_histograms_dispatch():
    field = _synthetic_field()
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: field
                     if target == "recon_complex" else None)
    out = r.dispatch(ctx, _call("inspect_field", bins=16))
    assert out.ok is True
    assert len(out.payload["amplitude_hist"]) == 16


def test_inspect_field_raw_target():
    raw = np.abs(_synthetic_field())
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: raw if target == "raw" else None)
    out = r.dispatch(ctx, _call("inspect_field", target="raw"))
    assert out.ok is True


# ---------------------------------------------------------------------------
# (c) render_view writes a PNG to a tmp _RENDER_DIR and returns the path
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_render_view_writes_png_to_tmp_render_dir(tmp_path, monkeypatch):
    import core.ai.tool_impls as tool_impls
    monkeypatch.setattr(tool_impls, "_RENDER_DIR", tmp_path / "renders")

    field = _synthetic_field()
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: field
                     if target == "recon_complex" else None)
    out = r.dispatch(ctx, _call("render_view", kind="amplitude"))
    assert out.ok is True
    assert out.payload["ok"] is True
    assert out.payload["kind"] == "amplitude"
    assert out.payload["shape"] == [64, 64]

    png_path = out.payload["path"]
    assert png_path.startswith(str(tmp_path))
    with open(png_path, "rb") as fh:
        magic = fh.read(8)
    assert magic == _PNG_MAGIC
    # No base64 / raw bytes leaked into the tool payload — the local
    # 8k-context model must not choke on an inline image.
    assert "base64" not in out.payload
    for v in out.payload.values():
        assert not (isinstance(v, str) and len(v) > 2000)


def test_render_view_depth_target(tmp_path, monkeypatch):
    import core.ai.tool_impls as tool_impls
    monkeypatch.setattr(tool_impls, "_RENDER_DIR", tmp_path / "renders")

    depth = np.random.default_rng(0).uniform(-2, 2, size=(32, 32))
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: depth if target == "depth" else None)
    out = r.dispatch(ctx, _call("render_view", kind="depth"))
    assert out.ok is True
    assert out.payload["kind"] == "depth"


def test_render_view_depth_missing_errors(tmp_path, monkeypatch):
    import core.ai.tool_impls as tool_impls
    monkeypatch.setattr(tool_impls, "_RENDER_DIR", tmp_path / "renders")

    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: None)
    out = r.dispatch(ctx, _call("render_view", kind="depth"))
    assert out.ok is False


def test_render_view_rejects_bad_kind_via_schema():
    r = build_tool_registry()
    ctx = _make_ctx(get_last_field=lambda target: _synthetic_field())
    out = r.dispatch(ctx, _call("render_view", kind="hologram"))
    assert out.ok is False


# ---------------------------------------------------------------------------
# (d) set_reconstruction_mode: delegate + None-hook error
# ---------------------------------------------------------------------------

def test_set_reconstruction_mode_delegates_to_hook():
    calls = []

    def hook(payload):
        calls.append(payload)
        return {"ok": True, **payload}

    r = build_tool_registry()
    ctx = _make_ctx(set_reference_mode=hook)
    out = r.dispatch(ctx, _call("set_reconstruction_mode", mode="reference"))
    assert out.ok is True
    assert calls[0]["mode"] == "reference"


def test_set_reconstruction_mode_passes_bg_params():
    calls = []

    def hook(payload):
        calls.append(payload)
        return {"ok": True}

    r = build_tool_registry()
    ctx = _make_ctx(set_reference_mode=hook)
    out = r.dispatch(ctx, _call(
        "set_reconstruction_mode", mode="reference_free",
        bg_method="zernike", bg_order=4,
    ))
    assert out.ok is True
    assert calls[0]["bg_method"] == "zernike"
    assert calls[0]["bg_order"] == 4


def test_set_reconstruction_mode_errors_when_hook_not_wired():
    r = build_tool_registry()
    ctx = _make_ctx(set_reference_mode=None)
    out = r.dispatch(ctx, _call("set_reconstruction_mode", mode="off"))
    assert out.ok is False
    assert "not wired" in out.payload["error"]
