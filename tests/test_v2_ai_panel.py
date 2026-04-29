"""v2 AI panel port tests — pure-data + bridge layer.

The DPG window itself is render-only; we test the state machine
(``ui2.ai_panel_state``) and the bridge (``ui2.ai_bridge``) here.
DPG-side smoke runs the controller without touching real DPG.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# DPG stub top-up — same idempotent pattern as test_line_profile_dialog
# ---------------------------------------------------------------------------

def _install_dpg_stub():
    parent = sys.modules.get("dearpygui") or types.ModuleType("dearpygui")
    dpg = sys.modules.get("dearpygui.dearpygui") or types.ModuleType(
        "dearpygui.dearpygui",
    )
    for k in ["mvKey_O", "mvKey_R", "mvXAxis", "mvYAxis", "mvAll",
              "mvThemeCol_WindowBg"]:
        if not hasattr(dpg, k):
            setattr(dpg, k, 0)
    for name in [
        "create_context", "create_viewport", "setup_dearpygui",
        "show_viewport", "set_primary_window",
        "is_dearpygui_running", "render_dearpygui_frame",
        "destroy_context", "stop_dearpygui",
        "set_value", "get_value", "configure_item", "delete_item",
        "add_text", "add_button", "add_input_text",
        "add_separator", "add_spacer", "add_menu_item",
        "add_combo", "add_checkbox", "add_input_float",
        "add_input_int", "focus_item", "show_item",
        "add_plot_axis", "add_plot_legend", "add_image_series",
        "add_line_series", "add_table_column",
    ]:
        if not hasattr(dpg, name):
            setattr(dpg, name, MagicMock(return_value=None))
    if not hasattr(dpg, "does_item_exist"):
        dpg.does_item_exist = MagicMock(return_value=False)
    if not hasattr(dpg, "get_item_children"):
        dpg.get_item_children = MagicMock(return_value=[])

    class _CM:
        def __enter__(self): return 0
        def __exit__(self, *a): return False
    dummy_cm = MagicMock(return_value=_CM())
    for name in ["theme", "theme_component", "texture_registry",
                 "window", "child_window", "group", "menu",
                 "file_dialog", "handler_registry",
                 "item_handler_registry", "viewport_menu_bar",
                 "plot", "tooltip", "plot_axis",
                 "collapsing_header", "table", "table_row"]:
        if not hasattr(dpg, name):
            setattr(dpg, name, dummy_cm)

    parent.dearpygui = dpg
    sys.modules["dearpygui"] = parent
    sys.modules["dearpygui.dearpygui"] = dpg


_install_dpg_stub()


from ui2.ai_panel_state import (  # noqa: E402
    AIChatState, AIRunStatus, ChatEntry, ChatRole,
)


# ---------------------------------------------------------------------------
# AIChatState
# ---------------------------------------------------------------------------

def test_chat_state_starts_idle_empty():
    s = AIChatState()
    assert s.status is AIRunStatus.IDLE
    assert s.entries == []
    assert s.transcript() == ""


def test_push_user_then_assistant_renders_in_order():
    s = AIChatState()
    s.push(ChatRole.USER, "hi")
    s.push(ChatRole.ASSISTANT, "hello back")
    text = s.transcript()
    assert text.index("hi") < text.index("hello back")


def test_handle_event_assistant_text():
    """The state's event handler maps ``assistant_text`` →
    ASSISTANT entry."""
    class _E:
        kind = "assistant_text"
        text = "OK, loading…"
    s = AIChatState()
    s.handle_event(_E())
    assert s.entries[-1].role is ChatRole.ASSISTANT
    assert s.entries[-1].text == "OK, loading…"


def test_handle_event_tool_call_records_args():
    class _E:
        kind = "tool_call"
        name = "load_hologram"
        arguments = {"path": "/tmp/x.tif"}
    s = AIChatState()
    s.handle_event(_E())
    e = s.entries[-1]
    assert e.role is ChatRole.TOOL_CALL
    assert e.tool_name == "load_hologram"
    assert e.tool_args == {"path": "/tmp/x.tif"}


def test_handle_event_tool_result_truncates_huge_payload():
    """Result rendering should cap at ~400 chars so the chat box
    doesn't blow up."""
    class _E:
        kind = "tool_result"
        name = "find_focus_candidates"
        result = {"candidates": [{"z_mm": i, "score": float(i)}
                                  for i in range(50)]}
    s = AIChatState()
    s.handle_event(_E())
    rendered = s.entries[-1].render_block()
    assert "find_focus_candidates" in rendered
    # Truncated → ends with the ellipsis we apply.
    assert "…" in rendered


def test_handle_event_error_flips_status():
    class _E:
        kind = "error"
        message = "endpoint unreachable"
    s = AIChatState()
    s.status = AIRunStatus.RUNNING
    s.handle_event(_E())
    assert s.status is AIRunStatus.ERROR
    assert s.last_error == "endpoint unreachable"


def test_handle_event_done_returns_to_idle():
    s = AIChatState(status=AIRunStatus.RUNNING)

    class _E:
        kind = "done"
    s.handle_event(_E())
    assert s.status is AIRunStatus.IDLE


def test_handle_event_unknown_logs_system_row():
    """Unknown event kinds become a SYSTEM row rather than
    raising — agent evolution shouldn't break the panel."""
    class _E:
        kind = "novel_event"
    s = AIChatState()
    s.handle_event(_E())
    assert s.entries[-1].role is ChatRole.SYSTEM
    assert "novel_event" in s.entries[-1].text


def test_clear_resets_history():
    s = AIChatState()
    s.push(ChatRole.USER, "hi")
    s.last_error = "boom"
    s.status = AIRunStatus.ERROR
    s.clear()
    assert s.entries == []
    assert s.last_error is None
    assert s.status is AIRunStatus.IDLE


def test_transcript_caps_at_max_entries():
    s = AIChatState()
    for i in range(500):
        s.push(ChatRole.USER, f"msg {i}")
    text = s.transcript(max_entries=10)
    # Only the latest 10 land; earliest msg 0..489 dropped.
    assert "msg 0\n" not in text
    assert "msg 499" in text


# ---------------------------------------------------------------------------
# Bridge — ToolContext factory
# ---------------------------------------------------------------------------

def _stub_app(tmp_path):
    """Minimal DhmApp-shaped object for the bridge."""
    from core.settings_schema import AppSettings

    class _DriverStub:
        def submit(self, path, params, *,
                   sample_id, on_result, on_error):
            on_result(types.SimpleNamespace(
                phase=__import__("numpy").zeros((4, 4),
                                                dtype="float32"),
                z_m=0.012,
            ))

    class _ScienceStub:
        def run_autofocus(self, path, params, *,
                          z_min_mm, z_max_mm, n_steps,
                          sample_id, on_result, on_error):
            on_result(types.SimpleNamespace(
                best_z_m=0.014, score=1.5, scanned=20,
                runtime_ms=42.0,
            ))

        def run_qpi(self, path, params, *,
                    z_mm, sample_id, on_result, on_error):
            qpi = types.SimpleNamespace(
                phase_stats=types.SimpleNamespace(range_nm=412.5),
                total_dry_mass_pg=88.0, step_height_m=6.2e-7,
            )
            on_result(types.SimpleNamespace(qpi=qpi))

        def run_depth_map(self, path, params, *,
                          z_min_mm, z_max_mm, n_steps,
                          sample_id, on_result, on_error):
            on_result(types.SimpleNamespace(
                result=types.SimpleNamespace(
                    z_map=__import__("numpy").full((4, 4), 0.012,
                                                    dtype="float32"),
                ),
                clusters=[1, 2],
            ))

        def find_focus_candidates(self, path, params, *,
                                   z_min_mm, z_max_mm, n_steps,
                                   sample_id, on_result, on_error):
            on_result(types.SimpleNamespace(
                candidates=[
                    types.SimpleNamespace(
                        z_m=0.012, score=1.0, prominence=0.5,
                    ),
                ],
            ))

    app = types.SimpleNamespace()
    app._settings = AppSettings.defaults()
    app._params = types.SimpleNamespace(
        wavelength_nm=632.8, pixel_um=2.5, z_mm=12.0,
        n_medium=1.33, mask_radius=40, method="ASM",
        magnification=1.0, autofocus_metric="ENTROPY",
        af_z_min_mm=8.0, af_z_max_mm=16.0, af_n_steps=20,
    )
    app._current_hologram = tmp_path / "frame.tif"
    app._sample_id = "test"
    app._driver = _DriverStub()
    app._science = _ScienceStub()
    app._mark_dirty = MagicMock()

    def _load(path):
        app._current_hologram = path
    app._load_hologram = _load

    def _post_result(_):
        pass
    app._post_result = _post_result
    return app


def test_bridge_builds_tool_context(tmp_path):
    from ui2.ai_bridge import make_v2_tool_context
    app = _stub_app(tmp_path)
    ctx = make_v2_tool_context(app)
    assert ctx is not None
    # Mock devices wired by the bridge.
    assert ctx.shutter is not None
    assert ctx.led is not None


def test_bridge_invoke_recon_blocks_and_summarises(tmp_path):
    from ui2.ai_bridge import make_v2_tool_context
    # Touch the file so loading checks pass when AI calls it.
    (tmp_path / "frame.tif").write_bytes(b"")
    app = _stub_app(tmp_path)
    ctx = make_v2_tool_context(app)
    out = ctx.invoke_recon({})
    assert out.get("ok") is True
    assert out["shape"] == [4, 4]


def test_bridge_invoke_autofocus_summarises_z_mm(tmp_path):
    from ui2.ai_bridge import make_v2_tool_context
    (tmp_path / "frame.tif").write_bytes(b"")
    app = _stub_app(tmp_path)
    ctx = make_v2_tool_context(app)
    out = ctx.invoke_autofocus(
        {"z_min_mm": 8.0, "z_max_mm": 16.0, "n_steps": 20},
    )
    assert out["z_mm"] == pytest.approx(14.0)


def test_bridge_invoke_qpi_extracts_opd_and_dry_mass(tmp_path):
    from ui2.ai_bridge import make_v2_tool_context
    (tmp_path / "frame.tif").write_bytes(b"")
    app = _stub_app(tmp_path)
    ctx = make_v2_tool_context(app)
    out = ctx.invoke_qpi({})
    assert out["opd_range_nm"] == pytest.approx(412.5)
    assert out["total_dry_mass_pg"] == pytest.approx(88.0)


def test_bridge_load_hologram_missing_file(tmp_path):
    from ui2.ai_bridge import make_v2_tool_context
    app = _stub_app(tmp_path)
    ctx = make_v2_tool_context(app)
    out = ctx.load_hologram(str(tmp_path / "no_such.tif"))
    assert "error" in out


def test_bridge_legacy_stage_get_position(tmp_path):
    """The legacy ``stage_*`` AI tools call ``ctx.stage.get_position``;
    the v2 stub must satisfy that interface even though v2's real
    stage abstraction is in ``core.devices``."""
    from ui2.ai_bridge import make_v2_tool_context
    app = _stub_app(tmp_path)
    ctx = make_v2_tool_context(app)
    assert ctx.stage.get_position() == (0.0, 0.0, 0.0)
    ctx.stage.move_relative(1.0, 2.0, 3.0)
    assert ctx.stage.get_position() == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Controller — opens window without DPG (uses stub)
# ---------------------------------------------------------------------------

def test_controller_send_empty_prompt_is_noop(tmp_path):
    from ui2.ai_panel import AIPanelController
    app = _stub_app(tmp_path)
    ctrl = AIPanelController(app)
    ctrl.send("   ")
    # No history change.
    assert ctrl.state.entries == []


def test_controller_clear_resets_state(tmp_path):
    from ui2.ai_panel import AIPanelController
    app = _stub_app(tmp_path)
    ctrl = AIPanelController(app)
    ctrl.state.push(ChatRole.USER, "hi")
    ctrl._history.append(object())
    ctrl.clear()
    assert ctrl.state.entries == []
    assert ctrl._history == []


def test_show_ai_panel_idempotent(tmp_path):
    from ui2.ai_panel import show_ai_panel
    app = _stub_app(tmp_path)
    a = show_ai_panel(app)
    b = show_ai_panel(app)
    assert a is b
