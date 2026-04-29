"""Line profile sampling tests (v2.0.6).

The interactive draw handlers depend on Dear PyGui's mouse API, so
the UI layer is tested via stubbed handler calls. The core — bilinear
sampling along an arbitrary line segment — is pure numpy/scipy and
deserves direct coverage.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _install_dpg_stub():
    # Even if an earlier test stubbed dearpygui, top up the attrs this
    # test needs (get_plot_mouse_pos, is_item_hovered, mouse handlers)
    # so the cross-test ordering doesn't flake.
    existing = sys.modules.get("dearpygui.dearpygui")
    if existing is not None:
        for name in ("add_mouse_click_handler", "add_mouse_release_handler",
                     "get_plot_mouse_pos", "is_item_hovered",
                     "mvMouseButton_Left"):
            if not hasattr(existing, name):
                setattr(
                    existing, name,
                    MagicMock(return_value=None)
                    if name != "mvMouseButton_Left" else 0,
                )
        if not hasattr(existing, "is_item_hovered"):
            existing.is_item_hovered = MagicMock(return_value=True)
        return
    dpg = types.ModuleType("dearpygui.dearpygui")
    for k in ["mvKey_O","mvKey_R","mvKey_K","mvKey_Slash","mvKey_Escape",
              "mvKey_LShift","mvKey_RShift","mvKey_LControl","mvKey_RControl",
              "mvKey_LSuper","mvKey_RSuper","mvKey_0","mvKey_1","mvKey_2",
              "mvKey_3","mvKey_4","mvXAxis","mvYAxis","mvAll",
              "mvThemeCol_WindowBg","mvMouseButton_Left"]:
        setattr(dpg, k, 0)
    for name in [
        "create_context","create_viewport","setup_dearpygui","show_viewport",
        "set_primary_window","is_dearpygui_running","render_dearpygui_frame",
        "destroy_context","stop_dearpygui","add_dynamic_texture","set_value",
        "get_value","configure_item","delete_item","add_menu_item","add_text",
        "add_button","add_combo","add_input_text","add_input_float",
        "add_input_int","add_checkbox","add_spacer","add_separator",
        "add_plot_axis","add_image_series","add_file_extension","bind_theme",
        "add_theme_color","add_theme_style","focus_item","show_item",
        "fit_axis_data","get_item_parent","get_item_label",
        "get_viewport_client_width","get_viewport_client_height",
        "set_viewport_drop_callback","set_viewport_resize_callback",
        "add_key_press_handler","add_item_clicked_handler",
        "bind_item_handler_registry","is_key_down","add_line_series",
        "add_mouse_click_handler","add_mouse_release_handler",
        "get_plot_mouse_pos","is_item_hovered",
    ]:
        setattr(dpg, name, MagicMock(return_value=None))
    dpg.does_item_exist = MagicMock(return_value=False)
    dpg.is_item_hovered = MagicMock(return_value=True)

    class _CM:
        def __enter__(self): return 0
        def __exit__(self, *a): return False
    dummy_cm = MagicMock(return_value=_CM())
    for name in ["theme","theme_component","texture_registry","window",
                 "child_window","group","menu","file_dialog",
                 "handler_registry","item_handler_registry",
                 "viewport_menu_bar","plot","tooltip","plot_axis",
                 "collapsing_header"]:
        setattr(dpg, name, dummy_cm)
    dpg.last_item = MagicMock(return_value="last")

    parent = types.ModuleType("dearpygui")
    parent.dearpygui = dpg
    sys.modules["dearpygui"] = parent
    sys.modules["dearpygui.dearpygui"] = dpg


_install_dpg_stub()

from ui2.app import DhmApp  # noqa: E402
import dearpygui.dearpygui as dpg  # noqa: E402


# ---------------------------------------------------------------------------
# Pure sampling — no UI needed
# ---------------------------------------------------------------------------

def test_horizontal_line_returns_matching_row():
    """Line from (0, H/2) to (W-1, H/2) should reproduce the centre row."""
    phase = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    row_idx = 32
    sampled = DhmApp._sample_line(phase, (0, row_idx), (63, row_idx), n=64)
    # Every step lands on an integer column so bilinear = exact row.
    expected = phase[row_idx].astype(np.float64)
    assert sampled.shape == expected.shape
    assert np.allclose(sampled, expected)


def test_vertical_line_returns_matching_column():
    phase = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    col_idx = 10
    sampled = DhmApp._sample_line(phase, (col_idx, 0), (col_idx, 63), n=64)
    expected = phase[:, col_idx].astype(np.float64)
    assert np.allclose(sampled, expected)


def test_diagonal_line_is_bilinear():
    """Diagonal across a linear ramp should produce a monotonic
    sequence — no NaNs, no negatives on a non-negative input."""
    phase = np.outer(np.linspace(0, 1, 128),
                     np.linspace(0, 1, 128)).astype(np.float32)
    sampled = DhmApp._sample_line(phase, (0, 0), (127, 127), n=256)
    assert np.all(np.isfinite(sampled))
    # Strictly non-decreasing along the diagonal (within numerical noise).
    diffs = np.diff(sampled)
    assert (diffs >= -1e-6).all()


def test_out_of_bounds_clamps_not_crashes():
    """Line endpoints outside the frame must clamp to the nearest
    edge — ``map_coordinates(mode="nearest")`` does that; we just
    verify there's no NaN or exception."""
    phase = np.ones((16, 16), dtype=np.float32)
    sampled = DhmApp._sample_line(phase, (-5, -5), (25, 25), n=32)
    assert np.all(np.isfinite(sampled))
    assert np.allclose(sampled, 1.0, atol=1e-6)


@pytest.mark.parametrize("n", [2, 10, 512])
def test_sample_count_matches_request(n):
    phase = np.zeros((32, 32), dtype=np.float32)
    sampled = DhmApp._sample_line(phase, (0, 0), (31, 31), n=n)
    assert sampled.shape == (n,)


# ---------------------------------------------------------------------------
# Mode toggle / handler gating
# ---------------------------------------------------------------------------

def _fake_app_with_recon(phase: np.ndarray | None = None) -> DhmApp:
    app = DhmApp.__new__(DhmApp)
    app._line_mode = False
    app._line_p1 = None
    app._line_p2 = None
    app._line_handlers_installed = True  # skip registry install
    app._set_status = MagicMock()
    if phase is None:
        phase = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    recon = MagicMock()
    recon.phase = phase
    app._last_recon = recon
    app.panel_phase = MagicMock(plot_tag="phase_plot_tag")
    app._render_line_profile_window = MagicMock()
    return app


def test_enable_line_mode_needs_a_recon():
    app = _fake_app_with_recon()
    app._last_recon = None
    app._enable_line_profile_mode()
    # Status should warn; line_mode stays off.
    assert app._line_mode is False
    app._set_status.assert_called()


def test_enable_line_mode_flips_state():
    app = _fake_app_with_recon()
    app._enable_line_profile_mode()
    assert app._line_mode is True
    assert app._line_p1 is None
    assert app._line_p2 is None


def test_mouse_down_ignored_when_mode_off():
    app = _fake_app_with_recon()
    app._line_mode = False
    app._on_phase_mouse_down(None, None)
    assert app._line_p1 is None


def test_full_click_release_samples_and_renders(monkeypatch):
    app = _fake_app_with_recon()
    app._enable_line_profile_mode()
    # Simulate the mouse coords DPG would give at each stage.
    monkeypatch.setattr(dpg, "get_plot_mouse_pos",
                        MagicMock(side_effect=[(0.0, 0.0), (10.0, 0.0)]))
    monkeypatch.setattr(dpg, "is_item_hovered",
                        MagicMock(return_value=True))
    app._on_phase_mouse_down(None, None)
    app._on_phase_mouse_release(None, None)
    assert app._line_mode is False   # mode auto-exits after gesture
    app._render_line_profile_window.assert_called_once()
    # Assert the rendered line's label encodes both endpoints.
    _, kwargs = app._render_line_profile_window.call_args
    label = kwargs.get("label", "")
    assert "0" in label and "10" in label


def test_identical_endpoints_show_warning(monkeypatch):
    app = _fake_app_with_recon()
    app._enable_line_profile_mode()
    monkeypatch.setattr(dpg, "get_plot_mouse_pos",
                        MagicMock(side_effect=[(3.0, 4.0), (3.0, 4.0)]))
    monkeypatch.setattr(dpg, "is_item_hovered",
                        MagicMock(return_value=True))
    app._on_phase_mouse_down(None, None)
    app._on_phase_mouse_release(None, None)
    app._render_line_profile_window.assert_not_called()
    # Status must surface the "too short" warning.
    warn_calls = [c for c in app._set_status.call_args_list
                  if c.kwargs.get("level") == "warn"]
    assert warn_calls
