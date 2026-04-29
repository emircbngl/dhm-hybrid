"""Pure-logic tests for the Dear PyGui app (no Dear PyGui context required).

These exercise the helpers on :class:`DhmApp` that don't touch the
render tree: viewport tier logic, the drop-zone copy, the shortcut
formatter, the info-text composer, and the workflow-section visibility
table. Dear PyGui is stubbed out at import time so we can import the
module without creating a GL context.
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


# ---------------------------------------------------------------------------
# Stub Dear PyGui before ui2 imports it
# ---------------------------------------------------------------------------

def _install_dpg_stub():
    """Make ``import dearpygui.dearpygui`` succeed with a no-op stub."""
    if "dearpygui" in sys.modules:
        return
    dpg = types.ModuleType("dearpygui.dearpygui")
    dpg.mvKey_O = 1
    dpg.mvKey_R = 2
    dpg.mvKey_K = 3
    dpg.mvKey_Slash = 4
    dpg.mvKey_Escape = 5
    dpg.mvKey_LShift = 6
    dpg.mvKey_RShift = 7
    dpg.mvKey_LControl = 8
    dpg.mvKey_RControl = 9
    dpg.mvKey_LSuper = 10
    dpg.mvKey_RSuper = 11
    dpg.mvXAxis = 100
    dpg.mvYAxis = 101
    dpg.mvAll = 0
    dpg.mvThemeCol_WindowBg = 0
    # Every method used by DhmApp — no-op.
    for name in (
        "create_context", "create_viewport", "setup_dearpygui",
        "show_viewport", "set_primary_window", "is_dearpygui_running",
        "render_dearpygui_frame", "destroy_context", "stop_dearpygui",
        "add_dynamic_texture", "set_value", "get_value",
        "configure_item", "delete_item", "add_menu_item",
        "add_text", "add_button", "add_combo", "add_input_text",
        "add_input_float", "add_input_int", "add_checkbox",
        "add_spacer", "add_separator", "add_plot_axis",
        "add_image_series", "add_file_extension", "bind_theme",
        "add_theme_color", "add_theme_style", "focus_item",
        "show_item", "fit_axis_data", "get_item_parent",
        "get_item_label", "get_viewport_client_width",
        "get_viewport_client_height", "set_viewport_drop_callback",
        "set_viewport_resize_callback", "add_key_press_handler",
        "add_item_clicked_handler", "bind_item_handler_registry",
        "is_key_down",
    ):
        setattr(dpg, name, MagicMock(return_value=None))
    dpg.does_item_exist = MagicMock(return_value=False)

    # Context managers (theme, texture_registry, window, group, etc.)
    class _CM:
        def __enter__(self_inner): return 0
        def __exit__(self_inner, *a): return False
    dummy_cm = MagicMock(return_value=_CM())
    for name in (
        "theme", "theme_component", "texture_registry", "window",
        "child_window", "group", "menu", "file_dialog",
        "handler_registry", "item_handler_registry", "viewport_menu_bar",
        "plot", "tooltip",
    ):
        setattr(dpg, name, dummy_cm)
    dpg.last_item = MagicMock(return_value="last")

    parent = types.ModuleType("dearpygui")
    parent.dearpygui = dpg
    sys.modules["dearpygui"] = parent
    sys.modules["dearpygui.dearpygui"] = dpg


_install_dpg_stub()

from ui2.app import DhmApp, _to_rgba  # noqa: E402
from ui2.reconstruction import ReconParams  # noqa: E402


# ---------------------------------------------------------------------------
# Viewport sizing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,expected_tier", [
    (1100, 288),
    (1249, 288),
    (1250, 384),
    (1499, 384),
    (1500, 512),
    (1920, 512),
    (2560, 512),
])
def test_tier_for_width_matches_invariant(width, expected_tier):
    assert DhmApp._tier_for_width(width) == expected_tier


def test_tier_invariant_no_overflow():
    """At every tier, the sidebar + both panels must fit in the width
    they were chosen for — this is the invariant we broke in v2.0.1."""
    for w in (1100, 1250, 1366, 1500, 1600, 1920, 2560):
        tier = DhmApp._tier_for_width(w)
        # sidebar 320 + 2 panels (tier + 16 each) must be ≤ tier-appropriate
        content_min = 320 + 2 * (tier + 16)
        # The chosen width bucket should comfortably fit content.
        # At the low end of a tier the margin is smallest; we don't
        # demand slack, only that the inequality holds.
        assert content_min <= max(w, 1100), (
            f"tier {tier} for width {w}: needs {content_min}px")


# ---------------------------------------------------------------------------
# Shortcut formatter — Mac vs elsewhere
# ---------------------------------------------------------------------------

def test_shortcut_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert DhmApp._shortcut("Ctrl+O") == "⌘O"
    assert DhmApp._shortcut("Ctrl+K") == "⌘K"


def test_shortcut_non_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert DhmApp._shortcut("Ctrl+O") == "Ctrl+O"


# ---------------------------------------------------------------------------
# Workflow section visibility table
# ---------------------------------------------------------------------------

def test_workflow_sections_cover_every_mode():
    modes = {"Reconstruct", "Analyse", "Report"}
    for tag, visible_for in DhmApp._WORKFLOW_SECTIONS.items():
        assert visible_for.issubset(modes), (
            f"section {tag} references unknown mode: "
            f"{visible_for - modes}")


def test_report_mode_hides_preset_and_reference():
    r = DhmApp._WORKFLOW_SECTIONS
    assert "Report" not in r["section_preset"]
    assert "Report" not in r["section_reference"]
    assert "Report" in r["section_sample"]


# ---------------------------------------------------------------------------
# Colormaps — the values stay in [0, 1] and depth is monotonic
# ---------------------------------------------------------------------------

def test_to_rgba_gray_in_range():
    arr = np.linspace(-5, 10, 64 * 64).reshape(64, 64).astype(np.float32)
    rgba = _to_rgba(arr, colormap="gray")
    assert rgba.shape == (64, 64, 4)
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0


def test_to_rgba_depth_is_monotonic_in_luminance():
    """Depth maps are physical z — colour should increase monotonically
    with z, not wrap like the phase wheel did in v2.0.1."""
    arr = np.linspace(0, 1, 512, dtype=np.float32)[None, :].repeat(4, 0)
    rgba = _to_rgba(arr, colormap="depth")
    luminance = (0.299 * rgba[..., 0]
                 + 0.587 * rgba[..., 1]
                 + 0.114 * rgba[..., 2])
    # Strictly non-decreasing along the x axis.
    diffs = np.diff(luminance, axis=1)
    assert (diffs >= -1e-6).all(), "depth ramp should be monotonic"


def test_to_rgba_phase_is_periodic():
    """Phase wheel *should* wrap — confirms we haven't accidentally
    collapsed the two colormaps into one."""
    arr = np.linspace(0, 1, 256, dtype=np.float32)[None, :].repeat(4, 0)
    rgba = _to_rgba(arr, colormap="phase")
    # Columns 0 and 255 are (0, 2π) on the wheel — the RGB triples
    # must agree within a small delta.
    first_rgb = rgba[0, 0, :3]
    last_rgb = rgba[0, -1, :3]
    assert np.allclose(first_rgb, last_rgb, atol=0.02), (
        f"phase wheel not periodic: start={first_rgb}, end={last_rgb}")


# ---------------------------------------------------------------------------
# Info text composer
# ---------------------------------------------------------------------------

def _app_with_fake_state():
    """Instantiate DhmApp without running the render loop."""
    # DhmApp.__init__ calls load_state() which reads STATE_PATH — in the
    # test environment the file may not exist, which is fine (defaults).
    app = DhmApp.__new__(DhmApp)
    # Populate the fields _compose_info_text actually reads.
    app._last_recon = None
    app._last_qpi = None
    app._last_qpi_batch = None
    app._last_depth = None
    app._current_hologram = None
    app._sample_id = ""
    app._params = ReconParams()
    return app


def test_compose_empty_state():
    app = _app_with_fake_state()
    assert app._compose_info_text() == "No reconstruction yet."


def test_compose_with_hologram_but_no_recon():
    app = _app_with_fake_state()
    app._current_hologram = Path("/tmp/sample.tif")
    txt = app._compose_info_text()
    assert "sample.tif" in txt
    # No recon → no Amplitude/Phase lines yet.
    assert "Amplitude" not in txt


def test_compose_with_reference_loaded():
    app = _app_with_fake_state()
    app._current_hologram = Path("/tmp/a.tif")
    app._params = ReconParams(reference_path=Path("/tmp/ref.tif"),
                              subtract_reference=True)
    txt = app._compose_info_text()
    assert "Reference: ref.tif" in txt
    assert "(on)" in txt


def test_compose_with_reference_armed_off():
    app = _app_with_fake_state()
    app._current_hologram = Path("/tmp/a.tif")
    app._params = ReconParams(reference_path=Path("/tmp/ref.tif"),
                              subtract_reference=False)
    txt = app._compose_info_text()
    assert "(loaded)" in txt


def test_compose_no_reference_says_none():
    app = _app_with_fake_state()
    app._current_hologram = Path("/tmp/a.tif")
    txt = app._compose_info_text()
    assert "Reference: (none)" in txt


def test_compose_with_recon_and_depth_then_cleared():
    """After clearing depth, the composed text must drop the depth line
    rather than keep the stale 'Depth: …' string around."""
    app = _app_with_fake_state()
    app._current_hologram = Path("/tmp/a.tif")

    recon = MagicMock()
    recon.input_image = np.zeros((16, 16), dtype=np.float32)
    recon.amplitude = np.ones((16, 16), dtype=np.float32)
    recon.phase = np.zeros((16, 16), dtype=np.float32)
    recon.offaxis_center = (8, 8)
    recon.runtime_ms = 12.0
    app._last_recon = recon

    depth = MagicMock()
    depth.result.z_map = np.linspace(-0.001, 0.002, 16 * 16).reshape(16, 16)
    depth.clusters = [object(), object()]
    depth.runtime_ms = 9.0
    app._last_depth = depth

    with_depth = app._compose_info_text()
    assert "Depth:" in with_depth

    # User clears depth.
    app._last_depth = None
    without_depth = app._compose_info_text()
    assert "Depth:" not in without_depth
    # Recon lines still present.
    assert "Amplitude" in without_depth


# ---------------------------------------------------------------------------
# Drop-zone label reflects capability
# ---------------------------------------------------------------------------

def test_drop_zone_label_supported():
    app = _app_with_fake_state()
    app._drop_supported = True
    txt = app._drop_zone_label()
    assert "Drop" in txt
    assert "click" in txt.lower()


def test_drop_zone_label_not_supported():
    app = _app_with_fake_state()
    app._drop_supported = False
    txt = app._drop_zone_label()
    assert "not supported" in txt.lower()
    assert "click" in txt.lower()


def test_status_ready_text_reflects_capability():
    app = _app_with_fake_state()
    app._drop_supported = True
    assert "drag" in app._status_ready_text().lower()
    app._drop_supported = False
    assert "drag" not in app._status_ready_text().lower()
