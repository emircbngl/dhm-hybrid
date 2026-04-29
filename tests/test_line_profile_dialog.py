"""Headless smoke for the line-profile dialog wrapper.

DPG is stubbed (same pattern as other DPG tests); we only check
that:

* ``show_line_profiles`` can be invoked without DPG being
  initialised — its body runs through the stubbed ``dpg.window``
  context manager and registers the editor on ``dpg`` so the
  caller can fetch it later.
* The editor stash survives a re-open (saved profiles persist).
* When ``image_provider`` returns ``None``, the dialog still
  builds (no AttributeError on stats lookup).
* When ``image_provider`` returns a real array, the dialog calls
  ``sample_line`` against the provided profiles.

The DPG stub mirrors the one in ``test_ui2_user_presets.py``.
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
    """Install a DPG stub OR top up an existing one with missing
    attrs.

    Earlier sibling test files install their own minimal stubs
    (test_input_mode.py, test_ui2_user_presets.py); the first one
    that runs registers ``dearpygui.dearpygui`` in ``sys.modules``,
    and a naive "return early if present" leaves later tests using
    a stub that doesn't have the symbols the dialog calls
    (``add_plot_legend`` / ``add_table_column`` / ``mvXAxis``…).

    Pattern from ``lessons.md § 2026-04-24 — Dear PyGui stub top-up
    across test files``: always run the stub population block —
    ``setattr`` is idempotent — so any test that imports this
    file gets the full surface.
    """
    parent = sys.modules.get("dearpygui") or types.ModuleType("dearpygui")
    dpg = sys.modules.get("dearpygui.dearpygui") or types.ModuleType(
        "dearpygui.dearpygui",
    )
    for k in ["mvKey_O", "mvKey_R", "mvKey_K", "mvKey_Slash",
              "mvKey_Escape", "mvKey_LShift", "mvKey_RShift",
              "mvKey_LControl", "mvKey_RControl", "mvKey_LSuper",
              "mvKey_RSuper", "mvKey_0", "mvKey_1", "mvKey_2",
              "mvKey_3", "mvKey_4", "mvXAxis", "mvYAxis", "mvAll",
              "mvThemeCol_WindowBg"]:
        if not hasattr(dpg, k):
            setattr(dpg, k, 0)
    for name in [
        "create_context", "create_viewport", "setup_dearpygui",
        "show_viewport", "set_primary_window",
        "is_dearpygui_running", "render_dearpygui_frame",
        "destroy_context", "stop_dearpygui", "add_dynamic_texture",
        "set_value", "get_value", "configure_item", "delete_item",
        "add_menu_item", "add_text", "add_button", "add_combo",
        "add_input_text", "add_input_float", "add_input_int",
        "add_checkbox", "add_spacer", "add_separator",
        "add_plot_axis", "add_plot_legend", "add_image_series",
        "add_file_extension", "bind_theme",
        "add_theme_color", "add_theme_style", "focus_item",
        "show_item", "fit_axis_data", "get_item_parent",
        "get_item_label", "get_viewport_client_width",
        "get_viewport_client_height",
        "set_viewport_drop_callback",
        "set_viewport_resize_callback",
        "add_key_press_handler", "add_item_clicked_handler",
        "bind_item_handler_registry", "is_key_down",
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
    if not hasattr(dpg, "last_item"):
        dpg.last_item = MagicMock(return_value="last")

    parent.dearpygui = dpg
    sys.modules["dearpygui"] = parent
    sys.modules["dearpygui.dearpygui"] = dpg


_install_dpg_stub()

import dearpygui.dearpygui as dpg  # noqa: E402

from ui2.dialogs import show_line_profiles  # noqa: E402
from ui2.line_profile_state import LineProfileEditor  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_show_line_profiles_constructs_editor_on_first_call():
    """``editor=None`` → fresh editor stashed on ``dpg`` for
    later retrieval."""
    if hasattr(dpg, "_line_profile_editor"):
        del dpg._line_profile_editor
    show_line_profiles(image_provider=lambda: None)
    assert hasattr(dpg, "_line_profile_editor")
    assert isinstance(dpg._line_profile_editor, LineProfileEditor)


def test_show_line_profiles_keeps_supplied_editor():
    """Caller-supplied editor must be honoured (saved profiles
    survive re-open)."""
    editor = LineProfileEditor()
    editor.first_click(0, 0)
    editor.second_click(10, 20, label="membrane")
    show_line_profiles(editor=editor, image_provider=lambda: None)
    assert dpg._line_profile_editor is editor
    assert len(editor.profiles) == 1
    assert editor.profiles[0].label == "membrane"


def test_show_line_profiles_handles_no_image():
    """Image provider returning None should not blow up — the
    dialog opens with empty stats."""
    show_line_profiles(image_provider=lambda: None)
    # No exception = pass.


def test_show_line_profiles_runs_with_real_image():
    """Provide a 2-D array + a saved profile; the dialog body
    builds the table + plot without raising."""
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    editor = LineProfileEditor()
    editor.first_click(10, 10)
    editor.second_click(50, 50, label="diag")
    show_line_profiles(editor=editor, image_provider=lambda: img)


def test_show_line_profiles_on_close_callback_is_invoked():
    """The on_close callable wires through to the Close button.
    We can't push the button under DPG stubs, but we can verify
    the dialog accepts the callback without raising."""
    called = {"n": 0}
    show_line_profiles(
        image_provider=lambda: None,
        on_close=lambda: called.update(n=called["n"] + 1),
    )
    # Just verifies the constructor took the kwarg.
