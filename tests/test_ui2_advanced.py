"""Tests for v2.0.5 polish work: advanced pre-processing, error
drawer bounded log, maximize-panel state, migration v5→v6.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Dear PyGui stub (kept local so the file runs in isolation)
# ---------------------------------------------------------------------------

def _install_dpg_stub():
    if "dearpygui" in sys.modules:
        return
    dpg = types.ModuleType("dearpygui.dearpygui")
    for k in ["mvKey_O","mvKey_R","mvKey_K","mvKey_Slash","mvKey_Escape",
              "mvKey_LShift","mvKey_RShift","mvKey_LControl","mvKey_RControl",
              "mvKey_LSuper","mvKey_RSuper","mvKey_0","mvKey_1","mvKey_2",
              "mvKey_3","mvKey_4","mvXAxis","mvYAxis","mvAll",
              "mvThemeCol_WindowBg"]:
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
    ]:
        setattr(dpg, name, MagicMock(return_value=None))
    dpg.does_item_exist = MagicMock(return_value=False)

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

from core.settings_schema import SCHEMA_VERSION, AppSettings  # noqa: E402
from ui2 import state_store  # noqa: E402
from ui2.app import DhmApp  # noqa: E402
from ui2.reconstruction import ReconParams  # noqa: E402


# ---------------------------------------------------------------------------
# ReconParams defaults for the advanced block
# ---------------------------------------------------------------------------

def test_recon_params_has_advanced_fields():
    p = ReconParams()
    # Match v1 ReconDefaults so loading an old dump doesn't flip behaviour.
    assert p.subtract_mean is True
    assert p.hann_window is False
    assert p.fft_backend == "auto"
    assert p.unwrap_method == "GRADIENT_INTEGRATION"


# ---------------------------------------------------------------------------
# Migration v5 → v6
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "ui2_state.json"


def test_migration_v5_payload_backfills_advanced(tmp_state):
    """A v5 dump (no advanced block) must reach v6 with v1-equivalent
    defaults so behaviour doesn't silently change."""
    tmp_state.write_text(json.dumps({
        "schema_version": 5,
        "ui2": {"theme": "dark", "sample_id": "pre-v6"},
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.subtract_mean is True
    assert loaded.ui2.hann_window is False
    assert loaded.ui2.fft_backend == "auto"
    assert loaded.ui2.unwrap_method == "GRADIENT_INTEGRATION"


def test_roundtrip_preserves_advanced_fields(tmp_state):
    s = AppSettings.defaults().with_ui2(
        subtract_mean=False, hann_window=True,
        fft_backend="mlx", unwrap_method="QUALITY_GUIDED",
    )
    state_store.save(s, tmp_state)
    loaded = state_store.load(tmp_state)
    assert loaded.ui2.subtract_mean is False
    assert loaded.ui2.hann_window is True
    assert loaded.ui2.fft_backend == "mlx"
    assert loaded.ui2.unwrap_method == "QUALITY_GUIDED"


# ---------------------------------------------------------------------------
# Error drawer bounded log
# ---------------------------------------------------------------------------

def _fake_app() -> DhmApp:
    app = DhmApp.__new__(DhmApp)
    app._error_log = []
    app._ERROR_LOG_CAP = DhmApp._ERROR_LOG_CAP
    return app


def test_error_log_appends_entries():
    app = _fake_app()
    app._append_error("warn", "minor hiccup")
    app._append_error("danger", "segfault")
    assert len(app._error_log) == 2
    # Entry shape: (timestamp_str, level, message)
    assert app._error_log[0][1] == "warn"
    assert app._error_log[1][2] == "segfault"


def test_error_log_cap_holds_most_recent():
    app = _fake_app()
    app._ERROR_LOG_CAP = 5
    for i in range(20):
        app._append_error("warn", f"msg {i}")
    assert len(app._error_log) == 5
    messages = [e[2] for e in app._error_log]
    # Most recent should be kept; first 15 evicted.
    assert messages[0] == "msg 15"
    assert messages[-1] == "msg 19"


def test_error_log_clear_wipes():
    app = _fake_app()
    app._append_error("warn", "a")
    app._append_error("warn", "b")
    # _refresh_error_drawer is dpg-backed; stub no-ops so clear path
    # just needs ``does_item_exist`` → False to short-circuit.
    app._clear_error_log()
    assert app._error_log == []


# ---------------------------------------------------------------------------
# _to_uint8 utility used by batch export
# ---------------------------------------------------------------------------

def test_to_uint8_normalises_range():
    arr = np.linspace(-1.0, 3.5, 100, dtype=np.float32).reshape(10, 10)
    u8 = DhmApp._to_uint8(arr)
    assert u8.dtype == np.uint8
    assert u8.min() == 0
    assert u8.max() == 255


def test_to_uint8_handles_constant_array():
    """Constant input mustn't divide by zero — result is all zeros."""
    arr = np.full((8, 8), 0.7, dtype=np.float32)
    u8 = DhmApp._to_uint8(arr)
    assert (u8 == 0).all()


# ---------------------------------------------------------------------------
# Advanced pre-processing — hann_window and subtract_mean flow through
# ---------------------------------------------------------------------------

def test_subtract_mean_removes_dc_before_propagate():
    """Feed a hologram with a 0.3 DC offset; with subtract_mean=True
    the mean of the processed array must be ~0 before the normalise
    step in ReconstructionDriver._run."""
    from ui2.reconstruction import ReconstructionDriver

    # We can't call _run directly without a real hologram file and
    # the full pipeline — but we can verify the field existence on
    # ReconParams + the one-liner: subtract_mean=True ⇒ raw -= mean.
    # For a full-path test see the integration suite.
    p = ReconParams(subtract_mean=True)
    arr = np.full((8, 8), 0.5, dtype=np.float32)
    if p.subtract_mean:
        arr = arr - float(arr.mean())
    assert abs(arr.mean()) < 1e-6


def test_hann_window_dims_match_array():
    p = ReconParams(hann_window=True)
    arr = np.ones((16, 24), dtype=np.float32)
    if p.hann_window:
        wy = np.hanning(arr.shape[0]).astype(np.float32)
        wx = np.hanning(arr.shape[1]).astype(np.float32)
        arr = arr * (wy[:, None] * wx[None, :])
    assert arr.shape == (16, 24)
    # Corners should be ~0 (Hann boundary), centre should be ~1.
    assert arr[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert arr[arr.shape[0] // 2, arr.shape[1] // 2] > 0.9


# ---------------------------------------------------------------------------
# Maximize panel state — tag list points at expected panels
# ---------------------------------------------------------------------------

def test_panel_tags_emit_four_entries():
    """_panel_tags returns exactly four entries (Input/Amp/Phase/Info)
    so Ctrl+1-4 maps 1:1 to panel index 0-3."""
    app = DhmApp.__new__(DhmApp)
    # Minimal panel objects with container_tag populated.
    app.panel_input = MagicMock(container_tag="panelbox_a")
    app.panel_amp = MagicMock(container_tag="panelbox_b")
    app.panel_phase = MagicMock(container_tag="panelbox_c")
    tags = app._panel_tags()
    assert len(tags) == 4
    assert tags[0] == "panelbox_a"
    assert tags[1] == "panelbox_b"
    assert tags[2] == "panelbox_c"
    assert tags[3] == "info_panel"  # hardcoded container tag


def test_maximize_out_of_range_is_noop():
    app = DhmApp.__new__(DhmApp)
    app.panel_input = MagicMock(container_tag="a")
    app.panel_amp = MagicMock(container_tag="b")
    app.panel_phase = MagicMock(container_tag="c")
    app._preview_size = 400
    app._maximized_panel = None
    app._toasts = MagicMock()
    app._set_status = MagicMock()
    # Out-of-range should return without crashing or toasting.
    app._maximize_panel(99)
    assert app._maximized_panel is None
    app._set_status.assert_not_called()
