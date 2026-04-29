"""Live vs File input-mode differentiation tests (v2.1.x, H6).

The lab's pilot review (2026-04-28): "live and file modes look
the same, operator can't tell which one they're working in."
This file pins:

* Default mode is ``"file"``.
* Camera start flips to ``"live"``; camera stop flips back.
* File load explicitly flips to ``"file"`` even when the camera
  was running (operator's deliberate action wins).
* Status prefix carries the active mode tag (``[FILE]`` /
  ``[● LIVE]``).
* Live frames cached to ``_latest_live_frame`` so the snapshot
  + reconstruct paths don't depend on disk I/O.
* `_snapshot_live_frame_to_tempfile` writes a TIFF round-trip-able
  by ``core.ingestion.load_any``.
* `_on_reconstruct` in live mode auto-snapshots if no file is
  loaded — the operator's intent ("reconstruct what I see") is
  honoured without a manual record/stop/load cycle.

DhmApp is constructed via ``__new__`` so we don't need a real
DPG context — every method we touch is pure-data or DPG-tolerant.
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
    if "dearpygui" in sys.modules:
        return
    dpg = types.ModuleType("dearpygui.dearpygui")
    for k in ["mvKey_O", "mvKey_R", "mvKey_K", "mvKey_Slash",
              "mvKey_Escape", "mvKey_LShift", "mvKey_RShift",
              "mvKey_LControl", "mvKey_RControl", "mvKey_LSuper",
              "mvKey_RSuper", "mvKey_0", "mvKey_1", "mvKey_2",
              "mvKey_3", "mvKey_4", "mvXAxis", "mvYAxis", "mvAll",
              "mvThemeCol_WindowBg"]:
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
        "add_plot_axis", "add_image_series", "add_file_extension",
        "bind_theme", "add_theme_color", "add_theme_style",
        "focus_item", "show_item", "fit_axis_data",
        "get_item_parent", "get_item_label",
        "get_viewport_client_width", "get_viewport_client_height",
        "set_viewport_drop_callback", "set_viewport_resize_callback",
        "add_key_press_handler", "add_item_clicked_handler",
        "bind_item_handler_registry", "is_key_down",
        "add_line_series",
    ]:
        setattr(dpg, name, MagicMock(return_value=None))
    dpg.does_item_exist = MagicMock(return_value=False)

    class _CM:
        def __enter__(self): return 0
        def __exit__(self, *a): return False
    dummy_cm = MagicMock(return_value=_CM())
    for name in ["theme", "theme_component", "texture_registry",
                 "window", "child_window", "group", "menu",
                 "file_dialog", "handler_registry",
                 "item_handler_registry", "viewport_menu_bar",
                 "plot", "tooltip", "plot_axis",
                 "collapsing_header"]:
        setattr(dpg, name, dummy_cm)
    dpg.last_item = MagicMock(return_value="last")

    parent = types.ModuleType("dearpygui")
    parent.dearpygui = dpg
    sys.modules["dearpygui"] = parent
    sys.modules["dearpygui.dearpygui"] = dpg


_install_dpg_stub()

from ui2.app import DhmApp  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture — bare DhmApp without DPG construction
# ---------------------------------------------------------------------------

def _make_bare_app():
    app = DhmApp.__new__(DhmApp)
    app._input_mode = "file"
    app._latest_live_frame = None
    app._current_hologram = None
    app._error_log = []
    app._set_status = MagicMock()
    app._toasts = MagicMock()
    return app


# ---------------------------------------------------------------------------
# Mode setter
# ---------------------------------------------------------------------------

def test_default_mode_is_file():
    app = _make_bare_app()
    assert app._input_mode == "file"


@pytest.mark.parametrize("mode", ["file", "live", "timelapse"])
def test_set_input_mode_accepts_valid_values(mode):
    app = _make_bare_app()
    app._set_input_mode(mode)
    assert app._input_mode == mode


def test_set_input_mode_falls_back_to_file_on_unknown():
    app = _make_bare_app()
    app._set_input_mode("bogus")
    assert app._input_mode == "file"


def test_mode_prefix_table_covers_known_modes():
    """The status-line prefix table must have an entry for every
    mode value the setter accepts. Catches a typo if someone adds
    a new mode without registering its prefix."""
    for mode in ("file", "live", "timelapse"):
        assert mode in DhmApp._MODE_PREFIX
        assert DhmApp._MODE_PREFIX[mode]


# ---------------------------------------------------------------------------
# _set_status prefix
# ---------------------------------------------------------------------------

def test_set_status_prefixes_with_mode_tag(monkeypatch):
    """``_set_status('hello')`` should produce a string that
    starts with the active mode's prefix. We capture the
    forwarded value via ``dpg.set_value`` mocking."""
    import dearpygui.dearpygui as dpg
    captured = {}

    def _capture(tag, value):
        captured.setdefault(tag, []).append(value)

    monkeypatch.setattr(dpg, "set_value", _capture)
    monkeypatch.setattr(dpg, "configure_item",
                        MagicMock(return_value=None))

    app = _make_bare_app()
    # Bypass the mocked _set_status from _make_bare_app — we want
    # to exercise the real implementation.
    real_set_status = DhmApp._set_status.__get__(app, DhmApp)
    real_set_status("Reconstructing…", level="info")
    assert any(v.startswith("[FILE]") for v in captured["status_text"])

    app._set_input_mode = DhmApp._set_input_mode.__get__(app, DhmApp)
    app._set_input_mode("live")
    real_set_status("Camera frame received", level="info")
    assert any(v.startswith("[● LIVE]") for v in captured["status_text"])


# ---------------------------------------------------------------------------
# Latest live frame cache
# ---------------------------------------------------------------------------

def test_latest_live_frame_starts_none():
    app = _make_bare_app()
    assert app._latest_live_frame is None


def test_handle_camera_frame_caches_a_copy():
    """Each incoming live frame should land in the cache as a
    *copy* — caller mutating its buffer must not corrupt our
    snapshot."""
    app = _make_bare_app()
    # Stub out the panel.set_image call.
    app.panel_input = MagicMock()
    real_handle = DhmApp._handle_camera_frame.__get__(app, DhmApp)
    src = np.random.rand(32, 32).astype(np.float32)
    real_handle(src)
    cached = app._latest_live_frame
    assert cached is not None
    assert cached.shape == src.shape
    # Mutate source — cache must not flip.
    src[0, 0] = 999.0
    assert cached[0, 0] != 999.0


# ---------------------------------------------------------------------------
# Snapshot to tempfile
# ---------------------------------------------------------------------------

def test_snapshot_returns_none_when_no_live_frame_yet():
    app = _make_bare_app()
    assert app._latest_live_frame is None
    out = DhmApp._snapshot_live_frame_to_tempfile.__get__(
        app, DhmApp,
    )()
    assert out is None


def test_snapshot_writes_tiff_round_trip(tmp_path, monkeypatch):
    """Snapshot a known frame, confirm the resulting TIFF can be
    read back and equals the input within uint16 quantisation."""
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    app = _make_bare_app()
    rng = np.random.default_rng(0)
    frame = rng.random((48, 48)).astype(np.float32)
    app._latest_live_frame = frame
    snap = DhmApp._snapshot_live_frame_to_tempfile.__get__(
        app, DhmApp,
    )()
    assert snap is not None
    assert snap.exists()
    # Path under our tmp_path so monkeypatch took effect.
    assert tmp_path in snap.parents
    import tifffile
    rec = tifffile.imread(str(snap))
    # uint16 round-trip: < 1/65535 max relative error.
    rec_f32 = rec.astype(np.float32) / 65535.0
    assert np.allclose(rec_f32, frame, atol=2.0 / 65535.0)


# ---------------------------------------------------------------------------
# Mode transitions
# ---------------------------------------------------------------------------

def test_loading_hologram_flips_to_file_mode(tmp_path):
    """``_load_hologram`` is the operator's deliberate file pick.
    Even if the camera was streaming, this action should flip the
    mode to ``file`` so the prefix matches what the user sees."""
    app = _make_bare_app()
    app._set_input_mode = DhmApp._set_input_mode.__get__(app, DhmApp)
    app._input_mode = "live"
    p = tmp_path / "hh.tif"
    p.write_bytes(b"\x00")  # exists()
    # Stub out the rest of _load_hologram body that needs DPG / IO.
    # We just need to verify the mode flip happens early.
    app._refresh_reconstruct_tooltip = MagicMock()
    app._mark_dirty = MagicMock()
    app._last_recon = None
    app._last_qpi = None
    app._last_qpi_batch = None
    app._last_depth = None
    app._sample_id = ""
    app._recent = []
    app.panel_input = MagicMock()
    app._push_texture = MagicMock()
    app._tier_for_size = MagicMock(return_value=384)
    # Trigger.
    real_load = DhmApp._load_hologram.__get__(app, DhmApp)
    try:
        real_load(p)
    except Exception:
        # The full body touches a lot of UI; we only care that
        # the mode flip happened before the exception.
        pass
    assert app._input_mode == "file"


def test_camera_stop_reverts_to_file_mode():
    """After camera stop the prefix should not lie that we're
    still in live mode. Empty-history operator (no file ever
    loaded) → mode goes back to ``file`` so status reads truthful
    rather than perpetually live."""
    app = _make_bare_app()
    app._set_input_mode = DhmApp._set_input_mode.__get__(app, DhmApp)
    app._input_mode = "live"
    app._camera_thread = MagicMock(is_alive=lambda: False)
    app._camera_recorder = None
    app._refresh_reconstruct_tooltip = MagicMock()
    real_stop = DhmApp._on_camera_stop.__get__(app, DhmApp)
    real_stop()
    assert app._input_mode == "file"


# ---------------------------------------------------------------------------
# Reconstruct uses snapshot in live mode
# ---------------------------------------------------------------------------

def test_reconstruct_in_live_mode_with_no_file_uses_snapshot(
    tmp_path, monkeypatch,
):
    """When in live mode, no file loaded, but a live frame has
    arrived, ``_on_reconstruct`` should snapshot it and submit
    that path. Operator presses Reconstruct → it just works."""
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    app = _make_bare_app()
    app._set_input_mode = DhmApp._set_input_mode.__get__(app, DhmApp)
    app._snapshot_live_frame_to_tempfile = (
        DhmApp._snapshot_live_frame_to_tempfile.__get__(app, DhmApp)
    )
    app._input_mode = "live"
    app._latest_live_frame = np.random.rand(32, 32).astype(np.float32)
    submitted: dict = {}

    class _Driver:
        def submit(self, path, params, *, sample_id,
                   on_result, on_error):
            submitted["path"] = path
    app._driver = _Driver()
    app._on_param_changed = MagicMock()
    app._refresh_reconstruct_tooltip = MagicMock()
    app._params = MagicMock()
    app._sample_id = ""
    app._post_result = MagicMock()
    app._post_error = MagicMock()

    real_recon = DhmApp._on_reconstruct.__get__(app, DhmApp)
    real_recon()
    assert "path" in submitted
    assert submitted["path"].exists()
    assert submitted["path"].suffix.lower() == ".tif"


def test_reconstruct_in_file_mode_keeps_existing_path(tmp_path):
    """In file mode with a hologram already loaded, reconstruct
    must NOT touch the snapshot path. Catches a regression
    where the live branch fired even when a file was loaded."""
    app = _make_bare_app()
    app._set_input_mode = DhmApp._set_input_mode.__get__(app, DhmApp)
    app._snapshot_live_frame_to_tempfile = MagicMock(
        side_effect=AssertionError("should not be called"),
    )
    app._input_mode = "file"
    app._current_hologram = tmp_path / "h.tif"
    submitted: dict = {}

    class _Driver:
        def submit(self, path, params, *, sample_id,
                   on_result, on_error):
            submitted["path"] = path
    app._driver = _Driver()
    app._on_param_changed = MagicMock()
    app._refresh_reconstruct_tooltip = MagicMock()
    app._params = MagicMock()
    app._sample_id = ""
    app._post_result = MagicMock()
    app._post_error = MagicMock()

    real_recon = DhmApp._on_reconstruct.__get__(app, DhmApp)
    real_recon()
    assert submitted.get("path") == app._current_hologram
