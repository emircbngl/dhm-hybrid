"""User-defined preset persistence tests (v2.0.6).

The feature has three moving parts:

* ``Ui2State.user_presets: dict[str, dict]`` — the on-disk store.
* ``_v6_to_v7`` migrator backfills an empty dict for older dumps.
* ``DhmApp._presets()`` merges built-ins + user presets; save path
  rejects collisions with built-ins, delete path refuses built-ins.

Tests verify each layer without launching Dear PyGui — the DhmApp
methods are called directly on an instance constructed with
``__new__`` so no GL context is needed.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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

from core.settings_schema import SCHEMA_VERSION, AppSettings, Ui2State  # noqa: E402
from ui2 import state_store  # noqa: E402
from ui2.app import DhmApp  # noqa: E402


# ---------------------------------------------------------------------------
# Schema + migration
# ---------------------------------------------------------------------------

def test_ui2_state_has_user_presets_field():
    state = Ui2State()
    assert state.user_presets == {}


def test_v6_payload_backfills_user_presets(tmp_path):
    state_path = tmp_path / "ui2_state.json"
    state_path.write_text(json.dumps({
        "schema_version": 6,
        "ui2": {"theme": "dark", "sample_id": "legacy"},
    }), encoding="utf-8")
    loaded = state_store.load(state_path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.user_presets == {}


def test_corrupted_user_presets_normalises_to_dict(tmp_path):
    """A state file written by a hand-edit might leave user_presets as
    a list or scalar. ``_hydrate_ui2`` must tolerate that — we reset
    to {} rather than crashing the next launch."""
    state_path = tmp_path / "ui2_state.json"
    state_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "ui2": {"user_presets": [1, 2, 3]},  # list, not dict
    }), encoding="utf-8")
    loaded = state_store.load(state_path)
    assert loaded.ui2.user_presets == {}


def test_nested_non_dict_values_dropped(tmp_path):
    """Each value in user_presets must itself be a dict. Anything
    weirder is skipped; the remaining entries survive."""
    state_path = tmp_path / "ui2_state.json"
    state_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "ui2": {
            "user_presets": {
                "good": {"wavelength_nm": 520.0},
                "bad":  "not a dict",
                "alsobad": 42,
            },
        },
    }), encoding="utf-8")
    loaded = state_store.load(state_path)
    assert "good" in loaded.ui2.user_presets
    assert "bad" not in loaded.ui2.user_presets
    assert "alsobad" not in loaded.ui2.user_presets


def test_roundtrip_preserves_user_presets(tmp_path):
    state_path = tmp_path / "ui2_state.json"
    s = AppSettings.defaults().with_ui2(
        user_presets={
            "Lab1 x40": {"wavelength_nm": 532.0, "magnification": 40.0,
                         "pixel_is_effective": False},
        },
    )
    state_store.save(s, state_path)
    loaded = state_store.load(state_path)
    assert "Lab1 x40" in loaded.ui2.user_presets
    assert loaded.ui2.user_presets["Lab1 x40"]["wavelength_nm"] == 532.0


# ---------------------------------------------------------------------------
# DhmApp preset merge + save/delete behaviour
# ---------------------------------------------------------------------------

def _make_app_with_user_presets(user_presets: dict) -> DhmApp:
    app = DhmApp.__new__(DhmApp)
    app._settings = AppSettings.defaults().with_ui2(
        user_presets=user_presets,
    )
    app._selected_preset = ""
    app._set_status = MagicMock()
    app._mark_dirty = MagicMock()
    app._toasts = MagicMock()
    app._preset_chips = MagicMock()
    return app


def test_presets_merge_builtin_and_user():
    app = _make_app_with_user_presets({
        "Lab1": {"wavelength_nm": 488.0},
    })
    merged = app._presets()
    for name in DhmApp._BUILTIN_PRESET_NAMES:
        assert name in merged
    assert "Lab1" in merged
    assert merged["Lab1"]["wavelength_nm"] == 488.0


def test_presets_reject_user_shadowing_builtin_name():
    """If someone stashes a user preset under 'Cell' via state edit,
    the sidebar must still show the built-in value — silent takeover
    would confuse lab operators who expect Cell to be canonical."""
    app = _make_app_with_user_presets({
        "Cell": {"wavelength_nm": 123.0},  # shadowing attempt
    })
    merged = app._presets()
    assert merged["Cell"]["wavelength_nm"] != 123.0
    # Built-in Cell has λ=632.8 (see _builtin_presets).
    assert merged["Cell"]["wavelength_nm"] == pytest.approx(632.8)


def test_delete_preset_rejects_builtin():
    app = _make_app_with_user_presets({})
    assert app._delete_preset("Cell") is False
    assert app._delete_preset("Film") is False
    assert app._delete_preset("USAF") is False
    assert app._delete_preset("Custom") is False


def test_delete_preset_removes_user_entry():
    app = _make_app_with_user_presets({
        "Lab1 x40": {"wavelength_nm": 532.0},
    })
    assert app._delete_preset("Lab1 x40") is True
    assert "Lab1 x40" not in app._settings.ui2.user_presets
    app._mark_dirty.assert_called()


def test_delete_missing_preset_is_noop():
    app = _make_app_with_user_presets({})
    assert app._delete_preset("Nonexistent") is False


def test_delete_clears_selected_preset_when_removed():
    app = _make_app_with_user_presets({
        "Lab1 x40": {"wavelength_nm": 532.0},
    })
    app._selected_preset = "Lab1 x40"
    app._delete_preset("Lab1 x40")
    assert app._selected_preset == ""


def test_current_params_as_preset_dict_captures_physics_fields():
    from ui2.reconstruction import ReconParams
    app = _make_app_with_user_presets({})
    app._params = ReconParams(
        wavelength_nm=488.0, pixel_um=3.45, magnification=40.0,
        pixel_is_effective=False, n_sample=1.42,
    )
    snap = app._current_params_as_preset_dict()
    assert snap["wavelength_nm"] == pytest.approx(488.0)
    assert snap["magnification"] == 40.0
    assert snap["pixel_is_effective"] is False
    assert snap["n_sample"] == pytest.approx(1.42)


def test_builtin_preset_names_match_static_tuple():
    """The delete-path guard uses ``_BUILTIN_PRESET_NAMES``. Make sure
    it matches the keys produced by ``_builtin_presets`` so we can't
    drift — a new built-in that isn't in the tuple would be deletable."""
    builtin_keys = set(DhmApp._builtin_presets().keys())
    tuple_names = set(DhmApp._BUILTIN_PRESET_NAMES)
    assert builtin_keys == tuple_names


# ---------------------------------------------------------------------------
# v2.0.7 — Edit existing user preset (collision → Replace flow)
# ---------------------------------------------------------------------------

def test_ui2_state_has_empty_user_preset_archive():
    state = Ui2State()
    assert state.user_preset_archive == {}


def test_v10_payload_backfills_user_preset_archive(tmp_path):
    """Loading a v10 dump (empty archive field) hydrates with {}."""
    state_path = tmp_path / "ui2_state.json"
    state_path.write_text(json.dumps({
        "schema_version": 10,
        "ui2": {
            "theme": "dark",
            "user_presets": {"Lab1": {"wavelength_nm": 532.0}},
        },
    }), encoding="utf-8")
    loaded = state_store.load(state_path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.user_preset_archive == {}
    # User presets must survive — migration is additive only.
    assert "Lab1" in loaded.ui2.user_presets


def test_corrupted_user_preset_archive_normalises_to_dict(tmp_path):
    """A hand-edited state file might leave archive as a list or
    scalar. ``_hydrate_ui2`` must tolerate that — reset to {} rather
    than crashing the next launch."""
    state_path = tmp_path / "ui2_state.json"
    state_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "ui2": {"user_preset_archive": [1, 2, 3]},
    }), encoding="utf-8")
    loaded = state_store.load(state_path)
    assert loaded.ui2.user_preset_archive == {}


def test_user_preset_archive_drops_non_list_entries(tmp_path):
    """Each value in user_preset_archive must be a list. Anything
    else is dropped silently — the rest survives."""
    state_path = tmp_path / "ui2_state.json"
    state_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "ui2": {
            "user_preset_archive": {
                "good": [{"wavelength_nm": 488.0}],
                "bad_scalar": 42,
                "bad_dict": {"not": "a list"},
            },
        },
    }), encoding="utf-8")
    loaded = state_store.load(state_path)
    assert "good" in loaded.ui2.user_preset_archive
    assert loaded.ui2.user_preset_archive["good"][0]["wavelength_nm"] == 488.0
    assert "bad_scalar" not in loaded.ui2.user_preset_archive
    assert "bad_dict" not in loaded.ui2.user_preset_archive


def test_user_preset_archive_drops_non_dict_items_in_lists(tmp_path):
    """Inside the per-name list, each item must be a dict (a preset
    snapshot). Strings / numbers / nested lists are filtered out."""
    state_path = tmp_path / "ui2_state.json"
    state_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "ui2": {
            "user_preset_archive": {
                "Lab1": [
                    {"wavelength_nm": 488.0},  # OK
                    "bad string",              # dropped
                    42,                         # dropped
                    {"wavelength_nm": 532.0},  # OK
                ],
            },
        },
    }), encoding="utf-8")
    loaded = state_store.load(state_path)
    archive = loaded.ui2.user_preset_archive["Lab1"]
    assert len(archive) == 2
    assert archive[0]["wavelength_nm"] == 488.0
    assert archive[1]["wavelength_nm"] == 532.0


def test_archive_roundtrip_preserves_history(tmp_path):
    state_path = tmp_path / "ui2_state.json"
    s = AppSettings.defaults().with_ui2(
        user_presets={"Lab1": {"wavelength_nm": 488.0}},
        user_preset_archive={
            "Lab1": [
                {"wavelength_nm": 632.8},  # original
                {"wavelength_nm": 532.0},  # 1st replace
            ],
        },
    )
    state_store.save(s, state_path)
    loaded = state_store.load(state_path)
    archive = loaded.ui2.user_preset_archive["Lab1"]
    assert len(archive) == 2
    assert archive[0]["wavelength_nm"] == 632.8
    assert archive[1]["wavelength_nm"] == 532.0


def test_save_user_preset_creates_new_entry_no_archive():
    """First-time save: no collision, archive stays empty."""
    app = _make_app_with_user_presets({})
    app._preset_chips = MagicMock()
    app._save_user_preset(
        "Lab1 x40", {"wavelength_nm": 488.0},
    )
    assert "Lab1 x40" in app._settings.ui2.user_presets
    assert app._settings.ui2.user_presets["Lab1 x40"]["wavelength_nm"] == 488.0
    # Brand-new save → no archive entry.
    assert "Lab1 x40" not in app._settings.ui2.user_preset_archive


def test_save_user_preset_with_archive_previous_appends_history():
    """Replace flow: old dict appended to archive[name], new dict in
    user_presets. State is fully recoverable."""
    app = _make_app_with_user_presets({
        "Lab1": {"wavelength_nm": 488.0},
    })
    app._preset_chips = MagicMock()
    app._save_user_preset(
        "Lab1",
        {"wavelength_nm": 532.0},
        archive_previous={"wavelength_nm": 488.0},
    )
    assert app._settings.ui2.user_presets["Lab1"]["wavelength_nm"] == 532.0
    archive = app._settings.ui2.user_preset_archive["Lab1"]
    assert len(archive) == 1
    assert archive[0]["wavelength_nm"] == 488.0


def test_archive_caps_at_ten_versions_per_name():
    """Repeated overwrites push the archive list — cap at 10 keeps
    state files small. Eleventh push drops the oldest."""
    pre_existing = [{"wavelength_nm": float(i)} for i in range(10)]
    app = _make_app_with_user_presets({
        "Lab1": {"wavelength_nm": 999.0},
    })
    app._settings = app._settings.with_ui2(
        user_preset_archive={"Lab1": list(pre_existing)},
    )
    app._preset_chips = MagicMock()
    # 11th push — should drop oldest (wavelength_nm=0.0)
    app._save_user_preset(
        "Lab1",
        {"wavelength_nm": 1000.0},
        archive_previous={"wavelength_nm": 999.0},
    )
    archive = app._settings.ui2.user_preset_archive["Lab1"]
    assert len(archive) == 10
    # Oldest dropped
    assert archive[0]["wavelength_nm"] == pytest.approx(1.0)
    assert archive[-1]["wavelength_nm"] == pytest.approx(999.0)


def test_save_user_preset_does_not_pollute_other_names_archive():
    """Saving 'A' shouldn't touch archive['B'] — list isolation."""
    app = _make_app_with_user_presets({
        "A": {"wavelength_nm": 488.0},
        "B": {"wavelength_nm": 532.0},
    })
    app._settings = app._settings.with_ui2(
        user_preset_archive={"B": [{"wavelength_nm": 405.0}]},
    )
    app._preset_chips = MagicMock()
    app._save_user_preset(
        "A",
        {"wavelength_nm": 633.0},
        archive_previous={"wavelength_nm": 488.0},
    )
    # B's archive untouched.
    assert app._settings.ui2.user_preset_archive["B"][0]["wavelength_nm"] \
        == pytest.approx(405.0)
    # A's archive grew by one.
    assert len(app._settings.ui2.user_preset_archive["A"]) == 1
    assert app._settings.ui2.user_preset_archive["A"][0]["wavelength_nm"] \
        == pytest.approx(488.0)
