"""User-defined preset persistence tests (v2.0.6).

The feature has three moving parts:

* ``Ui2State.user_presets: dict[str, dict]`` — the on-disk store.
* ``_v6_to_v7`` migrator backfills an empty dict for older dumps.
* ``DhmApp._presets()`` merges built-ins + user presets; save path
  rejects collisions with built-ins, delete path refuses built-ins.

Tests verify each layer without launching Dear PyGui — the DhmApp
methods are called directly on an instance constructed with
``__new__`` so no GL context is needed.

DPG/DhmApp tests removed 2026-07-06 with the ui2 frontend retirement; driver/state tests kept.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings_schema import SCHEMA_VERSION, AppSettings, Ui2State  # noqa: E402
from ui2 import state_store  # noqa: E402


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
