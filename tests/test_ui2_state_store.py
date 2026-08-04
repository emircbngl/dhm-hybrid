"""Tests for the Dear PyGui frontend's persistence (`ui2.state_store`).

All roundtrips go through a tmp_path — we never touch the real
``~/.dhm-reconstruction`` directory. The tests cover:

* save → load identity
* missing file → defaults
* corrupt JSON → defaults + no crash
* atomic write guarantee (no half-written state)
* schema migration chain (v1 → v2 → v3)
* debounced saver coalescing multiple mark_dirty calls
* forward-compat: unknown keys are dropped silently
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

# Make the v2 package importable directly — conftest doesn't add src.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.settings_schema import SCHEMA_VERSION, AppSettings, Ui2State  # noqa: E402
from ui2 import state_store  # noqa: E402


@pytest.fixture
def tmp_state_path(tmp_path):
    return tmp_path / "ui2_state.json"


# ---------------------------------------------------------------------------
# Load / save roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_state_path):
    s = AppSettings.defaults()
    s = s.with_ui2(
        viewport_w=1400, viewport_h=900, theme="midnight",
        sample_id="A-12", workflow_mode="Analyse",
        selected_preset="USAF",
        recent=["/Users/x/a.tif", "/Users/x/b.tif"],
        last_dir="/Users/x", last_hologram="/Users/x/a.tif",
        reference_path="/Users/x/ref.tif", subtract_reference=True,
        wavelength_nm=488.0, pixel_um=2.2, z_mm=4.5,
        mask_radius=55, method="Fresnel",
    )

    state_store.save(s, tmp_state_path)
    loaded = state_store.load(tmp_state_path)

    assert loaded.ui2.viewport_w == 1400
    assert loaded.ui2.viewport_h == 900
    assert loaded.ui2.theme == "midnight"
    assert loaded.ui2.sample_id == "A-12"
    assert loaded.ui2.workflow_mode == "Analyse"
    assert loaded.ui2.selected_preset == "USAF"
    assert loaded.ui2.recent == ["/Users/x/a.tif", "/Users/x/b.tif"]
    assert loaded.ui2.reference_path == "/Users/x/ref.tif"
    assert loaded.ui2.subtract_reference is True
    assert loaded.ui2.wavelength_nm == pytest.approx(488.0)
    assert loaded.ui2.mask_radius == 55
    assert loaded.ui2.method == "Fresnel"


def test_load_missing_file_returns_defaults(tmp_state_path):
    assert not tmp_state_path.exists()
    loaded = state_store.load(tmp_state_path)
    assert isinstance(loaded, AppSettings)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2 == Ui2State()


def test_load_corrupt_json_falls_back(tmp_state_path, caplog):
    tmp_state_path.write_text("{ not valid json", encoding="utf-8")
    loaded = state_store.load(tmp_state_path)
    assert isinstance(loaded, AppSettings)
    assert loaded.ui2 == Ui2State()


def test_load_empty_file_is_treated_as_defaults(tmp_state_path):
    """Zero-byte file happens when a save is interrupted before write."""
    tmp_state_path.write_text("", encoding="utf-8")
    loaded = state_store.load(tmp_state_path)
    assert loaded.ui2 == Ui2State()


# ---------------------------------------------------------------------------
# Atomic writes — crash-resistant replace
# ---------------------------------------------------------------------------

def test_save_leaves_no_tmp_on_success(tmp_state_path):
    s = AppSettings.defaults().with_ui2(sample_id="test")
    state_store.save(s, tmp_state_path)
    tmp = tmp_state_path.with_suffix(tmp_state_path.suffix + ".tmp")
    assert tmp_state_path.exists()
    assert not tmp.exists()


def test_save_overwrites_previous_state(tmp_state_path):
    s1 = AppSettings.defaults().with_ui2(sample_id="first")
    state_store.save(s1, tmp_state_path)
    s2 = AppSettings.defaults().with_ui2(sample_id="second")
    state_store.save(s2, tmp_state_path)
    assert state_store.load(tmp_state_path).ui2.sample_id == "second"


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migration_v1_to_v3_via_v2(tmp_state_path):
    """A file stamped as v1 should pick up ui2 defaults after migration."""
    tmp_state_path.write_text(
        json.dumps({"schema_version": 1, "recon": {"z_mm": 5.0}}),
        encoding="utf-8",
    )
    loaded = state_store.load(tmp_state_path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.recon.z_mm == pytest.approx(5.0)
    # ui2 section didn't exist in v1, we fill with defaults.
    assert loaded.ui2.theme == "dark"


def test_migration_v2_adds_ui2_block(tmp_state_path):
    """v2 payload (no ui2 key) still loads cleanly."""
    tmp_state_path.write_text(
        json.dumps({
            "schema_version": 2,
            "recon": {},
            "autofocus": {},
            "qpi": {},
            "io": {"last_folder": "/tmp"},
        }),
        encoding="utf-8",
    )
    loaded = state_store.load(tmp_state_path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.io.last_folder == "/tmp"
    # NOT plain Ui2State(): a pre-v9 file passes through the FROZEN v9
    # migration, which pins af_algorithm="zscan" so existing lab setups
    # keep their behaviour — while the dataclass default moved to
    # "robust" with the 2026-07-06 settlement (B-095). Missing/corrupt
    # files (no migration chain) get the pure defaults instead — see
    # test_load_missing_file_returns_defaults.
    assert loaded.ui2 == Ui2State(af_algorithm="zscan")


def test_forward_compat_unknown_keys_dropped(tmp_state_path):
    """A newer on-disk payload with unknown fields must not crash."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ui2": {
            "theme": "dark",
            "future_feature": "ignored",
            "sample_id": "X",
        },
    }
    tmp_state_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = state_store.load(tmp_state_path)
    assert loaded.ui2.theme == "dark"
    assert loaded.ui2.sample_id == "X"


def test_recent_list_normalises_non_strings(tmp_state_path):
    """Older dumps might serialise Path objects as weird types."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ui2": {"recent": [None, "/a/b", 123]},
    }
    tmp_state_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = state_store.load(tmp_state_path)
    # None → "None" string is fine — the UI will filter empties.
    assert all(isinstance(p, str) for p in loaded.ui2.recent)


# ---------------------------------------------------------------------------
# DebouncedSaver — coalesces rapid writes onto a worker thread
# ---------------------------------------------------------------------------

def test_debounced_saver_coalesces_rapid_edits(tmp_state_path, monkeypatch):
    """Rapid ``mark_dirty`` storm collapses to exactly one save call."""
    counter = {"n": 0}
    settings_box = {"v": AppSettings.defaults()}
    real_save = state_store.save   # capture before we patch

    def counted_save(s, path=None):
        counter["n"] += 1
        real_save(s, tmp_state_path)   # explicit target, no recursion

    monkeypatch.setattr(state_store, "save", counted_save)
    saver = state_store.DebouncedSaver(lambda: settings_box["v"], delay=0.05)

    for i in range(10):
        settings_box["v"] = AppSettings.defaults().with_ui2(
            sample_id=f"S{i}",
        )
        saver.mark_dirty()
    # Nothing should have fired yet — we're inside the delay window.
    saver.tick()
    assert counter["n"] == 0

    # Wait out the delay, then tick once.
    time.sleep(0.1)
    saver.tick()
    # Give the async worker a moment to actually run.
    deadline = time.monotonic() + 1.0
    while counter["n"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert counter["n"] == 1  # coalesced to a single save


def test_debounced_saver_tick_no_dirty_is_noop(tmp_state_path, monkeypatch):
    counter = {"n": 0}

    def counted_save(s, path=None):
        counter["n"] += 1

    monkeypatch.setattr(state_store, "save", counted_save)
    saver = state_store.DebouncedSaver(
        lambda: AppSettings.defaults(), delay=0.01,
    )
    saver.tick()  # never marked dirty
    time.sleep(0.05)
    saver.tick()
    assert counter["n"] == 0


def test_debounced_saver_flush_now_is_synchronous(tmp_state_path, monkeypatch):
    saver = state_store.DebouncedSaver(
        lambda: AppSettings.defaults().with_ui2(sample_id="flush"),
        delay=10.0,  # long debounce — flush_now bypasses it
    )
    real_save = state_store.save
    monkeypatch.setattr(state_store, "STATE_PATH", tmp_state_path)

    def redirected(s, path=None):
        real_save(s, tmp_state_path)

    monkeypatch.setattr(state_store, "save", redirected)
    saver.mark_dirty()
    saver.flush_now()
    loaded = state_store.load(tmp_state_path)
    assert loaded.ui2.sample_id == "flush"
