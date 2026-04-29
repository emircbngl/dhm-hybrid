"""Tests for v2.0.7 :mod:`core.session` — the longitudinal data
model that captures Karin's 3000-hologram time-lapse pain.

Coverage:

* construction (``new``, ``from_directory``)
* mutation (``add_frame``, ``with_params``, ``effective_params_for``)
* path resolution (relative vs absolute, ``root_dir`` anchor)
* signature determinism (``--resume-if-exists`` correctness)
* JSON round-trip (atomic write, malformed-JSON handling)
* index stability under reorder (we don't reorder, so this also
  documents the rule)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.session import HologramFrame, Session  # noqa: E402


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_new_session_assigns_uuid_and_timestamp():
    s = Session.new(operator="emir", sample_id="A549_pHluorin")
    # 32-char hex UUID
    assert len(s.id) == 32
    # ISO-8601 starts with YYYY-MM-DD
    assert s.created_at.startswith("20")
    assert s.operator == "emir"
    assert s.sample_id == "A549_pHluorin"
    assert s.frames == []


def test_new_sessions_are_unique():
    """Every ``Session.new`` must produce a fresh UUID — same-process
    collisions would corrupt audit trails."""
    a = Session.new()
    b = Session.new()
    assert a.id != b.id


def test_from_directory_picks_up_glob_matches(tmp_path):
    """Drop a folder of TIFFs, the session has one frame per file."""
    for i in range(5):
        (tmp_path / f"frame_{i:03d}.tif").write_bytes(b"x")
    (tmp_path / "ignored.txt").write_bytes(b"x")
    s = Session.from_directory(tmp_path)
    assert len(s) == 5
    assert all(f.index == i for i, f in enumerate(s.frames))
    # Filenames sorted lexicographically by default.
    assert s.frames[0].path == "frame_000.tif"
    assert s.frames[-1].path == "frame_004.tif"
    # root_dir anchors the manifest.
    assert s.root_dir == str(tmp_path.resolve())


def test_from_directory_sort_by_mtime(tmp_path):
    """When filenames don't sort the way you want, mtime is the
    fallback. Lab cameras occasionally write zzz_first.tif then
    aaa_second.tif if their counter wraps."""
    paths = []
    for i, name in enumerate(["zzz.tif", "aaa.tif", "mmm.tif"]):
        p = tmp_path / name
        p.write_bytes(b"x")
        # Set mtime so order is deterministic.
        ts = 1_700_000_000.0 + i
        os.utime(p, (ts, ts))
        paths.append(p)
    s = Session.from_directory(tmp_path, sort_by="mtime")
    assert [f.path for f in s.frames] == ["zzz.tif", "aaa.tif", "mmm.tif"]


def test_from_directory_unknown_sort_raises(tmp_path):
    (tmp_path / "x.tif").write_bytes(b"x")
    with pytest.raises(ValueError, match="unknown sort_by"):
        Session.from_directory(tmp_path, sort_by="bogus")


def test_from_directory_empty_match_raises(tmp_path):
    (tmp_path / "x.txt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        Session.from_directory(tmp_path, glob_pattern="*.tif*")


def test_from_directory_missing_dir_raises(tmp_path):
    bogus = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        Session.from_directory(bogus)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def test_add_frame_assigns_running_index(tmp_path):
    s = Session.new()
    p1 = tmp_path / "a.tif"
    p1.write_bytes(b"a")
    p2 = tmp_path / "b.tif"
    p2.write_bytes(b"b")
    f1 = s.add_frame(p1)
    f2 = s.add_frame(p2)
    assert f1.index == 0
    assert f2.index == 1
    assert len(s) == 2


def test_add_frame_uses_mtime_when_timestamp_omitted(tmp_path):
    """Lab cameras embed acquisition time in EXIF; until we parse
    that, st_mtime is a usable proxy. Test the proxy works."""
    p = tmp_path / "x.tif"
    p.write_bytes(b"x")
    target_ts = 1_700_000_000.0
    os.utime(p, (target_ts, target_ts))
    s = Session.new()
    f = s.add_frame(p)
    assert f.timestamp_s == pytest.approx(target_ts)


def test_add_frame_explicit_timestamp_wins(tmp_path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"x")
    s = Session.new()
    f = s.add_frame(p, timestamp_s=1_234_567_890.5)
    assert f.timestamp_s == pytest.approx(1_234_567_890.5)


def test_add_frame_carries_overrides_and_notes(tmp_path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"x")
    s = Session.new()
    f = s.add_frame(
        p,
        params_overrides={"z_mm": 17.5},
        notes="cell budded",
    )
    assert f.params_overrides == {"z_mm": 17.5}
    assert f.notes == "cell budded"


def test_with_params_returns_copy_with_merged_dict():
    s = Session.new(params={"wavelength_nm": 632.8, "z_mm": 12.0})
    s2 = s.with_params(z_mm=15.0, n_medium=1.33)
    # Original untouched.
    assert s.params == {"wavelength_nm": 632.8, "z_mm": 12.0}
    # Copy merged.
    assert s2.params == {
        "wavelength_nm": 632.8, "z_mm": 15.0, "n_medium": 1.33,
    }


def test_effective_params_merges_session_and_frame_overrides():
    s = Session.new(params={"z_mm": 12.0, "wavelength_nm": 632.8})
    f = HologramFrame(
        path="x.tif", timestamp_s=0.0, index=0,
        params_overrides={"z_mm": 18.0},
    )
    eff = s.effective_params_for(f)
    # Frame override wins.
    assert eff["z_mm"] == 18.0
    # Session-only field passed through.
    assert eff["wavelength_nm"] == 632.8


def test_effective_params_returns_fresh_copy():
    """Mutating the returned dict must not affect session state."""
    s = Session.new(params={"z_mm": 12.0})
    f = HologramFrame(path="x.tif", timestamp_s=0.0, index=0)
    eff = s.effective_params_for(f)
    eff["z_mm"] = 999.0
    assert s.params["z_mm"] == 12.0


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_resolve_frame_path_anchors_relative_to_root_dir(tmp_path):
    s = Session.new(root_dir=str(tmp_path))
    f = HologramFrame(path="sub/frame.tif", timestamp_s=0.0, index=0)
    resolved = s.resolve_frame_path(f)
    assert resolved == tmp_path / "sub" / "frame.tif"


def test_resolve_frame_path_passes_absolute_through(tmp_path):
    s = Session.new(root_dir=str(tmp_path))
    abs_path = (tmp_path / "elsewhere.tif").resolve()
    f = HologramFrame(path=str(abs_path), timestamp_s=0.0, index=0)
    assert s.resolve_frame_path(f) == abs_path


def test_resolve_frame_path_no_root_dir_returns_as_is():
    s = Session.new()  # root_dir defaults to ""
    f = HologramFrame(path="just_a_name.tif", timestamp_s=0.0, index=0)
    assert s.resolve_frame_path(f) == Path("just_a_name.tif")


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

def test_signature_is_deterministic(tmp_path):
    """Same session content → same signature, run-after-run."""
    p = tmp_path / "x.tif"
    p.write_bytes(b"hello")
    s1 = Session.new(params={"z_mm": 12.0}, root_dir=str(tmp_path))
    s1.add_frame(p)
    s2 = Session.new(params={"z_mm": 12.0}, root_dir=str(tmp_path))
    s2.add_frame(p)
    # Different IDs (UUIDs), same content signature.
    assert s1.id != s2.id
    assert s1.signature() == s2.signature()


def test_signature_changes_when_params_change(tmp_path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"hello")
    s = Session.new(params={"z_mm": 12.0}, root_dir=str(tmp_path))
    s.add_frame(p)
    sig_before = s.signature()
    s.params["z_mm"] = 18.0
    assert s.signature() != sig_before


def test_signature_changes_when_frame_added(tmp_path):
    p1 = tmp_path / "a.tif"; p1.write_bytes(b"a")
    p2 = tmp_path / "b.tif"; p2.write_bytes(b"b")
    s = Session.new(root_dir=str(tmp_path))
    s.add_frame(p1)
    sig_one = s.signature()
    s.add_frame(p2)
    assert s.signature() != sig_one


def test_signature_changes_when_frame_size_changes(tmp_path):
    """Re-acquisition: same filename, different bytes → different
    signature. Resume must NOT skip the frame."""
    p = tmp_path / "x.tif"
    p.write_bytes(b"v1")
    s = Session.new(root_dir=str(tmp_path))
    s.add_frame(p)
    sig_before = s.signature()
    p.write_bytes(b"version_two_with_more_bytes")
    assert s.signature() != sig_before


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_dict_round_trips_through_from_dict(tmp_path):
    p = tmp_path / "frame.tif"
    p.write_bytes(b"x")
    s = Session.new(
        operator="emir", sample_id="HeLa_001",
        params={"wavelength_nm": 632.8}, root_dir=str(tmp_path),
    )
    s.add_frame(p, notes="mitosis", params_overrides={"z_mm": 15.0})
    raw = s.to_dict()
    loaded = Session.from_dict(raw)
    assert loaded.id == s.id
    assert loaded.operator == "emir"
    assert loaded.sample_id == "HeLa_001"
    assert loaded.params == s.params
    assert len(loaded) == 1
    assert loaded.frames[0].notes == "mitosis"
    assert loaded.frames[0].params_overrides == {"z_mm": 15.0}


def test_save_load_json_round_trip(tmp_path):
    p = tmp_path / "frame.tif"
    p.write_bytes(b"x")
    s = Session.new(operator="karin", root_dir=str(tmp_path))
    s.add_frame(p)
    manifest = tmp_path / "session.json"
    s.save_json(manifest)
    assert manifest.exists()
    s2 = Session.load_json(manifest)
    assert s2.id == s.id
    assert len(s2) == 1


def test_save_json_is_atomic(tmp_path):
    """Write doesn't leave a temp file in the session directory
    after a successful save."""
    s = Session.new()
    target = tmp_path / "session.json"
    s.save_json(target)
    leftovers = [
        p for p in tmp_path.iterdir() if p.name.endswith(".tmp")
    ]
    assert leftovers == []
    assert target.exists()


def test_load_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Session.load_json(tmp_path / "nope.json")


def test_load_json_malformed_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not { valid : json")
    with pytest.raises(ValueError, match="not valid JSON"):
        Session.load_json(bad)


def test_from_dict_tolerant_to_missing_fields():
    """Hand-edited manifests should still load if they're missing
    optional fields. ``Session.from_dict`` falls back to defaults
    rather than raising — the CLI can warn about gaps separately."""
    raw = {"frames": [{"path": "x.tif", "timestamp_s": 0.0,
                       "index": 0}]}
    s = Session.from_dict(raw)
    assert s.id == ""        # optional, defaulted
    assert s.operator == ""  # optional, defaulted
    assert len(s) == 1


def test_from_dict_drops_invalid_frames():
    """A frame entry that's not a dict, or has un-parseable types,
    is silently skipped. The remaining frames survive."""
    raw = {
        "frames": [
            {"path": "good.tif", "timestamp_s": 1.0, "index": 0},
            "not a dict — dropped",
            {"path": "bad-index", "timestamp_s": 2.0, "index": "x"},
            {"path": "good2.tif", "timestamp_s": 3.0, "index": 1},
        ],
    }
    s = Session.from_dict(raw)
    assert len(s) == 2
    assert s.frames[0].path == "good.tif"
    assert s.frames[1].path == "good2.tif"


# ---------------------------------------------------------------------------
# __len__ + __iter__
# ---------------------------------------------------------------------------

def test_iteration_yields_frames_in_order(tmp_path):
    s = Session.new()
    for i in range(3):
        p = tmp_path / f"f{i}.tif"
        p.write_bytes(b"x")
        s.add_frame(p)
    indices = [f.index for f in s]
    assert indices == [0, 1, 2]


def test_get_frame_out_of_range_raises():
    s = Session.new()
    with pytest.raises(IndexError):
        s.get_frame(0)
    with pytest.raises(IndexError):
        s.get_frame(-1)
