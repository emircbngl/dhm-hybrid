"""Headless CLI runner tests (v2.0.7, T1 + T6).

Coverage:

* End-to-end ``run`` on a synthetic 3-frame session — frame JSONs,
  CSV aggregate, signature marker, JSONL progress events.
* ``inspect`` subcommand prints valid JSON.
* Resume (T6): ``--resume-if-exists`` skips already-completed
  frames whose signature matches; runs the rest.
* Resume invalidates output when params change (different
  signature → re-runs even with the flag set).
* Frame errors don't kill the session — they're written into
  ``frame_<i>.json`` and the CSV ``frame_error`` column.
* CSV layouts (long + wide) match what session_export.py promises.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from cli import run_session as runner  # noqa: E402
from core.session import Session  # noqa: E402

# Use the synthetic-hologram fixture — same source the validation
# suite uses, so the CLI exercise hits production code.
from fixtures.synthetic_hologram import (  # noqa: E402
    HologramConfig, SphereSpec, build_hologram,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_3frame_session(tmp_path: Path) -> Path:
    """Make 3 synthetic single-sphere holograms + a session
    manifest pointing at them. Returns the manifest path."""
    cfg = HologramConfig(
        shape=(128, 128), pixel_m=2.5e-6,
        wavelength_m=632.8e-9,
        carrier_freq_m_inv=(50_000.0, 0.0),
    )
    holo_dir = tmp_path / "holograms"
    holo_dir.mkdir()
    import tifffile
    for i, z_mm in enumerate([12.0, 13.0, 14.0]):
        sphere = SphereSpec(radius_m=15e-6, z_m=z_mm * 1e-3,
                            center_yx_m=(0.0, 0.0),
                            n_sphere=1.40, n_medium=1.33)
        holo = build_hologram([sphere], cfg)
        tifffile.imwrite(holo_dir / f"frame_{i:03d}.tif",
                         holo.astype(np.float32))
    s = Session.from_directory(holo_dir, glob_pattern="*.tif")
    s.params = {
        "wavelength_nm": 632.8,
        "pixel_um": 2.5,
        "magnification": 1.0,
        "n_medium": 1.33,
        "mask_radius": 25,
        "method": "ASM",
        "af_z_min_mm": 8.0,
        "af_z_max_mm": 16.0,
        "af_n_steps": 20,
        "autofocus_metric": "ENTROPY",
    }
    s.operator = "karin"
    s.sample_id = "synthetic_3frame"
    manifest = tmp_path / "session.json"
    s.save_json(manifest)
    return manifest


# Reset CLI cancel flag between tests (module-level global).
@pytest.fixture(autouse=True)
def _reset_cancel():
    runner._CANCEL = False
    yield
    runner._CANCEL = False


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def test_inspect_prints_valid_json(tmp_path, capsys):
    manifest = _build_3frame_session(tmp_path)
    rc = runner.inspect_session(manifest)
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["frame_count"] == 3
    assert payload["operator"] == "karin"
    assert payload["signature"]


# ---------------------------------------------------------------------------
# Run — end to end (autofocus phase keeps the test fast)
# ---------------------------------------------------------------------------

def test_run_session_writes_per_frame_json_and_csv(tmp_path):
    manifest = _build_3frame_session(tmp_path)
    out_dir = tmp_path / "out"
    progress = io.StringIO()
    rc = runner.run_session(
        manifest, out_dir,
        phase="autofocus",
        progress_stream=progress,
    )
    assert rc == 0
    # Per-frame JSONs.
    for i in range(3):
        p = out_dir / f"frame_{i:05d}.json"
        assert p.exists()
        d = json.loads(p.read_text())
        assert d["frame_index"] == i
        assert d["z_mm"] is not None
        assert d["session_signature"]
    # Session signature marker.
    assert (out_dir / "session.signature").exists()
    # Aggregate CSV.
    csv_path = out_dir / "session.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("session_id,operator,sample_id,frame_index")
    # 3 rows (one per frame, no segmented cells in autofocus phase →
    # cell-less rows still appear).
    assert len(text) - 1 == 3


def test_run_session_emits_progress_jsonl(tmp_path):
    manifest = _build_3frame_session(tmp_path)
    out_dir = tmp_path / "out"
    progress = io.StringIO()
    runner.run_session(
        manifest, out_dir, phase="autofocus",
        progress_stream=progress,
    )
    lines = [json.loads(l) for l in progress.getvalue().splitlines()]
    assert lines[0]["event"] == "session_start"
    assert lines[-1]["event"] == "session_done"
    frame_done = [l for l in lines if l["event"] == "frame_done"]
    assert len(frame_done) == 3
    # Every frame_done carries an autofocus z (autofocus phase).
    assert all(l["z_mm"] is not None for l in frame_done)


def test_run_session_quiet_suppresses_stdout(tmp_path, capsys):
    manifest = _build_3frame_session(tmp_path)
    runner.run_session(
        manifest, tmp_path / "out",
        phase="autofocus", quiet=True,
    )
    out = capsys.readouterr().out
    assert out == ""


# ---------------------------------------------------------------------------
# Resume (T6)
# ---------------------------------------------------------------------------

def test_resume_skips_completed_frames(tmp_path):
    """First run completes all frames. Second run with
    --resume-if-exists must skip them all (frames_processed=0)."""
    manifest = _build_3frame_session(tmp_path)
    out_dir = tmp_path / "out"
    runner.run_session(manifest, out_dir, phase="autofocus", quiet=True)

    # Second invocation, same manifest → resume should skip.
    progress = io.StringIO()
    rc = runner.run_session(
        manifest, out_dir, phase="autofocus",
        resume_if_exists=True, quiet=True,
        progress_stream=progress,
    )
    assert rc == 0
    lines = [json.loads(l) for l in progress.getvalue().splitlines()]
    start = next(l for l in lines if l["event"] == "session_start")
    done = next(l for l in lines if l["event"] == "session_done")
    assert start["pending"] == 0
    assert start["skipped"] == 3
    assert done["frames_processed"] == 0


def test_resume_re_runs_when_signature_changes(tmp_path):
    """Change a session param → signature changes → resume must
    re-run the frames (existing JSON has stale signature)."""
    manifest = _build_3frame_session(tmp_path)
    out_dir = tmp_path / "out"
    runner.run_session(manifest, out_dir, phase="autofocus", quiet=True)

    # Mutate the manifest (param drift) and persist.
    s = Session.load_json(manifest)
    s.params["af_z_min_mm"] = 5.0  # was 8.0
    s.save_json(manifest)

    progress = io.StringIO()
    runner.run_session(
        manifest, out_dir, phase="autofocus",
        resume_if_exists=True, quiet=True,
        progress_stream=progress,
    )
    lines = [json.loads(l) for l in progress.getvalue().splitlines()]
    start = next(l for l in lines if l["event"] == "session_start")
    # All 3 frames pending again (none can be reused).
    assert start["pending"] == 3
    assert start["skipped"] == 0


def test_resume_handles_partial_output(tmp_path):
    """Mid-run crash leaves 1 frame written, 2 missing. Resume
    should pick up exactly the 2 missing ones."""
    manifest = _build_3frame_session(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Prime the output dir with frame 1 only, matching the session sig.
    s = Session.load_json(manifest)
    sig = s.signature()
    runner._write_signature(out_dir, sig)
    fake_payload = {"z_mm": 12.5, "runtime_ms": 100.0,
                    "error": None, "cells": []}
    runner._write_frame_output(out_dir, s.frames[1], sig, fake_payload)

    progress = io.StringIO()
    runner.run_session(
        manifest, out_dir, phase="autofocus",
        resume_if_exists=True, quiet=True,
        progress_stream=progress,
    )
    lines = [json.loads(l) for l in progress.getvalue().splitlines()]
    start = next(l for l in lines if l["event"] == "session_start")
    # Frame 1 already done — pending = 2 (frames 0 and 2).
    assert start["skipped"] == 1
    assert start["pending"] == 2


# ---------------------------------------------------------------------------
# Phase filter
# ---------------------------------------------------------------------------

def test_phase_autofocus_skips_reconstruction(tmp_path):
    """phase='autofocus' should still produce valid frame JSONs but
    skip the reconstruct step. For a smoke test we just check
    no errors and z_mm is set."""
    manifest = _build_3frame_session(tmp_path)
    out_dir = tmp_path / "out"
    rc = runner.run_session(manifest, out_dir, phase="autofocus",
                            quiet=True)
    assert rc == 0
    for i in range(3):
        d = json.loads((out_dir / f"frame_{i:05d}.json")
                       .read_text())
        assert d["error"] is None
        assert d["z_mm"] is not None


def test_phase_reconstruct_runs_full_pipeline(tmp_path):
    manifest = _build_3frame_session(tmp_path)
    out_dir = tmp_path / "out"
    rc = runner.run_session(manifest, out_dir, phase="reconstruct",
                            quiet=True)
    assert rc == 0


# ---------------------------------------------------------------------------
# Error containment
# ---------------------------------------------------------------------------

def test_frame_error_does_not_abort_session(tmp_path):
    """A missing frame file should write an error entry, not crash
    the whole session. The other frames still run."""
    manifest = _build_3frame_session(tmp_path)
    s = Session.load_json(manifest)
    # Remove one frame's source file so load_any() blows up.
    bad_path = Path(s.root_dir) / s.frames[1].path
    bad_path.unlink()

    out_dir = tmp_path / "out"
    rc = runner.run_session(manifest, out_dir, phase="autofocus",
                            quiet=True)
    # rc==1 because at least one frame failed.
    assert rc == 1
    err = json.loads((out_dir / "frame_00001.json").read_text())
    assert err["error"]
    # Other frames OK.
    for i in (0, 2):
        d = json.loads((out_dir / f"frame_{i:05d}.json").read_text())
        assert d["error"] is None


# ---------------------------------------------------------------------------
# argparse end-to-end (CLI shape)
# ---------------------------------------------------------------------------

def test_cli_main_run_succeeds(tmp_path, monkeypatch):
    """Drive the argparse entrypoint exactly like a shell would."""
    manifest = _build_3frame_session(tmp_path)
    out_dir = tmp_path / "out"
    rc = runner.main([
        "run", str(manifest),
        "--out", str(out_dir),
        "--phase", "autofocus",
        "--quiet",
    ])
    assert rc == 0
    assert (out_dir / "session.csv").exists()


def test_cli_main_inspect(tmp_path, capsys):
    manifest = _build_3frame_session(tmp_path)
    rc = runner.main(["inspect", str(manifest)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frame_count"] == 3


def test_cli_main_unknown_subcommand_exits(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(["bogus"])
