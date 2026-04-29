"""Tests for the v2 camera live-feed pipeline (v2.0.6).

Three components, three owners:

* :class:`SyntheticCamera` — deterministic frame generator. Tests
  assert shape, dtype, value range, and that consecutive frames
  differ (else the animation is dead).
* :class:`AcquisitionThread` — start/grab/stop on a daemon thread.
  Tests verify the ``on_frame`` callback is invoked at ≈ target FPS
  and ``stop()`` returns quickly.
* :class:`TiffStackRecorder` — multi-page TIFF writer. Tests round-
  trip a 5-frame stack and confirm frame count + shape.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ui2.camera_feed import (  # noqa: E402
    AcquisitionThread,
    CameraSource,
    SyntheticCamera,
    TiffStackRecorder,
)


# ---------------------------------------------------------------------------
# SyntheticCamera basics
# ---------------------------------------------------------------------------

def test_synthetic_camera_fulfils_protocol():
    cam = SyntheticCamera(size_px=64, target_fps=15.0)
    assert isinstance(cam, CameraSource)


def test_synthetic_camera_frame_shape_and_range():
    cam = SyntheticCamera(size_px=64)
    cam.start()
    f = cam.grab()
    cam.stop()
    assert f.shape == (64, 64)
    assert f.dtype == np.float32
    assert f.min() >= 0.0
    assert f.max() <= 1.0 + 1e-6


def test_synthetic_camera_frames_are_animated():
    cam = SyntheticCamera(size_px=32)
    cam.start()
    a = cam.grab()
    b = cam.grab()
    cam.stop()
    assert not np.allclose(a, b), "consecutive frames look identical"


def test_synthetic_camera_grab_before_start_tolerated():
    """Users shouldn't be punished for skipping start() — grab() should
    just return a valid frame rather than raising."""
    cam = SyntheticCamera(size_px=16)
    f = cam.grab()
    assert f.shape == (16, 16)


def test_synthetic_camera_reports_size_and_fps():
    cam = SyntheticCamera(size_px=128, target_fps=24.0)
    assert cam.size == (128, 128)
    assert cam.fps == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# TiffStackRecorder roundtrip
# ---------------------------------------------------------------------------

def test_recorder_roundtrip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    out = tmp_path / "stack.tif"
    rec = TiffStackRecorder(out)
    rec.start()
    for i in range(5):
        frame = np.full((32, 32), i / 10.0, dtype=np.float32)
        rec.write_frame(frame)
    rec.stop()

    assert out.exists()
    assert rec.frames_written == 5
    # Read back — each page should be a (32, 32) uint16 frame scaled
    # by 65535 from our 0, 0.1, 0.2… inputs.
    with tifffile.TiffFile(out) as tf:
        assert len(tf.pages) == 5
        page_shapes = {p.shape for p in tf.pages}
        assert page_shapes == {(32, 32)}


def test_recorder_write_before_start_raises(tmp_path):
    rec = TiffStackRecorder(tmp_path / "early.tif")
    with pytest.raises(RuntimeError, match="before start"):
        rec.write_frame(np.zeros((16, 16), dtype=np.float32))


def test_recorder_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deeper" / "stack.tif"
    rec = TiffStackRecorder(out)
    rec.start()
    rec.write_frame(np.zeros((8, 8), dtype=np.float32))
    rec.stop()
    assert out.exists()


# ---------------------------------------------------------------------------
# AcquisitionThread lifecycle
# ---------------------------------------------------------------------------

def test_acquisition_thread_fires_on_frame_callback():
    cam = SyntheticCamera(size_px=32, target_fps=120.0)  # fast for the test
    frames: list = []
    lock = threading.Lock()

    def on_frame(f):
        with lock:
            frames.append(f)

    t = AcquisitionThread(source=cam, on_frame=on_frame)
    t.start()
    # Wait long enough to guarantee ≥ 3 frames at 120 fps target.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with lock:
            if len(frames) >= 3:
                break
        time.sleep(0.02)
    t.stop()
    t.join(timeout=2.0)
    with lock:
        count = len(frames)
    assert count >= 3, f"expected ≥ 3 frames, got {count}"
    assert frames[0].shape == (32, 32)
    assert not t.is_alive()


def test_acquisition_thread_stop_is_prompt():
    """stop() should return, and the thread should exit, well inside
    the per-iteration sleep (0.1 s) — otherwise the UI feels laggy."""
    cam = SyntheticCamera(size_px=16, target_fps=1.0)
    t = AcquisitionThread(source=cam, on_frame=lambda f: None)
    t.start()
    time.sleep(0.2)
    start = time.monotonic()
    t.stop()
    t.join(timeout=1.0)
    elapsed = time.monotonic() - start
    assert not t.is_alive(), "thread did not exit after stop()"
    assert elapsed < 0.5, f"stop() took {elapsed:.2f} s, expected < 0.5 s"


def test_acquisition_thread_forwards_to_recorder(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    cam = SyntheticCamera(size_px=16, target_fps=60.0)
    rec = TiffStackRecorder(tmp_path / "out.tif")
    rec.start()
    t = AcquisitionThread(source=cam, on_frame=lambda f: None,
                          recorder=rec)
    t.start()
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        if rec.frames_written >= 3:
            break
        time.sleep(0.02)
    t.stop()
    t.join(timeout=1.5)
    assert rec.frames_written >= 3
    # Recorder is closed by the thread when it exits.
    assert rec._writer is None


def test_on_frame_exception_does_not_kill_thread():
    """A broken callback should log and keep the acquisition alive
    so one buggy UI handler can't freeze the whole feed."""
    cam = SyntheticCamera(size_px=8, target_fps=100.0)
    fire_count = {"n": 0}

    def broken(_):
        fire_count["n"] += 1
        if fire_count["n"] == 1:
            raise RuntimeError("boom on first frame")

    t = AcquisitionThread(source=cam, on_frame=broken)
    t.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if fire_count["n"] >= 3:
            break
        time.sleep(0.02)
    t.stop()
    t.join(timeout=1.0)
    assert fire_count["n"] >= 3
