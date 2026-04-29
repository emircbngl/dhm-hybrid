"""v2.1.x real-hardware sprint tests.

Coverage:

* :mod:`core.cameras` registry — discovery, availability filter,
  factory, error paths.
* :mod:`core.cameras.mock` — pathology behaviours (drop rate,
  warmup, exposure jitter).
* :mod:`core.video_recorder` — round-trip MP4 (skip when
  imageio_ffmpeg unavailable).
* Vendor stubs (Pylon / IDS / Thorlabs) — registry advertises
  them, ``is_available()`` returns False on dev box, ``make()``
  raises NotImplementedError when an SDK isn't there.
* FPS perf — synthetic + mock cameras sustain ≥ 30 fps on the
  test box.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import cameras  # noqa: E402
from core.cameras import (  # noqa: E402
    CameraBackendInfo, all_backends, available_backends, make_camera,
)
from ui2.camera_feed import CameraSource  # noqa: E402


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_all_backends_includes_synthetic_mock_and_vendors():
    names = {b.name for b in all_backends()}
    # SDK-free always present.
    assert "synthetic" in names
    assert "mock" in names
    # Vendor stubs registered even if SDK missing.
    assert "pylon" in names
    assert "ids" in names
    assert "thorlabs" in names


def test_available_backends_excludes_unavailable_vendors():
    """Vendor backends without their SDK installed should be
    filtered out of ``available_backends`` (they remain in
    ``all_backends`` so the dialog can still show greyed rows)."""
    names = {b.name for b in available_backends()}
    assert "synthetic" in names
    assert "mock" in names
    # No vendor SDKs in the dev venv.
    assert "pylon" not in names
    assert "ids" not in names
    assert "thorlabs" not in names


def test_backend_info_carries_capabilities():
    pylon = next(b for b in all_backends() if b.name == "pylon")
    assert pylon.requires_sdk == ("pypylon",)
    assert pylon.capabilities["live"] is True
    assert pylon.capabilities["trigger"] is True
    assert pylon.capabilities["16-bit"] is True


def test_make_camera_synthetic_returns_camera_source():
    cam = make_camera("synthetic", size_px=64)
    assert isinstance(cam, CameraSource)
    assert cam.size == (64, 64)


def test_make_camera_mock_returns_camera_source():
    cam = make_camera("mock", size_px=64)
    assert isinstance(cam, CameraSource)


def test_make_camera_unknown_name_raises_valueerror():
    with pytest.raises(ValueError, match="unknown camera backend"):
        make_camera("not_a_backend")


def test_make_camera_unavailable_vendor_raises_runtime():
    """Pylon registered but SDK absent → RuntimeError pointing at
    the missing pip package. Caller catches and falls back to
    synthetic / mock."""
    with pytest.raises(RuntimeError, match="requires SDK"):
        make_camera("pylon")


# ---------------------------------------------------------------------------
# Mock camera pathologies
# ---------------------------------------------------------------------------

def test_mock_camera_drops_at_configured_rate():
    """drop_rate=1.0 → every frame is a zero-filled drop."""
    cam = make_camera("mock", size_px=32, drop_rate=1.0)
    cam.start()
    for _ in range(5):
        f = cam.grab()
        assert np.all(f == 0)


def test_mock_camera_warmup_returns_blank_frames():
    cam = make_camera("mock", size_px=32, warmup_frames=3)
    cam.start()
    for _ in range(3):
        assert np.all(cam.grab() == 0)
    # Frame 4 should have signal.
    assert np.any(cam.grab() > 0)


def test_mock_camera_deterministic_with_seed():
    a = make_camera("mock", size_px=32, rng_seed=7,
                    read_noise_sigma=0.1)
    b = make_camera("mock", size_px=32, rng_seed=7,
                    read_noise_sigma=0.1)
    a.start()
    b.start()
    np.testing.assert_array_equal(a.grab(), b.grab())


def test_mock_camera_protocol_surface():
    cam = make_camera("mock", size_px=16)
    assert cam.size == (16, 16)
    assert cam.fps > 0
    cam.start()
    f = cam.grab()
    assert f.shape == (16, 16)
    assert f.dtype == np.float32
    cam.stop()


# ---------------------------------------------------------------------------
# Vendor stubs raise on start (no SDK)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["pylon", "ids", "thorlabs"])
def test_vendor_stub_factory_returns_skeleton(name):
    """When the SDK *is* present the registry would let
    ``make_camera`` through; we can't test that path here, but we
    can prove the per-module ``make()`` factory builds a skeleton
    and that ``start()`` raises NotImplementedError until the SDK
    block is filled in."""
    from importlib import import_module
    mod = import_module(f"core.cameras.{name}")
    cam = mod.make()
    assert hasattr(cam, "start")
    with pytest.raises(NotImplementedError):
        cam.start()


# ---------------------------------------------------------------------------
# MP4 recorder — skip when imageio_ffmpeg missing
# ---------------------------------------------------------------------------

from core.video_recorder import MP4Recorder, is_available as mp4_available  # noqa: E402

mp4_only = pytest.mark.skipif(
    not mp4_available(),
    reason="imageio-ffmpeg optional; install with `pip install "
           "imageio-ffmpeg` for video recording tests",
)


@mp4_only
def test_mp4_recorder_round_trip(tmp_path):
    rec = MP4Recorder(tmp_path / "out.mp4", fps=10.0)
    rec.start()
    for _ in range(20):
        rec.write_frame(np.random.rand(64, 64).astype(np.float32))
    rec.stop()
    out = tmp_path / "out.mp4"
    assert out.exists()
    assert out.stat().st_size > 256  # non-empty MP4


@mp4_only
def test_mp4_recorder_frames_written_counter(tmp_path):
    rec = MP4Recorder(tmp_path / "out.mp4")
    rec.start()
    rec.write_frame(np.zeros((32, 32), dtype=np.float32))
    rec.write_frame(np.zeros((32, 32), dtype=np.float32))
    rec.stop()
    assert rec.frames_written == 2


@mp4_only
def test_mp4_recorder_creates_parent_dir(tmp_path):
    rec = MP4Recorder(tmp_path / "deeper" / "sub" / "x.mp4")
    rec.start()
    rec.write_frame(np.zeros((16, 16), dtype=np.float32))
    rec.stop()
    assert (tmp_path / "deeper" / "sub" / "x.mp4").exists()


def test_mp4_recorder_write_before_start_raises(tmp_path):
    rec = MP4Recorder(tmp_path / "x.mp4")
    with pytest.raises(RuntimeError, match="before start"):
        rec.write_frame(np.zeros((8, 8), dtype=np.float32))


def test_mp4_recorder_start_without_imageio_raises(tmp_path, monkeypatch):
    """When imageio_ffmpeg is missing we surface a clear error
    rather than silently skipping the recording."""
    import core.video_recorder as vr
    monkeypatch.setattr(vr, "is_available", lambda: False)
    rec = vr.MP4Recorder(tmp_path / "x.mp4")
    with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
        rec.start()


# ---------------------------------------------------------------------------
# FPS sustained throughput — smoke
# ---------------------------------------------------------------------------

def test_synthetic_camera_sustains_30fps_at_512px():
    """Best-effort: grab 60 frames; assert wall-clock matches at
    least 30 fps. CI flake-prone but a regression that drops it
    by 2× would be obvious."""
    from ui2.camera_feed import SyntheticCamera
    cam = SyntheticCamera(size_px=512)
    cam.start()
    cam.grab()  # warm
    n = 60
    t0 = time.monotonic()
    for _ in range(n):
        f = cam.grab()
        assert f.shape == (512, 512)
    elapsed = time.monotonic() - t0
    fps = n / elapsed
    cam.stop()
    # Lab box can hit 100+ fps on this; 30 is the floor that maps
    # to "live preview is real-time".
    assert fps >= 30, (
        f"synthetic camera sustained only {fps:.1f} fps at 512² "
        f"({elapsed * 1000 / n:.1f} ms/frame); expected ≥ 30"
    )


def test_mock_camera_sustains_30fps_at_512px():
    cam = make_camera("mock", size_px=512)
    cam.start()
    cam.grab()
    n = 60
    t0 = time.monotonic()
    for _ in range(n):
        cam.grab()
    elapsed = time.monotonic() - t0
    fps = n / elapsed
    cam.stop()
    assert fps >= 30, f"mock camera dropped to {fps:.1f} fps"
