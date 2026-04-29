"""Time-lapse acquisition runner tests (v2.1.x, H5).

Coverage:

* Schedule validation (interval, total_frames, max_duration).
* Frame count caps (total_frames, max_duration_s, both, neither).
* TIFF + manifest written, frame count matches schedule.
* Cancel mid-run flushes manifest with partial frames.
* Camera-error containment — single grab failure doesn't kill
  the run; gap is recorded in manifest.
* Camera-start failure short-circuits gracefully.
* `start_at` future timestamp delays first capture.
* Progress / frame callbacks fire.

A fake clock keeps a "12-hour" schedule down to microseconds.
"""
from __future__ import annotations

import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core.cameras.mock import MockCamera  # noqa: E402
from core.session import Session  # noqa: E402
from core.timelapse import (  # noqa: E402
    TimelapseRunner,
    TimelapseSchedule,
)


# ---------------------------------------------------------------------------
# Fake clock — never sleeps in real time
# ---------------------------------------------------------------------------

class _FakeClock:
    """Deterministic clock with controllable advance.

    ``now_utc`` and ``monotonic`` start at known values; ``sleep``
    just advances both forward by the requested amount and lets
    cancel events break out the same way the real clock does."""

    def __init__(self, *, start: datetime = None) -> None:
        self._t0_utc = (start or datetime(2026, 4, 28, 12, 0, 0,
                                           tzinfo=timezone.utc))
        self._t_mono = 0.0
        self._t_utc = self._t0_utc

    def now_utc(self) -> datetime:
        return self._t_utc

    def monotonic(self) -> float:
        return self._t_mono

    def sleep(self, seconds: float, cancel_event: threading.Event) -> None:
        # Cancel before any advance honours the "≤ 100 ms latency"
        # contract from the real clock.
        if cancel_event.is_set():
            return
        self._t_mono += float(seconds)
        self._t_utc = self._t_utc + timedelta(seconds=float(seconds))


# ---------------------------------------------------------------------------
# Schedule validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interval", [0, -1, -100])
def test_schedule_rejects_non_positive_interval(interval):
    with pytest.raises(ValueError, match="interval_s"):
        TimelapseSchedule(interval_s=interval)


def test_schedule_rejects_zero_total_frames():
    with pytest.raises(ValueError, match="total_frames"):
        TimelapseSchedule(interval_s=1.0, total_frames=0)


def test_schedule_rejects_negative_max_duration():
    with pytest.raises(ValueError, match="max_duration_s"):
        TimelapseSchedule(interval_s=1.0, max_duration_s=-5.0)


def test_schedule_expected_frame_count():
    s = TimelapseSchedule(interval_s=300.0, total_frames=144,
                          max_duration_s=12 * 3600)
    # 12h / 5min = 144 + 1 (the boundary frame); total_frames=144
    # is the smaller of the two so plan = 144.
    assert s.expected_frame_count() == 144


def test_schedule_expected_frame_count_unbounded():
    s = TimelapseSchedule(interval_s=300.0)
    assert s.expected_frame_count() is None


# ---------------------------------------------------------------------------
# Happy path: total_frames cap
# ---------------------------------------------------------------------------

def test_runner_writes_n_frames_and_manifest(tmp_path):
    cam = MockCamera(size_px=32)
    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(interval_s=300.0, total_frames=5),
        out_dir=tmp_path,
        sample_id="test", operator="karin",
        clock=_FakeClock(),
    )
    res = runner.run()
    assert res.frames_captured == 5
    assert not res.cancelled
    # 5 TIFFs on disk.
    tiffs = sorted(tmp_path.glob("frame_*.tif"))
    assert len(tiffs) == 5
    # Manifest written + parseable.
    assert res.manifest_path is not None
    s = Session.load_json(res.manifest_path)
    assert len(s) == 5
    assert s.operator == "karin"
    assert s.sample_id == "test"


def test_runner_frame_indices_monotonic(tmp_path):
    cam = MockCamera(size_px=16)
    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(interval_s=10.0, total_frames=4),
        out_dir=tmp_path,
        clock=_FakeClock(),
    )
    res = runner.run()
    indices = [f.index for f in res.session.frames]
    assert indices == [0, 1, 2, 3]


def test_runner_timestamps_advance_by_interval(tmp_path):
    """Per-frame UTC timestamps should land at start, start+interval,
    start+2*interval, ... within the fake clock's tick precision."""
    cam = MockCamera(size_px=8)
    clock = _FakeClock()
    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(interval_s=60.0, total_frames=3),
        out_dir=tmp_path,
        clock=clock,
    )
    res = runner.run()
    ts = [f.timestamp_s for f in res.session.frames]
    assert ts[1] - ts[0] == pytest.approx(60.0)
    assert ts[2] - ts[1] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Happy path: max_duration cap
# ---------------------------------------------------------------------------

def test_runner_stops_at_max_duration(tmp_path):
    """3 frames fit in 200s @ 100s interval; 4th would exceed it."""
    cam = MockCamera(size_px=8)
    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(
            interval_s=100.0, max_duration_s=200.0,
        ),
        out_dir=tmp_path,
        clock=_FakeClock(),
    )
    res = runner.run()
    # Frames at t=0, 100, 200 — at t=200 the duration check trips
    # so we stop with 3 captured.
    assert res.frames_captured == 3


# ---------------------------------------------------------------------------
# Cancel mid-run
# ---------------------------------------------------------------------------

def test_runner_cancel_after_two_frames(tmp_path):
    """Cancel from the on_frame callback after the 2nd frame.
    Manifest should still be written, 2 frames captured."""
    cam = MockCamera(size_px=8)
    runner_holder = []

    def _on_frame(i, _):
        if i == 1:  # after 2nd frame (indices 0, 1)
            runner_holder[0].cancel()

    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(interval_s=10.0, total_frames=10),
        out_dir=tmp_path,
        on_frame=_on_frame,
        clock=_FakeClock(),
    )
    runner_holder.append(runner)
    res = runner.run()
    assert res.cancelled
    assert res.frames_captured == 2
    assert res.manifest_path is not None
    assert res.manifest_path.exists()


# ---------------------------------------------------------------------------
# Camera errors
# ---------------------------------------------------------------------------

class _FailingCamera:
    """Camera that fails at construction-controlled grab indices."""

    def __init__(self, *, fail_grab_at: list[int]):
        self._fail = set(fail_grab_at)
        self._size = (16, 16)
        self._n = 0

    @property
    def size(self):
        return self._size

    @property
    def fps(self):
        return 30.0

    def start(self):
        self._n = 0

    def stop(self):
        pass

    def grab(self):
        i = self._n
        self._n += 1
        if i in self._fail:
            raise RuntimeError(f"simulated grab failure at {i}")
        return np.zeros(self._size, dtype=np.float32)


def test_grab_failure_records_gap_and_continues(tmp_path):
    cam = _FailingCamera(fail_grab_at=[1])
    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(interval_s=5.0, total_frames=4),
        out_dir=tmp_path,
        clock=_FakeClock(),
    )
    res = runner.run()
    # 4 manifest entries — the failed grab is annotated, not skipped.
    assert len(res.session.frames) == 4
    notes = [f.notes for f in res.session.frames]
    assert notes[1] == "grab_failed"
    # Other 3 wrote TIFFs to disk.
    assert len(list(tmp_path.glob("frame_*.tif"))) == 3


class _StartFailsCamera:
    @property
    def size(self):
        return (8, 8)

    @property
    def fps(self):
        return 30.0

    def start(self):
        raise RuntimeError("camera unplugged")

    def stop(self):
        pass

    def grab(self):
        raise RuntimeError("never reached")


def test_camera_start_failure_short_circuits(tmp_path):
    runner = TimelapseRunner(
        camera=_StartFailsCamera(),
        schedule=TimelapseSchedule(interval_s=10.0, total_frames=3),
        out_dir=tmp_path,
        clock=_FakeClock(),
    )
    res = runner.run()
    assert res.frames_captured == 0
    # Manifest still written so audit + retry tooling can inspect.
    assert res.manifest_path is not None
    s = Session.load_json(res.manifest_path)
    assert s.params.get("error", "").startswith("camera_start")


# ---------------------------------------------------------------------------
# start_at delay
# ---------------------------------------------------------------------------

def test_start_at_delays_first_capture(tmp_path):
    """``start_at`` 30 s in the future → fake clock advances 30 s
    before the first grab. Total elapsed ≈ 30s + 2 intervals
    for 3 captures."""
    clock = _FakeClock(start=datetime(2026, 4, 28, 12, 0, 0,
                                       tzinfo=timezone.utc))
    cam = MockCamera(size_px=8)
    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(
            interval_s=10.0, total_frames=3,
            start_at=datetime(2026, 4, 28, 12, 0, 30,
                              tzinfo=timezone.utc),
        ),
        out_dir=tmp_path,
        clock=clock,
    )
    res = runner.run()
    assert res.frames_captured == 3
    # First frame's timestamp is ≥ start_at.
    first_ts = res.session.frames[0].timestamp_s
    expected = datetime(2026, 4, 28, 12, 0, 30,
                        tzinfo=timezone.utc).timestamp()
    assert first_ts >= expected


# ---------------------------------------------------------------------------
# Progress + frame callbacks
# ---------------------------------------------------------------------------

def test_progress_callback_fires_per_frame(tmp_path):
    progress = []
    cam = MockCamera(size_px=8)
    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(interval_s=1.0, total_frames=4),
        out_dir=tmp_path,
        on_progress=lambda i, total: progress.append((i, total)),
        clock=_FakeClock(),
    )
    runner.run()
    assert len(progress) == 4
    assert progress[0] == (0, 4)
    assert progress[-1] == (3, 4)


def test_progress_callback_total_is_none_when_unbounded(tmp_path):
    progress = []
    cam = MockCamera(size_px=8)

    def _stop_after_two(i, total):
        progress.append((i, total))
        if i == 1:
            runner.cancel()  # noqa: F821 — closure ref

    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(interval_s=1.0),  # unbounded
        out_dir=tmp_path,
        on_progress=_stop_after_two,
        clock=_FakeClock(),
    )
    runner.run()
    # `total` should be None — no schedule cap.
    assert all(t is None for _, t in progress)


def test_on_frame_receives_array(tmp_path):
    seen = []
    cam = MockCamera(size_px=16)
    runner = TimelapseRunner(
        camera=cam,
        schedule=TimelapseSchedule(interval_s=1.0, total_frames=2),
        out_dir=tmp_path,
        on_frame=lambda i, arr: seen.append((i, arr.shape)),
        clock=_FakeClock(),
    )
    runner.run()
    assert seen == [(0, (16, 16)), (1, (16, 16))]
