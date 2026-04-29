"""Multi-position time-lapse runner tests.

Exercises the bridge between :mod:`core.timelapse` (interval) and
:mod:`core.devices.orchestrator` (multi-position).
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

from core.devices.orchestrator import StagePosition  # noqa: E402
from core.multi_position_timelapse import (  # noqa: E402
    MultiPositionResult,
    MultiPositionSchedule,
    MultiPositionTimelapseRunner,
)
from core.session import Session  # noqa: E402


# ---------------------------------------------------------------------------
# Fake clock — copy of the timelapse test pattern
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self, *, start: datetime = None) -> None:
        self._t0_utc = (
            start or datetime(2026, 4, 30, 12, 0, 0,
                              tzinfo=timezone.utc)
        )
        self._t_mono = 0.0
        self._t_utc = self._t0_utc

    def now_utc(self) -> datetime:
        return self._t_utc

    def monotonic(self) -> float:
        return self._t_mono

    def sleep(self, seconds: float,
              cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            return
        self._t_mono += float(seconds)
        self._t_utc = self._t_utc + timedelta(seconds=float(seconds))


# ---------------------------------------------------------------------------
# Stub camera + stage
# ---------------------------------------------------------------------------

class _StubCamera:
    def __init__(self):
        self._n = 0
        self._size = (8, 8)

    @property
    def size(self):
        return self._size

    @property
    def fps(self):
        return 30.0

    def start(self):
        pass

    def stop(self):
        pass

    def grab(self):
        self._n += 1
        return np.full(self._size, float(self._n), dtype=np.float32)


def _make_stage():
    from core.devices import make_device
    s = make_device("mock_stage")
    s.connect()
    return s


# ---------------------------------------------------------------------------
# Schedule validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interval", [0, -1])
def test_schedule_rejects_non_positive_interval(interval):
    with pytest.raises(ValueError, match="interval_s"):
        MultiPositionSchedule(interval_s=interval)


def test_schedule_rejects_zero_total_ticks():
    with pytest.raises(ValueError, match="total_ticks"):
        MultiPositionSchedule(interval_s=10.0, total_ticks=0)


def test_schedule_expected_total_frames():
    s = MultiPositionSchedule(
        interval_s=300.0,
        total_ticks=144,
        positions=[StagePosition(0, 0), StagePosition(100, 0)],
    )
    # 144 ticks × 2 positions = 288 frames.
    assert s.expected_total_frames() == 288


def test_schedule_unbounded_returns_none():
    s = MultiPositionSchedule(interval_s=300.0)
    assert s.expected_total_frames() is None


def test_schedule_as_simple_timelapse_round_trips():
    s = MultiPositionSchedule(
        interval_s=120.0, total_ticks=10, max_duration_s=2400.0,
    )
    simple = s.as_simple_timelapse()
    assert simple.interval_s == 120.0
    assert simple.total_frames == 10
    assert simple.max_duration_s == 2400.0


# ---------------------------------------------------------------------------
# Happy path: 3 ticks × 4 positions = 12 frames
# ---------------------------------------------------------------------------

def test_runner_writes_frames_and_manifest(tmp_path):
    sched = MultiPositionSchedule(
        interval_s=300.0,
        total_ticks=3,
        positions=[
            StagePosition(0, 0, label="A"),
            StagePosition(100, 0, label="B"),
            StagePosition(200, 0, label="C"),
            StagePosition(300, 0, label="D"),
        ],
        settle_time_s=0.0,
    )
    runner = MultiPositionTimelapseRunner(
        sched, tmp_path,
        camera=_StubCamera(),
        stage=_make_stage(),
        sample_id="multi_pos", operator="karin",
        clock=_FakeClock(),
    )
    res = runner.run()
    assert res.ticks_completed == 3
    assert res.frames_total == 12
    # Per-tick TIFFs landed.
    tiffs = sorted(tmp_path.glob("frame_*.tif"))
    assert len(tiffs) == 12
    # Manifest readable + has 12 frames.
    s = Session.load_json(res.manifest_path)
    assert len(s) == 12
    # Position labels folded into notes.
    notes = [f.notes for f in s.frames]
    assert any("pos=A" in n for n in notes)
    assert any("pos=D" in n for n in notes)


def test_per_frame_overrides_carry_position_metadata(tmp_path):
    """Position info lives on each HologramFrame.params_overrides
    so the CSV exporter can pivot."""
    sched = MultiPositionSchedule(
        interval_s=10.0, total_ticks=2,
        positions=[StagePosition(50, 75, 100, label="X")],
    )
    runner = MultiPositionTimelapseRunner(
        sched, tmp_path,
        camera=_StubCamera(),
        stage=_make_stage(),
        clock=_FakeClock(),
    )
    res = runner.run()
    f = res.session.frames[0]
    assert f.params_overrides["x_um"] == 50.0
    assert f.params_overrides["y_um"] == 75.0
    assert f.params_overrides["z_um"] == 100.0
    assert f.params_overrides["tick"] == 0
    # Tick 1 frame.
    f1 = res.session.frames[1]
    assert f1.params_overrides["tick"] == 1


def test_max_duration_cap(tmp_path):
    """3 ticks fit in 200 s @ 100 s interval; 4th would exceed."""
    sched = MultiPositionSchedule(
        interval_s=100.0,
        max_duration_s=200.0,
        positions=[StagePosition(0, 0)],
    )
    runner = MultiPositionTimelapseRunner(
        sched, tmp_path,
        camera=_StubCamera(),
        stage=_make_stage(),
        clock=_FakeClock(),
    )
    res = runner.run()
    # 3 ticks * 1 position = 3 frames.
    assert res.ticks_completed == 3
    assert res.frames_total == 3


def test_cancel_mid_run(tmp_path):
    sched = MultiPositionSchedule(
        interval_s=10.0, total_ticks=10,
        positions=[StagePosition(0, 0), StagePosition(100, 0)],
    )
    runner_holder: list = [None]

    def _on_tick(tick_idx, n_pos):
        if tick_idx >= 2:
            runner_holder[0].cancel()

    runner = MultiPositionTimelapseRunner(
        sched, tmp_path,
        camera=_StubCamera(),
        stage=_make_stage(),
        on_tick=_on_tick,
        clock=_FakeClock(),
    )
    runner_holder[0] = runner
    res = runner.run()
    assert res.cancelled is True
    # 2 ticks completed before cancel.
    assert res.ticks_completed == 2
    assert res.frames_total == 4


def test_callbacks_fire_in_order(tmp_path):
    seen_ticks = []
    seen_frames = []
    sched = MultiPositionSchedule(
        interval_s=1.0, total_ticks=2,
        positions=[StagePosition(i * 100, 0) for i in range(3)],
    )
    runner = MultiPositionTimelapseRunner(
        sched, tmp_path,
        camera=_StubCamera(),
        stage=_make_stage(),
        on_tick=lambda t, n: seen_ticks.append((t, n)),
        on_frame=lambda t, rec: seen_frames.append(
            (t, rec.position.x_um),
        ),
        clock=_FakeClock(),
    )
    runner.run()
    # 2 ticks × 3 positions = 6 frame callbacks.
    assert len(seen_frames) == 6
    # Ticks fire after each tick completes (after 3 frames).
    assert seen_ticks == [(1, 3), (2, 3)]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_positions_runs_one_grab_per_tick(tmp_path):
    """Empty positions = single grab at current location each tick.
    Mirrors run_plan's "no waypoints" behaviour."""
    sched = MultiPositionSchedule(
        interval_s=5.0, total_ticks=4, positions=[],
    )
    runner = MultiPositionTimelapseRunner(
        sched, tmp_path,
        camera=_StubCamera(),
        clock=_FakeClock(),
    )
    res = runner.run()
    assert res.ticks_completed == 4
    assert res.frames_total == 4


def test_manifest_robust_to_tifffile_missing(tmp_path, monkeypatch):
    """When tifffile isn't installed the runner still records the
    manifest — just with no on-disk TIFFs. Lab can rerun later."""
    import builtins
    real_import = builtins.__import__

    def _no_tiff(name, *a, **kw):
        if name == "tifffile":
            raise ImportError("simulated")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_tiff)
    sched = MultiPositionSchedule(
        interval_s=1.0, total_ticks=1,
        positions=[StagePosition(0, 0)],
    )
    runner = MultiPositionTimelapseRunner(
        sched, tmp_path,
        camera=_StubCamera(),
        stage=_make_stage(),
        clock=_FakeClock(),
    )
    res = runner.run()
    # Manifest written.
    assert res.manifest_path is not None
    assert res.manifest_path.exists()
    # No TIFFs (tifffile was missing).
    assert list(tmp_path.glob("frame_*.tif")) == []
