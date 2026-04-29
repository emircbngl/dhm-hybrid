"""Multi-position time-lapse bridge — v2.1.z follow-up.

Karin's overnight scenario from the Lindqvist Lab review: **24
positions, every 5 minutes, for 12 hours**. The two existing
runners cover one half each:

* :class:`core.timelapse.TimelapseRunner` — interval-based
  capture at one position.
* :func:`core.devices.orchestrator.run_plan` — multi-position
  capture in one shot.

This module composes them. ``MultiPositionTimelapseRunner``
schedules one **plan** (set of stage waypoints) per interval
tick, writes one ``frame_<position>_<tick>.tif`` per capture,
and produces a single :class:`core.session.Session` manifest the
v2.0.7 CLI runner / CSV exporter consumes.

Why a separate module
---------------------
Cramming this into ``timelapse.py`` would couple the simple
single-position runner to the device registry. Keeping it
separate lets headless rigs (no stage) run :class:`TimelapseRunner`
without ever importing ``core.devices``, while multi-position
labs opt in via this bridge.

Public API
----------
* :class:`MultiPositionSchedule` — cadence + list of positions +
  per-tick plan settings.
* :class:`MultiPositionTimelapseRunner` — single-shot runner.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from .devices.orchestrator import (
    AcquisitionPlan, FrameRecord, StagePosition, run_plan,
)
from .session import HologramFrame, Session
from .timelapse import TimelapseSchedule


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiPositionSchedule:
    """When + where to acquire.

    Attributes
    ----------
    interval_s
        Seconds between ticks (each tick visits every position).
    total_ticks
        Stop after this many ticks. ``None`` → unbounded by count.
    max_duration_s
        Stop after this many wall-clock seconds. ``None`` →
        unbounded by time.
    positions
        Stage waypoints visited every tick, in order.
    led_intensity_percent
        Optional LED setting, applied once per tick. ``None`` →
        no LED action.
    shutter_per_frame
        Open/close shutter around every grab when ``True``;
        otherwise leave open for the whole tick.
    settle_time_s
        Wait after each move before grabbing.
    """
    interval_s: float
    positions: Sequence[StagePosition] = ()
    total_ticks: Optional[int] = None
    max_duration_s: Optional[float] = None
    led_intensity_percent: Optional[float] = None
    shutter_per_frame: bool = True
    settle_time_s: float = 0.05

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError(
                f"interval_s must be > 0, got {self.interval_s}",
            )
        if self.total_ticks is not None and self.total_ticks <= 0:
            raise ValueError(
                f"total_ticks must be > 0 or None, "
                f"got {self.total_ticks}",
            )
        if self.max_duration_s is not None and self.max_duration_s <= 0:
            raise ValueError(
                f"max_duration_s must be > 0 or None, "
                f"got {self.max_duration_s}",
            )

    def expected_total_frames(self) -> Optional[int]:
        """Plan-ahead estimate. ``None`` = unbounded; useful for
        progress bars."""
        n_pos = max(1, len(self.positions))
        bounds: List[int] = []
        if self.total_ticks is not None:
            bounds.append(int(self.total_ticks) * n_pos)
        if self.max_duration_s is not None:
            bounds.append(
                (int(self.max_duration_s // self.interval_s) + 1) * n_pos,
            )
        if not bounds:
            return None
        return min(bounds)

    def as_simple_timelapse(self) -> TimelapseSchedule:
        """Return the equivalent single-position
        :class:`TimelapseSchedule` so callers can hand the cadence
        layer off to :class:`TimelapseRunner` if positions is empty."""
        return TimelapseSchedule(
            interval_s=self.interval_s,
            total_frames=self.total_ticks,
            max_duration_s=self.max_duration_s,
        )


@dataclass
class MultiPositionResult:
    """What the runner returns once stopped.

    ``session`` carries the manifest with every captured frame as
    a :class:`HologramFrame` (notes field encodes position label
    so the CSV exporter can pivot by position). ``per_tick`` is
    the parallel list of :class:`PlanResult` objects for tools
    that want the raw orchestrator outputs.
    """
    session: Session
    per_tick: List[Any] = field(default_factory=list)
    ticks_completed: int = 0
    frames_total: int = 0
    elapsed_s: float = 0.0
    cancelled: bool = False
    manifest_path: Optional[Path] = None


class _RealClock:
    """Same shape as core.timelapse._RealClock — duplicated so this
    module doesn't reach into a sibling private. Tests inject a
    :class:`_FakeClock`-style stand-in."""

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        import time
        return time.monotonic()

    def sleep(self, seconds: float, cancel_event: threading.Event) -> None:
        import time
        end = self.monotonic() + max(0.0, seconds)
        while True:
            remaining = end - self.monotonic()
            if remaining <= 0:
                return
            if cancel_event.is_set():
                return
            time.sleep(min(remaining, 0.1))


class MultiPositionTimelapseRunner:
    """Drive a :class:`MultiPositionSchedule` against camera + stage
    (+ optional shutter + LED).

    Per-tick flow:
    1. Build an :class:`AcquisitionPlan` from the schedule.
    2. Run the plan via :func:`run_plan`.
    3. For each :class:`FrameRecord`, write a TIFF +
       :class:`HologramFrame` row in the session manifest.
    4. Sleep until the next tick (or break out on cancel).

    Cancel-aware. Camera-error and stage-error containment delegate
    to ``run_plan`` (per-frame error fields).
    """

    def __init__(self,
                 schedule: MultiPositionSchedule,
                 out_dir: Path,
                 *,
                 camera: Any,
                 stage: Optional[Any] = None,
                 shutter: Optional[Any] = None,
                 led: Optional[Any] = None,
                 sample_id: str = "",
                 operator: str = "",
                 params: Optional[dict] = None,
                 on_tick: Optional[Callable[[int, int], None]] = None,
                 on_frame: Optional[Callable[[int, FrameRecord], None]] = None,
                 clock: Optional[_RealClock] = None,
                 ) -> None:
        self.schedule = schedule
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.camera = camera
        self.stage = stage
        self.shutter = shutter
        self.led = led
        self.sample_id = sample_id
        self.operator = operator
        self.params = dict(params or {})
        self._on_tick = on_tick
        self._on_frame = on_frame
        self._clock = clock or _RealClock()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request stop from any thread. Latency bounded by the
        ``run_plan`` cancel-check granularity (per-position) plus
        the inter-tick sleep slice (≤ 100 ms)."""
        self._cancel_event.set()

    def run(self) -> MultiPositionResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import tifffile  # noqa: WPS433
        except ImportError:
            tifffile = None  # type: ignore[assignment]

        session = Session.new(
            operator=self.operator,
            sample_id=self.sample_id,
            params=self.params,
            root_dir=str(self.out_dir),
        )

        result = MultiPositionResult(session=session)
        n_pos = max(1, len(self.schedule.positions))
        t0 = self._clock.monotonic()
        tick = 0

        # Build the per-tick AcquisitionPlan once — only the LED
        # intensity / shutter mode / settle apply across ticks; the
        # positions are fixed for the whole session.
        plan = AcquisitionPlan(
            positions=self.schedule.positions,
            led_intensity_percent=self.schedule.led_intensity_percent,
            shutter_per_frame=self.schedule.shutter_per_frame,
            settle_time_s=self.schedule.settle_time_s,
        )

        while True:
            if self._cancel_event.is_set():
                result.cancelled = True
                break

            # One tick = one plan execution.
            tick_result = run_plan(
                plan,
                camera=self.camera,
                stage=self.stage,
                shutter=self.shutter,
                led=self.led,
                cancel_check=self._cancel_event.is_set,
                on_frame=lambda rec, _t=tick:
                    self._handle_frame(_t, rec, tifffile, session, result),
            )
            result.per_tick.append(tick_result)
            if tick_result.cancelled:
                result.cancelled = True
                break

            tick += 1
            result.ticks_completed = tick
            if self._on_tick:
                try:
                    self._on_tick(tick, n_pos)
                except Exception:
                    _LOG.debug("on_tick raised", exc_info=True)

            # Stop conditions.
            if (self.schedule.total_ticks is not None
                    and tick >= self.schedule.total_ticks):
                break
            if (self.schedule.max_duration_s is not None
                    and self._clock.monotonic() - t0
                    >= self.schedule.max_duration_s):
                break

            self._clock.sleep(
                self.schedule.interval_s, self._cancel_event,
            )

        result.elapsed_s = self._clock.monotonic() - t0
        result.frames_total = len(session.frames)
        try:
            manifest_path = self.out_dir / "session.json"
            session.save_json(manifest_path)
            result.manifest_path = manifest_path
        except Exception:
            _LOG.exception("multi-position timelapse: manifest save failed")
        return result

    def _handle_frame(self, tick: int, rec: FrameRecord,
                      tifffile: Any, session: Session,
                      result: MultiPositionResult) -> None:
        """Per-position callback: write TIFF + append manifest row."""
        idx = len(session.frames)
        # Position label folds into the frame's notes — CSV
        # exporter can pivot by parsing this back out.
        note = (
            f"tick={tick};pos={rec.position.label or rec.index};"
            f"x_um={rec.position.x_um};y_um={rec.position.y_um};"
            f"z_um={rec.position.z_um}"
        )
        if rec.error:
            note += f";error={rec.error}"

        path = (
            f"frame_t{tick:04d}_p{rec.index:03d}.tif"
        )
        if rec.frame is not None and tifffile is not None:
            try:
                u16 = (np.clip(np.asarray(rec.frame, dtype=np.float32),
                               0.0, 1.0) * 65535.0).astype(np.uint16)
                tifffile.imwrite(str(self.out_dir / path), u16)
            except Exception as exc:
                note += f";write_error={exc}"

        session.frames.append(HologramFrame(
            path=path,
            timestamp_s=float(self._clock.now_utc().timestamp()),
            index=idx,
            params_overrides={
                "tick": tick,
                "position_index": rec.index,
                "x_um": rec.position.x_um,
                "y_um": rec.position.y_um,
                "z_um": rec.position.z_um,
            },
            notes=note,
        ))
        if self._on_frame:
            try:
                self._on_frame(tick, rec)
            except Exception:
                _LOG.debug("on_frame raised", exc_info=True)


__all__ = [
    "MultiPositionSchedule",
    "MultiPositionResult",
    "MultiPositionTimelapseRunner",
]
