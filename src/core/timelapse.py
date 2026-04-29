"""Time-lapse acquisition runner — v2.1.x sprint, H5.

Continuous live feed (``ui2.camera_feed.AcquisitionThread``) is
the wrong tool for the lab's overnight cell-cycle imaging — a
12-hour session at 5-minute intervals doesn't need 30 fps, it
needs **one frame every 300 seconds**. This module is the
time-lapse counterpart: a scheduler that triggers ``camera.grab()``
on a timer, writes each frame as it arrives, and produces a
:class:`core.session.Session` manifest the rest of the v2.0.7
pipeline (CLI runner, CSV export) consumes.

Capabilities
------------
* **interval_s** — seconds between captures. Required.
* **total_frames** — stop after N frames. None → run until
  ``max_duration_s`` or cancel.
* **max_duration_s** — stop after this many wall-clock seconds.
  None → run until ``total_frames`` or cancel.
* **start_at** — optional UTC datetime to delay the first
  capture (for "begin at 18:00 then run overnight").
* Cancel support — :meth:`TimelapseRunner.cancel` from any
  thread (SIGINT handler / UI button); the in-flight frame
  finishes, partial outputs flush, manifest still written.

What it does NOT do
-------------------
* Live preview during the long sleeps — the camera is
  parked between captures so this isn't a live-view scope.
  Lab uses live preview to set up the field of view, then
  switches to time-lapse mode.
* Stage automation. Out of scope; this is single-position
  acquisition with one camera.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

from .session import HologramFrame, Session


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimelapseSchedule:
    """How / when to capture frames.

    Attributes
    ----------
    interval_s
        Wall-clock seconds between captures. Must be > 0.
    total_frames
        Stop after this many frames. ``None`` → unbounded by
        count.
    max_duration_s
        Stop after this many wall-clock seconds. ``None`` →
        unbounded by time.
    start_at
        Optional UTC datetime to delay the first capture. The
        runner sleeps until ``now() >= start_at`` then proceeds.
        ``None`` → start immediately.

    At least one of ``total_frames`` / ``max_duration_s`` should
    be set unless caller wires a cancel — otherwise the runner
    loops forever.
    """
    interval_s: float
    total_frames: Optional[int] = None
    max_duration_s: Optional[float] = None
    start_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError(
                f"interval_s must be > 0, got {self.interval_s}",
            )
        if self.total_frames is not None and self.total_frames <= 0:
            raise ValueError(
                f"total_frames must be > 0 or None, "
                f"got {self.total_frames}",
            )
        if self.max_duration_s is not None and self.max_duration_s <= 0:
            raise ValueError(
                f"max_duration_s must be > 0 or None, "
                f"got {self.max_duration_s}",
            )

    def expected_frame_count(self) -> Optional[int]:
        """Plan-ahead estimate: smaller of ``total_frames`` and
        ``max_duration_s / interval_s``. ``None`` when both are
        unbounded (cancel is the only way out)."""
        bounds: List[int] = []
        if self.total_frames is not None:
            bounds.append(int(self.total_frames))
        if self.max_duration_s is not None:
            bounds.append(int(self.max_duration_s // self.interval_s) + 1)
        if not bounds:
            return None
        return min(bounds)


# ---------------------------------------------------------------------------
# Clock injection — tests pass a fake one
# ---------------------------------------------------------------------------

class _RealClock:
    """Wall-clock + sleep abstraction. Keeps tests fast (a 12-hour
    schedule under a fake clock takes microseconds)."""

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float, cancel_event: threading.Event) -> None:
        """Cancel-aware sleep — wakes up at most every 100 ms to
        check the event so cancel latency stays bounded."""
        end = self.monotonic() + max(0.0, seconds)
        while True:
            remaining = end - self.monotonic()
            if remaining <= 0:
                return
            if cancel_event.is_set():
                return
            time.sleep(min(remaining, 0.1))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class TimelapseResult:
    """What the runner gives back after stopping."""
    session: Session
    frames_captured: int
    elapsed_s: float
    cancelled: bool
    manifest_path: Optional[Path] = None
    paths_written: List[Path] = field(default_factory=list)


class TimelapseRunner:
    """Run a :class:`TimelapseSchedule` against a camera + recorder.

    Parameters
    ----------
    camera
        Anything implementing :class:`CameraSource` (synthetic,
        mock, or vendor backend with SDK installed).
    schedule
        :class:`TimelapseSchedule` controlling cadence + stop
        conditions.
    out_dir
        Output directory. Per-frame TIFFs land at
        ``out_dir / frame_<NNNN>.tif``; the session manifest at
        ``out_dir / session.json``.
    sample_id, operator
        Forwarded to the :class:`Session` for audit / paper bundle.
    on_progress
        Optional callable ``(frame_idx, expected_total)`` — UI
        hook. ``expected_total`` may be ``None`` for unbounded
        runs.
    on_frame
        Optional callable ``(frame_idx, ndarray)`` — useful for
        a live preview thumbnail of the most recent capture.

    The runner is **single-shot**: instantiate, call
    :meth:`run`, dispose. Re-running needs a fresh instance.
    """

    def __init__(self,
                 camera: Any,                       # CameraSource
                 schedule: TimelapseSchedule,
                 out_dir: Path,
                 *,
                 sample_id: str = "",
                 operator: str = "",
                 params: Optional[dict] = None,
                 on_progress: Optional[Callable[[int, Optional[int]], None]] = None,
                 on_frame: Optional[Callable[[int, np.ndarray], None]] = None,
                 clock: Optional[_RealClock] = None,
                 ) -> None:
        self.camera = camera
        self.schedule = schedule
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.sample_id = sample_id
        self.operator = operator
        self.params = dict(params or {})
        self._on_progress = on_progress
        self._on_frame = on_frame
        self._clock = clock or _RealClock()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request stop. Any thread can call. Latency ≤ 100 ms +
        the in-flight grab/write."""
        self._cancel_event.set()

    def run(self) -> TimelapseResult:
        """Execute the schedule. Returns when the schedule
        completes, the cancel flag is set, or an unrecoverable
        camera error occurs (all surfaced via the result, never
        as a raised exception)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Save tifffile import to runtime — keeps headless smoke
        # imports fast.
        try:
            import tifffile
        except ImportError:
            tifffile = None  # type: ignore[assignment]

        session = Session.new(
            operator=self.operator,
            sample_id=self.sample_id,
            params=self.params,
            root_dir=str(self.out_dir),
        )

        # Honour start_at by sleeping until then.
        if self.schedule.start_at is not None:
            now = self._clock.now_utc()
            if now < self.schedule.start_at:
                wait_s = (self.schedule.start_at - now).total_seconds()
                self._clock.sleep(wait_s, self._cancel_event)

        # Camera lifecycle: start once, stop in finally so a
        # mid-run error doesn't leak the device.
        try:
            self.camera.start()
        except Exception as exc:
            _LOG.exception("timelapse: camera start failed")
            session.params["error"] = f"camera_start: {exc}"
            manifest_path = self._save_manifest(session)
            return TimelapseResult(
                session=session, frames_captured=0,
                elapsed_s=0.0, cancelled=False,
                manifest_path=manifest_path,
            )

        expected = self.schedule.expected_frame_count()
        paths: List[Path] = []
        idx = 0
        t0_mono = self._clock.monotonic()
        cancelled = False
        try:
            while True:
                if self._cancel_event.is_set():
                    cancelled = True
                    break

                # Capture window — grab frame, write to disk, append
                # to session.
                ts = self._clock.now_utc()
                frame_path = self.out_dir / f"frame_{idx:04d}.tif"
                try:
                    arr = self.camera.grab()
                except Exception as exc:
                    _LOG.warning(
                        "timelapse: grab failed at frame %d: %s",
                        idx, exc,
                    )
                    # Record a placeholder frame so the manifest
                    # captures the gap; the runner keeps going.
                    arr = None

                if arr is not None and tifffile is not None:
                    try:
                        # 16-bit preserves dynamic range without
                        # the 4× cost of float32. Same convention
                        # as TiffStackRecorder.
                        u16 = np.clip(arr, 0.0, 1.0)
                        u16 = (u16 * 65535.0).astype(np.uint16)
                        tifffile.imwrite(str(frame_path), u16)
                        paths.append(frame_path)
                    except Exception as exc:
                        _LOG.warning(
                            "timelapse: write failed at frame %d: %s",
                            idx, exc,
                        )

                # Append to session regardless of write success —
                # the gap shows up in downstream CSV as an error
                # row (see core.session_export.FrameResult).
                session.frames.append(HologramFrame(
                    path=frame_path.name,
                    timestamp_s=float(ts.timestamp()),
                    index=idx,
                    notes="" if arr is not None else "grab_failed",
                ))

                if self._on_frame and arr is not None:
                    try:
                        self._on_frame(idx, arr)
                    except Exception:
                        _LOG.debug("on_frame callback raised",
                                   exc_info=True)
                if self._on_progress:
                    try:
                        self._on_progress(idx, expected)
                    except Exception:
                        _LOG.debug("on_progress callback raised",
                                   exc_info=True)

                idx += 1

                # Stop conditions — check BEFORE the next sleep
                # so a "1 frame" run doesn't sleep for 5 minutes
                # at the end.
                if self.schedule.total_frames is not None and \
                        idx >= self.schedule.total_frames:
                    break
                if self.schedule.max_duration_s is not None and \
                        (self._clock.monotonic() - t0_mono) \
                        >= self.schedule.max_duration_s:
                    break

                # Inter-frame sleep (cancel-aware).
                self._clock.sleep(
                    self.schedule.interval_s, self._cancel_event,
                )
        finally:
            try:
                self.camera.stop()
            except Exception:
                _LOG.debug("timelapse: camera stop raised",
                           exc_info=True)

        elapsed = self._clock.monotonic() - t0_mono
        manifest_path = self._save_manifest(session)
        return TimelapseResult(
            session=session,
            frames_captured=idx,
            elapsed_s=elapsed,
            cancelled=cancelled,
            manifest_path=manifest_path,
            paths_written=paths,
        )

    def _save_manifest(self, session: Session) -> Optional[Path]:
        manifest = self.out_dir / "session.json"
        try:
            session.save_json(manifest)
            return manifest
        except Exception:
            _LOG.exception("timelapse: failed to save manifest")
            return None


__all__ = [
    "TimelapseSchedule",
    "TimelapseResult",
    "TimelapseRunner",
]
