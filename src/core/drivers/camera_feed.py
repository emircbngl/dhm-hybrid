"""Camera live-feed + TIFF stack recording for the v2 UI.

v1 shipped a PySide-coupled NICamera stream + recording pipeline. The
v2 port re-implements the same contract (camera source → acquisition
thread → frame mailbox) but:

* backend-agnostic via :class:`CameraSource` Protocol,
* ships a :class:`SyntheticCamera` fallback so a dev laptop can
  exercise the full UI path without any hardware,
* pushes frames through a single-slot mailbox (the render loop only
  paints the newest frame, dropped frames are fine for live preview),
* lets the recording path open a separate :class:`TiffStackRecorder`
  that sits in the same worker thread and writes each frame as it's
  grabbed — no second copy, no ordering races.

Hardware support (NICamera, OpenCV, Pylon, …) lands by implementing
:class:`CameraSource` in a sibling module; ``CAMERA_SOURCES`` in
:mod:`ui2.app` decides which one the dialog picks.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source protocol + synthetic default
# ---------------------------------------------------------------------------

@runtime_checkable
class CameraSource(Protocol):
    """Minimal contract a camera implementation has to meet.

    ``start`` / ``stop`` are infallible (or at least no-raise for the
    synthetic case); ``grab`` returns a 2-D float32 frame normalised
    to ``[0, 1]``. Shape / fps / noise are implementation choices."""
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def grab(self) -> np.ndarray: ...

    @property
    def size(self) -> tuple[int, int]: ...
    @property
    def fps(self) -> float: ...


@dataclass
class SyntheticCamera:
    """Animated interference-fringe frame generator — useful for
    exercising the UI path without hardware.

    The pattern looks enough like a real DHM acquisition (carrier
    fringes + a slowly-moving Gaussian envelope) that downstream QPI
    / autofocus code gets something believable, but it's purely
    deterministic so automated tests don't flake."""
    size_px: int = 512
    target_fps: float = 30.0
    rng_seed: int = 42

    def __post_init__(self) -> None:
        self._frame = 0
        self._rng = np.random.default_rng(self.rng_seed)
        self._started = False

    def start(self) -> None:
        self._frame = 0
        self._started = True

    def stop(self) -> None:
        self._started = False

    def grab(self) -> np.ndarray:
        if not self._started:
            # Tolerate grab-before-start so callers don't have to sync
            # explicitly; nothing catastrophic happens.
            self._started = True
        self._frame += 1
        n = self.size_px
        # Off-axis carrier fringes + a slowly drifting envelope.
        y, x = np.mgrid[:n, :n].astype(np.float32)
        carrier = 0.5 + 0.35 * np.sin(
            2 * np.pi * (0.15 * x + 0.02 * self._frame)
        )
        cx = n / 2 + 30 * np.cos(self._frame * 0.05)
        cy = n / 2 + 30 * np.sin(self._frame * 0.03)
        env = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2)
                       / (0.2 * n) ** 2))
        noise = self._rng.normal(0.0, 0.01, size=(n, n)).astype(np.float32)
        frame = carrier * (0.6 + 0.4 * env) + noise
        return np.clip(frame, 0.0, 1.0).astype(np.float32)

    @property
    def size(self) -> tuple[int, int]:
        return (self.size_px, self.size_px)

    @property
    def fps(self) -> float:
        return float(self.target_fps)


# ---------------------------------------------------------------------------
# TIFF stack recorder — lazy tifffile import, append-on-each-frame
# ---------------------------------------------------------------------------

class TiffStackRecorder:
    """Persist each grabbed frame to a multi-page TIFF as it arrives.

    ``tifffile.TiffWriter(..., append=False)`` on first write then
    subsequent ``.write(...)`` calls append. bigtiff=True so long
    sessions don't hit the 4 GiB cap. Files are closed on ``stop()``
    so a mid-session crash doesn't leave an invalid TIFF.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._writer = None
        self._frames_written = 0

    def start(self) -> None:
        import tifffile  # lazy so base dep stays light
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = tifffile.TiffWriter(str(self.path), bigtiff=True)
        self._frames_written = 0

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("recorder: write_frame before start()")
        # 16-bit preserves dynamic range without the 4x cost of float32.
        f = np.clip(frame, 0.0, 1.0)
        self._writer.write(
            (f * 65535.0).astype(np.uint16),
            photometric="minisblack",
        )
        self._frames_written += 1

    def stop(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None

    @property
    def frames_written(self) -> int:
        return self._frames_written


# ---------------------------------------------------------------------------
# Acquisition thread — grab → mailbox, optional record
# ---------------------------------------------------------------------------

class AcquisitionThread(threading.Thread):
    """Background frame pump.

    Callers supply:
        * ``source`` — any :class:`CameraSource` implementation,
        * ``on_frame`` — a callable invoked with the latest frame
          (the UI main loop paints it),
        * ``on_fps`` — optional heartbeat, invoked once per second
          with the measured rate.

    The thread is daemon — a main-loop crash shouldn't leak a live
    camera. Call :meth:`stop` to end the loop deterministically."""

    def __init__(
        self,
        source: CameraSource,
        on_frame: Callable[[np.ndarray], None],
        on_fps: Optional[Callable[[float], None]] = None,
        recorder: Optional[TiffStackRecorder] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(daemon=True, name="dhm-camera")
        self._source = source
        self._on_frame = on_frame
        self._on_fps = on_fps
        self._recorder = recorder
        # Invoked (with a human-readable reason) when the pump dies on a
        # start/grab failure. Without it the thread logged + died silently and
        # the UI kept the last frame with "recording" still armed — a frozen
        # feed indistinguishable from a live one (2026-07-08 review, B-109).
        self._on_error = on_error
        self._stop_event = threading.Event()

    def _emit_error(self, message: str) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(message)
        except Exception:  # a broken error sink must not mask the original
            _LOG.debug("on_error callback raised", exc_info=True)

    def run(self) -> None:
        try:
            self._source.start()
        except Exception as exc:
            _LOG.exception("camera start failed")
            self._emit_error(f"camera start failed: {exc}")
            return
        target = max(1.0, float(self._source.fps))
        interval = 1.0 / target
        counter = 0
        t_window = time.perf_counter()
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                frame = self._source.grab()
            except Exception as exc:
                _LOG.exception("camera grab failed")
                self._emit_error(f"camera grab failed: {exc}")
                break
            try:
                self._on_frame(frame)
            except Exception:
                _LOG.debug("on_frame callback raised", exc_info=True)
            if self._recorder is not None:
                try:
                    self._recorder.write_frame(frame)
                except Exception:
                    _LOG.warning("recorder.write_frame failed",
                                 exc_info=True)
            counter += 1
            elapsed = time.perf_counter() - t_window
            if elapsed >= 1.0:
                if self._on_fps:
                    try:
                        self._on_fps(counter / elapsed)
                    except Exception:
                        pass
                counter = 0
                t_window = time.perf_counter()
            next_tick += interval
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                # Sleep in small chunks so stop() has ≤ 100 ms latency.
                while sleep_for > 0 and not self._stop_event.is_set():
                    chunk = min(sleep_for, 0.1)
                    time.sleep(chunk)
                    sleep_for -= chunk
            else:
                # We're already late — skip sleeping and recompute
                # the next tick from *now* to avoid cascading lag.
                next_tick = time.perf_counter()
        try:
            self._source.stop()
        except Exception:
            _LOG.debug("camera stop raised", exc_info=True)
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                _LOG.debug("recorder stop raised", exc_info=True)

    def stop(self) -> None:
        self._stop_event.set()


__all__ = [
    "CameraSource",
    "SyntheticCamera",
    "TiffStackRecorder",
    "AcquisitionThread",
]
