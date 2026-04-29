"""Multi-device acquisition orchestrator — the v2.1.z value-add.

The lab pattern that a generic device GUI doesn't capture: **one
acquire call** that does:

1. Open the shutter.
2. Set LED intensity.
3. Move the stage to the next position.
4. Wait for settle.
5. Trigger the camera grab.
6. Close the shutter (or leave open for the next position).
7. Tag the captured frame with the position metadata so the
   downstream Session manifest carries (x_um, y_um, z_um) per
   frame.

Without orchestration the operator opens three dialogs + races
the timing. With it, a 24-position 12-hour time-lapse becomes
one config object.

Design
------
* :class:`AcquisitionPlan` — declarative list of positions + a
  per-frame action list (open/grab/close, etc.). Pure data.
* :func:`run_plan` — drives the plan against a camera + (optional)
  stage / shutter / LED. Returns a :class:`PlanResult` with
  per-frame metadata + outcome (success / error).
* Mock-friendly — stage/shutter/LED can be the mock backends, the
  test suite drives the orchestrator deterministically without
  real hardware.

This is the orchestration layer; the v2.1.x time-lapse runner
covers the simpler "one position, fixed interval" case. The two
compose: an orchestrator plan can itself be triggered by a
time-lapse schedule.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagePosition:
    """One stage waypoint with optional label."""
    x_um: float
    y_um: float
    z_um: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class AcquisitionPlan:
    """Declarative acquisition recipe.

    Attributes
    ----------
    positions
        Ordered stage waypoints. Empty = single-frame at current
        position (no stage motion).
    led_intensity_percent
        Optional LED intensity. ``None`` → no LED action.
    shutter_per_frame
        ``True`` → open before grab, close after. ``False`` →
        leave open for the whole plan (faster, more photobleach).
    settle_time_s
        Wait after each move before grabbing. Must accommodate
        stage settle + camera exposure budget.
    """
    positions: Sequence[StagePosition] = ()
    led_intensity_percent: Optional[float] = None
    shutter_per_frame: bool = True
    settle_time_s: float = 0.05


@dataclass
class FrameRecord:
    """One position's outcome — what landed on disk + the
    position metadata. The Session/CSV exporter consumes this
    via the ``payload`` field."""
    index: int
    position: StagePosition
    frame: Optional[np.ndarray]      # captured image, may be None on error
    error: Optional[str] = None


@dataclass
class PlanResult:
    """Aggregated outcome of one plan run."""
    frames: List[FrameRecord] = field(default_factory=list)
    cancelled: bool = False
    elapsed_s: float = 0.0


def run_plan(
    plan: AcquisitionPlan,
    *,
    camera: Any,                       # CameraSource
    stage: Optional[Any] = None,        # StageDevice
    shutter: Optional[Any] = None,      # ShutterDevice
    led: Optional[Any] = None,          # LEDDevice
    cancel_check: Optional[Callable[[], bool]] = None,
    on_frame: Optional[Callable[[FrameRecord], None]] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> PlanResult:
    """Drive ``plan`` against the supplied devices.

    Lifecycle
    ---------
    1. Connect every supplied device that's not already connected.
    2. If ``led`` + ``led_intensity_percent`` set, apply intensity
       and turn LED on.
    3. If ``shutter_per_frame=False``, open shutter once.
    4. Iterate positions. For each: stage move (if stage given),
       optional shutter open, sleep ``settle_time_s``, grab,
       shutter close (if per-frame).
    5. Cleanup: shutter close, LED off (we leave them connected —
       caller manages connection lifecycle to allow chained plans).

    All exceptions are caught + recorded per-frame; the orchestrator
    never raises mid-plan, but returns a :class:`PlanResult` so
    the caller can decide what to do.

    Parameters
    ----------
    cancel_check
        Polled before each frame; ``True`` aborts the rest of the
        plan. Same shape as the autofocus / timelapse cancel
        contract.
    on_frame
        Per-frame callback (UI hook). Receives the
        :class:`FrameRecord` after capture/error.
    sleep
        Injectable sleep function for tests. Defaults to
        ``time.sleep``.
    """
    import time as _time
    sleep_fn = sleep if sleep is not None else _time.sleep

    # Connect devices that aren't already connected.
    for d in (stage, shutter, led, camera):
        if d is None:
            continue
        is_connected = getattr(d, "is_connected", None)
        if is_connected is None or not is_connected:
            try:
                # Camera uses ``start`` for "begin streaming";
                # other devices use ``connect``.
                if hasattr(d, "connect"):
                    d.connect()
                if hasattr(d, "start"):
                    d.start()
            except Exception as exc:
                _LOG.warning("orchestrator: device init failed: %s", exc)

    # LED setup (once for the whole plan).
    if led is not None:
        if plan.led_intensity_percent is not None:
            try:
                led.set_intensity(float(plan.led_intensity_percent))
            except Exception:
                _LOG.debug("LED set_intensity failed", exc_info=True)
        try:
            led.on()
        except Exception:
            _LOG.debug("LED on failed", exc_info=True)

    # Shutter — leave open for the whole plan if per_frame=False.
    if shutter is not None and not plan.shutter_per_frame:
        try:
            shutter.open()
        except Exception:
            _LOG.debug("shutter open failed", exc_info=True)

    # Empty positions = single grab at current position.
    positions = (plan.positions if plan.positions
                 else [StagePosition(0.0, 0.0, 0.0,
                                     label="current")])

    result = PlanResult()
    t0 = _time.monotonic()
    for i, pos in enumerate(positions):
        if cancel_check and cancel_check():
            result.cancelled = True
            break

        rec = FrameRecord(index=i, position=pos, frame=None)

        # Stage move.
        if stage is not None and plan.positions:
            try:
                stage.move_to(pos.x_um, pos.y_um, pos.z_um)
            except Exception as exc:
                rec.error = f"stage_move: {exc}"
                result.frames.append(rec)
                if on_frame:
                    on_frame(rec)
                continue

        # Shutter open (per-frame branch).
        if shutter is not None and plan.shutter_per_frame:
            try:
                shutter.open()
            except Exception:
                _LOG.debug("shutter per-frame open failed",
                           exc_info=True)

        # Settle.
        if plan.settle_time_s > 0:
            sleep_fn(plan.settle_time_s)

        # Grab.
        try:
            rec.frame = camera.grab()
        except Exception as exc:
            rec.error = f"grab: {exc}"

        # Shutter close (per-frame branch).
        if shutter is not None and plan.shutter_per_frame:
            try:
                shutter.close()
            except Exception:
                _LOG.debug("shutter per-frame close failed",
                           exc_info=True)

        result.frames.append(rec)
        if on_frame:
            try:
                on_frame(rec)
            except Exception:
                _LOG.debug("on_frame raised", exc_info=True)

    # Cleanup.
    if shutter is not None:
        try:
            shutter.close()
        except Exception:
            _LOG.debug("shutter close failed", exc_info=True)
    if led is not None:
        try:
            led.off()
        except Exception:
            _LOG.debug("LED off failed", exc_info=True)
    result.elapsed_s = _time.monotonic() - t0
    return result


__all__ = [
    "StagePosition",
    "AcquisitionPlan",
    "FrameRecord",
    "PlanResult",
    "run_plan",
]
