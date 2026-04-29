"""Mock XYZ stage — deterministic, headless-safe, APT-style features.

Tracks position in micrometres and adds the operator-facing knobs
real lab stage controllers expose (Thorlabs APT, Newport, Märzhäuser,
PI):

* **Speed control** — ``set_speed_um_per_s`` + ``speed_um_per_s``.
  Synthetic sleep on a long move is ``distance / speed`` so tests
  of "stage settle before grab" exercise the right wait budget.
* **Step-mode jogging** — operator picks a step size (1, 10, 100,
  1000 µm presets are typical) and clicks ``X+ / X- / Y+ / Y-
  / Z+ / Z-`` arrow buttons. Encoded as ``jog(axis, direction)``;
  agent tools route operator intent into the same primitive.
* **Soft limits** — every move clamps + raises on out-of-range.
* **Stop motion** — ``stop_motion()`` cancels any in-flight
  synthetic settle (real stages need an emergency-stop).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Tuple

from . import DeviceBackendInfo, DeviceKind


_LOG = logging.getLogger(__name__)


# Default speed for the mock — fast enough that tests don't sleep
# meaningfully but deterministic when the caller asks for it.
_DEFAULT_SPEED_UM_PER_S = 1000.0
# Default step size — 100 µm is the lab's common single-cell-shift
# magnitude; operator can change.
_DEFAULT_STEP_UM = 100.0
# Allowed jog axis labels — kept here so AI tools + UI buttons
# share one canonical list.
_JOG_AXES = ("x", "y", "z")


BACKEND = DeviceBackendInfo(
    name="mock_stage",
    display_name="Mock XYZ stage (APT-style)",
    kind=DeviceKind.STAGE,
    summary="Deterministic XYZ positioner with soft limits, "
            "speed control, and step-size jog (X+/X-/Y+/Y-/Z+/Z-). "
            "CI / dev-laptop default.",
    requires_sdk=(),
    capabilities={
        "xy": True, "z": True, "home": True,
        "limits_um": (-25_000.0, 25_000.0),
        "speed_um_per_s_default": _DEFAULT_SPEED_UM_PER_S,
        "step_um_default": _DEFAULT_STEP_UM,
        "jog": True,
        "stop_motion": True,
    },
)


def is_available() -> bool:
    return True


@dataclass
class MockStage:
    """``StageDevice`` Protocol implementation, APT-style features.

    Attributes
    ----------
    name
        Operator-visible label.
    limits_um
        ``(min, max)`` for every axis.
    settle_time_s
        Optional fixed-overhead settle (move-end mechanical
        damping). Zero by default so tests stay fast.
    simulate_motion_time
        When True, ``move_to`` / ``move_by`` sleep for
        ``distance_um / speed_um_per_s`` so the "wait for the
        stage to finish" code path is exercised. Default False.
    """
    name: str = "mock-stage-1"
    limits_um: Tuple[float, float] = (-25_000.0, 25_000.0)
    settle_time_s: float = 0.0
    simulate_motion_time: bool = False

    _connected: bool = field(default=False, init=False)
    _x: float = field(default=0.0, init=False)
    _y: float = field(default=0.0, init=False)
    _z: float = field(default=0.0, init=False)
    _speed: float = field(default=_DEFAULT_SPEED_UM_PER_S, init=False)
    _step: float = field(default=_DEFAULT_STEP_UM, init=False)
    _abort: threading.Event = field(default_factory=threading.Event,
                                    init=False)

    # ── lifecycle ─────────────────────────────────────────────
    def connect(self) -> None:
        self._connected = True
        _LOG.info("mock_stage[%s]: connected", self.name)

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── absolute + home ───────────────────────────────────────
    def home(self) -> None:
        if not self._connected:
            raise RuntimeError("home() before connect()")
        self.move_to(0.0, 0.0, 0.0)

    def move_to(self, x_um: float, y_um: float,
                z_um: float = 0.0) -> None:
        if not self._connected:
            raise RuntimeError("move_to() before connect()")
        self._abort.clear()
        for label, v in (("x", x_um), ("y", y_um), ("z", z_um)):
            lo, hi = self.limits_um
            if not lo <= v <= hi:
                raise ValueError(
                    f"{self.name}: {label}={v} µm outside "
                    f"[{lo}, {hi}]",
                )
        # Synthetic motion time = distance / speed. Skip in fast
        # CI mode (default).
        if self.simulate_motion_time and self._speed > 0:
            dx = float(x_um) - self._x
            dy = float(y_um) - self._y
            dz = float(z_um) - self._z
            distance = (dx * dx + dy * dy + dz * dz) ** 0.5
            travel_s = distance / self._speed
            self._sleep_abortable(travel_s)
        if self.settle_time_s > 0:
            self._sleep_abortable(self.settle_time_s)
        if self._abort.is_set():
            # Aborted mid-motion — leave position at request
            # target (mock approximation; a real controller would
            # report a partial trajectory).
            _LOG.info("mock_stage[%s]: motion aborted", self.name)
        self._x, self._y, self._z = float(x_um), float(y_um), float(z_um)

    @property
    def position_um(self) -> Tuple[float, float, float]:
        return (self._x, self._y, self._z)

    # ── speed ─────────────────────────────────────────────────
    def set_speed_um_per_s(self, v) -> None:
        """Set traverse speed. Pass ``None`` to reset to default.

        Real APT controllers cap at the actuator's max velocity —
        we accept any positive number. Negative / zero raise so
        an LLM hallucination ``speed=-5`` doesn't silently freeze
        the stage."""
        if v is None:
            self._speed = _DEFAULT_SPEED_UM_PER_S
            return
        v = float(v)
        if v <= 0:
            raise ValueError(
                f"speed must be > 0 µm/s (got {v})",
            )
        self._speed = v

    @property
    def speed_um_per_s(self) -> float:
        return self._speed

    # ── relative + jog ────────────────────────────────────────
    def move_by(self, dx_um: float, dy_um: float,
                dz_um: float = 0.0) -> None:
        """Shift relative to current position. Wraps ``move_to``
        so soft-limit + speed semantics carry through unchanged."""
        if not self._connected:
            raise RuntimeError("move_by() before connect()")
        self.move_to(
            self._x + float(dx_um),
            self._y + float(dy_um),
            self._z + float(dz_um),
        )

    def set_step_size_um(self, step_um: float) -> None:
        """Set the discrete jog increment. Zero / negative are
        rejected — operator wouldn't get visible motion."""
        v = float(step_um)
        if v <= 0:
            raise ValueError(
                f"step_size_um must be > 0 (got {v})",
            )
        self._step = v

    @property
    def step_size_um(self) -> float:
        return self._step

    def jog(self, axis: str, direction: int) -> None:
        """One discrete step on ``axis`` (``"x"`` / ``"y"`` /
        ``"z"``) in ``direction`` (+1 or −1). Distance = current
        ``step_size_um``. Caller-friendly: pass ``+2`` to take
        two consecutive steps."""
        if not self._connected:
            raise RuntimeError("jog() before connect()")
        ax = str(axis).lower()
        if ax not in _JOG_AXES:
            raise ValueError(
                f"jog axis must be one of {_JOG_AXES}, got {axis!r}",
            )
        dir_int = int(direction)
        if dir_int == 0:
            return
        delta = float(dir_int) * self._step
        if ax == "x":
            self.move_by(delta, 0.0, 0.0)
        elif ax == "y":
            self.move_by(0.0, delta, 0.0)
        else:
            self.move_by(0.0, 0.0, delta)

    def stop_motion(self) -> None:
        """Request the in-flight synthetic motion to bail out.
        Real controllers cut current to the actuators."""
        self._abort.set()

    # ── helpers ───────────────────────────────────────────────
    def _sleep_abortable(self, total_s: float) -> None:
        """``time.sleep`` in 50 ms slices so ``stop_motion``
        wakes us within ~50 ms instead of waiting full motion."""
        end = time.monotonic() + max(0.0, total_s)
        while time.monotonic() < end:
            if self._abort.is_set():
                return
            time.sleep(min(0.05, end - time.monotonic()))


def make(*,
         name: str = "mock-stage-1",
         limits_um: Tuple[float, float] = (-25_000.0, 25_000.0),
         settle_time_s: float = 0.0,
         simulate_motion_time: bool = False,
         ) -> MockStage:
    return MockStage(
        name=name,
        limits_um=limits_um,
        settle_time_s=settle_time_s,
        simulate_motion_time=simulate_motion_time,
    )
