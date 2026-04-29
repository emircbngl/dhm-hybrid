"""Mock XYZ stage — deterministic, headless-safe.

Tracks position in micrometres, applies optional travel-time
simulation so tests of multi-position acquisition exercise the
"wait for stage to settle" path. Position respects axis limits
(soft min/max) so a test asserting "move outside limits raises"
has a target.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Tuple

from . import DeviceBackendInfo, DeviceKind


_LOG = logging.getLogger(__name__)


BACKEND = DeviceBackendInfo(
    name="mock_stage",
    display_name="Mock XYZ stage",
    kind=DeviceKind.STAGE,
    summary="Deterministic XYZ positioner with soft limits and "
            "configurable settle time. CI / dev-laptop default.",
    requires_sdk=(),
    capabilities={
        "xy": True, "z": True, "home": True,
        "limits_um": (-25_000.0, 25_000.0),
        "speed_um_per_s": 1000.0,
    },
)


def is_available() -> bool:
    return True


@dataclass
class MockStage:
    """``StageDevice`` Protocol implementation.

    Attributes
    ----------
    name
        Operator-visible label.
    limits_um
        ``(min, max)`` for every axis. ``move_to`` raises
        ``ValueError`` for any out-of-range request.
    settle_time_s
        Synthetic delay applied inside ``move_to`` (sleep). 0 by
        default so tests don't slow down; bump for realism.
    """
    name: str = "mock-stage-1"
    limits_um: Tuple[float, float] = (-25_000.0, 25_000.0)
    settle_time_s: float = 0.0

    _connected: bool = field(default=False, init=False)
    _x: float = field(default=0.0, init=False)
    _y: float = field(default=0.0, init=False)
    _z: float = field(default=0.0, init=False)

    def connect(self) -> None:
        self._connected = True
        _LOG.info("mock_stage[%s]: connected", self.name)

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def home(self) -> None:
        if not self._connected:
            raise RuntimeError("home() before connect()")
        self._x = self._y = self._z = 0.0

    def move_to(self, x_um: float, y_um: float,
                z_um: float = 0.0) -> None:
        if not self._connected:
            raise RuntimeError("move_to() before connect()")
        for label, v in (("x", x_um), ("y", y_um), ("z", z_um)):
            lo, hi = self.limits_um
            if not lo <= v <= hi:
                raise ValueError(
                    f"{self.name}: {label}={v} µm outside "
                    f"[{lo}, {hi}]",
                )
        if self.settle_time_s > 0:
            time.sleep(self.settle_time_s)
        self._x, self._y, self._z = float(x_um), float(y_um), float(z_um)

    @property
    def position_um(self) -> Tuple[float, float, float]:
        return (self._x, self._y, self._z)


def make(*,
         name: str = "mock-stage-1",
         limits_um: Tuple[float, float] = (-25_000.0, 25_000.0),
         settle_time_s: float = 0.0,
         ) -> MockStage:
    return MockStage(
        name=name,
        limits_um=limits_um,
        settle_time_s=settle_time_s,
    )
