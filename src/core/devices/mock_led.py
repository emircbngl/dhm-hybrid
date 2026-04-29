"""Mock LED illumination controller — on/off + 0–100 % intensity."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import DeviceBackendInfo, DeviceKind


BACKEND = DeviceBackendInfo(
    name="mock_led",
    display_name="Mock LED",
    kind=DeviceKind.LED,
    summary="0–100 % intensity LED model. Honour clamping at the "
            "boundary so accidental over-drive doesn't slip past.",
    requires_sdk=(),
    capabilities={
        "intensity_min": 0.0,
        "intensity_max": 100.0,
        "intensity_resolution": 0.1,
    },
)


def is_available() -> bool:
    return True


@dataclass
class MockLED:
    name: str = "mock-led-1"
    _connected: bool = field(default=False, init=False)
    _on: bool = field(default=False, init=False)
    _intensity: float = field(default=0.0, init=False)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def on(self) -> None:
        if not self._connected:
            raise RuntimeError("on() before connect()")
        self._on = True

    def off(self) -> None:
        if not self._connected:
            raise RuntimeError("off() before connect()")
        self._on = False

    def set_intensity(self, percent: float) -> None:
        if not self._connected:
            raise RuntimeError("set_intensity() before connect()")
        # Clamp instead of raise — operator dragging a slider will
        # constantly hit the bound, raising would spam the log.
        self._intensity = max(0.0, min(100.0, float(percent)))

    @property
    def intensity_percent(self) -> float:
        return self._intensity

    @property
    def is_on(self) -> bool:
        return self._on


def make(*, name: str = "mock-led-1") -> MockLED:
    return MockLED(name=name)
