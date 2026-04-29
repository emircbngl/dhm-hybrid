"""Mock shutter — open/close, optional latency simulation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import DeviceBackendInfo, DeviceKind


BACKEND = DeviceBackendInfo(
    name="mock_shutter",
    display_name="Mock shutter",
    kind=DeviceKind.SHUTTER,
    summary="Binary state shutter with configurable open/close "
            "latency. Use in CI to test shutter-sync timing.",
    requires_sdk=(),
    capabilities={"latency_ms": 5.0},
)


def is_available() -> bool:
    return True


@dataclass
class MockShutter:
    name: str = "mock-shutter-1"
    latency_s: float = 0.0
    _connected: bool = field(default=False, init=False)
    _open: bool = field(default=False, init=False)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def open(self) -> None:
        if not self._connected:
            raise RuntimeError("open() before connect()")
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        self._open = True

    def close(self) -> None:
        if not self._connected:
            raise RuntimeError("close() before connect()")
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open


def make(*,
         name: str = "mock-shutter-1",
         latency_s: float = 0.0,
         ) -> MockShutter:
    return MockShutter(name=name, latency_s=latency_s)
