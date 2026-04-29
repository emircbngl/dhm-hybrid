"""Device Protocols — capability-typed interfaces every backend
implements a subset of.

Why granular Protocols
----------------------
Stages, shutters, and LEDs share a tiny common API (``connect``,
``disconnect``, ``is_connected``) but diverge sharply on what
actions make sense (``move_to`` vs ``open/close`` vs
``set_intensity``). One mega-Protocol with everything optional
makes consumer code do constant ``hasattr`` dances. A small base
plus per-kind extensions (``StageDevice``, ``ShutterDevice``,
``LEDDevice``) keeps consumers honest — pass typing's structural
checks at the right granularity.
"""
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable


@runtime_checkable
class Device(Protocol):
    """Common base — every backend honours these."""
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def name(self) -> str: ...


@runtime_checkable
class StageDevice(Device, Protocol):
    """Motorised XYZ stage."""
    def move_to(self, x_um: float, y_um: float,
                z_um: float = 0.0) -> None: ...
    def home(self) -> None: ...
    @property
    def position_um(self) -> Tuple[float, float, float]: ...


@runtime_checkable
class ShutterDevice(Device, Protocol):
    """Binary state shutter."""
    def open(self) -> None: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...


@runtime_checkable
class LEDDevice(Device, Protocol):
    """LED illumination controller — on/off + 0-100 % intensity."""
    def on(self) -> None: ...
    def off(self) -> None: ...
    def set_intensity(self, percent: float) -> None: ...
    @property
    def intensity_percent(self) -> float: ...
    @property
    def is_on(self) -> bool: ...


__all__ = [
    "Device",
    "StageDevice",
    "ShutterDevice",
    "LEDDevice",
]
