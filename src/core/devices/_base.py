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
    """Motorised XYZ stage.

    Core motion: absolute ``move_to`` + ``home`` + ``position_um``.
    APT-style controllers also need:

    * **Speed control** — ``set_speed_um_per_s(v)`` and the
      mirroring ``speed_um_per_s`` property. Movement uses this
      velocity until changed; reset to default with ``set_speed_um_per_s(None)``.
    * **Relative jog** — ``move_by(dx, dy, dz)`` shifts from the
      current position. Gives the AI agent a "step right by 50 µm"
      verb without having to read the position first.
    * **Step-mode** — operator-side discrete step size + jog
      verbs (``jog_x_plus`` etc.). The DPG control panel's
      arrow buttons translate to these; the agent's
      ``stage_jog_step`` tool calls into them.
    """
    def move_to(self, x_um: float, y_um: float,
                z_um: float = 0.0) -> None: ...
    def home(self) -> None: ...
    @property
    def position_um(self) -> Tuple[float, float, float]: ...

    # ── Speed control ─────────────────────────────────────────
    def set_speed_um_per_s(self, v: float) -> None: ...
    @property
    def speed_um_per_s(self) -> float: ...

    # ── Relative + step-by-step ──────────────────────────────
    def move_by(self, dx_um: float, dy_um: float,
                dz_um: float = 0.0) -> None: ...
    def set_step_size_um(self, step_um: float) -> None: ...
    @property
    def step_size_um(self) -> float: ...
    def jog(self, axis: str, direction: int) -> None: ...
    def stop_motion(self) -> None: ...


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
