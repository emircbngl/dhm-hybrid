"""Microscope stage interface — abstract base + in-memory mock.

The DHM app does not currently drive any motion hardware. This module
exists so the AI assistant has a stable surface to call into
(``stage_get_position``, ``stage_move_relative``, …) today, and so a
real driver — Leica K8, Thorlabs KDC101, Märzhäuser Tango, etc. — can
slot in later as a sibling implementation without churning the AI
tool layer.

``MockStage`` is the v1 default. It tracks an XYZ tuple in memory,
clamps every move to a sane bounding box, and notifies registered
listeners on each move. Threads can call concurrently — a single
``threading.Lock`` is enough for the kind of contention an LLM-driven
agent produces (one move at a time).

No Qt; the panel adapter wraps a listener as a ``QTimer.singleShot``
or queued signal as needed.
"""
from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

Position = Tuple[float, float, float]
Listener = Callable[[Position], None]


class StageError(RuntimeError):
    """Raised on motion-control failures (out of bounds, NaN, hardware fault)."""


class StageInterface(ABC):
    """Common surface for any XYZ stage. All units are millimetres.

    Real drivers should refuse moves while the previous one is still
    in flight; for v1 with the mock, every move completes synchronously,
    so this concern is just documented.
    """

    @abstractmethod
    def get_position(self) -> Position: ...

    @abstractmethod
    def move_relative(self, dx_mm: float, dy_mm: float, dz_mm: float) -> Position: ...

    @abstractmethod
    def move_absolute(self, x_mm: float, y_mm: float, z_mm: float) -> Position: ...

    @abstractmethod
    def home(self) -> Position: ...

    @abstractmethod
    def add_listener(self, fn: Listener) -> Callable[[], None]:
        """Register a position-change listener. Returns the unsubscribe callable."""
        ...


class MockStage(StageInterface):
    """In-memory XYZ stage, suitable for tests and GUI smoke runs.

    Bounds default to ±200 mm — wider than any commercial DHM stage
    we've seen, so it doesn't artificially constrain experimentation.
    Real drivers will publish their own bounds.
    """

    BOUNDS_MM: tuple[float, float] = (-200.0, 200.0)

    def __init__(self, start: Position = (0.0, 0.0, 0.0)) -> None:
        self._lock = threading.Lock()
        self._listeners: List[Listener] = []
        self._x, self._y, self._z = self._validate_position(start)

    # ----- public API ------------------------------------------------------

    def get_position(self) -> Position:
        with self._lock:
            return (self._x, self._y, self._z)

    def move_relative(self, dx_mm: float, dy_mm: float, dz_mm: float) -> Position:
        for name, val in (("dx_mm", dx_mm), ("dy_mm", dy_mm), ("dz_mm", dz_mm)):
            if val != val:
                raise StageError(f"{name} is NaN")
        with self._lock:
            new = (
                self._clamp_axis(self._x + float(dx_mm)),
                self._clamp_axis(self._y + float(dy_mm)),
                self._clamp_axis(self._z + float(dz_mm)),
            )
            self._x, self._y, self._z = new
        self._notify(new)
        return new

    def move_absolute(self, x_mm: float, y_mm: float, z_mm: float) -> Position:
        new = self._validate_position((x_mm, y_mm, z_mm))
        with self._lock:
            self._x, self._y, self._z = new
        self._notify(new)
        return new

    def home(self) -> Position:
        with self._lock:
            self._x = self._y = self._z = 0.0
            new = (self._x, self._y, self._z)
        self._notify(new)
        return new

    def add_listener(self, fn: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(fn)
                except ValueError:
                    pass

        return unsubscribe

    # ----- internals -------------------------------------------------------

    def _validate_position(self, pos: Position) -> Position:
        if len(pos) != 3:
            raise StageError(f"position must be 3 values, got {pos!r}")
        out = []
        for name, val in zip(("x_mm", "y_mm", "z_mm"), pos):
            if math.isnan(val):
                raise StageError(f"{name} is NaN")
            lo, hi = self.BOUNDS_MM
            if val < lo or val > hi:
                raise StageError(f"{name}={val} is out of range [{lo}, {hi}]")
            out.append(float(val))
        return (out[0], out[1], out[2])

    def _clamp_axis(self, v: float) -> float:
        lo, hi = self.BOUNDS_MM
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def _notify(self, pos: Position) -> None:
        # Snapshot listeners under the lock, fire outside — listener
        # exceptions must not break the queue.
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(pos)
            except Exception:  # noqa: BLE001
                # Listener raised — swallow; same policy as ErrorCenter.
                pass


__all__ = [
    "Position",
    "Listener",
    "StageError",
    "StageInterface",
    "MockStage",
]
