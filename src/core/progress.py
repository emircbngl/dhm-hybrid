"""Progress protocol for long-running operations.

Pure-Python dataclasses and a context-manager helper. Qt signals live in the
worker layer; core stays Qt-free so operations can be tested headless and
replayed from an audit log without spinning up a GUI.

Usage inside a worker:

    from core.progress import Operation, ProgressEvent

    def _process(self, job):
        op = Operation("recon", phases=["preprocess", "offaxis", "propagate",
                                        "refsub", "unwrap"])
        op.on_progress = lambda ev: self.progress.emit(ev)   # Qt wiring

        with op.phase("preprocess"):
            if self.isInterruptionRequested():
                raise OperationCancelled()
            ...

The important properties:
    * No Qt import — tests don't need PySide6.
    * Phases are declared up front so pct is monotonic and honest.
    * ETA is derived from elapsed time and remaining-phase count, not faked.
    * Cancellation is signalled by raising OperationCancelled; the context
      manager always emits a final event so subscribers see a clean close.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator, Optional, Sequence


class OperationCancelled(Exception):
    """Raised when a cooperative cancellation is observed inside a phase."""


@dataclass(frozen=True)
class ProgressEvent:
    """Single point-in-time snapshot of an operation.

    Subscribers (UI, audit log, logger) consume these; they are immutable so
    they can be queued across threads without defensive copies.
    """
    operation_id: str
    operation_kind: str          # e.g. "recon", "qpi", "autofocus"
    phase: str                   # phase name, or "" for init/done
    phase_index: int             # 0-based; -1 before start, == total at done
    phase_count: int             # total phases
    pct: float                   # 0.0 .. 1.0, monotonic within an operation
    elapsed_s: float             # wall time since op.start()
    eta_s: Optional[float]       # estimated seconds remaining, or None if unknown
    message: str = ""            # short, human, optional
    done: bool = False           # True only on the final event
    cancelled: bool = False      # True if op ended via OperationCancelled

    def as_dict(self) -> dict:
        """Audit-log friendly dict (flat, primitive types only)."""
        return {
            "operation_id": self.operation_id,
            "kind": self.operation_kind,
            "phase": self.phase,
            "phase_index": self.phase_index,
            "phase_count": self.phase_count,
            "pct": round(self.pct, 4),
            "elapsed_s": round(self.elapsed_s, 3),
            "eta_s": None if self.eta_s is None else round(self.eta_s, 3),
            "message": self.message,
            "done": self.done,
            "cancelled": self.cancelled,
        }


ProgressSink = Callable[[ProgressEvent], None]


@dataclass
class Operation:
    """Context for a multi-phase operation.

    Declare phases up front so percentage is meaningful. Call `.phase(name)`
    as a context manager around each segment of work. Subscribers get a start
    event, per-phase-start events, and a final done/cancelled event.
    """
    kind: str
    phases: Sequence[str]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    on_progress: Optional[ProgressSink] = None

    _t0: float = field(default=0.0, init=False, repr=False)
    _current_index: int = field(default=-1, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    # ---- lifecycle ---------------------------------------------------------

    def start(self, message: str = "") -> None:
        """Emit the initial event. Safe to call implicitly on first phase()."""
        if self._started:
            return
        self._t0 = time.perf_counter()
        self._started = True
        self._emit(ProgressEvent(
            operation_id=self.id,
            operation_kind=self.kind,
            phase="",
            phase_index=-1,
            phase_count=len(self.phases),
            pct=0.0,
            elapsed_s=0.0,
            eta_s=None,
            message=message,
        ))

    def finish(self, *, cancelled: bool = False, message: str = "") -> None:
        """Emit the final event. Idempotent."""
        if self._finished:
            return
        self._finished = True
        n = len(self.phases)
        self._emit(ProgressEvent(
            operation_id=self.id,
            operation_kind=self.kind,
            phase="",
            phase_index=n,
            phase_count=n,
            pct=1.0,
            elapsed_s=self._elapsed(),
            eta_s=0.0,
            message=message,
            done=True,
            cancelled=cancelled,
        ))

    @contextmanager
    def phase(self, name: str, *, message: str = "") -> Iterator["PhaseHandle"]:
        """Enter a phase. Emits a start event; auto-finishes on exit.

        If the body raises OperationCancelled the op is marked cancelled and
        the exception propagates so the caller can short-circuit cleanly.
        """
        if not self._started:
            self.start()
        try:
            idx = list(self.phases).index(name)
        except ValueError as e:
            raise ValueError(
                f"phase {name!r} not declared in Operation(kind={self.kind!r}, "
                f"phases={list(self.phases)!r})"
            ) from e

        self._current_index = idx
        pct = idx / max(len(self.phases), 1)
        elapsed = self._elapsed()
        eta = self._estimate_eta(pct, elapsed)
        self._emit(ProgressEvent(
            operation_id=self.id,
            operation_kind=self.kind,
            phase=name,
            phase_index=idx,
            phase_count=len(self.phases),
            pct=pct,
            elapsed_s=elapsed,
            eta_s=eta,
            message=message or name,
        ))

        handle = PhaseHandle(self, name, idx)
        try:
            yield handle
        except OperationCancelled:
            self.finish(cancelled=True, message=f"cancelled during {name}")
            raise

    def update(self, *, sub_pct: float, message: str = "") -> None:
        """Emit a within-phase progress update. `sub_pct` in [0, 1].

        Useful for long phases (e.g. an FFT with a reportable iteration count).
        The overall pct interpolates between this phase and the next.
        """
        if self._current_index < 0:
            return
        n = len(self.phases)
        sub_pct = max(0.0, min(1.0, sub_pct))
        pct = (self._current_index + sub_pct) / max(n, 1)
        elapsed = self._elapsed()
        self._emit(ProgressEvent(
            operation_id=self.id,
            operation_kind=self.kind,
            phase=self.phases[self._current_index],
            phase_index=self._current_index,
            phase_count=n,
            pct=pct,
            elapsed_s=elapsed,
            eta_s=self._estimate_eta(pct, elapsed),
            message=message,
        ))

    # ---- internals ---------------------------------------------------------

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0 if self._started else 0.0

    def _estimate_eta(self, pct: float, elapsed: float) -> Optional[float]:
        if pct <= 1e-3 or elapsed < 0.15:
            return None
        return max(0.0, elapsed * (1.0 - pct) / pct)

    def _emit(self, ev: ProgressEvent) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(ev)
            except Exception:  # noqa: BLE001 — sink must not break the op
                pass


@dataclass
class PhaseHandle:
    """Handle yielded by Operation.phase() for within-phase updates."""
    op: "Operation"
    name: str
    index: int

    def update(self, sub_pct: float, message: str = "") -> None:
        self.op.update(sub_pct=sub_pct, message=message)


# ---------------------------------------------------------------------------
# Convenience: snapshot an event into something friendly for an audit record.
# Used by core.audit when recording operation_started / operation_completed.
# ---------------------------------------------------------------------------

def audit_payload(ev: ProgressEvent) -> dict:
    """Return a compact dict suitable for audit-log params."""
    return ev.as_dict()
