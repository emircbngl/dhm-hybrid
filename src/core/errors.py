"""Structured error events and a process-wide error bus.

The v1.0.0 pattern — workers emitting ``str(exception)`` to a transient
status bar — loses severity, cause, action hint, traceback, and timing.
This module replaces that with a typed event and a central router.

Design:
    * ``ErrorEvent`` is frozen. It carries everything a UI needs to render
      a human message, plus everything an audit log needs to reconstruct
      what happened, plus the raw traceback for debugging.
    * ``ErrorCenter`` is a thread-safe singleton. Anywhere in the app can
      publish; anywhere else can subscribe.
    * Built-in subscribers: ``logger_sink`` (always on) and
      ``audit_sink`` (wired on first import). The UI layer adds its own
      toast/drawer subscribers in Tier 1.
    * A bounded in-memory history buffer (default 200) lets a "Recent
      Errors" drawer show what just happened without touching disk.

The distinction between *cause* and *action* is deliberate and matters
for copy quality:

    cause:  "Wavelength parameter is 0 nm"
    action: "Set a positive wavelength in the Parameters panel"

One sentence for each. The UI renders them on two lines. If a worker
can't provide an action, the cause is still shown alone — but the
convention is that workers SHOULD supply an action whenever the error
is user-correctable.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Optional

from core.audit import get_audit_log

_LOG = logging.getLogger(__name__)


class Severity(str, Enum):
    """Four-level severity ladder.

    * ``INFO``  — routine notice, no user action needed ("Autofocus fell
      back to PHASE_VARIANCE due to low contrast").
    * ``WARN``  — degraded result but computation completed ("Phase unwrap
      produced NaNs; using wrapped phase instead").
    * ``ERROR`` — operation failed; user correctable ("Wavelength is 0").
    * ``FATAL`` — app cannot continue ("Camera driver crashed").
    """
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


@dataclass(frozen=True)
class ErrorEvent:
    """One structured error, passed across the bus.

    Fields that appear verbatim in the UI copy:
        * ``title``  — one short noun phrase ("Reconstruction failed")
        * ``cause``  — one sentence ("Wavelength is 0 nm.")
        * ``action`` — one sentence of what to do ("Set a positive value
          in the Parameters panel."), or empty string if none.

    Fields for audit + debugging:
        * ``context`` — dict of anything relevant (job params, phase name,
          elapsed_s, operation_id). Must be JSON-serializable (primitive
          values only; the emitter is responsible for stringifying
          numpy scalars, Paths, etc.).
        * ``trace``   — the traceback string, or None if no exception.
    """
    severity: Severity
    title: str
    cause: str
    action: str = ""
    context: dict = field(default_factory=dict)
    trace: Optional[str] = None
    timestamp: str = ""      # ISO8601 UTC, filled by factory
    event_id: str = ""       # short uuid, filled by factory

    # ---- factories ---------------------------------------------------------

    @classmethod
    def make(
        cls,
        *,
        severity: Severity,
        title: str,
        cause: str,
        action: str = "",
        context: Optional[dict] = None,
        exc: Optional[BaseException] = None,
    ) -> "ErrorEvent":
        """Preferred constructor. Fills timestamp/event_id; flattens exc."""
        trace = None
        if exc is not None:
            trace = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        return cls(
            severity=severity,
            title=title,
            cause=cause,
            action=action,
            context=dict(context) if context else {},
            trace=trace,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=uuid.uuid4().hex[:12],
        )

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        title: str,
        action: str = "",
        context: Optional[dict] = None,
        severity: Severity = Severity.ERROR,
    ) -> "ErrorEvent":
        """Wrap a raw exception. ``cause`` is derived from ``str(exc)``."""
        cause = str(exc).strip() or type(exc).__name__
        return cls.make(
            severity=severity,
            title=title,
            cause=cause,
            action=action,
            context=context,
            exc=exc,
        )

    # ---- serialization -----------------------------------------------------

    def as_dict(self) -> dict:
        """JSON-friendly dict; used by audit sink and session history export."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "title": self.title,
            "cause": self.cause,
            "action": self.action,
            "context": dict(self.context),
            "has_trace": self.trace is not None,
        }


ErrorSink = Callable[[ErrorEvent], None]


class ErrorCenter:
    """Thread-safe publish/subscribe bus for ErrorEvents.

    ``emit`` is safe to call from a Qt worker thread; subscribers run
    inline on the emitter's thread. UI subscribers (toast widgets) must
    themselves marshal to the GUI thread — a Qt signal/slot connection
    does this automatically because Qt detects the cross-thread emission.
    """

    def __init__(self, history_size: int = 200) -> None:
        self._lock = threading.RLock()
        self._subscribers: list[ErrorSink] = []
        self._history: deque[ErrorEvent] = deque(maxlen=history_size)

    # ---- subscription ------------------------------------------------------

    def subscribe(self, sink: ErrorSink) -> Callable[[], None]:
        """Register a sink. Returns an unsubscribe callable."""
        with self._lock:
            self._subscribers.append(sink)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(sink)
                except ValueError:
                    pass

        return unsubscribe

    # ---- publication -------------------------------------------------------

    def emit(self, event: ErrorEvent) -> None:
        """Deliver to all subscribers. Subscriber exceptions are caught."""
        with self._lock:
            self._history.append(event)
            subs = list(self._subscribers)

        for sink in subs:
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001 — bus must not break
                _LOG.warning("error-center: sink %r raised: %s", sink, exc)

    # ---- convenience emitters ----------------------------------------------

    def error(self, title: str, cause: str, *, action: str = "",
              context: Optional[dict] = None,
              exc: Optional[BaseException] = None) -> None:
        self.emit(ErrorEvent.make(
            severity=Severity.ERROR, title=title, cause=cause,
            action=action, context=context, exc=exc,
        ))

    def warn(self, title: str, cause: str, *, action: str = "",
             context: Optional[dict] = None) -> None:
        self.emit(ErrorEvent.make(
            severity=Severity.WARN, title=title, cause=cause,
            action=action, context=context,
        ))

    def info(self, title: str, cause: str = "", *,
             context: Optional[dict] = None) -> None:
        self.emit(ErrorEvent.make(
            severity=Severity.INFO, title=title, cause=cause,
            context=context,
        ))

    # ---- history access ----------------------------------------------------

    def recent(self, limit: int = 50,
               min_severity: Severity = Severity.INFO) -> list[ErrorEvent]:
        """Most-recent-first list, filtered by severity floor."""
        rank = _severity_rank(min_severity)
        with self._lock:
            items = list(self._history)
        # newest first
        items.reverse()
        filtered = [e for e in items if _severity_rank(e.severity) >= rank]
        return filtered[:limit]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


def _severity_rank(s: Severity) -> int:
    return {Severity.INFO: 0, Severity.WARN: 1,
            Severity.ERROR: 2, Severity.FATAL: 3}[s]


# ---------------------------------------------------------------------------
# Built-in sinks — wired on first access of get_error_center()
# ---------------------------------------------------------------------------

def logger_sink(event: ErrorEvent) -> None:
    """Route to stdlib logging. Always registered."""
    level = {
        Severity.INFO: logging.INFO,
        Severity.WARN: logging.WARNING,
        Severity.ERROR: logging.ERROR,
        Severity.FATAL: logging.CRITICAL,
    }[event.severity]
    msg = f"[{event.severity.value}] {event.title}: {event.cause}"
    if event.action:
        msg += f"  ({event.action})"
    _LOG.log(level, msg)
    if event.trace:
        _LOG.debug("trace for %s:\n%s", event.event_id, event.trace)


def audit_sink(event: ErrorEvent) -> None:
    """Route to the JSONL audit log. Always registered.

    Only records ERROR and FATAL by default; WARN/INFO would flood the log
    during normal operation (fallbacks, unwrap warnings, etc.).
    """
    if _severity_rank(event.severity) < _severity_rank(Severity.ERROR):
        return
    try:
        get_audit_log().record(
            action="error_event",
            params=event.as_dict(),
        )
    except Exception as exc:  # pragma: no cover
        _LOG.warning("audit_sink failed: %s", exc)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_SINGLETON: Optional[ErrorCenter] = None
_SINGLETON_LOCK = threading.Lock()


def get_error_center() -> ErrorCenter:
    """Return the process-wide ErrorCenter, creating it on first call.

    The default logger + audit sinks are wired automatically.
    """
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            center = ErrorCenter()
            center.subscribe(logger_sink)
            center.subscribe(audit_sink)
            _SINGLETON = center
    return _SINGLETON


def reset_for_tests() -> None:
    """Clear the singleton. Test-only; not exported in __all__."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = None


__all__ = [
    "Severity",
    "ErrorEvent",
    "ErrorCenter",
    "ErrorSink",
    "get_error_center",
    "logger_sink",
    "audit_sink",
]
