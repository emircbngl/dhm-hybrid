"""Error bus — event structure, pub/sub, history, severity filtering."""
from __future__ import annotations

import threading
import time

import pytest

from core.errors import (
    ErrorCenter,
    ErrorEvent,
    Severity,
    get_error_center,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_for_tests()
    yield
    reset_for_tests()


# ---- ErrorEvent --------------------------------------------------------------


def test_event_make_fills_timestamp_and_id():
    ev = ErrorEvent.make(
        severity=Severity.ERROR,
        title="Recon failed",
        cause="Wavelength is 0 nm.",
        action="Set a positive value.",
    )
    assert ev.timestamp and "T" in ev.timestamp
    assert ev.event_id and len(ev.event_id) == 12
    assert ev.trace is None


def test_event_from_exception_captures_trace():
    try:
        raise ValueError("bad param")
    except ValueError as e:
        ev = ErrorEvent.from_exception(e, title="Test", action="Retry")

    assert ev.cause == "bad param"
    assert ev.trace is not None
    assert "ValueError" in ev.trace
    assert "bad param" in ev.trace


def test_event_as_dict_omits_trace_keeps_has_trace_flag():
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        ev = ErrorEvent.from_exception(e, title="T")

    d = ev.as_dict()
    assert d["has_trace"] is True
    assert "trace" not in d  # intentional: traces are only for the drawer


def test_event_is_frozen():
    ev = ErrorEvent.make(
        severity=Severity.INFO, title="T", cause="c",
    )
    with pytest.raises(Exception):
        ev.severity = Severity.FATAL  # type: ignore[misc]


# ---- ErrorCenter basic pub/sub ----------------------------------------------


def test_subscriber_receives_emitted_event():
    center = ErrorCenter()
    received: list[ErrorEvent] = []
    center.subscribe(received.append)

    center.error("Failed", "reason", action="fix it")

    assert len(received) == 1
    assert received[0].title == "Failed"
    assert received[0].severity == Severity.ERROR


def test_unsubscribe_stops_delivery():
    center = ErrorCenter()
    received: list[ErrorEvent] = []
    unsub = center.subscribe(received.append)

    center.info("First")
    unsub()
    center.info("Second")

    assert len(received) == 1
    assert received[0].title == "First"


def test_sink_exception_does_not_block_other_sinks():
    center = ErrorCenter()
    received: list[ErrorEvent] = []

    def broken(ev):
        raise RuntimeError("bad sink")

    center.subscribe(broken)
    center.subscribe(received.append)

    center.error("still delivered", "yes")
    assert len(received) == 1


# ---- History -----------------------------------------------------------------


def test_recent_returns_newest_first():
    center = ErrorCenter()
    center.info("first")
    center.info("second")
    center.info("third")

    recent = center.recent()
    titles = [e.title for e in recent]
    assert titles == ["third", "second", "first"]


def test_recent_filters_by_min_severity():
    center = ErrorCenter()
    center.info("info msg")
    center.warn("warn msg", "why")
    center.error("error msg", "cause")

    errors_only = center.recent(min_severity=Severity.ERROR)
    assert [e.title for e in errors_only] == ["error msg"]

    warn_and_up = center.recent(min_severity=Severity.WARN)
    assert [e.title for e in warn_and_up] == ["error msg", "warn msg"]


def test_history_is_bounded():
    center = ErrorCenter(history_size=5)
    for i in range(20):
        center.info(f"msg-{i}")
    recent = center.recent(limit=100)
    assert len(recent) == 5
    assert recent[0].title == "msg-19"
    assert recent[-1].title == "msg-15"


def test_clear_history_empties_buffer():
    center = ErrorCenter()
    center.error("a", "b")
    center.error("c", "d")
    assert len(center.recent()) == 2
    center.clear_history()
    assert center.recent() == []


# ---- Thread safety -----------------------------------------------------------


def test_emit_from_multiple_threads_is_safe():
    center = ErrorCenter(history_size=1000)
    received: list[ErrorEvent] = []
    lock = threading.Lock()

    def safe_sink(ev):
        with lock:
            received.append(ev)

    center.subscribe(safe_sink)

    def worker(worker_id: int):
        for i in range(50):
            center.info(f"w{worker_id}-{i}")

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(received) == 200
    # history should also hold them all
    assert len(center.recent(limit=500)) == 200


# ---- Singleton ---------------------------------------------------------------


def test_get_error_center_is_singleton():
    a = get_error_center()
    b = get_error_center()
    assert a is b


def test_singleton_has_default_sinks_wired():
    center = get_error_center()
    # Emit error; history holds it → default sinks registered without error
    center.error("T", "c")
    assert len(center.recent()) == 1
