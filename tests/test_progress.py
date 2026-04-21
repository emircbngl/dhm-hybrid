"""Progress protocol — event ordering, ETA, cancellation semantics."""
from __future__ import annotations

import time

import pytest

from core.progress import Operation, OperationCancelled, ProgressEvent


def _collect() -> tuple[list[ProgressEvent], callable]:
    events: list[ProgressEvent] = []
    return events, events.append


def test_operation_emits_start_and_done():
    events, sink = _collect()
    op = Operation(kind="test", phases=["a", "b"], on_progress=sink)
    with op.phase("a"):
        pass
    with op.phase("b"):
        pass
    op.finish()

    # Start + two phase events + done
    assert len(events) == 4
    assert events[0].phase == "" and events[0].phase_index == -1
    assert events[1].phase == "a" and events[1].phase_index == 0
    assert events[2].phase == "b" and events[2].phase_index == 1
    assert events[3].done is True and events[3].pct == 1.0


def test_pct_is_monotonic_within_operation():
    events, sink = _collect()
    op = Operation(kind="test", phases=["a", "b", "c", "d"], on_progress=sink)
    for p in ["a", "b", "c", "d"]:
        with op.phase(p):
            pass
    op.finish()

    pcts = [e.pct for e in events]
    assert pcts == sorted(pcts), f"pct not monotonic: {pcts}"
    assert pcts[0] == 0.0
    assert pcts[-1] == 1.0


def test_unknown_phase_raises():
    op = Operation(kind="test", phases=["a", "b"])
    with pytest.raises(ValueError, match="phase 'ghost' not declared"):
        with op.phase("ghost"):
            pass


def test_cancellation_emits_cancelled_event():
    events, sink = _collect()
    op = Operation(kind="test", phases=["a", "b"], on_progress=sink)
    with pytest.raises(OperationCancelled):
        with op.phase("a"):
            raise OperationCancelled()

    last = events[-1]
    assert last.done is True
    assert last.cancelled is True
    assert last.phase_index == len(op.phases)


def test_finish_is_idempotent():
    events, sink = _collect()
    op = Operation(kind="test", phases=["a"], on_progress=sink)
    with op.phase("a"):
        pass
    op.finish()
    op.finish()  # no-op
    op.finish()  # no-op
    # Exactly one done event
    done_events = [e for e in events if e.done]
    assert len(done_events) == 1


def test_within_phase_update_refines_pct():
    events, sink = _collect()
    op = Operation(kind="test", phases=["a", "b"], on_progress=sink)
    with op.phase("a") as h:
        h.update(0.5, "halfway")
    op.finish()

    # Find the update event
    updates = [e for e in events if e.message == "halfway"]
    assert len(updates) == 1
    # phase 'a' starts at pct 0.0 and phase 'b' at 0.5; halfway through 'a' should be 0.25
    assert abs(updates[0].pct - 0.25) < 1e-9


def test_eta_is_none_at_start_and_zero_at_end():
    events, sink = _collect()
    op = Operation(kind="test", phases=["a"], on_progress=sink)
    with op.phase("a"):
        time.sleep(0.01)
    op.finish()

    assert events[0].eta_s is None, "ETA at start should be unknown"
    assert events[-1].eta_s == 0.0, "ETA at done should be zero"


def test_event_as_dict_is_flat_primitives():
    op = Operation(kind="test", phases=["a"])
    events: list[ProgressEvent] = []
    op.on_progress = events.append
    with op.phase("a"):
        pass
    op.finish()

    for ev in events:
        d = ev.as_dict()
        for k, v in d.items():
            assert v is None or isinstance(v, (str, int, float, bool)), \
                f"{k} is not a primitive: {type(v)!r}"


def test_sink_exception_does_not_break_operation():
    def broken_sink(ev):
        raise RuntimeError("subscriber blew up")

    op = Operation(kind="test", phases=["a"], on_progress=broken_sink)
    # should not raise
    with op.phase("a"):
        pass
    op.finish()
