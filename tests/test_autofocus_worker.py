"""Autofocus worker — structured error protocol tests.

The worker's algorithm layer already has its own extensive tests
(``tests/test_autofocus_*.py``). What this file verifies is the thin
v1.0.1 contract: when the worker fails it must fan out both the legacy
``error(str)`` signal *and* a new structured ``error_event(ErrorEvent)``
signal, so the UI can bind to the same protocol used by the
reconstruction and QPI workers.

We deliberately exercise :meth:`AutofocusWorker._emit_error` directly
rather than driving the full ``run()`` pipeline — the pipeline's
behavior is covered elsewhere, and we want a fast, deterministic test
that doesn't spin up Qt's event loop or any real focus scan.
"""
from __future__ import annotations

from core.autofocus import FocusMetric
from core.errors import ErrorEvent, Severity
from gui.workers import autofocus_worker as afw


def _job(**overrides):
    base = dict(
        z_min_m=0.01,
        z_max_m=0.06,
        steps=25,
        metric=FocusMetric.PHASE_VARIANCE,
        algo="Linear Sweep",
    )
    base.update(overrides)
    return base


def test_emit_error_fans_out_structured_and_legacy():
    worker = afw.AutofocusWorker()
    structured: list[ErrorEvent] = []
    legacy: list[str] = []
    worker.error_event.connect(structured.append)
    worker.error.connect(legacy.append)

    worker._emit_error(ValueError("synthetic metric failure"), job=_job())

    assert len(structured) == 1
    ev = structured[0]
    assert ev.severity == Severity.ERROR
    assert ev.title == "Autofocus failed"
    assert "synthetic metric failure" in ev.cause
    assert ev.context["z_min_mm"] == 10.0
    assert ev.context["z_max_mm"] == 60.0
    assert ev.context["steps"] == 25
    assert ev.context["metric"] == FocusMetric.PHASE_VARIANCE.value
    assert ev.context["algo"] == "Linear Sweep"
    assert ev.trace is not None  # from_exception captured traceback

    # Legacy string signal stays wired for v1.0.0 call-sites.
    assert len(legacy) == 1
    assert "synthetic metric failure" in legacy[0]


def test_emit_error_without_job_still_works():
    """The ``run()`` guard emits before a job is configured — no context,
    no crash."""
    worker = afw.AutofocusWorker()
    events: list[ErrorEvent] = []
    worker.error_event.connect(events.append)

    worker._emit_error(
        ValueError("No autofocus job configured"),
        job=None,
        title="Autofocus failed",
        action="Configure z-range and metric, then retry.",
    )

    assert len(events) == 1
    assert events[0].context == {}
    assert events[0].action == "Configure z-range and metric, then retry."


def test_suggest_action_zrange_inverted():
    """Rams #4 hint: a user-correctable action when z-max ≤ z-min."""
    hint = afw._suggest_action(
        ValueError("something went wrong"),
        _job(z_min_m=0.05, z_max_m=0.01),
    )
    assert "z-max" in hint.lower()


def test_suggest_action_memory_error():
    hint = afw._suggest_action(MemoryError("out of memory"), _job())
    assert "step" in hint.lower() or "z-range" in hint.lower()


def test_suggest_action_wavelength_keyword():
    hint = afw._suggest_action(
        ValueError("wavelength produced invalid k"),
        _job(),
    )
    assert "wavelength" in hint.lower()


def test_suggest_action_empty_when_no_rule_matches():
    """If no heuristic fires, return an empty string rather than guessing."""
    hint = afw._suggest_action(RuntimeError("unrelated thing"), _job())
    assert hint == ""
