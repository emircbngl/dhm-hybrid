"""Reconstruction worker — phased pipeline, cancellation, structured errors.

These tests drive ``_process`` synchronously (without starting the QThread
event loop) so they're fast and deterministic. Heavy core calls
(``extract_complex_field_offaxis_debug``, ``propagate``,
``unwrap_phase_advanced``) are monkey-patched with cheap stand-ins whose
shapes match the real signatures — the worker's responsibility is
orchestration, not science, and those core routines have their own
dedicated tests.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.errors import ErrorEvent, Severity
from core.progress import ProgressEvent
from gui.workers import reconstruction_worker as rw


# --- lightweight synthetic pipeline ------------------------------------------

def _fake_offaxis(img, params):
    """Return (complex_field, center, spectrum_mag, mask) — shape-preserving."""
    fc = (img + 1j * img).astype(np.complex64)
    spec = np.abs(fc)
    return fc, (0, 0), spec, np.ones_like(img)


def _fake_propagate(fc, params, method, fft=None):
    return np.asarray(fc, dtype=np.complex64)


def _fake_unwrap(wrapped, **kwargs):
    return wrapped.astype(np.float64)


@pytest.fixture
def _patched(monkeypatch):
    monkeypatch.setattr(rw, "extract_complex_field_offaxis_debug", _fake_offaxis)
    monkeypatch.setattr(rw, "propagate", _fake_propagate)
    monkeypatch.setattr(rw, "unwrap_phase_advanced", _fake_unwrap)


def _ready_worker() -> rw.ReconstructionWorker:
    """Build a worker with ``running=True`` so ``_check_cancel`` does not
    trip before the test has a chance to exercise the pipeline. In
    production ``run()`` sets this flag; the synchronous test path bypasses
    ``run()`` and must set it manually."""
    w = rw.ReconstructionWorker()
    w.running = True
    return w


def _job(image=None, **overrides):
    base = {
        "frame_num": 1,
        "image": image if image is not None else
                 np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32),
        "method": "angular_spectrum",
        "wavelength": 632.8e-9,
        "pixel_size": 3.45e-6,
        "z": 0.044,
        "n_medium": 1.0,
        "subtract_mean": True,
        "hann_window": False,
        "mask_radius": 16,
    }
    base.update(overrides)
    return base


# --- tests -------------------------------------------------------------------


def test_process_returns_expected_shape(_patched):
    worker = _ready_worker()
    result = worker._process(_job())
    assert set(result) >= {
        "frame_num", "recon_complex", "recon_complex_raw",
        "spectrum_mag", "phase_unwrapped",
    }
    assert result["recon_complex"].dtype == np.complex64
    assert np.all(np.isfinite(result["phase_unwrapped"]))


def test_progress_events_cover_every_phase_in_order(_patched):
    worker = _ready_worker()
    events: list[ProgressEvent] = []
    worker.progress.connect(events.append)

    worker._process(_job())

    # Start + 7 phases + done = 9 events
    phase_names = [e.phase for e in events if e.phase]
    assert phase_names == list(rw._PHASES), (
        f"expected {list(rw._PHASES)}, got {phase_names}"
    )

    # First event is start (phase="", phase_index=-1, pct=0); last is done (pct=1).
    assert events[0].phase == "" and events[0].phase_index == -1
    assert events[-1].done is True and events[-1].pct == 1.0
    assert events[-1].cancelled is False


def test_progress_pct_monotonic(_patched):
    worker = _ready_worker()
    events: list[ProgressEvent] = []
    worker.progress.connect(events.append)
    worker._process(_job())

    pcts = [e.pct for e in events]
    assert pcts == sorted(pcts), f"pct not monotonic: {pcts}"


def test_cancellation_before_propagate_raises_and_flags_event(_patched, monkeypatch):
    """Interruption observed at a phase boundary must short-circuit the pipeline."""
    worker = _ready_worker()
    events: list[ProgressEvent] = []
    worker.progress.connect(events.append)

    # Flip interruption on after offaxis completes; the next phase's
    # _check_cancel should trip.
    original_check = worker._check_cancel
    calls = {"n": 0}

    def cancel_after_two(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 2:
            # Force the QThread interruption check to return True without
            # actually starting a thread.
            monkeypatch.setattr(worker, "isInterruptionRequested", lambda: True)
        return original_check(*a, **kw)

    monkeypatch.setattr(worker, "_check_cancel", cancel_after_two)

    with pytest.raises(rw.OperationCancelled):
        worker._process(_job())

    # Final event should be the cancel-done marker.
    assert events[-1].done is True
    assert events[-1].cancelled is True
    # Phases that ran: preprocess, offaxis (at most)
    ran = [e.phase for e in events if e.phase and not e.done]
    assert "preprocess" in ran
    assert "unwrap" not in ran


def test_error_in_phase_emits_structured_event(monkeypatch):
    """A failure inside a phase surfaces as an ErrorEvent with phase context."""
    def explode(*a, **kw):
        raise ValueError("synthetic propagate failure")

    monkeypatch.setattr(rw, "extract_complex_field_offaxis_debug", _fake_offaxis)
    monkeypatch.setattr(rw, "propagate", explode)
    monkeypatch.setattr(rw, "unwrap_phase_advanced", _fake_unwrap)

    worker = _ready_worker()
    events: list[ErrorEvent] = []
    legacy: list[str] = []
    worker.error_event.connect(events.append)
    worker.error_occurred.connect(legacy.append)

    # _process propagates the raw exception — the outer run() loop catches
    # it. We simulate that catch manually:
    job = _job()
    try:
        worker._process(job)
    except ValueError as e:
        worker._emit_error(e, job)

    assert len(events) == 1
    ev = events[0]
    assert ev.severity == Severity.ERROR
    assert ev.title == "Reconstruction failed"
    assert "synthetic propagate failure" in ev.cause
    assert ev.context["z_mm"] == pytest.approx(44.0)
    assert ev.context["wavelength_nm"] == pytest.approx(632.8)
    assert ev.trace is not None

    # Legacy string-signal still fires for backward-compat.
    assert len(legacy) == 1
    assert "synthetic propagate failure" in legacy[0]


def test_wavelength_zero_suggests_positive_wavelength_action(_patched, monkeypatch):
    """Error copy should include a Rams-#4 action hint when the job is
    clearly misconfigured."""
    def wavelength_error(*a, **kw):
        raise ValueError("wavelength produced invalid k")

    monkeypatch.setattr(rw, "propagate", wavelength_error)
    worker = _ready_worker()
    events: list[ErrorEvent] = []
    worker.error_event.connect(events.append)

    job = _job(wavelength=0.0)
    try:
        worker._process(job)
    except ValueError as e:
        worker._emit_error(e, job)

    assert len(events) == 1
    assert "positive wavelength" in events[0].action.lower()


def test_stop_sets_running_false_and_requests_interruption():
    worker = _ready_worker()
    worker.running = True
    worker.stop()
    assert worker.running is False
    # isInterruptionRequested is only True after start(); stop() calls
    # requestInterruption but without a running event loop the flag is
    # stored — just confirm no exception path was taken.
