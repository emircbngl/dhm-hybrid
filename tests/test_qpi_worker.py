"""QPI worker — phased pipeline, cancellation, structured errors.

The worker's responsibility is orchestration (sanitize input → optional
reference-wave subtraction → optional background correction → compute),
not the science of QPI itself — :mod:`core.qpi` has its own dedicated
tests. We therefore stub out ``compute_qpi`` / ``subtract_reference_wave``
/ ``correct_background_phase`` / ``unwrap_phase_advanced`` and assert the
worker's *contract*: the right phases run in the right order, progress
events fire, and failures surface as structured ``ErrorEvent`` payloads.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.errors import ErrorEvent, Severity
from core.progress import ProgressEvent
from gui.workers import qpi_worker as qw


# ---- test doubles -----------------------------------------------------------

class _StubResult:
    """Shape-only stand-in — the real QPIResult is a dataclass with many fields."""
    def __init__(self) -> None:
        self.opd_m = np.zeros((8, 8), dtype=np.float64)
        self.height_m = np.zeros((8, 8), dtype=np.float64)


def _fake_compute_qpi(**kwargs):
    # Assert the worker actually forwarded the expected kwargs — otherwise
    # silent key-renames would slip past these tests.
    assert "phase_unwrapped" in kwargs
    assert "wavelength_m" in kwargs
    assert "pixel_size_m" in kwargs
    assert "mode" in kwargs
    return _StubResult()


def _fake_subtract_reference_wave(recon, ref):
    return np.asarray(recon, dtype=np.complex64)


def _fake_correct_bg(phase, order=2):
    return np.asarray(phase, dtype=np.float64)


def _fake_unwrap(phase, **kwargs):
    return np.asarray(phase, dtype=np.float64)


@pytest.fixture
def _patched(monkeypatch):
    """Patch every heavy core entry point the worker lazy-imports."""
    import core.qpi as qpi_mod
    import core.phase_unwrap as pu_mod

    monkeypatch.setattr(qpi_mod, "compute_qpi", _fake_compute_qpi)
    monkeypatch.setattr(qpi_mod, "subtract_reference_wave", _fake_subtract_reference_wave)
    monkeypatch.setattr(qpi_mod, "correct_background_phase", _fake_correct_bg)
    monkeypatch.setattr(pu_mod, "unwrap_phase_advanced", _fake_unwrap)


def _job(**overrides):
    rng = np.random.default_rng(0)
    phase = rng.standard_normal((8, 8)).astype(np.float64)
    base = {
        "phase_unwrapped": phase,
        "wavelength_m": 632.8e-9,
        "pixel_size_m": 3.45e-6,
        "mode": "both",
        "n_sample": 1.38,
        "n_medium": 1.337,
        "alpha_SI": 1.8e-7,
        "known_thickness_m": None,
        "cell_threshold": 0.15,
        "compute_psd": True,
        # Optional toggles — default off so the base job skips refsub/bgcorrect.
        "ref_subtract": False,
        "bg_correction": False,
    }
    base.update(overrides)
    return base


# ---- tests ------------------------------------------------------------------


def test_process_returns_result(_patched):
    worker = qw.QPIWorker()
    worker.setup(_job())
    result = worker._process(worker._job)
    assert result is not None
    assert hasattr(result, "opd_m")


def test_progress_events_cover_every_phase_in_order(_patched):
    worker = qw.QPIWorker()
    worker.setup(_job())
    events: list[ProgressEvent] = []
    worker.progress.connect(events.append)

    worker._process(worker._job)

    phase_names = [e.phase for e in events if e.phase]
    assert phase_names == list(qw._PHASES), (
        f"expected {list(qw._PHASES)}, got {phase_names}"
    )

    assert events[0].phase == "" and events[0].phase_index == -1
    assert events[-1].done is True and events[-1].pct == 1.0
    assert events[-1].cancelled is False


def test_progress_pct_monotonic(_patched):
    worker = qw.QPIWorker()
    worker.setup(_job())
    events: list[ProgressEvent] = []
    worker.progress.connect(events.append)
    worker._process(worker._job)
    pcts = [e.pct for e in events]
    assert pcts == sorted(pcts), f"pct not monotonic: {pcts}"


def test_non_finite_phase_is_sanitized(_patched):
    """NaN/Inf in the input phase must not propagate — the worker sanitizes
    in the 'validate' phase."""
    phase = np.zeros((8, 8), dtype=np.float64)
    phase[0, 0] = np.nan
    phase[1, 1] = np.inf
    worker = qw.QPIWorker()
    worker.setup(_job(phase_unwrapped=phase))
    # Should not raise — the validator coerces non-finite → 0.
    worker._process(worker._job)


def test_emit_error_fans_out_structured_and_legacy():
    """Failures surface on both the structured and legacy signals."""
    worker = qw.QPIWorker()
    structured: list[ErrorEvent] = []
    legacy: list[str] = []
    worker.error_event.connect(structured.append)
    worker.qpi_failed.connect(legacy.append)

    worker._emit_error(ValueError("synthetic compute failure"), job=_job())

    assert len(structured) == 1
    ev = structured[0]
    assert ev.severity == Severity.ERROR
    assert ev.title == "QPI failed"
    assert "synthetic compute failure" in ev.cause
    assert ev.context["wavelength_nm"] == pytest.approx(632.8)
    assert ev.context["pixel_um"] == pytest.approx(3.45)
    assert ev.context["n_sample"] == pytest.approx(1.38)
    assert ev.context["n_medium"] == pytest.approx(1.337)
    assert ev.trace is not None

    assert len(legacy) == 1
    assert "synthetic compute failure" in legacy[0]


def test_suggest_action_memory_error():
    hint = qw._suggest_action(MemoryError("out of memory"), _job())
    assert "smaller" in hint.lower() or "crop" in hint.lower()


def test_suggest_action_bad_n_sample():
    hint = qw._suggest_action(
        ValueError("n_sample must be positive"),
        _job(n_sample=0.0),
    )
    assert "sample refractive index" in hint.lower()


def test_suggest_action_non_finite_phase():
    hint = qw._suggest_action(
        ValueError("phase contains non-finite values"),
        _job(),
    )
    assert "reconstruction" in hint.lower() or "non-finite" in hint.lower()


def test_no_job_configured_emits_error():
    """``run()`` is a no-op on an unconfigured worker, but it still fans
    out an ErrorEvent so the UI can show a sensible message."""
    worker = qw.QPIWorker()
    events: list[ErrorEvent] = []
    worker.error_event.connect(events.append)
    # Call run() directly — no QThread start, no event loop.
    worker.run()
    assert len(events) == 1
    assert "No QPI job" in events[0].cause


def test_cancellation_short_circuits(_patched):
    """Interruption observed at a phase boundary must raise
    :class:`OperationCancelled` without completing the pipeline."""
    worker = qw.QPIWorker()
    worker.setup(_job())
    # Force the interruption check to report True at the first phase boundary.
    worker.isInterruptionRequested = lambda: True  # type: ignore[assignment]

    with pytest.raises(qw.OperationCancelled):
        worker._process(worker._job)
