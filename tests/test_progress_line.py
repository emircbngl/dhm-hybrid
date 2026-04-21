"""Silent progress line — state transitions and signal handling.

The widget's entire API is ``on_progress(event)`` plus three internal
rendering paths; we drive it directly rather than wiring a QTimer
event loop, to keep the tests deterministic.
"""
from __future__ import annotations

import os
import time

import pytest

from core.progress import Operation, ProgressEvent


def _headless_qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return None
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception:
            return None
    return app


def _event(pct: float = 0.5, **overrides) -> ProgressEvent:
    defaults = dict(
        operation_id="op-1",
        operation_kind="reconstruct",
        phase="propagate",
        phase_index=3,
        phase_count=7,
        pct=pct,
        elapsed_s=0.0,
        eta_s=None,
        message="",
        done=False,
        cancelled=False,
    )
    defaults.update(overrides)
    return ProgressEvent(**defaults)


# ---- module import ----------------------------------------------------------

def test_module_imports():
    import gui.widgets.progress_line as pl  # noqa: F401
    assert hasattr(pl, "ProgressLine")


# ---- behaviour --------------------------------------------------------------

def _visible(w) -> bool:
    """Qt's ``isVisible()`` requires the whole parent chain to be shown.
    Tests here don't raise a real window, so we check the widget's own
    hidden bit — the one ``setVisible(True/False)`` actually flips."""
    return not w.isHidden()


def test_hidden_before_threshold():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.progress_line import ProgressLine

    widget = ProgressLine()
    widget.on_progress(_event(pct=0.1))
    # Under 500 ms elapsed — line stays hidden.
    assert _visible(widget._bar) is False
    assert _visible(widget._caption) is False
    assert _visible(widget._cancel_hint) is False


def test_line_only_in_mid_window():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.progress_line import ProgressLine

    widget = ProgressLine()
    widget.on_progress(_event(pct=0.3))
    # Back-date the start so the first render lands in the line-only band.
    widget._start_time = time.monotonic() - 1.0
    widget._on_tick()
    assert _visible(widget._bar) is True
    assert _visible(widget._caption) is False
    assert _visible(widget._cancel_hint) is False
    # pct maps into the 0..1000 integer bar.
    assert widget._bar.value() == 300


def test_caption_after_long_wait():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.progress_line import ProgressLine

    widget = ProgressLine()
    widget.on_progress(_event(pct=0.8, phase="propagate"))
    widget._start_time = time.monotonic() - 6.0
    widget._on_tick()
    assert _visible(widget._bar) is True
    assert _visible(widget._caption) is True
    assert _visible(widget._cancel_hint) is True
    assert "propagate" in widget._caption.text()
    assert "80%" in widget._caption.text()


def test_done_resets_state():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.progress_line import ProgressLine

    widget = ProgressLine()
    widget.on_progress(_event(pct=0.5))
    widget._start_time = time.monotonic() - 6.0
    widget._on_tick()
    assert _visible(widget._caption) is True

    widget.on_progress(_event(pct=1.0, done=True))
    assert _visible(widget._bar) is False
    assert _visible(widget._caption) is False
    assert widget._start_time is None


def test_non_progress_event_is_ignored():
    """A stray non-ProgressEvent payload must not crash the slot."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.progress_line import ProgressLine

    widget = ProgressLine()
    widget.on_progress("not an event")
    widget.on_progress(None)
    widget.on_progress(42)
    # Still in the hidden state.
    assert _visible(widget._bar) is False


def test_cancel_hint_click_emits_signal():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.progress_line import ProgressLine

    widget = ProgressLine()
    fired = []
    widget.cancel_requested.connect(lambda: fired.append(True))
    # We replaced mousePressEvent with a lambda in __init__; call it
    # directly — the widget owns that contract and tests should exercise it.
    widget._cancel_hint.mousePressEvent(None)  # type: ignore[arg-type]
    assert fired == [True]


def test_monotonic_pct_through_full_operation():
    """Driving the full ``Operation`` phase lifecycle must never drop
    ``pct`` backward on the widget's bar."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.progress_line import ProgressLine

    widget = ProgressLine()
    widget._start_time = time.monotonic() - 1.0  # land in line-only band

    seen: list[int] = []

    def sink(ev: ProgressEvent) -> None:
        widget.on_progress(ev)
        seen.append(widget._bar.value())

    op = Operation(kind="test", phases=("a", "b", "c"))
    op.on_progress = sink
    with op.phase("a"):
        pass
    with op.phase("b"):
        pass
    with op.phase("c"):
        pass
    op.finish()

    # Non-decreasing *during* the operation. The final ``done=True`` event
    # resets the bar to 0, which is the design: the widget erases itself
    # when work completes.
    in_flight = seen[:-1]
    assert in_flight == sorted(in_flight), in_flight
    assert seen[-1] == 0, "done event should reset bar to 0"
