"""MockStage: bounds, NaN, listener notifications, thread-safety smoke.

The mock implementation backs every AI stage tool in v1. These tests
also serve as the contract any real-hardware driver must satisfy.
"""
from __future__ import annotations

import threading

import pytest

from core.stage import MockStage, StageError


def test_initial_position_zero():
    s = MockStage()
    assert s.get_position() == (0.0, 0.0, 0.0)


def test_initial_position_custom():
    s = MockStage(start=(1.0, 2.0, 3.0))
    assert s.get_position() == (1.0, 2.0, 3.0)


def test_move_relative_accumulates():
    s = MockStage()
    s.move_relative(1.0, 2.0, 3.0)
    s.move_relative(0.5, 0.0, -1.0)
    assert s.get_position() == pytest.approx((1.5, 2.0, 2.0))


def test_move_absolute_replaces():
    s = MockStage()
    s.move_relative(10.0, 10.0, 10.0)
    s.move_absolute(0.0, 0.0, 0.0)
    assert s.get_position() == (0.0, 0.0, 0.0)


def test_move_relative_clamps_to_bounds():
    s = MockStage()
    s.move_relative(500.0, 0.0, 0.0)
    x, y, z = s.get_position()
    assert x == MockStage.BOUNDS_MM[1]


def test_move_absolute_rejects_out_of_bounds():
    s = MockStage()
    with pytest.raises(StageError):
        s.move_absolute(999.0, 0.0, 0.0)


def test_move_relative_rejects_nan():
    s = MockStage()
    with pytest.raises(StageError):
        s.move_relative(float("nan"), 0.0, 0.0)


def test_move_absolute_rejects_nan():
    s = MockStage()
    with pytest.raises(StageError):
        s.move_absolute(float("nan"), 0.0, 0.0)


def test_home_resets_to_origin():
    s = MockStage()
    s.move_relative(5.0, 5.0, 5.0)
    pos = s.home()
    assert pos == (0.0, 0.0, 0.0)
    assert s.get_position() == (0.0, 0.0, 0.0)


def test_listener_fires_on_every_move():
    s = MockStage()
    seen: list = []
    s.add_listener(lambda p: seen.append(p))
    s.move_relative(1.0, 0.0, 0.0)
    s.move_absolute(2.0, 0.0, 0.0)
    s.home()
    assert len(seen) == 3
    assert seen[-1] == (0.0, 0.0, 0.0)


def test_listener_unsubscribe_works():
    s = MockStage()
    seen: list = []
    unsub = s.add_listener(lambda p: seen.append(p))
    unsub()
    s.move_relative(1.0, 0.0, 0.0)
    assert seen == []


def test_listener_exception_does_not_crash_stage():
    s = MockStage()

    def raising(_):
        raise RuntimeError("nope")

    s.add_listener(raising)
    # Move must not raise even though listener does.
    s.move_relative(1.0, 0.0, 0.0)
    assert s.get_position() == (1.0, 0.0, 0.0)


def test_concurrent_moves_serialize_safely():
    s = MockStage()

    def worker():
        for _ in range(50):
            s.move_relative(0.01, 0.0, 0.0)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    x, _, _ = s.get_position()
    # 4 workers * 50 moves * 0.01 mm = 2.0 mm
    assert x == pytest.approx(2.0, abs=1e-9)
