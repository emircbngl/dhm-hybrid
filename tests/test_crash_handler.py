"""Crash handler regression tests — v2.0.6.

Two hooks, four invariants:

* ``install_crash_handler`` writes a JSON dump per uncaught exception
  in the main thread + chains to the previously installed hook.
* ``install_threading_excepthook`` routes worker-thread crashes into
  the same JSON dump directory + audit log.
* Both hooks ignore KeyboardInterrupt — CTRL-C is a clean exit, not
  a crash worth a dump.
* ``uninstall_crash_handler`` restores whatever was there before so
  tests don't leak global state.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import crash_handler  # noqa: E402


@pytest.fixture
def restore_hooks():
    """Snapshot + restore sys.excepthook + threading.excepthook so
    installing crash hooks in one test can't poison subsequent ones."""
    prev_sys = sys.excepthook
    prev_thr = threading.excepthook
    yield
    sys.excepthook = prev_sys
    threading.excepthook = prev_thr
    # Also reset the module-level bookkeeping so the next call to
    # ``install_crash_handler`` doesn't chain to our test hook.
    crash_handler._previous_excepthook = None
    crash_handler._previous_threading_excepthook = None


@pytest.fixture
def crash_dir(tmp_path, monkeypatch):
    d = tmp_path / "crash"
    monkeypatch.setattr(crash_handler, "_CRASH_DIR", d)
    return d


# ---------------------------------------------------------------------------
# Main-thread hook
# ---------------------------------------------------------------------------

def test_install_writes_dump_for_main_thread(crash_dir, restore_hooks):
    # Pin a benign prior hook: the handler correctly CHAINS to whatever was
    # installed before it, and under this test runner that is pytest-qt's
    # exception capture (active since pytest-qt entered the env with ui3),
    # which turns the chained call into a false "Exceptions caught in Qt
    # event loop" failure. Chaining itself is covered by
    # test_install_chains_to_previous_hook (2026-07-06, B-084).
    sys.excepthook = lambda *exc: None
    crash_handler.install_crash_handler()
    try:
        raise RuntimeError("boom from main")
    except RuntimeError:
        import sys as _s
        exc_type, exc_value, exc_tb = _s.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)

    dumps = list(crash_dir.glob("*.json"))
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text())
    assert payload["exception_type"] == "RuntimeError"
    assert "boom from main" in payload["exception_message"]
    assert payload["thread_name"] == "MainThread"


def test_install_chains_to_previous_hook(crash_dir, restore_hooks):
    """Previous hook must still run — otherwise pdb, IDE hooks, and
    default stderr output would be silently dropped."""
    seen = {}

    def prior_hook(et, ev, tb):
        seen["called"] = (et, str(ev))

    sys.excepthook = prior_hook
    crash_handler.install_crash_handler()
    try:
        raise ValueError("chain me")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)

    assert seen.get("called") is not None
    assert seen["called"][0] is ValueError
    assert "chain me" in seen["called"][1]


def test_keyboard_interrupt_not_persisted(crash_dir, restore_hooks):
    """Ctrl-C → clean exit; we must NOT leave a scary crash file."""
    crash_handler.install_crash_handler()
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exc_type, exc_value, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)
    assert not list(crash_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Threading hook
# ---------------------------------------------------------------------------

def test_threading_hook_captures_worker_crash(crash_dir, restore_hooks):
    # Pin benign prior hooks — see test_install_writes_dump_for_main_thread.
    # The threading chain otherwise lands in pytest's threadexception
    # capture and false-fails the test (2026-07-06, B-084).
    sys.excepthook = lambda *exc: None
    threading.excepthook = lambda args: None
    crash_handler.install_crash_handler()
    crash_handler.install_threading_excepthook()

    def worker():
        raise RuntimeError("boom from thread")

    t = threading.Thread(target=worker, name="dhm-worker-42")
    t.start()
    t.join()

    dumps = list(crash_dir.glob("*.json"))
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text())
    assert payload["exception_type"] == "RuntimeError"
    assert "boom from thread" in payload["exception_message"]
    assert payload["thread_name"] == "dhm-worker-42"


def test_threading_hook_ignores_keyboard_interrupt(crash_dir, restore_hooks):
    """Matches the stdlib default — KI in a worker is a no-op, only
    the main thread's KI gets handled."""
    crash_handler.install_crash_handler()
    crash_handler.install_threading_excepthook()

    def worker():
        raise KeyboardInterrupt()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not list(crash_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Audit integration — every crash should also land in the audit log
# ---------------------------------------------------------------------------

def test_crash_also_emits_audit_record(crash_dir, restore_hooks,
                                       tmp_path, monkeypatch):
    from core import audit as audit_mod
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    monkeypatch.setattr(audit_mod, "_AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit_mod, "_SINGLETON", None)

    # Pin a benign prior hook — see test_install_writes_dump_for_main_thread
    # (2026-07-06, B-084).
    sys.excepthook = lambda *exc: None
    crash_handler.install_crash_handler()
    try:
        raise RuntimeError("auditable crash")
    except RuntimeError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_value, exc_tb)

    audit_files = list(audit_dir.glob("*.jsonl"))
    assert audit_files, "no audit records written on crash"
    records = [json.loads(line) for line in
               audit_files[0].read_text().splitlines() if line.strip()]
    crash_records = [r for r in records if r["action"] == "crash"]
    assert crash_records, f"no 'crash' action in audit: " \
                          f"{[r['action'] for r in records]}"
    assert crash_records[0]["params"]["exception_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Uninstall — restore to the previous state
# ---------------------------------------------------------------------------

def test_uninstall_restores_previous_hooks(restore_hooks):
    baseline_sys = sys.excepthook
    baseline_thr = threading.excepthook

    crash_handler.install_crash_handler()
    crash_handler.install_threading_excepthook()
    assert sys.excepthook is not baseline_sys
    assert threading.excepthook is not baseline_thr

    crash_handler.uninstall_crash_handler()
    assert sys.excepthook is baseline_sys
    assert threading.excepthook is baseline_thr
