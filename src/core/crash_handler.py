"""Local crash-dump handler for DHM Reconstruction.

Installs a ``sys.excepthook`` that persists uncaught exceptions to
``~/.dhm-reconstruction/crash/YYYYMMDD_HHMMSS_<pid>.json`` before chaining
to the previously installed hook. Never raises from within the hook — any
failure inside the handler is logged via ``logging`` but must not crash
the interpreter further.

The dump is self-contained (version, git SHA, platform, Python version,
PID, thread name, exception type/message, formatted traceback frames) so
a user can attach it to a bug report without any additional context.

No network, no telemetry — crash files live entirely on the local disk.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import platform
import sys
import threading
import traceback
from datetime import datetime, timezone
from types import TracebackType
from typing import Optional, Type

try:
    from __version__ import __version__, __git_sha__, __app_name__  # type: ignore
except Exception:  # pragma: no cover - best-effort import
    __version__ = "unknown"
    __git_sha__ = "unknown"
    __app_name__ = "DHM Reconstruction"


_LOG = logging.getLogger(__name__)

_CRASH_DIR = pathlib.Path.home() / ".dhm-reconstruction" / "crash"

# Preserve the hook that was in place before we installed ours so we can
# chain to it (pdb, IDE hooks, etc.). Falls back to sys.__excepthook__.
_previous_excepthook = None  # type: ignore[var-annotated]


def crash_dir() -> pathlib.Path:
    """Return the directory where crash files live.

    Exposed so a future "Show crash folder" menu item can open it in the
    OS file browser.
    """
    return _CRASH_DIR


def _build_payload(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: Optional[TracebackType],
) -> dict:
    """Assemble the JSON-serializable crash payload."""
    now = datetime.now(timezone.utc).astimezone()
    frames = traceback.format_exception(exc_type, exc_value, exc_traceback)
    return {
        "timestamp": now.isoformat(timespec="milliseconds"),
        "app": __app_name__,
        "version": __version__,
        "git_sha": __git_sha__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "exception_type": getattr(exc_type, "__name__", str(exc_type)),
        "exception_message": str(exc_value),
        "traceback": [frame.rstrip("\n") for frame in frames],
        "thread_name": threading.current_thread().name,
    }


def _dump_path() -> pathlib.Path:
    """Compute the crash file path for the current moment and PID."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _CRASH_DIR / f"{stamp}_{os.getpid()}.json"


def _write_atomic(path: pathlib.Path, payload: dict) -> None:
    """Write the payload atomically via temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _handle_exception(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: Optional[TracebackType],
) -> None:
    """The hook itself. Preserves the original hook for chaining.

    Wrapped entirely in try/except — a crash in the crash handler must
    never escalate. After writing the local dump we call the previously
    installed excepthook (falling back to ``sys.__excepthook__``) so the
    normal traceback still prints to stderr and any debugger hook runs.
    """
    # KeyboardInterrupt is intentionally not persisted — treat it as a
    # clean user abort and defer straight to the default hook.
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    try:
        payload = _build_payload(exc_type, exc_value, exc_traceback)
        path = _dump_path()
        _write_atomic(path, payload)
        _LOG.critical("uncaught exception persisted to %s", path)
        # Also stream the crash into the append-only audit log so tools
        # that already scrape it (LIMS correlation, ops dashboards) see
        # every crash without having to sniff the crash/ folder too.
        try:
            from core.audit import get_audit_log
            get_audit_log().record(
                action="crash",
                params={
                    "exception_type": payload["exception_type"],
                    "thread_name": payload["thread_name"],
                    "crash_file": str(path),
                },
                result_summary={
                    "exception_message": payload["exception_message"],
                },
            )
        except Exception:
            _LOG.debug("audit emit for crash failed", exc_info=True)
    except Exception:  # noqa: BLE001 - last-resort guard
        _LOG.critical("crash handler itself failed", exc_info=True)

    # Chain to the previous hook (pdb, IDE, or the default).
    try:
        chained = _previous_excepthook or sys.__excepthook__
        chained(exc_type, exc_value, exc_traceback)
    except Exception:  # noqa: BLE001
        _LOG.critical("crash handler itself failed", exc_info=True)


def install_crash_handler() -> None:
    """Install a ``sys.excepthook`` that writes local crash dumps and re-raises.

    Idempotent-ish: re-installing captures whatever hook was current,
    which means calling this twice will chain our own hook to itself.
    Don't do that. Call it once near the start of ``main()``.

    For PySide6 6.x, uncaught exceptions raised inside signal slots are
    routed through ``sys.excepthook`` by default, so no extra Qt-side
    plumbing is required here.
    """
    global _previous_excepthook
    _previous_excepthook = sys.excepthook
    sys.excepthook = _handle_exception


# ---------------------------------------------------------------------------
# Threading hook — Python 3.8+. Science workers spawn threads (audit log,
# ScienceDriver, batch_reconstruct, camera acquisition); an unhandled
# exception in one of them would otherwise vanish into the logger noise.
# Wiring threading.excepthook through the same ``_handle_exception`` means
# the crash dump JSON lands in the same folder regardless of origin thread.
# ---------------------------------------------------------------------------

_previous_threading_excepthook = None  # type: ignore[var-annotated]


def install_threading_excepthook() -> None:
    """Route ``threading.Thread`` uncaught exceptions into the same crash
    handler. Safe to call after or before :func:`install_crash_handler`.

    KeyboardInterrupt inside a thread is ignored (matches the stdlib
    default — only the main thread's KI turns into a graceful shutdown)."""
    global _previous_threading_excepthook
    _previous_threading_excepthook = threading.excepthook

    def _thread_hook(args: "threading.ExceptHookArgs") -> None:
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        _handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

    threading.excepthook = _thread_hook


def uninstall_crash_handler() -> None:
    """Restore the hooks that were in place before installation. Exists
    so tests can undo the global state they install — never call from
    production code, the crash dumps are valuable."""
    global _previous_excepthook, _previous_threading_excepthook
    if _previous_excepthook is not None:
        sys.excepthook = _previous_excepthook
        _previous_excepthook = None
    if _previous_threading_excepthook is not None:
        threading.excepthook = _previous_threading_excepthook
        _previous_threading_excepthook = None
