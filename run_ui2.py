"""Launcher for the Dear PyGui frontend (v2 UI).

Runs ``python run_ui2.py`` from the project root. Adds ``src`` to
``sys.path`` so ``ui2`` and ``core`` both import cleanly as
top-level packages, then installs the crash-dump handlers that shared
``core.crash_handler`` exposes so any uncaught exception (main thread
or worker) lands in ``~/.dhm-reconstruction/crash/`` AND the shared
audit log.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    # Install crash + threading excepthooks BEFORE importing the app —
    # an ImportError during app load is exactly the kind of failure the
    # crash dump is most useful for.
    from core.crash_handler import (
        install_crash_handler, install_threading_excepthook,
    )
    install_crash_handler()
    install_threading_excepthook()
    from ui2.app import main as app_main  # type: ignore
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
