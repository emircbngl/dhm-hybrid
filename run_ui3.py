#!/usr/bin/env python3
"""Launch the ui3 (PySide6 + pyqtgraph) DHM desktop app.

    python run_ui3.py

ui3 is the from-scratch Qt rebuild of the frontend; the Dear PyGui ``ui2``
app (``run_ui2.py``) stays available until ui3 reaches full parity.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ui3.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
