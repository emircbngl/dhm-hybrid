"""Bug regression — Phase: PRE_PILOT (Faz 1 / v1.0.0 prep).

Covers early autofocus direction findings, Qt 3D viewer fixes, and
the 2026-04-17 lesson cluster. Six entries (B-001 to B-006) — three
``test`` (autofocus direction, wrapped-phase variance, greedy
walker), three ``lesson_only`` (venv ladder, silent ImportError,
Qt destroy/replace).

Run::

    python scripts/check_bugs_phase_pre_pilot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_phase(Phase.PRE_PILOT))
