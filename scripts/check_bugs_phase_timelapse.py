"""Bug regression — Phase: TIMELAPSE_FOUNDATION (v2.0.7, current).

The current sprint. As of 2026-04-27 the registry is empty —
T0 (Session model) and T4 (preset edit) shipped clean, no
regressions captured here yet. T1 (CLI runner), T2 (CSV export),
T3 (multi-user), T5 (audit viewer), T6 (batch resume) still in
flight; whichever surfaces a real bug earns an entry in the
registry.

Empty-phase output is expected to print a banner and exit 0 —
"no bugs registered yet" is a valid result, not a failure.

Run::

    python scripts/check_bugs_phase_timelapse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_phase(Phase.TIMELAPSE_FOUNDATION))
