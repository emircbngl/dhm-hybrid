"""Bug regression — Phase: PILOT_PATCHES (5-bug acil + end-to-end +
multi-focus refactor, 2026-04-24 evening).

Pilot pilot dönüşü acil sprint: Bug #1 hologram flip, Bug #2 ref
subtract, Bug #3 first load, Bug #4 autofocus speed, Bug #5 scroll;
plus end-to-end synthesis tests (lateral + depth correction at
autofocus z) and the multi-focus refactor that put 7 search
algorithms on a single ``_make_fast_evaluator`` path.

Ten entries (B-026 to B-035). Includes one of the "perf prediction
was 4200× off" lessons (B-034) and the carrier-residual-makes-lateral-
diameter-unmeasurable insight (B-033).

Run::

    python scripts/check_bugs_phase_pilot_patches.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_phase(Phase.PILOT_PATCHES))
