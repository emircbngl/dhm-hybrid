"""Bug regression — Phase: UX_PATCH (v1.0.1-ux pilot patch).

Tier 0 plumbing + Tier 1 visible design sprint. Two PySide6 6.x
port quirks captured here as ``lesson_only`` — they're version-
upgrade artifacts, not testable in the current 6.x baseline.

Run::

    python scripts/check_bugs_phase_ux_patch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_phase(Phase.UX_PATCH))
