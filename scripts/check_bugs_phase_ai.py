"""Bug regression — Phase: AI_FAZ_2 (v3.0).

Cellpose + cell-cycle classifier + onboarding wizard. Faz 2
omurgası. Empty until v3.0 sprint surfaces bugs.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402
if __name__ == "__main__":
    sys.exit(run_phase(Phase.AI_FAZ_2))
