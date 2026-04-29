"""Bug regression — Phase: PAPER_READY (v2.0.9).

Vector PDF + Zenodo bundle + line profile click-drag ROI + crash
handler + WCAG-AA. Empty until v2.0.9 sprint surfaces bugs.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402
if __name__ == "__main__":
    sys.exit(run_phase(Phase.PAPER_READY))
