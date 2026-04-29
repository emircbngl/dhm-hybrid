"""Bug regression — Phase: TRACKING (v2.0.8).

Drift correction + per-cell tracking + NIST bead calibration +
multi-line profile. Empty until v2.0.8 sprint kicks off; the
script exists so the per-phase ritual is in place from day 1.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402
if __name__ == "__main__":
    sys.exit(run_phase(Phase.TRACKING))
