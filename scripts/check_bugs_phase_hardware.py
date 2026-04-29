"""Bug regression — Phase: HARDWARE (v2.1.x).

Pylon / IDS / Thorlabs CameraSource + MP4 recording + live feed.
Empty until v2.1.x sprint surfaces bugs.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402
if __name__ == "__main__":
    sys.exit(run_phase(Phase.HARDWARE))
