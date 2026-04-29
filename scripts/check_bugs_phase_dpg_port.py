"""Bug regression — Phase: DPG_PORT (v2.0.0 → v2.0.6).

Dear PyGui frontend port + scientific param audit + compliance +
workflow tools mega sprints. The biggest phase by entry count
(~17). Surfaces every non-trivial v1→v2 port issue + DPG-specific
macOS quirks (file_dialog, viewport_menu_bar, drop callback,
mvClickedHandler).

Most entries are ``manual`` because they're hardware/env quirks
that can't be tested headlessly — keep the lesson refs current and
verify in real DPG sessions.

Run::

    python scripts/check_bugs_phase_dpg_port.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bug_runner import run_phase  # noqa: E402
from bug_registry import Phase  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_phase(Phase.DPG_PORT))
