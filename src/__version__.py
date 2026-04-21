"""Version metadata for DHM Reconstruction.

Surfaced in the About dialog, audit log entries, and generated reports.
Keep `__version__` in lockstep with CHANGELOG.md.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

__version__ = "1.0.0"
__build_date__ = "2026-04-20"
__app_name__ = "DHM Reconstruction"
__vendor__ = "Hybrid Optics"


def _read_git_sha() -> str:
    """Best-effort git SHA. Returns 'unknown' if git is unavailable or not a repo."""
    try:
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=1.0, check=False,
        )
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"


__git_sha__ = _read_git_sha()


def version_string() -> str:
    """Human-readable one-liner, e.g. 'DHM Reconstruction 1.0.0 (a1b2c3d, 2026-04-20)'."""
    return f"{__app_name__} {__version__} ({__git_sha__}, {__build_date__})"


def version_info() -> dict:
    """Machine-readable version dict, suitable for audit logs and reports."""
    return {
        "app": __app_name__,
        "version": __version__,
        "build_date": __build_date__,
        "git_sha": __git_sha__,
        "vendor": __vendor__,
    }
