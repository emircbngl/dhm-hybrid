"""Shared runner / formatter for bug-history regression scripts.

Imports :mod:`scripts.bug_registry` and provides:

* :func:`run_one` — invoke pytest for a single :class:`BugEntry`.
* :func:`print_table` — colourised terminal table.
* :func:`run_phase` / :func:`run_all` — convenience helpers for the
  per-phase wrappers.

Each ``check_bugs_*.py`` wrapper stays trivial — declare the phase,
call ``run_phase(phase)``. New phases earn a 5-line script.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# Same-directory siblings — make sure `scripts/` is importable when
# these CLIs are invoked directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bug_registry import (  # noqa: E402
    BUG_REGISTRY,
    BugEntry,
    Phase,
    entries_for_phase,
    phase_summary,
)


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    bug: BugEntry
    status: str            # PASS | FAIL | SKIP | LESSON | MANUAL
    duration_ms: float = 0.0
    detail: str = ""


# ---------------------------------------------------------------------------
# Pytest invocation
# ---------------------------------------------------------------------------

def _pytest_cmd() -> List[str]:
    """Pick the right Python — sibling Phyton venv is the project's
    de-facto runner because Hybrid/venv lacks the full test stack
    (lesson 2026-04-17 'Wrong venv ladder')."""
    candidates = [
        ROOT.parent / "Phyton" / "venv" / "bin" / "python",
        ROOT / "venv" / "bin" / "python",
        Path(sys.executable),
    ]
    py = next((p for p in candidates if p.exists()), Path(sys.executable))
    return [str(py), "-m", "pytest"]


def run_one(entry: BugEntry, *,
            env_extras: Optional[dict] = None) -> RunResult:
    """Run one entry. ``lesson_only`` / ``manual`` entries are reported
    as such without invoking pytest — they have no automated
    surface."""
    if entry.status == "lesson_only":
        return RunResult(
            entry, "LESSON", 0.0,
            f"no automated test surface — see lessons.md "
            f"§ {entry.lesson_ref or '(none)'}",
        )
    if entry.status == "manual":
        return RunResult(
            entry, "MANUAL", 0.0,
            "hardware/env-dependent — verify in real session "
            f"(lessons.md § {entry.lesson_ref or '(none)'})",
        )
    if not entry.test:
        return RunResult(
            entry, "FAIL", 0.0,
            "registry says status='test' but no test node provided",
        )

    cmd = _pytest_cmd() + [entry.test, "--no-header", "-q",
                           "--tb=line"]
    env = {"PYTHONPATH": "src:tests"}
    if env_extras:
        env.update(env_extras)
    full_env = {**os.environ, **env}

    t0 = time.monotonic()
    try:
        cp = subprocess.run(
            cmd, cwd=ROOT, env=full_env,
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return RunResult(entry, "FAIL", 300_000.0, "timed out at 300 s")
    except FileNotFoundError as e:
        return RunResult(entry, "FAIL", 0.0, f"runner not found: {e}")
    dt = (time.monotonic() - t0) * 1000.0

    if cp.returncode == 0:
        if "passed" in cp.stdout:
            return RunResult(entry, "PASS", dt, "")
        if "skipped" in cp.stdout:
            return RunResult(entry, "SKIP", dt,
                             "no test ran (selector didn't match?)")
        return RunResult(entry, "PASS", dt, "(no count info)")
    last_lines = (cp.stdout or "").strip().splitlines()[-3:]
    return RunResult(entry, "FAIL", dt,
                     " | ".join(last_lines) or cp.stderr[:200])


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_STATUS_COLOR = {
    "PASS":   "\033[92m",
    "FAIL":   "\033[91m",
    "SKIP":   "\033[93m",
    "LESSON": "\033[94m",
    "MANUAL": "\033[95m",
}
_RESET = "\033[0m"


def _color(s: str, status: str, use_color: bool) -> str:
    if not use_color:
        return s
    return f"{_STATUS_COLOR.get(status, '')}{s}{_RESET}"


def print_table(results: List[RunResult], *,
                use_color: bool = True,
                show_phase: bool = False) -> None:
    """Pretty terminal table. ``show_phase=True`` adds a phase column
    — useful for the all-phases runner; per-phase runners suppress it
    since every row repeats the same value."""
    if show_phase:
        print(f"{'ID':<7} {'Date':<12} {'Phase':<28} "
              f"{'Status':<8} {'ms':>7}  Topic")
        print("-" * 130)
    else:
        print(f"{'ID':<7} {'Date':<12} {'Status':<8} {'ms':>7}  Topic")
        print("-" * 110)
    for r in results:
        if show_phase:
            line = (
                f"{r.bug.bug_id:<7} {r.bug.date:<12} "
                f"{r.bug.phase.value:<28} "
                f"{_color(r.status.ljust(8), r.status, use_color)} "
                f"{int(r.duration_ms):>7}  {r.bug.topic}"
            )
        else:
            line = (
                f"{r.bug.bug_id:<7} {r.bug.date:<12} "
                f"{_color(r.status.ljust(8), r.status, use_color)} "
                f"{int(r.duration_ms):>7}  {r.bug.topic}"
            )
        print(line)
        if r.detail and r.status == "FAIL":
            print(f"           └─ {r.detail}")


def summary(results: List[RunResult]) -> dict:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0,
              "LESSON": 0, "MANUAL": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    counts["total"] = len(results)
    return counts


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def run_phase(phase: Phase, *,
              json_output: bool = False,
              use_color: bool = True) -> int:
    """Run every entry tagged with ``phase``. Empty phases (future
    sprints not yet started) print a banner and return 0 — they're
    valid, just empty.

    Returns
    -------
    int
        ``0`` if no FAILs, ``1`` if one or more, ``2`` for runner
        errors (no-such-phase etc).
    """
    entries = entries_for_phase(phase)
    if not entries:
        print(f"(phase {phase.value!r} has no bugs registered yet — "
              f"add entries to scripts/bug_registry.py as they surface)")
        return 0
    return _run_and_report(
        entries, json_output=json_output, use_color=use_color,
        header=f"Phase: {phase.value}  ({len(entries)} entries)",
        show_phase=False,
    )


def run_all(*,
            json_output: bool = False,
            use_color: bool = True) -> int:
    """Run the entire registry. Header prints a phase summary so
    high-level state is visible at a glance."""
    summary_dict = phase_summary()
    summary_str = ", ".join(
        f"{k}={v}" for k, v in summary_dict.items()
    )
    return _run_and_report(
        BUG_REGISTRY, json_output=json_output, use_color=use_color,
        header=(f"All phases  ({len(BUG_REGISTRY)} entries):  "
                f"{summary_str}"),
        show_phase=True,
    )


def _run_and_report(entries: List[BugEntry], *,
                    json_output: bool, use_color: bool,
                    header: str,
                    show_phase: bool) -> int:
    if not json_output:
        print(header)
        print()
    results = [run_one(e) for e in entries]
    if json_output:
        print(json.dumps({
            "header": header,
            "summary": summary(results),
            "results": [
                {**asdict(r.bug), "phase": r.bug.phase.value,
                 "result": r.status, "duration_ms": r.duration_ms,
                 "detail": r.detail}
                for r in results
            ],
        }, indent=2, default=str))
    else:
        print_table(results, use_color=use_color, show_phase=show_phase)
        s = summary(results)
        print()
        print(
            f"Total: {s['total']}   "
            f"PASS={s.get('PASS', 0)}   FAIL={s.get('FAIL', 0)}   "
            f"SKIP={s.get('SKIP', 0)}   LESSON={s.get('LESSON', 0)}   "
            f"MANUAL={s.get('MANUAL', 0)}"
        )
    return 1 if any(r.status == "FAIL" for r in results) else 0


__all__ = [
    "RunResult",
    "run_one",
    "print_table",
    "summary",
    "run_phase",
    "run_all",
]
