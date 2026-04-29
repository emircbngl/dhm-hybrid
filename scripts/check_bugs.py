"""Bug-history regression — all phases.

Runs every entry in :data:`scripts.bug_registry.BUG_REGISTRY` across
every phase and prints a unified table. For per-phase scope use the
lighter wrappers under ``scripts/check_bugs_phase_*.py``.

Usage
-----
::

    python scripts/check_bugs.py                # full table
    python scripts/check_bugs.py --phase pre_pilot
    python scripts/check_bugs.py --filter B-029
    python scripts/check_bugs.py --json         # machine-readable
    python scripts/check_bugs.py --no-color

Exit
----
``0`` = every testable bug passes. ``1`` = at least one regressed.
``2`` = invalid CLI args.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/ on path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bug_runner import run_all, run_one, print_table, summary  # noqa: E402
from bug_registry import BUG_REGISTRY, Phase  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=[p.value for p in Phase],
                    help="Run only one phase (default: all)")
    ap.add_argument("--filter", metavar="ID",
                    help="Run only one bug by id (e.g. B-029)")
    ap.add_argument("--json", action="store_true",
                    help="JSON output")
    ap.add_argument("--no-color", action="store_true",
                    help="Disable ANSI colour")
    args = ap.parse_args(argv)
    use_color = not args.no_color

    if args.filter:
        entries = [e for e in BUG_REGISTRY if e.bug_id == args.filter]
        if not entries:
            print(f"no bug with id={args.filter}", file=sys.stderr)
            return 2
        results = [run_one(e) for e in entries]
        if not args.json:
            print_table(results, use_color=use_color, show_phase=True)
            s = summary(results)
            print()
            print(f"PASS={s['PASS']}  FAIL={s['FAIL']}  "
                  f"LESSON={s['LESSON']}  MANUAL={s['MANUAL']}")
        else:
            import json as _json
            from dataclasses import asdict
            print(_json.dumps({
                "summary": summary(results),
                "results": [
                    {**asdict(r.bug), "phase": r.bug.phase.value,
                     "result": r.status,
                     "duration_ms": r.duration_ms, "detail": r.detail}
                    for r in results
                ],
            }, indent=2, default=str))
        return 1 if any(r.status == "FAIL" for r in results) else 0

    if args.phase:
        from _bug_runner import run_phase
        return run_phase(Phase(args.phase),
                         json_output=args.json, use_color=use_color)

    return run_all(json_output=args.json, use_color=use_color)


if __name__ == "__main__":
    sys.exit(main())
