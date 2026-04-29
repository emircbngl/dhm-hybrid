"""Append a new entry to ``scripts/bug_registry.py`` from the CLI.

Sprint hygiene tool: editing ``BUG_REGISTRY`` by hand is error-
prone (forget the next ``B-NNN`` id, mis-indent the BugEntry,
forget to commit). This helper:

* Parses the existing registry to find the next free id.
* Validates the phase enum value.
* Splices a fresh ``BugEntry(...)`` into the right place.
* Writes the file atomically (temp + rename).
* Re-runs the bug-registry script as a sanity check.

Usage
-----
::

    python scripts/add_bug.py \\
        --phase pilot_patches_v2_0_6_post \\
        --topic "Bug summary in one line." \\
        --status test \\
        --test "tests/test_xxx.py::test_yyy" \\
        --lesson-ref "(sprint reference)"

Status options: ``test`` | ``lesson_only`` | ``manual``. ``--test``
is required when ``--status test`` (otherwise ignored). ``--date``
defaults to today (UTC).

Re-validation runs ``scripts/check_bugs.py --filter <new_id>``;
non-zero exit aborts the helper *after* the file is written so
you can debug, but you'll see a clear "DOES NOT pass" message.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "bug_registry.py"
RUNNER_PATH = ROOT / "check_bugs.py"


def _next_bug_id(registry_text: str) -> str:
    """Walk every ``bug_id="B-NNN"`` literal, return ``B-(max+1)``.

    The walk is regex-based to avoid importing the registry — we
    don't want to side-effect anything during a hygiene run.
    """
    ids = re.findall(r'bug_id="B-(\d+)"', registry_text)
    if not ids:
        return "B-001"
    next_n = max(int(x) for x in ids) + 1
    return f"B-{next_n:03d}"


def _known_phases(registry_text: str) -> list[str]:
    """All ``Phase`` enum string values from the registry source."""
    return re.findall(r'^\s*[A-Z_]+\s*=\s*"([a-z_0-9]+)"',
                      registry_text, flags=re.MULTILINE)


def _phase_enum_member(phase_value: str, registry_text: str) -> str:
    """Map a ``phase`` enum string back to its ``Phase.<MEMBER>``
    name so we can write valid Python."""
    pat = (
        r'^\s*([A-Z_]+)\s*=\s*"' + re.escape(phase_value) + r'"'
    )
    m = re.search(pat, registry_text, flags=re.MULTILINE)
    if not m:
        raise SystemExit(
            f"phase {phase_value!r} not found in Phase enum. "
            f"Known: {_known_phases(registry_text)}",
        )
    return f"Phase.{m.group(1)}"


def _splice_entry(registry_text: str, entry_block: str) -> str:
    """Insert ``entry_block`` immediately before the closing
    ``]`` of ``BUG_REGISTRY``.

    Walks the file from the ``BUG_REGISTRY: List[BugEntry] = [``
    declaration, tracking ``[`` / ``]`` depth so we splice before
    the *matching* close, not the last close in the file (which
    might belong to ``__all__`` or a helper).

    Bug fix 2026-04-28: previous version used ``rfind("\\n]\\n")``
    which matched the ``__all__`` closing bracket and inserted the
    new BugEntry inside ``__all__`` — silent corruption that only
    surfaced when ``check_bugs.py --filter B-049`` returned
    "no bug with id". Bracket-matching is the right primitive."""
    decl = "BUG_REGISTRY: List[BugEntry] = ["
    decl_idx = registry_text.find(decl)
    if decl_idx < 0:
        raise SystemExit(
            "could not find `BUG_REGISTRY: List[BugEntry] = [` "
            "declaration in registry file",
        )
    # Start scanning *after* the opening ``[`` of the declaration.
    scan_start = decl_idx + len(decl)
    depth = 1
    i = scan_start
    while i < len(registry_text) and depth > 0:
        ch = registry_text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
        i += 1
    else:
        raise SystemExit(
            "could not find matching closing `]` for BUG_REGISTRY",
        )
    # Splice point is the line containing the matching `]`. Walk
    # back to its newline so the new entry lands on its own line
    # immediately above it.
    line_start = registry_text.rfind("\n", 0, close_idx)
    head = registry_text[: line_start + 1]
    tail = registry_text[line_start + 1:]
    # Drop trailing whitespace from head so the new BugEntry block
    # has a single blank line of separation, matching the file's
    # existing style.
    return head.rstrip() + "\n" + entry_block + "\n" + tail


def _format_entry(*,
                  bug_id: str,
                  date: str,
                  phase: str,
                  topic: str,
                  status: str,
                  test: str | None,
                  lesson_ref: str | None) -> str:
    """Render a ``BugEntry(...)`` block matching the file's style."""
    lines = [
        "    BugEntry(",
        f'        bug_id="{bug_id}", date="{date}",',
        f"        phase={phase},",
        f"        topic={_repr_long(topic)},",
        f'        status="{status}",',
    ]
    if status == "test":
        if not test:
            raise SystemExit("--test required when --status=test")
        lines.append(f'        test="{test}",')
    if lesson_ref:
        lines.append(f"        lesson_ref={_repr_long(lesson_ref)},")
    lines.append("    ),")
    return "\n".join(lines)


def _repr_long(s: str) -> str:
    """Choose between single-line ``"..."`` and multi-line wrapped
    strings so long topics stay readable. Returns a Python literal
    safe to splice directly into source."""
    s = s.replace('"', '\\"')
    if len(s) <= 64:
        return f'"{s}"'
    # Greedy wrap at 64 chars between words. Output multi-line
    # implicit-concat literal so each line stays readable.
    words = s.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= 64:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    out = ['"' + lines[0] + ' "']
    for line in lines[1:]:
        out.append('              "' + line + ' "')
    return "\n".join(out).rstrip(' "') + '"'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True,
                    help="Phase enum string value, e.g. "
                         "'pilot_patches_v2_0_6_post'")
    ap.add_argument("--topic", required=True,
                    help="One-line summary of the bug + fix.")
    ap.add_argument("--status",
                    choices=["test", "lesson_only", "manual"],
                    default="test")
    ap.add_argument("--test", default=None,
                    help="pytest node id; required when status=test")
    ap.add_argument("--lesson-ref", default=None,
                    help="lessons.md heading reference")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the proposed entry, don't write")
    args = ap.parse_args(argv)

    if not REGISTRY_PATH.exists():
        print(f"registry not found at {REGISTRY_PATH}", file=sys.stderr)
        return 2
    text = REGISTRY_PATH.read_text(encoding="utf-8")

    bug_id = _next_bug_id(text)
    phase_member = _phase_enum_member(args.phase, text)
    date = args.date or _dt.date.today().isoformat()

    block = _format_entry(
        bug_id=bug_id, date=date, phase=phase_member,
        topic=args.topic, status=args.status,
        test=args.test, lesson_ref=args.lesson_ref,
    )

    if args.dry_run:
        print("# would append:\n" + block)
        return 0

    new_text = _splice_entry(text, block)

    # Atomic write — same dir tempfile + os.replace.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=str(REGISTRY_PATH.parent),
        prefix=REGISTRY_PATH.stem + ".", suffix=".tmp",
        delete=False, encoding="utf-8",
    )
    try:
        tmp.write(new_text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, REGISTRY_PATH)
    except Exception:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        raise

    print(f"appended {bug_id} ({args.phase})")

    # Sanity-check: run the bug runner filtered to the new id.
    if RUNNER_PATH.exists():
        try:
            cp = subprocess.run(
                [sys.executable, str(RUNNER_PATH),
                 "--filter", bug_id, "--no-color"],
                cwd=ROOT.parent, capture_output=True, text=True,
                timeout=120,
            )
        except Exception as exc:
            print(f"warning: validation run failed: {exc}",
                  file=sys.stderr)
            return 0
        print(cp.stdout)
        if cp.returncode != 0:
            print(f"WARNING: {bug_id} DOES NOT pass — entry was "
                  f"written, fix the test or registry detail.",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
