#!/usr/bin/env python3
"""Fail on Turkish-specific characters inside ``src/gui/``.

The lab pilot (Anna / Lindqvist) shipped a mixed-language build; v1.0.1-ux
normalises everything user-facing to English. This guard exists so a future
well-meaning edit doesn't silently slip a Turkish word back into the UI.

Scope: ``src/gui/**/*.py`` — that's where user-visible strings live. Core
science modules and tests are unaffected; so are docs, CHANGELOG, and the
``tasks/`` notes (those can stay bilingual for the author's sanity).

Exits 0 on clean, 1 on hit. Prints every offending line so the committer
can fix them before re-trying.

Invoke directly, or via ``pre-commit`` using the hook entry in
``.pre-commit-config.yaml``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

_TURKISH_RX = re.compile(r"[ğşıçöüĞŞİÇÖÜâîû]")


def _targets(argv: list[str]) -> Iterable[Path]:
    """If pre-commit handed us files, use those; otherwise sweep ``src/gui``."""
    if argv:
        for a in argv:
            p = Path(a)
            if p.is_file() and p.suffix == ".py":
                yield p
        return
    root = Path(__file__).resolve().parents[1] / "src" / "gui"
    yield from root.rglob("*.py")


def main(argv: list[str]) -> int:
    hits: list[str] = []
    for path in _targets(argv):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"check-language: could not read {path}: {exc}", file=sys.stderr)
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _TURKISH_RX.search(line):
                hits.append(f"{path}:{lineno}: {line.rstrip()}")

    if hits:
        print("check-language: Turkish characters found in src/gui/:", file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        print(
            "\nv1.0.1-ux shipped English-only UI. Translate the line, or",
            "move the message out of src/gui/ if it's truly author-facing.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
