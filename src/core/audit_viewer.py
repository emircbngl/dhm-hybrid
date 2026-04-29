"""Audit log reader + filter — v2.0.7 sprint, T5.

The dialog UI in :mod:`ui2.dialogs` is a thin DPG wrapper around
this module. Filter / format logic lives here so it's pure-data
and testable without DPG.

Sven's compliance ask (Lindqvist 2026-04-27):

    > "Audit log şu an her run için ayrı kayıt. 3000 run = 3000
    > satır. Bizim QA hangi parametrelerin tüm session boyunca
    > aynı kaldığını istiyor — diff view yok."

The viewer can't synthesise that diff yet (v2.0.8 backlog), but it
*does* solve the prerequisite: a usable browse + filter UI on top
of the JSONL log instead of `cat ~/.dhm-reconstruction/audit/*.jsonl
| jq`.

Capabilities
------------
* Glob across multiple daily files (date range).
* Filter by ``operator``, ``action``, free-text substring.
* Stable iteration order (newest first) — newest is what the lab
  wants to see when they open the viewer.
* Streaming-friendly read (one JSONL line at a time) so a 12-hour
  session's 3000 audit lines don't stall the dialog.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence


_LOG = logging.getLogger(__name__)

DEFAULT_AUDIT_DIR = Path.home() / ".dhm-reconstruction" / "audit"


@dataclass
class AuditEntry:
    """One parsed JSONL record. Free-form ``raw`` keeps the
    original dict around so the viewer can show full details on
    expand without re-parsing."""
    timestamp: str
    action: str
    operator: str
    user: str
    app_version: str
    git_sha: str
    params: dict = field(default_factory=dict)
    result_summary: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "AuditEntry":
        return cls(
            timestamp=str(raw.get("timestamp", "")),
            action=str(raw.get("action", "")),
            operator=str(raw.get("operator") or raw.get("user") or ""),
            user=str(raw.get("user", "")),
            app_version=str(raw.get("app_version", "")),
            git_sha=str(raw.get("git_sha", "")),
            params=dict(raw.get("params") or {}),
            result_summary=dict(raw.get("result_summary") or {}),
            raw=dict(raw),
        )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def list_log_files(directory: Optional[Path] = None) -> List[Path]:
    """Return every ``YYYY-MM-DD.jsonl`` file in ``directory``,
    newest first.

    Missing dir → empty list (lab freshly installed). Non-jsonl
    files ignored. Symlinks followed."""
    d = Path(directory) if directory else DEFAULT_AUDIT_DIR
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir()
             if p.is_file() and p.suffix == ".jsonl"]
    return sorted(files, reverse=True)


def iter_entries(directory: Optional[Path] = None,
                 *,
                 since: Optional[date] = None,
                 until: Optional[date] = None,
                 ) -> Iterator[AuditEntry]:
    """Yield every entry across daily files, newest file first,
    newest line within each file first.

    Parameters
    ----------
    directory
        Audit dir; defaults to ``~/.dhm-reconstruction/audit``.
    since, until
        Inclusive date filters on the filename's ``YYYY-MM-DD``
        component. ``None`` means no bound.
    """
    for path in list_log_files(directory):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            file_date = None
        if since and file_date and file_date < since:
            continue
        if until and file_date and file_date > until:
            continue
        # Read in reverse so newest line within a file emerges first.
        try:
            with path.open(encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            _LOG.warning("audit: skipping %s (%s)", path, exc)
            continue
        for raw_line in reversed(lines):
            line = raw_line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue
            yield AuditEntry.from_dict(d)


def read_entries(directory: Optional[Path] = None,
                 *,
                 since: Optional[date] = None,
                 until: Optional[date] = None,
                 limit: Optional[int] = None,
                 ) -> List[AuditEntry]:
    """Materialise :func:`iter_entries` into a list, with optional
    cap. Defaults: every entry, newest first."""
    out: List[AuditEntry] = []
    for entry in iter_entries(directory, since=since, until=until):
        out.append(entry)
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditFilter:
    """Composable filter applied after reading.

    Empty / None fields = no filter. ``query`` is a free-text
    case-insensitive substring match against the JSON-encoded
    record (params, result_summary, etc.) — so the user can search
    "z_mm=12" or "QPI" and find what they meant."""
    operators: Optional[Sequence[str]] = None
    actions: Optional[Sequence[str]] = None
    query: Optional[str] = None

    def matches(self, e: AuditEntry) -> bool:
        if self.operators:
            ops = {o.lower() for o in self.operators}
            if (e.operator or e.user).lower() not in ops:
                return False
        if self.actions:
            if e.action not in self.actions:
                return False
        if self.query:
            q = self.query.lower()
            blob = json.dumps(e.raw, default=str).lower()
            if q not in blob:
                return False
        return True


def apply_filter(entries: Iterable[AuditEntry],
                 filt: AuditFilter) -> List[AuditEntry]:
    return [e for e in entries if filt.matches(e)]


# ---------------------------------------------------------------------------
# Aggregations — for filter dropdown population
# ---------------------------------------------------------------------------

def known_operators(directory: Optional[Path] = None) -> List[str]:
    """All distinct operators seen across the log, sorted."""
    out = set()
    for e in iter_entries(directory):
        op = (e.operator or e.user).strip()
        if op:
            out.add(op)
    return sorted(out)


def known_actions(directory: Optional[Path] = None) -> List[str]:
    """All distinct actions seen across the log, sorted."""
    out = set()
    for e in iter_entries(directory):
        if e.action:
            out.add(e.action)
    return sorted(out)


__all__ = [
    "DEFAULT_AUDIT_DIR",
    "AuditEntry",
    "AuditFilter",
    "list_log_files",
    "iter_entries",
    "read_entries",
    "apply_filter",
    "known_operators",
    "known_actions",
]
