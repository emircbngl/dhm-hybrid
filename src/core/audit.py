"""Append-only JSONL audit log.

Each day is a separate file at
``~/.dhm-reconstruction/audit/YYYY-MM-DD.jsonl``. One JSON object per line.
Records include timestamp (ISO8601 UTC), action, app version, git SHA, the
params dict passed by the caller, and an optional result_summary.

The log is intentionally dumb and crash-safe: the file is opened in append
mode and closed after every write, so a crash mid-reconstruction can at worst
lose a single in-flight record. All exceptions inside ``record`` are caught
and logged via ``logging``; audit failure must never crash science code.
"""
from __future__ import annotations

import getpass
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from __version__ import __version__, __git_sha__  # type: ignore
except Exception:  # pragma: no cover - fallback for tests / direct runs
    __version__ = "unknown"
    __git_sha__ = "unknown"


_LOG = logging.getLogger(__name__)

_AUDIT_DIR = Path.home() / ".dhm-reconstruction" / "audit"


class AuditLog:
    """Singleton-style append-only JSONL audit log.

    Use :func:`get_audit_log` to obtain the cached instance rather than
    constructing this directly.
    """

    def __init__(self, directory: Optional[Path] = None) -> None:
        self._dir = Path(directory) if directory is not None else _AUDIT_DIR
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - permission edge cases
            _LOG.warning("audit: could not create directory %s: %s", self._dir, exc)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def log_path(self) -> Path:
        """Return today's log file path (YYYY-MM-DD.jsonl)."""
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._dir / f"{day}.jsonl"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def record(
        self,
        action: str,
        params: dict,
        result_summary: Optional[dict] = None,
    ) -> None:
        """Append one record to today's log.

        Any exception (disk full, permission, JSON-unserializable value that
        even ``default=str`` cannot handle, etc.) is caught and logged at
        WARNING level.  A failed audit must never propagate.
        """
        try:
            record: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "app_version": __version__,
                "git_sha": __git_sha__,
                "params": params if params is not None else {},
            }
            if result_summary is not None:
                record["result_summary"] = result_summary
            # v2.0.7: user_profile.current_user() honours DHM_USER env
            # override + sanitises the result. We import lazily so the
            # audit log keeps working when running standalone scripts
            # that don't pull in core.user_profile (e.g. CLI smoke
            # tests). Both 'user' and 'operator' carry the same value
            # — 'user' is the v1 field name (back-compat for existing
            # log readers); 'operator' is the v2.0.7 idiom the new
            # audit viewer prefers.
            try:
                from core import user_profile  # noqa: WPS433
                op = user_profile.current_user()
                record["user"] = op
                record["operator"] = op
            except Exception:
                try:
                    record["user"] = getpass.getuser()
                except Exception:
                    pass

            line = json.dumps(record, default=str, ensure_ascii=False)
            path = self.log_path()
            # Ensure directory still exists (user could have wiped it between calls).
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append-mode open + close per write: O_APPEND writes are atomic for
            # small lines on POSIX, and closing flushes buffers -> crash-safe.
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            _LOG.warning("audit: failed to record action=%r: %s", action, exc)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def entries_today(self, limit: int = 100) -> list[dict]:
        """Return up to ``limit`` most-recent entries from today's log.

        Silently returns an empty list if the log does not yet exist or cannot
        be parsed.  Malformed lines are skipped rather than raising.
        """
        path = self.log_path()
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as exc:
            _LOG.warning("audit: could not read %s: %s", path, exc)
            return []

        # Take the last `limit` lines, newest last.
        if limit > 0:
            lines = lines[-limit:]

        out: list[dict] = []
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_SINGLETON: Optional[AuditLog] = None


def get_audit_log() -> AuditLog:
    """Return the process-wide :class:`AuditLog` instance, creating it once."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = AuditLog()
    return _SINGLETON
