"""Multi-user profile resolution — v2.0.7 sprint, T3.

Up to v2.0.6 the app stored UI state at a single shared path
``~/.dhm-reconstruction/ui2_state.json``. That breaks the moment
two operators share a workstation: Erik's defaults overwrite
Karin's, and the audit trail can't say who clicked Reconstruct.

This module gives each operator their own folder under
``~/.dhm-reconstruction/users/<username>/`` while preserving the
single-user upgrade path: existing ``ui2_state.json`` is migrated
to ``users/<default_user>/ui2_state.json`` on first read.

Public API
----------
* :func:`current_user` — username for this process (env > OS > "default").
* :func:`user_state_dir` — folder for one user.
* :func:`user_state_path` — ``ui2_state.json`` for one user.
* :func:`migrate_legacy_state_if_needed` — one-shot migration.
* :func:`list_known_users` — directory listing for an audit viewer.

Design rules
------------
* No DPG / no Qt. Pure pathlib + filesystem. Tests can override
  the root via :func:`set_root_dir`.
* Username sanitisation is deterministic — only ``[a-z0-9_-]``,
  others collapse to ``_``. Means a Windows user "Karin Berg"
  ends up at ``users/karin_berg/``.
* ``current_user()`` honours ``DHM_USER`` env var first. Useful
  for tests, CI, headless CLI runners (Sven's Linux box).
"""
from __future__ import annotations

import getpass
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional


_LOG = logging.getLogger(__name__)

# Root override is mutable so tests can sandbox to ``tmp_path``.
_ROOT_OVERRIDE: Optional[Path] = None

DEFAULT_USER = "default"
ENV_VAR = "DHM_USER"
ENV_ROOT = "DHM_ROOT"


def set_root_dir(path: Optional[Path]) -> None:
    """Test seam — point the user-profile system at a sandbox dir.

    Call with ``None`` to clear and revert to ``~/.dhm-reconstruction``.
    """
    global _ROOT_OVERRIDE
    _ROOT_OVERRIDE = Path(path) if path is not None else None


def root_dir() -> Path:
    """Resolve the per-host root.

    Priority: explicit override (test) > ``DHM_ROOT`` env var >
    ``~/.dhm-reconstruction``. The directory is created on first
    access — every caller below assumes it exists.
    """
    if _ROOT_OVERRIDE is not None:
        root = _ROOT_OVERRIDE
    elif os.environ.get(ENV_ROOT):
        root = Path(os.environ[ENV_ROOT])
    else:
        root = Path.home() / ".dhm-reconstruction"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Username
# ---------------------------------------------------------------------------

_SAFE_RE = re.compile(r"[^a-z0-9_-]+")


def sanitise_username(name: str) -> str:
    """Normalise to ``[a-z0-9_-]``. Empty / whitespace-only → DEFAULT.

    Names that consist *only* of separator chars (``_``, ``-``) are
    treated as empty — a folder called ``---`` is technically valid
    on POSIX but useless as an operator label. Collapse to DEFAULT
    so the audit log doesn't end up sorting on phantom usernames.
    """
    if not name:
        return DEFAULT_USER
    n = name.strip().lower()
    n = _SAFE_RE.sub("_", n)
    n = n.strip("_-") or DEFAULT_USER
    return n


def current_user() -> str:
    """Determine which operator is using this process.

    Resolution order:

    1. ``DHM_USER`` environment variable — explicit override the lab
       sets when running under shared logins or in CI batch jobs.
    2. ``getpass.getuser()`` — OS username. The classic single-user
       Mac dev path.
    3. ``DEFAULT_USER`` — fallback when both fail (sandboxes).

    Returns
    -------
    str
        Sanitised username, safe to use as a directory name.
    """
    raw = os.environ.get(ENV_VAR)
    if not raw:
        try:
            raw = getpass.getuser()
        except Exception:
            raw = ""
    return sanitise_username(raw or DEFAULT_USER)


# ---------------------------------------------------------------------------
# Per-user paths
# ---------------------------------------------------------------------------

def user_state_dir(username: Optional[str] = None) -> Path:
    """Folder for a given user — ``<root>/users/<sanitised>``.

    Created on first access so callers can directly write into it.
    """
    name = sanitise_username(username) if username else current_user()
    d = root_dir() / "users" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_state_path(username: Optional[str] = None) -> Path:
    """``ui2_state.json`` for a given user — the v2.0.7 successor of
    the single ``~/.dhm-reconstruction/ui2_state.json``."""
    return user_state_dir(username) / "ui2_state.json"


def list_known_users() -> List[str]:
    """All usernames with an existing per-user folder. Used by the
    audit viewer to populate the operator filter dropdown.

    Returns sorted list — empty when nobody has saved yet."""
    users_root = root_dir() / "users"
    if not users_root.is_dir():
        return []
    return sorted(
        d.name for d in users_root.iterdir() if d.is_dir()
    )


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------

def migrate_legacy_state_if_needed(*,
                                   target_user: Optional[str] = None,
                                   ) -> Optional[Path]:
    """Copy the legacy single-user state file to the per-user
    location on first run.

    Logic:

    * Source: ``<root>/ui2_state.json``
    * Target: ``<root>/users/<target_user>/ui2_state.json``
    * If source missing → no-op, returns ``None``.
    * If target already exists → leave both, return ``None``
      (operator already migrated, don't clobber).
    * Otherwise: copy source → target, remove source, return target.

    Removing the legacy file is intentional: keeping it around would
    invite two stores to drift. The migration is logged in the audit
    log via the calling layer (state_store).

    Returns
    -------
    Optional[Path]
        The new target path if migration ran, else ``None``.
    """
    target = (sanitise_username(target_user)
              if target_user else current_user())
    legacy = root_dir() / "ui2_state.json"
    new = user_state_path(target)

    if not legacy.exists():
        return None
    if new.exists():
        # Both already exist — operator likely cleared per-user but
        # legacy lingered. Do nothing; the operator can clean up.
        _LOG.info(
            "user_profile: legacy %s and per-user %s both exist; "
            "leaving legacy alone.",
            legacy, new,
        )
        return None
    try:
        shutil.copy2(legacy, new)
        legacy.unlink()
        _LOG.info(
            "user_profile: migrated legacy state %s → %s",
            legacy, new,
        )
        return new
    except Exception as exc:  # pragma: no cover - permission edge cases
        _LOG.warning(
            "user_profile: failed to migrate legacy state: %s",
            exc,
        )
        return None


__all__ = [
    "DEFAULT_USER",
    "ENV_VAR",
    "ENV_ROOT",
    "set_root_dir",
    "root_dir",
    "sanitise_username",
    "current_user",
    "user_state_dir",
    "user_state_path",
    "list_known_users",
    "migrate_legacy_state_if_needed",
]
