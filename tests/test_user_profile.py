"""Multi-user profile resolution + legacy state migration (v2.0.7,
T3 of the time-lapse foundation sprint).

Coverage:

* :func:`current_user` honours ``DHM_USER`` env, falls back to
  ``getpass.getuser`` then to ``DEFAULT_USER``.
* :func:`sanitise_username` is deterministic + safe for filesystem.
* Per-user state dir is created on first access.
* Legacy single-user ``ui2_state.json`` migrates to
  ``users/<default_user>/`` on first read and the legacy file is
  removed (no drift between two stores).
* Audit log embeds the resolved operator under ``user`` AND
  ``operator`` (back-compat with v1 reader + v2.0.7 idiom).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import user_profile  # noqa: E402
from core.audit import AuditLog  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — sandbox the user-profile root for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def sandbox_root(tmp_path, monkeypatch):
    """Each test gets its own private root + cleared env."""
    monkeypatch.delenv(user_profile.ENV_VAR, raising=False)
    monkeypatch.delenv(user_profile.ENV_ROOT, raising=False)
    user_profile.set_root_dir(tmp_path)
    yield tmp_path
    user_profile.set_root_dir(None)


# ---------------------------------------------------------------------------
# sanitise_username
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("emir",          "emir"),
    ("Emir",          "emir"),
    ("Emir Cebnegil", "emir_cebnegil"),
    ("Karin Berg",    "karin_berg"),
    ("user@host",     "user_host"),
    ("__weird__",     "weird"),
    ("",              "default"),
    ("   ",           "default"),
    ("---",           "default"),
])
def test_sanitise_username_normalises_to_safe(raw, expected):
    assert user_profile.sanitise_username(raw) == expected


# ---------------------------------------------------------------------------
# current_user
# ---------------------------------------------------------------------------

def test_current_user_honours_dhm_user_env(monkeypatch):
    monkeypatch.setenv(user_profile.ENV_VAR, "Karin Berg")
    assert user_profile.current_user() == "karin_berg"


def test_current_user_falls_back_to_getpass(monkeypatch):
    """When DHM_USER is unset, current_user calls getpass.getuser
    and sanitises that. We don't assert the exact name (tests run
    under unknown OS users) — only that it resolves to *something*
    parseable."""
    monkeypatch.delenv(user_profile.ENV_VAR, raising=False)
    name = user_profile.current_user()
    assert name
    # Sanitised + non-empty.
    assert name == user_profile.sanitise_username(name)


def test_current_user_falls_back_to_default_when_getpass_blows_up(
    monkeypatch,
):
    monkeypatch.delenv(user_profile.ENV_VAR, raising=False)
    monkeypatch.setattr(
        user_profile.getpass, "getuser",
        lambda: (_ for _ in ()).throw(OSError("no user")),
    )
    assert user_profile.current_user() == user_profile.DEFAULT_USER


# ---------------------------------------------------------------------------
# Per-user paths
# ---------------------------------------------------------------------------

def test_user_state_dir_creates_subfolder(sandbox_root, monkeypatch):
    monkeypatch.setenv(user_profile.ENV_VAR, "karin")
    d = user_profile.user_state_dir()
    assert d == sandbox_root / "users" / "karin"
    assert d.is_dir()


def test_user_state_path_targets_ui2_state_json(sandbox_root, monkeypatch):
    monkeypatch.setenv(user_profile.ENV_VAR, "erik")
    p = user_profile.user_state_path()
    assert p == sandbox_root / "users" / "erik" / "ui2_state.json"


def test_user_state_path_isolates_users(sandbox_root, monkeypatch):
    """Two users hitting the same root see different files."""
    monkeypatch.setenv(user_profile.ENV_VAR, "karin")
    a = user_profile.user_state_path()
    monkeypatch.setenv(user_profile.ENV_VAR, "erik")
    b = user_profile.user_state_path()
    assert a != b
    assert a.parent.name == "karin"
    assert b.parent.name == "erik"


# ---------------------------------------------------------------------------
# list_known_users
# ---------------------------------------------------------------------------

def test_list_known_users_returns_sorted_directory_names(
    sandbox_root, monkeypatch,
):
    """Folders under <root>/users/ are the truth source for known
    operators. Sorted output keeps the audit viewer dropdown
    deterministic."""
    for name in ["erik", "anna", "karin"]:
        (sandbox_root / "users" / name).mkdir(parents=True)
    # Non-folder entry should be ignored.
    (sandbox_root / "users").mkdir(exist_ok=True)
    (sandbox_root / "users" / "ignore_me.txt").write_bytes(b"x")
    assert user_profile.list_known_users() == ["anna", "erik", "karin"]


def test_list_known_users_empty_when_no_users(sandbox_root):
    """Fresh install — no folders. Empty list, not error."""
    assert user_profile.list_known_users() == []


# ---------------------------------------------------------------------------
# Legacy state migration
# ---------------------------------------------------------------------------

def test_migrate_legacy_no_op_when_legacy_missing(sandbox_root):
    """No legacy file → migrate is a noop, returns None."""
    res = user_profile.migrate_legacy_state_if_needed()
    assert res is None


def test_migrate_legacy_copies_to_user_dir_and_unlinks(
    sandbox_root, monkeypatch,
):
    """Legacy file → moved to users/<default>/ui2_state.json.
    Source is removed so two stores can't drift."""
    monkeypatch.setenv(user_profile.ENV_VAR, "karin")
    legacy = sandbox_root / "ui2_state.json"
    legacy.write_text(json.dumps({"schema_version": 11}))
    res = user_profile.migrate_legacy_state_if_needed()
    assert res == sandbox_root / "users" / "karin" / "ui2_state.json"
    assert res.exists()
    assert not legacy.exists()
    # Content preserved.
    assert json.loads(res.read_text())["schema_version"] == 11


def test_migrate_legacy_preserves_target_when_already_present(
    sandbox_root, monkeypatch,
):
    """If both legacy and per-user exist, keep both untouched.
    This is the operator-already-migrated case + a stray legacy
    file. Don't clobber recent work."""
    monkeypatch.setenv(user_profile.ENV_VAR, "anna")
    legacy = sandbox_root / "ui2_state.json"
    legacy.write_text('{"schema_version": 1}')
    new_dir = sandbox_root / "users" / "anna"
    new_dir.mkdir(parents=True)
    new_path = new_dir / "ui2_state.json"
    new_path.write_text('{"schema_version": 11, "real": "current"}')
    res = user_profile.migrate_legacy_state_if_needed()
    assert res is None
    # Both still present; per-user content untouched.
    assert legacy.exists()
    assert json.loads(new_path.read_text())["real"] == "current"


def test_migrate_to_explicit_target_user_overrides_env(
    sandbox_root, monkeypatch,
):
    monkeypatch.setenv(user_profile.ENV_VAR, "anna")
    legacy = sandbox_root / "ui2_state.json"
    legacy.write_text('{"schema_version": 11}')
    res = user_profile.migrate_legacy_state_if_needed(
        target_user="erik",
    )
    assert res == sandbox_root / "users" / "erik" / "ui2_state.json"


# ---------------------------------------------------------------------------
# Audit log carries operator field
# ---------------------------------------------------------------------------

def test_audit_log_records_operator_from_dhm_user_env(
    sandbox_root, monkeypatch, tmp_path,
):
    """Audit lines under DHM_USER=karin must carry 'karin' (sanitised)
    in both 'user' (back-compat) and 'operator' (v2.0.7 idiom).
    No leakage from the OS username — env wins."""
    monkeypatch.setenv(user_profile.ENV_VAR, "Karin Berg")
    audit = AuditLog(directory=tmp_path / "audit")
    audit.record(action="reconstruct", params={"z_mm": 12.0})
    entries = audit.entries_today()
    assert len(entries) == 1
    rec = entries[0]
    assert rec["user"] == "karin_berg"
    assert rec["operator"] == "karin_berg"


def test_audit_log_skips_operator_when_user_profile_unavailable(
    sandbox_root, monkeypatch, tmp_path,
):
    """If user_profile import fails (e.g. extracted standalone
    audit util), we still want a 'user' field — fall back to
    getpass. 'operator' may be missing in that branch."""
    # Force the user_profile import to raise.
    import builtins
    real_import = builtins.__import__

    def _raise(name, *a, **kw):
        if name.startswith("core.user_profile") or name == "core":
            raise ImportError("simulated")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _raise)
    audit = AuditLog(directory=tmp_path / "audit")
    audit.record(action="reconstruct", params={})
    entries = audit.entries_today()
    assert len(entries) == 1
    # 'user' falls back to getpass — should be present.
    assert "user" in entries[0]
    # 'operator' may be absent in fallback branch — that's OK.
