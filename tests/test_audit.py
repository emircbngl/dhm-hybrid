"""Audit log — JSONL append, unserializable input safety, entry limit."""
from __future__ import annotations

import json

from core.audit import AuditLog


def test_audit_record_writes_jsonl(tmp_audit_dir):
    log = AuditLog(directory=tmp_audit_dir)
    log.record("unit_test", {"param": 1}, result_summary={"ok": True})
    path = log.log_path()
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    for key in ("timestamp", "action", "app_version", "git_sha", "params"):
        assert key in entry
    assert entry["action"] == "unit_test"
    assert entry["params"] == {"param": 1}
    assert entry["result_summary"] == {"ok": True}


def test_audit_never_raises_on_bad_input(tmp_audit_dir):
    log = AuditLog(directory=tmp_audit_dir)
    # Lambdas are not JSON-serializable by default, but default=str should stringify them.
    log.record("bad", {"fn": lambda x: x}, result_summary=None)
    path = log.log_path()
    assert path.exists()
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    entry = json.loads(line)
    # The lambda should have been converted to its repr string.
    assert "<function" in entry["params"]["fn"]


def test_audit_entries_today_limit_respected(tmp_audit_dir):
    log = AuditLog(directory=tmp_audit_dir)
    for i in range(5):
        log.record("probe", {"i": i})
    got = log.entries_today(limit=3)
    assert len(got) == 3
    # Newest last → our last 3 writes are i=2,3,4.
    assert [e["params"]["i"] for e in got] == [2, 3, 4]
