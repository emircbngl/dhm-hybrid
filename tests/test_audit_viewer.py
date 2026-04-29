"""Audit log viewer reader + filter (v2.0.7, T5).

The DPG dialog is a thin wrapper around :mod:`core.audit_viewer`.
This file tests the data layer that the dialog depends on — read,
parse, filter, aggregate. Headless, no DPG.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.audit_viewer import (  # noqa: E402
    AuditEntry,
    AuditFilter,
    apply_filter,
    iter_entries,
    known_actions,
    known_operators,
    list_log_files,
    read_entries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_log(dir_path: Path, fname: str, lines: list[dict]) -> Path:
    """Write a daily JSONL file. ``lines`` order = file order
    (oldest at top, newest at bottom — matches AuditLog.record)."""
    p = dir_path / fname
    with p.open("w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")
    return p


def _make_record(*, action: str = "reconstruct",
                 operator: str = "karin",
                 timestamp: str = "2026-04-25T10:00:00+00:00",
                 z_mm: float = 12.0,
                 **extra) -> dict:
    rec = {
        "timestamp": timestamp,
        "action": action,
        "operator": operator,
        "user": operator,
        "app_version": "2.0.7",
        "git_sha": "abc1234",
        "params": {"z_mm": z_mm},
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# list_log_files
# ---------------------------------------------------------------------------

def test_list_log_files_returns_newest_first(tmp_path):
    """Sorted reverse so the dialog opens on today's events."""
    _write_log(tmp_path, "2026-04-23.jsonl", [_make_record()])
    _write_log(tmp_path, "2026-04-25.jsonl", [_make_record()])
    _write_log(tmp_path, "2026-04-24.jsonl", [_make_record()])
    files = list_log_files(tmp_path)
    assert [p.stem for p in files] == ["2026-04-25", "2026-04-24",
                                       "2026-04-23"]


def test_list_log_files_ignores_non_jsonl(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl", [_make_record()])
    (tmp_path / "README.txt").write_text("not an audit log")
    files = list_log_files(tmp_path)
    assert len(files) == 1


def test_list_log_files_handles_missing_dir(tmp_path):
    assert list_log_files(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# iter_entries / read_entries
# ---------------------------------------------------------------------------

def test_iter_entries_yields_newest_within_file_first(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl", [
        _make_record(timestamp="2026-04-25T08:00:00+00:00"),
        _make_record(timestamp="2026-04-25T09:00:00+00:00"),
        _make_record(timestamp="2026-04-25T10:00:00+00:00"),
    ])
    entries = read_entries(tmp_path)
    # File order = oldest top, newest bottom; iter reverses → newest first.
    timestamps = [e.timestamp for e in entries]
    assert timestamps == ["2026-04-25T10:00:00+00:00",
                          "2026-04-25T09:00:00+00:00",
                          "2026-04-25T08:00:00+00:00"]


def test_iter_entries_yields_newest_file_first(tmp_path):
    """Across multiple daily files, iter visits newest file first."""
    _write_log(tmp_path, "2026-04-23.jsonl",
               [_make_record(timestamp="2026-04-23T10:00:00+00:00")])
    _write_log(tmp_path, "2026-04-25.jsonl",
               [_make_record(timestamp="2026-04-25T08:00:00+00:00")])
    entries = read_entries(tmp_path)
    assert entries[0].timestamp.startswith("2026-04-25")
    assert entries[-1].timestamp.startswith("2026-04-23")


def test_iter_entries_skips_malformed_lines(tmp_path):
    """A truncated/corrupted line shouldn't kill the read — it's
    skipped, the rest of the file proceeds. Audit log readers must
    be charitable; the writer is dumb-append and a power-cut can
    leave half-lines."""
    p = tmp_path / "2026-04-25.jsonl"
    p.write_text("\n".join([
        json.dumps(_make_record()),
        "not valid {json:",          # malformed
        "",                            # blank
        json.dumps(_make_record(z_mm=15.0)),
    ]), encoding="utf-8")
    entries = read_entries(tmp_path)
    assert len(entries) == 2


def test_read_entries_limit_caps_results(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl",
               [_make_record(z_mm=float(i)) for i in range(50)])
    out = read_entries(tmp_path, limit=10)
    assert len(out) == 10


def test_iter_entries_filters_by_date_range(tmp_path):
    _write_log(tmp_path, "2026-04-22.jsonl", [_make_record()])
    _write_log(tmp_path, "2026-04-25.jsonl", [_make_record()])
    _write_log(tmp_path, "2026-04-28.jsonl", [_make_record()])
    out = list(iter_entries(
        tmp_path,
        since=date(2026, 4, 25),
        until=date(2026, 4, 27),
    ))
    # Only the 2026-04-25 file matches.
    assert len(out) == 1


# ---------------------------------------------------------------------------
# AuditEntry.from_dict tolerance
# ---------------------------------------------------------------------------

def test_from_dict_tolerates_missing_optional_fields():
    e = AuditEntry.from_dict({"action": "x"})
    assert e.action == "x"
    assert e.timestamp == ""
    assert e.params == {}


def test_operator_falls_back_to_user_field():
    """Old v1 records have only 'user'; viewer should still surface
    them as the operator label."""
    e = AuditEntry.from_dict({"user": "old_emir"})
    assert e.operator == "old_emir"


# ---------------------------------------------------------------------------
# AuditFilter
# ---------------------------------------------------------------------------

def test_filter_by_operator_substring(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl", [
        _make_record(operator="karin"),
        _make_record(operator="erik"),
        _make_record(operator="karin"),
    ])
    entries = read_entries(tmp_path)
    filt = AuditFilter(operators=["karin"])
    matched = apply_filter(entries, filt)
    assert len(matched) == 2
    assert all(e.operator == "karin" for e in matched)


def test_filter_by_action_set(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl", [
        _make_record(action="reconstruct"),
        _make_record(action="qpi"),
        _make_record(action="autofocus"),
    ])
    entries = read_entries(tmp_path)
    filt = AuditFilter(actions=["reconstruct", "qpi"])
    matched = apply_filter(entries, filt)
    assert len(matched) == 2


def test_filter_query_substring_searches_full_record(tmp_path):
    """Free-text search hits anywhere in the JSON: params,
    result_summary, action, etc. Operator-friendly catch-all."""
    _write_log(tmp_path, "2026-04-25.jsonl", [
        _make_record(action="reconstruct", z_mm=12.4),
        _make_record(action="qpi", z_mm=99.0),
        _make_record(action="reconstruct", z_mm=12.4,
                     result_summary={"opd_range_nm": 412.5}),
    ])
    entries = read_entries(tmp_path)
    # Looking for the OPD record by its numeric value substring.
    matched = apply_filter(entries, AuditFilter(query="412.5"))
    assert len(matched) == 1
    matched = apply_filter(entries, AuditFilter(query="qpi"))
    assert len(matched) == 1
    # Substring "12.4" hits two records that have z_mm=12.4.
    matched = apply_filter(entries, AuditFilter(query="12.4"))
    assert len(matched) == 2


def test_combined_filters_intersect(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl", [
        _make_record(operator="karin", action="reconstruct"),
        _make_record(operator="erik",  action="reconstruct"),
        _make_record(operator="karin", action="qpi"),
    ])
    entries = read_entries(tmp_path)
    filt = AuditFilter(operators=["karin"], actions=["reconstruct"])
    matched = apply_filter(entries, filt)
    assert len(matched) == 1
    assert matched[0].operator == "karin"
    assert matched[0].action == "reconstruct"


def test_empty_filter_passes_everything(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl", [
        _make_record() for _ in range(5)
    ])
    entries = read_entries(tmp_path)
    filt = AuditFilter()
    assert len(apply_filter(entries, filt)) == 5


# ---------------------------------------------------------------------------
# known_operators / known_actions
# ---------------------------------------------------------------------------

def test_known_operators_returns_sorted_set(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl", [
        _make_record(operator="karin"),
        _make_record(operator="erik"),
        _make_record(operator="anna"),
        _make_record(operator="karin"),  # duplicate
    ])
    assert known_operators(tmp_path) == ["anna", "erik", "karin"]


def test_known_actions_returns_sorted_set(tmp_path):
    _write_log(tmp_path, "2026-04-25.jsonl", [
        _make_record(action="reconstruct"),
        _make_record(action="autofocus"),
        _make_record(action="qpi"),
        _make_record(action="reconstruct"),
    ])
    assert known_actions(tmp_path) == [
        "autofocus", "qpi", "reconstruct",
    ]


def test_known_operators_skips_blank_strings(tmp_path):
    """A record without operator/user (very old log) shouldn't
    add a phantom blank operator to the dropdown."""
    p = tmp_path / "2026-04-25.jsonl"
    p.write_text("\n".join([
        json.dumps({"action": "x"}),  # no operator/user
        json.dumps(_make_record(operator="karin")),
    ]), encoding="utf-8")
    assert known_operators(tmp_path) == ["karin"]
