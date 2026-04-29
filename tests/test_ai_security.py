"""Guardrails: numeric clamp, path validation, audit redaction.

Pure-stdlib tests — no Qt, no HTTP. The ``validate_path`` cases use
``tmp_path`` to build legitimate files and a temporary HOME so the
restrict-to-home check has something predictable to compare against.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.ai.security import (
    NUMERIC_BOUNDS,
    SecurityError,
    clamp,
    redact_for_llm,
    validate_path,
)


# ----- clamp -----------------------------------------------------------------


def test_clamp_passes_in_range_value():
    assert clamp("z_mm", 10.0) == 10.0


def test_clamp_passes_unknown_name_through():
    # Unknown numeric → identity. JSON schema is the first gate.
    assert clamp("unknown_param", 1234.0) == 1234.0


def test_clamp_rejects_above_upper():
    with pytest.raises(SecurityError) as exc:
        clamp("z_mm", 999.0)
    assert "out of range" in str(exc.value)


def test_clamp_rejects_below_lower():
    with pytest.raises(SecurityError):
        clamp("wavelength_nm", 50.0)


def test_clamp_rejects_nan():
    with pytest.raises(SecurityError) as exc:
        clamp("z_mm", float("nan"))
    assert "NaN" in str(exc.value)


def test_clamp_table_covers_each_known_param():
    for name, (lo, hi) in NUMERIC_BOUNDS.items():
        # In-range value at the midpoint must pass.
        mid = (lo + hi) / 2.0
        assert clamp(name, mid) == pytest.approx(mid)


# ----- validate_path ---------------------------------------------------------


def _make_tif(tmp_path: Path, name: str = "h.tif", size: int = 1024) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x00" * size)
    return p


def test_validate_path_accepts_tif_in_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = _make_tif(tmp_path)
    out = validate_path(str(p), restrict_to_home=True)
    assert out == p.resolve()


def test_validate_path_rejects_outside_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    p = elsewhere / "h.tif"
    p.write_bytes(b"\x00" * 16)
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(SecurityError) as exc:
        validate_path(str(p), restrict_to_home=True)
    assert "outside the home directory" in str(exc.value)


def test_validate_path_allows_outside_home_when_unrestricted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    p = elsewhere / "h.tif"
    p.write_bytes(b"\x00" * 16)
    monkeypatch.setenv("HOME", str(home))
    out = validate_path(str(p), restrict_to_home=False)
    assert out == p.resolve()


def test_validate_path_rejects_traversal_outside_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    sibling = tmp_path / "sneaky.tif"
    sibling.write_bytes(b"\x00")
    monkeypatch.setenv("HOME", str(home))
    raw = str(home / ".." / "sneaky.tif")
    with pytest.raises(SecurityError):
        validate_path(raw, restrict_to_home=True)


def test_validate_path_rejects_bad_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    bad = tmp_path / "secret.txt"
    bad.write_text("hello")
    with pytest.raises(SecurityError) as exc:
        validate_path(str(bad), restrict_to_home=True)
    assert "not allowed" in str(exc.value)


def test_validate_path_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SecurityError) as exc:
        validate_path(str(tmp_path / "nope.tif"), restrict_to_home=True)
    assert "does not exist" in str(exc.value)


def test_validate_path_rejects_huge_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = _make_tif(tmp_path, "big.tif")
    with pytest.raises(SecurityError):
        validate_path(str(p), restrict_to_home=True, max_size_bytes=8)


def test_validate_path_rejects_empty_string():
    with pytest.raises(SecurityError):
        validate_path("", restrict_to_home=False)


def test_validate_path_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _make_tif(tmp_path, "h.tif")
    out = validate_path("~/h.tif", restrict_to_home=True)
    assert out.name == "h.tif"


# ----- redact_for_llm --------------------------------------------------------


def test_redact_replaces_operator():
    record = {"action": "x", "operator": "alice", "user": "alice"}
    out = redact_for_llm(record)
    assert out["operator"] == "<user>"
    assert out["user"] == "<user>"


def test_redact_trims_absolute_paths_in_params():
    record = {
        "action": "load",
        "params": {"path": "/Users/alice/secret/sample.tif",
                   "shape": [1024, 1024]},
    }
    out = redact_for_llm(record)
    assert out["params"]["path"] == "sample.tif"
    assert out["params"]["shape"] == [1024, 1024]


def test_redact_keeps_non_path_strings():
    record = {"params": {"label": "ROI A", "metric": "PHASE_VARIANCE"}}
    out = redact_for_llm(record)
    assert out["params"]["label"] == "ROI A"
    assert out["params"]["metric"] == "PHASE_VARIANCE"


def test_redact_recurses_into_nested_dict():
    record = {"params": {"nested": {"path": "/abs/foo.tif"}}}
    out = redact_for_llm(record)
    assert out["params"]["nested"]["path"] == "foo.tif"


def test_redact_does_not_mutate_input():
    record = {"operator": "alice"}
    snapshot = dict(record)
    redact_for_llm(record)
    assert record == snapshot
