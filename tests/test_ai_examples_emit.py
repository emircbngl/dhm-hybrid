"""Training-data emission script: shape, count, registry sync.

The ``scripts/ai_training_examples.py`` writes a JSONL file whose every
line is one OpenAI fine-tune chat example. We verify:

* The script runs cleanly to a tmp path.
* Every line is valid JSON.
* Each example carries a ``messages`` list and a ``tools`` list.
* Every tool name referenced in the examples is registered in the
  current ToolRegistry — drift between training data and runtime is
  exactly the kind of silent bug a pre-merge test should catch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "ai_training_examples.py"


@pytest.fixture
def jsonl_out(tmp_path) -> Path:
    out = tmp_path / "examples.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"script failed: stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    return out


def _read_lines(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_script_emits_at_least_ten_examples(jsonl_out):
    examples = _read_lines(jsonl_out)
    assert len(examples) >= 10


def test_each_example_has_messages_and_tools(jsonl_out):
    for ex in _read_lines(jsonl_out):
        assert "messages" in ex
        assert isinstance(ex["messages"], list)
        assert ex["messages"][0]["role"] == "system"
        assert "tools" in ex
        assert isinstance(ex["tools"], list)
        assert len(ex["tools"]) > 0


def test_each_example_starts_with_user_after_system(jsonl_out):
    for ex in _read_lines(jsonl_out):
        roles = [m["role"] for m in ex["messages"]]
        assert roles[0] == "system"
        assert "user" in roles


def test_referenced_tool_names_match_registry(jsonl_out):
    from core.ai.tool_impls import build_tool_registry
    registry_names = set(build_tool_registry().names())
    referenced: set[str] = set()
    for ex in _read_lines(jsonl_out):
        for msg in ex["messages"]:
            calls = msg.get("tool_calls") or []
            for c in calls:
                fn = (c or {}).get("function", {}) or {}
                if fn.get("name"):
                    referenced.add(fn["name"])
    unknown = referenced - registry_names
    assert unknown == set(), f"examples reference unknown tools: {unknown}"


def test_every_tool_call_arguments_are_valid_json(jsonl_out):
    for ex in _read_lines(jsonl_out):
        for msg in ex["messages"]:
            for c in msg.get("tool_calls") or []:
                fn = (c or {}).get("function", {}) or {}
                args = fn.get("arguments")
                if args is None:
                    continue
                # Should always be a JSON-decodable string.
                json.loads(args)
