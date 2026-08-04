"""Adaptive-autofocus settlement pins (2026-07-06, B-095).

The six search algorithms were benchmarked on 9 real lab scenes
(scripts/benchmark_af_real.py → tasks/af_real_benchmark.json,
write-up in docs/AUTOFOCUS_ADAPTIVE.md). Verdict: "robust" is the
only top performer across both metrics at the default 40-eval
budget (78%/67% ≤0.5 mm hit vs the old zscan default's 33%/44%).
These tests pin the settled defaults so a refactor can't silently
revert them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_recon_params_default_algorithm_is_robust():
    from ui2.reconstruction import ReconParams
    assert ReconParams().af_algorithm == "robust"


def test_settings_schema_default_algorithm_is_robust():
    from core.settings_schema import Ui2State
    assert Ui2State().af_algorithm == "robust"


def test_state_store_v9_migration_stays_zscan():
    """The v9 migration semantics are FROZEN: state files written before
    the algorithm field existed were deliberately pinned to the then-current
    zscan behaviour. The 2026-07-06 default change applies to NEW state
    only — a migration must never silently change what an existing lab
    setup does."""
    from ui2.state_store import _v8_to_v9  # type: ignore
    raw: dict = {"schema_version": 8}
    out = _v8_to_v9(raw)
    assert out["ui2"]["af_algorithm"] == "zscan"


def test_headless_autofocus_uses_robust_search(monkeypatch):
    """MCP/headless autofocus dispatches robust_coarse_to_fine_search
    (previously hardcoded adaptive_gradient_search, which the real-data
    benchmark showed is unreliable on entropy-like flat landscapes)."""
    import core.autofocus.search_classic as sc
    from dhm_mcp.headless import HeadlessSession

    called = {}

    def spy(*args, **kw):
        called["kw"] = kw
        return SimpleNamespace(best_z_m=0.005, best_score=1.23)

    monkeypatch.setattr(sc, "robust_coarse_to_fine_search", spy)

    s = HeadlessSession()
    # A tiny raw frame is enough — demodulate runs for real on 64×64.
    rng = np.random.default_rng(0)
    s.raw = (rng.random((64, 64)) * 1000).astype(np.float32)
    out = s.invoke_autofocus({"n_steps": 10})

    assert called, "robust_coarse_to_fine_search was not dispatched"
    assert called["kw"]["n_coarse"] == 10
    assert out["summary"]["best_z_mm"] == pytest.approx(5.0)
    # GUI-parity default metric (summary stores the enum VALUE, lowercase).
    assert out["summary"]["metric"] == "laplacian_variance"


def test_algorithm_profile_tips_carry_benchmark_verdicts():
    from ui2.workers import af_algorithm_input_profile
    assert "DEFAULT" in af_algorithm_input_profile("robust")["steps_tip"]
    assert "UNRELIABLE" in af_algorithm_input_profile("adaptive_distance")["steps_tip"]
    assert "ENTROPY" in af_algorithm_input_profile("adaptive_gradient")["steps_tip"]
