"""Autofocus flat/degenerate-landscape diagnostic (2026-07-08, B-098).

Ported from the original Phyton app's autofocus during the version audit:
Hybrid's plain linear z-scan silently returned argmax-of-noise when the
focus-score curve was flat (wrong pixel size / magnification / wavelength /
z-range / +1-order mask). It now attaches a non-fatal ``warning`` to the
result — best-guess z preserved, but the UI can flag it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# --- Qt-free: the diagnostic helper --------------------------------------

def test_flat_curve_warns():
    from core.autofocus.evaluator import focus_landscape_warning
    zs = np.linspace(-1e-3, 1e-3, 40)
    flat = {float(z): 1.0 + 1e-6 * i for i, z in enumerate(zs)}  # ~constant
    w = focus_landscape_warning(flat)
    assert w is not None and "flat" in w.lower()


def test_real_peak_does_not_warn():
    from core.autofocus.evaluator import focus_landscape_warning
    zs = np.linspace(-1e-3, 1e-3, 41)
    peak = {float(z): float(np.exp(-((z / 2e-4) ** 2))) for z in zs}  # clear bump
    assert focus_landscape_warning(peak) is None


def test_mostly_nonfinite_warns():
    from core.autofocus.evaluator import focus_landscape_warning
    scores = {0.0: float("nan"), 1e-4: float("inf"), 2e-4: 1.0}  # <3 finite
    w = focus_landscape_warning(scores)
    assert w is not None and "non-finite" in w.lower()


def test_empty_scores_no_warning():
    from core.autofocus.evaluator import focus_landscape_warning
    assert focus_landscape_warning({}) is None


# --- zscan wires the diagnostic + healthy run stays clean ----------------

def _synthetic_field(n: int = 96):
    """A carrier-modulated Gaussian bump → a real off-axis field with an
    actual focus, so a normal scan produces a non-flat curve."""
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.6 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 10) ** 2)))
    return (0.3 + bump) * np.exp(1j * (0.2 * X + bump)).astype(np.complex64)


def test_zscan_result_has_warning_field_and_stays_none_on_real_field():
    from core.autofocus import FocusMetric
    from core.autofocus.search_classic import autofocus_zscan
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    field = _synthetic_field()
    base = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=1e-6,
                                z_m=0.0, n=1.0)
    zs = list(np.linspace(-2e-3, 2e-3, 25))
    res = autofocus_zscan(field, base, zs, ReconstructionMethod.ASM,
                          FocusMetric.LAPLACIAN_VARIANCE)
    assert hasattr(res, "warning")           # field exists
    assert res.warning is None               # a real field is not flagged


def test_zscan_flags_a_degenerate_all_equal_field():
    """A uniform field has no z-dependence → every metric is identical →
    flat landscape → warning fires (the exact mis-parameterised case)."""
    from core.autofocus import FocusMetric
    from core.autofocus.search_classic import autofocus_zscan
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    flat_field = np.ones((64, 64), dtype=np.complex64)
    base = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=1e-6,
                                z_m=0.0, n=1.0)
    zs = list(np.linspace(-1e-3, 1e-3, 20))
    res = autofocus_zscan(flat_field, base, zs, ReconstructionMethod.ASM,
                          FocusMetric.LAPLACIAN_VARIANCE)
    assert res.warning is not None


# --- the diagnostic now covers the DEFAULT 'robust' path (B-100) ---------
# Until 2026-07-08 the driver only forwarded the linear-sweep warning; every
# other algorithm — including the settled default 'robust' — returned
# warning=None, so a mis-parameterised run on the default path degraded
# silently. _landscape_warning diagnoses from each search's retained landscape.

def test_landscape_warning_passes_through_linear_result():
    from core.drivers.workers import _landscape_warning
    assert _landscape_warning(SimpleNamespace(warning="flat!")) == "flat!"
    assert _landscape_warning(SimpleNamespace(warning=None)) is None


def test_landscape_warning_diagnoses_flat_robust_coarse_scores():
    """Default 'robust' keeps a uniform coarse landscape — a flat one must
    now warn (it silently returned None before B-100)."""
    from core.drivers.workers import _landscape_warning
    from core.autofocus.search_classic import RobustSearchResult
    zs = np.linspace(-1e-3, 1e-3, 40)
    flat = np.array([1.0 + 1e-6 * i for i in range(zs.size)])  # ~constant
    res = RobustSearchResult(
        best_z_m=0.0, best_score=1.0, total_evaluations=40,
        coarse_best_z_m=0.0, fine_range=(-1e-4, 1e-4),
        coarse_z=zs, coarse_scores=flat)
    w = _landscape_warning(res)
    assert w is not None and "flat" in w.lower()


def test_landscape_warning_none_on_peaked_robust_coarse_scores():
    from core.drivers.workers import _landscape_warning
    from core.autofocus.search_classic import RobustSearchResult
    zs = np.linspace(-1e-3, 1e-3, 41)
    peak = np.exp(-((zs / 2e-4) ** 2))  # clear bump
    res = RobustSearchResult(
        best_z_m=0.0, best_score=1.0, total_evaluations=41,
        coarse_best_z_m=0.0, fine_range=(-1e-4, 1e-4),
        coarse_z=zs, coarse_scores=peak)
    assert _landscape_warning(res) is None


def test_landscape_warning_none_for_scoreless_golden_result():
    """golden / coarse_to_fine keep only the best point — no landscape to
    diagnose, so None (not a false alarm)."""
    from core.drivers.workers import _landscape_warning
    from core.autofocus.search_classic import GoldenSearchResult
    res = GoldenSearchResult(best_z_m=0.0, best_score=1.0,
                             iterations=10, evaluations=20)
    assert _landscape_warning(res) is None


def test_robust_search_on_degenerate_field_is_flagged_end_to_end():
    """Full default-path check: robust search on a z-invariant field yields a
    flat coarse landscape → _landscape_warning fires (the exact
    mis-parameterised case, now caught on the default algorithm)."""
    from core.autofocus import FocusMetric
    from core.autofocus.search_classic import robust_coarse_to_fine_search
    from core.drivers.workers import _landscape_warning
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    flat_field = np.ones((64, 64), dtype=np.complex64)
    base = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=1e-6,
                                z_m=0.0, n=1.0)
    res = robust_coarse_to_fine_search(
        flat_field, base, ReconstructionMethod.ASM,
        FocusMetric.LAPLACIAN_VARIANCE, z_min_m=-1e-3, z_max_m=1e-3,
        n_coarse=40)
    assert _landscape_warning(res) is not None


# --- worker + ui3 surfacing ----------------------------------------------

def test_worker_autofocus_result_carries_warning():
    from core.drivers.workers import AutofocusResult
    r = AutofocusResult(best_z_m=0.0, score=1.0, scanned=10, runtime_ms=5.0,
                        warning="flat!")
    assert r.warning == "flat!"
    # default stays None (non-breaking for the other search branches)
    assert AutofocusResult(0.0, 1.0, 10, 5.0).warning is None


@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def test_focus_panel_surfaces_warning_as_warn_status(qapp, monkeypatch):
    from ui3.main_window import MainWindow
    win = MainWindow()
    panel = win._panels.get("focus")
    assert panel is not None

    toasts = []
    monkeypatch.setattr(win._toasts, "show_toast",
                        lambda text, level="info": toasts.append((text, level)))

    # With a warning: status carries the diagnostic, result label gets ⚠.
    warn_result = SimpleNamespace(best_z_m=0.012, score=1.0, scanned=20,
                                  runtime_ms=5.0,
                                  warning="focus scores are nearly flat")
    panel._on_autofocus_done(warn_result)
    assert "flat" in win._status_label.text()
    assert "⚠" in panel.lbl_result.text()
    assert any(lvl == "warn" for _t, lvl in toasts)

    # Without a warning: normal "ok" path, no ⚠.
    toasts.clear()
    ok_result = SimpleNamespace(best_z_m=0.008, score=1.0, scanned=20,
                                runtime_ms=5.0, warning=None)
    panel._on_autofocus_done(ok_result)
    assert "Autofocus done" in win._status_label.text()
    assert "⚠" not in panel.lbl_result.text()
    assert any(lvl == "ok" for _t, lvl in toasts)
    win.close()
