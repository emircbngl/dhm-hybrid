"""Tests for ``ui3.panels.focus_panel.FocusPanel``.

Offscreen Qt (QT_QPA_PLATFORM=offscreen), real ``WorkerBridge`` (which
wraps the real, already-tested ``ui2`` drivers) driven against a
synthetic hologram — same pattern as ``tests/test_ui3_spine.py``. No
compute is re-implemented or mocked: this proves the panel actually
drives the real autofocus / focus-candidate pipeline end to end.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DHM_USER", "ci")

import numpy as np
import pytest
from PySide6.QtCore import Qt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _synthetic_hologram(tmp_path, n=192):
    """Copied from tests/test_ui3_spine.py's pattern — a simple
    off-axis interference pattern with a Gaussian bump so the
    demodulated field has a clean, findable focus peak."""
    import tifffile
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.5 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * 0.28 * X + bump))
    holo = np.abs(1 + obj) ** 2
    p = tmp_path / "holo.tif"
    tifffile.imwrite(p, (holo * 1000).astype(np.uint16))
    return p


@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


class _FakeContext:
    """Minimal stand-in for ``ui3.context.PanelContext`` — real
    ``WorkerBridge`` (real compute drivers), a live mutable
    ``ReconParams`` instance, and simple recorder callables for the
    feedback channels so tests can assert on what the panel reported."""

    def __init__(self, bridge, params, hologram_path: Optional[Path]):
        self.bridge = bridge
        self._params = params
        self._hologram_path = hologram_path
        self.params_changed_calls = 0
        self.status_log: List[tuple] = []
        self.toast_log: List[tuple] = []
        self.palette = None

    def get_params(self):
        return self._params

    def params_changed(self):
        self.params_changed_calls += 1

    def hologram_path(self):
        return self._hologram_path

    def last_recon(self):
        return None

    def last_depth(self):
        return None

    def get_field(self, target: str):
        return None

    def set_status(self, text: str, level: str = "info") -> None:
        self.status_log.append((text, level))

    def toast(self, text: str, level: str = "info") -> None:
        self.toast_log.append((text, level))

    def show_in_panel(self, target: str, array, **kw) -> None:
        pass


def _pump_until(qapp, predicate, deadline_s: float = 20.0) -> None:
    deadline = time.time() + deadline_s
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)


@pytest.fixture()
def bridge():
    from ui3.bridge import WorkerBridge
    b = WorkerBridge()
    yield b
    b.shutdown()


@pytest.fixture()
def params():
    from ui2.reconstruction import ReconParams
    return ReconParams(
        wavelength_nm=632.8, pixel_um=1.0, mask_radius=40, z_mm=0.0,
        af_z_min_mm=-2.0, af_z_max_mm=2.0, af_n_steps=12,
    )


# ---------------------------------------------------------------------------
# Construction / initial sync
# ---------------------------------------------------------------------------

def test_panel_builds_and_syncs_from_params(qapp, bridge, params):
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)

    # "robust" = settled default since the 2026-07-06 real-data benchmark
    # (B-095); the panel mirrors whatever the params instance carries.
    assert panel.cmb_algorithm.currentText() == params.af_algorithm == "robust"
    assert panel.spin_z_min.value() == pytest.approx(-2.0)
    assert panel.spin_z_max.value() == pytest.approx(2.0)
    assert panel.spin_n_steps.value() == 12
    # Panel default metric is PHASE_VARIANCE when the field was unset
    # on a fresh ReconParams — but ReconParams.autofocus_metric
    # dataclass default is "LAPLACIAN_VARIANCE", so respect whatever
    # is actually on the params instance.
    assert panel.cmb_metric.currentText() == params.autofocus_metric


def test_algorithm_combo_lists_all_ui2_algorithms(qapp, bridge, params):
    from ui2.workers import available_autofocus_algorithms
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)
    items = [panel.cmb_algorithm.itemText(i)
             for i in range(panel.cmb_algorithm.count())]
    assert items == available_autofocus_algorithms()


def test_metric_combo_lists_all_focus_metrics(qapp, bridge, params):
    from core.autofocus import FocusMetric
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)
    items = [panel.cmb_metric.itemText(i)
             for i in range(panel.cmb_metric.count())]
    assert items == [m.name for m in FocusMetric]


# ---------------------------------------------------------------------------
# Controls mutate ReconParams
# ---------------------------------------------------------------------------

def test_algorithm_change_writes_params_and_triggers_params_changed(
        qapp, bridge, params):
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)
    before = ctx.params_changed_calls

    panel.cmb_algorithm.setCurrentText("adaptive_gradient")

    assert params.af_algorithm == "adaptive_gradient"
    assert ctx.params_changed_calls > before
    # Adaptive sub-section becomes visible for an adaptive algorithm.
    # (isVisibleTo, not isVisible — the panel is never shown/top-level
    # in this offscreen test, so isVisible() would always read False
    # regardless of setVisible(); isVisibleTo(panel) reflects the
    # widget's own visibility flag.)
    assert panel.grp_adaptive.isVisibleTo(panel)
    assert panel.lbl_adaptive_info.text() != ""


def test_metric_change_writes_params(qapp, bridge, params):
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)
    panel.cmb_metric.setCurrentText("TENENGRAD")
    assert params.autofocus_metric == "TENENGRAD"


def test_z_range_and_steps_change_write_params(qapp, bridge, params):
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)

    panel.spin_z_min.setValue(-5.0)
    panel.spin_z_max.setValue(5.0)
    panel.spin_n_steps.setValue(30)

    assert params.af_z_min_mm == pytest.approx(-5.0)
    assert params.af_z_max_mm == pytest.approx(5.0)
    assert params.af_n_steps == 30


def test_adaptive_distance_disables_z_range_and_labels_auto(
        qapp, bridge, params):
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)
    panel.cmb_algorithm.setCurrentText("adaptive_distance")

    assert not panel.spin_z_min.isEnabled()
    assert not panel.spin_z_max.isEnabled()
    assert "(auto)" in panel.lbl_z_min.text()
    assert "(auto)" in panel.lbl_z_max.text()
    assert panel.grp_adaptive.isVisibleTo(panel)


def test_classic_algorithm_keeps_z_range_enabled(qapp, bridge, params):
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)
    panel.cmb_algorithm.setCurrentText("coarse_to_fine")

    assert panel.spin_z_min.isEnabled()
    assert panel.spin_z_max.isEnabled()
    assert not panel.grp_adaptive.isVisibleTo(panel)


# ---------------------------------------------------------------------------
# No-hologram guards
# ---------------------------------------------------------------------------

def test_autofocus_without_hologram_warns_and_does_not_call_bridge(
        qapp, bridge, params, monkeypatch):
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)

    called = {}
    monkeypatch.setattr(bridge, "autofocus",
                        lambda *a, **kw: called.setdefault("hit", True))
    panel.btn_autofocus.click()

    assert "hit" not in called
    assert any(lvl == "warn" for _, lvl in ctx.status_log)


def test_find_candidates_without_hologram_warns(qapp, bridge, params, monkeypatch):
    from ui3.panels.focus_panel import FocusPanel

    ctx = _FakeContext(bridge, params, None)
    panel = FocusPanel(ctx)

    called = {}
    monkeypatch.setattr(bridge, "find_focus_candidates",
                        lambda *a, **kw: called.setdefault("hit", True))
    panel.btn_find_candidates.click()

    assert "hit" not in called
    assert any(lvl == "warn" for _, lvl in ctx.status_log)


# ---------------------------------------------------------------------------
# End-to-end: real autofocus / focus-candidate scan through the real bridge
# ---------------------------------------------------------------------------

def test_autofocus_button_runs_real_pipeline_and_updates_z(
        qapp, bridge, params, tmp_path):
    from ui3.panels.focus_panel import FocusPanel

    holo = _synthetic_hologram(tmp_path)
    ctx = _FakeContext(bridge, params, holo)
    panel = FocusPanel(ctx)

    panel.btn_autofocus.click()
    _pump_until(qapp, lambda: panel.lbl_result.text() != "" or
                any(lvl == "error" for _, lvl in ctx.status_log))

    assert not any(lvl == "error" for _, lvl in ctx.status_log), ctx.status_log
    assert panel.lbl_result.text() != ""
    assert ctx.params_changed_calls > 0
    # z_mm on the SAME params instance was updated to best_z (mm).
    assert params.z_mm != 0.0
    assert any("Autofocus" in t for t, _ in ctx.toast_log)


def test_find_candidates_button_runs_real_pipeline_and_populates_list(
        qapp, bridge, params, tmp_path):
    from ui3.panels.focus_panel import FocusPanel

    holo = _synthetic_hologram(tmp_path)
    ctx = _FakeContext(bridge, params, holo)
    panel = FocusPanel(ctx)

    got_signal: List[Any] = []
    panel.candidates_found.connect(lambda cands: got_signal.append(cands))

    panel.btn_find_candidates.click()
    _pump_until(qapp, lambda: panel.list_candidates.count() > 0 or
                any(lvl == "error" for _, lvl in ctx.status_log))

    assert not any(lvl == "error" for _, lvl in ctx.status_log), ctx.status_log
    assert panel.list_candidates.count() >= 1
    assert len(got_signal) == 1
    assert len(got_signal[0]) == panel.list_candidates.count()


def test_candidate_double_click_jumps_z_and_reconstructs(
        qapp, bridge, params, tmp_path):
    from ui3.panels.focus_panel import FocusPanel

    holo = _synthetic_hologram(tmp_path)
    ctx = _FakeContext(bridge, params, holo)
    panel = FocusPanel(ctx)

    panel.btn_find_candidates.click()
    _pump_until(qapp, lambda: panel.list_candidates.count() > 0 or
                any(lvl == "error" for _, lvl in ctx.status_log))
    assert panel.list_candidates.count() >= 1, ctx.status_log

    recon_done = {}
    bridge.recon_done.connect(lambda r: recon_done.setdefault("res", r))
    bridge.recon_error.connect(lambda m: recon_done.setdefault("err", m))

    item = panel.list_candidates.item(0)
    expected_z = float(item.data(Qt.UserRole))
    panel._on_candidate_double_clicked(item)

    assert params.z_mm == pytest.approx(expected_z)
    assert ctx.params_changed_calls > 0

    _pump_until(qapp, lambda: "res" in recon_done or "err" in recon_done)
    assert "res" in recon_done, f"reconstruction failed: {recon_done.get('err')}"
