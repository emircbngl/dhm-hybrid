"""Offscreen tests for ``ui3.panels.depth_panel.DepthPanel``.

Builds a real ``PanelContext`` around a real ``WorkerBridge`` (which wraps
the genuine ``ui2`` compute drivers) and a synthetic hologram — mirrors the
pattern in ``tests/test_ui3_spine.py::_synthetic_hologram`` /
``test_mainwindow_end_to_end_reconstruct``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def _synthetic_hologram(tmp_path, n=192):
    import tifffile
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.5 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * 0.28 * X + bump))
    holo = np.abs(1 + obj) ** 2
    p = tmp_path / "holo.tif"
    tifffile.imwrite(p, (holo * 1000).astype(np.uint16))
    return p


class _FakeShell:
    """Minimal stand-in for MainWindow's PanelContext plumbing.

    Real ``WorkerBridge`` + real ``ReconParams`` (live, mutable instance) +
    a captured ``show_in_panel`` / ``set_status`` / ``toast`` so the test can
    assert on what the panel pushed without needing the full MainWindow.
    """

    def __init__(self, hologram_path: Path):
        from ui2.reconstruction import ReconParams
        from ui3.bridge import WorkerBridge

        self.bridge = WorkerBridge()
        self._params = ReconParams(
            wavelength_nm=632.8, pixel_um=1.0, mask_radius=40, z_mm=0.0,
            af_z_min_mm=-2.0, af_z_max_mm=2.0, af_n_steps=6,
        )
        self._hologram_path = hologram_path
        self.status_log = []
        self.toast_log = []
        self.panel_calls = []
        self.params_changed_calls = 0

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

    def get_field(self, target):
        return None

    def set_status(self, text, level="info"):
        self.status_log.append((text, level))

    def toast(self, text, level="info"):
        self.toast_log.append((text, level))

    def show_in_panel(self, target, array, **kw):
        self.panel_calls.append((target, np.asarray(array).copy(), kw))

    def build_context(self):
        from ui3.context import PanelContext
        return PanelContext(
            bridge=self.bridge,
            get_params=self.get_params,
            params_changed=self.params_changed,
            hologram_path=self.hologram_path,
            last_recon=self.last_recon,
            last_depth=self.last_depth,
            get_field=self.get_field,
            set_status=self.set_status,
            toast=self.toast,
            show_in_panel=self.show_in_panel,
            palette=None,
        )


def test_depth_panel_builds(qapp, tmp_path):
    from ui3.panels.depth_panel import DepthPanel

    holo = _synthetic_hologram(tmp_path)
    shell = _FakeShell(holo)
    ctx = shell.build_context()
    panel = DepthPanel(ctx)

    assert panel._sp_z_min.value() == pytest.approx(-2.0)
    assert panel._sp_z_max.value() == pytest.approx(2.0)
    assert panel._sp_n_steps.value() == 6
    assert panel._cb_metric.currentText() == "LAPLACIAN_VARIANCE"
    panel.deleteLater()
    shell.bridge.shutdown()


def test_depth_panel_controls_update_live_params(qapp, tmp_path):
    from ui3.panels.depth_panel import DepthPanel

    holo = _synthetic_hologram(tmp_path)
    shell = _FakeShell(holo)
    ctx = shell.build_context()
    panel = DepthPanel(ctx)

    panel._sp_z_min.setValue(-3.5)
    panel._sp_z_max.setValue(3.5)
    panel._sp_n_steps.setValue(8)
    panel._cb_metric.setCurrentText("BRENNER")

    p = shell.get_params()
    assert p.af_z_min_mm == pytest.approx(-3.5)
    assert p.af_z_max_mm == pytest.approx(3.5)
    assert p.af_n_steps == 8
    assert p.autofocus_metric == "BRENNER"
    assert shell.params_changed_calls > 0
    panel.deleteLater()
    shell.bridge.shutdown()


def test_depth_panel_compute_updates_readout_and_panel(qapp, tmp_path):
    from ui3.panels.depth_panel import DepthPanel

    holo = _synthetic_hologram(tmp_path)
    shell = _FakeShell(holo)
    ctx = shell.build_context()
    panel = DepthPanel(ctx)

    panel._sp_n_steps.setValue(6)
    panel._sp_window.setValue(3)

    panel._on_compute()

    deadline = time.time() + 20
    while panel.last_z_map() is None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)

    assert panel.last_z_map() is not None, (
        f"depth map never completed; status_log={shell.status_log}"
    )
    assert any(lvl == "error" for _, lvl in shell.status_log) is False

    # show_in_panel was called with the 'spectrum' target in mm units.
    targets = [t for t, _arr, _kw in shell.panel_calls]
    assert "spectrum" in targets
    _, arr, kw = [c for c in shell.panel_calls if c[0] == "spectrum"][-1]
    assert kw.get("contrast") == "minmax"
    z_map_mm = panel.last_z_map() * 1e3
    assert np.allclose(arr, z_map_mm)

    text = panel._readout.text()
    assert "mm" in text
    assert float(z_map_mm.min()) - 1e-3 <= float(z_map_mm.max())

    panel.deleteLater()
    shell.bridge.shutdown()


def test_depth_panel_clear_overlay(qapp, tmp_path):
    from ui3.panels.depth_panel import DepthPanel

    holo = _synthetic_hologram(tmp_path)
    shell = _FakeShell(holo)
    ctx = shell.build_context()
    panel = DepthPanel(ctx)

    panel._on_compute()
    deadline = time.time() + 20
    while panel.last_z_map() is None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert panel.last_z_map() is not None

    panel._on_clear()
    assert panel.last_z_map() is None
    assert "—" in panel._readout.text()

    panel.deleteLater()
    shell.bridge.shutdown()


def test_depth_panel_requires_hologram(qapp):
    from ui3.panels.depth_panel import DepthPanel

    shell = _FakeShell(None)
    ctx = shell.build_context()
    panel = DepthPanel(ctx)

    panel._on_compute()
    assert any("Load a hologram" in t for t, _lvl in shell.status_log)
    assert panel.last_z_map() is None

    panel.deleteLater()
    shell.bridge.shutdown()


def test_surface_requested_signal_emits_and_falls_back_to_status(qapp, tmp_path):
    from ui3.panels.depth_panel import DepthPanel

    holo = _synthetic_hologram(tmp_path)
    shell = _FakeShell(holo)
    ctx = shell.build_context()
    panel = DepthPanel(ctx)

    got = []
    panel.surface_requested.connect(lambda kind: got.append(kind))

    panel._surface_depth_btn.click()
    panel._surface_phase_btn.click()

    assert got == ["depth", "phase"]
    assert any("3D surface" in t for t, _lvl in shell.status_log)

    panel.deleteLater()
    shell.bridge.shutdown()
