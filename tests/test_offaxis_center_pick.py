"""Manual off-axis +1-order center override (2026-07-08).

The core (OffAxisParams.center_yx / build_plus_one_order_mask) always
accepted a manual center, but it was never plumbed from ReconParams into
the three OffAxisParams call sites, and ui3 had no way to set it. This
wires a click on the Spectrum viewport → ReconParams.offaxis_center →
reconstruction, with a reset-to-auto escape hatch.
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


# --- Qt-free: params + plumbing + persistence ----------------------------

def test_recon_params_has_offaxis_center_default_none():
    from core.drivers.reconstruction import ReconParams
    assert ReconParams().offaxis_center is None


def test_offaxis_params_threads_center_and_coerces_list():
    from core.drivers.reconstruction import ReconParams
    from core.drivers.workers import _offaxis_params
    assert _offaxis_params(ReconParams(mask_radius=55)).center_yx is None
    p = ReconParams(mask_radius=55)
    p.offaxis_center = [452, 635]          # as if from a JSON state round-trip
    oa = _offaxis_params(p)
    assert oa.center_yx == (452, 635)
    assert all(isinstance(v, int) for v in oa.center_yx)
    assert oa.radius == 55                 # radius still threaded


def test_state_round_trip_preserves_offaxis_center_tuple(tmp_path):
    from core.drivers.reconstruction import ReconParams
    from ui3.state import Ui3State, load_state, save_state
    st = Ui3State()
    st.set_recon_params(ReconParams(offaxis_center=(452, 635)))
    assert st.params["offaxis_center"] == [452, 635]     # JSON list on disk
    fp = tmp_path / "s.json"
    save_state(st, fp)
    back = load_state(fp).recon_params()
    assert back.offaxis_center == (452, 635)
    assert isinstance(back.offaxis_center, tuple)


def _write_offaxis_hologram(path: Path, n: int = 128) -> Path:
    """A carrier-fringe hologram whose +1 order sits off-center, so the
    chosen mask center actually changes the extracted field."""
    import tifffile
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    obj = 0.4 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    holo = np.abs(1 + obj * np.exp(1j * (2 * np.pi * 0.3 * X))) ** 2
    tifffile.imwrite(path, (holo * 1000).astype(np.uint16))
    return path


def test_driver_prepare_field_honours_manual_center(tmp_path):
    """Through a real call site (_prepare_field): a deliberately wrong
    manual center must yield a different field than auto-detect — proving
    offaxis_center is actually consumed, not ignored."""
    from core.drivers.reconstruction import ReconParams
    from core.drivers.workers import _prepare_field

    holo = _write_offaxis_hologram(tmp_path / "h.tif")

    field_auto, _, _, _ = _prepare_field(holo, ReconParams(mask_radius=30))
    field_manual, _, _, _ = _prepare_field(
        holo, ReconParams(mask_radius=30, offaxis_center=(5, 5)))

    assert field_auto.shape == field_manual.shape
    assert not np.allclose(field_auto, field_manual), (
        "manual center had no effect — it is not being plumbed into the "
        "offaxis extractor")


# --- ui3 viewport: pick mode + marker + click mapping --------------------

@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def test_viewport_pick_mode_and_marker(qapp):
    from ui3.design import DARK
    from ui3.viewport import ImagePanel
    panel = ImagePanel("spectrum", DARK)
    panel.set_image(np.random.default_rng(0).random((64, 64)))

    assert panel._pick_mode is False
    panel.set_pick_mode(True)
    assert panel._pick_mode is True
    panel.set_marker(10, 20)
    assert panel._marker is not None
    panel.clear_marker()
    assert panel._marker is None
    panel.set_pick_mode(False)
    assert panel._pick_mode is False


def test_viewport_click_maps_and_emits_only_when_armed(qapp, monkeypatch):
    from PySide6.QtCore import QPointF, Qt
    from ui3.design import DARK
    from ui3.viewport import ImagePanel
    panel = ImagePanel("spectrum", DARK)
    panel.set_image(np.zeros((64, 96)))       # h=64, w=96

    got = []
    panel.pixel_clicked.connect(lambda r, c: got.append((r, c)))
    # Deterministic mapping: pretend the click landed at data coords x=30.4,
    # y=12.6 (col, row) — bypasses pyqtgraph scene geometry.
    monkeypatch.setattr(panel._vb, "mapSceneToView",
                        lambda _pos: QPointF(30.4, 12.6))
    ev = SimpleNamespace(button=lambda: Qt.LeftButton,
                         scenePos=lambda: QPointF(0, 0),
                         accept=lambda: None)

    # Not armed → no emission.
    panel._on_scene_click(ev)
    assert got == []

    # Armed → emits the rounded (row, col).
    panel.set_pick_mode(True)
    panel._on_scene_click(ev)
    assert got == [(13, 30)]                   # round(12.6)=13, round(30.4)=30


def test_viewport_click_clamps_out_of_bounds(qapp, monkeypatch):
    from PySide6.QtCore import QPointF, Qt
    from ui3.design import DARK
    from ui3.viewport import ImagePanel
    panel = ImagePanel("spectrum", DARK)
    panel.set_image(np.zeros((64, 96)))
    got = []
    panel.pixel_clicked.connect(lambda r, c: got.append((r, c)))
    monkeypatch.setattr(panel._vb, "mapSceneToView",
                        lambda _pos: QPointF(9999, -50))
    panel.set_pick_mode(True)
    panel._on_scene_click(SimpleNamespace(
        button=lambda: Qt.LeftButton, scenePos=lambda: QPointF(0, 0),
        accept=lambda: None))
    assert got == [(0, 95)]                     # clamped to (row 0, col w-1)


# --- ui3 shell integration -----------------------------------------------

def test_shell_pick_sets_param_marker_and_reconstructs(qapp, monkeypatch):
    from ui3.main_window import MainWindow
    win = MainWindow()
    recons = []
    monkeypatch.setattr(win, "_on_reconstruct", lambda: recons.append(True))

    win._on_spectrum_center_picked(120, 200)
    assert win._params.offaxis_center == (120, 200)
    assert win.panel_spectrum._marker is not None
    assert recons == [True]                      # a reconstruct was triggered
    assert win._act_pick_order.isChecked() is False   # one-shot disarm
    win.close()


def test_shell_reset_clears_center_and_reconstructs(qapp, monkeypatch):
    from ui3.main_window import MainWindow
    win = MainWindow()
    win._params.offaxis_center = (120, 200)
    win.panel_spectrum.set_marker(120, 200)
    recons = []
    monkeypatch.setattr(win, "_on_reconstruct", lambda: recons.append(True))

    win._reset_order_center()
    assert win._params.offaxis_center is None
    assert win.panel_spectrum._marker is None
    assert recons == [True]
    win.close()


def test_shell_arm_without_hologram_warns_and_unchecks(qapp):
    from ui3.main_window import MainWindow
    win = MainWindow()
    assert win._raw is None
    win._act_pick_order.setChecked(True)         # fires _toggle_pick_order_center
    assert "Load a hologram" in win._status_label.text()
    assert win._act_pick_order.isChecked() is False
    assert win.panel_spectrum._pick_mode is False
    win.close()
