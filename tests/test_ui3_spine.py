"""ui3 spine tests — design tokens, wcag, state, command registry, and an
offscreen end-to-end MainWindow smoke (load → reconstruct → display).

Qt runs headless via the offscreen platform (set before any Qt import), so no
DPG-style stubbing is needed — a real QApplication drives real widgets.
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


# --- Qt-free: design + wcag ------------------------------------------------

def test_all_palettes_pass_wcag():
    from ui3 import design, wcag
    for name, p in design.PALETTES.items():
        assert wcag.failing_pairs(p) == [], f"{name} fails WCAG: {wcag.failing_pairs(p)}"


def test_build_qss_is_a_nonempty_string_for_each_palette():
    from ui3.design import PALETTES, build_qss
    for p in PALETTES.values():
        qss = build_qss(p)
        assert isinstance(qss, str) and len(qss) > 500
        assert p.accent in qss and p.bg in qss


# --- Qt-free: state --------------------------------------------------------

def test_state_roundtrip_and_recon_params(tmp_path):
    from ui3.state import Ui3State, load_state, save_state, push_recent
    st = Ui3State(theme="midnight", sample_id="cellA")
    st.params = {"wavelength_nm": 633.0, "pixel_um": 0.5, "reference_mode": "reference_free"}
    push_recent(st, "/a/b/holo1.tif")
    push_recent(st, "/a/b/holo2.tif")
    p = tmp_path / "ui3_state.json"
    save_state(st, p)
    back = load_state(p)
    assert back.theme == "midnight"
    assert back.recent[0] == "/a/b/holo2.tif"       # most-recent first
    rp = back.recon_params()
    assert rp.wavelength_nm == 633.0
    assert rp.reference_mode == "reference_free"


def test_load_state_missing_file_returns_defaults(tmp_path):
    from ui3.state import load_state
    st = load_state(tmp_path / "nope.json")
    assert st.theme == "dark"


def test_push_recent_dedups_and_caps():
    from ui3.state import Ui3State, push_recent
    st = Ui3State()
    for i in range(20):
        push_recent(st, f"/f/{i}.tif")
    push_recent(st, "/f/0.tif")
    assert st.recent[0] == "/f/0.tif"
    assert len(st.recent) <= 12
    assert st.recent.count("/f/0.tif") == 1


# --- Qt-free: command registry --------------------------------------------

def test_command_registry_search():
    from ui3.widgets.command_palette import Command, CommandRegistry
    reg = CommandRegistry()
    reg.register(Command("a", "Reconstruct", lambda: None, "Process"))
    reg.register(Command("b", "Compute QPI", lambda: None, "Analyse"))
    reg.register(Command("c", "Autofocus", lambda: None, "Process"))
    assert [c.id for c in reg.search("recon")] == ["a"]
    assert {c.id for c in reg.search("process")} == {"a", "c"}
    assert len(reg.search("")) == 3


# --- Offscreen Qt: app + widgets ------------------------------------------

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


def test_image_panel_displays_array(qapp):
    from ui3.design import DARK
    from ui3.viewport import ImagePanel
    panel = ImagePanel("t", DARK)
    assert not panel.has_image()
    panel.set_image(np.random.default_rng(0).random((64, 64)), pixel_size_um=0.1)
    assert panel.has_image()
    panel.clear()
    assert not panel.has_image()


def test_mainwindow_end_to_end_reconstruct(qapp, tmp_path):
    from ui3.main_window import MainWindow
    from ui3.state import Ui3State

    holo = _synthetic_hologram(tmp_path)
    st = Ui3State()
    st.params = {"wavelength_nm": 632.8, "pixel_um": 1.0, "mask_radius": 40, "z_mm": 0.0}
    win = MainWindow(st)

    assert set(win._image_panels) == {"input", "amplitude", "phase", "spectrum"}
    assert len(win.commands.all()) >= 20

    got = {}
    win.bridge.recon_done.connect(lambda r: got.update(recon=r))
    win.bridge.recon_error.connect(lambda m: got.update(err=m))

    win._load_path(str(holo))
    assert win.panel_input.has_image()

    win._on_reconstruct()
    deadline = time.time() + 20
    while "recon" not in got and "err" not in got and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)

    assert "recon" in got, f"reconstruction failed: {got.get('err')}"
    assert win.panel_amp.has_image()
    assert win.panel_phase.has_image()

    # theme + mode + maximize don't raise
    for t in ("midnight", "light", "high_contrast", "dark"):
        win._apply_theme(t)
    for m in ("Acquire", "Reconstruct", "Analyse", "Report"):
        win._apply_workflow_mode(m)
    win._maximize_panel("phase")
    win._restore_grid()

    # state export carries the params back
    out = win.export_state()
    assert out.params.get("wavelength_nm") == 632.8
    win.close()


def test_mainwindow_requires_hologram_before_compute(qapp):
    from ui3.main_window import MainWindow
    win = MainWindow()
    # No hologram loaded → compute actions no-op with a status warning.
    win._on_reconstruct()
    win._on_qpi()
    win._on_autofocus()
    assert "Load a hologram" in win._status_label.text()
    win.close()


def test_autofocus_done_converts_best_z_m_to_mm(qapp):
    """Review CONFIRMED: _on_autofocus_done read a nonexistent best_z_mm and
    silently reconstructed at the stale z. It must read best_z_m (metres) and
    convert to mm."""
    from types import SimpleNamespace
    from ui3.main_window import MainWindow
    win = MainWindow()
    win._params.z_mm = 0.0
    # avoid the auto-reconstruct side effect (no hologram loaded → early return)
    win._on_autofocus_done(SimpleNamespace(best_z_m=0.0123))
    assert win._params.z_mm == pytest.approx(12.3, rel=1e-6)
    win.close()


def test_close_shuts_down_panels_without_error(qapp, tmp_path):
    """closeEvent must tear down thread-owning panels (Qt doesn't deliver
    closeEvent to docked content widgets) — no hang/crash on close."""
    from ui3.main_window import MainWindow
    from ui3.state import Ui3State
    st = Ui3State()
    st.onboarding_seen = True
    win = MainWindow(st)
    # panels that own threads expose shutdown(); assert they're reachable
    for key in ("ai", "timelapse"):
        panel = win._panels.get(key)
        if panel is not None:
            assert hasattr(panel, "shutdown")
    win.close()   # must not raise / hang
    qapp.processEvents()


def test_app_sets_macos_paint_engine_workaround():
    """Regression: ui3 must set QT_WIDGETS_RHI=0 before QApplication —
    without it, PySide6 6.10 on macOS Sonoma+ segfaults on first paint
    (QPainter engine==0). Importing ui3.app applies it (2026-07-06)."""
    import os
    import ui3.app  # noqa: F401 — import applies the env guard
    assert os.environ.get("QT_WIDGETS_RHI") == "0"
