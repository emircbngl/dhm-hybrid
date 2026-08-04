"""ui3 QPIPanel tests — offscreen Qt, real WorkerBridge (ui2 drivers),
synthetic hologram. Mirrors the ``_synthetic_hologram`` pattern from
``tests/test_ui3_spine.py``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DHM_USER", "ci")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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


@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def _make_ctx(qapp, tmp_path, holo_path):
    """A minimal-but-real PanelContext: a live WorkerBridge, a mutable
    ReconParams, and no-op display/feedback hooks (recorded for assertions)."""
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.context import PanelContext
    from ui3.design import DARK

    bridge = WorkerBridge()
    params = ReconParams(
        wavelength_nm=632.8, pixel_um=1.0, z_mm=0.0, mask_radius=40,
        n_sample=1.38, n_medium=1.337,
        af_z_min_mm=-0.5, af_z_max_mm=0.5, af_n_steps=12,
    )
    events: dict = {"status": [], "toast": []}

    ctx = PanelContext(
        bridge=bridge,
        get_params=lambda: params,
        params_changed=lambda: None,
        hologram_path=lambda: holo_path,
        last_recon=lambda: None,
        last_depth=lambda: None,
        get_field=lambda target: None,
        set_status=lambda text, level="info": events["status"].append((text, level)),
        toast=lambda text, level="info": events["toast"].append((text, level)),
        show_in_panel=lambda target, array, **kw: None,
        palette=DARK,
    )
    return ctx, bridge, params, events


def _pump_until(qapp, predicate, deadline_s=20):
    deadline = time.time() + deadline_s
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)


def test_qpi_panel_builds(qapp, tmp_path):
    from ui3.panels.qpi_panel import QPIPanel

    holo = _synthetic_hologram(tmp_path)
    ctx, bridge, params, events = _make_ctx(qapp, tmp_path, holo)
    panel = QPIPanel(ctx)

    assert panel.mode_combo.count() == 3
    assert panel.n_sample_spin.value() == pytest.approx(1.38)
    assert panel.n_medium_spin.value() == pytest.approx(1.337)
    for lbl in panel._readout_labels.values():
        assert lbl.text() == "—"

    bridge.shutdown()


def test_qpi_panel_controls_update_params(qapp, tmp_path):
    from ui3.panels.qpi_panel import QPIPanel

    holo = _synthetic_hologram(tmp_path)
    ctx, bridge, params, events = _make_ctx(qapp, tmp_path, holo)
    panel = QPIPanel(ctx)

    panel.n_sample_spin.setValue(1.40)
    assert params.n_sample == pytest.approx(1.40)
    panel.n_medium_spin.setValue(1.335)
    assert params.n_medium == pytest.approx(1.335)

    bridge.shutdown()


def test_compute_qpi_populates_readouts(qapp, tmp_path):
    from ui3.panels.qpi_panel import QPIPanel

    holo = _synthetic_hologram(tmp_path)
    ctx, bridge, params, events = _make_ctx(qapp, tmp_path, holo)
    panel = QPIPanel(ctx)

    got = {}
    bridge.qpi_done.connect(lambda r: got.update(done=r))
    bridge.qpi_error.connect(lambda m: got.update(err=m))

    panel._on_compute_clicked()
    _pump_until(qapp, lambda: "done" in got or "err" in got)

    assert "done" in got, f"QPI failed: {got.get('err')}"
    # The readout grid must reflect the real QPIResult — raw data, not
    # placeholder text — for at least the always-computed OPD range.
    assert panel._readout_labels["opd_range"].text() != "—"
    assert panel._readout_labels["runtime"].text() != "—"
    assert any(level == "ok" for _, level in events["toast"])

    bridge.shutdown()


def test_qpi_batch_populates_table(qapp, tmp_path):
    from ui3.panels.qpi_panel import QPIPanel

    holo = _synthetic_hologram(tmp_path)
    ctx, bridge, params, events = _make_ctx(qapp, tmp_path, holo)
    panel = QPIPanel(ctx)

    got = {}
    bridge.qpi_batch_done.connect(lambda r: got.update(done=r))
    bridge.qpi_batch_error.connect(lambda m: got.update(err=m))

    panel._on_batch_clicked()
    _pump_until(qapp, lambda: "done" in got or "err" in got, deadline_s=30)

    assert "done" in got, f"QPI batch failed: {got.get('err')}"
    entries = got["done"].entries
    assert panel.batch_table.rowCount() == len(entries)
    if entries:
        # First column (z_mm) must be a real numeric readout, never
        # "—" — every candidate carries a z_m.
        item = panel.batch_table.item(0, 0)
        assert item is not None and item.text() != "—"

    bridge.shutdown()


def test_no_hologram_warns_without_crashing(qapp, tmp_path):
    from ui3.panels.qpi_panel import QPIPanel

    ctx, bridge, params, events = _make_ctx(qapp, tmp_path, None)
    panel = QPIPanel(ctx)

    panel._on_compute_clicked()
    panel._on_batch_clicked()

    assert any("Load a hologram" in text for text, _ in events["status"])
    bridge.shutdown()
