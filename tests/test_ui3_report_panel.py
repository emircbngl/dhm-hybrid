"""ui3 ReportPanel tests — offscreen Qt, real WorkerBridge (ui2 drivers),
synthetic hologram. Mirrors the ``_synthetic_hologram`` pattern from
``tests/test_ui3_spine.py`` / ``tests/test_ui3_qpi_panel.py``.
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


def _make_ctx(tmp_path, holo_path, *, last_recon=None, last_depth=None,
              field_map=None):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.context import PanelContext
    from ui3.design import DARK

    bridge = WorkerBridge()
    params = ReconParams(
        wavelength_nm=632.8, pixel_um=1.0, z_mm=0.0, mask_radius=40,
        n_sample=1.38, n_medium=1.337,
        af_z_min_mm=-0.5, af_z_max_mm=0.5, af_n_steps=8,
    )
    events: dict = {"status": [], "toast": []}
    field_map = field_map or {}

    ctx = PanelContext(
        bridge=bridge,
        get_params=lambda: params,
        params_changed=lambda: None,
        hologram_path=lambda: holo_path,
        last_recon=lambda: last_recon["value"] if last_recon else None,
        last_depth=lambda: last_depth["value"] if last_depth else None,
        get_field=lambda target: field_map.get(target),
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


def _run_reconstruction(qapp, bridge, holo, params):
    """Drive a real reconstruction through the bridge and return the
    ReconResult (blocks the test until done, pumping the event loop)."""
    got = {}
    bridge.recon_done.connect(lambda r: got.update(done=r))
    bridge.recon_error.connect(lambda m: got.update(err=m))
    bridge.reconstruct(holo, params)
    _pump_until(qapp, lambda: "done" in got or "err" in got)
    assert "done" in got, f"reconstruction failed: {got.get('err')}"
    return got["done"]


def _run_depth_map(qapp, bridge, holo, params):
    got = {}
    bridge.depth_done.connect(lambda r: got.update(done=r))
    bridge.depth_error.connect(lambda m: got.update(err=m))
    bridge.depth_map(
        holo, params,
        z_min_mm=params.af_z_min_mm, z_max_mm=params.af_z_max_mm,
        n_steps=params.af_n_steps, window_size=5,
    )
    _pump_until(qapp, lambda: "done" in got or "err" in got, deadline_s=30)
    assert "done" in got, f"depth map failed: {got.get('err')}"
    return got["done"]


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_report_panel_builds(qapp, tmp_path):
    from ui3.panels.report_panel import ReportPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReportPanel(ctx)

    assert panel.html_btn.text() == "Export HTML report"
    assert panel.pdf_btn.text() == "Export PDF report"
    assert panel.qpi_csv_btn.text() == "Export QPI CSV"
    assert panel.bundle_btn.text() == "Export tomography bundle"
    assert panel._path_label.text() == "No export yet."

    bridge.shutdown()


# ---------------------------------------------------------------------------
# Warn-without-crashing paths (no data yet)
# ---------------------------------------------------------------------------

def test_export_html_warns_without_recon(qapp, tmp_path):
    from ui3.panels.report_panel import ReportPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReportPanel(ctx)

    panel._on_export_html()

    assert any("reconstruction" in text.lower() for text, _ in events["status"])
    bridge.shutdown()


def test_export_pdf_warns_without_recon(qapp, tmp_path):
    from ui3.panels.report_panel import ReportPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReportPanel(ctx)

    panel._on_export_pdf()

    assert any("reconstruction" in text.lower() for text, _ in events["status"])
    bridge.shutdown()


def test_export_qpi_csv_warns_without_recon(qapp, tmp_path):
    from ui3.panels.report_panel import ReportPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReportPanel(ctx)

    panel._on_export_qpi_csv()

    assert any("reconstruction" in text.lower() or "qpi" in text.lower()
               for text, _ in events["status"])
    bridge.shutdown()


def test_export_qpi_csv_warns_without_phase_unwrapped(qapp, tmp_path):
    """last_recon() present but get_field('phase_unwrapped') is None ->
    'run QPI first' style warning, no crash."""
    from ui3.panels.report_panel import ReportPanel

    last_recon = {"value": object()}  # non-None sentinel is enough
    ctx, bridge, params, events = _make_ctx(
        tmp_path, None, last_recon=last_recon, field_map={})
    panel = ReportPanel(ctx)

    panel._on_export_qpi_csv()

    assert any("qpi" in text.lower() for text, _ in events["status"])
    bridge.shutdown()


def test_export_bundle_warns_without_depth(qapp, tmp_path):
    from ui3.panels.report_panel import ReportPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReportPanel(ctx)

    panel._on_export_bundle()

    assert any("depth" in text.lower() for text, _ in events["status"])
    bridge.shutdown()


# ---------------------------------------------------------------------------
# Real exports — file actually written to a tmp target
# ---------------------------------------------------------------------------

def test_export_html_report_writes_file(qapp, tmp_path, monkeypatch):
    from ui3.panels.report_panel import ReportPanel
    from PySide6.QtWidgets import QFileDialog

    holo = _synthetic_hologram(tmp_path)
    last_recon = {"value": None}
    ctx, bridge, params, events = _make_ctx(tmp_path, holo, last_recon=last_recon)
    panel = ReportPanel(ctx)

    recon_result = _run_reconstruction(qapp, bridge, holo, params)
    last_recon["value"] = recon_result

    target = tmp_path / "out_report.html"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **kw: (str(target), "")))

    panel._on_export_html()

    assert target.exists() and target.stat().st_size > 0
    assert any(level == "ok" for _, level in events["toast"])
    assert str(target) in panel._path_label.text()

    bridge.shutdown()


def test_export_pdf_report_writes_file(qapp, tmp_path, monkeypatch):
    from ui3.panels.report_panel import ReportPanel
    from PySide6.QtWidgets import QFileDialog

    holo = _synthetic_hologram(tmp_path)
    last_recon = {"value": None}
    ctx, bridge, params, events = _make_ctx(tmp_path, holo, last_recon=last_recon)
    panel = ReportPanel(ctx)

    recon_result = _run_reconstruction(qapp, bridge, holo, params)
    last_recon["value"] = recon_result

    target = tmp_path / "out_report.pdf"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **kw: (str(target), "")))

    panel._on_export_pdf()

    assert target.exists() and target.stat().st_size > 0
    assert any(level == "ok" for _, level in events["toast"])

    bridge.shutdown()


def test_export_qpi_csv_writes_file(qapp, tmp_path, monkeypatch):
    from ui3.panels.report_panel import ReportPanel
    from PySide6.QtWidgets import QFileDialog

    holo = _synthetic_hologram(tmp_path)
    last_recon = {"value": None}
    ctx, bridge, params, events = _make_ctx(tmp_path, holo, last_recon=last_recon)
    panel = ReportPanel(ctx)

    recon_result = _run_reconstruction(qapp, bridge, holo, params)
    last_recon["value"] = recon_result
    # Wire get_field("phase_unwrapped") to the real recon result, same as
    # MainWindow.get_field does.
    phase = getattr(recon_result, "unwrapped_phase", None)
    if phase is None or not getattr(phase, "size", 0):
        phase = recon_result.phase
    ctx.get_field = lambda target: (
        phase if target == "phase_unwrapped" else None)

    target = tmp_path / "out_qpi.csv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **kw: (str(target), "")))

    panel._on_export_qpi_csv()

    assert target.exists() and target.stat().st_size > 0
    content = target.read_text()
    assert "opd_range_nm" in content or "total_dry_mass_pg" in content
    assert any(level == "ok" for _, level in events["toast"])

    bridge.shutdown()


def test_export_bundle_writes_files(qapp, tmp_path, monkeypatch):
    from ui3.panels.report_panel import ReportPanel
    from PySide6.QtWidgets import QFileDialog

    holo = _synthetic_hologram(tmp_path)
    last_depth = {"value": None}
    ctx, bridge, params, events = _make_ctx(tmp_path, holo, last_depth=last_depth)
    panel = ReportPanel(ctx)

    depth_result = _run_depth_map(qapp, bridge, holo, params)
    last_depth["value"] = depth_result

    out_dir = tmp_path / "bundle_out"
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **kw: str(out_dir)))

    panel._on_export_bundle()

    written = list(out_dir.glob("tomography_*"))
    assert written, "expected at least one tomography bundle file"
    assert any(level == "ok" for _, level in events["toast"])

    bridge.shutdown()


def test_default_export_dir_is_created():
    from ui3.panels.report_panel import _default_export_dir

    d = _default_export_dir()
    assert d.exists()
    assert d == Path.home() / ".dhm-reconstruction" / "exports"
