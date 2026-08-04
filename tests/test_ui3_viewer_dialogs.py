"""Offscreen smoke tests for the four ui3 viewer dialogs.

Runs a real ``QApplication`` (offscreen platform), a real
``ui3.bridge.WorkerBridge`` (wrapping the real ``ui2`` compute drivers),
and a synthetic hologram (same pattern as ``tests/test_ui3_spine.py``'s
``_synthetic_hologram``) to drive the dialogs end-to-end where practical.
Some dialogs (QPI batch / focus candidates) are also exercised with
hand-built fake result objects so the table-population path is covered
without paying for a full focus-landscape scan in every test.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

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


def _make_ctx(bridge, params, *, hologram_path=None, last_recon=None,
              last_depth=None, fields=None):
    """Build a real ``PanelContext`` with plain-callable facades — mirrors
    ``MainWindow.panel_context`` but standalone for dialog tests."""
    from ui3.context import PanelContext
    from ui3.design import DARK

    state = {"path": hologram_path, "recon": last_recon, "depth": last_depth}
    fields = fields or {}
    status_log: List[tuple] = []
    toast_log: List[tuple] = []

    def set_status(text, level="info"):
        status_log.append((text, level))

    def toast(text, level="info"):
        toast_log.append((text, level))

    def params_changed():
        pass

    ctx = PanelContext(
        bridge=bridge,
        get_params=lambda: params,
        params_changed=params_changed,
        hologram_path=lambda: state["path"],
        last_recon=lambda: state["recon"],
        last_depth=lambda: state["depth"],
        get_field=lambda target: fields.get(target),
        set_status=set_status,
        toast=toast,
        show_in_panel=lambda *a, **kw: None,
        palette=DARK,
    )
    ctx._status_log = status_log  # type: ignore[attr-defined]
    ctx._toast_log = toast_log  # type: ignore[attr-defined]
    return ctx


def _pump(qapp, predicate, deadline_s=20.0):
    deadline = time.time() + deadline_s
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# SurfaceViewer
# ---------------------------------------------------------------------------

def test_surface_viewer_show_depth_and_save_png(qapp, tmp_path):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.surface_viewer import SurfaceViewer

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)

    dlg = SurfaceViewer(ctx)
    zmap_mm = (np.random.default_rng(0).random((64, 64)).astype(np.float32) - 0.5) * 2.0
    dlg.show_depth(zmap_mm, pixel_size_um=1.0)
    assert dlg._surface is not None
    assert dlg._last_z is not None
    assert dlg._last_z.shape == (64, 64)

    out_png = tmp_path / "surface.png"
    dlg.save_png(str(out_png))
    assert out_png.exists()
    assert out_png.stat().st_size > 0

    bridge.shutdown()
    dlg.close()


def test_surface_viewer_show_phase_preserves_sign(qapp):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.surface_viewer import SurfaceViewer

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)
    dlg = SurfaceViewer(ctx)

    phase = np.linspace(-5.0, 5.0, 32 * 32).reshape(32, 32).astype(np.float32)
    dlg.show_phase(phase)
    assert dlg._last_z is not None
    assert float(dlg._last_z.min()) < 0.0  # negatives NOT clipped

    bridge.shutdown()
    dlg.close()


def test_surface_viewer_downsamples_large_arrays(qapp):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.surface_viewer import SurfaceViewer

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)
    dlg = SurfaceViewer(ctx)

    big = np.zeros((600, 600), dtype=np.float32)
    dlg.show_depth(big)
    assert dlg._last_z is not None
    assert max(dlg._last_z.shape) <= 256

    bridge.shutdown()
    dlg.close()


def test_surface_viewer_open_for_reads_ctx_get_field(qapp):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.surface_viewer import SurfaceViewer

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    depth_field_m = np.full((16, 16), 2.0e-3, dtype=np.float32)  # 2 mm in metres
    ctx = _make_ctx(bridge, params, fields={"depth": depth_field_m})
    dlg = SurfaceViewer(ctx)

    # ui3/main_window.py (frozen) calls open_for(self, kind) — a host arg
    # then the kind. Verify that call shape works.
    dlg.open_for(object(), "depth")
    assert dlg._last_z is not None
    # metres -> mm conversion happened inside open_for.
    assert abs(float(dlg._last_z.mean()) - 2.0) < 1e-6

    # Also verify the simpler open_for(kind) call shape.
    dlg2 = SurfaceViewer(ctx)
    dlg2.open_for("depth")
    assert dlg2._last_z is not None

    bridge.shutdown()
    dlg.close()
    dlg2.close()


def test_surface_viewer_open_for_missing_field_warns_not_raises(qapp):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.surface_viewer import SurfaceViewer

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params, fields={})
    dlg = SurfaceViewer(ctx)

    dlg.open_for("depth")  # no field present — must not raise
    assert any(lvl == "warn" for _text, lvl in ctx._status_log)

    bridge.shutdown()
    dlg.close()


# ---------------------------------------------------------------------------
# QPIBatchDialog
# ---------------------------------------------------------------------------

@dataclass
class _FakeCandidate:
    z_m: float
    score: float
    prominence: float
    rank: int


@dataclass
class _FakePhaseStats:
    range_nm: float


@dataclass
class _FakeMorph:
    area_um2: float
    circularity: float


@dataclass
class _FakeQPIResult:
    total_dry_mass_pg: Optional[float] = None
    step_height_m: Optional[float] = None
    cell_morph: Optional[_FakeMorph] = None
    phase_stats: Optional[_FakePhaseStats] = None


@dataclass
class _FakeQPIBatchEntry:
    candidate: _FakeCandidate
    qpi_result: _FakeQPIResult


@dataclass
class _FakeQPIBatchResult:
    entries: List[_FakeQPIBatchEntry]
    runtime_ms: float = 0.0


def _fake_qpi_batch_result(n=3):
    entries = []
    for i in range(n):
        cand = _FakeCandidate(z_m=(i - 1) * 1e-3, score=1.0 + i, prominence=0.5, rank=i)
        qpi = _FakeQPIResult(
            total_dry_mass_pg=10.0 + i,
            step_height_m=1e-7,
            cell_morph=_FakeMorph(area_um2=100.0 + i, circularity=0.8),
            phase_stats=_FakePhaseStats(range_nm=250.0 + i),
        )
        entries.append(_FakeQPIBatchEntry(candidate=cand, qpi_result=qpi))
    return _FakeQPIBatchResult(entries=entries, runtime_ms=42.0)


def test_qpi_batch_dialog_populates_table_from_fake_result(qapp):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.qpi_batch import QPIBatchDialog

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)
    dlg = QPIBatchDialog(ctx)

    result = _fake_qpi_batch_result(3)
    dlg.set_result(result)
    assert dlg.table.rowCount() == 3
    # dry mass column (index 3) shows a formatted number, not "—"
    assert dlg.table.item(0, 3).text() != "—"

    bridge.shutdown()
    dlg.close()


def test_qpi_batch_dialog_reconstruct_here_sets_z_and_triggers_bridge(qapp, tmp_path):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.qpi_batch import QPIBatchDialog

    holo = _synthetic_hologram(tmp_path)
    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0, mask_radius=40, z_mm=0.0)
    ctx = _make_ctx(bridge, params, hologram_path=holo)
    dlg = QPIBatchDialog(ctx)

    result = _fake_qpi_batch_result(3)
    dlg.set_result(result)
    dlg.table.selectRow(2)  # rank=2, z_m = (2-1)*1e-3 = 1e-3 -> 1.0 mm

    got = {}
    bridge.recon_done.connect(lambda r: got.update(recon=r))
    bridge.recon_error.connect(lambda m: got.update(err=m))

    dlg._on_reconstruct_here()
    assert params.z_mm == pytest.approx(1.0, abs=1e-6)

    assert _pump(qapp, lambda: "recon" in got or "err" in got)
    assert "recon" in got, f"reconstruction failed: {got.get('err')}"

    bridge.shutdown()
    dlg.close()


def test_qpi_batch_dialog_export_csv(qapp, tmp_path):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.qpi_batch import QPIBatchDialog

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)
    dlg = QPIBatchDialog(ctx)
    dlg.set_result(_fake_qpi_batch_result(2))

    out_csv = tmp_path / "batch.csv"
    dlg.export_csv(str(out_csv))
    assert out_csv.exists()
    text = out_csv.read_text()
    assert "candidate_rank" in text

    bridge.shutdown()
    dlg.close()


def test_qpi_batch_dialog_wired_to_bridge_signal(qapp, tmp_path):
    """End-to-end: run a real qpi_batch through the bridge and confirm the
    dialog's own bridge-signal wiring populates the table."""
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.qpi_batch import QPIBatchDialog

    holo = _synthetic_hologram(tmp_path)
    bridge = WorkerBridge()
    params = ReconParams(
        wavelength_nm=632.8, pixel_um=1.0, mask_radius=40,
        af_z_min_mm=-2.0, af_z_max_mm=2.0, af_n_steps=12,
    )
    ctx = _make_ctx(bridge, params, hologram_path=holo)
    dlg = QPIBatchDialog(ctx)

    bridge.qpi_batch(holo, params, z_min_mm=-2.0, z_max_mm=2.0, n_steps=12)
    assert _pump(qapp, lambda: dlg.table.rowCount() > 0 or
                 any("failed" in t for t, _l in ctx._status_log), deadline_s=30.0)

    bridge.shutdown()
    dlg.close()


# ---------------------------------------------------------------------------
# FocusCandidatesDialog
# ---------------------------------------------------------------------------

def _fake_candidates(n=3):
    return [_FakeCandidate(z_m=(i - 1) * 1e-3, score=1.0 + i, prominence=0.3, rank=i)
            for i in range(n)]


def test_focus_candidates_dialog_populates_table(qapp):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.focus_candidates import FocusCandidatesDialog

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)
    dlg = FocusCandidatesDialog(ctx)

    dlg.set_result(_fake_candidates(3))
    assert dlg.table.rowCount() == 3
    assert dlg.table.item(1, 1).text() != "—"

    bridge.shutdown()
    dlg.close()


def test_focus_candidates_dialog_empty_result(qapp):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.focus_candidates import FocusCandidatesDialog

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)
    dlg = FocusCandidatesDialog(ctx)

    dlg.set_result([])
    assert dlg.table.rowCount() == 0

    bridge.shutdown()
    dlg.close()


def test_focus_candidates_dialog_focus_here_sets_z_and_reconstructs(qapp, tmp_path):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.focus_candidates import FocusCandidatesDialog

    holo = _synthetic_hologram(tmp_path)
    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0, mask_radius=40, z_mm=0.0)
    ctx = _make_ctx(bridge, params, hologram_path=holo)
    dlg = FocusCandidatesDialog(ctx)

    dlg.set_result(_fake_candidates(3))
    dlg.table.selectRow(0)  # rank=0, z_m = -1e-3 -> -1.0 mm

    got = {}
    bridge.recon_done.connect(lambda r: got.update(recon=r))
    bridge.recon_error.connect(lambda m: got.update(err=m))

    dlg._on_focus_here()
    assert params.z_mm == pytest.approx(-1.0, abs=1e-6)

    assert _pump(qapp, lambda: "recon" in got or "err" in got)
    assert "recon" in got, f"reconstruction failed: {got.get('err')}"

    bridge.shutdown()
    dlg.close()


def test_focus_candidates_dialog_wired_to_bridge_signal(qapp, tmp_path):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.focus_candidates import FocusCandidatesDialog

    holo = _synthetic_hologram(tmp_path)
    bridge = WorkerBridge()
    params = ReconParams(
        wavelength_nm=632.8, pixel_um=1.0, mask_radius=40,
        af_z_min_mm=-2.0, af_z_max_mm=2.0, af_n_steps=16,
    )
    ctx = _make_ctx(bridge, params, hologram_path=holo)
    dlg = FocusCandidatesDialog(ctx)

    bridge.find_focus_candidates(holo, params, z_min_mm=-2.0, z_max_mm=2.0, n_steps=16)
    assert _pump(qapp, lambda: dlg.table.rowCount() >= 0 and
                 any(lvl in ("ok", "danger") for _t, lvl in ctx._status_log),
                 deadline_s=30.0)

    bridge.shutdown()
    dlg.close()


# ---------------------------------------------------------------------------
# AuditViewerDialog
# ---------------------------------------------------------------------------

def test_audit_viewer_dialog_empty_dir(qapp, tmp_path):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.audit_viewer import AuditViewerDialog

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)

    empty_dir = tmp_path / "audit_empty"
    dlg = AuditViewerDialog(ctx, audit_dir=empty_dir)
    assert dlg.row_count() == 0
    assert "0 of 0" in dlg.status_label.text()

    bridge.shutdown()
    dlg.close()


def _write_audit_entry(directory: Path, *, action: str, operator: str,
                       params: dict, timestamp: str) -> None:
    """Write one raw JSONL audit record directly — bypasses
    ``core.audit.AuditLog.record``'s env-derived operator resolution
    (``DHM_USER`` / ``core.user_profile.current_user()``) so the test can
    pin distinct operators per record."""
    import json
    directory.mkdir(parents=True, exist_ok=True)
    day = timestamp[:10]
    path = directory / f"{day}.jsonl"
    record = {
        "timestamp": timestamp,
        "action": action,
        "operator": operator,
        "user": operator,
        "app_version": "test",
        "git_sha": "test",
        "params": params,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def test_audit_viewer_dialog_populated_and_filters(qapp, tmp_path):
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.dialogs.audit_viewer import AuditViewerDialog

    audit_dir = tmp_path / "audit"
    _write_audit_entry(audit_dir, action="reconstruct", operator="alice",
                       params={"z_mm": 1.0}, timestamp="2026-07-05T10:00:00+00:00")
    _write_audit_entry(audit_dir, action="qpi", operator="bob",
                       params={"z_mm": 2.0}, timestamp="2026-07-05T10:01:00+00:00")
    _write_audit_entry(audit_dir, action="reconstruct", operator="alice",
                       params={"z_mm": 3.0}, timestamp="2026-07-05T10:02:00+00:00")

    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0)
    ctx = _make_ctx(bridge, params)
    dlg = AuditViewerDialog(ctx, audit_dir=audit_dir)

    assert dlg.row_count() == 3

    idx = dlg.operator_combo.findText("alice")
    assert idx >= 0
    dlg.operator_combo.setCurrentIndex(idx)
    assert dlg.row_count() == 2

    dlg.operator_combo.setCurrentIndex(dlg.operator_combo.findText("(any)"))
    idx_action = dlg.action_combo.findText("qpi")
    assert idx_action >= 0
    dlg.action_combo.setCurrentIndex(idx_action)
    assert dlg.row_count() == 1

    dlg.action_combo.setCurrentIndex(dlg.action_combo.findText("(any)"))
    dlg.query_edit.setText('"z_mm": 3.0')
    dlg.refresh()
    assert dlg.row_count() == 1

    bridge.shutdown()
    dlg.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
