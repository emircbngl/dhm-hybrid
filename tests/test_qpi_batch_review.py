"""QPIBatchReviewDialog — inline comparison picker for the batch result.

Headless Qt dials + hand-crafted entries so the test is fast (no
reconstruction, no QPI pipeline) and deterministic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pytest


def _headless_qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return None
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception:
            return None
    return app


# ---- Lightweight stand-ins for QPIResult sub-fields -----------------------

@dataclass
class _PhaseStats:
    range_nm: float
    mean_nm: float = 0.0
    std_nm: float = 0.0


@dataclass
class _Morph:
    area_um2: float
    volume_fL: float = 0.0
    dry_mass_density_pg_um2: float = 0.0
    max_height_nm: float = 0.0
    mean_height_nm: float = 0.0
    perimeter_um: float = 0.0
    circularity: float = 0.0
    aspect_ratio: float = 0.0
    eccentricity: float = 0.0
    mean_phase_rad: float = 0.0
    phase_std_rad: float = 0.0


@dataclass
class _QPIResult:
    total_dry_mass_pg: Optional[float] = None
    step_height_m: Optional[float] = None
    phase_stats: Optional[_PhaseStats] = None
    cell_morph: Optional[_Morph] = None
    roughness: Optional[object] = None


def _make_entries(n: int = 2):
    from core.autofocus import FocusCandidate
    from core.qpi_batch import QPIBatchEntry

    out = []
    for i in range(n):
        q = _QPIResult(
            total_dry_mass_pg=100.0 + 10.0 * i,
            step_height_m=(1.0 + i) * 1e-6,
            phase_stats=_PhaseStats(range_nm=300.0 + 20.0 * i),
            cell_morph=_Morph(area_um2=150.0 + 5.0 * i,
                              circularity=0.8 - 0.05 * i),
        )
        out.append(QPIBatchEntry(
            candidate=FocusCandidate(
                z_m=-(i + 1) * 8e-3,
                score=0.9 - 0.1 * i,
                prominence=0.5 - 0.1 * i,
                rank=i,
            ),
            qpi_result=q,
        ))
    return out


# ---- import-free check ----------------------------------------------------

def test_module_imports():
    import gui.widgets.qpi_batch_review as m  # noqa: F401
    assert hasattr(m, "QPIBatchReviewDialog")


# ---- Qt-dependent --------------------------------------------------------

def test_dialog_renders_one_row_per_entry():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.qpi_batch_review import QPIBatchReviewDialog

    entries = _make_entries(3)
    dlg = QPIBatchReviewDialog(entries)
    try:
        assert dlg._table.rowCount() == 3
        # Rank is 1-based in the UI.
        assert dlg._table.item(0, 0).text() == "1"
        # z is formatted with sign + 3 decimals.
        z_text = dlg._table.item(0, 1).text()
        assert z_text.startswith("-8.000") or z_text.startswith("-08.000")
        # Dry mass rendered.
        assert dlg._table.item(0, 3).text() == "100"
        # Area, OPD range, step, circularity all populated.
        for col in (4, 5, 6, 7):
            assert dlg._table.item(0, col).text() not in ("", "—")
    finally:
        dlg.deleteLater()


def test_dialog_empty_entries_shows_placeholder():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.qpi_batch_review import QPIBatchReviewDialog

    dlg = QPIBatchReviewDialog([])
    try:
        assert dlg._table.rowCount() == 0
        assert "No QPI batch" in dlg._header.text()
        # Export button disabled when there's nothing to export.
        assert dlg._export_btn.isEnabled() is False
    finally:
        dlg.deleteLater()


def test_dialog_missing_qpi_fields_render_as_dash():
    """Partial QPI runs (e.g. micro-structure mode with no cell_morph)
    must render as em-dashes without crashing the row."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from core.autofocus import FocusCandidate
    from core.qpi_batch import QPIBatchEntry
    from gui.widgets.qpi_batch_review import QPIBatchReviewDialog

    entries = [
        QPIBatchEntry(
            candidate=FocusCandidate(z_m=-10e-3, score=0.5, prominence=0.2, rank=0),
            qpi_result=_QPIResult(),  # all None
        ),
    ]
    dlg = QPIBatchReviewDialog(entries)
    try:
        # Dry mass / area / OPD / step / circularity all missing → em-dash.
        for col in (3, 4, 5, 6, 7):
            assert dlg._table.item(0, col).text() == "—"
    finally:
        dlg.deleteLater()


def test_focus_here_button_emits_z_and_closes():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.qpi_batch_review import QPIBatchReviewDialog

    entries = _make_entries(2)
    dlg = QPIBatchReviewDialog(entries)
    try:
        fired: list[float] = []
        dlg.focus_requested.connect(lambda z: fired.append(z))
        btn = dlg._table.cellWidget(1, 8)  # second row, last column
        assert btn is not None
        btn.click()
        assert len(fired) == 1
        assert fired[0] == pytest.approx(entries[1].candidate.z_m)
    finally:
        dlg.deleteLater()


def test_export_csv_button_emits_signal():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.qpi_batch_review import QPIBatchReviewDialog

    entries = _make_entries(1)
    dlg = QPIBatchReviewDialog(entries)
    try:
        fired: list[bool] = []
        dlg.export_csv_requested.connect(lambda: fired.append(True))
        dlg._export_btn.click()
        assert fired == [True]
    finally:
        dlg.deleteLater()


def test_entries_api_returns_defensive_copy():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable")
    from gui.widgets.qpi_batch_review import QPIBatchReviewDialog

    entries = _make_entries(2)
    dlg = QPIBatchReviewDialog(entries)
    try:
        snap = dlg.entries()
        assert snap == entries
        assert snap is not entries
    finally:
        dlg.deleteLater()
