"""Session-level CSV export tests (v2.0.7, T2).

Karin needs hücre × zaman matrisi out of a session run. Two layouts:

* ``long`` (tidy) — one row per (frame, cell). Cell-less frames
  still get a row with ``cell_id=""`` so the audit covers
  autofocus-only runs.
* ``wide`` — one row per frame, columns ``cell_<id>_<metric>``.
  Friendly to Excel pivot tables.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.session import HologramFrame, Session  # noqa: E402
from core.session_export import (  # noqa: E402
    CellMeasurement,
    FrameResult,
    write_session_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _frame(idx: int, path: str = None) -> HologramFrame:
    return HologramFrame(
        path=path or f"f{idx:03d}.tif",
        timestamp_s=1_700_000_000.0 + idx * 5.0,
        index=idx,
    )


def _two_frame_session_results():
    """A 2-frame session, one with two cells, one with no cells.
    Covers the "frame still recorded" branch + the cell-row branch."""
    s = Session(
        id="abc123",
        operator="karin",
        sample_id="A549",
        params={},
        frames=[_frame(0), _frame(1)],
    )
    r0 = FrameResult(
        frame=s.frames[0], z_mm=12.4, runtime_ms=315.2,
        cells=[
            CellMeasurement(
                cell_id=1, cy_px=128.0, cx_px=128.0,
                z_mm=12.4, dry_mass_pg=88.5, area_um2=42.0,
                height_nm=620.0,
            ),
            CellMeasurement(
                cell_id=2, cy_px=180.0, cx_px=180.0,
                z_mm=12.5, dry_mass_pg=95.1, area_um2=51.0,
                height_nm=710.0,
            ),
        ],
    )
    r1 = FrameResult(
        frame=s.frames[1], z_mm=12.6, runtime_ms=298.7,
        # No cells — autofocus-only branch.
        cells=[],
    )
    return s, [r0, r1]


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


# ---------------------------------------------------------------------------
# Long format
# ---------------------------------------------------------------------------

def test_long_format_one_row_per_cell(tmp_path):
    s, rs = _two_frame_session_results()
    out = write_session_csv(s, rs, tmp_path / "out.csv", layout="long")
    header, body = _read_csv(out)
    # 2 cells (from frame 0) + 1 cell-less frame (from frame 1) = 3 rows
    assert len(body) == 3
    # Column set sanity
    assert "cell_dry_mass_pg" in header
    assert "frame_index" in header
    assert "operator" in header
    # Per-cell row carries cell metrics; cell-less row has empty
    # cell columns.
    cell_id_idx = header.index("cell_id")
    frame_index_idx = header.index("frame_index")
    cells_seen = sorted(int(r[cell_id_idx]) for r in body
                        if r[cell_id_idx])
    assert cells_seen == [1, 2]
    # Frame-1 row is cell-less.
    frame_one_rows = [r for r in body if r[frame_index_idx] == "1"]
    assert len(frame_one_rows) == 1
    assert frame_one_rows[0][cell_id_idx] == ""


def test_long_format_carries_session_metadata(tmp_path):
    s, rs = _two_frame_session_results()
    out = write_session_csv(s, rs, tmp_path / "out.csv", layout="long")
    header, body = _read_csv(out)
    sid_idx = header.index("session_id")
    op_idx = header.index("operator")
    samp_idx = header.index("sample_id")
    for row in body:
        assert row[sid_idx] == "abc123"
        assert row[op_idx] == "karin"
        assert row[samp_idx] == "A549"


def test_long_format_carries_error_field(tmp_path):
    """A frame that errored out still gets a row, with the error
    text in ``frame_error``. Karin needs to see which frames failed
    inside the same CSV she opens for analysis — no separate log
    chase."""
    s = Session(id="x", operator="erik", sample_id="HeLa",
                frames=[_frame(0)])
    r = FrameResult(
        frame=s.frames[0], z_mm=None, runtime_ms=12.0,
        error="autofocus diverged",
    )
    out = write_session_csv(s, [r], tmp_path / "err.csv", layout="long")
    header, body = _read_csv(out)
    err_idx = header.index("frame_error")
    assert body[0][err_idx] == "autofocus diverged"


def test_long_format_blank_cell_when_metric_none(tmp_path):
    """``CellMeasurement(area_um2=None)`` writes empty string,
    not 'None'. Excel must show blank cell."""
    s = Session(id="x", operator="", sample_id="", frames=[_frame(0)])
    cell = CellMeasurement(cell_id=5, dry_mass_pg=88.0)
    r = FrameResult(frame=s.frames[0], z_mm=10.0, cells=[cell])
    out = write_session_csv(s, [r], tmp_path / "x.csv", layout="long")
    header, body = _read_csv(out)
    area_idx = header.index("cell_area_um2")
    assert body[0][area_idx] == ""
    dry_idx = header.index("cell_dry_mass_pg")
    assert float(body[0][dry_idx]) == pytest.approx(88.0)


# ---------------------------------------------------------------------------
# Wide format
# ---------------------------------------------------------------------------

def test_wide_format_one_row_per_frame(tmp_path):
    s, rs = _two_frame_session_results()
    out = write_session_csv(s, rs, tmp_path / "wide.csv", layout="wide")
    header, body = _read_csv(out)
    # 2 frames → 2 rows
    assert len(body) == 2
    # Per-cell metric columns expanded for every cell id seen
    # anywhere in the results — cell 1 + cell 2 from frame 0.
    assert "cell_1_dry_mass_pg" in header
    assert "cell_2_dry_mass_pg" in header


def test_wide_format_blank_for_missing_cell_in_frame(tmp_path):
    """Frame 1 has no cells. Its row's per-cell columns must be
    blank — not 0, not None."""
    s, rs = _two_frame_session_results()
    out = write_session_csv(s, rs, tmp_path / "wide.csv", layout="wide")
    header, body = _read_csv(out)
    frame_idx = header.index("frame_index")
    cell1_dm = header.index("cell_1_dry_mass_pg")
    frame_one = next(r for r in body if r[frame_idx] == "1")
    assert frame_one[cell1_dm] == ""


def test_wide_format_dry_mass_values_match(tmp_path):
    s, rs = _two_frame_session_results()
    out = write_session_csv(s, rs, tmp_path / "wide.csv", layout="wide")
    header, body = _read_csv(out)
    frame_idx = header.index("frame_index")
    c1_dm = header.index("cell_1_dry_mass_pg")
    c2_dm = header.index("cell_2_dry_mass_pg")
    frame_zero = next(r for r in body if r[frame_idx] == "0")
    assert float(frame_zero[c1_dm]) == pytest.approx(88.5)
    assert float(frame_zero[c2_dm]) == pytest.approx(95.1)


def test_wide_format_custom_metrics(tmp_path):
    """Caller can ask for a specific metric subset — prevents the
    Excel matrix from blowing up to dozens of columns when the
    operator only cares about dry mass."""
    s, rs = _two_frame_session_results()
    out = write_session_csv(
        s, rs, tmp_path / "wide.csv",
        layout="wide", wide_metrics=("dry_mass_pg",),
    )
    header, _ = _read_csv(out)
    assert "cell_1_dry_mass_pg" in header
    # Other per-cell metrics excluded — only check cell_<id>_<metric>
    # columns; ``frame_z_mm`` (the frame-level autofocus result) is
    # always present and not part of the wide_metrics filter.
    cell_cols = [h for h in header if h.startswith("cell_")]
    assert not any(h.endswith("_z_mm") for h in cell_cols)
    assert not any(h.endswith("_area_um2") for h in cell_cols)


# ---------------------------------------------------------------------------
# Errors / edge cases
# ---------------------------------------------------------------------------

def test_unknown_layout_raises():
    s = Session.new()
    with pytest.raises(ValueError, match="unknown layout"):
        write_session_csv(s, [], "x.csv", layout="bogus")


def test_writes_to_nested_path_creates_dirs(tmp_path):
    """``out_path`` may include parent dirs that don't exist —
    write_session_csv mkdirs them."""
    s, rs = _two_frame_session_results()
    target = tmp_path / "deeper" / "still" / "out.csv"
    out = write_session_csv(s, rs, target, layout="long")
    assert out.exists()


def test_empty_results_writes_header_only(tmp_path):
    """Zero frames → CSV with header, no body. Audit-friendly even
    if a session was cancelled before any frame ran."""
    s = Session.new(operator="erik")
    out = write_session_csv(s, [], tmp_path / "empty.csv", layout="long")
    header, body = _read_csv(out)
    assert header[0] == "session_id"
    assert body == []
