"""Session-level CSV export — v2.0.7 sprint, T2.

Karin's pain point (Lindqvist 2026-04-27):

    > "PNG'ler güzel ama dry mass time-series çıkmıyor. Ben 30
    > hücrenin 144 zaman noktasındaki dry mass'ini istiyorum —
    > CSV, hücre × zaman matrisi. Şu an her hologramı tek tek
    > işleyip Excel'de elle birleştiriyorum. Üç gün."

This module is the answer. Two layouts depending on downstream
analysis:

* **Long format** (default) — one row per ``(frame, cell)`` pair.
  Tidy: friendly to pandas, ggplot, seaborn. Required when cell
  IDs are not stable across frames (early v2.0.7 — full tracking
  in v2.0.8).

* **Wide format** — one row per frame, columns ``cell_<id>_<metric>``.
  Friendly to Excel pivots + plotting; fails when cell IDs aren't
  globally consistent (a frame where cell 7 was missed leaves a
  blank slot — that's by design).

The exporter is decoupled from the CLI / pipeline: callers pass a
:class:`Session` plus a list of :class:`FrameResult` (one per frame).
The CLI runner produces ``FrameResult`` objects; tests build them
synthetically. This separation lets the lab post-process old
results directly with ``write_session_csv(session, results, ...)``
without re-running pipelines.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .session import HologramFrame, Session


# ---------------------------------------------------------------------------
# Per-frame result payload
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellMeasurement:
    """One cell's measurements at one frame.

    Optional fields (None) mean the pipeline didn't compute that
    metric for this cell — e.g. ``area_um2`` is missing when no
    segmentation ran. The CSV writer leaves the cell blank rather
    than guessing.
    """
    cell_id: int
    cy_px: Optional[float] = None
    cx_px: Optional[float] = None
    z_mm: Optional[float] = None
    dry_mass_pg: Optional[float] = None
    area_um2: Optional[float] = None
    height_nm: Optional[float] = None


@dataclass
class FrameResult:
    """All compute outputs from one hologram in a session run.

    ``frame`` is the input descriptor (path, timestamp, index).
    ``cells`` is the list of cell measurements. ``z_mm`` /
    ``runtime_ms`` / ``error`` are scalar fields the runner fills
    in even when no cells were segmented (autofocus on a sample
    with no objects, or a frame that errored mid-pipeline).
    """
    frame: HologramFrame
    z_mm: Optional[float] = None
    runtime_ms: float = 0.0
    error: Optional[str] = None
    cells: List[CellMeasurement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Long format
# ---------------------------------------------------------------------------

_LONG_COLUMNS = [
    "session_id",
    "operator",
    "sample_id",
    "frame_index",
    "frame_path",
    "timestamp_s",
    "z_mm_frame",
    "cell_id",
    "cell_cy_px",
    "cell_cx_px",
    "cell_z_mm",
    "cell_dry_mass_pg",
    "cell_area_um2",
    "cell_height_nm",
    "frame_runtime_ms",
    "frame_error",
]


def _row_long(session: Session, fr: FrameResult,
              cell: Optional[CellMeasurement]) -> List:
    """One row of the long-format CSV. ``cell=None`` produces a
    cell-less row — used when a frame finished without segmenting
    any cells (autofocus-only) or errored out (Karin still wants
    that frame in the CSV with ``frame_error`` populated)."""
    return [
        session.id,
        session.operator,
        session.sample_id,
        fr.frame.index,
        fr.frame.path,
        fr.frame.timestamp_s,
        _fmt(fr.z_mm),
        _fmt(cell.cell_id) if cell else "",
        _fmt(cell.cy_px) if cell else "",
        _fmt(cell.cx_px) if cell else "",
        _fmt(cell.z_mm) if cell else "",
        _fmt(cell.dry_mass_pg) if cell else "",
        _fmt(cell.area_um2) if cell else "",
        _fmt(cell.height_nm) if cell else "",
        _fmt(fr.runtime_ms),
        fr.error or "",
    ]


def _fmt(v) -> str:
    """Empty string for ``None`` (so Excel shows blank, not 'None'),
    pass-through otherwise. Floats get default repr to keep precision
    — caller can post-process for display."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _all_cell_ids(results: Sequence[FrameResult]) -> List[int]:
    """Sorted unique cell IDs across the whole results list. Wide
    CSV's column set."""
    ids = set()
    for r in results:
        for c in r.cells:
            ids.add(int(c.cell_id))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Wide format
# ---------------------------------------------------------------------------

_WIDE_BASE_COLS = [
    "session_id",
    "operator",
    "sample_id",
    "frame_index",
    "frame_path",
    "timestamp_s",
    "frame_z_mm",
    "frame_runtime_ms",
    "frame_error",
]


def _wide_columns(cell_ids: List[int],
                  metrics: Sequence[str]) -> List[str]:
    """Header for wide format. One block per cell id × metric."""
    cols = list(_WIDE_BASE_COLS)
    for cid in cell_ids:
        for m in metrics:
            cols.append(f"cell_{cid}_{m}")
    return cols


def _wide_row(session: Session, fr: FrameResult,
              cell_ids: List[int], metrics: Sequence[str]) -> List:
    base = [
        session.id,
        session.operator,
        session.sample_id,
        fr.frame.index,
        fr.frame.path,
        fr.frame.timestamp_s,
        _fmt(fr.z_mm),
        _fmt(fr.runtime_ms),
        fr.error or "",
    ]
    by_id = {int(c.cell_id): c for c in fr.cells}
    for cid in cell_ids:
        c = by_id.get(cid)
        for m in metrics:
            base.append(_fmt(getattr(c, m, None)) if c else "")
    return base


# Default wide-format metrics — covers the lab's day-1 needs
# (dry mass over time, plus axial position per cell). Caller can
# override with their own ``metrics`` arg.
DEFAULT_WIDE_METRICS = ("dry_mass_pg", "z_mm", "area_um2", "height_nm")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_session_csv(
    session: Session,
    results: Sequence[FrameResult],
    out_path: Path | str,
    *,
    layout: str = "long",
    wide_metrics: Sequence[str] = DEFAULT_WIDE_METRICS,
) -> Path:
    """Persist session results to ``out_path``.

    Parameters
    ----------
    session
        Source manifest. Used for global metadata in every row
        (session_id, operator, sample_id).
    results
        One :class:`FrameResult` per processed frame. Order matters
        — written in the order given, no re-sorting. The caller
        decides whether to sort by ``frame.index`` or by timestamp.
    out_path
        Target CSV path.
    layout
        ``"long"`` (default — tidy) or ``"wide"``.
    wide_metrics
        Metrics to expand into per-cell columns when ``layout='wide'``.
        Ignored for long format.

    Returns
    -------
    Path
        The output path (absolute).
    """
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if layout == "long":
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(_LONG_COLUMNS)
            for fr in results:
                if not fr.cells:
                    # Frame still gets a row — Karin wants to see
                    # the autofocus result + error message even
                    # when nothing segmented.
                    w.writerow(_row_long(session, fr, cell=None))
                    continue
                for c in fr.cells:
                    w.writerow(_row_long(session, fr, c))
        return out

    if layout == "wide":
        cell_ids = _all_cell_ids(results)
        cols = _wide_columns(cell_ids, wide_metrics)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for fr in results:
                w.writerow(
                    _wide_row(session, fr, cell_ids, wide_metrics),
                )
        return out

    raise ValueError(f"unknown layout={layout!r} "
                     f"(must be 'long' or 'wide')")


__all__ = [
    "CellMeasurement",
    "FrameResult",
    "DEFAULT_WIDE_METRICS",
    "write_session_csv",
]
