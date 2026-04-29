"""Sample-plane map: where cells are, in stage coordinates.

Built by sweeping the XY stage over a grid, capturing a hologram at
each position, segmenting cells, and recording each cell's stage XY,
in-frame centroid, and a few characterising scalars (area, dry mass).

The map is the data backbone for two consumers:
  * the AI ``goto_cell`` tool — move the stage to a previously-mapped
    cell so it lands under the objective again;
  * a future GUI overlay (operator builds it; we just supply the data)
    that draws the cells on a 2-D plane and lets the user click to
    teleport.

JSON-persistable so the map survives a session restart. Pure stdlib +
dataclasses; no numpy in the persisted form (though callers can pass
numpy scalars — we coerce at write time).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class CellLocation:
    """One cell, anchored in stage coordinates.

    ``stage_x_mm`` / ``stage_y_mm`` is the stage position the operator
    would dial in to recentre the cell under the objective. The stage Z
    is recorded too because focus drifts laterally in non-flat samples
    (a tilted slide, an uneven gel) — knowing the per-cell focus z
    short-circuits a full re-search when the operator revisits.

    ``in_frame_y_px`` / ``in_frame_x_px`` is the cell's pixel centroid
    *within* the hologram captured at this stage position; (0, 0) is
    top-left, matching numpy convention. Useful when zooming the GUI
    map onto a cell and overlaying the actual reconstruction.
    """
    cell_id: int
    stage_x_mm: float
    stage_y_mm: float
    stage_z_mm: float
    in_frame_y_px: int
    in_frame_x_px: int
    area_um2: float = 0.0
    dry_mass_pg: float = 0.0
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleMap:
    """All cells found during one mapping sweep.

    A map is *replaced* on each new sweep, not extended — the operator
    typically wants today's snapshot, not a cumulative history. If
    extension is wanted later we'll add a ``merge`` method; the
    contract today is "one sweep, one map".

    ``grid_extent`` records the XY bounds the sweep covered so the GUI
    can draw the empty area too (where no cells were detected). Keeping
    it stops the operator from staring at a sparse scatter and
    wondering "did the map miss something on the right edge, or was
    the sweep range narrower than I thought?".
    """
    cells: list[CellLocation] = field(default_factory=list)
    grid_extent_mm: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    grid_step_mm: float = 0.0
    created_at: str = ""
    sample_id: str = ""

    # ---- mutation ---------------------------------------------------------

    def reset(self) -> None:
        self.cells = []
        self.grid_extent_mm = (0.0, 0.0, 0.0, 0.0)
        self.grid_step_mm = 0.0
        self.created_at = ""
        self.sample_id = ""

    def add(self, cell: CellLocation) -> None:
        self.cells.append(cell)

    def stamp(self, *, x_min: float, x_max: float, y_min: float, y_max: float,
              step_mm: float, sample_id: str = "") -> None:
        self.grid_extent_mm = (
            float(x_min), float(x_max), float(y_min), float(y_max),
        )
        self.grid_step_mm = float(step_mm)
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.sample_id = sample_id

    # ---- queries ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cells)

    def by_id(self, cell_id: int) -> Optional[CellLocation]:
        for c in self.cells:
            if c.cell_id == cell_id:
                return c
        return None

    def nearest(self, x_mm: float, y_mm: float) -> Optional[CellLocation]:
        """Return the cell closest in XY to (x_mm, y_mm), or None if empty.

        Useful for "go to the cell nearest the current stage position"
        without the LLM having to enumerate ids.
        """
        if not self.cells:
            return None
        return min(
            self.cells,
            key=lambda c: (c.stage_x_mm - x_mm) ** 2 + (c.stage_y_mm - y_mm) ** 2,
        )

    # ---- serialization ----------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "cells": [c.as_dict() for c in self.cells],
            "grid_extent_mm": list(self.grid_extent_mm),
            "grid_step_mm": self.grid_step_mm,
            "created_at": self.created_at,
            "sample_id": self.sample_id,
        }

    def summary(self) -> dict:
        """Compact dict for the AI panel / status bar.

        Returns counts + extent + step but elides the full cell list
        (which can be hundreds of rows). Callers that want the full
        list use ``as_dict``.
        """
        return {
            "count": len(self.cells),
            "grid_extent_mm": list(self.grid_extent_mm),
            "grid_step_mm": self.grid_step_mm,
            "created_at": self.created_at,
            "sample_id": self.sample_id,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "SampleMap":
        cells_raw = raw.get("cells") or []
        cells = []
        for r in cells_raw:
            cells.append(CellLocation(
                cell_id=int(r.get("cell_id", 0)),
                stage_x_mm=float(r.get("stage_x_mm", 0.0)),
                stage_y_mm=float(r.get("stage_y_mm", 0.0)),
                stage_z_mm=float(r.get("stage_z_mm", 0.0)),
                in_frame_y_px=int(r.get("in_frame_y_px", 0)),
                in_frame_x_px=int(r.get("in_frame_x_px", 0)),
                area_um2=float(r.get("area_um2", 0.0)),
                dry_mass_pg=float(r.get("dry_mass_pg", 0.0)),
                notes=str(r.get("notes", "")),
            ))
        ext = raw.get("grid_extent_mm") or [0.0, 0.0, 0.0, 0.0]
        ext_t = (float(ext[0]), float(ext[1]), float(ext[2]), float(ext[3]))
        m = cls(
            cells=cells,
            grid_extent_mm=ext_t,
            grid_step_mm=float(raw.get("grid_step_mm", 0.0)),
            created_at=str(raw.get("created_at", "")),
            sample_id=str(raw.get("sample_id", "")),
        )
        return m

    # ---- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomic JSON write. Caller is responsible for the dir existing.

        We write to a sibling ``.tmp`` file then ``os.replace`` — this is
        the same crash-safe pattern the audit log uses.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=2)
        # ``os.replace`` is atomic on POSIX + Windows for files in the
        # same directory; the path resolution above guarantees that.
        import os
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "SampleMap":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)


__all__ = [
    "CellLocation",
    "SampleMap",
]
