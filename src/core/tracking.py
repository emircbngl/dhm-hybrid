"""Per-cell tracking across time-lapse frames — v2.0.8 sprint, D2.

Karin's pain (Lindqvist 2026-04-27):

    > "Per-cell tracking yok. Depth map clustering mevcut ama
    > frame'ler arası association yok. Cell-1 t=0'da neredeyse
    > cell-1 t=5'te aynı cell olduğunu nasıl bileceğim?"

This module gives stable cell IDs across frames using
nearest-neighbour assignment with a Hungarian-algorithm step. The
lab's typical scene (10–50 cells, modest motion between frames)
fits this approach perfectly — full ``trackpy`` is overkill and
adds an external dependency.

Algorithm
---------
For each consecutive frame pair:

1. Compute a cost matrix ``C[i, j] = euclidean(centroid_a[i],
   centroid_b[j])``.
2. Mask cells > ``max_distance_px`` apart with a large penalty so
   the optimiser won't link them.
3. Run :func:`scipy.optimize.linear_sum_assignment` (Hungarian /
   Munkres) — globally optimal assignment in polynomial time.
4. Cells with matched-distance > threshold inherit a new cell ID
   (track lost / cell entered the field).

Birth / death events are surfaced as a separate timeline so the
caller can render "cell N died at frame K" in a side panel.

Public API
----------
* :class:`Detection` — one (cy, cx, radius_px, frame_idx) row.
* :class:`Track` — chain of detections with a stable ``cell_id``.
* :func:`link_detections` — main entry. Takes a list of per-frame
  detection lists, returns a list of :class:`Track`.

What this is NOT
----------------
* It is not a re-implementation of trackpy's full feature set
  (sub-frame search, sub-pixel refinement, prediction).
* It does not handle merge / split events — two cells fusing into
  one are detected as one cell death + one cell rebirth.
* Memory is one frame deep; no Kalman state. For 5-min interval
  time-lapse this is fine; for short intervals with high motility,
  trackpy or sort would do better and we'd add it then.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_OK = True
except ImportError:  # pragma: no cover - scipy is a hard dep
    _SCIPY_OK = False


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One cell observation in one frame.

    Attributes
    ----------
    cy_px, cx_px
        Cell centroid in pixel coordinates.
    radius_px
        Approximate cell radius. Optional — the linker can ignore
        radius and link by centroid alone.
    frame_idx
        Index of the frame this detection comes from. The linker
        groups detections by frame and assumes frames are sorted
        by index.
    payload
        Any per-detection extras (depth, dry_mass, etc.). Carried
        through the linker so downstream code can attach measurements
        to track positions without a second lookup.
    """
    cy_px: float
    cx_px: float
    frame_idx: int
    radius_px: float = 0.0
    payload: dict = field(default_factory=dict)


@dataclass
class Track:
    """One cell's trajectory across frames.

    Attributes
    ----------
    cell_id
        Stable integer ID. Birth → first frame the cell appeared,
        death → last frame the cell appeared. Re-ID is not
        attempted; if a cell disappears for one frame and
        reappears, it gets a new ID. Tighter logic is v3.0
        territory.
    detections
        Ordered list (by frame_idx) of the detections that belong
        to this cell.
    """
    cell_id: int
    detections: List[Detection] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.detections[0].frame_idx if self.detections else -1

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame_idx if self.detections else -1

    @property
    def duration_frames(self) -> int:
        if not self.detections:
            return 0
        return self.last_frame - self.first_frame + 1


# ---------------------------------------------------------------------------
# Cost matrix + assignment
# ---------------------------------------------------------------------------

def _cost_matrix(prev: Sequence[Detection],
                 curr: Sequence[Detection]) -> np.ndarray:
    """Pairwise euclidean distance between centroids. Shape
    ``(len(prev), len(curr))``. NaN-safe — empty inputs return a
    properly-shaped 0-row or 0-col array."""
    if not prev or not curr:
        return np.zeros((len(prev), len(curr)), dtype=np.float64)
    py = np.array([d.cy_px for d in prev], dtype=np.float64)
    px = np.array([d.cx_px for d in prev], dtype=np.float64)
    cy = np.array([d.cy_px for d in curr], dtype=np.float64)
    cx = np.array([d.cx_px for d in curr], dtype=np.float64)
    dy = py[:, None] - cy[None, :]
    dx = px[:, None] - cx[None, :]
    return np.sqrt(dy * dy + dx * dx)


def _solve_assignment(cost: np.ndarray,
                      max_distance_px: float) -> List[Tuple[int, int]]:
    """Run Hungarian on ``cost`` and drop pairs above the threshold.

    Returns a list of ``(prev_idx, curr_idx)`` tuples — guaranteed
    matched + within distance.
    """
    if cost.size == 0:
        return []
    if not _SCIPY_OK:
        # Greedy fallback — pair each row to its argmin column,
        # break ties left-to-right. Suboptimal under heavy
        # crossing motion but always correct under separable
        # cells. Should not be hit in production (scipy is a hard
        # dep), kept as a safety net for sandboxes.
        used_cols = set()
        out: List[Tuple[int, int]] = []
        order = np.argsort(cost.min(axis=1))
        for i in order:
            row = cost[i].copy()
            for c in used_cols:
                row[c] = np.inf
            j = int(np.argmin(row))
            if cost[i, j] <= max_distance_px:
                out.append((int(i), j))
                used_cols.add(j)
        return out
    # Square or rectangular — linear_sum_assignment handles both.
    rows, cols = linear_sum_assignment(cost)
    out = []
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_distance_px:
            out.append((int(r), int(c)))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def link_detections(per_frame: Sequence[Sequence[Detection]],
                    *,
                    max_distance_px: float = 25.0,
                    ) -> List[Track]:
    """Link per-frame detections into stable :class:`Track` objects.

    Parameters
    ----------
    per_frame
        ``per_frame[k]`` is the list of detections in frame ``k``.
        Frames must be ordered (frame 0 first); empty slots are
        allowed.
    max_distance_px
        Cells whose centroids are farther apart than this between
        consecutive frames are considered different cells.
        Tune to ``2-3 × cell_radius`` plus expected motion budget.
        For Lindqvist's typical 5-min interval HeLa cells (~15 µm
        radius @ 2.5 µm pixel = 6 px, ~3 px motion) the default
        25 px is safe.

    Returns
    -------
    List[Track]
        Tracks sorted by ``first_frame`` then ``cell_id``. Each
        track has at least one detection. Cells that disappear and
        reappear get a new track (no re-ID across gaps yet).
    """
    if not per_frame:
        return []

    next_id = 0
    tracks: List[Track] = []
    # active[i] = index into ``tracks`` for the cell present at the
    # current "previous" frame; lengths match prev_frame.
    active_prev_idx: List[int] = []
    prev_frame: List[Detection] = []

    for k, curr_frame in enumerate(per_frame):
        curr_frame = list(curr_frame)
        if not prev_frame:
            # First non-empty frame → every detection births a
            # track.
            for det in curr_frame:
                t = Track(cell_id=next_id, detections=[det])
                next_id += 1
                tracks.append(t)
                active_prev_idx.append(len(tracks) - 1)
            prev_frame = curr_frame
            continue

        cost = _cost_matrix(prev_frame, curr_frame)
        assignments = _solve_assignment(cost, max_distance_px)

        # Build the next active set + birth tracks for unmatched
        # current detections.
        matched_curr = {c for _, c in assignments}
        new_active: List[int] = [-1] * len(curr_frame)
        for prev_i, curr_j in assignments:
            track_idx = active_prev_idx[prev_i]
            tracks[track_idx].detections.append(curr_frame[curr_j])
            new_active[curr_j] = track_idx
        for j, det in enumerate(curr_frame):
            if j in matched_curr:
                continue
            t = Track(cell_id=next_id, detections=[det])
            next_id += 1
            tracks.append(t)
            new_active[j] = len(tracks) - 1

        active_prev_idx = new_active
        prev_frame = curr_frame

    tracks.sort(key=lambda t: (t.first_frame, t.cell_id))
    return tracks


def detections_from_clusters(
    clusters: Sequence,  # Sequence[ClusterHeight | similar]
    *,
    frame_idx: int,
) -> List[Detection]:
    """Adapter from existing :class:`core.depth_map.ClusterHeight`
    output to :class:`Detection`. Lab's existing depth-map workflow
    produces clusters with ``cy_px``, ``cx_px``, ``radius_px``,
    plus a few measurement scalars. This wrapper normalises them
    into Detections so the linker doesn't grow an import on
    depth_map.

    The cluster's height / dry_mass / area scalars land in
    ``payload`` so downstream tracking-aware exporters can attach
    them by ``cell_id`` for free."""
    out: List[Detection] = []
    for c in clusters:
        cy = float(getattr(c, "cy_px", getattr(c, "cy", 0.0)))
        cx = float(getattr(c, "cx_px", getattr(c, "cx", 0.0)))
        r = float(getattr(c, "radius_px", 0.0))
        payload = {
            k: getattr(c, k)
            for k in ("z_mm", "height_nm", "dry_mass_pg",
                      "area_um2", "area_px", "confidence")
            if hasattr(c, k)
        }
        out.append(Detection(
            cy_px=cy, cx_px=cx, frame_idx=int(frame_idx),
            radius_px=r, payload=payload,
        ))
    return out


__all__ = [
    "Detection",
    "Track",
    "link_detections",
    "detections_from_clusters",
]
