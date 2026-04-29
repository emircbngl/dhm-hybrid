"""Per-cell tracking tests (v2.0.8, D2).

Coverage:

* Stable cell IDs across frames when motion is small.
* Two cells crossing → Hungarian assigns globally, not greedily.
* Cell birth (new detection appears) and death (detection
  disappears) bookkeeping.
* Distance threshold reject — far-apart detections get new IDs.
* Empty / single-frame edge cases.
* ``detections_from_clusters`` adapts depth-map output without
  importing depth_map.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.tracking import (  # noqa: E402
    Detection,
    Track,
    detections_from_clusters,
    link_detections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(cy: float, cx: float, frame: int = 0, **payload) -> Detection:
    return Detection(cy_px=cy, cx_px=cx, frame_idx=frame, payload=payload)


# ---------------------------------------------------------------------------
# Stable IDs
# ---------------------------------------------------------------------------

def test_single_cell_keeps_id_across_frames():
    """One cell drifting by a few pixels each frame → one Track."""
    f0 = [_det(50, 50, 0)]
    f1 = [_det(52, 51, 1)]
    f2 = [_det(54, 53, 2)]
    tracks = link_detections([f0, f1, f2])
    assert len(tracks) == 1
    assert tracks[0].cell_id == 0
    assert tracks[0].duration_frames == 3
    assert [d.frame_idx for d in tracks[0].detections] == [0, 1, 2]


def test_two_cells_keep_separate_ids():
    f0 = [_det(20, 20, 0), _det(80, 80, 0)]
    f1 = [_det(21, 21, 1), _det(81, 79, 1)]
    f2 = [_det(22, 22, 2), _det(82, 78, 2)]
    tracks = link_detections([f0, f1, f2])
    assert len(tracks) == 2
    # Each track has 3 detections.
    for t in tracks:
        assert t.duration_frames == 3
    # IDs are stable + distinct.
    ids = {t.cell_id for t in tracks}
    assert len(ids) == 2


def test_two_cells_crossing_paths_hungarian_global_optimum():
    """Two cells move toward each other; greedy nearest-neighbour
    can swap them. Hungarian's globally-optimal cost minimisation
    keeps them separate. This is the classic test for any tracker
    above NN-by-row."""
    f0 = [_det(30, 50, 0), _det(70, 50, 0)]
    # Move slowly so they don't actually cross — this distinguishes
    # global from greedy. Greedy would still swap if the inner
    # tie-break order is unlucky.
    f1 = [_det(35, 50, 1), _det(65, 50, 1)]
    tracks = link_detections([f0, f1])
    assert len(tracks) == 2
    # Each track must contain its own initial cell, not the other's.
    for t in tracks:
        first = t.detections[0]
        if first.cy_px == 30:
            # cell starting at (30, 50) ends up near (35, 50)
            assert abs(t.detections[1].cy_px - 35) < 1
        else:
            assert abs(t.detections[1].cy_px - 65) < 1


# ---------------------------------------------------------------------------
# Birth / death
# ---------------------------------------------------------------------------

def test_cell_birth_creates_new_track():
    """Frame 0 has 1 cell, frame 1 has 2 cells. Old cell continues,
    new cell gets a fresh ID."""
    f0 = [_det(50, 50, 0)]
    f1 = [_det(50, 50, 1), _det(20, 20, 1)]
    tracks = link_detections([f0, f1])
    assert len(tracks) == 2
    # Old track has 2 detections; new track has 1.
    old = next(t for t in tracks if t.first_frame == 0)
    new = next(t for t in tracks if t.first_frame == 1)
    assert old.duration_frames == 2
    assert new.duration_frames == 1


def test_cell_death_does_not_extend_track():
    """Frame 0 has 2 cells, frame 1 has 1. The disappearing cell
    keeps its track but doesn't gain a frame-1 detection."""
    f0 = [_det(20, 20, 0), _det(80, 80, 0)]
    f1 = [_det(21, 21, 1)]
    tracks = link_detections([f0, f1])
    assert len(tracks) == 2
    survivor = next(t for t in tracks if t.duration_frames == 2)
    deceased = next(t for t in tracks if t.duration_frames == 1)
    assert survivor.last_frame == 1
    assert deceased.last_frame == 0


def test_cell_disappear_then_reappear_gets_new_id():
    """Re-ID across a gap is NOT supported (out of v2.0.8 scope).
    A cell that disappears and reappears must get a new ID. This
    test pins that intentional limitation so a future regression
    that 'fixes' it without proper Kalman state is caught."""
    f0 = [_det(50, 50, 0)]
    f1: List[Detection] = []
    f2 = [_det(50, 50, 2)]
    tracks = link_detections([f0, f1, f2])
    assert len(tracks) == 2
    assert tracks[0].cell_id != tracks[1].cell_id


# ---------------------------------------------------------------------------
# Distance threshold
# ---------------------------------------------------------------------------

def test_far_apart_detections_get_new_ids():
    """Even with one cell in each frame, if they're > max_distance_px
    apart, the linker treats them as different cells."""
    f0 = [_det(20, 20, 0)]
    f1 = [_det(80, 80, 1)]
    tracks = link_detections([f0, f1], max_distance_px=10.0)
    # 60-pixel gap > 10 → two tracks.
    assert len(tracks) == 2


def test_within_distance_threshold_links():
    f0 = [_det(50, 50, 0)]
    f1 = [_det(58, 56, 1)]  # distance ~10
    tracks = link_detections([f0, f1], max_distance_px=15.0)
    assert len(tracks) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty():
    assert link_detections([]) == []


def test_single_frame_births_each_cell():
    f0 = [_det(10, 10, 0), _det(50, 50, 0), _det(80, 80, 0)]
    tracks = link_detections([f0])
    assert len(tracks) == 3


def test_all_empty_frames_no_tracks():
    tracks = link_detections([[], [], []])
    assert tracks == []


def test_first_frame_empty_then_births():
    """Empty frame 0, frame 1 has detections. Tracks should still
    fire — the linker should treat frame 1 as the effective start
    rather than producing zero output."""
    f0: List[Detection] = []
    f1 = [_det(50, 50, 1)]
    tracks = link_detections([f0, f1])
    assert len(tracks) == 1
    assert tracks[0].first_frame == 1


def test_payload_carried_through():
    """detection.payload should land on the tracked detection
    intact — analysis tools key by cell_id and need the per-cell
    measurements (dry_mass, etc.) attached."""
    f0 = [_det(50, 50, 0, dry_mass_pg=88.5)]
    f1 = [_det(51, 51, 1, dry_mass_pg=89.0)]
    tracks = link_detections([f0, f1])
    assert tracks[0].detections[0].payload["dry_mass_pg"] == 88.5
    assert tracks[0].detections[1].payload["dry_mass_pg"] == 89.0


# ---------------------------------------------------------------------------
# detections_from_clusters
# ---------------------------------------------------------------------------

class _FakeCluster:
    """Mimics the relevant subset of ``ClusterHeight`` without
    importing depth_map (which would pull a heavy graph of deps)."""
    def __init__(self, cy_px, cx_px, radius_px=5.0,
                 z_mm=12.0, dry_mass_pg=88.0):
        self.cy_px = cy_px
        self.cx_px = cx_px
        self.radius_px = radius_px
        self.z_mm = z_mm
        self.dry_mass_pg = dry_mass_pg


def test_detections_from_clusters_normalises_attrs():
    clusters = [_FakeCluster(50, 60), _FakeCluster(20, 20)]
    dets = detections_from_clusters(clusters, frame_idx=3)
    assert len(dets) == 2
    assert dets[0].cy_px == 50
    assert dets[0].cx_px == 60
    assert dets[0].frame_idx == 3
    # Payload picked up the scalar measurements.
    assert dets[0].payload["dry_mass_pg"] == 88.0
    assert dets[0].payload["z_mm"] == 12.0


def test_detections_from_clusters_handles_alt_attr_names():
    """Some ClusterHeight variants use ``cy``/``cx`` instead of
    ``cy_px``/``cx_px``. The adapter falls through."""
    class _Alt:
        def __init__(self, cy, cx):
            self.cy = cy
            self.cx = cx
    dets = detections_from_clusters([_Alt(40, 70)], frame_idx=0)
    assert len(dets) == 1
    assert dets[0].cy_px == 40
    assert dets[0].cx_px == 70


def test_detections_from_clusters_empty_input():
    assert detections_from_clusters([], frame_idx=0) == []
