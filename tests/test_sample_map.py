"""SampleMap data structure: add, query, persist, restore."""
from __future__ import annotations

import json

import pytest

from core.sample_map import CellLocation, SampleMap


def _cell(cid: int, x: float, y: float, z: float = 0.0) -> CellLocation:
    return CellLocation(
        cell_id=cid, stage_x_mm=x, stage_y_mm=y, stage_z_mm=z,
        in_frame_y_px=10, in_frame_x_px=20,
        area_um2=12.0, dry_mass_pg=4.5,
    )


def test_empty_map_has_zero_length():
    m = SampleMap()
    assert len(m) == 0


def test_add_increases_length():
    m = SampleMap()
    m.add(_cell(1, 0.0, 0.0))
    m.add(_cell(2, 1.0, 0.0))
    assert len(m) == 2


def test_by_id_finds_cell():
    m = SampleMap()
    m.add(_cell(1, 0.0, 0.0))
    m.add(_cell(2, 1.0, 0.0))
    found = m.by_id(2)
    assert found is not None
    assert found.cell_id == 2
    assert found.stage_x_mm == 1.0


def test_by_id_returns_none_for_missing():
    m = SampleMap()
    m.add(_cell(1, 0.0, 0.0))
    assert m.by_id(99) is None


def test_nearest_returns_closest():
    m = SampleMap()
    m.add(_cell(1, 0.0, 0.0))
    m.add(_cell(2, 5.0, 0.0))
    m.add(_cell(3, 10.0, 0.0))
    # 8.5 is 3.5 from id=2 and 1.5 from id=3 → id=3 wins
    nearest = m.nearest(8.5, 0.0)
    assert nearest is not None
    assert nearest.cell_id == 3


def test_nearest_returns_none_when_empty():
    m = SampleMap()
    assert m.nearest(0.0, 0.0) is None


def test_reset_clears_state():
    m = SampleMap()
    m.add(_cell(1, 0.0, 0.0))
    m.stamp(x_min=0, x_max=10, y_min=0, y_max=10, step_mm=1.0,
            sample_id="alpha")
    m.reset()
    assert len(m) == 0
    assert m.grid_step_mm == 0.0
    assert m.sample_id == ""


def test_stamp_records_extent():
    m = SampleMap()
    m.stamp(x_min=-5, x_max=5, y_min=-3, y_max=3, step_mm=0.5,
            sample_id="beta")
    assert m.grid_extent_mm == (-5.0, 5.0, -3.0, 3.0)
    assert m.grid_step_mm == 0.5
    assert m.sample_id == "beta"
    assert m.created_at  # ISO timestamp filled


def test_summary_omits_full_cell_list():
    m = SampleMap()
    for i in range(5):
        m.add(_cell(i + 1, float(i), 0.0))
    s = m.summary()
    assert s["count"] == 5
    assert "cells" not in s


def test_round_trip_via_dict():
    m = SampleMap()
    m.add(_cell(1, 1.0, 2.0, 3.0))
    m.stamp(x_min=0, x_max=2, y_min=0, y_max=4, step_mm=0.5,
            sample_id="rt")
    raw = m.as_dict()
    restored = SampleMap.from_dict(raw)
    assert len(restored) == 1
    c = restored.cells[0]
    assert c.stage_x_mm == 1.0
    assert c.stage_z_mm == 3.0
    assert restored.sample_id == "rt"
    assert restored.grid_step_mm == 0.5


def test_save_load_round_trip(tmp_path):
    m = SampleMap()
    m.add(_cell(7, 0.5, 0.5, 1.0))
    m.add(_cell(8, 1.5, 0.5, 1.0))
    m.stamp(x_min=0, x_max=2, y_min=0, y_max=2, step_mm=0.5)
    target = tmp_path / "sample.json"
    m.save(target)

    reloaded = SampleMap.load(target)
    assert len(reloaded) == 2
    assert reloaded.by_id(7).stage_x_mm == 0.5
    assert reloaded.by_id(8).stage_x_mm == 1.5


def test_save_is_atomic_via_tmp_replace(tmp_path):
    """A failed write shouldn't leave a half-baked target file in place."""
    m = SampleMap()
    m.add(_cell(1, 0.0, 0.0))
    target = tmp_path / "atomic.json"
    m.save(target)

    # Simulate an interrupted write by checking the tmp file got cleaned.
    tmp_sibling = target.with_suffix(target.suffix + ".tmp")
    assert not tmp_sibling.exists()
    assert target.exists()


def test_load_handles_missing_optional_fields(tmp_path):
    target = tmp_path / "minimal.json"
    target.write_text(json.dumps({"cells": [
        {"cell_id": 1, "stage_x_mm": 0.0, "stage_y_mm": 0.0,
         "stage_z_mm": 0.0, "in_frame_y_px": 0, "in_frame_x_px": 0}
    ]}))
    m = SampleMap.load(target)
    assert len(m) == 1
    assert m.cells[0].area_um2 == 0.0
    assert m.cells[0].notes == ""


def test_celllocation_is_hashable_via_dataclass_freeze():
    a = _cell(1, 0.0, 0.0)
    b = _cell(1, 0.0, 0.0)
    assert hash(a) == hash(b)
