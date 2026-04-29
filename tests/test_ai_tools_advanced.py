"""Advanced AI tools: stage focus search, sample mapping, time-lapse.

Uses the same scaffolding as ``test_ai_tools.py`` plus a varying-Z
sharpness mock so we can verify ``stage_focus_search`` actually picks
the peak.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.ai.protocol import ToolCall
from core.ai.tool_impls import build_tool_registry
from core.ai.tools import ToolContext
from core.sample_map import SampleMap


# ---------------------------------------------------------------------------
# Test scaffolding (slimmer copy of test_ai_tools — readable in isolation)
# ---------------------------------------------------------------------------

class _Audit:
    def __init__(self):
        self.records = []

    def record(self, action, params, result_summary=None):
        self.records.append((action, params, result_summary))


class _ErrorCenter:
    def __init__(self):
        self.events = []

    def error(self, **kwargs):
        self.events.append(kwargs)


class _Settings:
    restrict_to_home = False
    audit_redact_for_llm = True


class _Snap:
    def __init__(self, **kw):
        self.loaded_path = kw.get("loaded_path")
        self.loaded_shape = kw.get("loaded_shape")
        self.loaded_dtype = kw.get("loaded_dtype")
        self.recon_params = {}
        self.autofocus_params = {}
        self.qpi_params = {}
        self.last_recon_summary = None
        self.last_af_summary = None
        self.last_qpi_summary = None
        self.last_depth_summary = None
        self.stage_position_mm = (0.0, 0.0, 0.0)
        self.audit_tail = ()

    def as_dict(self):
        return {"loaded_path": self.loaded_path}


def _make_ctx(*, sample_map=None, sharpness_fn=None,
              capture_summaries=None, capture_summary=None):
    from core.stage import MockStage

    sm = sample_map if sample_map is not None else SampleMap()
    if sharpness_fn is None:
        # Default: constant sharpness — every Z step looks equally sharp.
        sharpness_fn = lambda f: 1.0

    summaries_iter = iter(capture_summaries or [])

    def cap_proc(args):
        try:
            return next(summaries_iter)
        except StopIteration:
            return capture_summary or {"phase_std": 0.0, "cells": []}

    return ToolContext(
        state=lambda: _Snap(),
        last_recon_summary=lambda: None,
        last_af_summary=lambda: None,
        last_qpi_summary=lambda: None,
        last_depth_summary=lambda: None,
        audit_tail=lambda lim: [],
        load_hologram=lambda p: {"path": p},
        set_recon_param=lambda a: {"updated": a},
        invoke_recon=lambda a: {"submitted": True},
        invoke_autofocus=lambda a: {"submitted": True, "best_z_mm": 0.42},
        invoke_qpi=lambda a: {"submitted": True},
        invoke_depth_map=lambda a: {"submitted": True},
        invoke_find_focus_candidates=lambda a: {"submitted": True},
        stage=MockStage(),
        capture_frame=lambda: np.ones((32, 32), dtype=np.float32),
        sample_map=sm,
        measure_sharpness=sharpness_fn,
        persist_sample_map=lambda: "/tmp/pretend.json",
        capture_and_process=cap_proc,
        audit=_Audit(),
        error_center=_ErrorCenter(),
        confirm=lambda n, a: True,
        settings=_Settings(),
    )


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id="x", name=name, arguments_json=json.dumps(args))


# ---------------------------------------------------------------------------
# stage_focus_search
# ---------------------------------------------------------------------------

def test_stage_focus_search_picks_sharpness_peak():
    """Build a sharpness function that peaks at z=2 mm and verify
    the tool lands the stage there."""
    target_z = 2.0
    state = {"calls": 0}

    def varying_sharpness(_frame):
        state["calls"] += 1
        # Use the stage's Z to compute distance from target. We need
        # the stage instance — we'll close over it through the ctx
        # by setting it from outside.
        return -(state["last_z"] - target_z) ** 2

    state["last_z"] = 0.0
    ctx = _make_ctx(sharpness_fn=varying_sharpness)

    # Wrap stage moves so we can record which Z each capture sees.
    real_move_abs = ctx.stage.move_absolute

    def tracked_move(x, y, z):
        state["last_z"] = float(z)
        return real_move_abs(x, y, z)

    ctx.stage.move_absolute = tracked_move  # type: ignore[method-assign]

    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=10.0, step_mm=0.5))
    assert out.ok is True
    # Best z should be near target_z within one step.
    assert abs(out.payload["best_z_mm"] - target_z) <= 0.5
    assert out.payload["n_steps_evaluated"] >= 3
    assert out.payload["stage_position"]["z_mm"] == out.payload["best_z_mm"]


def test_stage_focus_search_rejects_bad_step():
    """step bigger than range is nonsense; handler rejects after schema."""
    ctx = _make_ctx()
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=2.0, step_mm=3.0))
    assert out.ok is False
    # Handler returns "step_mm=... invalid for range ..." once schema passes.
    assert "invalid" in out.payload["error"].lower()


def test_stage_focus_search_refine_calls_digital_af():
    ctx = _make_ctx()
    af_called: list[dict] = []

    def fake_af(args):
        af_called.append(dict(args))
        return {"submitted": True, "best_z_mm": 12.4}

    ctx.invoke_autofocus = fake_af  # type: ignore[assignment]

    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=2.0, step_mm=0.5,
                                refine_with_digital_af=True))
    assert out.ok is True
    assert len(af_called) == 1
    assert "digital_af" in out.payload


def test_stage_focus_search_returns_error_when_capture_returns_none():
    ctx = _make_ctx()
    ctx.capture_frame = lambda: None  # type: ignore[assignment]
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=2.0, step_mm=0.5))
    assert out.ok is False
    assert "no frame" in out.payload["error"]


# ---- Cell-aware mode --------------------------------------------------------


def _cell_frame(shape=(64, 64), cells=((20, 30, 5),)):
    """Synthetic frame: low-amplitude background + bright disks at the
    given (cy, cx, radius) positions. Triggers the >mean+std mask."""
    img = np.full(shape, 0.05, dtype=np.float32)
    yy, xx = np.indices(shape)
    for cy, cx, r in cells:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = 1.0
    return img


def _flat_frame(shape=(64, 64)):
    """Uniform grey — no cells should trigger."""
    return np.full(shape, 0.5, dtype=np.float32)


def test_stage_focus_search_auto_picks_cell_aware_when_cells_present():
    ctx = _make_ctx()
    ctx.capture_frame = lambda: _cell_frame()  # type: ignore[assignment]
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=2.0, step_mm=0.5,
                                mode="auto"))
    assert out.ok is True
    assert out.payload["mode_used"] == "cell_aware"
    assert out.payload["mask_pixels_at_probe"] > 0


def test_stage_focus_search_auto_falls_back_to_full_frame_when_uniform():
    ctx = _make_ctx()
    ctx.capture_frame = lambda: _flat_frame()  # type: ignore[assignment]
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=2.0, step_mm=0.5,
                                mode="auto"))
    assert out.ok is True
    assert out.payload["mode_used"] == "full_frame"
    assert out.payload["mask_pixels_at_probe"] == 0


def test_stage_focus_search_cell_aware_forced_errors_on_uniform_frame():
    """Operator forces cell_aware on a calibration target → fail loud
    so they know the wrong mode was picked."""
    ctx = _make_ctx()
    ctx.capture_frame = lambda: _flat_frame()  # type: ignore[assignment]
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=2.0, step_mm=0.5,
                                mode="cell_aware"))
    assert out.ok is False
    assert "no cells detected" in out.payload["error"]


def test_stage_focus_search_full_frame_forced_uses_full_metric():
    ctx = _make_ctx()
    # Cell frame + force full_frame → mode_used must respect override.
    ctx.capture_frame = lambda: _cell_frame()  # type: ignore[assignment]
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=2.0, step_mm=0.5,
                                mode="full_frame"))
    assert out.ok is True
    assert out.payload["mode_used"] == "full_frame"


def test_stage_focus_search_unknown_mode_rejected():
    ctx = _make_ctx()
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("stage_focus_search",
                                search_range_mm=2.0, step_mm=0.5,
                                mode="random_string"))
    # JSON-Schema enum rejects this before the handler runs.
    assert out.ok is False


def test_stage_focus_search_mask_dilate_grows_mask_pixels():
    """With a single bright disk, larger dilation → strictly more mask
    pixels at probe time. Sanity-check the dilation argument actually
    propagates."""
    from core.ai.tool_impls import _detect_cell_mask
    f = _cell_frame(cells=((30, 30, 4),))
    small = _detect_cell_mask(f, dilate_px=2)
    large = _detect_cell_mask(f, dilate_px=20)
    assert small is not None and large is not None
    assert int(large.sum()) > int(small.sum())


# ---------------------------------------------------------------------------
# map_sample_grid + list_mapped_cells + goto_cell
# ---------------------------------------------------------------------------

def test_map_sample_grid_visits_every_grid_point():
    sm = SampleMap()
    visits: list[tuple[float, float]] = []

    def cap_proc(args):
        # Record the stage XY this iteration ran at (we'll read it
        # via the tool's stage interface).
        return {"cells": []}

    summaries = [{"cells": []}] * 25
    ctx = _make_ctx(sample_map=sm, capture_summaries=summaries)

    real_move = ctx.stage.move_absolute

    def tracked(x, y, z):
        visits.append((float(x), float(y)))
        return real_move(x, y, z)

    ctx.stage.move_absolute = tracked  # type: ignore[method-assign]

    r = build_tool_registry()
    out = r.dispatch(ctx, _call("map_sample_grid",
                                x_min_mm=0.0, x_max_mm=2.0,
                                y_min_mm=0.0, y_max_mm=2.0,
                                step_mm=1.0))
    assert out.ok is True
    # 3×3 grid (0, 1, 2 inclusive on both axes).
    assert out.payload["visited_points"] == 9
    assert sm.summary()["count"] == 0


def test_map_sample_grid_records_segmented_cells():
    sm = SampleMap()
    summaries = [
        {"cells": [{"centroid_y_px": 10, "centroid_x_px": 20,
                    "area_um2": 12.0, "dry_mass_pg": 3.0}]},
        {"cells": []},
        {"cells": [{"centroid_y_px": 50, "centroid_x_px": 60,
                    "area_um2": 24.0, "dry_mass_pg": 6.0}]},
        {"cells": []},
    ]
    ctx = _make_ctx(sample_map=sm, capture_summaries=summaries)

    r = build_tool_registry()
    out = r.dispatch(ctx, _call("map_sample_grid",
                                x_min_mm=0.0, x_max_mm=1.0,
                                y_min_mm=0.0, y_max_mm=1.0,
                                step_mm=1.0))
    assert out.ok is True
    assert out.payload["visited_points"] == 4
    assert out.payload["cell_count"] == 2
    assert len(sm) == 2


def test_map_sample_grid_rejects_oversized_grid():
    ctx = _make_ctx()
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("map_sample_grid",
                                x_min_mm=-100.0, x_max_mm=100.0,
                                y_min_mm=-100.0, y_max_mm=100.0,
                                step_mm=1.0))
    assert out.ok is False
    assert "too large" in out.payload["error"]


def test_map_sample_grid_skip_recon_dry_run():
    sm = SampleMap()
    ctx = _make_ctx(sample_map=sm)
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("map_sample_grid",
                                x_min_mm=0.0, x_max_mm=1.0,
                                y_min_mm=0.0, y_max_mm=1.0,
                                step_mm=0.5,
                                skip_recon=True))
    assert out.ok is True
    assert out.payload["visited_points"] == 9
    assert out.payload["cell_count"] == 0


def test_list_mapped_cells_returns_cells_and_summary():
    sm = SampleMap()
    sm.add_dummy = None  # silence type checker
    from core.sample_map import CellLocation
    sm.add(CellLocation(cell_id=1, stage_x_mm=0.0, stage_y_mm=0.0,
                        stage_z_mm=0.0, in_frame_y_px=0, in_frame_x_px=0))
    ctx = _make_ctx(sample_map=sm)
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("list_mapped_cells", limit=10))
    assert out.ok is True
    assert out.payload["summary"]["count"] == 1
    assert len(out.payload["cells"]) == 1


def test_list_mapped_cells_truncates():
    from core.sample_map import CellLocation
    sm = SampleMap()
    for i in range(10):
        sm.add(CellLocation(cell_id=i + 1, stage_x_mm=float(i),
                            stage_y_mm=0.0, stage_z_mm=0.0,
                            in_frame_y_px=0, in_frame_x_px=0))
    ctx = _make_ctx(sample_map=sm)
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("list_mapped_cells", limit=3))
    assert out.payload["truncated"] is True
    assert len(out.payload["cells"]) == 3


def test_goto_cell_by_id_moves_stage():
    from core.sample_map import CellLocation
    sm = SampleMap()
    sm.add(CellLocation(cell_id=42, stage_x_mm=3.0, stage_y_mm=4.0,
                        stage_z_mm=1.0, in_frame_y_px=0, in_frame_x_px=0))
    ctx = _make_ctx(sample_map=sm)
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("goto_cell", cell_id=42))
    assert out.ok is True
    assert out.payload["stage_position"]["x_mm"] == 3.0
    assert out.payload["stage_position"]["y_mm"] == 4.0


def test_goto_cell_nearest_to_picks_closest():
    from core.sample_map import CellLocation
    sm = SampleMap()
    sm.add(CellLocation(cell_id=1, stage_x_mm=0.0, stage_y_mm=0.0,
                        stage_z_mm=0.0, in_frame_y_px=0, in_frame_x_px=0))
    sm.add(CellLocation(cell_id=2, stage_x_mm=10.0, stage_y_mm=0.0,
                        stage_z_mm=0.0, in_frame_y_px=0, in_frame_x_px=0))
    ctx = _make_ctx(sample_map=sm)
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("goto_cell",
                                nearest_to={"x_mm": 9.0, "y_mm": 0.0}))
    assert out.ok is True
    assert out.payload["cell"]["cell_id"] == 2


def test_goto_cell_empty_map_errors():
    ctx = _make_ctx()
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("goto_cell", cell_id=1))
    assert out.ok is False
    assert "no map" in out.payload["error"]


def test_goto_cell_unknown_id_errors():
    from core.sample_map import CellLocation
    sm = SampleMap()
    sm.add(CellLocation(cell_id=1, stage_x_mm=0.0, stage_y_mm=0.0,
                        stage_z_mm=0.0, in_frame_y_px=0, in_frame_x_px=0))
    ctx = _make_ctx(sample_map=sm)
    r = build_tool_registry()
    out = r.dispatch(ctx, _call("goto_cell", cell_id=99))
    assert out.ok is False
    assert "not in map" in out.payload["error"]


# ---------------------------------------------------------------------------
# record_timelapse
# ---------------------------------------------------------------------------

def test_record_timelapse_emits_n_frames(monkeypatch):
    summaries = [{"phase_std": 0.5}, {"phase_std": 0.6}, {"phase_std": 0.7}]
    ctx = _make_ctx(capture_summaries=summaries)

    # Skip real sleeps so the test runs fast.
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    r = build_tool_registry()
    out = r.dispatch(ctx, _call("record_timelapse",
                                n_frames=3, interval_s=0.1, run_recon=True))
    assert out.ok is True
    assert out.payload["n_frames"] == 3
    assert len(out.payload["frames"]) == 3
    assert out.payload["frames"][0]["phase_std"] == 0.5
    assert out.payload["frames"][2]["phase_std"] == 0.7


def test_record_timelapse_propagates_capture_failure(monkeypatch):
    summaries = [{"phase_std": 0.5}]
    ctx = _make_ctx(capture_summaries=summaries)

    state = {"calls": 0}

    def cap(args):
        state["calls"] += 1
        if state["calls"] == 2:
            raise RuntimeError("simulated camera fault")
        return {"phase_std": 0.5}

    ctx.capture_and_process = cap  # type: ignore[assignment]

    monkeypatch.setattr("time.sleep", lambda s: None)

    r = build_tool_registry()
    out = r.dispatch(ctx, _call("record_timelapse",
                                n_frames=4, interval_s=0.1))
    assert out.ok is False
    assert out.payload.get("frame") == 1


def test_registry_now_has_19_tools():
    """Sprint 2 added 5 — original 14 + 5 = 19. v2.1.z added 9
    device tools behind ``include_devices=True``; this test pins
    the canonical 19-tool surface (devices opted out)."""
    r = build_tool_registry(include_devices=False)
    assert len(r) == 19
    expected_new = {"stage_focus_search", "map_sample_grid",
                    "list_mapped_cells", "goto_cell", "record_timelapse"}
    assert expected_new.issubset(set(r.names()))
