"""Concrete tool implementations for the DHM AI assistant.

Each handler is a small ``(ctx, args) -> dict`` function. Heavy lifting
(reconstruction, autofocus, QPI) is delegated to the panel via
``ctx.invoke_*`` callables — those marshal the work back to the GUI
thread where the existing pipeline workers run.

State queries (``get_state``, ``get_last_result``, ``get_audit_tail``)
read from the cached snapshot the panel keeps on the GUI thread; they
run on the AI thread without blocking.

Stage tools talk to ``ctx.stage`` (a :class:`MockStage` in v1).

The ``build_tool_registry`` factory at the bottom registers all 19
canonical tools (10 pipeline + 4 stage + 5 sprint-2 workflows) plus
the v2.1.z device family (shutter / led / orchestrator) when the
panel supplies the corresponding ``ctx.shutter`` / ``ctx.led`` /
``ctx.orchestrator`` hooks. Call it from the panel to build the
registry the agent will use; pass ``include_devices=False`` to
opt out of the device tools entirely.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from core.ai.security import (
    ALLOWED_HOLOGRAM_EXTENSIONS,
    SecurityError,
    clamp,
    redact_for_llm,
    validate_path,
)
from core.ai.tools import ToolContext, ToolRegistry, ToolSpec
from core.sample_map import CellLocation

_LOG = logging.getLogger(__name__)


# Allowed metric names — keep the LLM honest. Mirrors
# ``core.autofocus.metrics.FocusMetric`` enum values.
_FOCUS_METRICS = (
    "TOTAL_VARIATION", "GRADIENT", "TENENGRAD", "LAPLACIAN_VARIANCE",
    "BRENNER", "ENTROPY", "VOLLATH_F4", "VOLLATH_F5",
    "NORMALIZED_VARIANCE", "SPECTRAL_ENERGY", "PHASE_VARIANCE",
)


# ===========================================================================
# 1. load_hologram
# ===========================================================================

_LOAD_HOLOGRAM_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Filesystem path to a hologram. Tilde expansion supported.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


def _tool_load_hologram(ctx: ToolContext, args: dict) -> dict:
    raw = args["path"]
    restrict = bool(getattr(ctx.settings, "restrict_to_home", True))
    p = validate_path(
        raw,
        must_exist=True,
        allowed_extensions=ALLOWED_HOLOGRAM_EXTENSIONS,
        restrict_to_home=restrict,
    )
    return ctx.load_hologram(str(p))


# ===========================================================================
# 2. set_recon_param
# ===========================================================================

_SET_RECON_PARAM_SCHEMA = {
    "type": "object",
    "properties": {
        "wavelength_nm": {"type": "number", "minimum": 100, "maximum": 2000},
        "pixel_um": {"type": "number", "minimum": 0.01, "maximum": 100},
        "z_mm": {"type": "number", "minimum": -200, "maximum": 200},
        "n_medium": {"type": "number", "minimum": 1.0, "maximum": 2.0},
        "mask_radius": {"type": "integer", "minimum": 1, "maximum": 1024},
        "subtract_mean": {"type": "boolean"},
        "hann_window": {"type": "boolean"},
    },
    "additionalProperties": False,
}


def _tool_set_recon_param(ctx: ToolContext, args: dict) -> dict:
    if not args:
        return {"error": "no parameters provided"}
    cleaned: dict = {}
    for key, val in args.items():
        if key in ("subtract_mean", "hann_window"):
            cleaned[key] = bool(val)
        elif key == "mask_radius":
            cleaned[key] = int(clamp(key, val))
        else:
            cleaned[key] = float(clamp(key, val))
    return ctx.set_recon_param(cleaned)


# ===========================================================================
# 3. run_reconstruction
# ===========================================================================

_RUN_RECONSTRUCTION_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _tool_run_reconstruction(ctx: ToolContext, args: dict) -> dict:
    snap = ctx.state()
    if not snap.loaded_path:
        return {"error": "no hologram loaded; call load_hologram first"}
    return ctx.invoke_recon({})


# ===========================================================================
# 4. run_autofocus
# ===========================================================================

_RUN_AUTOFOCUS_SCHEMA = {
    "type": "object",
    "properties": {
        "z_min_mm": {"type": "number"},
        "z_max_mm": {"type": "number"},
        "metric": {"type": "string", "enum": list(_FOCUS_METRICS)},
        "n_steps": {"type": "integer", "minimum": 4, "maximum": 256},
    },
    "required": ["z_min_mm", "z_max_mm"],
    "additionalProperties": False,
}


def _tool_run_autofocus(ctx: ToolContext, args: dict) -> dict:
    z_min = clamp("z_min_mm", args["z_min_mm"])
    z_max = clamp("z_max_mm", args["z_max_mm"])
    if z_max <= z_min:
        return {"error": f"z_max_mm ({z_max}) must exceed z_min_mm ({z_min})"}
    payload = {"z_min_mm": z_min, "z_max_mm": z_max}
    if "metric" in args:
        payload["metric"] = str(args["metric"])
    if "n_steps" in args:
        payload["n_steps"] = int(clamp("n_steps", args["n_steps"]))
    return ctx.invoke_autofocus(payload)


# ===========================================================================
# 5. find_focus_candidates
# ===========================================================================

_FIND_FOCUS_CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "z_min_mm": {"type": "number"},
        "z_max_mm": {"type": "number"},
        "n_steps": {"type": "integer", "minimum": 8, "maximum": 256},
        "metric": {"type": "string", "enum": list(_FOCUS_METRICS)},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    "required": ["z_min_mm", "z_max_mm"],
    "additionalProperties": False,
}


def _tool_find_focus_candidates(ctx: ToolContext, args: dict) -> dict:
    z_min = clamp("z_min_mm", args["z_min_mm"])
    z_max = clamp("z_max_mm", args["z_max_mm"])
    if z_max <= z_min:
        return {"error": f"z_max_mm ({z_max}) must exceed z_min_mm ({z_min})"}
    payload = {
        "z_min_mm": z_min,
        "z_max_mm": z_max,
        "n_steps": int(clamp("n_steps", args.get("n_steps", 60))),
        "metric": str(args.get("metric", "ENTROPY")),
        "top_k": int(clamp("top_k", args.get("top_k", 5))),
    }
    return ctx.invoke_find_focus_candidates(payload)


# ===========================================================================
# 6. run_qpi
# ===========================================================================

_RUN_QPI_SCHEMA = {
    "type": "object",
    "properties": {
        "n_sample": {"type": "number", "minimum": 1.0, "maximum": 2.0},
        "n_medium": {"type": "number", "minimum": 1.0, "maximum": 2.0},
        "phase_offset": {"type": "number"},
    },
    "additionalProperties": False,
}


def _tool_run_qpi(ctx: ToolContext, args: dict) -> dict:
    payload: dict = {}
    if "n_sample" in args:
        payload["n_sample"] = clamp("n_sample", args["n_sample"])
    if "n_medium" in args:
        payload["n_medium"] = clamp("n_medium", args["n_medium"])
    if "phase_offset" in args:
        payload["phase_offset"] = clamp("phase_offset", args["phase_offset"])
    return ctx.invoke_qpi(payload)


# ===========================================================================
# 7. compute_depth_map
# ===========================================================================

_DEPTH_MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "z_min_mm": {"type": "number"},
        "z_max_mm": {"type": "number"},
        "n_steps": {"type": "integer", "minimum": 4, "maximum": 256},
        "metric": {"type": "string", "enum": list(_FOCUS_METRICS)},
        "window_size": {"type": "integer", "minimum": 3, "maximum": 33},
    },
    "required": ["z_min_mm", "z_max_mm"],
    "additionalProperties": False,
}


def _tool_compute_depth_map(ctx: ToolContext, args: dict) -> dict:
    z_min = clamp("z_min_mm", args["z_min_mm"])
    z_max = clamp("z_max_mm", args["z_max_mm"])
    if z_max <= z_min:
        return {"error": f"z_max_mm ({z_max}) must exceed z_min_mm ({z_min})"}
    payload = {
        "z_min_mm": z_min,
        "z_max_mm": z_max,
        "n_steps": int(clamp("n_steps", args.get("n_steps", 40))),
        "metric": str(args.get("metric", "LAPLACIAN_VARIANCE")),
        "window_size": int(clamp("window_size", args.get("window_size", 5))),
    }
    return ctx.invoke_depth_map(payload)


# ===========================================================================
# 8. get_state
# ===========================================================================

_GET_STATE_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def _tool_get_state(ctx: ToolContext, args: dict) -> dict:
    snap = ctx.state()
    return snap.as_dict()


# ===========================================================================
# 9. get_last_result
# ===========================================================================

_GET_LAST_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "stage": {
            "type": "string",
            "enum": ["recon", "autofocus", "qpi", "depth"],
        },
    },
    "required": ["stage"],
    "additionalProperties": False,
}


def _tool_get_last_result(ctx: ToolContext, args: dict) -> dict:
    stage = args["stage"]
    fn_map = {
        "recon": ctx.last_recon_summary,
        "autofocus": ctx.last_af_summary,
        "qpi": ctx.last_qpi_summary,
        "depth": ctx.last_depth_summary,
    }
    fn = fn_map.get(stage)
    if fn is None:
        return {"error": f"unknown stage {stage!r}"}
    summary = fn()
    if summary is None:
        return {"stage": stage, "available": False}
    return {"stage": stage, "available": True, "summary": summary}


# ===========================================================================
# 10. get_audit_tail
# ===========================================================================

_GET_AUDIT_TAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
    },
    "additionalProperties": False,
}


def _tool_get_audit_tail(ctx: ToolContext, args: dict) -> dict:
    limit = int(clamp("limit", args.get("limit", 20)))
    entries = list(ctx.audit_tail(limit))
    redact = bool(getattr(ctx.settings, "audit_redact_for_llm", True))
    if redact:
        entries = [redact_for_llm(e) for e in entries]
    return {"limit": limit, "count": len(entries), "entries": entries}


# ===========================================================================
# 11–14. Stage tools
# ===========================================================================

_STAGE_GET_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def _tool_stage_get_position(ctx: ToolContext, args: dict) -> dict:
    x, y, z = ctx.stage.get_position()
    return {"x_mm": float(x), "y_mm": float(y), "z_mm": float(z)}


_STAGE_REL_SCHEMA = {
    "type": "object",
    "properties": {
        "dx_mm": {"type": "number"},
        "dy_mm": {"type": "number"},
        "dz_mm": {"type": "number"},
    },
    "additionalProperties": False,
}


def _tool_stage_move_relative(ctx: ToolContext, args: dict) -> dict:
    dx = clamp("dx_mm", args.get("dx_mm", 0.0))
    dy = clamp("dy_mm", args.get("dy_mm", 0.0))
    dz = clamp("dz_mm", args.get("dz_mm", 0.0))
    x, y, z = ctx.stage.move_relative(dx, dy, dz)
    return {"x_mm": float(x), "y_mm": float(y), "z_mm": float(z)}


_STAGE_ABS_SCHEMA = {
    "type": "object",
    "properties": {
        "x_mm": {"type": "number"},
        "y_mm": {"type": "number"},
        "z_mm": {"type": "number"},
    },
    "required": ["x_mm", "y_mm", "z_mm"],
    "additionalProperties": False,
}


def _tool_stage_move_absolute(ctx: ToolContext, args: dict) -> dict:
    x = clamp("x_mm", args["x_mm"])
    y = clamp("y_mm", args["y_mm"])
    z = clamp("z_mm", args["z_mm"])
    nx, ny, nz = ctx.stage.move_absolute(x, y, z)
    return {"x_mm": float(nx), "y_mm": float(ny), "z_mm": float(nz)}


_STAGE_HOME_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def _tool_stage_home(ctx: ToolContext, args: dict) -> dict:
    x, y, z = ctx.stage.home()
    return {"x_mm": float(x), "y_mm": float(y), "z_mm": float(z), "homed": True}


# ===========================================================================
# 15. stage_focus_search — coarse Z sweep until image is sharp
# ===========================================================================
#
# When the operator doesn't know the calibrated focus z (new sample
# holder, sample slipped, drift overnight), we sweep the stage Z over
# a search range, capture the camera frame at each step, and pick the
# z with the highest sharpness. After landing on the coarse peak, the
# caller can refine with ``run_autofocus`` (digital propagation z) so
# both physical and digital focus agree.
#
# Two scoring strategies, switched by ``mode``:
#   * ``full_frame``  — Laplacian variance over the whole captured
#     image. Robust when the sample fills the frame uniformly
#     (calibration target, USAF chart, dense cell sheet).
#   * ``cell_aware``  — detect bright objects, dilate the mask by a
#     configurable margin, score only inside the mask. Robust when
#     cells are sparse and most of the frame is background — the
#     full-frame metric would dilute the signal otherwise.
#   * ``auto`` (default) — capture once, build a mask. If anything
#     was found, lock cell_aware for the rest of the sweep; else
#     fall through to full_frame. The lock matters: switching mid-
#     sweep would compare apples and oranges across z steps.

_STAGE_FOCUS_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "search_range_mm": {
            "type": "number", "minimum": 0.1, "maximum": 50.0,
            "description": "Total Z window centred on current stage z.",
        },
        "step_mm": {
            "type": "number", "minimum": 0.05, "maximum": 5.0,
            "description": "Step size — smaller = slower, more accurate.",
        },
        "mode": {
            "type": "string",
            "enum": ["auto", "cell_aware", "full_frame"],
            "description": "Sharpness scoring strategy. 'auto' picks "
                           "cell_aware if cells are detected, else "
                           "full_frame. Override only if you know better.",
        },
        "mask_dilate_px": {
            "type": "integer", "minimum": 0, "maximum": 256,
            "description": "Pixels to grow the cell mask outward — pulls "
                           "in surrounding context so sharpness covers "
                           "object edges, not just the bright core.",
        },
        "metric": {
            "type": "string", "enum": ["LAPLACIAN_VARIANCE", "TENENGRAD"],
            "description": "Sharpness metric on the captured frame.",
        },
        "refine_with_digital_af": {"type": "boolean"},
    },
    "additionalProperties": False,
}


def _detect_cell_mask(frame, *, dilate_px: int = 32):
    """Heuristic: pixels above ``mean + std`` are objects; dilate outward.

    Returns ``None`` when no objects are found — the caller falls back
    to full-frame sharpness in that case. Uses ``scipy.ndimage`` if
    available (precise dilation); falls through to a numpy max-filter
    approximation otherwise (rougher but never crashes).
    """
    a = np.abs(np.asarray(frame)).astype(np.float32, copy=False)
    if a.ndim != 2 or a.size == 0:
        return None
    thr = float(a.mean() + a.std())
    raw = a > thr
    if not raw.any():
        return None

    if dilate_px <= 0:
        return raw

    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(raw, iterations=int(dilate_px))
    except ImportError:  # pragma: no cover — scipy is in requirements
        # Cheap approximation: 1-pixel dilation repeated.
        out = raw.copy()
        for _ in range(int(dilate_px)):
            shifted = np.zeros_like(out)
            shifted[1:, :] |= out[:-1, :]
            shifted[:-1, :] |= out[1:, :]
            shifted[:, 1:] |= out[:, :-1]
            shifted[:, :-1] |= out[:, 1:]
            shifted |= out
            out = shifted
        return out


def _masked_sharpness(frame, mask) -> float:
    """Laplacian variance restricted to ``mask`` pixels.

    The Laplacian is computed over the full frame, then only the
    mask-interior values feed the variance. Falls back to the full
    Laplacian's variance when the mask is None — matches the
    ``measure_sharpness`` semantics so callers can swap freely.
    """
    a = np.abs(np.asarray(frame)).astype(np.float32, copy=False)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        return -float("inf")
    lap = (
        a[1:-1, :-2] + a[1:-1, 2:]
        + a[:-2, 1:-1] + a[2:, 1:-1]
        - 4.0 * a[1:-1, 1:-1]
    )
    if mask is None:
        v = float(np.var(lap))
    else:
        m = np.asarray(mask, dtype=bool)[1:-1, 1:-1]
        if not m.any():
            return -float("inf")
        sel = lap[m]
        v = float(np.var(sel))
    if not np.isfinite(v):
        return -float("inf")
    return v


def _tool_stage_focus_search(ctx: ToolContext, args: dict) -> dict:
    # v2.1.z fix: previous version clamped ``search_range_mm`` against
    # ``z_mm`` bounds (-200..200) — wrong axis. ``search_range_mm`` is
    # a window width, not a position. Use the matching bound now that
    # security.NUMERIC_BOUNDS carries it explicitly. ``step_mm`` was
    # only schema-validated; clamp it too in case the schema dep is
    # missing (jsonschema is lazy-imported).
    rng = float(clamp("search_range_mm", args.get("search_range_mm", 5.0)))
    step = float(clamp("step_mm", args.get("step_mm", 0.5)))
    if step <= 0 or step > rng:
        return {"error": f"step_mm={step} invalid for range {rng}"}
    n_steps = max(3, int(rng / step) + 1)

    requested_mode = str(args.get("mode", "auto"))
    if requested_mode not in ("auto", "cell_aware", "full_frame"):
        return {"error": f"unknown mode {requested_mode!r}"}
    # NUMERIC_BOUNDS owns the [0, 256] range — keep this in sync with
    # ``security.clamp`` instead of duplicating the bounds inline.
    try:
        dilate_px = int(clamp("mask_dilate_px", args.get("mask_dilate_px", 32)))
    except SecurityError as exc:
        return {"error": "argument out of range", "details": str(exc)}

    # Centre the sweep on current Z and sweep symmetrically.
    cx, cy, cz = ctx.stage.get_position()
    z_axis = np.linspace(cz - rng / 2.0, cz + rng / 2.0, n_steps).tolist()

    # ---------- Mode resolution: capture once at current z to decide ----
    # We don't move the stage for the probe — we just look at whatever the
    # camera sees right now, build a candidate mask, and let the result
    # pick the strategy. This locks the mode for the whole sweep so each
    # z step's score is comparable to the others.
    try:
        probe_frame = ctx.capture_frame()
    except Exception as exc:  # noqa: BLE001
        return {"error": "capture failed", "details": str(exc),
                "at_z": float(cz)}
    if probe_frame is None:
        return {"error": "no frame to capture (load a hologram first)"}

    if requested_mode == "auto":
        candidate_mask = _detect_cell_mask(probe_frame, dilate_px=dilate_px)
        chosen_mode = "cell_aware" if candidate_mask is not None else "full_frame"
        mask_pixels = int(candidate_mask.sum()) if candidate_mask is not None else 0
    elif requested_mode == "cell_aware":
        candidate_mask = _detect_cell_mask(probe_frame, dilate_px=dilate_px)
        if candidate_mask is None:
            return {
                "error": "cell_aware requested but no cells detected",
                "hint": "lower the threshold by adjusting mask_dilate_px, "
                        "or fall back to mode='full_frame' / 'auto'",
            }
        chosen_mode = "cell_aware"
        mask_pixels = int(candidate_mask.sum())
    else:  # full_frame forced
        chosen_mode = "full_frame"
        mask_pixels = 0

    measurements: list[dict] = []
    best_z = cz
    best_score = -float("inf")

    for z in z_axis:
        try:
            ctx.stage.move_absolute(cx, cy, float(z))
        except Exception as exc:  # noqa: BLE001 — clamp/NaN come back here
            return {"error": "stage move failed",
                    "details": str(exc), "at_z": float(z)}
        try:
            frame = ctx.capture_frame()
        except Exception as exc:  # noqa: BLE001
            return {"error": "capture failed",
                    "details": str(exc), "at_z": float(z)}
        if frame is None:
            return {"error": "no frame to capture mid-sweep"}

        try:
            if chosen_mode == "cell_aware":
                # Re-detect per frame — cells move into focus as we
                # change z, so the mask itself sharpens. Using a fresh
                # mask each step gives a stronger peak signal than a
                # single mask reused across z.
                mask = _detect_cell_mask(frame, dilate_px=dilate_px)
                score = _masked_sharpness(frame, mask)
            else:
                score = float(ctx.measure_sharpness(frame))
        except Exception as exc:  # noqa: BLE001
            return {"error": "sharpness measurement failed",
                    "details": str(exc)}

        measurements.append({"z_mm": float(z), "sharpness": score})
        if score > best_score:
            best_score = score
            best_z = float(z)

    # Land the stage on the best z.
    try:
        ctx.stage.move_absolute(cx, cy, best_z)
    except Exception as exc:  # noqa: BLE001
        return {"error": "stage settle failed", "details": str(exc)}

    out: dict = {
        "best_z_mm": best_z,
        "best_sharpness": best_score,
        "n_steps_evaluated": len(measurements),
        "search_range_mm": rng,
        "step_mm": step,
        "mode_requested": requested_mode,
        "mode_used": chosen_mode,
        "mask_pixels_at_probe": mask_pixels,
        "measurements": measurements,
        "stage_position": {"x_mm": cx, "y_mm": cy, "z_mm": best_z},
    }

    # Optional digital-AF refinement — bridges physical (stage) and
    # digital (propagation) z onto the same image plane.
    if bool(args.get("refine_with_digital_af", False)):
        af_args = {"z_min_mm": best_z - 1.0, "z_max_mm": best_z + 1.0,
                   "n_steps": 40}
        af_args["z_min_mm"] = clamp("z_min_mm", af_args["z_min_mm"])
        af_args["z_max_mm"] = clamp("z_max_mm", af_args["z_max_mm"])
        try:
            af_out = ctx.invoke_autofocus(af_args)
            out["digital_af"] = af_out
        except Exception as exc:  # noqa: BLE001
            out["digital_af_error"] = str(exc)

    return out


# ===========================================================================
# 16. map_sample_grid — XY sweep, segment each frame, store cells
# ===========================================================================
#
# Sweeps the stage over an XY grid; at each grid point captures a
# hologram, runs reconstruction + cell segmentation, and stores
# detected cells in the SampleMap. Heavy operation — for an 8×8 grid
# at 0.5 mm step, expect 64 reconstructions.

_MAP_SAMPLE_GRID_SCHEMA = {
    "type": "object",
    "properties": {
        "x_min_mm": {"type": "number", "minimum": -200, "maximum": 200},
        "x_max_mm": {"type": "number", "minimum": -200, "maximum": 200},
        "y_min_mm": {"type": "number", "minimum": -200, "maximum": 200},
        "y_max_mm": {"type": "number", "minimum": -200, "maximum": 200},
        "step_mm": {"type": "number", "minimum": 0.05, "maximum": 10.0},
        "sample_id": {"type": "string"},
        "skip_recon": {
            "type": "boolean",
            "description": "Skip reconstruction; record stage points only "
                           "(useful for dry-run timing).",
        },
    },
    "required": ["x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm", "step_mm"],
    "additionalProperties": False,
}


def _tool_map_sample_grid(ctx: ToolContext, args: dict) -> dict:
    x0 = clamp("x_mm", args["x_min_mm"])
    x1 = clamp("x_mm", args["x_max_mm"])
    y0 = clamp("y_mm", args["y_min_mm"])
    y1 = clamp("y_mm", args["y_max_mm"])
    step = float(args["step_mm"])
    if x1 <= x0 or y1 <= y0:
        return {"error": "min must be < max for both X and Y"}
    if step <= 0:
        return {"error": "step_mm must be positive"}

    sample_id = str(args.get("sample_id", "")) or "sample"
    skip_recon = bool(args.get("skip_recon", False))

    xs = np.arange(x0, x1 + 1e-9, step).tolist()
    ys = np.arange(y0, y1 + 1e-9, step).tolist()
    n_grid = len(xs) * len(ys)
    if n_grid > 1024:
        return {"error": "grid too large",
                "details": f"{n_grid} points; reduce range or increase step"}

    # Reset map; each sweep is its own snapshot per the SampleMap
    # contract.
    sample_map = ctx.sample_map
    sample_map.reset()
    sample_map.stamp(x_min=x0, x_max=x1, y_min=y0, y_max=y1,
                     step_mm=step, sample_id=sample_id)

    cell_id_seq = 0
    visited = 0

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            try:
                _, _, z = ctx.stage.get_position()
                ctx.stage.move_absolute(float(x), float(y), float(z))
            except Exception as exc:  # noqa: BLE001
                return {"error": "stage move failed",
                        "details": str(exc),
                        "visited": visited}

            visited += 1
            if skip_recon:
                continue

            # Drive a reconstruction + a quick cell segmentation. The
            # panel's invoke_recon already updates the cache; the
            # cells we record here come from capture_and_process,
            # which the panel implements to call segmentation on the
            # current reconstruction's phase.
            try:
                summary = ctx.capture_and_process({"segment": True})
            except Exception as exc:  # noqa: BLE001
                return {"error": "capture+process failed",
                        "details": str(exc), "visited": visited}

            for cell in summary.get("cells") or []:
                cell_id_seq += 1
                sample_map.add(CellLocation(
                    cell_id=cell_id_seq,
                    stage_x_mm=float(x),
                    stage_y_mm=float(y),
                    stage_z_mm=float(z),
                    in_frame_y_px=int(cell.get("centroid_y_px", 0)),
                    in_frame_x_px=int(cell.get("centroid_x_px", 0)),
                    area_um2=float(cell.get("area_um2", 0.0)),
                    dry_mass_pg=float(cell.get("dry_mass_pg", 0.0)),
                ))

    # Persist after each successful sweep so the GUI map can read it
    # the moment we return.
    persisted_path = None
    try:
        persisted_path = ctx.persist_sample_map()
    except Exception:  # noqa: BLE001
        pass

    return {
        "ok": True,
        "visited_points": visited,
        "cell_count": len(sample_map),
        "summary": sample_map.summary(),
        "persisted_path": persisted_path,
    }


# ===========================================================================
# 17. list_mapped_cells
# ===========================================================================

_LIST_MAPPED_CELLS_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 500},
    },
    "additionalProperties": False,
}


def _tool_list_mapped_cells(ctx: ToolContext, args: dict) -> dict:
    limit = int(clamp("limit", args.get("limit", 50)))
    sample_map = ctx.sample_map
    cells = [c.as_dict() for c in sample_map.cells[:limit]]
    return {
        "summary": sample_map.summary(),
        "cells": cells,
        "truncated": len(sample_map.cells) > limit,
    }


# ===========================================================================
# 18. goto_cell
# ===========================================================================

_GOTO_CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "cell_id": {"type": "integer", "minimum": 1},
        "nearest_to": {
            "type": "object",
            "properties": {
                "x_mm": {"type": "number"},
                "y_mm": {"type": "number"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


def _tool_goto_cell(ctx: ToolContext, args: dict) -> dict:
    sample_map = ctx.sample_map
    if not sample_map.cells:
        return {"error": "no map yet", "hint": "run map_sample_grid first"}

    cell = None
    if "cell_id" in args:
        cell = sample_map.by_id(int(args["cell_id"]))
        if cell is None:
            return {"error": f"cell_id {args['cell_id']} not in map",
                    "available_ids": [c.cell_id for c in sample_map.cells[:20]]}
    elif "nearest_to" in args:
        nt = args["nearest_to"] or {}
        cell = sample_map.nearest(
            float(nt.get("x_mm", 0.0)),
            float(nt.get("y_mm", 0.0)),
        )
    else:
        return {"error": "specify cell_id or nearest_to"}
    assert cell is not None  # narrowed above

    try:
        x, y, z = ctx.stage.move_absolute(
            cell.stage_x_mm, cell.stage_y_mm, cell.stage_z_mm,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": "stage move failed", "details": str(exc),
                "target_cell": cell.as_dict()}

    return {
        "ok": True,
        "cell": cell.as_dict(),
        "stage_position": {"x_mm": x, "y_mm": y, "z_mm": z},
    }


# ===========================================================================
# 19. record_timelapse
# ===========================================================================
#
# Periodic capture + optional pipeline runs. Sleeps between iterations
# in chunks so cancel propagates within ~200 ms. The summary dict
# includes per-frame scalars so the LLM can reason about drift over
# time.

_RECORD_TIMELAPSE_SCHEMA = {
    "type": "object",
    "properties": {
        "n_frames": {"type": "integer", "minimum": 1, "maximum": 1000},
        "interval_s": {"type": "number", "minimum": 0.1, "maximum": 3600.0},
        "run_recon": {"type": "boolean"},
        "run_qpi": {"type": "boolean"},
    },
    "required": ["n_frames", "interval_s"],
    "additionalProperties": False,
}


def _tool_record_timelapse(ctx: ToolContext, args: dict) -> dict:
    n_frames = int(args["n_frames"])
    interval_s = float(args["interval_s"])
    run_recon = bool(args.get("run_recon", True))
    run_qpi = bool(args.get("run_qpi", False))

    if n_frames < 1:
        return {"error": "n_frames must be >= 1"}

    started_at = time.time()
    frames: list[dict] = []

    cancelled = False
    for i in range(n_frames):
        # v2.1.z: poll the agent-level cancel hook before each capture.
        # Previous version's "100 ms slice sleep" comment claimed
        # cancellation propagated, but the cancel callable wasn't
        # in scope and the loop ran to completion regardless of
        # the worker's requestInterruption().
        if ctx.is_cancelled is not None and ctx.is_cancelled():
            cancelled = True
            break
        capture_args = {"run_recon": run_recon, "run_qpi": run_qpi}
        try:
            summary = ctx.capture_and_process(capture_args)
        except Exception as exc:  # noqa: BLE001
            return {"error": "capture+process failed at frame",
                    "frame": i, "details": str(exc),
                    "completed_frames": i,
                    "cancelled": cancelled}
        frame_record = {
            "frame": i,
            "elapsed_s": round(time.time() - started_at, 3),
        }
        # Pull through any compact scalars the panel returned so the
        # LLM can compare across frames without us hard-coding a
        # specific payload shape.
        for key in ("phase_std", "z_mm", "total_dry_mass_pg",
                    "cell_count", "opd_range_nm"):
            if isinstance(summary, dict) and key in summary:
                frame_record[key] = summary[key]
        frames.append(frame_record)

        # Sleep between frames in 100-ms slices, polling the cancel
        # hook each slice so Stop has ≤ 100 ms latency.
        if i < n_frames - 1:
            sleep_left = interval_s
            while sleep_left > 0:
                if ctx.is_cancelled is not None and ctx.is_cancelled():
                    cancelled = True
                    break
                chunk = min(sleep_left, 0.1)
                time.sleep(chunk)
                sleep_left -= chunk
            if cancelled:
                break

    return {
        "ok": True,
        "n_frames": n_frames,
        "interval_s": interval_s,
        "run_recon": run_recon,
        "run_qpi": run_qpi,
        "duration_s": round(time.time() - started_at, 3),
        "frames": frames,
        "cancelled": cancelled,
        "completed_frames": len(frames),
    }


# ===========================================================================
# 20–23. v2.1.z device tools — APT-aligned, use core.devices Protocols
# ===========================================================================
#
# The legacy ``stage_*`` family above talks to ``ctx.stage`` (a v1
# MockStage with ``move_relative`` / ``move_absolute`` / ``home``).
# These newer tools target the v2.1.z device registry
# (``core.devices.StageDevice`` / ``ShutterDevice`` / ``LEDDevice``)
# so a real APT-compatible driver — once installed — works as a
# drop-in replacement. ``ctx.shutter`` / ``ctx.led`` are populated
# only when the panel wires them; tools return a friendly
# ``{"error": "<kind> device not configured"}`` otherwise.

_LIST_DEVICES_SCHEMA = {
    "type": "object", "properties": {}, "additionalProperties": False,
}


def _tool_list_devices(ctx: ToolContext, args: dict) -> dict:
    """Discovery: which device backends does the host advertise +
    which subset is actually usable (SDK installed)?

    Returns a dict with `available` (immediate-use names + capability
    summaries) and `all` (everything registered, including SDK-gated
    vendor stubs that need pip wheels). Lets the LLM ask the user
    "do you want me to use mock_stage or do you have a Thorlabs
    KDC101 installed?" instead of guessing.
    """
    try:
        from core import devices as _devices
    except ImportError:
        return {"error": "core.devices not importable"}
    try:
        all_b = _devices.all_backends()
        avail_b = _devices.available_backends()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"device registry query failed: {exc}"}

    def _row(b):
        return {
            "name": b.name,
            "kind": b.kind.value,
            "display_name": b.display_name,
            "summary": b.summary,
            "requires_sdk": list(b.requires_sdk),
        }

    avail_names = {b.name for b in avail_b}
    return {
        "available": [_row(b) for b in avail_b],
        "all": [
            {**_row(b), "available": b.name in avail_names}
            for b in all_b
        ],
    }


_SHUTTER_OPEN_SCHEMA = {
    "type": "object", "properties": {}, "additionalProperties": False,
}
_SHUTTER_CLOSE_SCHEMA = _SHUTTER_OPEN_SCHEMA
_SHUTTER_STATUS_SCHEMA = _SHUTTER_OPEN_SCHEMA


def _shutter_or_error(ctx: ToolContext) -> dict:
    """Returns {} when shutter is configured + connected; an error
    dict otherwise. Avoid duplicating the "is the device wired"
    boilerplate across every shutter tool."""
    if ctx.shutter is None:
        return {"error": "shutter device not configured"}
    if not getattr(ctx.shutter, "is_connected", False):
        try:
            ctx.shutter.connect()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"shutter connect failed: {exc}"}
    return {}


def _tool_shutter_open(ctx: ToolContext, args: dict) -> dict:
    err = _shutter_or_error(ctx)
    if err:
        return err
    try:
        ctx.shutter.open()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"shutter open failed: {exc}"}
    return {"ok": True, "is_open": bool(ctx.shutter.is_open)}


def _tool_shutter_close(ctx: ToolContext, args: dict) -> dict:
    err = _shutter_or_error(ctx)
    if err:
        return err
    try:
        ctx.shutter.close()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"shutter close failed: {exc}"}
    return {"ok": True, "is_open": bool(ctx.shutter.is_open)}


def _tool_shutter_status(ctx: ToolContext, args: dict) -> dict:
    if ctx.shutter is None:
        return {"configured": False}
    return {
        "configured": True,
        "is_connected": bool(getattr(ctx.shutter, "is_connected", False)),
        "is_open": bool(getattr(ctx.shutter, "is_open", False)),
        "name": str(getattr(ctx.shutter, "name", "shutter")),
    }


_LED_INTENSITY_SCHEMA = {
    "type": "object",
    "properties": {
        "intensity_percent": {
            "type": "number", "minimum": 0.0, "maximum": 100.0,
            "description": "0–100 % drive level. Clamped — out-of-range "
                           "values silently snap to bounds.",
        },
    },
    "required": ["intensity_percent"],
    "additionalProperties": False,
}
_LED_ON_OFF_SCHEMA = {
    "type": "object", "properties": {}, "additionalProperties": False,
}
_LED_STATUS_SCHEMA = _LED_ON_OFF_SCHEMA


def _led_or_error(ctx: ToolContext) -> dict:
    if ctx.led is None:
        return {"error": "led device not configured"}
    if not getattr(ctx.led, "is_connected", False):
        try:
            ctx.led.connect()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"led connect failed: {exc}"}
    return {}


def _tool_led_set_intensity(ctx: ToolContext, args: dict) -> dict:
    err = _led_or_error(ctx)
    if err:
        return err
    pct = float(clamp("intensity_percent", args["intensity_percent"]))
    try:
        ctx.led.set_intensity(pct)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"led set_intensity failed: {exc}"}
    return {
        "ok": True,
        "intensity_percent": float(ctx.led.intensity_percent),
    }


def _tool_led_on(ctx: ToolContext, args: dict) -> dict:
    err = _led_or_error(ctx)
    if err:
        return err
    try:
        ctx.led.on()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"led on failed: {exc}"}
    return {"ok": True, "is_on": bool(ctx.led.is_on)}


def _tool_led_off(ctx: ToolContext, args: dict) -> dict:
    err = _led_or_error(ctx)
    if err:
        return err
    try:
        ctx.led.off()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"led off failed: {exc}"}
    return {"ok": True, "is_on": bool(ctx.led.is_on)}


def _tool_led_status(ctx: ToolContext, args: dict) -> dict:
    if ctx.led is None:
        return {"configured": False}
    return {
        "configured": True,
        "is_connected": bool(getattr(ctx.led, "is_connected", False)),
        "is_on": bool(getattr(ctx.led, "is_on", False)),
        "intensity_percent": float(
            getattr(ctx.led, "intensity_percent", 0.0),
        ),
        "name": str(getattr(ctx.led, "name", "led")),
    }


_ACQUIRE_GRID_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {"type": "integer", "minimum": 1, "maximum": 64},
        "cols": {"type": "integer", "minimum": 1, "maximum": 64},
        "spacing_x_um": {"type": "number", "minimum": 0.1,
                         "maximum": 50_000.0},
        "spacing_y_um": {"type": "number", "minimum": 0.1,
                         "maximum": 50_000.0},
        "led_intensity_percent": {
            "type": "number", "minimum": 0.0, "maximum": 100.0,
        },
        "shutter_per_frame": {"type": "boolean"},
        "settle_time_s": {"type": "number", "minimum": 0.0,
                          "maximum": 10.0},
    },
    "required": ["rows", "cols", "spacing_x_um", "spacing_y_um"],
    "additionalProperties": False,
}


def _tool_acquire_grid(ctx: ToolContext, args: dict) -> dict:
    """Multi-position acquisition via the v2.1.z orchestrator.

    LLM declares ``rows × cols`` grid + spacings; the panel-side
    ``ctx.orchestrator`` callable runs an ``AcquisitionPlan``
    against the wired camera + stage + shutter + LED. Returns a
    summary the LLM can compare across positions.
    """
    if ctx.orchestrator is None:
        return {"error": "orchestrator not configured "
                         "(panel didn't wire run_plan callback)"}
    rows = int(clamp("rows", args["rows"]))
    cols = int(clamp("cols", args["cols"]))
    spacing_x = float(clamp("spacing_x_um", args["spacing_x_um"]))
    spacing_y = float(clamp("spacing_y_um", args["spacing_y_um"]))
    led_pct = args.get("led_intensity_percent")
    if led_pct is not None:
        led_pct = float(clamp("intensity_percent", led_pct))
    shutter_per_frame = bool(args.get("shutter_per_frame", True))
    settle_s = float(clamp(
        "settle_time_s", args.get("settle_time_s", 0.05),
    ))
    plan_args = {
        "rows": rows, "cols": cols,
        "spacing_x_um": spacing_x, "spacing_y_um": spacing_y,
        "led_intensity_percent": led_pct,
        "shutter_per_frame": shutter_per_frame,
        "settle_time_s": settle_s,
    }
    try:
        return ctx.orchestrator(plan_args)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"acquire_grid failed: {exc}"}


# ===========================================================================
# Registry factory
# ===========================================================================

def build_tool_registry(*, include_devices: bool = True) -> ToolRegistry:
    """Construct a :class:`ToolRegistry` with the DHM tools.

    Default registers 19 canonical tools (10 pipeline + 4 stage + 5
    sprint-2) plus the v2.1.z device family (shutter / led /
    list_devices / acquire_grid) → 23 total. Pass
    ``include_devices=False`` to skip the device tools when the
    panel hasn't wired ``ctx.shutter`` / ``ctx.led`` / ``ctx.orchestrator``.

    Called once per panel construction. Re-registration is rejected by
    the registry so each call returns a fresh instance.
    """
    reg = ToolRegistry()

    reg.register(ToolSpec(
        name="load_hologram",
        description="Load a hologram from disk into the input panel. "
                    "Use this whenever the operator names a file or asks to switch samples.",
        parameters=_LOAD_HOLOGRAM_SCHEMA,
        handler=_tool_load_hologram,
        requires_gui_thread=True,
    ))
    reg.register(ToolSpec(
        name="set_recon_param",
        description="Update one or more reconstruction parameters in the sidebar "
                    "(wavelength_nm, pixel_um, z_mm, n_medium, mask_radius, subtract_mean, "
                    "hann_window). Returns the updated and currently-active parameter dicts.",
        parameters=_SET_RECON_PARAM_SCHEMA,
        handler=_tool_set_recon_param,
        requires_gui_thread=True,
    ))
    reg.register(ToolSpec(
        name="run_reconstruction",
        description="Run the off-axis reconstruction pipeline on the loaded hologram "
                    "with the current sidebar parameters. Returns shape and phase summary scalars.",
        parameters=_RUN_RECONSTRUCTION_SCHEMA,
        handler=_tool_run_reconstruction,
        requires_gui_thread=True,
    ))
    reg.register(ToolSpec(
        name="run_autofocus",
        description="Find the best-focus z within [z_min_mm, z_max_mm] using the chosen "
                    "metric. Default metric is PHASE_VARIANCE. Returns best_z_mm and the metric value.",
        parameters=_RUN_AUTOFOCUS_SCHEMA,
        handler=_tool_run_autofocus,
        requires_gui_thread=True,
    ))
    reg.register(ToolSpec(
        name="find_focus_candidates",
        description="Scan the focus landscape and return the top-K plausible focus planes — "
                    "use this for multi-object scenes (cells at different z, USAF chart at angle).",
        parameters=_FIND_FOCUS_CANDIDATES_SCHEMA,
        handler=_tool_find_focus_candidates,
        requires_gui_thread=True,
    ))
    reg.register(ToolSpec(
        name="run_qpi",
        description="Compute Quantitative Phase Imaging metrics on the latest reconstruction "
                    "(OPD, height, dry mass, phase stats). Optional refractive indices override sidebar values.",
        parameters=_RUN_QPI_SCHEMA,
        handler=_tool_run_qpi,
        requires_gui_thread=True,
    ))
    reg.register(ToolSpec(
        name="compute_depth_map",
        description="Build a per-pixel best-focus z map across the requested range. "
                    "Use when the sample has structure at different depths (tomography use case).",
        parameters=_DEPTH_MAP_SCHEMA,
        handler=_tool_compute_depth_map,
        requires_gui_thread=True,
    ))
    reg.register(ToolSpec(
        name="get_state",
        description="Return a snapshot of the app's current state — loaded hologram, current "
                    "parameters, last result summaries, stage position, and recent actions.",
        parameters=_GET_STATE_SCHEMA,
        handler=_tool_get_state,
    ))
    reg.register(ToolSpec(
        name="get_last_result",
        description="Return the summary scalars for the most recent run of one stage "
                    "('recon', 'autofocus', 'qpi', or 'depth').",
        parameters=_GET_LAST_RESULT_SCHEMA,
        handler=_tool_get_last_result,
    ))
    reg.register(ToolSpec(
        name="get_audit_tail",
        description="Return the last N entries from the JSONL audit log. Useful for "
                    "answering 'what was tried recently?' or replaying a debug session.",
        parameters=_GET_AUDIT_TAIL_SCHEMA,
        handler=_tool_get_audit_tail,
    ))
    reg.register(ToolSpec(
        name="stage_get_position",
        description="Return the current XYZ stage position in mm. Stage is virtual until "
                    "real hardware is connected; positions are tracked correctly anyway.",
        parameters=_STAGE_GET_SCHEMA,
        handler=_tool_stage_get_position,
    ))
    reg.register(ToolSpec(
        name="stage_move_relative",
        description="Move the stage by the given delta in mm. Missing axes default to 0.",
        parameters=_STAGE_REL_SCHEMA,
        handler=_tool_stage_move_relative,
    ))
    reg.register(ToolSpec(
        name="stage_move_absolute",
        description="Move the stage to an absolute XYZ position in mm.",
        parameters=_STAGE_ABS_SCHEMA,
        handler=_tool_stage_move_absolute,
    ))
    reg.register(ToolSpec(
        name="stage_home",
        description="Send the stage back to (0, 0, 0). Mock stage is cheap; "
                    "real-hardware drivers will require confirmation.",
        parameters=_STAGE_HOME_SCHEMA,
        handler=_tool_stage_home,
    ))

    # ---- Sprint 2: focus search, mapping, time-lapse -------------------
    reg.register(ToolSpec(
        name="stage_focus_search",
        description="Sweep the stage Z over a search range and pick the position "
                    "with the sharpest captured image. Use when the calibrated focus "
                    "z is unknown, the sample drifted, or different cells in the "
                    "sample sit at different depths.\n"
                    "Modes:\n"
                    "  - 'auto' (default): probes the current frame; uses cell-aware "
                    "scoring if cells are visible, else full-frame.\n"
                    "  - 'cell_aware': scores sharpness only inside a dilated cell "
                    "mask. Strongest signal for sparse cells over background.\n"
                    "  - 'full_frame': scores the whole image. Best for dense "
                    "samples / calibration targets.\n"
                    "Set `mask_dilate_px` larger when cells are small or you want "
                    "to capture surrounding fringe context (default 32 px). "
                    "Optionally refines with digital autofocus afterwards.",
        parameters=_STAGE_FOCUS_SEARCH_SCHEMA,
        handler=_tool_stage_focus_search,
    ))
    reg.register(ToolSpec(
        name="map_sample_grid",
        description="Sweep the XY stage over a rectangular grid, capture and segment "
                    "at each step, and record every detected cell's stage XY into the "
                    "sample map. Heavy operation — for an 8×8 grid expect dozens of "
                    "reconstructions. Use sparingly, then `goto_cell` to teleport.",
        parameters=_MAP_SAMPLE_GRID_SCHEMA,
        handler=_tool_map_sample_grid,
    ))
    reg.register(ToolSpec(
        name="list_mapped_cells",
        description="Return the cells currently in the sample map (built by "
                    "map_sample_grid). Useful after a sweep to check what was found "
                    "or to pick a cell id to revisit.",
        parameters=_LIST_MAPPED_CELLS_SCHEMA,
        handler=_tool_list_mapped_cells,
    ))
    reg.register(ToolSpec(
        name="goto_cell",
        description="Move the stage to a previously-mapped cell. Specify either "
                    "`cell_id` (from list_mapped_cells) or `nearest_to: {x_mm, y_mm}` "
                    "to grab the closest cell to a coordinate.",
        parameters=_GOTO_CELL_SCHEMA,
        handler=_tool_goto_cell,
    ))
    reg.register(ToolSpec(
        name="record_timelapse",
        description="Capture N frames at the given interval, optionally running "
                    "reconstruction and/or QPI per frame. Returns per-frame summary "
                    "scalars (phase_std, dry_mass, cell count) so you can compare "
                    "drift across the recording.",
        parameters=_RECORD_TIMELAPSE_SCHEMA,
        handler=_tool_record_timelapse,
    ))

    if include_devices:
        # v2.1.z device tools — registered only when caller asks for
        # them (panel may keep them off when no real hardware is
        # wired). Even when registered the runtime tools surface a
        # friendly "device not configured" error if ``ctx.shutter``
        # / ``ctx.led`` / ``ctx.orchestrator`` are still ``None``.
        reg.register(ToolSpec(
            name="list_devices",
            description="List every device backend the host advertises. "
                        "Returns 'available' (immediately usable, mock + "
                        "any vendor SDK installed) and 'all' (everything "
                        "registered, including SDK-gated stubs). Use this "
                        "before stage/shutter/LED tools to confirm the "
                        "hardware is reachable.",
            parameters=_LIST_DEVICES_SCHEMA,
            handler=_tool_list_devices,
        ))
        reg.register(ToolSpec(
            name="shutter_open",
            description="Open the illumination shutter. Lab uses this "
                        "around long acquisitions to minimise photobleach.",
            parameters=_SHUTTER_OPEN_SCHEMA,
            handler=_tool_shutter_open,
        ))
        reg.register(ToolSpec(
            name="shutter_close",
            description="Close the illumination shutter (block beam).",
            parameters=_SHUTTER_CLOSE_SCHEMA,
            handler=_tool_shutter_close,
        ))
        reg.register(ToolSpec(
            name="shutter_status",
            description="Return shutter device state: configured, "
                        "connected, currently open or closed.",
            parameters=_SHUTTER_STATUS_SCHEMA,
            handler=_tool_shutter_status,
        ))
        reg.register(ToolSpec(
            name="led_set_intensity",
            description="Set illumination LED brightness (0–100 %). "
                        "Out-of-range values clamp instead of raising.",
            parameters=_LED_INTENSITY_SCHEMA,
            handler=_tool_led_set_intensity,
        ))
        reg.register(ToolSpec(
            name="led_on",
            description="Turn the illumination LED on (intensity stays "
                        "at last set value).",
            parameters=_LED_ON_OFF_SCHEMA,
            handler=_tool_led_on,
        ))
        reg.register(ToolSpec(
            name="led_off",
            description="Turn the illumination LED off (intensity preserved).",
            parameters=_LED_ON_OFF_SCHEMA,
            handler=_tool_led_off,
        ))
        reg.register(ToolSpec(
            name="led_status",
            description="Return LED state: configured, connected, on/off, "
                        "current intensity %.",
            parameters=_LED_STATUS_SCHEMA,
            handler=_tool_led_status,
        ))
        reg.register(ToolSpec(
            name="acquire_grid",
            description="Run a rows × cols grid acquisition through the "
                        "v2.1.z orchestrator: stage waypoints + optional "
                        "shutter + LED choreography. Returns per-position "
                        "frame summaries. Use for tile / mosaic captures "
                        "without scripting each move yourself.",
            parameters=_ACQUIRE_GRID_SCHEMA,
            handler=_tool_acquire_grid,
            requires_gui_thread=True,
        ))

    return reg


__all__ = [
    "build_tool_registry",
]
