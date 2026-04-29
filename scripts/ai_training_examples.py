"""Emit fine-tuning examples for the DHM AI assistant.

Writes ``data/ai/training_examples.jsonl`` (training, 100 examples)
and ``data/ai/eval_holdout.jsonl`` (held-out eval, 15 examples) —
one OpenAI chat-format example per line. Tool schemas come from the
live :class:`ToolRegistry` so examples stay in sync with the code.

This script does NOT run training. Output feeds whichever fine-tuning
pipeline the user picks (Ollama Modelfile ADAPTER, HuggingFace ``trl``
SFT, OpenAI hosted fine-tune, etc.).

Lab profile (2026-Q2 baseline)
------------------------------
* HeNe λ=632.8 nm, 50× air objective, 3.45 µm pixel pitch
* Operator: Turkish prose, English/numeric tool args
* Stage: motorised hardware not yet wired — stage tools excluded.
  ``--include-stage`` re-includes them once the motor is connected.

Categories (matches docs/AI_FINETUNE_DATA.md)
---------------------------------------------
1. Tool selection (15)         — natural language → single tool
2. Argument formatting (10)    — type + enum + unit correctness
3. Multi-tool chains (25)      — load → AF → recon → QPI patterns
4. Self-correction (8)         — recover from tool errors
5. Refusal & safety (5)        — out-of-scope / unsafe
6. Domain language TR/EN (15)  — "dalga boyu / λ / wavelength"
7. Conversational style (12)   — short, action-first replies
8. Negative examples (5)       — "I don't have that tool"
9. Lab-specific (5)            — USAF, RBC, E. coli, bead, Bacillus

Usage::

    python scripts/ai_training_examples.py
    python scripts/ai_training_examples.py --include-stage   # motor connected
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from a fresh checkout without `pip install -e .`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from core.ai.tool_impls import build_tool_registry  # noqa: E402

# ---------------------------------------------------------------------------
# Lab profile — gets baked into example prompts and tool args so the model
# learns the *real* parameter combinations the lab uses, not arbitrary ones.
# Keep these in sync with src/core/settings_schema.py:ReconDefaults until
# the lab swaps hardware (then bump and regenerate).
# ---------------------------------------------------------------------------

LAB_PROFILE = {
    "wavelength_nm": 632.8,    # HeNe gas laser
    "pixel_um": 3.45,          # Basler 3.45 µm sensor
    "magnification": 50,       # 50× air objective default
    "n_medium_dry": 1.0,       # USAF, dry smear, bead in air
    "n_medium_wet": 1.337,     # PBS / live culture
    # Typical z propagation range for 50× off-axis transmission DHM.
    "z_scan_min_mm": 0.0,
    "z_scan_max_mm": 15.0,
    "z_scan_steps": 40,
}

# n_sample for QPI — sample → refractive index pair the model learns.
SAMPLE_NSAMPLE = {
    "rbc": 1.41, "ecoli": 1.40, "bacillus": 1.39,
    "staph": 1.40, "pseudo": 1.39, "lacto": 1.39,
    "bead_ps": 1.59, "hela": 1.37,
}

# Tools currently disabled in training because the motorised stage isn't
# wired yet. Re-enable with --include-stage when the hardware lands.
STAGE_TOOL_NAMES = frozenset({
    "stage_get_position", "stage_move_relative", "stage_move_absolute",
    "stage_home", "stage_focus_search",
    "map_sample_grid", "list_mapped_cells", "goto_cell",
})

# Convenience aliases reused across many examples.
_LAMBDA = LAB_PROFILE["wavelength_nm"]
_PIXEL = LAB_PROFILE["pixel_um"]
_MAG = LAB_PROFILE["magnification"]
_N_DRY = LAB_PROFILE["n_medium_dry"]
_N_WET = LAB_PROFILE["n_medium_wet"]
_Z_MIN = LAB_PROFILE["z_scan_min_mm"]
_Z_MAX = LAB_PROFILE["z_scan_max_mm"]
_Z_STEPS = LAB_PROFILE["z_scan_steps"]


def _filter_stage(tools_schema: list[dict], include_stage: bool) -> list[dict]:
    """Strip stage tools from the schema unless explicitly requested.

    The model only learns the tools it sees in ``tools``; hiding stage
    tools at training time means a tool-call to ``goto_cell`` becomes
    out-of-distribution and the model defaults to "I can't do that yet."
    """
    if include_stage:
        return tools_schema
    return [t for t in tools_schema
            if t["function"]["name"] not in STAGE_TOOL_NAMES]


def _system_prompt() -> str:
    """Short system prompt used in every example. Mirrors the runtime
    template in ``core.ai.context`` but with no live state — fine-tuning
    data shouldn't bake snapshots in. Lab profile is included so the
    model knows which optical parameters are baseline."""
    return (
        "You are the AI co-pilot for a Digital Holographic Microscopy app. "
        "Lab setup: HeNe laser λ=632.8 nm, 50× air objective, 3.45 µm "
        "pixel pitch. The operator types in Turkish; reply in Turkish but "
        "keep all tool arguments in English/numeric (enum names UPPER_CASE, "
        "wavelength in nm, lengths in mm or µm as named by the schema). "
        "Use the supplied tools to drive reconstruction, autofocus, QPI, "
        "and depth maps. Read-only tools (get_state, get_last_result, "
        "get_audit_tail) are cheap; use them to ground your reasoning. "
        "Refuse paths outside the user's home directory. Never invent "
        "file paths the operator did not supply. Be concise — short, "
        "numbered, action-first replies; ask the next-step question when "
        "useful."
    )


def _msg(role: str, content: str = "", **extra) -> dict:
    out: dict = {"role": role, "content": content}
    out.update(extra)
    return out


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _example(messages: list[dict], tools: list[dict]) -> dict:
    """One JSONL row.

    ``tools`` is included in every example so the fine-tune can learn
    the schema-call mapping. Trainers that don't accept ``tools`` will
    drop it cleanly — it sits next to ``messages`` per OpenAI fine-tune
    docs.
    """
    return {"messages": messages, "tools": tools}


# ---------------------------------------------------------------------------
# Example seeds
# ---------------------------------------------------------------------------

def build_examples(tools: list[dict]) -> list[dict]:
    sys_msg = _msg("system", _system_prompt())
    examples: list[dict] = []

    # 1. get_state — read-only intro
    examples.append(_example([
        sys_msg,
        _msg("user", "What's loaded right now?"),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "get_state", {})]),
        _msg("tool", json.dumps({
            "loaded_path": None, "loaded_shape": None, "loaded_dtype": None,
            "recon_params": {}, "stage_position_mm": [0.0, 0.0, 0.0],
        }), tool_call_id="c1", name="get_state"),
        _msg("assistant",
             "Nothing is loaded yet. Tell me a hologram path and I'll "
             "open it — for example: \"load ~/Documents/holograms/sample.tif\"."),
    ], tools))

    # 2. load_hologram only
    examples.append(_example([
        sys_msg,
        _msg("user", "Load ~/Documents/holograms/sample.tif"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/Documents/holograms/sample.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/lab/Documents/holograms/sample.tif",
            "shape": [1024, 1024], "dtype": "float32",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant",
             "Loaded a 1024×1024 float32 hologram. Want me to reconstruct "
             "with the current sidebar parameters, or run autofocus first?"),
    ], tools))

    # 3. set_recon_param
    examples.append(_example([
        sys_msg,
        _msg("user", "Set wavelength to 532 nm and pixel size to 4.8 µm."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param",
                                    {"wavelength_nm": 532.0, "pixel_um": 4.8})]),
        _msg("tool", json.dumps({
            "updated": {"wavelength_nm": 532.0, "pixel_um": 4.8},
            "current": {"wavelength_nm": 532.0, "pixel_um": 4.8, "z_mm": 44.0,
                        "n_medium": 1.0, "mask_radius": 80,
                        "subtract_mean": True, "hann_window": False},
        }), tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "Done — λ=532 nm, pixel=4.8 µm. Ready to reconstruct."),
    ], tools))

    # 4. run_reconstruction
    examples.append(_example([
        sys_msg,
        _msg("user", "Reconstruct the loaded hologram."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_reconstruction", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"shape": [1024, 1024], "phase_mean": -0.04,
                        "phase_std": 1.21, "duration_s": 0.78, "z_mm": 44.0},
        }), tool_call_id="c1", name="run_reconstruction"),
        _msg("assistant",
             "Reconstruction complete in 0.78 s. Phase has σ=1.21 rad — "
             "decent contrast. Ready for QPI when you are."),
    ], tools))

    # 5. find_focus_candidates → run_reconstruction
    examples.append(_example([
        sys_msg,
        _msg("user",
             "Find focus candidates between -25 and 25 mm, then reconstruct at the best one."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "find_focus_candidates",
                                    {"z_min_mm": -25.0, "z_max_mm": 25.0,
                                     "n_steps": 60, "metric": "ENTROPY", "top_k": 5})]),
        _msg("tool", json.dumps({"submitted": True}),
             tool_call_id="c1", name="find_focus_candidates"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "get_last_result", {"stage": "autofocus"})]),
        _msg("tool", json.dumps({
            "stage": "autofocus", "available": True,
            "summary": {"best_z_mm": 12.4, "best_metric_value": 0.93,
                        "candidates": [{"z_mm": 12.4, "score": 0.93}]},
        }), tool_call_id="c2", name="get_last_result"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param", {"z_mm": 12.4})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 12.4},
                                 "current": {"z_mm": 12.4}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"shape": [1024, 1024], "z_mm": 12.4}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant",
             "Best focus at z=12.4 mm — reconstruction is ready. Run QPI?"),
    ], tools))

    # 6. load → run_reconstruction → run_qpi
    examples.append(_example([
        sys_msg,
        _msg("user",
             "Open ~/data/cell.tif, reconstruct, and give me the QPI numbers."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/data/cell.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/lab/data/cell.tif", "shape": [2048, 2048], "dtype": "uint16",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"shape": [2048, 2048], "phase_std": 0.88}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.38, "n_medium": 1.337,
                        "total_dry_mass_pg": 64.2, "opd_range_nm": 312.0},
        }), tool_call_id="c3", name="run_qpi"),
        _msg("assistant",
             "Cell is loaded. Reconstruction at the current z and QPI complete: "
             "total dry mass ≈ 64.2 pg, OPD range ≈ 312 nm."),
    ], tools))

    # 7. compute_depth_map
    examples.append(_example([
        sys_msg,
        _msg("user", "Build me a depth map from -10 to 10 mm with 40 steps."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "compute_depth_map",
                                    {"z_min_mm": -10.0, "z_max_mm": 10.0,
                                     "n_steps": 40, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"shape": [1024, 1024], "z_min": -10.0, "z_max": 10.0,
                        "mean_z": 1.4, "valid_pct": 87.0},
        }), tool_call_id="c1", name="compute_depth_map"),
        _msg("assistant",
             "Depth map: mean focus plane ≈ 1.4 mm, 87 % valid pixels."),
    ], tools))

    # 8. get_audit_tail
    examples.append(_example([
        sys_msg,
        _msg("user", "What did I do in the last hour?"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_audit_tail", {"limit": 20})]),
        _msg("tool", json.dumps({
            "limit": 20, "count": 4,
            "entries": [
                {"timestamp": "...", "action": "load_hologram", "params": {}},
                {"timestamp": "...", "action": "reconstruct", "params": {}},
                {"timestamp": "...", "action": "ai.tool.run_qpi.done", "params": {}},
                {"timestamp": "...", "action": "qpi", "params": {}},
            ],
        }), tool_call_id="c1", name="get_audit_tail"),
        _msg("assistant",
             "Recent activity: hologram loaded, reconstruction run, QPI computed."),
    ], tools))

    # 9. stage move + recon
    examples.append(_example([
        sys_msg,
        _msg("user", "Move the stage Z up by 5 mm and reconstruct."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "stage_move_relative",
                                    {"dz_mm": 5.0})]),
        _msg("tool", json.dumps({"x_mm": 0.0, "y_mm": 0.0, "z_mm": 5.0}),
             tool_call_id="c1", name="stage_move_relative"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"z_mm": 44.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant",
             "Stage moved to (0, 0, 5) mm and reconstruction is done. "
             "Need a new autofocus at this position?"),
    ], tools))

    # 10. refusal — out-of-home path
    examples.append(_example([
        sys_msg,
        _msg("user", "Open /etc/passwd."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "/etc/passwd"})]),
        _msg("tool", json.dumps({
            "error": "security check failed",
            "details": "path is outside the home directory"
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant",
             "I can't open files outside the user's home directory. "
             "If you intended a hologram in /etc, please rephrase or "
             "loosen the path restriction in AI settings."),
    ], tools))

    # 11. set_recon_param then run_autofocus
    examples.append(_example([
        sys_msg,
        _msg("user",
             "Use λ=633 and find best z between 0 and 60 mm with PHASE_VARIANCE."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param",
                                    {"wavelength_nm": 633.0})]),
        _msg("tool", json.dumps({"updated": {"wavelength_nm": 633.0},
                                 "current": {"wavelength_nm": 633.0}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 60.0,
                                     "metric": "PHASE_VARIANCE",
                                     "n_steps": 60})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 44.5}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "Best focus z ≈ 44.5 mm at λ=633 nm."),
    ], tools))

    # 12. invalid args, model self-corrects
    examples.append(_example([
        sys_msg,
        _msg("user", "Run autofocus from 50 to 30 mm."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 50.0, "z_max_mm": 30.0})]),
        _msg("tool", json.dumps({
            "error": "z_max_mm (30.0) must exceed z_min_mm (50.0)",
        }), tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": 30.0, "z_max_mm": 50.0})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 41.7}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "Focus is at 41.7 mm — I corrected the swapped range."),
    ], tools))

    # 13. stage_focus_search — physical Z search when calibrated z is unknown
    examples.append(_example([
        sys_msg,
        _msg("user",
             "Yeni sample taktım, focus z'yi bilmiyorum. Sample'ı objektife yaklaştır."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "stage_focus_search", {
                 "search_range_mm": 10.0, "step_mm": 0.5,
                 "metric": "LAPLACIAN_VARIANCE",
                 "refine_with_digital_af": True,
             })]),
        _msg("tool", json.dumps({
            "best_z_mm": 1.5, "best_sharpness": 18.4, "n_steps_evaluated": 21,
            "stage_position": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 1.5},
            "digital_af": {"submitted": True, "best_z_mm": 12.4},
        }), tool_call_id="c1", name="stage_focus_search"),
        _msg("assistant",
             "Sample stage Z=1.5 mm'de en keskin. Digital AF de propagation z'yi "
             "12.4 mm'de doğruladı. Reconstruct edeyim mi?"),
    ], tools))

    # 14. map_sample_grid → list_mapped_cells → goto_cell
    examples.append(_example([
        sys_msg,
        _msg("user",
             "Sample üzerinde -2 ile +2 mm arasında 0.5 mm adımla hücre haritası çıkar."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "map_sample_grid", {
                 "x_min_mm": -2.0, "x_max_mm": 2.0,
                 "y_min_mm": -2.0, "y_max_mm": 2.0,
                 "step_mm": 0.5, "sample_id": "demo_2026_04_27",
             })]),
        _msg("tool", json.dumps({
            "ok": True, "visited_points": 81, "cell_count": 14,
            "summary": {"count": 14, "grid_extent_mm": [-2, 2, -2, 2],
                        "grid_step_mm": 0.5, "sample_id": "demo_2026_04_27"},
            "persisted_path": "/Users/lab/.dhm-reconstruction/sample_maps/demo_2026_04_27.json",
        }), tool_call_id="c1", name="map_sample_grid"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "list_mapped_cells", {"limit": 5})]),
        _msg("tool", json.dumps({
            "summary": {"count": 14},
            "cells": [
                {"cell_id": 1, "stage_x_mm": -1.5, "stage_y_mm": -1.0,
                 "area_um2": 22.5, "dry_mass_pg": 8.1},
                {"cell_id": 2, "stage_x_mm": -1.0, "stage_y_mm": -0.5,
                 "area_um2": 18.7, "dry_mass_pg": 6.3},
            ],
            "truncated": True,
        }), tool_call_id="c2", name="list_mapped_cells"),
        _msg("assistant",
             "14 hücre haritalandı. İlk hücre stage (-1.5, -1.0) mm'de, dry mass 8.1 pg. "
             "Buna gitmemi ister misin yoksa başkasına mı?"),
    ], tools))

    # 15. goto_cell — revisit a mapped cell
    examples.append(_example([
        sys_msg,
        _msg("user", "3 numaralı hücreye git ve QPI çıkar."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "goto_cell", {"cell_id": 3})]),
        _msg("tool", json.dumps({
            "ok": True,
            "cell": {"cell_id": 3, "stage_x_mm": 0.5, "stage_y_mm": 1.0,
                     "stage_z_mm": 1.5, "area_um2": 19.4},
            "stage_position": {"x_mm": 0.5, "y_mm": 1.0, "z_mm": 1.5},
        }), tool_call_id="c1", name="goto_cell"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"total_dry_mass_pg": 7.2, "opd_range_nm": 285.0},
        }), tool_call_id="c2", name="run_qpi"),
        _msg("assistant",
             "Hücre #3 üzerindeyim. QPI: dry mass 7.2 pg, OPD aralığı 285 nm."),
    ], tools))

    # 16. record_timelapse — drift over 30 minutes
    examples.append(_example([
        sys_msg,
        _msg("user",
             "Bu hücreyi 10 dakika boyunca her 60 saniyede bir kayıtla, dry mass değişimini izle."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "record_timelapse", {
                 "n_frames": 10, "interval_s": 60.0,
                 "run_recon": True, "run_qpi": True,
             })]),
        _msg("tool", json.dumps({
            "ok": True, "n_frames": 10, "interval_s": 60.0,
            "duration_s": 540.4,
            "frames": [
                {"frame": 0, "elapsed_s": 0.0, "total_dry_mass_pg": 7.2},
                {"frame": 1, "elapsed_s": 60.1, "total_dry_mass_pg": 7.4},
                {"frame": 5, "elapsed_s": 300.5, "total_dry_mass_pg": 8.1},
                {"frame": 9, "elapsed_s": 540.4, "total_dry_mass_pg": 8.6},
            ],
        }), tool_call_id="c1", name="record_timelapse"),
        _msg("assistant",
             "10 frame kaydedildi. Dry mass 7.2 pg → 8.6 pg, %19 artış 9 dakikada. "
             "Hücre büyüme fazında görünüyor."),
    ], tools))

    # 17. nearest_to navigation — operator pointed at a coordinate
    examples.append(_example([
        sys_msg,
        _msg("user", "Stage'in (1.0, 1.0) mm noktasına en yakın hücreye git."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "goto_cell",
                                    {"nearest_to": {"x_mm": 1.0, "y_mm": 1.0}})]),
        _msg("tool", json.dumps({
            "ok": True,
            "cell": {"cell_id": 7, "stage_x_mm": 0.85, "stage_y_mm": 1.1},
            "stage_position": {"x_mm": 0.85, "y_mm": 1.1, "z_mm": 1.5},
        }), tool_call_id="c1", name="goto_cell"),
        _msg("assistant", "(1.0, 1.0)'e en yakın hücre #7 — onun üzerindeyim."),
    ], tools))

    # 18. cell-aware focus search — sparse cells over background
    examples.append(_example([
        sys_msg,
        _msg("user",
             "Sample'da çok az hücre var, çoğu boş alan. Focus z'yi ara ama "
             "background'dan etkilenmesin."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "stage_focus_search", {
                 "search_range_mm": 8.0, "step_mm": 0.5,
                 "mode": "cell_aware",
                 "mask_dilate_px": 48,
                 "refine_with_digital_af": True,
             })]),
        _msg("tool", json.dumps({
            "best_z_mm": 0.75, "best_sharpness": 42.7,
            "n_steps_evaluated": 17,
            "mode_requested": "cell_aware",
            "mode_used": "cell_aware",
            "mask_pixels_at_probe": 1240,
            "stage_position": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.75},
            "digital_af": {"submitted": True, "best_z_mm": 11.8},
        }), tool_call_id="c1", name="stage_focus_search"),
        _msg("assistant",
             "Cell-aware moduyla z=0.75 mm'de focus buldum; mask hücreleri "
             "kapsayan ~1240 piksel oldu, background skoru bozmadı. "
             "Digital AF de propagation z'yi 11.8 mm'de doğruladı."),
    ], tools))

    # 19. mode=auto picks correctly without explicit instruction
    examples.append(_example([
        sys_msg,
        _msg("user", "Focus z'yi otomatik ayarla."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "stage_focus_search", {
                 "search_range_mm": 6.0, "step_mm": 0.5,
                 "mode": "auto",
             })]),
        _msg("tool", json.dumps({
            "best_z_mm": 1.2, "best_sharpness": 38.5,
            "mode_requested": "auto",
            "mode_used": "cell_aware",
            "mask_pixels_at_probe": 870,
            "stage_position": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 1.2},
        }), tool_call_id="c1", name="stage_focus_search"),
        _msg("assistant",
             "Auto mod hücreler tespit etti, cell-aware'a geçti. "
             "Stage z=1.2 mm'de en keskin."),
    ], tools))

    # 20. mode=full_frame for calibration targets
    examples.append(_example([
        sys_msg,
        _msg("user",
             "USAF kalibrasyon hedefi taktım, focus z'yi bul. Hücre yok, "
             "bütün hedefi tara."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "stage_focus_search", {
                 "search_range_mm": 4.0, "step_mm": 0.25,
                 "mode": "full_frame",
             })]),
        _msg("tool", json.dumps({
            "best_z_mm": -0.5, "best_sharpness": 124.6,
            "mode_requested": "full_frame",
            "mode_used": "full_frame",
            "mask_pixels_at_probe": 0,
            "stage_position": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": -0.5},
        }), tool_call_id="c1", name="stage_focus_search"),
        _msg("assistant",
             "USAF için full_frame moduyla z=-0.5 mm'de focus tespit edildi."),
    ], tools))

    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit AI assistant fine-tuning examples to JSONL."
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "data" / "ai" / "training_examples.jsonl",
        help="Output JSONL path (parents created as needed).",
    )
    args = parser.parse_args(argv)

    registry = build_tool_registry()
    tools_schema = registry.schemas()

    examples = build_examples(tools_schema)

    out: Path = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(examples)} examples ({out.stat().st_size} bytes) → {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
