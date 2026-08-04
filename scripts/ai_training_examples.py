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

# Tools currently disabled in training because the lab hardware isn't
# wired yet. Two groups, both off until the corresponding driver lands:
#   * stage tools — need a motorised stage
#   * device tools — need a programmable shutter + LED + acquisition rig
# Re-enable with --include-stage / --include-devices when hardware lands.
STAGE_TOOL_NAMES = frozenset({
    "stage_get_position", "stage_move_relative", "stage_move_absolute",
    "stage_home", "stage_focus_search",
    "map_sample_grid", "list_mapped_cells", "goto_cell",
})

DEVICE_TOOL_NAMES = frozenset({
    "list_devices",
    "shutter_open", "shutter_close", "shutter_status",
    "led_set_intensity", "led_on", "led_off", "led_status",
    "acquire_grid",
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


def _filter_hardware(
    tools_schema: list[dict],
    include_stage: bool,
    include_devices: bool,
) -> list[dict]:
    """Strip hardware-bound tools from the schema unless explicitly requested.

    The model only learns the tools it sees in ``tools``; hiding stage
    or device tools at training time means a tool-call to ``goto_cell``
    or ``shutter_open`` becomes out-of-distribution and the model
    defaults to "I can't do that yet."
    """
    excluded: set[str] = set()
    if not include_stage:
        excluded |= STAGE_TOOL_NAMES
    if not include_devices:
        excluded |= DEVICE_TOOL_NAMES
    if not excluded:
        return tools_schema
    return [t for t in tools_schema
            if t["function"]["name"] not in excluded]


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
# Example builders — one function per category. Order in build_examples()
# defines the order they appear in the JSONL (mixing categories rather
# than clustering by type gives a more balanced gradient signal).
# ---------------------------------------------------------------------------


def _tool_selection_examples(tools: list[dict]) -> list[dict]:
    """15 examples — natural language → exactly one correct tool call.

    Covers each of the 11 active tools at least once + set-vs-run
    ambiguity (the model has to learn that "λ=X yap" is set-only and
    "λ=X yap ve recon" is set-then-run)."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    out.append(_example([
        sys_msg,
        _msg("user", "Şu an ne yüklü?"),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "get_state", {})]),
        _msg("tool", json.dumps({
            "loaded_path": None, "loaded_shape": None, "loaded_dtype": None,
            "recon_params": {}, "stage_position_mm": [0.0, 0.0, 0.0],
        }), tool_call_id="c1", name="get_state"),
        _msg("assistant",
             "Henüz bir şey yüklü değil. Bir hologram yolu ver, açayım — "
             "örneğin: \"~/Documents/holograms/sample.tif yükle\"."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Son yarım saatte ne yaptım?"),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "get_audit_tail", {"limit": 30})]),
        _msg("tool", json.dumps({
            "limit": 30, "count": 3,
            "entries": [
                {"timestamp": "...", "action": "load_hologram", "params": {}},
                {"timestamp": "...", "action": "reconstruct", "params": {}},
                {"timestamp": "...", "action": "qpi", "params": {}},
            ],
        }), tool_call_id="c1", name="get_audit_tail"),
        _msg("assistant", "Son 30 dk: hologram → recon → QPI. Üç adım, hepsi temiz."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "~/Documents/holograms/USAF_ref.tif aç"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/Documents/holograms/USAF_ref.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/<lab>/Documents/holograms/USAF_ref.tif",
            "shape": [2048, 2048], "dtype": "uint16",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant",
             "USAF referans yüklü — 2048×2048 uint16. Kalibrasyon için recon "
             "edeyim mi yoksa önce AF mi koşturayım?"),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Load ~/data/ecoli_dilution_3.tif please."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/data/ecoli_dilution_3.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/<lab>/data/ecoli_dilution_3.tif",
            "shape": [1024, 1024], "dtype": "float32",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant",
             "E. coli örneği yüklü — 1024×1024 float32. AF ile odak z'yi "
             "bulup recon edeyim mi?"),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Mask radius'u 100 yap, başka bir şey yapma."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"mask_radius": 100})]),
        _msg("tool", json.dumps({
            "updated": {"mask_radius": 100},
            "current": {"wavelength_nm": _LAMBDA, "pixel_um": _PIXEL,
                        "z_mm": 8.0, "n_medium": _N_DRY,
                        "mask_radius": 100, "subtract_mean": True,
                        "hann_window": False},
        }), tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "Mask radius 100 olarak ayarlandı. Recon istediğinde haber ver."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Recon çalıştır."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "run_reconstruction", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"shape": [1024, 1024], "phase_mean": -0.02,
                        "phase_std": 1.05, "duration_s": 0.62, "z_mm": 8.0},
        }), tool_call_id="c1", name="run_reconstruction"),
        _msg("assistant", "Recon tamam — phase σ=1.05 rad, 0.62 s. Devam edeyim mi?"),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "0 ile 15 mm arasında autofocus yap, PHASE_VARIANCE metric ile."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 6.8,
                                             "best_metric_value": 0.71}}),
             tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "Best z = 6.8 mm (PHASE_VARIANCE 0.71). Recon edeyim mi?"),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Bana en iyi 5 odak adayını listele, 0-15 mm tarayarak."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "find_focus_candidates",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "n_steps": _Z_STEPS, "metric": "ENTROPY",
                                     "top_k": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"best_z_mm": 7.2,
                        "candidates": [
                            {"z_mm": 7.2, "score": 0.94},
                            {"z_mm": 6.8, "score": 0.91},
                            {"z_mm": 7.6, "score": 0.88},
                            {"z_mm": 6.4, "score": 0.83},
                            {"z_mm": 8.0, "score": 0.79},
                        ]},
        }), tool_call_id="c1", name="find_focus_candidates"),
        _msg("assistant",
             "5 aday: z = 7.2 / 6.8 / 7.6 / 6.4 / 8.0 mm "
             "(ENTROPY 0.94 → 0.79). En tepedeki 7.2'yi seçip recon edeyim mi?"),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "QPI çıkar."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.38, "n_medium": _N_DRY,
                        "total_dry_mass_pg": 142.0, "opd_range_nm": 410.0},
        }), tool_call_id="c1", name="run_qpi"),
        _msg("assistant", "QPI tamam — toplam dry mass ≈ 142 pg, OPD aralığı 410 nm."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Depth map çıkar — 0-12 mm arası, 50 step."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "compute_depth_map",
                                    {"z_min_mm": 0.0, "z_max_mm": 12.0,
                                     "n_steps": 50, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"shape": [1024, 1024], "z_min": 0.0, "z_max": 12.0,
                        "mean_z": 5.2, "valid_pct": 91.0},
        }), tool_call_id="c1", name="compute_depth_map"),
        _msg("assistant",
             "Depth map: ortalama odak düzlemi 5.2 mm, %91 geçerli pixel."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Şu pozisyonda 5 dakika boyunca her 30 saniyede bir kayıtla."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "record_timelapse",
                                    {"n_frames": 10, "interval_s": 30.0,
                                     "run_recon": True, "run_qpi": True})]),
        _msg("tool", json.dumps({
            "ok": True, "n_frames": 10, "interval_s": 30.0, "duration_s": 270.4,
            "frames": [
                {"frame": 0, "elapsed_s": 0.0, "total_dry_mass_pg": 28.1},
                {"frame": 9, "elapsed_s": 270.4, "total_dry_mass_pg": 28.4},
            ],
        }), tool_call_id="c1", name="record_timelapse"),
        _msg("assistant",
             "10 frame, 4.5 dk. Dry mass 28.1 → 28.4 pg, %1 değişim — drift sınırları içinde."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Son recon ne durumdaydı?"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_last_result", {"stage": "reconstruction"})]),
        _msg("tool", json.dumps({
            "stage": "reconstruction", "available": True,
            "summary": {"shape": [1024, 1024], "phase_std": 1.21,
                        "z_mm": 7.2, "duration_s": 0.78},
        }), tool_call_id="c1", name="get_last_result"),
        _msg("assistant", "Son recon: 1024×1024, z=7.2, σ=1.21 rad, 0.78 s. Sağlıklı."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "AF sonucunu hatırlat."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_last_result", {"stage": "autofocus"})]),
        _msg("tool", json.dumps({
            "stage": "autofocus", "available": True,
            "summary": {"best_z_mm": 6.8, "best_metric_value": 0.71,
                        "metric": "PHASE_VARIANCE"},
        }), tool_call_id="c1", name="get_last_result"),
        _msg("assistant", "Son AF: best z = 6.8 mm, PHASE_VARIANCE 0.71."),
    ], tools))

    # set-only (no run)
    out.append(_example([
        sys_msg,
        _msg("user", "Dalga boyunu 532 yap. Recon sonra."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"wavelength_nm": 532.0})]),
        _msg("tool", json.dumps({
            "updated": {"wavelength_nm": 532.0},
            "current": {"wavelength_nm": 532.0, "pixel_um": _PIXEL,
                        "z_mm": 8.0, "n_medium": _N_DRY},
        }), tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "λ = 532 nm. Recon istediğinde haber ver."),
    ], tools))

    # set + run (sequential)
    out.append(_example([
        sys_msg,
        _msg("user", "Dalga boyunu 532 yap ve hemen recon çalıştır."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"wavelength_nm": 532.0})]),
        _msg("tool", json.dumps({
            "updated": {"wavelength_nm": 532.0},
            "current": {"wavelength_nm": 532.0, "pixel_um": _PIXEL, "z_mm": 8.0},
        }), tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"shape": [1024, 1024], "phase_std": 1.18,
                        "duration_s": 0.74, "z_mm": 8.0},
        }), tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "λ = 532 nm uygulandı, recon hazır — phase σ=1.18, 0.74 s."),
    ], tools))

    return out


def _argument_formatting_examples(tools: list[dict]) -> list[dict]:
    """10 examples — type, enum case, unit-name correctness.

    The model has to produce the exact JSON-Schema-valid argument shape,
    not a paraphrase. Common failure modes the examples target: number
    wrapped in string, enum lowercase, wrong unit name, unit conversion."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    out.append(_example([
        sys_msg,
        _msg("user", "Dalga boyu 632.8 nm olsun."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"wavelength_nm": 632.8})]),
        _msg("tool", json.dumps({"updated": {"wavelength_nm": 632.8},
                                 "current": {"wavelength_nm": 632.8}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "λ = 632.8 nm — HeNe baseline."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Kameranın pixel boyutu 3.45 mikron."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"pixel_um": 3.45})]),
        _msg("tool", json.dumps({"updated": {"pixel_um": 3.45},
                                 "current": {"pixel_um": 3.45}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "Pixel pitch = 3.45 µm kaydedildi."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "z'yi 7.2 mm yap."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"z_mm": 7.2})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 7.2}, "current": {"z_mm": 7.2}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "z = 7.2 mm."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "AF'yi laplacian variance ile koş, 0-15 mm."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "LAPLACIAN_VARIANCE",
                                     "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True, "summary": {"best_z_mm": 7.4}}),
             tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "Best z = 7.4 mm (LAPLACIAN_VARIANCE)."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Entropy metric ile depth map, 2-10 mm."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "compute_depth_map",
                                    {"z_min_mm": 2.0, "z_max_mm": 10.0,
                                     "n_steps": 32, "metric": "ENTROPY",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"mean_z": 6.1, "valid_pct": 88.0},
        }), tool_call_id="c1", name="compute_depth_map"),
        _msg("assistant", "Depth map ENTROPY ile çıktı — ortalama z = 6.1 mm, %88 geçerli."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "RBC örneği — n_medium 1.337 (PBS)."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"n_medium": 1.337},
                                 "current": {"n_medium": 1.337}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant",
             "n_medium = 1.337 ayarlandı. QPI çağrısında n_sample=1.41 kullanılacak. "
             "Recon edeyim mi?"),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Mask radius 80 olsun."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"mask_radius": 80})]),
        _msg("tool", json.dumps({"updated": {"mask_radius": 80},
                                 "current": {"mask_radius": 80}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "mask_radius = 80 px."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Hann window aç, mean subtraction kapalı kalsın."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param",
                                    {"hann_window": True, "subtract_mean": False})]),
        _msg("tool", json.dumps({
            "updated": {"hann_window": True, "subtract_mean": False},
            "current": {"hann_window": True, "subtract_mean": False},
        }), tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "Hann window AÇIK, mean subtraction KAPALI."),
    ], tools))

    out.append(_example([
        sys_msg,
        _msg("user", "Son 50 kayıdı listele."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_audit_tail", {"limit": 50})]),
        _msg("tool", json.dumps({"limit": 50, "count": 12, "entries": []}),
             tool_call_id="c1", name="get_audit_tail"),
        _msg("assistant", "Son 50 kayıt isteğine 12 entry geldi — sessiz bir gün."),
    ], tools))

    # unit-conversion: operator says "12 400 µm", schema needs mm
    out.append(_example([
        sys_msg,
        _msg("user", "z'yi 12400 mikrometre yap."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"z_mm": 12.4})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 12.4},
                                 "current": {"z_mm": 12.4}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "12 400 µm = 12.4 mm olarak çevirdim."),
    ], tools))

    return out


def _chain_examples(tools: list[dict]) -> list[dict]:
    """25 examples — multi-step lab workflow patterns.

    Lab workflows almost always chain: load → AF → recon → QPI is the
    canonical sequence. Chain ends with a *summary* the operator can
    read, not raw tool output. Sample-specific n_sample / n_medium
    pairs (RBC, E. coli, Bacillus, etc.) appear here so the model
    learns the lab inventory."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    # 3.1 load → recon
    out.append(_example([
        sys_msg,
        _msg("user", "~/data/sample01.tif aç ve recon et."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/sample01.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/sample01.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.02, "duration_s": 0.71,
                                             "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "Yüklendi + recon: phase σ=1.02 rad, 0.71 s."),
    ], tools))

    # 3.2 load → AF → recon (new sample, unknown z)
    out.append(_example([
        sys_msg,
        _msg("user", "Yeni sample: ~/data/unknown.tif. Odakla ve recon et."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/unknown.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/unknown.tif",
                                 "shape": [1024, 1024], "dtype": "uint16"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 6.4}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.96, "z_mm": 6.4}}),
             tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant", "Yüklendi → AF z=6.4 → recon σ=0.96 rad. QPI istersen söyle."),
    ], tools))

    # 3.3 load → find_focus_candidates → set z → recon
    out.append(_example([
        sys_msg,
        _msg("user",
             "~/data/cells.tif yükle, en iyi 5 odak adayını listele, "
             "en yükseğine ayarla ve recon et."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/cells.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/cells.tif",
                                 "shape": [2048, 2048], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "find_focus_candidates",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": _Z_STEPS, "metric": "ENTROPY",
                                     "top_k": 5})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.2,
                                             "candidates": [{"z_mm": 7.2, "score": 0.92}]}}),
             tool_call_id="c2", name="find_focus_candidates"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param", {"z_mm": 7.2})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 7.2},
                                 "current": {"z_mm": 7.2}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.34, "z_mm": 7.2}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant",
             "Top candidate z=7.2 mm (ENTROPY 0.92). Recon σ=1.34 — kontrast iyi."),
    ], tools))

    # 3.4 load → recon → QPI (HeLa default)
    out.append(_example([
        sys_msg,
        _msg("user", "~/data/hela.tif aç, recon at, QPI çıkar."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/hela.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/hela.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.18, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c3", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.38, "n_medium": _N_WET,
                        "total_dry_mass_pg": 218.0, "opd_range_nm": 380.0},
        }), tool_call_id="c3", name="run_qpi"),
        _msg("assistant",
             "HeLa: dry mass ≈ 218 pg, OPD aralığı 380 nm. Tipik adherent hücre."),
    ], tools))

    # 3.5 RBC chain — n_sample 1.41 + n_medium 1.337
    out.append(_example([
        sys_msg,
        _msg("user", "~/data/rbc_smear.tif aç, recon ve QPI yap. PBS içinde, RBC."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/data/rbc_smear.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/rbc_smear.tif",
                                 "shape": [2048, 2048], "dtype": "uint16"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "set_recon_param", {"n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"n_medium": 1.337},
                                 "current": {"n_medium": 1.337}}),
             tool_call_id="c2", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.88, "z_mm": 8.0}}),
             tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c4", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.41, "n_medium": 1.337,
                        "total_dry_mass_pg": 32.4, "opd_range_nm": 168.0},
        }), tool_call_id="c4", name="run_qpi"),
        _msg("assistant",
             "RBC popülasyonu: dry mass ≈ 32 pg, OPD ≈ 168 nm. Sağlıklı eritrosit."),
    ], tools))

    # 3.6 E. coli chain
    out.append(_example([
        sys_msg,
        _msg("user", "E. coli sıvı kültür: ~/data/ecoli.tif aç, AF + recon + QPI."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/ecoli.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/ecoli.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 5.6}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param",
                                    {"z_mm": 5.6, "n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 5.6, "n_medium": 1.337},
                                 "current": {"z_mm": 5.6}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.62, "z_mm": 5.6}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c5", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.40, "n_medium": 1.337,
                        "total_dry_mass_pg": 0.85, "opd_range_nm": 38.0},
        }), tool_call_id="c5", name="run_qpi"),
        _msg("assistant",
             "E. coli: AF→z=5.6, σ=0.62, QPI dry mass ≈ 0.85 pg/hücre. Tipik."),
    ], tools))

    # 3.7 Bacillus chain
    out.append(_example([
        sys_msg,
        _msg("user", "Bacillus subtilis ~/data/bacillus.tif. Tam pipeline."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/bacillus.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/bacillus.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 6.0}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param",
                                    {"z_mm": 6.0, "n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 6.0, "n_medium": 1.337},
                                 "current": {"z_mm": 6.0}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.71, "z_mm": 6.0}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c5", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.39, "n_medium": 1.337,
                        "total_dry_mass_pg": 2.6, "opd_range_nm": 52.0},
        }), tool_call_id="c5", name="run_qpi"),
        _msg("assistant",
             "Bacillus: dry mass ≈ 2.6 pg, OPD 52 nm. Vegetatif form."),
    ], tools))

    # 3.8 USAF calibration
    out.append(_example([
        sys_msg,
        _msg("user", "Sabah kalibrasyonu — USAF target ~/data/USAF.tif. Yükle, AF, recon."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/USAF.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/USAF.tif",
                                 "shape": [2048, 2048], "dtype": "uint16"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "LAPLACIAN_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.0}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param",
                                    {"z_mm": 7.0, "n_medium": _N_DRY})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 7.0, "n_medium": _N_DRY},
                                 "current": {"z_mm": 7.0}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.42, "z_mm": 7.0}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant",
             "USAF kalibrasyon: best z=7.0, σ=1.42. Bar grupları net görünmeli."),
    ], tools))

    # 3.9 set λ → load → AF → recon
    out.append(_example([
        sys_msg,
        _msg("user", "Önce dalga boyunu 632.8 yap, sonra ~/data/test.tif aç, AF + recon."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param",
                                    {"wavelength_nm": _LAMBDA})]),
        _msg("tool", json.dumps({"updated": {"wavelength_nm": _LAMBDA},
                                 "current": {"wavelength_nm": _LAMBDA}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "load_hologram", {"path": "~/data/test.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/test.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c2", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 8.2}}),
             tool_call_id="c3", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.05, "z_mm": 8.2}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "λ=632.8 → load → AF z=8.2 → recon σ=1.05. Hazır."),
    ], tools))

    # 3.10 load → recon → compute_depth_map (surface topography)
    out.append(_example([
        sys_msg,
        _msg("user",
             "~/data/surface.tif yükle, recon at, sonra depth map çıkar — yüzey topografisi."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/surface.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/surface.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.98, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "compute_depth_map",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": 50, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"mean_z": 6.8, "valid_pct": 92.0,
                        "z_min": _Z_MIN, "z_max": _Z_MAX},
        }), tool_call_id="c3", name="compute_depth_map"),
        _msg("assistant",
             "Yüzey topografisi: ortalama z=6.8 mm, %92 geçerli pixel."),
    ], tools))

    # 3.11 load → recon → record_timelapse (drift gözlem)
    out.append(_example([
        sys_msg,
        _msg("user", "~/data/cell_drift.tif aç, recon, sonra 5 dakika timelapse."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/cell_drift.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/cell_drift.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.12, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "record_timelapse",
                                    {"n_frames": 10, "interval_s": 30.0,
                                     "run_recon": True, "run_qpi": True})]),
        _msg("tool", json.dumps({
            "ok": True, "n_frames": 10, "interval_s": 30.0, "duration_s": 270.5,
            "frames": [
                {"frame": 0, "elapsed_s": 0.0, "total_dry_mass_pg": 215.0},
                {"frame": 9, "elapsed_s": 270.5, "total_dry_mass_pg": 219.0},
            ],
        }), tool_call_id="c3", name="record_timelapse"),
        _msg("assistant",
             "10 frame, 4.5 dk. Dry mass 215 → 219 pg, %2 artış. Drift sınırlarında."),
    ], tools))

    # 3.12 get_state → load → recon
    out.append(_example([
        sys_msg,
        _msg("user", "Devam edelim — load et ve recon at."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "get_state", {})]),
        _msg("tool", json.dumps({
            "loaded_path": None, "loaded_shape": None,
            "recon_params": {"wavelength_nm": _LAMBDA, "z_mm": 8.0},
        }), tool_call_id="c1", name="get_state"),
        _msg("assistant",
             "State boş — hangi dosyayı load edeyim? (~/Documents/holograms altından "
             "bir path ver, sonra recon edeyim.)"),
    ], tools))

    # 3.13 get_state shows loaded → skip load
    out.append(_example([
        sys_msg,
        _msg("user", "Recon at."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "get_state", {})]),
        _msg("tool", json.dumps({
            "loaded_path": "/Users/<lab>/data/sample.tif",
            "loaded_shape": [1024, 1024], "loaded_dtype": "float32",
            "recon_params": {"wavelength_nm": _LAMBDA, "z_mm": 8.0,
                             "n_medium": _N_DRY},
        }), tool_call_id="c1", name="get_state"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.04, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "Sample zaten yüklüydü — recon σ=1.04 rad."),
    ], tools))

    # 3.14 multi-set + run
    out.append(_example([
        sys_msg,
        _msg("user", "λ=632.8, pixel=3.45, mask=80 yap ve recon."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param",
                                    {"wavelength_nm": _LAMBDA, "pixel_um": _PIXEL,
                                     "mask_radius": 80})]),
        _msg("tool", json.dumps({
            "updated": {"wavelength_nm": _LAMBDA, "pixel_um": _PIXEL, "mask_radius": 80},
            "current": {"wavelength_nm": _LAMBDA, "pixel_um": _PIXEL, "mask_radius": 80},
        }), tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.10, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "3 parametre tek seferde set + recon: σ=1.10."),
    ], tools))

    # 3.15 audit-driven summary
    out.append(_example([
        sys_msg,
        _msg("user", "Bugün ne işler döndü? Özet ver."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_audit_tail", {"limit": 100})]),
        _msg("tool", json.dumps({
            "limit": 100, "count": 8,
            "entries": [
                {"action": "load_hologram", "params": {"path": "USAF.tif"}},
                {"action": "reconstruct", "params": {"z_mm": 7.0}},
                {"action": "load_hologram", "params": {"path": "rbc.tif"}},
                {"action": "reconstruct", "params": {"z_mm": 7.4}},
                {"action": "qpi", "params": {}},
                {"action": "load_hologram", "params": {"path": "ecoli.tif"}},
                {"action": "reconstruct", "params": {"z_mm": 5.6}},
                {"action": "qpi", "params": {}},
            ],
        }), tool_call_id="c1", name="get_audit_tail"),
        _msg("assistant",
             "Bugün 3 sample: USAF (recon) → RBC (recon + QPI) → "
             "E. coli (recon + QPI). 8 etkinlik, sıralı çalışma."),
    ], tools))

    # 3.16 set n_medium → recon → QPI (sıvı kültür)
    out.append(_example([
        sys_msg,
        _msg("user", "Sıvı kültür sample — n_medium 1.337 yap, recon at, QPI çıkar."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"n_medium": 1.337},
                                 "current": {"n_medium": 1.337}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.92, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c3", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.38, "n_medium": 1.337,
                        "total_dry_mass_pg": 18.4, "opd_range_nm": 92.0},
        }), tool_call_id="c3", name="run_qpi"),
        _msg("assistant",
             "Sıvı kültür: σ=0.92, dry mass ≈ 18 pg/hücre. n_medium=1.337 etkisi belirgin."),
    ], tools))

    # 3.17 wide AF → narrow AF → recon (refine)
    out.append(_example([
        sys_msg,
        _msg("user", "Önce geniş AF, sonra dar bölgede tekrar, sonra recon."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "PHASE_VARIANCE", "n_steps": 30})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.0}}),
             tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": 6.5, "z_max_mm": 7.5,
                                     "metric": "PHASE_VARIANCE", "n_steps": 50})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 6.92}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "", tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.28, "z_mm": 6.92}}),
             tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant",
             "Coarse AF: 7.0 → fine AF: 6.92 mm → recon σ=1.28. 80 µm rafine."),
    ], tools))

    # 3.18 batch — 3 dosya sırayla
    out.append(_example([
        sys_msg,
        _msg("user",
             "Şu 3 dosyayı sırayla işle: ~/data/s1.tif, ~/data/s2.tif, ~/data/s3.tif. "
             "Her birine recon + QPI."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/s1.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/s1.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.02}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c3", "run_qpi", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"total_dry_mass_pg": 18.0}}),
             tool_call_id="c3", name="run_qpi"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "load_hologram", {"path": "~/data/s2.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/s2.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c4", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c5", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.98}}),
             tool_call_id="c5", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c6", "run_qpi", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"total_dry_mass_pg": 21.0}}),
             tool_call_id="c6", name="run_qpi"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c7", "load_hologram", {"path": "~/data/s3.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/s3.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c7", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c8", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.05}}),
             tool_call_id="c8", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c9", "run_qpi", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"total_dry_mass_pg": 19.5}}),
             tool_call_id="c9", name="run_qpi"),
        _msg("assistant",
             "3 dosya tamam: dry mass 18.0 / 21.0 / 19.5 pg. Ortalama 19.5 pg, σ ≈ 1.5 pg."),
    ], tools))

    # 3.19 set z (skip AF) → recon
    out.append(_example([
        sys_msg,
        _msg("user", "z=6.4 mm biliyorum, recon hemen at."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"z_mm": 6.4})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 6.4},
                                 "current": {"z_mm": 6.4}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.16, "z_mm": 6.4}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "z=6.4 mm uygulandı, recon σ=1.16."),
    ], tools))

    # 3.20 Staphylococcus full pipeline
    out.append(_example([
        sys_msg,
        _msg("user", "Yeni sample: ~/data/staph.tif. Tam pipeline."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/staph.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/staph.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 5.8}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param",
                                    {"z_mm": 5.8, "n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 5.8, "n_medium": 1.337},
                                 "current": {"z_mm": 5.8}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.55, "z_mm": 5.8}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c5", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.40, "n_medium": 1.337,
                        "total_dry_mass_pg": 0.6, "opd_range_nm": 28.0},
        }), tool_call_id="c5", name="run_qpi"),
        _msg("assistant",
             "Staphylococcus: AF z=5.8, σ=0.55, QPI dry mass ≈ 0.6 pg. Kok şekli."),
    ], tools))

    # 3.21 full analysis (recon → depth → QPI)
    out.append(_example([
        sys_msg,
        _msg("user", "Tam analiz: ~/data/full.tif → recon → depth map → QPI."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/full.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/full.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.08, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "compute_depth_map",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": 50, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"mean_z": 7.6, "valid_pct": 89.0},
        }), tool_call_id="c3", name="compute_depth_map"),
        _msg("assistant", "", tool_calls=[_tool_call("c4", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.38, "n_medium": _N_DRY,
                        "total_dry_mass_pg": 95.0, "opd_range_nm": 285.0},
        }), tool_call_id="c4", name="run_qpi"),
        _msg("assistant",
             "Recon σ=1.08 + depth (mean z=7.6, %89) + QPI 95 pg. Tam analiz."),
    ], tools))

    # 3.22 candidates only — no recon
    out.append(_example([
        sys_msg,
        _msg("user", "Adayları bul ama recon atma — önce listeyi göreyim."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "find_focus_candidates",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": _Z_STEPS, "metric": "ENTROPY",
                                     "top_k": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"best_z_mm": 6.4,
                        "candidates": [
                            {"z_mm": 6.4, "score": 0.91},
                            {"z_mm": 6.0, "score": 0.87},
                            {"z_mm": 6.8, "score": 0.84},
                            {"z_mm": 5.6, "score": 0.78},
                            {"z_mm": 7.2, "score": 0.74},
                        ]},
        }), tool_call_id="c1", name="find_focus_candidates"),
        _msg("assistant",
             "5 aday: z = 6.4 / 6.0 / 6.8 / 5.6 / 7.2 mm "
             "(0.91 → 0.74). Hangisinde recon edeyim?"),
    ], tools))

    # 3.23 explicit candidate selection
    out.append(_example([
        sys_msg,
        _msg("user", "İkinci adayda recon at."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_last_result",
                                    {"stage": "find_focus_candidates"})]),
        _msg("tool", json.dumps({
            "stage": "find_focus_candidates", "available": True,
            "summary": {"candidates": [
                {"z_mm": 6.4, "score": 0.91},
                {"z_mm": 6.0, "score": 0.87},
                {"z_mm": 6.8, "score": 0.84},
            ]},
        }), tool_call_id="c1", name="get_last_result"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "set_recon_param", {"z_mm": 6.0})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 6.0},
                                 "current": {"z_mm": 6.0}}),
             tool_call_id="c2", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.21, "z_mm": 6.0}}),
             tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant", "İkinci aday z=6.0 mm — recon σ=1.21."),
    ], tools))

    # 3.24 bead-size verification chain
    out.append(_example([
        sys_msg,
        _msg("user",
             "10 µm polystyrene bead testi: ~/data/bead10.tif → recon → QPI ile boyut."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/bead10.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/bead10.tif",
                                 "shape": [1024, 1024], "dtype": "uint16"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.34, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c3", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.59, "n_medium": _N_DRY,
                        "total_dry_mass_pg": 28.5, "opd_range_nm": 590.0,
                        "particle_diameter_um": 9.8},
        }), tool_call_id="c3", name="run_qpi"),
        _msg("assistant",
             "10 µm bead testi: ölçülen çap 9.8 µm — nominalden %2 sapma. Spec içinde."),
    ], tools))

    # 3.25 Lactobacillus full pipeline
    out.append(_example([
        sys_msg,
        _msg("user", "Lactobacillus: ~/data/lacto.tif. Sıvı kültür, tüm adımlar."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/lacto.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/lacto.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 5.4}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param",
                                    {"z_mm": 5.4, "n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 5.4, "n_medium": 1.337},
                                 "current": {"z_mm": 5.4}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "", tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.68, "z_mm": 5.4}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c5", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.39, "n_medium": 1.337,
                        "total_dry_mass_pg": 1.4, "opd_range_nm": 44.0},
        }), tool_call_id="c5", name="run_qpi"),
        _msg("assistant",
             "Lactobacillus: AF z=5.4, σ=0.68, dry mass ≈ 1.4 pg/hücre. Uzun rod."),
    ], tools))

    return out


def _self_correction_examples(tools: list[dict]) -> list[dict]:
    """8 examples — recover from tool errors and retry cleanly.

    The pattern is "see error → adjust argument → retry" — never
    "retry the same call". Common error types: range swap, enum case,
    path traversal, type mismatch, missing prerequisite, flat AF."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    # 4.1 z_min > z_max swap
    out.append(_example([
        sys_msg,
        _msg("user", "AF: 15'ten 0'a tara."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 15.0, "z_max_mm": 0.0})]),
        _msg("tool", json.dumps({
            "error": "z_max_mm (0.0) must exceed z_min_mm (15.0)",
        }), tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "PHASE_VARIANCE",
                                     "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.0}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "Sıralı yazılmıştı, ben min↔max çevirdim — best z = 7.0 mm."),
    ], tools))

    # 4.2 enum case fix
    out.append(_example([
        sys_msg,
        _msg("user", "AF, phase variance ile 0-15."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "phase_variance"})]),
        _msg("tool", json.dumps({
            "error": "invalid arguments",
            "details": "metric must be one of PHASE_VARIANCE, ENTROPY, LAPLACIAN_VARIANCE, ...",
        }), tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "PHASE_VARIANCE",
                                     "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 6.8}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "Enum case düzeltildi (PHASE_VARIANCE) — best z=6.8 mm."),
    ], tools))

    # 4.3 path home dışı — refuse + ask
    out.append(_example([
        sys_msg,
        _msg("user", "/etc/passwd'i aç."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "/etc/passwd"})]),
        _msg("tool", json.dumps({
            "error": "security check failed",
            "details": "path is outside the home directory",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant",
             "Ev klasörü dışındaki yolları açamam (path-traversal koruması). "
             "Hologram dosyan ev altında bir yerde mi? Tam yolu ver, açayım."),
    ], tools))

    # 4.4 QPI before recon → run recon first
    out.append(_example([
        sys_msg,
        _msg("user", "QPI çıkar."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "error": "no reconstruction available",
            "details": "QPI requires a recent reconstruction; run run_reconstruction first.",
        }), tool_call_id="c1", name="run_qpi"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.10, "z_mm": 8.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"total_dry_mass_pg": 142.0, "opd_range_nm": 405.0},
        }), tool_call_id="c3", name="run_qpi"),
        _msg("assistant",
             "Recon yoktu — önce onu koşturdum (σ=1.10), sonra QPI: dry mass ≈ 142 pg."),
    ], tools))

    # 4.5 file not found
    out.append(_example([
        sys_msg,
        _msg("user", "~/data/missing.tif yükle."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/missing.tif"})]),
        _msg("tool", json.dumps({
            "error": "FileNotFoundError",
            "message": "/Users/<lab>/data/missing.tif: no such file",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant",
             "Dosya bulunamadı — ~/data/ altında missing.tif yok. "
             "Doğru yolu yazar mısın?"),
    ], tools))

    # 4.6 invalid n_medium > n_sample (QPI nonsense)
    out.append(_example([
        sys_msg,
        _msg("user", "QPI'yı n_sample=1.30, n_medium=1.337 ile çıkar."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"n_medium": 1.337},
                                 "current": {"n_medium": 1.337}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant",
             "Bir saniye — n_sample (1.30) < n_medium (1.337) olamaz; "
             "QPI dry mass formülü negatif çıkar. RBC için 1.41, E. coli için 1.40 "
             "tipik. Hangi sample bu, doğru n_sample'ı yazayım?"),
    ], tools))

    # 4.7 invalid arg type — string instead of float
    out.append(_example([
        sys_msg,
        _msg("user", "λ '532 nm' yap."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param",
                                    {"wavelength_nm": "532 nm"})]),
        _msg("tool", json.dumps({
            "error": "invalid arguments",
            "details": "wavelength_nm must be a number, got string '532 nm'",
        }), tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "set_recon_param",
                                    {"wavelength_nm": 532.0})]),
        _msg("tool", json.dumps({"updated": {"wavelength_nm": 532.0},
                                 "current": {"wavelength_nm": 532.0}}),
             tool_call_id="c2", name="set_recon_param"),
        _msg("assistant", "Birim string'di, sayıya çevirdim: λ = 532 nm."),
    ], tools))

    # 4.8 AF flat → widen range
    out.append(_example([
        sys_msg,
        _msg("user", "AF 7-8 mm arasında."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 7.0, "z_max_mm": 8.0,
                                     "metric": "PHASE_VARIANCE", "n_steps": 20})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"best_z_mm": 7.0, "best_metric_value": 0.21,
                        "warning": "metric profile is flat; peak is ambiguous"},
        }), tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 5.6,
                                             "best_metric_value": 0.78}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant",
             "7-8 mm'de metric flat'ti — geniş aralık (0-15) ile best z=5.6 mm bulundu."),
    ], tools))

    return out


def _refusal_examples(tools: list[dict]) -> list[dict]:
    """5 examples — refuse out-of-scope or unsafe requests.

    Three of these don't call any tool — model replies with text only.
    Two call a tool and surface the security error. The mix teaches
    the model when to *not* call vs. when calling+failing is fine."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    # 5.1 system file read — refuse without tool call
    out.append(_example([
        sys_msg,
        _msg("user", "/etc/shadow'u oku."),
        _msg("assistant",
             "Bu dosyaya erişimim yok — sistem dosyaları (/etc altı) ev klasörü dışında, "
             "AI panelin path-traversal koruması bunu engelliyor. Hologram dosyan başka "
             "bir yerde mi?"),
    ], tools))

    # 5.2 reset all settings — irreversible, no tool
    out.append(_example([
        sys_msg,
        _msg("user", "Tüm ayarları sıfırla."),
        _msg("assistant",
             "Toplu sıfırlama tool'um yok — bu geri alınamaz bir işlem ve Settings → "
             "Reset to defaults menüsünden yapılması gerekir. UI üzerinden gitmeni "
             "öneririm; istersen hangi parametrenin değişmesini istediğini söyle, "
             "onu tek tek geri alayım."),
    ], tools))

    # 5.3 internet fetch — no tool
    out.append(_example([
        sys_msg,
        _msg("user", "İnternetten USAF target görüntüsünü indir."),
        _msg("assistant",
             "İnternet erişimi olan bir tool'um yok — lokal dosyalarla çalışıyorum. "
             "USAF görüntün lab'da bir yerde olmalı; ~/Documents altından yolunu "
             "verirsen yükleyebilirim."),
    ], tools))

    # 5.4 hallucinated tool — Cellpose
    out.append(_example([
        sys_msg,
        _msg("user", "Cellpose ile hücreleri segment et."),
        _msg("assistant",
             "Cellpose entegrasyonu yok (v3.0 roadmap'inde). Şu an "
             "compute_depth_map veya QPI ile dolaylı segmentasyon yapabilirim — "
             "phase haritası üzerinden eşik tabanlı maskeleme. İstersen recon + QPI "
             "çıkarayım, sonuç haritasını birlikte yorumlayalım."),
    ], tools))

    # 5.5 stage move attempt — hardware not connected
    out.append(_example([
        sys_msg,
        _msg("user", "Stage'i x=2 mm'e götür."),
        _msg("assistant",
             "Motorize stage henüz bağlı değil — stage hareket tool'ları bu sürümde "
             "askıya alındı. Manuel stage'i elle hareket ettirip yeni hologramı "
             "load edersen, bu yeni pozisyonda recon/QPI yapabilirim."),
    ], tools))

    return out


def _domain_language_examples(tools: list[dict]) -> list[dict]:
    """15 examples — Turkish prose maps to English/numeric tool args.

    Turkish ↔ English term pairs the model has to learn:
    dalga boyu/wavelength, odakla/AF, kuru kütle/dry mass,
    eritrosit/RBC, basil/coccus, derinlik/depth. Lab abbreviations
    (AF, QPI, OPD, FFT) appear bare in operator prompts — model has
    to expand them to the right tool."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    # 6.1 dalga boyu → wavelength_nm
    out.append(_example([
        sys_msg,
        _msg("user", "Dalga boyu 660 nm yap, kırmızı diode laser var."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"wavelength_nm": 660.0})]),
        _msg("tool", json.dumps({"updated": {"wavelength_nm": 660.0},
                                 "current": {"wavelength_nm": 660.0}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "λ = 660 nm — kırmızı diode setup."),
    ], tools))

    # 6.2 odakla → run_autofocus
    out.append(_example([
        sys_msg,
        _msg("user", "Odakla."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.2}}),
             tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "Odak z = 7.2 mm bulundu."),
    ], tools))

    # 6.3 kuru kütle → run_qpi
    out.append(_example([
        sys_msg,
        _msg("user", "Kuru kütle çıkar."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"total_dry_mass_pg": 28.0, "opd_range_nm": 145.0,
                        "n_sample": 1.41, "n_medium": 1.337},
        }), tool_call_id="c1", name="run_qpi"),
        _msg("assistant", "Kuru kütle ≈ 28 pg, OPD aralığı 145 nm."),
    ], tools))

    # 6.4 OPD term
    out.append(_example([
        sys_msg,
        _msg("user", "OPD ne durumda?"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_last_result", {"stage": "qpi"})]),
        _msg("tool", json.dumps({
            "stage": "qpi", "available": True,
            "summary": {"opd_range_nm": 145.0, "total_dry_mass_pg": 28.0},
        }), tool_call_id="c1", name="get_last_result"),
        _msg("assistant", "Son QPI'da OPD aralığı 145 nm."),
    ], tools))

    # 6.5 büyütme — explanation, no tool change
    out.append(_example([
        sys_msg,
        _msg("user", "50x objektifle çekildi, parametreleri ayarla."),
        _msg("assistant",
             "Büyütme bilgisi (50×) recon parametresinden ziyade pixel pitch'in "
             "etkin değerini etkiler. Settings → Reconstruction'da `magnification` "
             "alanı zaten 50 (lab default). Recon parametrelerinde değişiklik "
             "gerekmiyor — λ ve pixel zaten lab profilinde."),
    ], tools))

    # 6.6 FFT backend
    out.append(_example([
        sys_msg,
        _msg("user", "FFT backend'i numpy yap."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"fft_backend": "numpy"})]),
        _msg("tool", json.dumps({"updated": {"fft_backend": "numpy"},
                                 "current": {"fft_backend": "numpy"}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "FFT backend = numpy. CPU yolundan gidecek."),
    ], tools))

    # 6.7 eritrosit → RBC
    out.append(_example([
        sys_msg,
        _msg("user", "Eritrosit örneği yükledim ~/data/eritrosit.tif. PBS içinde."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/data/eritrosit.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/eritrosit.tif",
                                 "shape": [2048, 2048], "dtype": "uint16"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "set_recon_param", {"n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"n_medium": 1.337},
                                 "current": {"n_medium": 1.337}}),
             tool_call_id="c2", name="set_recon_param"),
        _msg("assistant",
             "Eritrosit (RBC) PBS içinde — n_medium=1.337 ayarlandı. "
             "QPI'da n_sample=1.41 kullanacağım. Recon edeyim mi?"),
    ], tools))

    # 6.8 basil → ambiguous, ask
    out.append(_example([
        sys_msg,
        _msg("user", "Basil sample yükledim."),
        _msg("assistant",
             "Basil = rod şekilli bakteri. Hangisi: E. coli (n_sample 1.40), "
             "Bacillus subtilis (1.39), Pseudomonas (1.39), yoksa başka? "
             "Hangi suşa göre QPI ayarlayayım?"),
    ], tools))

    # 6.9 kok → coccus = Staph
    out.append(_example([
        sys_msg,
        _msg("user", "Kok bakterisi, ~/data/staph.tif. Tam pipeline."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/staph.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/staph.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "set_recon_param", {"n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"n_medium": 1.337},
                                 "current": {"n_medium": 1.337}}),
             tool_call_id="c2", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.51, "z_mm": 8.0}}),
             tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.40, "n_medium": 1.337,
                        "total_dry_mass_pg": 0.55, "opd_range_nm": 24.0},
        }), tool_call_id="c4", name="run_qpi"),
        _msg("assistant",
             "Staphylococcus (kok): recon σ=0.51, dry mass ≈ 0.55 pg/hücre."),
    ], tools))

    # 6.10 USAF target
    out.append(_example([
        sys_msg,
        _msg("user", "USAF target ile kalibrasyon yapacağım."),
        _msg("assistant",
             "Kalibrasyon için USAF dosya yolunu ver — yükleyeyim, AF + recon koşturayım. "
             "Sonra phase profilinden bar grupları görünür mü diye kontrol edersin "
             "(n_medium=1.0 hava varsayımıyla)."),
    ], tools))

    # 6.11 haritalama — stage disabled
    out.append(_example([
        sys_msg,
        _msg("user", "Sample'ı haritala."),
        _msg("assistant",
             "Sample haritalama (map_sample_grid) motorize stage gerektiriyor; "
             "şu an stage manuel olduğu için bu tool askıda. Manuel taramada her "
             "pozisyonu ayrı .tif olarak kaydedersen, hepsini sırayla load → recon "
             "→ QPI ile işleyebilirim."),
    ], tools))

    # 6.12 geçen saat → audit_tail
    out.append(_example([
        sys_msg,
        _msg("user", "Geçen saat ne yaptım?"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_audit_tail", {"limit": 60})]),
        _msg("tool", json.dumps({
            "limit": 60, "count": 5,
            "entries": [
                {"action": "load_hologram", "params": {}},
                {"action": "autofocus", "params": {}},
                {"action": "reconstruct", "params": {}},
                {"action": "qpi", "params": {}},
                {"action": "load_hologram", "params": {}},
            ],
        }), tool_call_id="c1", name="get_audit_tail"),
        _msg("assistant",
             "Son saatte: 1 sample yüklendi → AF → recon → QPI tamamlandı, "
             "ardından ikinci sample yüklendi (henüz işlenmedi)."),
    ], tools))

    # 6.13 derinlik haritası → compute_depth_map
    out.append(_example([
        sys_msg,
        _msg("user", "Derinlik haritası çıkar, geniş tarama."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "compute_depth_map",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": 60, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"mean_z": 6.4, "valid_pct": 86.0,
                        "z_min": _Z_MIN, "z_max": _Z_MAX},
        }), tool_call_id="c1", name="compute_depth_map"),
        _msg("assistant",
             "Derinlik haritası: ortalama z=6.4 mm, %86 geçerli pixel."),
    ], tools))

    # 6.14 AF abbreviation
    out.append(_example([
        sys_msg,
        _msg("user", "AF yap 0-15."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.6}}),
             tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "AF tamam — best z = 7.6 mm."),
    ], tools))

    # 6.15 QPI abbreviation
    out.append(_example([
        sys_msg,
        _msg("user", "QPI."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"total_dry_mass_pg": 95.0, "opd_range_nm": 280.0},
        }), tool_call_id="c1", name="run_qpi"),
        _msg("assistant", "QPI: dry mass 95 pg, OPD 280 nm."),
    ], tools))

    return out


def _conversational_style_examples(tools: list[dict]) -> list[dict]:
    """12 examples — short, action-first replies.

    Models the *good* conversational style described in
    AI_FINETUNE_DATA.md §7: numbers + interpretation + next-step
    question. The opposite (long, no question, ceremonial) is
    deliberately *not* shown — the model never sees that pattern
    paired with positive completion."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    # 7.1 recon → short numeric + next step
    out.append(_example([
        sys_msg,
        _msg("user", "Recon."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "run_reconstruction", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"shape": [1024, 1024], "phase_mean": 0.01,
                        "phase_std": 1.21, "duration_s": 0.78, "z_mm": 8.0},
        }), tool_call_id="c1", name="run_reconstruction"),
        _msg("assistant", "Recon — phase σ=1.21 rad, 0.78 s. QPI ister misin?"),
    ], tools))

    # 7.2 AF → short summary + next
    out.append(_example([
        sys_msg,
        _msg("user", "AF 0-15."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 6.4,
                                             "best_metric_value": 0.81}}),
             tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "Best z=6.4 mm (PV 0.81). Recon edeyim mi?"),
    ], tools))

    # 7.3 QPI → numbers + interpretation
    out.append(_example([
        sys_msg,
        _msg("user", "QPI."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"total_dry_mass_pg": 215.0, "opd_range_nm": 380.0,
                        "n_sample": 1.38, "n_medium": _N_DRY},
        }), tool_call_id="c1", name="run_qpi"),
        _msg("assistant", "Dry mass 215 pg, OPD 380 nm. Tipik adherent hücre. Başka analiz?"),
    ], tools))

    # 7.4 depth_map → percent valid
    out.append(_example([
        sys_msg,
        _msg("user", "Depth map çıkar."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "compute_depth_map",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": 50, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"mean_z": 7.2, "valid_pct": 88.0},
        }), tool_call_id="c1", name="compute_depth_map"),
        _msg("assistant", "Mean z=7.2 mm, %88 geçerli pixel."),
    ], tools))

    # 7.5 timelapse → trend
    out.append(_example([
        sys_msg,
        _msg("user", "5 dk timelapse."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "record_timelapse",
                                    {"n_frames": 10, "interval_s": 30.0,
                                     "run_recon": True, "run_qpi": True})]),
        _msg("tool", json.dumps({
            "ok": True, "n_frames": 10, "duration_s": 270.4,
            "frames": [
                {"frame": 0, "elapsed_s": 0.0, "total_dry_mass_pg": 28.0},
                {"frame": 4, "elapsed_s": 120.0, "total_dry_mass_pg": 29.5},
                {"frame": 9, "elapsed_s": 270.4, "total_dry_mass_pg": 31.2},
            ],
        }), tool_call_id="c1", name="record_timelapse"),
        _msg("assistant",
             "10 frame: 28 → 31.2 pg (%11 artış 4.5 dk'da). Hücre büyüme fazında."),
    ], tools))

    # 7.6 get_state empty → suggest
    out.append(_example([
        sys_msg,
        _msg("user", "Ne durumdayım?"),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "get_state", {})]),
        _msg("tool", json.dumps({
            "loaded_path": None, "loaded_shape": None,
            "recon_params": {}, "stage_position_mm": [0.0, 0.0, 0.0],
        }), tool_call_id="c1", name="get_state"),
        _msg("assistant", "Boş — bir hologram yolu ver, başlayalım."),
    ], tools))

    # 7.7 audit_tail summary
    out.append(_example([
        sys_msg,
        _msg("user", "Bugün özet."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_audit_tail", {"limit": 100})]),
        _msg("tool", json.dumps({
            "limit": 100, "count": 6,
            "entries": [
                {"action": "load_hologram"}, {"action": "reconstruct"},
                {"action": "qpi"}, {"action": "load_hologram"},
                {"action": "reconstruct"}, {"action": "qpi"},
            ],
        }), tool_call_id="c1", name="get_audit_tail"),
        _msg("assistant", "2 sample, her birine recon + QPI. Sessiz bir gün."),
    ], tools))

    # 7.8 load → ask next-step
    out.append(_example([
        sys_msg,
        _msg("user", "~/data/x.tif yükle."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/x.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/x.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "1024×1024 float32 yüklü. Recon mu, AF mi?"),
    ], tools))

    # 7.9 set → confirm + suggest
    out.append(_example([
        sys_msg,
        _msg("user", "z=6.0."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"z_mm": 6.0})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 6.0},
                                 "current": {"z_mm": 6.0}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "z=6.0 mm. Recon?"),
    ], tools))

    # 7.10 candidates → numbered list
    out.append(_example([
        sys_msg,
        _msg("user", "Top 3 aday."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "find_focus_candidates",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": _Z_STEPS, "metric": "ENTROPY",
                                     "top_k": 3})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"candidates": [
                {"z_mm": 6.8, "score": 0.92},
                {"z_mm": 7.2, "score": 0.88},
                {"z_mm": 6.4, "score": 0.84},
            ]},
        }), tool_call_id="c1", name="find_focus_candidates"),
        _msg("assistant", "1) z=6.8 (0.92), 2) z=7.2 (0.88), 3) z=6.4 (0.84). Hangisi?"),
    ], tools))

    # 7.11 batch summary
    out.append(_example([
        sys_msg,
        _msg("user", "İki dosya işle: ~/data/a.tif, ~/data/b.tif → recon + QPI."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/a.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/a.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.0}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c3", "run_qpi", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"total_dry_mass_pg": 18.0}}),
             tool_call_id="c3", name="run_qpi"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "load_hologram", {"path": "~/data/b.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/b.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c4", name="load_hologram"),
        _msg("assistant", "", tool_calls=[_tool_call("c5", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.05}}),
             tool_call_id="c5", name="run_reconstruction"),
        _msg("assistant", "", tool_calls=[_tool_call("c6", "run_qpi", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"total_dry_mass_pg": 19.0}}),
             tool_call_id="c6", name="run_qpi"),
        _msg("assistant", "a: 18 pg, b: 19 pg. Ortalama ≈ 18.5 pg."),
    ], tools))

    # 7.12 ambiguous prompt → short clarification
    out.append(_example([
        sys_msg,
        _msg("user", "Şunu yap."),
        _msg("assistant",
             "Hangisi: load, AF, recon, yoksa QPI? Tek kelime yeter."),
    ], tools))

    return out


def _negative_examples(tools: list[dict]) -> list[dict]:
    """5 examples — 'I don't have that tool' refusals.

    Stops the model from inventing handlers. None of these call any
    tool — the assistant explains *why* the request can't be served
    and points at a UI alternative or roadmap item."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    # 8.1 Cellpose / deep segmentation
    out.append(_example([
        sys_msg,
        _msg("user", "Cellpose ile hücreleri segment et."),
        _msg("assistant",
             "Cellpose entegrasyonu yok (v3.0 roadmap). Şu an phase haritasından "
             "eşik tabanlı dolaylı segmentasyon mümkün — recon + QPI çıkarayım, "
             "QPI panel'inden segmentasyon maskesi gösterilir."),
    ], tools))

    # 8.2 colormap / visualisation tweak
    out.append(_example([
        sys_msg,
        _msg("user", "Phase görüntüsünü inferno colormap ile göster."),
        _msg("assistant",
             "Görselleştirme tool'um yok — colormap UI tarafında. "
             "View → Colormap menüsünden inferno seçebilirsin. Phase verisi hazırsa "
             "değişiklik anında yansır."),
    ], tools))

    # 8.3 Excel export
    out.append(_example([
        sys_msg,
        _msg("user", "QPI sonuçlarını Excel'e aktar."),
        _msg("assistant",
             "Excel export tool'um yok. Mevcut export: File → Export Bundle → ZIP "
             "(JSON + CSV + .tif çıktıları). CSV'yi Excel'de açabilirsin. İstersen "
             "şu anki sonuçları audit-tail üzerinden özetleyeyim."),
    ], tools))

    # 8.4 FastAPI / web server
    out.append(_example([
        sys_msg,
        _msg("user", "FastAPI server başlat, sonuçları API üzerinden vereyim."),
        _msg("assistant",
             "Server başlatma tool'um yok — bu uygulamanın kapsamı dışı. Eğer dış "
             "sistemle entegrasyon lazımsa export bundle (ZIP) JSON + CSV verir, "
             "onu pipeline'ından besleyebilirsin."),
    ], tools))

    # 8.5 matplotlib / direct plot
    out.append(_example([
        sys_msg,
        _msg("user", "Phase histogramını pyplot ile çiz."),
        _msg("assistant",
             "Direkt plot tool'um yok — histogram QPI panelinde otomatik çıkıyor "
             "(Tools → QPI Histogram). Phase istatistikleri get_last_result ile "
             "alınabilir; istersen yazılı özet veririm."),
    ], tools))

    return out


def _lab_specific_examples(tools: list[dict]) -> list[dict]:
    """5 examples — end-to-end realistic lab sessions for the actual
    sample inventory (USAF, RBC, E. coli, bead, Bacillus). These are
    the high-value ground-truth examples — they should mirror what
    the operator literally types in a real lab session."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    # 9.1 USAF morning calibration
    out.append(_example([
        sys_msg,
        _msg("user",
             "Sabah kalibrasyonu — USAF target ~/Documents/holograms/USAF_morning.tif. "
             "Yükle, AF, recon, sonra phase σ rapor et."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/Documents/holograms/USAF_morning.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/<lab>/Documents/holograms/USAF_morning.tif",
            "shape": [2048, 2048], "dtype": "uint16",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "LAPLACIAN_VARIANCE",
                                     "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.0,
                                             "best_metric_value": 1.42}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"shape": [2048, 2048], "phase_std": 1.42,
                        "z_mm": 7.0, "duration_s": 1.21},
        }), tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant",
             "Sabah kalibrasyonu: USAF z=7.0 mm, recon σ=1.42 rad. "
             "Bar grupları net görünmeli — phase panel'de Group 7 / Element 6'ya "
             "kadar çözünmeyi gözle kontrol et."),
    ], tools))

    # 9.2 RBC dry-mass session
    out.append(_example([
        sys_msg,
        _msg("user",
             "RBC kan smear — ~/Documents/holograms/blood_2026_04_28.tif. PBS içinde, "
             "kuru kütle dağılımı lazım."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/Documents/holograms/blood_2026_04_28.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/<lab>/Documents/holograms/blood_2026_04_28.tif",
            "shape": [2048, 2048], "dtype": "uint16",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE",
                                     "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.4}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param",
                                    {"z_mm": 7.4, "n_medium": 1.337})]),
        _msg("tool", json.dumps({
            "updated": {"z_mm": 7.4, "n_medium": 1.337},
            "current": {"z_mm": 7.4, "n_medium": 1.337},
        }), tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.92, "z_mm": 7.4}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c5", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.41, "n_medium": 1.337,
                        "total_dry_mass_pg": 31.8, "opd_range_nm": 165.0,
                        "cell_count_estimate": 142},
        }), tool_call_id="c5", name="run_qpi"),
        _msg("assistant",
             "RBC popülasyonu (~142 hücre): toplam dry mass 31.8 pg. "
             "Detaylı dağılım için QPI panel'inde histogramı kontrol et."),
    ], tools))

    # 9.3 E. coli enumeration
    out.append(_example([
        sys_msg,
        _msg("user",
             "E. coli OD600=0.4 sıvı kültür, ~/Documents/holograms/ecoli_t0.tif. "
             "Tam pipeline + depth map (rod oryantasyonu)."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/Documents/holograms/ecoli_t0.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/<lab>/Documents/holograms/ecoli_t0.tif",
            "shape": [1024, 1024], "dtype": "float32",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE",
                                     "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 5.6}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param",
                                    {"z_mm": 5.6, "n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 5.6, "n_medium": 1.337},
                                 "current": {"z_mm": 5.6}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.62, "z_mm": 5.6}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c5", "compute_depth_map",
                                    {"z_min_mm": 4.5, "z_max_mm": 6.5,
                                     "n_steps": 30, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 3})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"mean_z": 5.6, "valid_pct": 78.0},
        }), tool_call_id="c5", name="compute_depth_map"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c6", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.40, "n_medium": 1.337,
                        "total_dry_mass_pg": 0.92, "opd_range_nm": 38.0},
        }), tool_call_id="c6", name="run_qpi"),
        _msg("assistant",
             "E. coli OD600=0.4: AF z=5.6, σ=0.62, depth %78 geçerli "
             "(rod için pencere=3 daralttım), QPI tek-hücre dry mass ≈ 0.92 pg."),
    ], tools))

    # 9.4 Bead size verification
    out.append(_example([
        sys_msg,
        _msg("user",
             "10 µm polystyrene bead kalibrasyonu, ~/Documents/holograms/bead10_test.tif. "
             "Boyut doğrula."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/Documents/holograms/bead10_test.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/<lab>/Documents/holograms/bead10_test.tif",
            "shape": [1024, 1024], "dtype": "uint16",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "ENTROPY", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 8.2}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param", {"z_mm": 8.2})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 8.2},
                                 "current": {"z_mm": 8.2}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.38, "z_mm": 8.2}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c5", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.59, "n_medium": _N_DRY,
                        "particle_diameter_um": 9.85,
                        "total_dry_mass_pg": 28.2, "opd_range_nm": 580.0},
        }), tool_call_id="c5", name="run_qpi"),
        _msg("assistant",
             "10 µm bead: ölçülen çap 9.85 µm (nominal 10.0 ± 0.1 µm). "
             "Sapma %1.5 — sistem doğru raporluyor."),
    ], tools))

    # 9.5 Bacillus mixed (vegetative + spore)
    out.append(_example([
        sys_msg,
        _msg("user",
             "Bacillus subtilis karışık (vegetatif + spor), "
             "~/Documents/holograms/bacillus_mix.tif. Tam pipeline."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram",
                                    {"path": "~/Documents/holograms/bacillus_mix.tif"})]),
        _msg("tool", json.dumps({
            "path": "/Users/<lab>/Documents/holograms/bacillus_mix.tif",
            "shape": [1024, 1024], "dtype": "float32",
        }), tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE",
                                     "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 6.0}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "set_recon_param",
                                    {"z_mm": 6.0, "n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 6.0, "n_medium": 1.337},
                                 "current": {"z_mm": 6.0}}),
             tool_call_id="c3", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.82, "z_mm": 6.0}}),
             tool_call_id="c4", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c5", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.39, "n_medium": 1.337,
                        "total_dry_mass_pg": 4.1, "opd_range_nm": 76.0,
                        "phase_distribution_bimodal": True},
        }), tool_call_id="c5", name="run_qpi"),
        _msg("assistant",
             "Bacillus karışık: AF z=6.0, σ=0.82, dry mass ≈ 4.1 pg. "
             "Phase dağılımı bimodal (vegetatif vs spor) — QPI panel'inde net görür."),
    ], tools))

    return out


def _holdout_examples(tools: list[dict]) -> list[dict]:
    """15 examples — held-out eval set, NEVER trained on.

    Same shape as the training set, deliberately different surface
    phrasing so eval measures generalisation, not memorisation. Spread
    across categories: 2 tool-selection, 1 arg-format, 4 chain,
    2 self-correction, 1 refusal, 2 domain-TR, 2 conversational,
    1 negative."""
    sys_msg = _msg("system", _system_prompt())
    out: list[dict] = []

    # H1 — tool selection: get_state
    out.append(_example([
        sys_msg,
        _msg("user", "Şu anki durumumu söyle."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "get_state", {})]),
        _msg("tool", json.dumps({
            "loaded_path": "/Users/<lab>/data/x.tif",
            "loaded_shape": [1024, 1024], "loaded_dtype": "float32",
            "recon_params": {"wavelength_nm": _LAMBDA, "z_mm": 7.2},
        }), tool_call_id="c1", name="get_state"),
        _msg("assistant", "x.tif yüklü (1024×1024), z=7.2 mm. Recon hazır."),
    ], tools))

    # H2 — tool selection: run_qpi
    out.append(_example([
        sys_msg,
        _msg("user", "Kuru kütlesini ölç."),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"total_dry_mass_pg": 142.0, "opd_range_nm": 380.0},
        }), tool_call_id="c1", name="run_qpi"),
        _msg("assistant", "Dry mass 142 pg, OPD 380 nm."),
    ], tools))

    # H3 — argument formatting: λ
    out.append(_example([
        sys_msg,
        _msg("user", "532 nanometre dalga boyu."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"wavelength_nm": 532.0})]),
        _msg("tool", json.dumps({"updated": {"wavelength_nm": 532.0},
                                 "current": {"wavelength_nm": 532.0}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "λ = 532 nm."),
    ], tools))

    # H4 — chain: load → AF → recon
    out.append(_example([
        sys_msg,
        _msg("user", "~/data/new.tif aç, focus bul ve recon at."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/new.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/new.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.0}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.10, "z_mm": 7.0}}),
             tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant", "Yüklendi → AF z=7.0 → recon σ=1.10."),
    ], tools))

    # H5 — chain: RBC pipeline
    out.append(_example([
        sys_msg,
        _msg("user", "RBC sample ~/data/rbc2.tif aç ve dry mass ölç."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/rbc2.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/rbc2.tif",
                                 "shape": [2048, 2048], "dtype": "uint16"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "set_recon_param", {"n_medium": 1.337})]),
        _msg("tool", json.dumps({"updated": {"n_medium": 1.337},
                                 "current": {"n_medium": 1.337}}),
             tool_call_id="c2", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 0.91}}),
             tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c4", "run_qpi", {})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"n_sample": 1.41, "n_medium": 1.337,
                        "total_dry_mass_pg": 30.5, "opd_range_nm": 162.0},
        }), tool_call_id="c4", name="run_qpi"),
        _msg("assistant", "RBC: dry mass ≈ 30 pg, OPD 162 nm."),
    ], tools))

    # H6 — chain: load → recon → depth_map
    out.append(_example([
        sys_msg,
        _msg("user", "~/data/topo.tif yükle, recon, derinlik haritası."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "load_hologram", {"path": "~/data/topo.tif"})]),
        _msg("tool", json.dumps({"path": "/Users/<lab>/data/topo.tif",
                                 "shape": [1024, 1024], "dtype": "float32"}),
             tool_call_id="c1", name="load_hologram"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.04}}),
             tool_call_id="c2", name="run_reconstruction"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "compute_depth_map",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": 50, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"mean_z": 7.0, "valid_pct": 90.0},
        }), tool_call_id="c3", name="compute_depth_map"),
        _msg("assistant", "Yüklendi + recon + depth: mean z=7.0 mm, %90 geçerli."),
    ], tools))

    # H7 — chain: candidates + manual pick
    out.append(_example([
        sys_msg,
        _msg("user", "Top 3 odak adayını listele, ortancayı seç recon at."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "find_focus_candidates",
                                    {"z_min_mm": _Z_MIN, "z_max_mm": _Z_MAX,
                                     "n_steps": _Z_STEPS, "metric": "ENTROPY",
                                     "top_k": 3})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"candidates": [
                {"z_mm": 7.0, "score": 0.91},
                {"z_mm": 6.6, "score": 0.86},
                {"z_mm": 7.4, "score": 0.81},
            ]},
        }), tool_call_id="c1", name="find_focus_candidates"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "set_recon_param", {"z_mm": 6.6})]),
        _msg("tool", json.dumps({"updated": {"z_mm": 6.6},
                                 "current": {"z_mm": 6.6}}),
             tool_call_id="c2", name="set_recon_param"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c3", "run_reconstruction", {})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"phase_std": 1.18, "z_mm": 6.6}}),
             tool_call_id="c3", name="run_reconstruction"),
        _msg("assistant",
             "3 aday: 7.0 / 6.6 / 7.4 mm. Ortanca z=6.6 — recon σ=1.18."),
    ], tools))

    # H8 — self-correction: z swap
    out.append(_example([
        sys_msg,
        _msg("user", "Autofocus 12'den 4'e tara."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 12.0, "z_max_mm": 4.0})]),
        _msg("tool", json.dumps({
            "error": "z_max_mm (4.0) must exceed z_min_mm (12.0)",
        }), tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "run_autofocus",
                                    {"z_min_mm": 4.0, "z_max_mm": 12.0,
                                     "metric": "PHASE_VARIANCE",
                                     "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.8}}),
             tool_call_id="c2", name="run_autofocus"),
        _msg("assistant", "Sıralama düzeltildi (4 → 12), best z = 7.8 mm."),
    ], tools))

    # H9 — self-correction: enum case
    out.append(_example([
        sys_msg,
        _msg("user", "Depth map laplacian variance, 0-15."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "compute_depth_map",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "n_steps": 40, "metric": "laplacian_variance",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "error": "invalid arguments",
            "details": "metric must be one of LAPLACIAN_VARIANCE, ENTROPY, PHASE_VARIANCE, ...",
        }), tool_call_id="c1", name="compute_depth_map"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c2", "compute_depth_map",
                                    {"z_min_mm": 0.0, "z_max_mm": 15.0,
                                     "n_steps": 40, "metric": "LAPLACIAN_VARIANCE",
                                     "window_size": 5})]),
        _msg("tool", json.dumps({
            "submitted": True,
            "summary": {"mean_z": 6.5, "valid_pct": 87.0},
        }), tool_call_id="c2", name="compute_depth_map"),
        _msg("assistant", "Enum büyük harfe çevrildi — depth mean z=6.5, %87."),
    ], tools))

    # H10 — refusal: out-of-home path
    out.append(_example([
        sys_msg,
        _msg("user", "/var/log/system.log dosyasını yükle."),
        _msg("assistant",
             "Ev klasörü dışındaki yolları açamam (path-traversal koruması). "
             "Hologram dosyan ev altında bir yerde mi?"),
    ], tools))

    # H11 — domain TR: "dalga boyu"
    out.append(_example([
        sys_msg,
        _msg("user", "Dalga boyunu 660 yap."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "set_recon_param", {"wavelength_nm": 660.0})]),
        _msg("tool", json.dumps({"updated": {"wavelength_nm": 660.0},
                                 "current": {"wavelength_nm": 660.0}}),
             tool_call_id="c1", name="set_recon_param"),
        _msg("assistant", "λ = 660 nm."),
    ], tools))

    # H12 — domain TR: "AF"
    out.append(_example([
        sys_msg,
        _msg("user", "AF 2-12 mm."),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "run_autofocus",
                                    {"z_min_mm": 2.0, "z_max_mm": 12.0,
                                     "metric": "PHASE_VARIANCE", "n_steps": _Z_STEPS})]),
        _msg("tool", json.dumps({"submitted": True,
                                 "summary": {"best_z_mm": 7.4}}),
             tool_call_id="c1", name="run_autofocus"),
        _msg("assistant", "Best z = 7.4 mm."),
    ], tools))

    # H13 — conversational: short summary
    out.append(_example([
        sys_msg,
        _msg("user", "QPI nasıl?"),
        _msg("assistant", "",
             tool_calls=[_tool_call("c1", "get_last_result", {"stage": "qpi"})]),
        _msg("tool", json.dumps({
            "stage": "qpi", "available": True,
            "summary": {"total_dry_mass_pg": 18.0, "opd_range_nm": 92.0},
        }), tool_call_id="c1", name="get_last_result"),
        _msg("assistant", "Son QPI: dry mass 18 pg, OPD 92 nm."),
    ], tools))

    # H14 — conversational: empty state
    out.append(_example([
        sys_msg,
        _msg("user", "Hazır mıyım?"),
        _msg("assistant", "", tool_calls=[_tool_call("c1", "get_state", {})]),
        _msg("tool", json.dumps({
            "loaded_path": None,
            "recon_params": {}, "stage_position_mm": [0.0, 0.0, 0.0],
        }), tool_call_id="c1", name="get_state"),
        _msg("assistant", "Hayır — sample yüklü değil. Bir hologram yolu ver."),
    ], tools))

    # H15 — negative: scikit-image
    out.append(_example([
        sys_msg,
        _msg("user", "Scikit-image ile watershed segmentasyon yap."),
        _msg("assistant",
             "scikit-image entegrasyonu yok — watershed tool'um yok. "
             "Phase haritası üzerinden eşik tabanlı dolaylı segmentasyon QPI panelinde "
             "var; recon + QPI çalıştırırsan onu kullanabilirsin."),
    ], tools))

    return out


def build_examples(tools: list[dict]) -> list[dict]:
    """100 training examples spread across 9 categories."""
    out: list[dict] = []
    out.extend(_tool_selection_examples(tools))         # 15
    out.extend(_argument_formatting_examples(tools))    # 10
    out.extend(_chain_examples(tools))                  # 25
    out.extend(_self_correction_examples(tools))        #  8
    out.extend(_refusal_examples(tools))                #  5
    out.extend(_domain_language_examples(tools))        # 15
    out.extend(_conversational_style_examples(tools))   # 12
    out.extend(_negative_examples(tools))               #  5
    out.extend(_lab_specific_examples(tools))           #  5
    return out


def build_holdout(tools: list[dict]) -> list[dict]:
    """15 evaluation examples — never train on these."""
    return _holdout_examples(tools)


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
        help="Training set output JSONL path (parents created as needed).",
    )
    parser.add_argument(
        "--holdout-out", type=Path,
        default=ROOT / "data" / "ai" / "eval_holdout.jsonl",
        help="Holdout eval set output JSONL path.",
    )
    parser.add_argument(
        "--include-stage", action="store_true",
        help="Include stage tools in the tool schema "
             "(re-enable after motorised stage is connected).",
    )
    parser.add_argument(
        "--include-devices", action="store_true",
        help="Include device tools (shutter / LED / acquire_grid) in the "
             "tool schema (re-enable after hardware drivers are wired).",
    )
    args = parser.parse_args(argv)

    registry = build_tool_registry()
    tools_schema = _filter_hardware(
        registry.schemas(),
        include_stage=args.include_stage,
        include_devices=args.include_devices,
    )

    train = build_examples(tools_schema)
    holdout = build_holdout(tools_schema)

    train_out: Path = args.out.expanduser().resolve()
    holdout_out: Path = args.holdout_out.expanduser().resolve()
    train_out.parent.mkdir(parents=True, exist_ok=True)
    holdout_out.parent.mkdir(parents=True, exist_ok=True)

    with open(train_out, "w", encoding="utf-8") as fh:
        for ex in train:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(holdout_out, "w", encoding="utf-8") as fh:
        for ex in holdout:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(train)} training examples "
        f"({train_out.stat().st_size} bytes) → {train_out}"
    )
    print(
        f"wrote {len(holdout)} holdout examples "
        f"({holdout_out.stat().st_size} bytes) → {holdout_out}"
    )
    stage_state = "INCLUDED" if args.include_stage else "EXCLUDED"
    dev_state = "INCLUDED" if args.include_devices else "EXCLUDED"
    print(
        f"tool schema: {len(tools_schema)} tools "
        f"(stage {stage_state}, devices {dev_state})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
