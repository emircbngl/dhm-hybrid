"""State snapshot + system-prompt builder.

The LLM's context window has to hold the system prompt plus the
conversation. We keep both small: the snapshot is scalar-only (no numpy
arrays) and the prompt is short and prescriptive.

The snapshot is built on the GUI thread before each turn so the AI
worker reads frozen data without crossing threads. ``build_system_prompt``
is pure-function — given a snapshot and a tool count, it returns a
multi-line string. Both halves are import-cheap (no Qt).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class StateSnapshot:
    """What the AI sees about the app's current state.

    Kept small on purpose: numpy arrays don't serialize well into a
    chat prompt, and the LLM doesn't need them — it needs *summaries*
    (shape, mean, std, min, max) of the last few results. The handlers
    that compute new things take what they need from the live app via
    the panel's helpers, not from this snapshot.
    """
    loaded_path: Optional[str] = None
    loaded_shape: Optional[tuple[int, int]] = None
    loaded_dtype: Optional[str] = None

    recon_params: dict = field(default_factory=dict)
    autofocus_params: dict = field(default_factory=dict)
    qpi_params: dict = field(default_factory=dict)

    last_recon_summary: Optional[dict] = None
    last_af_summary: Optional[dict] = None
    last_qpi_summary: Optional[dict] = None
    last_depth_summary: Optional[dict] = None

    stage_position_mm: Optional[tuple[float, float, float]] = None
    audit_tail: tuple[dict, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are the AI co-pilot for a Digital Holographic Microscopy (DHM) desktop app.

Your job is to help the operator analyse holograms by chaining tool calls.
You have {tool_count} tools available — they are the only way you can affect
the app. Pure-text responses are fine for explanations and summaries; for
*actions*, always go through tools.

Pipeline you are operating:
  load_hologram → set_recon_param → run_reconstruction →
    run_autofocus / find_focus_candidates → run_qpi / compute_depth_map.

Stage tools (stage_get_position, stage_move_*) drive a virtual XYZ stage —
real hardware is not yet connected, but moves are tracked and reported.

Behaviour rules:
  * Prefer tool calls over freehand instructions. The operator wants action.
  * Read-only tools (get_state, get_last_result, get_audit_tail) are cheap;
    use them to ground your reasoning whenever you are unsure.
  * Do not invent file paths. If the operator hasn't named a hologram, ask.
  * Numbers must include units — wavelength in nm, pixel in µm, z in mm.
  * If a tool returns {{"error": ...}}, read the message and try a
    different argument. Do not retry the same call unchanged.
  * Be concise. The operator is at a microscope, not at a chat window.

Current state snapshot:
{state_block}

When the conversation starts, propose a concrete next step based on the
state above — e.g. "I see no hologram loaded; want me to open one?", or
"a reconstruction is ready; should I run autofocus next?".
"""


def build_system_prompt(snapshot: StateSnapshot, tool_count: int) -> str:
    """Compose the system message for one turn.

    Re-built every turn so the assistant always sees the latest state.
    Keeping the format stable helps the model reason consistently across
    turns.
    """
    lines: list[str] = []
    if snapshot.loaded_path:
        shape = snapshot.loaded_shape
        dtype = snapshot.loaded_dtype
        lines.append(
            f"  - Loaded hologram: {snapshot.loaded_path} "
            f"({shape[0]}×{shape[1]} {dtype})" if shape else
            f"  - Loaded hologram: {snapshot.loaded_path}"
        )
    else:
        lines.append("  - Loaded hologram: (none)")

    if snapshot.recon_params:
        lines.append(
            "  - Recon params: " + _kv(snapshot.recon_params, keys=(
                "wavelength_nm", "pixel_um", "z_mm", "n_medium",
                "mask_radius", "subtract_mean", "hann_window",
            ))
        )
    if snapshot.autofocus_params:
        lines.append(
            "  - Autofocus params: " + _kv(snapshot.autofocus_params, keys=(
                "z_min_mm", "z_max_mm", "metric", "n_steps",
            ))
        )
    if snapshot.qpi_params:
        lines.append(
            "  - QPI params: " + _kv(snapshot.qpi_params, keys=(
                "n_sample", "n_medium", "phase_offset",
            ))
        )

    if snapshot.last_recon_summary:
        lines.append(
            "  - Last reconstruction: " + _kv(snapshot.last_recon_summary)
        )
    if snapshot.last_af_summary:
        lines.append("  - Last autofocus: " + _kv(snapshot.last_af_summary))
    if snapshot.last_qpi_summary:
        lines.append("  - Last QPI: " + _kv(snapshot.last_qpi_summary))
    if snapshot.last_depth_summary:
        lines.append("  - Last depth map: " + _kv(snapshot.last_depth_summary))

    if snapshot.stage_position_mm is not None:
        x, y, z = snapshot.stage_position_mm
        lines.append(f"  - Stage position: x={x:.3f} mm, y={y:.3f} mm, z={z:.3f} mm")

    if snapshot.audit_tail:
        lines.append("  - Recent actions:")
        for entry in snapshot.audit_tail[-5:]:
            action = entry.get("action", "?")
            ts = entry.get("timestamp", "")
            lines.append(f"      • {ts}  {action}")

    state_block = "\n".join(lines) if lines else "  (state is empty)"
    return _SYSTEM_TEMPLATE.format(
        tool_count=int(tool_count),
        state_block=state_block,
    )


def _kv(d: dict, *, keys: Optional[tuple[str, ...]] = None) -> str:
    items = []
    use = keys if keys is not None else tuple(d.keys())
    for k in use:
        if k in d:
            v = d[k]
            if isinstance(v, float):
                items.append(f"{k}={v:.4g}")
            else:
                items.append(f"{k}={v!s}")
    return ", ".join(items) if items else "(empty)"


__all__ = [
    "StateSnapshot",
    "build_system_prompt",
]
