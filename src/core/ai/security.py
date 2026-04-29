"""Guardrails for AI tool calls.

Even with a local LLM, accepting tool args verbatim is unsafe: a fumbled
``z_mm: 1e9`` would crash propagation, and ``load_hologram("/etc/passwd")``
would read the wrong file. This module is the single place that defines
*what's allowed* — every tool handler imports the helpers from here.

No Qt, no HTTP. Pure stdlib so this can be unit-tested without a Qt event
loop or running app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Default file extensions accepted by ``load_hologram``. Adding to this
# list is fine — the loader downstream already rejects formats it can't
# decode. Removing entries is a behaviour change.
ALLOWED_HOLOGRAM_EXTENSIONS: frozenset[str] = frozenset({
    ".tif", ".tiff", ".png", ".npy", ".h5", ".mat", ".bmp",
})

# 4 GB. Holograms are typically 4–64 MB; this is a sanity ceiling
# that catches an LLM hallucinating a path to a video file.
MAX_FILE_SIZE_BYTES: int = 4 * 1024 * 1024 * 1024


class SecurityError(ValueError):
    """Raised when an argument fails a guardrail check.

    We use a subclass of ``ValueError`` so handlers can either catch it
    explicitly or let it propagate to the registry's friendly-error
    branch. Either way the LLM sees a structured failure, not a stack
    trace.
    """


# ---------------------------------------------------------------------------
# Numeric clamping
# ---------------------------------------------------------------------------

# (lo, hi) tuples. Every numeric tool arg ends up here. Choose ranges
# that are wider than any sane lab use but tight enough that an obvious
# LLM hallucination (z_mm: 1e9) is rejected.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "z_mm": (-200.0, 200.0),
    "z_min_mm": (-200.0, 200.0),
    "z_max_mm": (-200.0, 200.0),
    "wavelength_nm": (100.0, 2000.0),
    "pixel_um": (0.01, 100.0),
    "n_medium": (1.0, 2.0),
    "n_sample": (1.0, 2.0),
    "cell_n": (1.0, 2.0),
    "medium_n": (1.0, 2.0),
    "phase_offset": (-100.0, 100.0),
    "mask_radius": (1.0, 1024.0),
    "n_steps": (1.0, 256.0),
    "top_k": (1.0, 256.0),
    "window_size": (3.0, 64.0),
    "limit": (1.0, 500.0),
    "dx_mm": (-200.0, 200.0),
    "dy_mm": (-200.0, 200.0),
    "dz_mm": (-200.0, 200.0),
    "x_mm": (-200.0, 200.0),
    "y_mm": (-200.0, 200.0),
    "max_iterations": (1.0, 32.0),
    # v2.1.z — stage_focus_search args + device tool args.
    "search_range_mm": (0.1, 50.0),
    "step_mm": (0.05, 5.0),
    "mask_dilate_px": (0.0, 256.0),
    "intensity_percent": (0.0, 100.0),
    # acquire_grid extents — wider than typical lab stage limits so
    # the SDK-side limit check still gates the actual move.
    "rows": (1.0, 64.0),
    "cols": (1.0, 64.0),
    "spacing_x_um": (0.1, 50_000.0),
    "spacing_y_um": (0.1, 50_000.0),
    "settle_time_s": (0.0, 10.0),
    # v2.1.z+ — APT-style stage controls
    "speed_um_per_s": (0.1, 100_000.0),
    "step_um_jog": (0.01, 50_000.0),
    "dx_um": (-50_000.0, 50_000.0),
    "dy_um": (-50_000.0, 50_000.0),
    "dz_um": (-50_000.0, 50_000.0),
}


def clamp(name: str, value: float) -> float:
    """Clamp ``value`` against :data:`NUMERIC_BOUNDS` for ``name``.

    Raises :class:`SecurityError` if the name has bounds and the value
    would land outside them. Names without bounds are returned as-is —
    meaning an unknown numeric is the caller's responsibility (the JSON
    schema should already have validated it).

    Rationale for raising rather than silently clamping: an LLM that
    sends ``z_mm = 999`` is confused; silently making it 200 would let
    a wrong reasoning step leak into the result. A friendly error in
    the tool result lets the model re-plan.
    """
    if name not in NUMERIC_BOUNDS:
        return float(value)
    lo, hi = NUMERIC_BOUNDS[name]
    v = float(value)
    if v != v:  # NaN
        raise SecurityError(f"{name} is NaN")
    if v < lo or v > hi:
        raise SecurityError(
            f"{name}={v} is out of range [{lo}, {hi}]"
        )
    return v


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def validate_path(
    raw: str,
    *,
    must_exist: bool = True,
    allowed_extensions: Iterable[str] = ALLOWED_HOLOGRAM_EXTENSIONS,
    restrict_to_home: bool = True,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Path:
    """Resolve and validate a filesystem path provided by the assistant.

    The resolution + checks together rule out: (a) traversal via ``..``;
    (b) symlinks pointing into system dirs; (c) wrong file types; (d)
    accidentally huge files; (e) paths outside the user's home when
    the operator has chosen the conservative default.

    Any failure raises :class:`SecurityError` with a message that
    mentions both the rejected path and the reason — handlers convert
    that into a tool result the LLM can read.
    """
    if not isinstance(raw, str) or not raw:
        raise SecurityError("path must be a non-empty string")

    # ``expanduser`` lets the LLM say ``~/holograms/sample.tif`` and have
    # it Just Work; ``resolve`` collapses ``..`` and follows symlinks so
    # the final path is what we actually check.
    p = Path(raw).expanduser().resolve()

    # Restrict-to-home is the default. Lab installs with NAS mounts
    # outside ~ can flip it off in settings.
    if restrict_to_home:
        home = Path.home().resolve()
        try:
            p.relative_to(home)
        except ValueError as exc:
            raise SecurityError(
                f"path is outside the home directory ({home}): {p}"
            ) from exc

    if must_exist and not p.exists():
        raise SecurityError(f"path does not exist: {p}")

    if must_exist and not p.is_file():
        raise SecurityError(f"path is not a regular file: {p}")

    suffix = p.suffix.lower()
    allowed = {ext.lower() for ext in allowed_extensions}
    if suffix not in allowed:
        raise SecurityError(
            f"extension {suffix!r} not allowed (need one of {sorted(allowed)})"
        )

    if must_exist:
        try:
            size = p.stat().st_size
        except OSError as exc:
            raise SecurityError(f"could not stat {p}: {exc}") from exc
        if size > max_size_bytes:
            raise SecurityError(
                f"file is {size} bytes, larger than the {max_size_bytes}-byte limit"
            )

    return p


# ---------------------------------------------------------------------------
# Audit redaction (LLM-bound)
# ---------------------------------------------------------------------------

def redact_for_llm(record: dict) -> dict:
    """Strip lab-private fields out of an audit record before sending to the LLM.

    The audit log carries the OS username and absolute paths. Neither
    helps the assistant reason; both leak out of the lab. Replace
    operator with a placeholder and trim absolute paths to basenames
    inside ``params`` and ``result_summary``.
    """
    redacted: dict = dict(record)

    if "user" in redacted:
        redacted["user"] = "<user>"
    if "operator" in redacted:
        redacted["operator"] = "<user>"

    for key in ("params", "result_summary"):
        sub = redacted.get(key)
        if isinstance(sub, dict):
            redacted[key] = {k: _maybe_basename(v) for k, v in sub.items()}

    return redacted


def _maybe_basename(value):
    """Trim absolute paths to their basename, recurse into dicts/lists.

    Heuristic: a string starting with ``/`` or ``~`` and at least one
    ``/`` is treated as a path. Anything else passes through.
    """
    if isinstance(value, str):
        if value.startswith(("/", "~")) and "/" in value[1:]:
            return Path(value).name
        return value
    if isinstance(value, dict):
        return {k: _maybe_basename(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_maybe_basename(v) for v in value]
    return value


__all__ = [
    "SecurityError",
    "ALLOWED_HOLOGRAM_EXTENSIONS",
    "MAX_FILE_SIZE_BYTES",
    "NUMERIC_BOUNDS",
    "clamp",
    "validate_path",
    "redact_for_llm",
]
