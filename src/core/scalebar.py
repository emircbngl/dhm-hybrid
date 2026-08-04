"""Scalebar overlay helpers.

Both Qt (v1) and Dear PyGui (v2) panels paint a horizontal bar plus a
length label so lab reports can size objects from a screenshot. The
shared logic lives here:

* :func:`nice_length_um` — pick a "round" length close to a target
  fraction of the frame width, snapped to the 1/2/5 × 10ⁿ ladder so
  bars always read 1 µm, 2 µm, 5 µm, 10 µm, … instead of 7.4 µm.
* :func:`compute_scalebar` — given frame width in pixels and the
  effective pixel size in µm, returns ``(length_um, length_px, label)``
  so the panel just paints two endpoints + a string.

The functions are pure — no Qt or DPG imports — so they can be unit
tested without spinning up a window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


_TICKS = (1.0, 2.0, 5.0)


def nice_length_um(target_um: float) -> float:
    """Snap ``target_um`` to the nearest 1/2/5 × 10ⁿ tick.

    The 1-2-5 ladder is the standard "engineering scale" choice — it
    keeps bar labels predictable across magnifications. Picking 7 µm
    instead of 5 µm forces the lab to do mental arithmetic; picking
    5 µm does not.

    Decade-boundary cases (target = 78 should snap to 100, not 50)
    are handled by including the neighbouring decades in the
    candidate set and picking the closest in log space.
    """
    if target_um <= 0:
        return 1.0
    import math
    exp = math.floor(math.log10(target_um))
    log_target = math.log10(target_um)
    candidates = [
        t * (10 ** e)
        for e in (exp - 1, exp, exp + 1)
        for t in _TICKS
    ]
    return min(candidates, key=lambda c: abs(math.log10(c) - log_target))


@dataclass(frozen=True)
class ScaleBarSpec:
    """Output of :func:`compute_scalebar` — what the panel needs to paint."""
    length_um: float
    length_px: float
    label: str   # e.g. "25 µm"


def compute_scalebar(
    frame_width_px: int,
    pixel_size_um: float,
    *,
    target_fraction: float = 0.15,
    fixed_length_um: float | None = None,
) -> ScaleBarSpec | None:
    """Pick a sensible scalebar for a frame.

    Parameters
    ----------
    frame_width_px
        Width of the displayed frame in pixels (the panel's image, not
        the screen).
    pixel_size_um
        Effective pixel size at the sample plane (camera_pixel / M).
        Must be > 0; returns ``None`` otherwise so callers can hide the
        bar when the recon tab hasn't been wired yet.
    target_fraction
        Fraction of the frame width the bar should occupy. 0.10–0.20 is
        the readable range; smaller bars get lost, larger bars cut into
        the image content.
    fixed_length_um
        Override the auto-pick. When provided, the function still
        returns the px length and label; useful when the lab wants a
        fixed 25 µm bar across all reports for visual consistency.

    Returns
    -------
    ``ScaleBarSpec`` describing the bar to paint, or ``None`` when
    parameters are unusable (zero width, zero/negative pixel size).
    """
    if frame_width_px <= 0 or pixel_size_um <= 0:
        return None
    if fixed_length_um is not None and fixed_length_um > 0:
        length_um = float(fixed_length_um)
    else:
        target_um = frame_width_px * pixel_size_um * target_fraction
        length_um = nice_length_um(target_um)
    length_px = length_um / pixel_size_um
    if length_px > frame_width_px * 0.95:
        # Bar would overflow the frame — nudge down one ladder step.
        for tick in reversed(_TICKS):
            candidate = tick * (10 ** _floor_log10(length_um / 2.0))
            if candidate / pixel_size_um <= frame_width_px * 0.7:
                length_um = candidate
                length_px = length_um / pixel_size_um
                break
    label = _format_label_um(length_um)
    return ScaleBarSpec(length_um=length_um, length_px=length_px, label=label)


def _floor_log10(x: float) -> int:
    import math
    return int(math.floor(math.log10(x))) if x > 0 else 0


def _format_label_um(length_um: float) -> str:
    """Render a label with sensible precision: '25 µm', '0.5 µm', '1 mm'."""
    if length_um >= 1000.0:
        mm = length_um / 1000.0
        if mm == int(mm):
            return f"{int(mm)} mm"
        return f"{mm:g} mm"
    if length_um >= 1.0 and length_um == int(length_um):
        return f"{int(length_um)} µm"
    return f"{length_um:g} µm"


__all__ = ["ScaleBarSpec", "compute_scalebar", "nice_length_um"]
