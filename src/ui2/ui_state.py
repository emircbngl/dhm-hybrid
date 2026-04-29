"""Lightweight UI-state helpers — v2.1.y sprint, P2 + P3.

Two small state machines that live outside :mod:`ui2.app` so they
can be tested without a DPG context:

* :class:`DropZoneState` — drives the drop-zone label / colour as
  the operator goes "ready → loading → reconstructing → ready".
  Lab review (v2.0.3 backlog): "I clicked Reconstruct, was the
  click received? The status text scrolled away too fast."
* :func:`workflow_export_buttons_visible` / :func:`is_export_button_id`
  — pure helpers for the v2.0.3 backlog item "report mode export
  buttons highlight". The ``Report`` workflow mode now both
  *shows* + *highlights* the export buttons; other modes hide
  them.

Both are framework-agnostic. ``ui2.app`` consumes them and feeds
the output into DPG calls (configure_item show=, theme bind).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Set


# ---------------------------------------------------------------------------
# DropZoneState
# ---------------------------------------------------------------------------

class DropZoneStage(str, Enum):
    """Discrete phases the drop zone can be in."""
    READY = "ready"             # waiting for input
    LOADING = "loading"         # TIFF read + off-axis extraction in flight
    RECON = "reconstructing"    # core pipeline running
    DONE = "done"               # last action finished, briefly highlighted


# Operator-visible label per stage. Includes leading glyph so the
# box reads as live without a colour bar.
DROP_ZONE_LABEL = {
    DropZoneStage.READY: "Drop hologram here  /  Click to load",
    DropZoneStage.LOADING: "● Loading hologram…",
    DropZoneStage.RECON: "● Reconstructing…",
    DropZoneStage.DONE: "✓ Done",
}

# Colour role per stage — caller maps to ScientificTheme constants.
DROP_ZONE_COLOUR_ROLE = {
    DropZoneStage.READY: "TEXT_MUTED",
    DropZoneStage.LOADING: "ACCENT",
    DropZoneStage.RECON: "ACCENT",
    DropZoneStage.DONE: "SUCCESS",
}


@dataclass
class DropZoneState:
    """Current stage + last-known FPS / progress hint.

    ``stage`` is the discrete phase; ``hint`` is a free-form
    string the dialog appends to the label (e.g. "12 of 40
    steps", "frame 1 of 3"). ``hint`` empty = no hint shown.
    """
    stage: DropZoneStage = DropZoneStage.READY
    hint: str = ""

    def label(self) -> str:
        """Combine stage label + hint into a single string."""
        base = DROP_ZONE_LABEL[self.stage]
        if not self.hint:
            return base
        return f"{base}  ({self.hint})"

    def colour_role(self) -> str:
        return DROP_ZONE_COLOUR_ROLE[self.stage]

    def transition(self, new_stage: DropZoneStage,
                   *, hint: str = "") -> None:
        """Transition to ``new_stage`` and replace the hint.
        Empty hint = clear the previous one."""
        self.stage = new_stage
        self.hint = hint


# ---------------------------------------------------------------------------
# Workflow → export-button visibility / accent
# ---------------------------------------------------------------------------

# DPG button tag strings. Centralised so both the renderer and
# the test suite agree on the names.
EXPORT_BUTTON_TAGS: Set[str] = {
    "btn_export_report",
    "btn_export_csv",
    "btn_export_bundle",
    "btn_export_pdf",
}


def workflow_export_buttons_visible(workflow_mode: str) -> bool:
    """Are the export buttons shown for this workflow mode?

    Rule: only ``Report`` mode renders the export panel. ``Reconstruct``
    + ``Analyse`` keep export buttons hidden so the sidebar stays
    focused on their primary task.
    """
    return str(workflow_mode).lower() == "report"


def workflow_export_buttons_accented(workflow_mode: str) -> bool:
    """Should the export buttons render with an accent border?

    True only when:
    1. Visible (``Report`` mode).
    2. Operator hasn't dismissed the highlight (handled by the
       caller; we just signal "this is your primary action").

    The renderer maps this to a coloured border + slightly bolder
    label.
    """
    return workflow_export_buttons_visible(workflow_mode)


def is_export_button_id(tag: str) -> bool:
    """``True`` if ``tag`` is one of the export buttons we
    track. Centralised so we don't fan out tag-name string
    comparisons across the codebase."""
    return str(tag) in EXPORT_BUTTON_TAGS


__all__ = [
    "DropZoneStage",
    "DROP_ZONE_LABEL",
    "DROP_ZONE_COLOUR_ROLE",
    "DropZoneState",
    "EXPORT_BUTTON_TAGS",
    "workflow_export_buttons_visible",
    "workflow_export_buttons_accented",
    "is_export_button_id",
]
