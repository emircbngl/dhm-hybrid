"""v2.1.y UI polish mini-sprint tests.

Three pure-state modules so DPG-free unit tests cover the logic.

* P1 — ``ui2.line_profile_state.LineProfileEditor``: click-drag
  state machine.
* P2 — ``ui2.ui_state.DropZoneState``: ready → loading → recon →
  done transitions.
* P3 — ``ui2.ui_state`` workflow helpers: export buttons visible
  + accented only in Report mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ui2.line_profile_state import EditorState, LineProfileEditor  # noqa: E402
from ui2.ui_state import (  # noqa: E402
    DROP_ZONE_COLOUR_ROLE,
    DROP_ZONE_LABEL,
    DropZoneStage,
    DropZoneState,
    EXPORT_BUTTON_TAGS,
    is_export_button_id,
    workflow_export_buttons_accented,
    workflow_export_buttons_visible,
)


# ---------------------------------------------------------------------------
# P1 — LineProfileEditor
# ---------------------------------------------------------------------------

def test_editor_starts_idle():
    e = LineProfileEditor()
    assert e.state is EditorState.IDLE
    assert e.profiles == []
    assert e.draft_endpoints is None


def test_first_click_transitions_to_awaiting_end():
    e = LineProfileEditor()
    e.first_click(10, 20)
    assert e.state is EditorState.AWAITING_END
    assert e.has_pending_first_click
    # Draft endpoints are exposed only in PREVIEW.
    assert e.draft_endpoints is None


def test_second_click_creates_profile():
    e = LineProfileEditor()
    e.first_click(10, 20)
    p = e.second_click(30, 40, label="membrane")
    assert p is not None
    assert p.label == "membrane"
    assert (p.y0, p.x0) == (10, 20)
    assert (p.y1, p.x1) == (30, 40)
    # After commit we're back to IDLE so a fresh first_click
    # starts another line.
    assert e.state is EditorState.IDLE
    assert len(e.profiles) == 1


def test_default_label_is_indexed():
    e = LineProfileEditor()
    e.first_click(0, 0)
    p1 = e.second_click(10, 10)
    e.first_click(0, 0)
    p2 = e.second_click(20, 20)
    assert p1.label == "line-1"
    assert p2.label == "line-2"


def test_second_click_without_first_returns_none():
    e = LineProfileEditor()
    p = e.second_click(10, 10)
    assert p is None
    assert e.state is EditorState.IDLE


def test_second_click_zero_length_rejected():
    """End == start → no profile (bilinear sampler returns NaN
    everywhere; the dialog plot would be empty + confusing)."""
    e = LineProfileEditor()
    e.first_click(15, 25)
    p = e.second_click(15, 25)
    assert p is None
    # State stays AWAITING_END so operator can click a different
    # end point.
    assert e.state is EditorState.AWAITING_END


def test_first_click_during_awaiting_end_resets_draft():
    """If the operator clicks somewhere new while still awaiting
    the end point, the new click becomes the new start. Forgive
    the wandering hand."""
    e = LineProfileEditor()
    e.first_click(5, 5)
    e.first_click(50, 50)
    assert e.state is EditorState.AWAITING_END
    p = e.second_click(60, 60)
    assert p is not None
    assert (p.y0, p.x0) == (50, 50)


def test_cancel_drops_draft_keeps_saved():
    e = LineProfileEditor()
    e.first_click(0, 0)
    e.second_click(10, 10)
    e.first_click(20, 20)
    e.cancel()
    assert e.state is EditorState.IDLE
    assert len(e.profiles) == 1


def test_drop_last_pops_most_recent():
    e = LineProfileEditor()
    e.first_click(0, 0); e.second_click(10, 10, label="a")
    e.first_click(0, 0); e.second_click(20, 20, label="b")
    removed = e.drop_last()
    assert removed.label == "b"
    assert len(e.profiles) == 1
    assert e.profiles[0].label == "a"


def test_drop_last_on_empty_returns_none():
    e = LineProfileEditor()
    assert e.drop_last() is None


def test_clear_all_empties_list_and_state():
    e = LineProfileEditor()
    for i in range(3):
        e.first_click(i, i)
        e.second_click(i + 5, i + 5)
    e.first_click(50, 50)  # mid-draft
    e.clear_all()
    assert e.profiles == []
    assert e.state is EditorState.IDLE


def test_rename_updates_label_in_place():
    e = LineProfileEditor()
    e.first_click(0, 0); e.second_click(10, 10, label="old")
    assert e.rename(0, "new") is True
    assert e.profiles[0].label == "new"


def test_rename_out_of_bounds_returns_false():
    e = LineProfileEditor()
    assert e.rename(0, "x") is False


def test_set_colour_updates_in_place():
    e = LineProfileEditor()
    e.first_click(0, 0); e.second_click(5, 5)
    assert e.set_colour(0, (1.0, 0.0, 0.0)) is True
    assert e.profiles[0].colour_rgb == (1.0, 0.0, 0.0)


def test_palette_cycles():
    """Profiles 1..N pick palette[(i-1) % len]; profile 7 wraps
    to slot 0."""
    e = LineProfileEditor()
    for _ in range(7):
        e.first_click(0, 0)
        e.second_click(5, 5)
    assert e.profiles[0].colour_rgb == e.profiles[6].colour_rgb


def test_custom_palette_honoured():
    e = LineProfileEditor(palette=[(1.0, 0.0, 0.0)])
    e.first_click(0, 0); e.second_click(5, 5)
    assert e.profiles[0].colour_rgb == (1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# P2 — DropZoneState
# ---------------------------------------------------------------------------

def test_drop_zone_starts_ready():
    s = DropZoneState()
    assert s.stage is DropZoneStage.READY
    assert s.label() == DROP_ZONE_LABEL[DropZoneStage.READY]
    assert s.colour_role() == "TEXT_MUTED"


def test_drop_zone_transition_with_hint():
    s = DropZoneState()
    s.transition(DropZoneStage.RECON, hint="step 12 of 40")
    assert s.stage is DropZoneStage.RECON
    assert "step 12 of 40" in s.label()
    assert s.colour_role() == "ACCENT"


def test_drop_zone_transition_clears_hint_when_empty():
    s = DropZoneState(hint="leftover")
    s.transition(DropZoneStage.READY)
    assert s.hint == ""


def test_drop_zone_done_uses_success_colour():
    s = DropZoneState()
    s.transition(DropZoneStage.DONE)
    assert s.colour_role() == "SUCCESS"


@pytest.mark.parametrize("stage", list(DropZoneStage))
def test_drop_zone_label_table_complete(stage):
    """Every enum value must have a label entry — catches typo on
    enum extension."""
    assert stage in DROP_ZONE_LABEL
    assert DROP_ZONE_LABEL[stage]


@pytest.mark.parametrize("stage", list(DropZoneStage))
def test_drop_zone_colour_table_complete(stage):
    assert stage in DROP_ZONE_COLOUR_ROLE


# ---------------------------------------------------------------------------
# P3 — Workflow export buttons
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,visible", [
    ("Reconstruct", False),
    ("Analyse", False),
    ("Report", True),
    ("report", True),     # case-insensitive comparison
    ("REPORT", True),
])
def test_export_buttons_visible_only_in_report(mode, visible):
    assert workflow_export_buttons_visible(mode) is visible


def test_export_buttons_accented_when_visible():
    assert workflow_export_buttons_accented("Report") is True


def test_export_buttons_not_accented_when_hidden():
    """No point accenting a hidden button."""
    assert workflow_export_buttons_accented("Reconstruct") is False


def test_is_export_button_id_recognises_known_tags():
    for tag in EXPORT_BUTTON_TAGS:
        assert is_export_button_id(tag) is True


def test_is_export_button_id_rejects_others():
    assert is_export_button_id("btn_reconstruct") is False
    assert is_export_button_id("") is False
    assert is_export_button_id("anything") is False
