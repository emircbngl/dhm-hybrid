"""Line-profile click-drag state machine — v2.1.y sprint, P1.

The data layer (``core.line_profile``) was shipped in v2.0.8 D4 —
``LineProfile`` dataclass + bilinear sampler. This module is the
**interaction state machine** the DPG dialog drives:

```
              first_click()         second_click()
   IDLE ───────────────────► AWAITING_END ──────► PREVIEW ──┐
     ▲                                                       │
     │              cancel() / commit() / drop_last()        │
     └───────────────────────────────────────────────────────┘
```

Each successful (start, end) pair becomes a :class:`LineProfile`
appended to the active list. The dialog renders live preview by
asking the state for the current draft endpoints.

Why split state from DPG
------------------------
Headless tests can drive the state machine directly + assert
transitions; the DPG side stays a thin "render this state" view.
Big behavioural classes that only exist inside a render loop
(like the existing v1 line-profile widget) are hard to unit-test
and easy to break. Splitting reverses that.

Public API
----------
* :class:`LineProfileEditor` — owns the (start, end, profiles)
  state.
* Methods: ``first_click(y, x)``, ``second_click(y, x, label)``,
  ``cancel()``, ``drop_last()``, ``clear_all()``,
  ``rename(idx, new_label)``, ``set_colour(idx, rgb)``.
* Properties: ``state``, ``draft_endpoints``, ``profiles``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from core.line_profile import LineProfile


class EditorState(str, Enum):
    """Discrete UI states the dialog can be in."""
    IDLE = "idle"                 # waiting for first click
    AWAITING_END = "awaiting_end"  # have first point, awaiting end
    PREVIEW = "preview"           # full profile drawn, can commit


# Default colour palette — six visually-distinct accents the
# dialog cycles through as the operator adds profiles. After six,
# wraps around.
_DEFAULT_PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (0.27, 0.55, 0.95),  # blue
    (0.95, 0.45, 0.30),  # orange
    (0.40, 0.80, 0.40),  # green
    (0.80, 0.45, 0.85),  # magenta
    (0.95, 0.85, 0.30),  # yellow
    (0.40, 0.85, 0.85),  # teal
)


@dataclass
class LineProfileEditor:
    """Stateful editor for a list of line profiles.

    Attributes
    ----------
    profiles
        Current saved list. Order = stacking order in the overlay.
    state
        :class:`EditorState`. Caller transitions explicitly via
        ``first_click`` / ``second_click`` / ``cancel``.
    palette
        Colour cycle for newly-added profiles. Pass a custom one
        (e.g. WCAG-AA-safe) for accessibility-tuned themes.
    """
    profiles: List[LineProfile] = field(default_factory=list)
    state: EditorState = EditorState.IDLE
    palette: Sequence[Tuple[float, float, float]] = _DEFAULT_PALETTE

    _draft_start: Optional[Tuple[float, float]] = field(
        default=None, init=False,
    )
    _draft_end: Optional[Tuple[float, float]] = field(
        default=None, init=False,
    )

    # ── Public properties ─────────────────────────────────────
    @property
    def draft_endpoints(self) -> Optional[Tuple[Tuple[float, float],
                                                 Tuple[float, float]]]:
        """``((y0, x0), (y1, x1))`` while in PREVIEW; ``None``
        otherwise. Dialog uses this to draw the rubber-band line."""
        if self.state is EditorState.PREVIEW \
                and self._draft_start is not None \
                and self._draft_end is not None:
            return (self._draft_start, self._draft_end)
        return None

    @property
    def has_pending_first_click(self) -> bool:
        return self.state is EditorState.AWAITING_END

    # ── Click flow ────────────────────────────────────────────
    def first_click(self, y: float, x: float) -> None:
        """Operator clicked the start of a line. Transitions
        IDLE / PREVIEW → AWAITING_END. From PREVIEW the previous
        draft is dropped (operator started over)."""
        self._draft_start = (float(y), float(x))
        self._draft_end = None
        self.state = EditorState.AWAITING_END

    def second_click(self, y: float, x: float,
                     *, label: str = "") -> Optional[LineProfile]:
        """Operator clicked the end. Validates non-zero length,
        transitions AWAITING_END → PREVIEW, returns the freshly
        committed :class:`LineProfile` or ``None`` if rejected."""
        if self.state is not EditorState.AWAITING_END:
            return None
        if self._draft_start is None:
            return None
        end = (float(y), float(x))
        # Reject zero-length lines — bilinear sampler returns NaN
        # everywhere and the dialog plot is empty.
        if end == self._draft_start:
            return None
        self._draft_end = end
        self.state = EditorState.PREVIEW
        return self._commit_current(label=label)

    # ── Cancel / drop / clear ────────────────────────────────
    def cancel(self) -> None:
        """Drop the current draft + revert to IDLE. Does not touch
        the saved profile list."""
        self._draft_start = None
        self._draft_end = None
        self.state = EditorState.IDLE

    def drop_last(self) -> Optional[LineProfile]:
        """Pop the most recently committed profile. Returns the
        removed entry (caller can offer Undo)."""
        if not self.profiles:
            return None
        removed = self.profiles.pop()
        return removed

    def clear_all(self) -> None:
        """Remove every saved profile + reset state."""
        self.profiles.clear()
        self.cancel()

    def rename(self, idx: int, new_label: str) -> bool:
        """Update an existing profile's label. Returns True on
        success."""
        if not 0 <= idx < len(self.profiles):
            return False
        self.profiles[idx] = LineProfile(
            y0=self.profiles[idx].y0,
            x0=self.profiles[idx].x0,
            y1=self.profiles[idx].y1,
            x1=self.profiles[idx].x1,
            label=str(new_label),
            colour_rgb=self.profiles[idx].colour_rgb,
            n_samples=self.profiles[idx].n_samples,
        )
        return True

    def set_colour(self, idx: int,
                   rgb: Tuple[float, float, float]) -> bool:
        """Recolour an existing profile."""
        if not 0 <= idx < len(self.profiles):
            return False
        self.profiles[idx] = LineProfile(
            y0=self.profiles[idx].y0,
            x0=self.profiles[idx].x0,
            y1=self.profiles[idx].y1,
            x1=self.profiles[idx].x1,
            label=self.profiles[idx].label,
            colour_rgb=tuple(float(c) for c in rgb),
            n_samples=self.profiles[idx].n_samples,
        )
        return True

    # ── Internal ───────────────────────────────────────────────
    def _next_colour(self) -> Tuple[float, float, float]:
        """Cycle through the palette as profiles are added.
        Stable: profile N always lands on the same palette slot."""
        return self.palette[len(self.profiles) % len(self.palette)]

    def _commit_current(self, *, label: str) -> LineProfile:
        assert self._draft_start is not None
        assert self._draft_end is not None
        profile = LineProfile(
            y0=self._draft_start[0], x0=self._draft_start[1],
            y1=self._draft_end[0], x1=self._draft_end[1],
            label=label or f"line-{len(self.profiles) + 1}",
            colour_rgb=self._next_colour(),
        )
        self.profiles.append(profile)
        # After commit, transition back to IDLE so a fresh click
        # starts a new line. Dialog can call first_click again
        # immediately on the same coordinate without dropping.
        self._draft_start = None
        self._draft_end = None
        self.state = EditorState.IDLE
        return profile


__all__ = [
    "EditorState",
    "LineProfileEditor",
]
