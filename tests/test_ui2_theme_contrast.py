"""WCAG 2.1 Level AA contrast tests for every palette role.

The Dear PyGui themes ship four palettes (dark / light / midnight /
high_contrast). A11y compliance requires the contrast ratio between
foreground text and the background it lands on to be ≥ 4.5:1 for
normal-size text. We enumerate the role pairs that actually render
together in the UI and assert the ratio for each palette.

Non-text chrome (borders, the outline colour of a button) is exempt
from 4.5:1 per WCAG; for the roles we check here the user is reading
actual text, so AA applies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ui2.theme import PALETTES  # noqa: E402


# ---------------------------------------------------------------------------
# WCAG relative luminance + contrast ratio (sRGB linearisation)
# ---------------------------------------------------------------------------

def _relative_luminance(rgba):
    """Per https://www.w3.org/TR/WCAG21/#dfn-relative-luminance."""
    def _chan(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.2126 * _chan(r) + 0.7152 * _chan(g) + 0.0722 * _chan(b)


def _contrast_ratio(fg, bg) -> float:
    l_fg = _relative_luminance(fg)
    l_bg = _relative_luminance(bg)
    lighter, darker = max(l_fg, l_bg), min(l_fg, l_bg)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# Role-pair matrix — which foreground lands on which background at runtime
# ---------------------------------------------------------------------------

# (foreground_role, background_role) pairs. Each pair is rendered in the
# app somewhere (status text, toast body, sidebar label, etc.). Kept
# deliberately short so a single failure points at a concrete visual.
CRITICAL_PAIRS = [
    ("text", "window_bg"),
    ("text", "panel_bg"),
    ("text_muted", "panel_bg"),
    ("success", "panel_bg"),
    ("warn", "panel_bg"),
    ("danger", "panel_bg"),
    ("accent", "panel_bg"),
]


@pytest.mark.parametrize("palette_name", list(PALETTES.keys()))
@pytest.mark.parametrize("fg,bg", CRITICAL_PAIRS)
def test_contrast_meets_aa(palette_name, fg, bg):
    palette = PALETTES[palette_name]
    ratio = _contrast_ratio(getattr(palette, fg), getattr(palette, bg))
    # Level AA for normal-size text is 4.5:1.
    assert ratio >= 4.5, (
        f"{palette_name}.{fg} on {palette_name}.{bg} = "
        f"{ratio:.2f}:1 (needs ≥ 4.5:1 for WCAG AA)")


def test_high_contrast_exceeds_aaa():
    """The high_contrast palette is the fallback for visually-impaired
    users — it must clear AAA (7:1) on every critical pair, not just AA.
    This is the one palette where we can afford to be strict."""
    palette = PALETTES["high_contrast"]
    for fg, bg in CRITICAL_PAIRS:
        ratio = _contrast_ratio(getattr(palette, fg),
                                getattr(palette, bg))
        assert ratio >= 7.0, (
            f"high_contrast.{fg} on {bg} = {ratio:.2f} (needs ≥ 7.0)")


# ---------------------------------------------------------------------------
# Sanity check on the formula itself
# ---------------------------------------------------------------------------

def test_luminance_white_is_one():
    assert _relative_luminance((255, 255, 255)) == pytest.approx(1.0,
                                                                 abs=1e-6)


def test_luminance_black_is_zero():
    assert _relative_luminance((0, 0, 0)) == pytest.approx(0.0, abs=1e-6)


def test_contrast_white_black_is_21():
    # 1.0 + 0.05 / 0.0 + 0.05 = 21.0 exactly — WCAG's fixed upper bound.
    assert _contrast_ratio((255, 255, 255), (0, 0, 0)) == pytest.approx(
        21.0, abs=1e-6)
