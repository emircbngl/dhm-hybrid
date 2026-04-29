"""WCAG 2.1 contrast computation + theme audit — v2.0.9 sprint, P4.

Sven's compliance backlog item from v2.0.6:

    > "High-contrast theme palette var, compliance test yok."

This module provides:

* :func:`relative_luminance` — sRGB → linear → Y per WCAG.
* :func:`contrast_ratio`   — 4.5:1 / 3:1 thresholds.
* :func:`audit_palette` — programmatic check that every text /
  state colour clears WCAG AA on its intended background.

The auditor is the test surface — pinning the palette so a
future PR that drops contrast below threshold flips a CI failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


_RGB = Tuple[float, float, float]


def _to_unit(rgb255: Sequence[float]) -> _RGB:
    return (rgb255[0] / 255.0, rgb255[1] / 255.0, rgb255[2] / 255.0)


def _linearise(c: float) -> float:
    """sRGB gamma-decode for one channel, per WCAG 2.x.

    The 0.03928 break-point + 12.92 / 2.4 curve matches the WCAG
    Appendix A definition. Same numbers Mozilla's a11y inspector
    + axe-core use, so values agree with the popular tooling.
    """
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb255: Sequence[float]) -> float:
    """WCAG relative luminance Y in [0, 1] for an sRGB colour.

    ``rgb255`` may be a 3- or 4-tuple (alpha ignored). Float input
    is accepted; values are clamped to [0, 255]."""
    r, g, b = _to_unit([
        max(0.0, min(255.0, c)) for c in list(rgb255)[:3]
    ])
    return (
        0.2126 * _linearise(r)
        + 0.7152 * _linearise(g)
        + 0.0722 * _linearise(b)
    )


def contrast_ratio(fg: Sequence[float], bg: Sequence[float]) -> float:
    """WCAG contrast ratio in [1.0, 21.0]. 1.0 = identical, 21.0
    = pure black on pure white. Order of fg/bg doesn't matter
    (max/min normalisation)."""
    L1 = relative_luminance(fg)
    L2 = relative_luminance(bg)
    hi, lo = (L1, L2) if L1 >= L2 else (L2, L1)
    return (hi + 0.05) / (lo + 0.05)


# WCAG 2.1 AA thresholds.
AA_NORMAL = 4.5  # body text (< 18 pt regular / < 14 pt bold)
AA_LARGE = 3.0   # ≥ 18 pt regular / ≥ 14 pt bold
AAA_NORMAL = 7.0
AAA_LARGE = 4.5


@dataclass
class ContrastFinding:
    """One audit row — readable + filterable."""
    role_fg: str
    role_bg: str
    fg_rgb: Tuple[int, int, int]
    bg_rgb: Tuple[int, int, int]
    ratio: float
    passes_aa_normal: bool
    passes_aa_large: bool


def audit_palette(palette,
                  *,
                  text_pairs: Sequence[Tuple[str, str]] = None,
                  ) -> List[ContrastFinding]:
    """Run WCAG ratios for every (foreground, background) pair on
    ``palette`` (a :class:`ui2.theme.Palette`).

    Default pair set tests the colour roles that actually render
    text in the v2.0.x app. Caller may pass ``text_pairs`` for a
    custom audit (e.g. the high-contrast theme exercise).

    Returns a list — caller filters by ``passes_aa_normal`` /
    ``passes_aa_large`` for compliance gates.
    """
    if text_pairs is None:
        text_pairs = [
            ("text", "panel_bg"),
            ("text", "panel_alt_bg"),
            ("text", "window_bg"),
            ("text_muted", "panel_bg"),
            ("text_muted", "panel_alt_bg"),
            ("accent", "panel_bg"),
            ("success", "panel_bg"),
            ("warn", "panel_bg"),
            ("danger", "panel_bg"),
        ]
    out: List[ContrastFinding] = []
    for fg_name, bg_name in text_pairs:
        fg_rgb = getattr(palette, fg_name)
        bg_rgb = getattr(palette, bg_name)
        ratio = contrast_ratio(fg_rgb, bg_rgb)
        out.append(ContrastFinding(
            role_fg=fg_name,
            role_bg=bg_name,
            fg_rgb=tuple(int(x) for x in fg_rgb[:3]),
            bg_rgb=tuple(int(x) for x in bg_rgb[:3]),
            ratio=ratio,
            passes_aa_normal=ratio >= AA_NORMAL,
            passes_aa_large=ratio >= AA_LARGE,
        ))
    return out


def find_aa_failures(findings: Sequence[ContrastFinding],
                     *, allow_large: bool = False,
                     ) -> List[ContrastFinding]:
    """Return only the findings that fail AA. ``allow_large=True``
    is the right default for icon-overlay text (≥18 pt regular)."""
    threshold_check = (lambda f: f.passes_aa_large) if allow_large \
        else (lambda f: f.passes_aa_normal)
    return [f for f in findings if not threshold_check(f)]


__all__ = [
    "AA_NORMAL",
    "AA_LARGE",
    "AAA_NORMAL",
    "AAA_LARGE",
    "ContrastFinding",
    "relative_luminance",
    "contrast_ratio",
    "audit_palette",
    "find_aa_failures",
]
