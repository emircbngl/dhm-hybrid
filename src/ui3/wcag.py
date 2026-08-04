"""WCAG 2.1 contrast audit for the ui3 palettes (Qt-free).

Ports the ui2 approach: sRGB→linear luminance (WCAG Appendix A), contrast
ratio, and an ``audit_palette`` that checks the load-bearing foreground/
background role pairs against AA thresholds. Used by tests to pin the
design tokens — a token change that drops a pair below AA fails the suite.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ui3.design import Palette

AA_NORMAL = 4.5
AA_LARGE = 3.0
AAA_NORMAL = 7.0


def _hex_to_rgb(value: str) -> Tuple[float, float, float]:
    v = value.lstrip("#")
    if len(v) != 6:
        raise ValueError(f"expected #rrggbb, got {value!r}")
    return tuple(int(v[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


def _linear_channel(c: float) -> float:
    # WCAG Appendix A sRGB → linear.
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    r, g, b = (_linear_channel(c) for c in _hex_to_rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str) -> float:
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    hi, lo = (l1, l2) if l1 >= l2 else (l2, l1)
    return (hi + 0.05) / (lo + 0.05)


# Role pairs that must stay legible. (foreground, background, threshold).
def _pairs(p: Palette) -> List[Tuple[str, str, str, float]]:
    return [
        ("text/surface", p.text, p.surface, AA_NORMAL),
        ("text/bg", p.text, p.bg, AA_NORMAL),
        ("text_muted/surface", p.text_muted, p.surface, AA_LARGE),
        ("accent/surface", p.accent, p.surface, AA_LARGE),
        ("ok/surface", p.ok, p.surface, AA_LARGE),
        ("warn/surface", p.warn, p.surface, AA_LARGE),
        ("danger/surface", p.danger, p.surface, AA_LARGE),
    ]


def audit_palette(p: Palette) -> Dict[str, Dict[str, float]]:
    """Return {pair_name: {ratio, threshold, passes}} for a palette."""
    out: Dict[str, Dict[str, float]] = {}
    for name, fg, bg, thr in _pairs(p):
        ratio = contrast_ratio(fg, bg)
        out[name] = {
            "ratio": round(ratio, 2),
            "threshold": thr,
            "passes": ratio >= thr,
        }
    return out


def failing_pairs(p: Palette) -> List[str]:
    return [name for name, res in audit_palette(p).items() if not res["passes"]]
