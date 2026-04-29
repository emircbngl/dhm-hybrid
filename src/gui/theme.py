"""Theme system — light / dark / system (v1.4 UI Redesign).

A thin wrapper around :class:`QPalette`. We pick palette colours for a
handful of well-defined roles (window, text, base, button, accent,
mid) and let Qt style-sheets inside individual widgets *reference*
them via ``palette(highlight)`` / ``palette(mid)`` — the existing
widgets (toast, command palette, error drawer, progress line) already
do this, so a palette change propagates without touching those files.

Design constraints:

* **Token dict is pure data** — buildable without a QApplication, so
  tests can assert the palette shape without Qt.
* **``SYSTEM`` is real deference** — we reset to whatever default the
  platform style produced at startup, not a best-guess approximation.
* **No custom QSS** at this layer. Widget-specific polish lives in
  each widget's own ``_apply_style`` (see ``ToastHost``, etc.) and
  reads from the palette tokens we set here.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

_LOG = logging.getLogger(__name__)


class ThemeName(str, Enum):
    """Which theme the user has selected.

    * ``LIGHT``         — explicit light palette (works on any platform)
    * ``DARK``          — explicit dark palette (warm blue accent)
    * ``SYSTEM``        — reset to the platform style's default palette
                          (macOS light/dark follows the OS automatically
                          on Qt ≥ 6.5; older builds fall back to the
                          build-time system palette)
    * ``HIGH_CONTRAST`` — black background, pure-white text, saturated
                          yellow accent. Targets WCAG-AA contrast
                          (≥ 4.5:1) on every reachable role.
    """
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"
    HIGH_CONTRAST = "high_contrast"


# ---------------------------------------------------------------------------
# Token dictionaries — pure data, Qt-free.
# ---------------------------------------------------------------------------

# Keys map directly to QPalette.ColorRole names in apply_theme(); the
# string form is used so tests can assert the shape of the dict
# without importing Qt.
# Palettes are calibrated for a scientific-instrument look: restrained
# contrast, slightly warm neutrals (easier on the eye during long
# reconstruction sessions), and a muted blue accent instead of the
# over-saturated default. Nothing decorative — the intent is "reads
# well on a CCD frame sitting beside it."
_LIGHT_TOKENS: dict[str, tuple[int, int, int]] = {
    "Window":          (246, 247, 249),   # warm off-white
    "WindowText":      ( 33,  38,  45),   # near-black with warmth
    "Base":            (252, 253, 254),   # panel body
    "AlternateBase":   (236, 238, 241),
    "Text":            ( 33,  38,  45),
    "Button":          (239, 241, 244),
    "ButtonText":      ( 33,  38,  45),
    "Highlight":       ( 52, 120, 198),   # muted scientific blue
    "HighlightedText": (255, 255, 255),
    "Mid":             (148, 155, 165),   # secondary text / borders
    "Dark":            ( 98, 105, 116),
    "Light":           (250, 251, 252),
    "ToolTipBase":     (250, 247, 225),   # soft manila
    "ToolTipText":     ( 33,  38,  45),
}

_DARK_TOKENS: dict[str, tuple[int, int, int]] = {
    "Window":          ( 33,  36,  42),   # warm charcoal
    "WindowText":      (222, 226, 232),   # near-white, slight blue tint
    "Base":            ( 26,  29,  34),   # panel body (a touch darker)
    "AlternateBase":   ( 42,  46,  54),
    "Text":            (222, 226, 232),
    "Button":          ( 44,  48,  56),
    "ButtonText":      (222, 226, 232),
    "Highlight":       ( 82, 142, 208),   # muted blue — not neon
    "HighlightedText": ( 15,  18,  22),
    "Mid":             (140, 148, 160),   # slightly lifted for readability
    "Dark":            ( 64,  68,  76),
    "Light":           ( 88,  94, 104),
    "ToolTipBase":     ( 58,  62,  72),
    "ToolTipText":     (222, 226, 232),
}

# High-contrast palette tuned for WCAG-AA legibility:
# * Text 255/255/255 on Window 0/0/0  → contrast ratio 21:1 (AAA)
# * Highlight 255/215/0 on 0/0/0      → ratio 15.5:1 (AAA)
# * Muted grey 178/178/178 on black   → ratio 9.7:1 (AAA)
# Every role that can appear as text-on-background meets AA (≥ 4.5:1).
_HIGH_CONTRAST_TOKENS: dict[str, tuple[int, int, int]] = {
    "Window":          (  0,   0,   0),
    "WindowText":      (255, 255, 255),
    "Base":            (  0,   0,   0),
    "AlternateBase":   ( 20,  20,  20),
    "Text":            (255, 255, 255),
    "Button":          ( 20,  20,  20),
    "ButtonText":      (255, 255, 255),
    "Highlight":       (255, 215,   0),  # saturated yellow
    "HighlightedText": (  0,   0,   0),
    "Mid":             (178, 178, 178),
    "Dark":            (100, 100, 100),
    "Light":           (140, 140, 140),
    "ToolTipBase":     (255, 255,   0),
    "ToolTipText":     (  0,   0,   0),
}


def get_theme_tokens(theme: ThemeName) -> dict[str, tuple[int, int, int]]:
    """Return the RGB token dict for an explicit theme.

    Raises :class:`ValueError` for :data:`ThemeName.SYSTEM` — the
    system theme has no tokens, it defers to the platform style.
    """
    if theme is ThemeName.LIGHT:
        return dict(_LIGHT_TOKENS)
    if theme is ThemeName.DARK:
        return dict(_DARK_TOKENS)
    if theme is ThemeName.HIGH_CONTRAST:
        return dict(_HIGH_CONTRAST_TOKENS)
    if theme is ThemeName.SYSTEM:
        raise ValueError("SYSTEM theme has no tokens — defer to platform style")
    raise ValueError(f"unknown theme {theme!r}")


# ---------------------------------------------------------------------------
# Qt-side application.
# ---------------------------------------------------------------------------

# One-shot cache of the platform's default palette, captured on the
# very first ``apply_theme`` call so SYSTEM mode can reset to it later
# without fighting any palette we installed on top.
_DEFAULT_PALETTE = None


def apply_theme(app: Any, theme: ThemeName) -> None:
    """Install ``theme`` on a :class:`QApplication`.

    ``app`` is accepted as ``Any`` so unit tests can pass a real
    ``QApplication`` under offscreen Qt without this module needing a
    hard top-level Qt import (the tests that run in CI-lite mode
    skip this path cleanly).

    SYSTEM: resets to the style's default palette on first call (we
    remember it). Subsequent SYSTEM applies reuse the cached handle
    so a light/dark toggle doesn't drift.
    """
    from PySide6.QtGui import QColor, QPalette

    global _DEFAULT_PALETTE
    if _DEFAULT_PALETTE is None:
        try:
            _DEFAULT_PALETTE = QPalette(app.palette())
        except Exception:
            _DEFAULT_PALETTE = QPalette()

    if theme is ThemeName.SYSTEM:
        app.setPalette(QPalette(_DEFAULT_PALETTE))
        _LOG.info("theme: reset to SYSTEM palette")
        return

    tokens = get_theme_tokens(theme)
    palette = QPalette()
    for role_name, rgb in tokens.items():
        role = getattr(QPalette.ColorRole, role_name, None)
        if role is None:
            continue
        palette.setColor(role, QColor(*rgb))
    app.setPalette(palette)
    _LOG.info("theme: applied %s", theme.value)


def current_default_palette():
    """For tests: return the cached SYSTEM palette (or ``None`` if
    :func:`apply_theme` hasn't been called yet)."""
    return _DEFAULT_PALETTE


def reset_for_tests() -> None:
    """Clear the cached SYSTEM palette — tests rely on this to avoid
    state leaking across suites."""
    global _DEFAULT_PALETTE
    _DEFAULT_PALETTE = None


__all__ = [
    "ThemeName",
    "get_theme_tokens",
    "apply_theme",
    "current_default_palette",
    "reset_for_tests",
]
