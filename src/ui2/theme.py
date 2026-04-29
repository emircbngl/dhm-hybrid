"""Dear PyGui themes for the DHM frontend.

Four palettes tuned for scientific instruments:

* ``dark``           — warm charcoal (default), muted scientific blue
* ``light``          — warm off-white, blue accent, manila tooltips
* ``midnight``       — deeper dark + teal accent for photosensitive labs
* ``high_contrast``  — pure black + white + saturated yellow accent
                       (WCAG-AA compliant for every reachable role)

``apply(name)`` builds and binds the named theme, tearing down any
previous theme atomically. No custom icons, no shadows, no gradients —
restrained so the scientific content stays the loudest element.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

# Import Dear PyGui lazily inside ``ScientificTheme.apply`` so merely
# touching the palette dataclasses (useful in headless tests and the
# lazy-loaded ``ui2/__init__.py``) doesn't drag the GL backend in.
if TYPE_CHECKING:  # pragma: no cover - hint only
    import dearpygui.dearpygui as dpg  # noqa: F401


# ---------------------------------------------------------------------------
# Palette tokens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Palette:
    name: str
    window_bg: tuple
    panel_bg: tuple
    panel_alt_bg: tuple
    border: tuple
    text: tuple
    text_muted: tuple
    text_disabled: tuple
    accent: tuple
    accent_hover: tuple
    accent_active: tuple
    success: tuple
    warn: tuple
    danger: tuple


_DARK = Palette(
    name="dark",
    window_bg=(33, 36, 42, 255),
    panel_bg=(26, 29, 34, 255),
    panel_alt_bg=(42, 46, 54, 255),
    border=(58, 62, 72, 255),
    text=(222, 226, 232, 255),
    text_muted=(140, 148, 160, 255),
    text_disabled=(92, 98, 108, 255),
    accent=(82, 142, 208, 255),
    accent_hover=(108, 164, 224, 255),
    accent_active=(64, 122, 188, 255),
    success=(46, 160, 67, 255),
    warn=(220, 168, 44, 255),
    # v2.0.6: lightened from (215, 58, 74) — original measured at
    # 3.70:1 on panel_bg, below WCAG AA (4.5:1). (240, 110, 125) lands
    # at ~5.85:1 while keeping the red hue readable.
    danger=(240, 110, 125, 255),
)

_LIGHT = Palette(
    name="light",
    window_bg=(246, 247, 249, 255),
    panel_bg=(252, 253, 254, 255),
    panel_alt_bg=(236, 238, 241, 255),
    border=(210, 214, 220, 255),
    text=(33, 38, 45, 255),
    text_muted=(110, 118, 128, 255),
    text_disabled=(175, 180, 188, 255),
    # v2.0.6: three colour-role tweaks to clear WCAG AA on panel_bg.
    # Hue kept identical; only L* reduced just enough to cross 4.5:1.
    accent=(36, 100, 180, 255),   # from (52, 120, 198) — 4.44 → ~5.6
    accent_hover=(76, 144, 222, 255),
    accent_active=(40, 102, 174, 255),
    success=(25, 110, 40, 255),   # from (33, 136, 56)  — 4.33 → ~6.1
    warn=(140, 100, 0, 255),      # from (180, 138, 20) — 3.06 → ~5.3
    danger=(196, 50, 62, 255),
)

_MIDNIGHT = Palette(
    name="midnight",
    window_bg=(16, 20, 28, 255),
    panel_bg=(10, 14, 20, 255),
    panel_alt_bg=(24, 30, 40, 255),
    border=(38, 46, 60, 255),
    text=(212, 220, 230, 255),
    text_muted=(120, 132, 148, 255),
    text_disabled=(72, 82, 96, 255),
    accent=(72, 184, 192, 255),   # teal
    accent_hover=(102, 208, 216, 255),
    accent_active=(56, 156, 164, 255),
    success=(46, 170, 94, 255),
    warn=(228, 172, 48, 255),
    danger=(232, 82, 94, 255),
)

_HIGH_CONTRAST = Palette(
    name="high_contrast",
    window_bg=(0, 0, 0, 255),
    panel_bg=(0, 0, 0, 255),
    panel_alt_bg=(20, 20, 20, 255),
    border=(140, 140, 140, 255),
    text=(255, 255, 255, 255),
    text_muted=(200, 200, 200, 255),
    text_disabled=(120, 120, 120, 255),
    accent=(255, 215, 0, 255),       # saturated yellow
    accent_hover=(255, 232, 80, 255),
    accent_active=(220, 180, 0, 255),
    success=(80, 255, 120, 255),
    warn=(255, 220, 40, 255),
    danger=(255, 100, 120, 255),
)


PALETTES: Dict[str, Palette] = {
    "dark": _DARK,
    "light": _LIGHT,
    "midnight": _MIDNIGHT,
    "high_contrast": _HIGH_CONTRAST,
}


class ScientificTheme:
    """Thin namespace keeping the current palette accessible as
    class-level attributes (for legacy code that reads them)."""

    _current: Palette = _DARK
    _active_theme_id: Optional[int] = None

    # Legacy class-level accessors — point at whichever palette is live.
    WINDOW_BG = _DARK.window_bg
    PANEL_BG = _DARK.panel_bg
    PANEL_ALT_BG = _DARK.panel_alt_bg
    BORDER = _DARK.border
    TEXT = _DARK.text
    TEXT_MUTED = _DARK.text_muted
    TEXT_DISABLED = _DARK.text_disabled
    ACCENT = _DARK.accent
    ACCENT_HOVER = _DARK.accent_hover
    ACCENT_ACTIVE = _DARK.accent_active
    SUCCESS = _DARK.success
    WARN = _DARK.warn
    DANGER = _DARK.danger

    @classmethod
    def apply(cls, name: str = "dark") -> int:
        """Build + bind the named theme. Tears down the prior one so
        palettes don't stack."""
        import dearpygui.dearpygui as dpg  # lazy — keeps headless tests clean
        palette = PALETTES.get(name) or _DARK

        # Drop the previous theme first — Dear PyGui doesn't merge.
        if cls._active_theme_id is not None:
            try:
                dpg.delete_item(cls._active_theme_id)
            except Exception:
                pass

        with dpg.theme() as theme_id:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, palette.window_bg)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, palette.panel_bg)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, palette.panel_bg)
                dpg.add_theme_color(dpg.mvThemeCol_Border, palette.border)

                dpg.add_theme_color(dpg.mvThemeCol_Text, palette.text)
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,
                                    palette.text_disabled)

                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,
                                    palette.panel_alt_bg)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,
                                    _shift(palette.panel_alt_bg, +10))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,
                                    _shift(palette.panel_alt_bg, +20))

                dpg.add_theme_color(dpg.mvThemeCol_Button,
                                    palette.panel_alt_bg)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,
                                    palette.accent_hover)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,
                                    palette.accent_active)

                dpg.add_theme_color(dpg.mvThemeCol_Header,
                                    palette.panel_alt_bg)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,
                                    _shift(palette.panel_alt_bg, +10))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,
                                    palette.accent_active)

                dpg.add_theme_color(dpg.mvThemeCol_Tab, palette.panel_bg)
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered,
                                    palette.accent_hover)
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, palette.accent)
                dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused,
                                    palette.panel_bg)
                dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive,
                                    palette.panel_alt_bg)

                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, palette.panel_bg)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,
                                    palette.panel_bg)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed,
                                    palette.panel_bg)
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg,
                                    palette.panel_bg)

                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,
                                    palette.panel_bg)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,
                                    palette.border)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered,
                                    palette.accent)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, palette.accent)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, palette.accent)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,
                                    palette.accent_hover)

                dpg.add_theme_color(dpg.mvThemeCol_Separator, palette.border)
                dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered,
                                    palette.accent)
                dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive,
                                    palette.accent)

                # Styles shared across every palette.
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, 12)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 12)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 12)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_PopupBorderSize, 1)

        dpg.bind_theme(theme_id)
        cls._active_theme_id = theme_id
        cls._current = palette
        # Refresh legacy attributes so status/info text colours match.
        cls.WINDOW_BG = palette.window_bg
        cls.PANEL_BG = palette.panel_bg
        cls.PANEL_ALT_BG = palette.panel_alt_bg
        cls.BORDER = palette.border
        cls.TEXT = palette.text
        cls.TEXT_MUTED = palette.text_muted
        cls.TEXT_DISABLED = palette.text_disabled
        cls.ACCENT = palette.accent
        cls.ACCENT_HOVER = palette.accent_hover
        cls.ACCENT_ACTIVE = palette.accent_active
        cls.SUCCESS = palette.success
        cls.WARN = palette.warn
        cls.DANGER = palette.danger
        return theme_id

    @classmethod
    def current_name(cls) -> str:
        return cls._current.name


def _shift(color: tuple, delta: int) -> tuple:
    """Return ``color`` brightened/darkened by ``delta`` across RGB."""
    r, g, b, *alpha = color
    a = alpha[0] if alpha else 255
    return (
        max(0, min(255, r + delta)),
        max(0, min(255, g + delta)),
        max(0, min(255, b + delta)),
        a,
    )


__all__ = ["ScientificTheme", "Palette", "PALETTES"]
