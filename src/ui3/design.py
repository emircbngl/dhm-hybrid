"""ui3 design system — pure-data tokens + a Qt stylesheet builder.

Qt-free on purpose: the palettes and scales are plain dataclasses/dicts so
they can be unit-tested and WCAG-audited (``ui3.wcag``) without a running
QApplication. ``build_qss(palette)`` is the only Qt-adjacent function and it
only emits a string.

Aesthetic — "instrument": a calm dark scientific-instrument look. Hierarchy
by scale & space, not weight. Accent colours are reserved for *live state*
(running, armed, connected, error). Numerals are mono so readouts align.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class Palette:
    """A full colour set. All values are ``#rrggbb`` strings."""
    name: str
    bg: str            # window background (the darkest surface)
    surface: str       # panels / cards
    surface_high: str  # raised controls, headers
    border: str        # 1px separators / control outlines
    text: str          # primary text
    text_muted: str    # secondary text, labels, disabled
    accent: str        # primary accent — selection, focus, running
    armed: str         # a control is armed / pending (amber)
    ok: str            # success / connected (green)
    warn: str          # caution (amber-ish)
    danger: str        # error / destructive (red)
    # Image display defaults (viewport background + fallback colormap name).
    view_bg: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


# --- The three shipped palettes ------------------------------------------
# Values chosen so text/text_muted/accent/ok/warn/danger all clear WCAG-AA
# (>=4.5:1 for text, >=3.0:1 for large/UI) on ``surface`` — verified in
# tests/test_ui3_design.py via ui3.wcag.

DARK = Palette(
    name="dark",
    bg="#0d1117",
    surface="#151b23",
    surface_high="#1c2330",
    border="#2a3441",
    text="#e6edf3",
    text_muted="#9aa7b4",
    accent="#4cc3d6",
    armed="#e0a458",
    ok="#5fb56a",
    warn="#e0a458",
    danger="#f0645c",
    view_bg="#05070a",
)

MIDNIGHT = Palette(
    name="midnight",
    bg="#080a0f",
    surface="#0f1420",
    surface_high="#161d2c",
    border="#232c3d",
    text="#e8eef7",
    text_muted="#93a1b5",
    accent="#5aa0ff",
    armed="#d9a441",
    ok="#57b06a",
    warn="#d9a441",
    danger="#ec6a62",
    view_bg="#03040a",
)

LIGHT = Palette(
    name="light",
    bg="#f2f4f7",
    surface="#ffffff",
    surface_high="#eef1f5",
    border="#d3d9e0",
    text="#1a2230",
    text_muted="#5a6675",
    accent="#0e7d90",
    armed="#9a6512",
    ok="#2f7d3a",
    warn="#9a6512",
    danger="#c0392b",
    view_bg="#e9edf2",
)

HIGH_CONTRAST = Palette(
    name="high_contrast",
    bg="#000000",
    surface="#000000",
    surface_high="#111111",
    border="#ffffff",
    text="#ffffff",
    text_muted="#e0e0e0",
    accent="#ffd000",
    armed="#ffd000",
    ok="#48ff6a",
    warn="#ffd000",
    danger="#ff5a54",
    view_bg="#000000",
)

PALETTES: Dict[str, Palette] = {
    p.name: p for p in (DARK, MIDNIGHT, LIGHT, HIGH_CONTRAST)
}
DEFAULT_PALETTE = "dark"


def get_palette(name: str) -> Palette:
    return PALETTES.get(str(name or "").lower(), DARK)


# --- Scales ---------------------------------------------------------------

class Space:
    """4-based spacing scale (px)."""
    xs = 4
    sm = 8
    md = 12
    lg = 16
    xl = 24
    xxl = 32


class Type:
    """Type scale (px). ``mono`` marks readouts that must align."""
    small = 11
    body = 13
    label = 13
    title = 15
    heading = 20
    display = 28
    mono_family = '"SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace'
    sans_family = ('-apple-system, "SF Pro Text", "Segoe UI", '
                   '"Helvetica Neue", sans-serif')


class Radius:
    card = 6
    control = 4


# --- Stylesheet -----------------------------------------------------------

# Arrow-shape version — bump if the drawn triangle geometry changes so cached
# PNGs from an older shape are not reused.
_ARROW_V = "v2"


def _ensure_arrow_icons(palette: Palette):
    """Render crisp up/down/chevron PNGs in ``palette``'s muted colour and
    return their file paths, or ``None`` when there is no live QApplication.

    Qt's stylesheet loader draws no arrow for the CSS border-triangle trick and
    ignores ``data:`` URIs (both verified empirically), so a real image file is
    the only reliable way to get a clean spin/combo arrow. Icons are cached by
    colour under ``~/.dhm-reconstruction/icons`` so this is a no-op after the
    first theme apply. Headless/Qt-free callers (tests) get ``None`` and fall
    back to arrow-less controls.
    """
    try:
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            return None
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtGui import QColor, QPainter, QPixmap, QPolygon
    except Exception:
        return None

    import pathlib

    color = palette.text_muted
    tag = color.lstrip("#")
    cache = pathlib.Path.home() / ".dhm-reconstruction" / "icons"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    # key -> (direction, logical width, logical height). The QSS references
    # these logical sizes; the PNG is rendered at 2x so it stays crisp when Qt
    # upscales it on a Retina/HiDPI display.
    scale = 2
    specs = {"up": ("up", 9, 6), "down": ("down", 9, 6), "chev": ("down", 11, 7)}
    paths = {}
    for key, (direction, w, h) in specs.items():
        fp = cache / f"arrow-{key}-{tag}-{_ARROW_V}.png"
        if not fp.exists():
            pw, ph = w * scale, h * scale
            pm = QPixmap(pw, ph)
            pm.fill(Qt.transparent)
            pt = QPainter(pm)
            pt.setRenderHint(QPainter.Antialiasing)
            pt.setPen(Qt.NoPen)
            pt.setBrush(QColor(color))
            if direction == "up":
                poly = QPolygon([QPoint(0, ph - 1), QPoint(pw - 1, ph - 1), QPoint(pw // 2, 0)])
            else:
                poly = QPolygon([QPoint(0, 0), QPoint(pw - 1, 0), QPoint(pw // 2, ph - 1)])
            pt.drawPolygon(poly)
            pt.end()
            if not pm.save(str(fp), "PNG"):
                return None
        paths[key] = str(fp).replace("\\", "/")
    return paths


def build_qss(palette: Palette) -> str:
    """Return a Qt stylesheet string for ``palette``.

    Kept in one place so a theme switch is a single ``setStyleSheet``.
    Widgets reference roles via object names / classes where they need
    accenting; everything else inherits these base rules.
    """
    p = palette
    icons = _ensure_arrow_icons(p)
    if icons:
        # Quote the paths so a space in $HOME doesn't break the QSS url().
        spin_arrow_qss = (
            f"QSpinBox::up-arrow, QDoubleSpinBox::up-arrow "
            f'{{ image: url("{icons["up"]}"); width: 9px; height: 6px; }}\n'
            f"    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow "
            f'{{ image: url("{icons["down"]}"); width: 9px; height: 6px; }}'
        )
        combo_arrow_qss = (
            f"QComboBox::down-arrow "
            f'{{ image: url("{icons["chev"]}"); width: 11px; height: 7px; }}'
        )
    else:
        # No QApplication (headless/pure): suppress the native arrow rather than
        # fall back to the border-trick, which renders as blank squares.
        spin_arrow_qss = (
            "QSpinBox::up-arrow, QDoubleSpinBox::up-arrow, "
            "QSpinBox::down-arrow, QDoubleSpinBox::down-arrow "
            "{ width: 0; height: 0; }"
        )
        combo_arrow_qss = "QComboBox::down-arrow { width: 0; height: 0; }"
    return f"""
    QWidget {{
        background: {p.bg};
        color: {p.text};
        font-family: {Type.sans_family};
        font-size: {Type.body}px;
    }}
    QMainWindow::separator {{ background: {p.bg}; width: 6px; height: 6px; }}
    QMainWindow::separator:hover {{ background: {p.border}; }}
    /* Splitter handles (the central imaging grid dividers) — match the dock
       separators so the whole window divides consistently. */
    QSplitter::handle {{ background: {p.bg}; }}
    QSplitter::handle:horizontal {{ width: 6px; }}
    QSplitter::handle:vertical {{ height: 6px; }}
    QSplitter::handle:hover {{ background: {p.border}; }}
    QToolTip {{ background: {p.surface_high}; color: {p.text};
                border: 1px solid {p.border}; padding: {Space.xs}px {Space.sm}px; }}

    QLabel {{ background: transparent; }}
    QLabel[role="muted"] {{ color: {p.text_muted}; }}
    QLabel[role="heading"] {{
        font-size: {Type.heading}px; font-weight: 600; color: {p.text};
    }}
    QLabel[role="mono"] {{ font-family: {Type.mono_family}; }}
    QLabel[role="readout"] {{
        font-family: {Type.mono_family}; font-size: {Type.title}px;
        color: {p.text};
    }}
    /* Status-tinted labels (e.g. the AI health pill) — the ● dot + text
       colour signals connected/caution/error at a glance. */
    QLabel[role="ok"] {{ color: {p.ok}; }}
    QLabel[role="warn"] {{ color: {p.warn}; }}
    QLabel[role="danger"] {{ color: {p.danger}; }}

    /* Cards / panels */
    QFrame[role="card"], QGroupBox {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {Radius.card}px;
    }}
    QGroupBox {{
        margin-top: {Space.lg}px;
        padding: {Space.lg}px {Space.md}px {Space.md}px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left;
        left: {Space.md}px; padding: 0 {Space.xs}px;
        color: {p.text_muted}; font-size: {Type.small}px;
        text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
    }}
    /* Checkable group titles carry a checkbox — style it like a real
       QCheckBox so it stays visible on every theme (the native indicator
       vanished on the high-contrast black background). */
    QGroupBox::indicator {{
        width: 14px; height: 14px; border: 1px solid {p.border};
        border-radius: 3px; background: {p.surface_high};
    }}
    QGroupBox::indicator:hover {{ border-color: {p.accent}; }}
    QGroupBox::indicator:checked {{
        background: {p.accent}; border-color: {p.accent};
    }}
    /* A collapsed checkable group (property set in recon_panel) sheds its
       card frame so it reads as a disclosure row, not an empty bordered box. */
    QGroupBox[collapsed="true"] {{
        border: none; background: transparent;
        padding: 0 {Space.md}px; margin-top: {Space.lg}px;
    }}

    /* Docks */
    QDockWidget {{ titlebar-close-icon: none; color: {p.text_muted};
                   font-size: {Type.small}px; }}
    QDockWidget::title {{
        background: {p.surface_high}; padding: {Space.sm}px {Space.md}px;
        border-bottom: 1px solid {p.border};
        text-transform: uppercase; letter-spacing: 1px;
    }}

    /* Inputs — comfortable, consistent control height */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background: {p.surface_high};
        border: 1px solid {p.border};
        border-radius: {Radius.control}px;
        padding: {Space.xs}px {Space.sm}px;
        min-height: {Space.xl}px;
        selection-background-color: {p.accent};
        selection-color: {p.bg};
    }}
    QPlainTextEdit, QTextEdit {{ min-height: 0; }}
    QSpinBox, QDoubleSpinBox {{ font-family: {Type.mono_family};
                               padding-right: 2px; }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
        border: 1px solid {p.text_muted};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
    QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {p.accent}; }}

    /* Spin buttons — integrated into the field (a divider + arrow), not
       floating dark squares. Blend with the field, light up on hover. */
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border; subcontrol-position: top right;
        width: 16px; border-left: 1px solid {p.border}; background: transparent;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border; subcontrol-position: bottom right;
        width: 16px; border-left: 1px solid {p.border}; background: transparent;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {p.accent};
    }}
    {spin_arrow_qss}
    QComboBox::drop-down {{
        subcontrol-origin: border; subcontrol-position: top right;
        width: 20px; border-left: 1px solid {p.border}; background: transparent;
    }}
    {combo_arrow_qss}
    QComboBox QAbstractItemView {{
        background: {p.surface_high}; border: 1px solid {p.border};
        padding: {Space.xs}px;
        selection-background-color: {p.accent}; selection-color: {p.bg};
        outline: none;
    }}

    /* Buttons */
    QPushButton {{
        background: {p.surface_high};
        border: 1px solid {p.border};
        border-radius: {Radius.control}px;
        padding: {Space.sm}px {Space.md}px;
        min-height: {Space.xl}px;
        color: {p.text};
    }}
    QPushButton:hover {{ border: 1px solid {p.accent}; background: {p.surface}; }}
    QPushButton:pressed {{ background: {p.bg}; }}
    QPushButton:disabled {{ color: {p.text_muted}; border-color: {p.border};
                            background: {p.surface}; }}
    QPushButton[role="primary"] {{
        background: {p.accent}; color: {p.bg}; border: 1px solid {p.accent};
        font-weight: 600;
    }}
    QPushButton[role="primary"]:hover {{ border: 1px solid {p.text}; }}
    QPushButton[role="primary"]:pressed {{ background: {p.accent}; }}
    QPushButton[role="danger"] {{ border-color: {p.danger}; color: {p.danger}; }}
    QPushButton[role="danger"]:hover {{ border-color: {p.danger};
                                        background: {p.danger}; color: {p.bg}; }}

    QCheckBox, QRadioButton {{ spacing: {Space.sm}px; background: transparent; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px; height: 16px; border: 1px solid {p.border};
        border-radius: 3px; background: {p.surface_high};
    }}
    QRadioButton::indicator {{ border-radius: 9px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {p.accent};
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: {p.accent}; border-color: {p.accent};
    }}

    /* Tabs — the feature-panel switcher. Do NOT set an elide-inducing max
       width or per-tab font shrink; keep labels full ('Autofocus', not
       'Autof…'). */
    QTabWidget::pane {{ border: none; }}
    QTabBar::tab {{
        background: transparent; color: {p.text_muted};
        padding: {Space.sm}px {Space.md}px;
        border: none; border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:hover {{ color: {p.text}; }}
    QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}

    /* Tables / lists */
    QTableWidget, QTableView, QListWidget, QTreeWidget {{
        background: {p.surface}; border: 1px solid {p.border};
        border-radius: {Radius.control}px;
        gridline-color: {p.border};
        alternate-background-color: {p.surface_high};
        outline: none;
    }}
    QListWidget::item, QTreeWidget::item {{ padding: {Space.xs}px; }}
    QListWidget::item:selected, QTreeWidget::item:selected,
    QTableWidget::item:selected {{ background: {p.accent}; color: {p.bg}; }}
    QHeaderView::section {{
        background: {p.surface_high}; color: {p.text_muted};
        border: none; border-bottom: 1px solid {p.border};
        padding: {Space.sm}px {Space.sm}px;
        text-transform: uppercase; font-size: {Type.small}px;
        letter-spacing: 1px;
    }}

    /* Menus + status */
    QMenuBar {{ background: {p.surface}; border-bottom: 1px solid {p.border}; }}
    QMenuBar::item {{ padding: {Space.xs}px {Space.sm}px; background: transparent; }}
    QMenuBar::item:selected {{ background: {p.surface_high}; }}
    QMenu {{ background: {p.surface_high}; border: 1px solid {p.border};
             padding: {Space.xs}px; }}
    QMenu::item {{ padding: {Space.xs}px {Space.lg}px; border-radius: {Radius.control}px; }}
    QMenu::item:selected {{ background: {p.accent}; color: {p.bg}; }}
    QMenu::separator {{ height: 1px; background: {p.border};
                        margin: {Space.xs}px {Space.sm}px; }}
    QStatusBar {{ background: {p.surface}; border-top: 1px solid {p.border};
                  color: {p.text_muted}; }}
    QStatusBar::item {{ border: none; }}

    /* Scrollbars — thin, muted, fade in on hover */
    QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 4px;
                                   min-height: 32px; margin: 2px; }}
    QScrollBar::handle:vertical:hover {{ background: {p.text_muted}; }}
    QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {p.border}; border-radius: 4px;
                                     min-width: 32px; margin: 2px; }}
    QScrollBar::handle:horizontal:hover {{ background: {p.text_muted}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* Sliders */
    QSlider::groove:horizontal {{ height: 4px; background: {p.border};
                                  border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {p.accent}; width: 14px;
                                  margin: -6px 0; border-radius: 7px; }}
    QSlider::handle:horizontal:hover {{ background: {p.text}; }}

    QProgressBar {{ background: {p.surface_high}; border: 1px solid {p.border};
                    border-radius: {Radius.control}px; text-align: center;
                    min-height: 6px; color: {p.text_muted}; }}
    QProgressBar::chunk {{ background: {p.accent}; border-radius: {Radius.control}px; }}
    """
