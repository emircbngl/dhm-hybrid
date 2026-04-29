"""Register the main-window's commands into the global :class:`CommandRegistry`.

This is the one place the ids, titles, shortcuts, and callbacks for the
window's menu/toolbar/shortcut surface are spelled out. Before v1.0.1
the same information lived in three places (``_setup_shortcuts`` in
``main_window.py``, ``_init_menus`` also in ``main_window.py``, and
``Toolbar`` in ``toolbar.py``) and disagreed with itself. Now each
command has one definition; the shortcut loop and the menu builder both
read from here, and the ⌘K palette (T1.1) will too.

Keep this module thin: it references ``window`` attributes (panels,
``image_grid``, ``_trigger_reconstruction``, etc.) but doesn't do any
work itself. The callbacks are short lambdas that delegate back to the
window. That keeps the *semantics* of a command — id, title, shortcut —
separable from the *implementation*, which remains in ``main_window``.
"""
from __future__ import annotations

import logging
from typing import Any

from gui.commands import (
    Categories,
    Command,
    CommandRegistry,
    get_registry,
)

_LOG = logging.getLogger(__name__)


# Command ids the main window owns. Keeping this list explicit means
# ``install_main_window_commands`` can be called more than once (hot
# reload, test fixture rebuild) without leaving stale entries behind.
MAIN_WINDOW_COMMAND_IDS: tuple[str, ...] = (
    # View / layout
    "view.maximize.input",
    "view.maximize.amplitude",
    "view.maximize.phase",
    "view.maximize.spectrum",
    "view.restore_grid",
    "view.reset_layout",
    # Reconstruct
    "reconstruct.run",
    # Autofocus
    "autofocus.find_multiple",
    # Tools
    "tools.generate_report",
    "tools.export_qpi_csv",
    "tools.compute_depth_map",
    "tools.toggle_depth_overlay",
    "tools.qpi_batch_candidates",
    "tools.export_tomography_bundle",
    # View / theme
    "view.theme.light",
    "view.theme.dark",
    "view.theme.system",
    "view.theme.high_contrast",
    # Help
    "help.about",
    "help.show_onboarding",
    "help.show_overlay",
    # AI assistant (v3.0 prep)
    "ai.toggle_panel",
)


def install_main_window_commands(
    window: Any, *, registry: CommandRegistry | None = None,
) -> CommandRegistry:
    """Register every main-window-owned command.

    Safe to call multiple times — each call clears the previously
    registered set by id and re-registers. Returns the registry used so
    the caller can thread it into a palette or shortcut installer.
    """
    reg = registry or get_registry()

    # Idempotency: unregister window-owned ids before re-adding. Unknown
    # ids are silent, so this is cheap on a fresh registry too.
    for cid in MAIN_WINDOW_COMMAND_IDS:
        reg.unregister(cid)

    # -- View ---------------------------------------------------------------
    # The maximize commands rely on ``image_grid`` + the four panel
    # attributes already being constructed. ``install_main_window_commands``
    # must be called after panel setup.
    reg.register(Command(
        id="view.maximize.input",
        title="Maximize input panel",
        category=Categories.VIEW,
        shortcut="Ctrl+1",
        hint="Toggle focus on the raw hologram",
        callback=lambda: window.image_grid.toggle_maximize(window.panel_input),
    ))
    reg.register(Command(
        id="view.maximize.amplitude",
        title="Maximize amplitude panel",
        category=Categories.VIEW,
        shortcut="Ctrl+2",
        hint="Toggle focus on the reconstructed amplitude",
        callback=lambda: window.image_grid.toggle_maximize(window.panel_amp),
    ))
    reg.register(Command(
        id="view.maximize.phase",
        title="Maximize phase panel",
        category=Categories.VIEW,
        shortcut="Ctrl+3",
        hint="Toggle focus on the unwrapped phase",
        callback=lambda: window.image_grid.toggle_maximize(window.panel_phase),
    ))
    reg.register(Command(
        id="view.maximize.spectrum",
        title="Maximize spectrum panel",
        category=Categories.VIEW,
        shortcut="Ctrl+4",
        hint="Toggle focus on the Fourier spectrum",
        callback=lambda: window.image_grid.toggle_maximize(window.panel_spectrum),
    ))
    reg.register(Command(
        id="view.restore_grid",
        title="Restore grid layout",
        category=Categories.VIEW,
        shortcut="Ctrl+0",
        hint="Return all four panels to the default 2×2 grid",
        callback=lambda: window.image_grid.restore_grid(),
    ))
    reg.register(Command(
        id="view.reset_layout",
        title="Reset layout to defaults",
        category=Categories.VIEW,
        shortcut="Ctrl+Shift+R",
        hint="Forget dock positions and grid state",
        callback=lambda: window._reset_layout_to_defaults(),
    ))

    # -- Reconstruct --------------------------------------------------------
    reg.register(Command(
        id="reconstruct.run",
        title="Reconstruct",
        category=Categories.RECONSTRUCT,
        shortcut="Ctrl+R",
        hint="Run the off-axis pipeline on the current frame",
        callback=lambda: window._trigger_reconstruction(),
    ))

    # -- Tools --------------------------------------------------------------
    reg.register(Command(
        id="tools.generate_report",
        title="Generate report…",
        category=Categories.TOOLS,
        hint="Export an HTML report from the last reconstruction, autofocus, and QPI",
        callback=lambda: window._on_generate_report_triggered(),
    ))

    # -- Autofocus (multi-focus prototype) ---------------------------------
    reg.register(Command(
        id="autofocus.find_multiple",
        title="Find multiple focus planes…",
        category=Categories.AUTOFOCUS,
        hint="Scan the focus landscape and list every plausible focus z",
        callback=lambda: window._on_find_focus_candidates_triggered(),
        # Gated on a loaded hologram — without one, the scan has no input.
        when=lambda: getattr(window, "_loaded_array", None) is not None,
    ))

    reg.register(Command(
        id="tools.export_qpi_csv",
        title="Export QPI CSV…",
        category=Categories.TOOLS,
        hint="Write the last QPI result to a CSV row (LIMS/R/Python friendly)",
        callback=lambda: window._on_export_qpi_csv_triggered(),
    ))

    reg.register(Command(
        id="tools.compute_depth_map",
        title="Compute depth map…",
        category=Categories.TOOLS,
        hint="Per-pixel best-focus z — tomographic depth map saved as NPZ + CSV",
        callback=lambda: window._on_compute_depth_map_triggered(),
        when=lambda: getattr(window, "_loaded_array", None) is not None,
    ))

    reg.register(Command(
        id="tools.qpi_batch_candidates",
        title="Run QPI batch for focus candidates…",
        category=Categories.TOOLS,
        hint="Compute QPI once per detected focus plane, save per-candidate CSV",
        callback=lambda: window._on_qpi_batch_candidates_triggered(),
        when=lambda: getattr(window, "_loaded_array", None) is not None,
    ))

    reg.register(Command(
        id="tools.toggle_depth_overlay",
        title="Toggle depth overlay on phase panel",
        category=Categories.TOOLS,
        hint="Compute depth map and tint the phase panel with colormap",
        callback=lambda: window._on_toggle_depth_overlay_triggered(),
        when=lambda: getattr(window, "_loaded_array", None) is not None,
    ))

    reg.register(Command(
        id="tools.export_tomography_bundle",
        title="Export tomography bundle…",
        category=Categories.TOOLS,
        hint="Write depth NPZ + depth CSV + cluster CSV + QPI batch into one directory",
        callback=lambda: window._on_export_tomography_bundle_triggered(),
        when=lambda: getattr(window, "_loaded_array", None) is not None,
    ))

    # -- View / theme (v1.4 UI Redesign) -----------------------------------
    reg.register(Command(
        id="view.theme.light",
        title="Theme: Light",
        category=Categories.VIEW,
        hint="Switch the UI to the explicit light palette",
        callback=lambda: window._apply_theme_by_name("light"),
    ))
    reg.register(Command(
        id="view.theme.dark",
        title="Theme: Dark",
        category=Categories.VIEW,
        hint="Switch the UI to the explicit dark palette",
        callback=lambda: window._apply_theme_by_name("dark"),
    ))
    reg.register(Command(
        id="view.theme.system",
        title="Theme: System default",
        category=Categories.VIEW,
        hint="Follow the platform's system appearance (macOS light/dark)",
        callback=lambda: window._apply_theme_by_name("system"),
    ))
    reg.register(Command(
        id="view.theme.high_contrast",
        title="Theme: High contrast",
        category=Categories.VIEW,
        hint="WCAG-AA high-contrast palette — pure black / white / yellow",
        callback=lambda: window._apply_theme_by_name("high_contrast"),
    ))

    # -- Help ---------------------------------------------------------------
    reg.register(Command(
        id="help.about",
        title="About DHM Reconstruction",
        category=Categories.HELP,
        hint="Show version information",
        callback=lambda: window._show_about_dialog(),
    ))
    reg.register(Command(
        id="help.show_onboarding",
        title="Show onboarding",
        category=Categories.HELP,
        hint="Re-open the first-run walkthrough of the app",
        callback=lambda: window._show_onboarding(),
    ))
    reg.register(Command(
        id="help.show_overlay",
        title="Show contextual help (?)",
        category=Categories.HELP,
        hint="List tooltips of every currently-visible control",
        callback=lambda: window._show_help_overlay(),
    ))

    # -- AI assistant -------------------------------------------------------
    reg.register(Command(
        id="ai.toggle_panel",
        title="Toggle AI assistant panel",
        category=Categories.HELP,
        shortcut="Ctrl+Shift+A",
        hint="Open or close the local-LLM AI co-pilot dock",
        callback=lambda: window._toggle_ai_panel(),
    ))

    _LOG.debug(
        "commands: installed %d main-window commands",
        len(MAIN_WINDOW_COMMAND_IDS),
    )
    return reg


# ---------------------------------------------------------------------------
# Shortcut installation — walks the registry and materializes one
# ``QShortcut`` per command that has a non-empty default shortcut. Kept
# here (not in ``commands.py``) so the core module has no Qt imports.
# ---------------------------------------------------------------------------

def install_shortcuts(
    window: Any, *, registry: CommandRegistry | None = None,
) -> list[Any]:
    """Create QShortcut instances for every command with a default shortcut.

    Returns the list of created QShortcuts — Qt parents them via
    ``window``, so the caller only needs to hold the return value if it
    wants to iterate (e.g. for unregistering).
    """
    # QShortcut moved from QtWidgets to QtGui in PySide6 6.0.
    from PySide6.QtGui import QKeySequence, QShortcut

    reg = registry or get_registry()
    made: list[Any] = []
    for cmd in reg:
        if not cmd.shortcut:
            continue
        sc = QShortcut(QKeySequence(cmd.shortcut), window)
        sc.setObjectName(f"sc_{cmd.id.replace('.', '_')}")
        sc.activated.connect(cmd.callback)
        made.append(sc)
    _LOG.debug("commands: installed %d QShortcuts", len(made))
    return made


__all__ = [
    "MAIN_WINDOW_COMMAND_IDS",
    "install_main_window_commands",
    "install_shortcuts",
]
