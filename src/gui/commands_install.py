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
    # Tools
    "tools.generate_report",
    # Help
    "help.about",
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

    # -- Help ---------------------------------------------------------------
    reg.register(Command(
        id="help.about",
        title="About DHM Reconstruction",
        category=Categories.HELP,
        hint="Show version information",
        callback=lambda: window._show_about_dialog(),
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
