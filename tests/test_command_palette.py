"""⌘K command palette — smoke import + registry wiring.

We can't test the QDialog interactions without ``pytest-qt`` and a live
``QApplication`` — the project doesn't ship with either. What we *can*
test is the integration seam:

* The module imports cleanly under a headless interpreter (so the main
  window can lazy-import it without risk).
* When a real ``QApplication`` is available, the palette instantiates
  and wires itself up against a supplied registry.
* The populate/search path reflects registry contents — since the
  filtering is delegated to :class:`CommandRegistry.search`, which has
  its own exhaustive suite (``test_commands.py``), we only need a
  lightweight "did the list get the right number of rows" check here.

The Qt tests are guarded by ``QApplication.instance()`` so they skip
cleanly on CI machines without a display.
"""
from __future__ import annotations

import importlib

import pytest

from gui.commands import Categories, Command, CommandRegistry


def _headless_qapp():
    """Return a ``QApplication`` instance if offscreen-capable, else None.

    PySide6 will happily start a QApplication with ``platform=offscreen``,
    which is enough for widget construction tests. We try that once and
    fall back to skipping when the environment blocks GUI init entirely.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return None
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception:
            return None
    return app


def test_module_imports_without_qapplication():
    """Regression guard: lazy-import paths must not execute Qt widget
    constructors at import time."""
    mod = importlib.import_module("gui.widgets.command_palette")
    assert hasattr(mod, "CommandPalette")


def _make_registry() -> CommandRegistry:
    r = CommandRegistry()
    r.register(Command(
        id="reconstruct.run", title="Reconstruct",
        category=Categories.RECONSTRUCT, callback=lambda: None,
        shortcut="Ctrl+R",
    ))
    r.register(Command(
        id="view.restore_grid", title="Restore grid layout",
        category=Categories.VIEW, callback=lambda: None, shortcut="Ctrl+0",
    ))
    r.register(Command(
        id="help.about", title="About DHM Reconstruction",
        category=Categories.HELP, callback=lambda: None,
    ))
    return r


def test_palette_populates_from_registry():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable in this environment")

    from gui.widgets.command_palette import CommandPalette

    reg = _make_registry()
    palette = CommandPalette(registry=reg)
    # ``showEvent`` is what normally calls ``_populate("")``; call it
    # directly to exercise the branch without raising a window.
    palette._populate(list(reg))
    assert palette._results_view.count() == 3


def test_palette_query_filters_results():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable in this environment")

    from gui.widgets.command_palette import CommandPalette

    reg = _make_registry()
    palette = CommandPalette(registry=reg)
    # "grid" pins to exactly one fixture command — a more distinctive query
    # than "recon" which substring-matches both "Reconstruct" and the About
    # dialog's "Reconstruction".
    palette._on_query_changed("grid")
    assert palette._results_view.count() == 1
    item = palette._results_view.item(0)
    assert item is not None
    assert item.data(0x0100) == "view.restore_grid"  # Qt.UserRole


def test_palette_greys_out_when_blocked_commands():
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable in this environment")

    from PySide6.QtCore import Qt as _Qt
    from gui.widgets.command_palette import CommandPalette

    reg = CommandRegistry()
    reg.register(Command(
        id="a.run", title="Always runnable",
        category=Categories.TOOLS, callback=lambda: None,
    ))
    reg.register(Command(
        id="b.blocked", title="Blocked right now",
        category=Categories.TOOLS, callback=lambda: None,
        when=lambda: False,
    ))

    palette = CommandPalette(registry=reg)
    palette._populate(list(reg))

    assert palette._results_view.count() == 2
    blocked_item = palette._results_view.item(1)
    assert blocked_item is not None
    assert not (blocked_item.flags() & _Qt.ItemFlag.ItemIsEnabled)


def test_palette_selecting_empty_does_not_crash():
    """Enter on an empty list should be a no-op, not an exception."""
    app = _headless_qapp()
    if app is None:
        pytest.skip("QApplication unavailable in this environment")

    from gui.widgets.command_palette import CommandPalette

    palette = CommandPalette(registry=CommandRegistry())
    palette._run_selected()  # must not raise
