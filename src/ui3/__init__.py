"""ui3 — the DHM desktop app, rebuilt on PySide6 (Qt 6) + pyqtgraph.

A from-scratch replacement for the Dear PyGui ``ui2`` frontend. Reuses the
Qt-free compute layer (``core.drivers.*`` + ``core.*`` — no ui2 import
anywhere in ui3) and rebuilds the presentation with a native Qt shell, a cohesive
"instrument" design system (``ui3.design``), and pyqtgraph for imaging/3D.

Entry point: ``ui3.app.main()`` (see ``run_ui3.py`` at the repo root).

Lazy attribute access so ``import ui3.design`` / ``ui3.wcag`` / ``ui3.state``
stay Qt-free and importable without a QApplication (for headless tests).
"""
from __future__ import annotations

__all__ = ["main"]


def __getattr__(name: str):  # PEP 562 lazy — avoid importing Qt at package import
    if name == "main":
        from ui3.app import main
        return main
    raise AttributeError(f"module 'ui3' has no attribute {name!r}")
