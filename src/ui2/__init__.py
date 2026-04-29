"""DHM Reconstruction — Dear PyGui frontend (v2 UI, 2026-04-22).

Parallel to the PySide6 ``src/gui`` package; reuses ``src/core/*`` for
reconstruction / autofocus / QPI / depth-map pipelines without change.
Not a drop-in replacement — ships the essential DHM workflow only
(load → reconstruct → display + QPI batch) while the full PySide app
stays available via ``run_app.py``.

Entry point: ``python -m ui2.app`` or ``python src/ui2/app.py``.

DhmApp is intentionally *lazy-loaded*: importing :mod:`ui2` should
not force Dear PyGui to initialise, because many sibling modules
(``state_store``, ``reconstruction``, ``workers``) are Qt/GL-free
and useful in headless tests. Only ``from ui2.app import DhmApp``
(or attribute access via ``ui2.DhmApp``) pulls in the GL backend.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .theme import ScientificTheme

if TYPE_CHECKING:  # pragma: no cover - hint only
    from .app import DhmApp as DhmApp


def __getattr__(name: str):
    """PEP 562 lazy attribute. Resolves ``DhmApp`` only when something
    actually asks for it, so ``import ui2.state_store`` stays free of
    the Dear PyGui C extension. Tests that stub ``dearpygui`` can then
    import ``ui2`` without loading the real backend first."""
    if name == "DhmApp":
        from .app import DhmApp as _DhmApp
        return _DhmApp
    raise AttributeError(f"module 'ui2' has no attribute {name!r}")


__all__ = ["DhmApp", "ScientificTheme"]
