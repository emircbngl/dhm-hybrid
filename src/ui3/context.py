"""PanelContext — the frozen contract every ui3 feature panel builds against.

A panel is a ``QWidget`` in ``ui3/panels/<name>_panel.py`` (or a dialog in
``ui3/dialogs/``) whose ``__init__(self, ctx: PanelContext, parent=None)`` takes
this context. It gives the panel everything it needs from the shell without a
back-reference to MainWindow internals, so panels are independently testable
(construct with a fake/real context under offscreen Qt) and the integration
step is mechanical (``MainWindow.mount_panel``).

The context is a thin facade over MainWindow: callables + the shared bridge.
Nothing here imports a panel, so there is no import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core.drivers.reconstruction import ReconParams
from ui3.bridge import WorkerBridge


@dataclass
class PanelContext:
    """Everything a feature panel is allowed to touch on the shell."""

    bridge: WorkerBridge
    # Current live reconstruction params (the SAME instance the shell edits;
    # panels may read and mutate fields, then call ``params_changed``).
    get_params: Callable[[], ReconParams]
    params_changed: Callable[[], None]
    # Current inputs / last results.
    hologram_path: Callable[[], Optional[Path]]
    last_recon: Callable[[], Any]          # ReconResult | None
    last_depth: Callable[[], Any]          # DepthMapResultWrap | None
    # Field access for observation/vision tools (target ∈
    # {"recon_complex","phase_unwrapped","raw","depth"}); None if absent.
    get_field: Callable[[str], Any]
    # Feedback.
    set_status: Callable[..., None]        # (text, level="info")
    toast: Callable[..., None]             # (text, level="info")
    # Display: push an array into one of the four image panels
    # (target ∈ {"input","amplitude","phase","spectrum"}).
    show_in_panel: Callable[..., None]     # (target, array, **kw)
    # The active palette (ui3.design.Palette) for panel styling.
    palette: Any = None
