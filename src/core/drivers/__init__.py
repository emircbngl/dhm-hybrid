"""Framework-free compute drivers (moved from ``src/ui2/`` 2026-07-06).

These modules were born inside the Dear PyGui frontend package but were
always GUI-free (ThreadPoolExecutor + plain callbacks) and carry the
hard-won pipeline correctness: subtract_mean parity, reference-mode
gating, reference-free QPI correction, reflection halving, per-z
reference division, the six autofocus algorithms + their dispatch/audit/
cancel semantics, and the camera feed protocol. ui3 (PySide6), the
MCP/headless session, and ui2 itself all drive the same physics through
them — so they live in ``core`` where the layering says they belong
(this also fixes core.cameras.synthetic importing UP into ui2).

Backward compatibility: ``ui2.reconstruction``, ``ui2.workers`` and
``ui2.camera_feed`` remain importable — each is a shim that aliases
itself to the module here via ``sys.modules``, so the OLD and NEW paths
are literally the same module object. Tests that
``patch("ui2.workers.X")`` therefore keep patching the real globals.

LAZY on purpose (PEP 562, same pattern as ``ui2/__init__``): eagerly
importing ``.workers`` here dragged matplotlib + skimage + the whole
recon/qpi/depth/report stack into every ``core.drivers.*`` submodule
import — ``import core.drivers.camera_feed`` (numpy + stdlib on its own)
went to ~0.7 s and newly required matplotlib (2026-07-06 relocation
review, B-097). Submodules stay independent; the convenience re-exports
below resolve only when actually referenced.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — hints only, no runtime import
    from .reconstruction import (  # noqa: F401
        ReconError,
        ReconParams,
        ReconResult,
        ReconstructionDriver,
    )
    from .workers import (  # noqa: F401
        ScienceDriver,
        af_algorithm_input_profile,
        available_autofocus_algorithms,
    )

_EXPORTS = {
    "ReconError": "reconstruction",
    "ReconParams": "reconstruction",
    "ReconResult": "reconstruction",
    "ReconstructionDriver": "reconstruction",
    "ScienceDriver": "workers",
    "af_algorithm_input_profile": "workers",
    "available_autofocus_algorithms": "workers",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve convenience re-exports on first use only."""
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module 'core.drivers' has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value  # cache — subsequent access skips __getattr__
    return value
