"""Relocation shim — the feed moved to ``core.drivers.camera_feed``
(2026-07-06; see ``core/drivers/__init__.py`` for the rationale —
this also un-inverts core.cameras.synthetic importing up into ui2).

``sys.modules`` aliasing makes the old and new import paths the SAME
module object, so patches and class identity keep working unchanged.
"""
import sys as _sys

import core.drivers.camera_feed as _target

_sys.modules[__name__] = _target
