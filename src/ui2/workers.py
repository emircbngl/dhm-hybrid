"""Relocation shim — the driver moved to ``core.drivers.workers``
(2026-07-06; see ``core/drivers/__init__.py`` for the rationale).

``sys.modules`` aliasing makes the old and new import paths the SAME
module object, so ``patch("ui2.workers.X")``, private-name imports
(``_AUTOFOCUS_ALGORITHMS``, ``_prepare_field``, ...), and class
identity all keep working unchanged.
"""
import sys as _sys

import core.drivers.workers as _target

_sys.modules[__name__] = _target
