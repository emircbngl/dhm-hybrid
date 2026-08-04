"""Relocation shim — the driver moved to ``core.drivers.reconstruction``
(2026-07-06; see ``core/drivers/__init__.py`` for the rationale).

``sys.modules`` aliasing makes the old and new import paths the SAME
module object, so ``patch("ui2.reconstruction.X")``, private-name
imports, and class identity all keep working unchanged. (The import
machinery re-reads ``sys.modules[__name__]`` after executing this file
and binds the parent-package attribute to the aliased target.)
"""
import sys as _sys

import core.drivers.reconstruction as _target

_sys.modules[__name__] = _target
