"""ui2 — RETIRED Dear PyGui frontend (2026-04-22 → 2026-07-06).

The DPG presentation layer was removed on 2026-07-06 after the PySide6
``ui3`` rebuild reached feature parity (see docs/UI3_DESIGN.md coverage
matrix and the wiki page "DHM ui3 — Qt Rebuild"). What remains here:

* ``reconstruction`` / ``workers`` / ``camera_feed`` — sys.modules
  aliasing shims to ``core.drivers.*`` (the compute layer relocated
  there earlier the same day); kept so historical imports and test
  ``patch("ui2.workers.X")`` targets stay valid.
* ``state_store`` — the persisted-settings store + frozen schema
  migrations (v1..v10); GUI-free and still the single source of truth
  for on-disk state compatibility.

The full DPG implementation is recoverable from git history
(last complete state: the commit preceding the 2026-07-06 retirement).
"""
from __future__ import annotations

__all__: list[str] = []
