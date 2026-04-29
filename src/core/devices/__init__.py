"""Lab device control — v2.1.z sprint.

Companion to :mod:`core.cameras`: a registry + Protocol for the
**non-camera** instruments that make a microscopy session run.
Common types in a DHM lab:

* **Motorised stage** — XYZ positioning + homing.
* **Shutter** — block / pass illumination (long acquisitions need
  shutter-sync to avoid photobleach).
* **LED controller** — set exposure intensity.
* **Generic serial device** — anything else over RS-232 / USB-CDC.

The lab's existing in-house tool (separate codebase) handles some
of this already; this module is the in-tree counterpart so:

1. Time-lapse runs can drive a stage between frames (multi-position
   imaging out of one DHM Reconstruction session).
2. Camera + shutter + LED can be coordinated as one "expose"
   action, not three independent button clicks.
3. Mock backends keep CI green without lab hardware.

API mirrors :mod:`core.cameras` deliberately:

    from core.devices import available_backends, make_device
    print([b.name for b in available_backends()])
    stage = make_device("mock_stage")
    stage.connect()
    stage.move_to(x_um=100, y_um=200, z_um=0)

Device backends are sub-modules; ``BACKEND = DeviceBackendInfo(...)``
+ ``is_available()`` + ``make()`` factory makes them discoverable
by the registry. Same shape as :mod:`core.cameras` so adding new
hardware is one file.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


_LOG = logging.getLogger(__name__)


class DeviceKind(str, Enum):
    """Coarse classification — drives the dialog grouping. Add a
    member when introducing a fundamentally new device class
    (e.g. PIEZO, ROTATOR)."""
    STAGE = "stage"
    SHUTTER = "shutter"
    LED = "led"
    GENERIC = "generic"


@dataclass(frozen=True)
class DeviceBackendInfo:
    """Backend metadata advertised to the dialog.

    Attributes
    ----------
    name
        Short identifier (``mock_stage``, ``serial_shutter``,
        vendor-specific names like ``thorlabs_kdc101``).
    display_name
        Human label.
    kind
        :class:`DeviceKind` — drives grouping in the device panel.
    summary
        One-line description for the dropdown subtitle.
    requires_sdk
        Tuple of pip packages required (e.g. ``("pyserial",)``).
        Empty = no external dep (mock / synthetic).
    capabilities
        Dict of per-feature flags. Example for a stage:
        ``{'xy': True, 'z': True, 'home': True,
        'limits_um': (-25000, 25000)}``. UI greys out controls
        that wouldn't apply.
    """
    name: str
    display_name: str
    kind: DeviceKind
    summary: str
    requires_sdk: tuple = ()
    capabilities: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_backend_modules() -> List[str]:
    """Sub-module names; same logic as :mod:`core.cameras`."""
    out: List[str] = []
    package = importlib.import_module(__name__)
    for finder, name, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg or name.startswith("_"):
            continue
        out.append(name)
    return sorted(out)


def _load_backend(name: str):
    return importlib.import_module(f"{__name__}.{name}")


def all_backends() -> List[DeviceBackendInfo]:
    """Every registered backend, regardless of SDK availability."""
    out: List[DeviceBackendInfo] = []
    for mod_name in _discover_backend_modules():
        try:
            mod = _load_backend(mod_name)
            info = getattr(mod, "BACKEND", None)
            if isinstance(info, DeviceBackendInfo):
                out.append(info)
        except Exception as exc:  # pragma: no cover
            _LOG.warning("device backend %s: import failed: %s",
                         mod_name, exc)
    return out


def available_backends() -> List[DeviceBackendInfo]:
    """Subset whose SDK is actually importable on this host."""
    out: List[DeviceBackendInfo] = []
    for mod_name in _discover_backend_modules():
        try:
            mod = _load_backend(mod_name)
        except Exception:
            continue
        info = getattr(mod, "BACKEND", None)
        if not isinstance(info, DeviceBackendInfo):
            continue
        is_avail = getattr(mod, "is_available", lambda: True)
        try:
            ok = bool(is_avail())
        except Exception:
            ok = False
        if ok:
            out.append(info)
    return out


def backends_by_kind(kind: DeviceKind) -> List[DeviceBackendInfo]:
    """Filter helper for the device panel — group rows by stage /
    shutter / LED / generic so the operator scans them quickly."""
    return [b for b in all_backends() if b.kind is kind]


def make_device(name: str, **kwargs: Any) -> Any:
    """Construct a device by registry name.

    ``ValueError`` if the name isn't registered, ``RuntimeError``
    if the SDK is missing — matching the camera registry's
    error contract.
    """
    info_map = {b.name: b for b in all_backends()}
    if name not in info_map:
        raise ValueError(
            f"unknown device backend {name!r}; "
            f"known: {sorted(info_map.keys())}",
        )
    mod = _load_backend(name)
    is_avail = getattr(mod, "is_available", lambda: True)
    if not is_avail():
        info = info_map[name]
        sdk = ", ".join(info.requires_sdk) if info.requires_sdk else "?"
        raise RuntimeError(
            f"device backend {name!r} requires SDK ({sdk}) — "
            f"install vendor wheels and retry."
        )
    factory = getattr(mod, "make", None)
    if factory is None:
        raise RuntimeError(
            f"device backend {name!r} has no make() factory.",
        )
    return factory(**kwargs)


__all__ = [
    "DeviceKind",
    "DeviceBackendInfo",
    "all_backends",
    "available_backends",
    "backends_by_kind",
    "make_device",
]
