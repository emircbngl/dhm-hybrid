"""Camera backend registry — v2.1.x sprint, H1.

Vendor SDKs (Basler Pylon, IDS uEye, Thorlabs SciCam) ship their
own pip wheels that aren't installed by default and won't be
present on a dev laptop. This package gives the lab a single
discovery API:

    from core.cameras import available_backends, make_camera
    print(available_backends())   # → ['synthetic', 'mock']
    cam = make_camera('synthetic', size_px=512)

Each vendor backend is a separate sub-module that lazy-imports its
SDK. ``available_backends()`` returns only the names whose SDK is
actually importable, so the dialog dropdown reflects reality
instead of advertising hardware that won't work.

Integration shape
-----------------
A new vendor (NextLab Imaging or whatever) becomes available by
adding ``cameras/<vendor>.py`` with:

* a class implementing :class:`ui2.camera_feed.CameraSource`
* a module-level ``BACKEND = CameraBackendInfo(...)`` describing
  the capability set
* a module-level ``is_available() -> bool`` predicate

The registry below picks them up automatically — no central list
to forget to update.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraBackendInfo:
    """What the dialog needs to know about a backend.

    Attributes
    ----------
    name
        Short identifier the user picks (e.g. ``'pylon'``).
    display_name
        Human label (e.g. ``'Basler Pylon'``).
    vendor
        Free-form vendor name (Basler, IDS, Thorlabs, …).
    summary
        One-line description for the dropdown subtitle.
    requires_sdk
        Tuple of pip package names whose installation is required.
        Empty tuple = no SDK (synthetic / mock).
    capabilities
        Dict of feature flags — ``{'live': True, 'trigger': True,
        '16-bit': True, 'roi': False}``. UI uses these to grey out
        controls that wouldn't apply.
    """
    name: str
    display_name: str
    vendor: str
    summary: str
    requires_sdk: tuple = ()
    capabilities: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        # dataclass(frozen=True) doesn't allow setattr; build the
        # default dict via object.__setattr__ when caller leaves
        # capabilities=None.
        if self.capabilities is None:
            object.__setattr__(self, "capabilities", {})


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_backend_modules() -> List[str]:
    """Return import names of every sub-module in this package
    (synthetic, mock, pylon, ids, thorlabs, …) — order matters
    only for stability, not correctness."""
    out: List[str] = []
    package = importlib.import_module(__name__)
    for finder, name, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg or name.startswith("_"):
            continue
        out.append(name)
    return sorted(out)


def _load_backend(name: str):
    """Import + return a backend module. Re-raises on failure so
    the registry can mark it unavailable."""
    return importlib.import_module(f"{__name__}.{name}")


def all_backends() -> List[CameraBackendInfo]:
    """Every registered backend, regardless of SDK availability.

    Useful for showing the lab "we *could* support Pylon if you
    install pypylon" rather than silently hiding it. The dialog
    can still grey-out unavailable rows.
    """
    out: List[CameraBackendInfo] = []
    for mod_name in _discover_backend_modules():
        try:
            mod = _load_backend(mod_name)
            info = getattr(mod, "BACKEND", None)
            if isinstance(info, CameraBackendInfo):
                out.append(info)
        except Exception as exc:  # pragma: no cover - import errors
            _LOG.warning("camera backend %s: import failed: %s",
                         mod_name, exc)
    return out


def available_backends() -> List[CameraBackendInfo]:
    """Subset of :func:`all_backends` whose SDK is actually
    importable on this machine. Drives the dropdown population."""
    out: List[CameraBackendInfo] = []
    for mod_name in _discover_backend_modules():
        try:
            mod = _load_backend(mod_name)
        except Exception:
            continue
        info = getattr(mod, "BACKEND", None)
        if not isinstance(info, CameraBackendInfo):
            continue
        is_avail = getattr(mod, "is_available", lambda: True)
        try:
            ok = bool(is_avail())
        except Exception:
            ok = False
        if ok:
            out.append(info)
    return out


def make_camera(name: str, **kwargs: Any) -> Any:
    """Construct a camera by registry name.

    Raises ``ValueError`` if the name isn't registered, or
    ``RuntimeError`` if the SDK isn't installed. Caller catches
    the latter to fall back on synthetic / mock for CI etc.
    """
    info_map = {b.name: b for b in all_backends()}
    if name not in info_map:
        raise ValueError(
            f"unknown camera backend {name!r}; "
            f"known: {sorted(info_map.keys())}",
        )
    mod = _load_backend(name)
    is_avail = getattr(mod, "is_available", lambda: True)
    if not is_avail():
        info = info_map[name]
        sdk = ", ".join(info.requires_sdk) if info.requires_sdk else "?"
        raise RuntimeError(
            f"camera backend {name!r} requires SDK ({sdk}) — "
            f"install vendor wheels and retry."
        )
    factory = getattr(mod, "make", None)
    if factory is None:
        raise RuntimeError(
            f"camera backend {name!r} module has no make() factory.",
        )
    return factory(**kwargs)


__all__ = [
    "CameraBackendInfo",
    "all_backends",
    "available_backends",
    "make_camera",
]
