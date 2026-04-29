"""Basler Pylon camera backend stub — v2.1.x sprint, H3.

This module exposes the registry shape (``BACKEND`` info,
``is_available()`` predicate, ``make()`` factory) that
:mod:`core.cameras` expects. The actual SDK calls live in TODOs
that the lab IT team fills in once Pypylon is installed on the
target box.

Why a stub
----------
Pypylon (the official Basler Python wheel) bundles the GenICam
runtime; it can't ship in the test venv without the underlying
.deb / .pkg installation. This stub:

1. Checks at import time whether ``pypylon.pylon`` is reachable.
2. If yes, ``is_available()`` returns True and the dialog
   advertises the backend.
3. ``make()`` constructs a :class:`PylonCamera` whose methods
   call into the SDK at the documented integration points (see
   ``# TODO`` markers).

When the integration lands, replace each TODO body with the real
SDK call. Tests can stay where they are — the mock camera covers
the same Protocol surface.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import CameraBackendInfo


_LOG = logging.getLogger(__name__)


BACKEND = CameraBackendInfo(
    name="pylon",
    display_name="Basler Pylon",
    vendor="Basler",
    summary="Basler ace / dart / boost via the official Pypylon "
            "wheel. Requires Basler runtime installed on host.",
    requires_sdk=("pypylon",),
    capabilities={
        "live": True,
        "trigger": True,
        "16-bit": True,
        "roi": True,
        "exposure": True,
    },
)


def is_available() -> bool:
    """Pypylon detection — import test, no side effects."""
    try:
        import pypylon.pylon  # noqa: F401
        return True
    except Exception:
        return False


class PylonCamera:
    """Skeleton wrapper around a single Basler camera.

    Actual SDK calls are TODO; the structure is correct. Test
    surface is via :class:`core.cameras.mock.MockCamera` until
    real hardware is online.
    """

    def __init__(self, *, serial: Optional[str] = None,
                 exposure_us: float = 5000.0,
                 size_px: int = 1024,
                 target_fps: float = 30.0):
        self.serial = serial
        self.exposure_us = float(exposure_us)
        self.size_px = int(size_px)
        self._target_fps = float(target_fps)
        self._handle = None  # SDK handle, set by start()
        self._size = (self.size_px, self.size_px)

    def start(self) -> None:
        # TODO(integration): call pypylon to open + configure.
        #   import pypylon.pylon as pylon
        #   tlf = pylon.TlFactory.GetInstance()
        #   devs = tlf.EnumerateDevices()
        #   pick = next((d for d in devs if d.GetSerialNumber() == self.serial),
        #               devs[0])
        #   self._handle = pylon.InstantCamera(tlf.CreateDevice(pick))
        #   self._handle.Open()
        #   self._handle.ExposureTime.SetValue(self.exposure_us)
        #   self._handle.PixelFormat.SetValue("Mono16")
        #   self._size = (self._handle.Height.Value, self._handle.Width.Value)
        #   self._handle.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        raise NotImplementedError(
            "Pylon backend is a stub. Install pypylon + uncomment "
            "the integration block in core/cameras/pylon.py.",
        )

    def stop(self) -> None:
        # TODO(integration):
        #   if self._handle is not None:
        #       self._handle.StopGrabbing()
        #       self._handle.Close()
        #       self._handle = None
        self._handle = None

    def grab(self) -> np.ndarray:
        # TODO(integration):
        #   res = self._handle.RetrieveResult(
        #       1000, pylon.TimeoutHandling_ThrowException)
        #   try:
        #       arr = res.Array.astype(np.float32) / 65535.0
        #   finally:
        #       res.Release()
        #   return arr
        raise NotImplementedError(
            "Pylon backend stub: grab() needs SDK integration",
        )

    @property
    def size(self) -> tuple:
        return self._size

    @property
    def fps(self) -> float:
        return self._target_fps


def make(*, serial: Optional[str] = None,
         exposure_us: float = 5000.0,
         size_px: int = 1024,
         target_fps: float = 30.0,
         ) -> PylonCamera:
    return PylonCamera(
        serial=serial,
        exposure_us=exposure_us,
        size_px=size_px,
        target_fps=target_fps,
    )
