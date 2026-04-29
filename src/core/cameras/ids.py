"""IDS uEye camera backend stub — v2.1.x sprint, H3.

Same shape as the Pylon stub. SDK is ``pyueye`` (IDS official) or
``ids_peak`` (newer line). We probe both because the lab might
have either installed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import CameraBackendInfo


BACKEND = CameraBackendInfo(
    name="ids",
    display_name="IDS uEye / Peak",
    vendor="IDS Imaging",
    summary="IDS uEye legacy or IDS peak SDK. Probes pyueye and "
            "ids_peak; either satisfies the requires_sdk gate.",
    requires_sdk=("pyueye", "ids_peak"),  # either one
    capabilities={
        "live": True,
        "trigger": True,
        "16-bit": True,
        "roi": True,
        "exposure": True,
    },
)


def is_available() -> bool:
    try:
        import pyueye  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import ids_peak  # noqa: F401
        return True
    except Exception:
        return False


class IDSCamera:
    """Skeleton; SDK calls TODO."""

    def __init__(self, *, device_id: Optional[int] = None,
                 exposure_us: float = 5000.0,
                 size_px: int = 1024,
                 target_fps: float = 30.0):
        self.device_id = device_id
        self.exposure_us = float(exposure_us)
        self.size_px = int(size_px)
        self._target_fps = float(target_fps)
        self._handle = None
        self._size = (self.size_px, self.size_px)

    def start(self) -> None:
        # TODO(integration, pyueye path):
        #   from pyueye import ueye
        #   self._handle = ueye.HIDS(self.device_id or 0)
        #   nRet = ueye.is_InitCamera(self._handle, None)
        #   ... configure exposure + pixel format + ROI ...
        # TODO(integration, ids_peak path):
        #   import ids_peak.ids_peak as peak
        #   peak.Library.Initialize()
        #   manager = peak.DeviceManager.Instance()
        #   manager.Update()
        #   pick = manager.Devices()[self.device_id or 0]
        #   self._handle = pick.OpenDevice(peak.DeviceAccessType_Control)
        raise NotImplementedError(
            "IDS backend is a stub. Pick pyueye or ids_peak path "
            "and fill in the corresponding TODO block.",
        )

    def stop(self) -> None:
        # TODO(integration): release handle + library.
        self._handle = None

    def grab(self) -> np.ndarray:
        # TODO(integration): grab buffer + convert to float32 [0,1].
        raise NotImplementedError(
            "IDS backend stub: grab() needs SDK integration",
        )

    @property
    def size(self) -> tuple:
        return self._size

    @property
    def fps(self) -> float:
        return self._target_fps


def make(*, device_id: Optional[int] = None,
         exposure_us: float = 5000.0,
         size_px: int = 1024,
         target_fps: float = 30.0,
         ) -> IDSCamera:
    return IDSCamera(
        device_id=device_id,
        exposure_us=exposure_us,
        size_px=size_px,
        target_fps=target_fps,
    )
