"""Thorlabs SciCam camera backend stub — v2.1.x sprint, H3.

SDK: ``thorlabs_tsi_sdk`` (the official Python wrapper around the
TSI native library). Lab Linux box typically has the .so files
under ``/usr/local/lib/`` plus the Python wheel; macOS port is
limited to monochrome cameras.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import CameraBackendInfo


BACKEND = CameraBackendInfo(
    name="thorlabs",
    display_name="Thorlabs SciCam",
    vendor="Thorlabs",
    summary="Thorlabs Scientific Camera (Quantalux / Kiralux / "
            "Zelux). Requires thorlabs_tsi_sdk wheel + native "
            "TSI library on PATH/LD_LIBRARY_PATH.",
    requires_sdk=("thorlabs_tsi_sdk",),
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
        import thorlabs_tsi_sdk.tl_camera  # noqa: F401
        return True
    except Exception:
        return False


class ThorlabsCamera:
    """Skeleton; SDK calls TODO."""

    def __init__(self, *, serial: Optional[str] = None,
                 exposure_us: float = 5000.0,
                 size_px: int = 1024,
                 target_fps: float = 30.0):
        self.serial = serial
        self.exposure_us = float(exposure_us)
        self.size_px = int(size_px)
        self._target_fps = float(target_fps)
        self._sdk = None
        self._handle = None
        self._size = (self.size_px, self.size_px)

    def start(self) -> None:
        # TODO(integration):
        #   from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
        #   self._sdk = TLCameraSDK()
        #   serials = self._sdk.discover_available_cameras()
        #   pick = self.serial or serials[0]
        #   self._handle = self._sdk.open_camera(pick)
        #   self._handle.exposure_time_us = int(self.exposure_us)
        #   self._handle.frames_per_trigger_zero_for_unlimited = 0
        #   self._handle.image_poll_timeout_ms = 1000
        #   self._handle.arm(2)
        #   self._handle.issue_software_trigger()
        raise NotImplementedError(
            "Thorlabs backend is a stub. Install thorlabs_tsi_sdk "
            "and replace the TODO block in this module.",
        )

    def stop(self) -> None:
        # TODO(integration):
        #   if self._handle is not None: self._handle.disarm(); self._handle.dispose()
        #   if self._sdk is not None: self._sdk.dispose()
        self._handle = None
        self._sdk = None

    def grab(self) -> np.ndarray:
        # TODO(integration):
        #   frame = self._handle.get_pending_frame_or_null()
        #   if frame is None: return np.zeros(self._size, dtype=np.float32)
        #   arr = frame.image_buffer.astype(np.float32) / 65535.0
        #   return arr
        raise NotImplementedError(
            "Thorlabs backend stub: grab() needs SDK integration",
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
         ) -> ThorlabsCamera:
    return ThorlabsCamera(
        serial=serial,
        exposure_us=exposure_us,
        size_px=size_px,
        target_fps=target_fps,
    )
