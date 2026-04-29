"""Synthetic camera backend — deterministic fringe pattern, no
hardware. Re-exports the v2 UI's existing ``SyntheticCamera`` so
both the dialog dropdown and ``make_camera('synthetic')`` end up
on the same source.
"""
from __future__ import annotations

from . import CameraBackendInfo


BACKEND = CameraBackendInfo(
    name="synthetic",
    display_name="Synthetic (no hardware)",
    vendor="DHM Reconstruction",
    summary="Deterministic interference-fringe generator. "
            "Use for development + CI when no camera is attached.",
    requires_sdk=(),
    capabilities={
        "live": True,
        "trigger": False,
        "16-bit": False,
        "roi": False,
        "exposure": False,
    },
)


def is_available() -> bool:
    return True


def make(*, size_px: int = 512, target_fps: float = 30.0,
         rng_seed: int = 42):
    from ui2.camera_feed import SyntheticCamera
    return SyntheticCamera(
        size_px=size_px,
        target_fps=target_fps,
        rng_seed=rng_seed,
    )
