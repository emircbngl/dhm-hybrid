"""Mock camera backend with realistic pathologies — v2.1.x sprint.

The synthetic camera produces clean frames at full FPS. Real
hardware doesn't: occasional dropped frames, exposure jitter,
read noise that varies with gain. The mock backend gives CI a
target that resembles those pathologies so we can test cancel /
recovery / dropped-frame handling without owning a real camera.

Capabilities
------------
* ``drop_rate`` — fraction of frames returned as None (dropped).
* ``exposure_jitter_us`` — gaussian noise added to ``exposure_us``
  per frame; surfaces as brightness drift.
* ``read_noise_sigma`` — additive gaussian on top of the synthetic
  pattern.
* ``warmup_frames`` — the first N grabs return all-zero (camera
  not yet streaming) so cancel-during-warmup gets exercised.

Determinism
-----------
Seeded RNG → identical frame stream for identical seed. Lab can
record an old session against a known seed and replay it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import CameraBackendInfo


BACKEND = CameraBackendInfo(
    name="mock",
    display_name="Mock (realistic synthetic)",
    vendor="DHM Reconstruction",
    summary="Synthetic stream with drop rate, exposure jitter, and "
            "warmup behaviour mimicking real hardware. CI smoke "
            "test target.",
    requires_sdk=(),
    capabilities={
        "live": True,
        "trigger": True,
        "16-bit": False,
        "roi": False,
        "exposure": True,
    },
)


def is_available() -> bool:
    return True


@dataclass
class MockCamera:
    """``CameraSource`` implementation with adjustable
    pathologies."""
    size_px: int = 512
    target_fps: float = 30.0
    rng_seed: int = 0

    drop_rate: float = 0.0
    exposure_us: float = 5000.0
    exposure_jitter_us: float = 0.0
    read_noise_sigma: float = 0.0
    warmup_frames: int = 0

    _started: bool = field(default=False, init=False)
    _frame: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    # ── Protocol surface ──────────────────────────────────────
    def start(self) -> None:
        self._started = True
        self._frame = 0

    def stop(self) -> None:
        self._started = False

    def grab(self) -> np.ndarray:
        if not self._started:
            self._started = True
        self._frame += 1
        n = self.size_px

        if self._frame <= self.warmup_frames:
            # Camera streaming hasn't stabilised — black frame.
            return np.zeros((n, n), dtype=np.float32)

        if self.drop_rate > 0 and self._rng.random() < self.drop_rate:
            # Dropped frame surfaces as zeros so the consumer sees
            # explicit "nothing here" without the type changing.
            return np.zeros((n, n), dtype=np.float32)

        y, x = np.mgrid[:n, :n].astype(np.float32)
        carrier = 0.5 + 0.35 * np.sin(
            2 * np.pi * (0.15 * x + 0.02 * self._frame),
        )
        cx = n / 2 + 30 * np.cos(self._frame * 0.05)
        cy = n / 2 + 30 * np.sin(self._frame * 0.03)
        env = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2)
                       / (0.2 * n) ** 2))

        # Exposure jitter scales global brightness.
        if self.exposure_jitter_us > 0:
            jitter = float(self._rng.normal(
                0.0, self.exposure_jitter_us,
            ))
            scale = max(0.1, 1.0 + jitter / max(1.0, self.exposure_us))
        else:
            scale = 1.0

        frame = scale * carrier * (0.6 + 0.4 * env)
        if self.read_noise_sigma > 0:
            frame = frame + self._rng.normal(
                0.0, self.read_noise_sigma, size=(n, n),
            ).astype(np.float32)
        return np.clip(frame, 0.0, 1.0).astype(np.float32)

    @property
    def size(self) -> tuple:
        return (self.size_px, self.size_px)

    @property
    def fps(self) -> float:
        return float(self.target_fps)


def make(*,
         size_px: int = 512,
         target_fps: float = 30.0,
         rng_seed: int = 0,
         drop_rate: float = 0.0,
         exposure_us: float = 5000.0,
         exposure_jitter_us: float = 0.0,
         read_noise_sigma: float = 0.0,
         warmup_frames: int = 0,
         ) -> MockCamera:
    return MockCamera(
        size_px=size_px,
        target_fps=target_fps,
        rng_seed=rng_seed,
        drop_rate=drop_rate,
        exposure_us=exposure_us,
        exposure_jitter_us=exposure_jitter_us,
        read_noise_sigma=read_noise_sigma,
        warmup_frames=warmup_frames,
    )
