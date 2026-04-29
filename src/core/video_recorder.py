"""MP4 video recorder for live-feed sessions — v2.1.x sprint, H2.

Mirrors :class:`ui2.camera_feed.TiffStackRecorder`'s shape so the
acquisition thread can plug either one in. Uses
``imageio-ffmpeg`` for the actual encoding, which bundles a
static FFmpeg binary so the lab doesn't have to install it.

Why MP4 alongside TIFF
----------------------
TIFF stack is the lab's preferred data format (lossless, opens
in ImageJ). But:

* TIFF stacks are hard to share with collaborators outside the
  imaging community — most reviewers want a video file.
* TIFFs of long sessions get into BigTIFF territory + need a
  separate viewer.
* Time-lapse demos in talks need a quick play-through.

The recorder writes 8-bit MP4 (or H.264 if available) at the
camera's nominal FPS. Lossy by design — the TIFF stack remains
the source of truth for science.

Public API
----------
* :class:`MP4Recorder` — same Protocol surface (``start`` / write
  / ``stop``) as :class:`TiffStackRecorder`.
* :func:`is_available` — predicate the dialog uses to decide
  whether to expose the recording-format dropdown.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np


_LOG = logging.getLogger(__name__)


def is_available() -> bool:
    """``True`` when imageio + imageio_ffmpeg are importable, the
    pre-flight needed before the FFmpeg pipe is opened."""
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
        return True
    except Exception:
        return False


class MP4Recorder:
    """Persist each grabbed frame to an H.264 MP4.

    Frames are normalised to [0, 1] (same convention as
    SyntheticCamera / Mock) and quantised to 8-bit on write so
    the resulting file plays in any browser / media player.

    For science-grade preservation use :class:`TiffStackRecorder`
    instead; this is a presentation artefact.
    """

    def __init__(self, path: Path, *,
                 fps: float = 30.0,
                 codec: str = "libx264",
                 quality: int = 8,
                 ) -> None:
        """Parameters
        ----------
        path
            Destination file. Parent directory is created on
            ``start()``.
        fps
            Playback rate. Should match the camera's nominal FPS
            so live and recording timing line up.
        codec
            FFmpeg codec id. ``libx264`` is the broadest-compat
            default; switch to ``libx265`` for 4K sources.
        quality
            ``imageio_ffmpeg`` quality knob (0 = best, 10 = small).
            8 is a reasonable lab default.
        """
        self.path = Path(path)
        self.fps = float(fps)
        self.codec = codec
        self.quality = int(quality)
        self._writer: Any = None
        self._frames_written = 0

    # ── Protocol surface (matches TiffStackRecorder) ──────────
    def start(self) -> None:
        if not is_available():
            raise RuntimeError(
                "MP4Recorder needs imageio + imageio-ffmpeg "
                "installed (pip install imageio-ffmpeg).",
            )
        import imageio.v2 as imageio
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            str(self.path),
            fps=self.fps,
            codec=self.codec,
            quality=self.quality,
            macro_block_size=1,  # avoid mismatch with arbitrary frame sizes
        )
        self._frames_written = 0

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("MP4Recorder: write_frame before start()")
        f = np.clip(frame, 0.0, 1.0)
        # imageio expects 8-bit RGB (or single-channel grey).
        u8 = (f * 255.0).astype(np.uint8)
        if u8.ndim == 2:
            u8 = np.stack([u8, u8, u8], axis=-1)
        self._writer.append_data(u8)
        self._frames_written += 1

    def stop(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None

    @property
    def frames_written(self) -> int:
        return self._frames_written


__all__ = [
    "MP4Recorder",
    "is_available",
]
