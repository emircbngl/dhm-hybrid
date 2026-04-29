"""Multi-line profile sampling — v2.0.8 sprint, D4.

v2.0.5 shipped a single-line profile (centre row only). This is
the multi-line follow-up the v2.0.7+ backlog called for: any
number of lines, each one click-drag defined, sampled with sub-
pixel bilinear interpolation, ready for an overlay plot dialog.

Lab use case
------------
Karin compares phase profiles across the cell membrane vs through
the cytoplasm vs along the long axis — three lines on the same
frame. Currently she opens the centre-row profile, eyeballs it,
moves on. Multi-line lets her drop three named samples in one
view and plot them together for a paper figure.

Public API
----------
* :class:`LineProfile` — frozen dataclass: endpoints, label, colour.
* :func:`sample_line` — bilinear sample N points along a line.
* :func:`sample_profiles` — apply :func:`sample_line` to a list of
  :class:`LineProfile` and return parallel arrays.

The DPG dialog wrapper (``ui2.dialogs.show_line_profiles``, future
work) will own click-drag input + colour picker; this module is the
pure-data sampler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineProfile:
    """One straight-line sample on a 2-D image.

    Attributes
    ----------
    y0, x0
        Start point in pixel coordinates.
    y1, x1
        End point in pixel coordinates.
    label
        Free-form identifier the lab uses (e.g. "membrane",
        "long axis"). Empty string OK.
    colour_rgb
        RGB tuple in [0, 1] for the dialog overlay. Defaults to
        a ScientificTheme accent — caller can override.
    n_samples
        Number of samples to take along the line. ``None`` →
        sampler picks ``round(line_length_px) + 1`` so each
        sample covers ~1 pixel.
    """
    y0: float
    x0: float
    y1: float
    x1: float
    label: str = ""
    colour_rgb: Tuple[float, float, float] = (0.27, 0.55, 0.95)
    n_samples: int = 0  # 0 = auto

    @property
    def length_px(self) -> float:
        return float(np.hypot(self.y1 - self.y0, self.x1 - self.x0))


@dataclass
class SampledProfile:
    """Output of :func:`sample_line`.

    Attributes
    ----------
    profile
        The :class:`LineProfile` source.
    distance_px
        Per-sample distance along the line, starting at 0.
    values
        Sampled image values. ``np.nan`` for out-of-bounds samples
        (caller can mask cleanly when plotting).
    """
    profile: LineProfile
    distance_px: np.ndarray
    values: np.ndarray


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _bilinear(image: np.ndarray, ys: np.ndarray, xs: np.ndarray,
              *, fill_value: float = float("nan")) -> np.ndarray:
    """Sub-pixel bilinear sample. Out-of-bounds → ``fill_value``.

    Vectorised over ``ys``/``xs`` (1-D arrays of equal length).
    """
    ny, nx = image.shape[:2]
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    # Out-of-bounds mask — we still index but overwrite at the end.
    in_bounds = (y0 >= 0) & (x0 >= 0) & (y1 < ny) & (x1 < nx)
    # Clamp indices for indexing safety; values get nuked below.
    y0c = np.clip(y0, 0, ny - 1)
    x0c = np.clip(x0, 0, nx - 1)
    y1c = np.clip(y1, 0, ny - 1)
    x1c = np.clip(x1, 0, nx - 1)
    fy = ys - y0
    fx = xs - x0
    Ia = image[y0c, x0c].astype(np.float64)
    Ib = image[y0c, x1c].astype(np.float64)
    Ic = image[y1c, x0c].astype(np.float64)
    Id = image[y1c, x1c].astype(np.float64)
    out = (Ia * (1 - fy) * (1 - fx)
           + Ib * (1 - fy) * fx
           + Ic * fy * (1 - fx)
           + Id * fy * fx)
    out = np.where(in_bounds, out, fill_value)
    return out


def sample_line(image: np.ndarray,
                profile: LineProfile,
                *,
                fill_value: float = float("nan"),
                ) -> SampledProfile:
    """Sample ``image`` along ``profile`` using bilinear interpolation.

    Parameters
    ----------
    image
        2-D real-valued array.
    profile
        :class:`LineProfile` with endpoints (and optional length).
    fill_value
        Value for samples falling outside the image rectangle.
        Default NaN — pandas / matplotlib treat NaN as "missing"
        and don't draw through it.

    Returns
    -------
    SampledProfile
        Distances + values arrays. Length = ``profile.n_samples``
        (or auto-resolved when ``n_samples=0``).
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D; got {image.shape}")
    n = profile.n_samples
    if n <= 0:
        # Auto: ~1 sample per pixel along the line. Minimum 2 so
        # the output has a well-defined start/end.
        n = max(2, int(round(profile.length_px)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    ys = profile.y0 + (profile.y1 - profile.y0) * ts
    xs = profile.x0 + (profile.x1 - profile.x0) * ts
    distances = ts * profile.length_px
    values = _bilinear(image, ys, xs, fill_value=fill_value)
    return SampledProfile(
        profile=profile,
        distance_px=distances,
        values=values,
    )


def sample_profiles(image: np.ndarray,
                    profiles: Sequence[LineProfile],
                    *,
                    fill_value: float = float("nan"),
                    ) -> List[SampledProfile]:
    """Apply :func:`sample_line` to every line. Order preserved
    so the dialog can map sample-N → profile-N → colour-N."""
    return [sample_line(image, p, fill_value=fill_value)
            for p in profiles]


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def stats_for(sampled: SampledProfile) -> dict:
    """Per-profile summary the dialog can show alongside the plot.
    Skips NaN cleanly so out-of-bounds samples don't poison."""
    v = sampled.values
    if v.size == 0 or np.all(np.isnan(v)):
        return {"min": None, "max": None, "mean": None,
                "std": None, "n": 0}
    return {
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
        "n": int(np.count_nonzero(~np.isnan(v))),
    }


__all__ = [
    "LineProfile",
    "SampledProfile",
    "sample_line",
    "sample_profiles",
    "stats_for",
]
