"""Tomographic depth map — per-pixel best-focus z.

This is the scientific step beyond :func:`core.autofocus.find_focus_candidates`.
Instead of reducing the focus landscape to a handful of candidate planes
for the *whole* frame, we ask the question locally: *for this pixel,
which z plane contains its in-focus content?*  The answer is a 2D array
of z values — a depth map.

Algorithm
---------
1. Propagate the demodulated field to ``n_steps`` equally-spaced z
   planes in ``[z_min, z_max]``. Same ASM kernel the rest of the app
   uses.
2. At each plane compute a **local sharpness** metric — variance of
   the Laplacian inside a ``window_size`` × ``window_size`` patch
   around each pixel. This is the classic Tenengrad-style focus
   measure; variance of Laplacian is particularly good at picking up
   phase-edge transitions (ringing away from focus, sharp near focus).
3. Stack the metric maps into a ``(N_z, H, W)`` volume.
4. For each pixel take ``argmax`` along the z axis → ``z_map``.
5. Take ``max`` along the z axis → ``confidence`` map. Low confidence
   marks pixels where the scan never produced a clear peak (background,
   out-of-range objects).

Trade-offs
----------
* ``window_size=5`` balances locality vs noise. 3×3 is too noisy on
  most phase-object holograms; 9×9 blurs the depth boundaries.
* ``metric=LAPLACIAN_VARIANCE`` responds well to phase edges (what a
  sphere's refocused silhouette looks like). ``TENENGRAD`` works on
  amplitude objects; ``ENTROPY`` is a global metric that doesn't
  decompose cleanly into a local form.
* Runtime scales as ``n_steps × H × W × window_size²``. At 50×256×256
  with a 5×5 window on a modern CPU the whole calculation runs in
  ~0.5 s; at 1024×1024 consider downsampling first.

Out of scope (deferred to v1.2-beta):
* UI overlay of the depth map on the phase panel.
* Confidence-mask integration with QPI (per-cluster depth averaging).
* Multi-object Z-resolved QPI batch.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import uniform_filter

from .autofocus import FocusMetric
from .fft_backend import get_best_fft_backend
from .reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
    safe_reference_divide,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthMapResult:
    """Output of :func:`compute_depth_map`.

    Attributes
    ----------
    z_map
        ``(H, W)`` array: for each pixel, the z (metres) at which the
        local sharpness metric was maximised.
    confidence
        ``(H, W)`` array: the peak sharpness value itself. Useful for
        masking — pixels with low confidence probably sit over
        background or the landscape was ambiguous.
    z_values_m
        1-D array of the z values that were actually scanned. Same
        length as the axis the arg-max was taken over.
    metric
        Which focus metric produced the map.
    window_size
        Lateral window size used for the local sharpness estimate.
    """
    z_map: np.ndarray
    confidence: np.ndarray
    z_values_m: np.ndarray
    metric: FocusMetric
    window_size: int


# ---------------------------------------------------------------------------
# Local sharpness kernels
# ---------------------------------------------------------------------------

def _laplacian(field_complex: np.ndarray) -> np.ndarray:
    """5-point discrete Laplacian of the amplitude ``|field|``.

    The phase's Laplacian is wrap-hazardous (jumps at ±π) so we use
    amplitude — for phase objects under off-axis DHM the amplitude
    sidelobes carry the sharp transitions we want to detect.
    """
    amp = np.abs(field_complex).astype(np.float32, copy=False)
    # ∇²f ≈ f[i,j-1] + f[i,j+1] + f[i-1,j] + f[i+1,j] − 4 f[i,j]
    lap = np.zeros_like(amp)
    lap[1:-1, 1:-1] = (
        amp[1:-1, :-2] + amp[1:-1, 2:]
        + amp[:-2, 1:-1] + amp[2:, 1:-1]
        - 4.0 * amp[1:-1, 1:-1]
    )
    return lap


def _local_laplacian_variance(
    field_complex: np.ndarray, window_size: int,
) -> np.ndarray:
    """Variance of the amplitude Laplacian over an ``N×N`` sliding window.

    ``Var = E[L²] − (E[L])²``, both E[.]s done with
    :func:`scipy.ndimage.uniform_filter` (box mean, separable —
    inexpensive compared to the reconstruction itself).
    """
    lap = _laplacian(field_complex)
    mean_lap = uniform_filter(lap, size=window_size, mode="reflect")
    mean_lap_sq = uniform_filter(lap * lap, size=window_size, mode="reflect")
    var = mean_lap_sq - mean_lap * mean_lap
    # Floating-point noise can make tiny negative numbers — clamp.
    np.maximum(var, 0.0, out=var)
    return var


def _local_tenengrad(
    field_complex: np.ndarray, window_size: int,
) -> np.ndarray:
    """Sum of squared gradients over an ``N×N`` sliding window."""
    amp = np.abs(field_complex).astype(np.float32, copy=False)
    gy, gx = np.gradient(amp)
    g2 = gy * gy + gx * gx
    return uniform_filter(g2, size=window_size, mode="reflect")


def _local_phase_variance(
    field_complex: np.ndarray, window_size: int,
) -> np.ndarray:
    """Local variance of the complex field's real + imaginary parts.

    Amplitude-only kernels (``_local_laplacian_variance``,
    ``_local_tenengrad``) miss *transparent phase objects* — at
    focus the amplitude is essentially uniform, and the Fresnel
    rings that light up these kernels actually appear as the sample
    *defocuses*. Argmax along z then returns the wrong plane.

    This kernel operates on ``Re(field)`` and ``Im(field)`` directly
    — wrap-safe replacements for phase gradients. A phase-only
    sphere at focus looks like a sharp-edged disk in ``Im(field)``
    (when amplitude ≈ 1 and ``φ`` is small, ``Im ≈ Aφ``), so its
    local variance is *maximised* at the true z plane. Works
    gracefully for bio cells too because amplitude variations just
    add to the same variance sum.
    """
    re = field_complex.real.astype(np.float32, copy=False)
    im = field_complex.imag.astype(np.float32, copy=False)
    mean_re = uniform_filter(re, size=window_size, mode="reflect")
    mean_im = uniform_filter(im, size=window_size, mode="reflect")
    mean_re2 = uniform_filter(re * re, size=window_size, mode="reflect")
    mean_im2 = uniform_filter(im * im, size=window_size, mode="reflect")
    var = (mean_re2 - mean_re * mean_re) + (mean_im2 - mean_im * mean_im)
    np.maximum(var, 0.0, out=var)
    return var


_LOCAL_KERNELS = {
    FocusMetric.LAPLACIAN_VARIANCE: _local_laplacian_variance,
    FocusMetric.TENENGRAD: _local_tenengrad,
    # ``PHASE_VARIANCE`` here = local complex-field variance (not the
    # global ``phase_variance`` used by autofocus). It's the right
    # pick for transparent-phase samples where amplitude sits flat
    # at focus and lights up at defocus — the amplitude kernels
    # above return the wrong z for those scenes.
    FocusMetric.PHASE_VARIANCE: _local_phase_variance,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_depth_map(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    z_min_m: float,
    z_max_m: float,
    *,
    n_steps: int = 40,
    window_size: int = 5,
    metric: FocusMetric = FocusMetric.LAPLACIAN_VARIANCE,
    cancel_check: Optional[Callable[[], bool]] = None,
    ref_field: Optional[np.ndarray] = None,
) -> DepthMapResult:
    """Build a per-pixel best-focus z map.

    Parameters
    ----------
    field
        Demodulated complex field at the hologram plane, same shape the
        reconstruction expects.
    base_params
        Carries wavelength / pixel size / medium index. ``z_m`` is
        ignored (overridden per scan step).
    method
        ``ASM`` or ``FRESNEL`` — whichever the app is configured for.
    z_min_m, z_max_m
        Scan range (metres). Must be strictly increasing.
    n_steps
        Number of z planes to scan. Higher resolves depth better; cost
        is linear.
    window_size
        Side length of the local sharpness window. Must be ≥ 3 and
        odd-ish (even values work but bias the window centre).
    metric
        One of :data:`_LOCAL_KERNELS`' keys. ``LAPLACIAN_VARIANCE`` is
        the default — handles phase-edge objects. ``TENENGRAD`` is a
        gradient alternative.
    ref_field
        Optional pre-propagation reference (same +1-order Fourier
        complex field as the sample). When supplied, both sample and
        reference are propagated to each trial z and complex-divided
        before the local sharpness kernel runs — the metric judges the
        same referenced field the user gets back from the live
        reconstruct path. Without this the depth map is reference-blind
        and best-z saturates to the scan boundaries on real lab data
        (illumination drift dominates the local Laplacian).

    Returns
    -------
    :class:`DepthMapResult`
    """
    if z_max_m <= z_min_m:
        raise ValueError(f"z_max_m ({z_max_m}) must exceed z_min_m ({z_min_m})")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2 (got {n_steps})")
    if window_size < 3:
        raise ValueError(f"window_size must be >= 3 (got {window_size})")

    try:
        local_kernel = _LOCAL_KERNELS[metric]
    except KeyError as exc:
        raise ValueError(
            f"metric {metric!r} has no local form — supported: "
            f"{sorted(k.name for k in _LOCAL_KERNELS)}"
        ) from exc

    fft = get_best_fft_backend()
    z_values = np.linspace(z_min_m, z_max_m, n_steps)

    h, w = field.shape[-2:]
    # Stack ``(N_z, H, W)`` of local sharpness maps.
    stack = np.empty((n_steps, h, w), dtype=np.float32)

    use_ref = ref_field is not None
    if use_ref and ref_field.shape[-2:] != (h, w):
        raise ValueError(
            f"ref_field shape {ref_field.shape} doesn't match field {field.shape}"
        )

    for i, z in enumerate(z_values):
        # Cooperative cancellation — depth scans are the longest jobs
        # in the pipeline (O(n_steps * H * W)), so Esc must actually
        # stop them rather than just hide the spinner.
        if cancel_check and cancel_check():
            from core.autofocus.evaluator import AutofocusCancelled
            raise AutofocusCancelled()
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=float(z),
            n=base_params.n,
        )
        recon = propagate(field, params, method, fft=fft, force_python=True)
        if use_ref:
            ref_recon = propagate(ref_field, params, method, fft=fft,
                                  force_python=True)
            recon = safe_reference_divide(recon, ref_recon)
        stack[i] = local_kernel(recon, window_size)

    # Per-pixel arg-max + max.
    idx = np.argmax(stack, axis=0)
    confidence = np.take_along_axis(stack, idx[None, :, :], axis=0)[0]
    z_map = _refine_subpixel_z(stack, idx, z_values)

    # Boundary saturation diagnostic. When the scan range is too narrow,
    # too wide, or the field carries no usable focus signal (flat metric
    # → noise argmax), most pixels pin to the scan extremes. We log a
    # warning so the operator notices instead of trusting a pinned map.
    boundary_frac = float(((idx == 0) | (idx == n_steps - 1)).mean())
    if boundary_frac > 0.5:
        _LOG.warning(
            "depth-map: %.0f%% of pixels saturated to scan boundary "
            "(z_min=%.3f, z_max=%.3f mm). Likely causes: scan range "
            "doesn't cover the focus plane, no reference division on a "
            "ref-required setup, or the metric is flat on this field.",
            boundary_frac * 100,
            z_min_m * 1e3, z_max_m * 1e3,
        )

    _LOG.info(
        "depth-map: %d×%d → z in [%.3f, %.3f] mm, mean conf %.3g, "
        "boundary %.0f%%",
        h, w, float(z_map.min()) * 1e3, float(z_map.max()) * 1e3,
        float(confidence.mean()), boundary_frac * 100,
    )
    return DepthMapResult(
        z_map=z_map.astype(np.float32),
        confidence=confidence.astype(np.float32),
        z_values_m=z_values.astype(np.float32),
        metric=metric,
        window_size=int(window_size),
    )


def _refine_subpixel_z(
    stack: np.ndarray,
    idx: np.ndarray,
    z_values: np.ndarray,
) -> np.ndarray:
    """Parabolic 3-point interpolation around each pixel's argmax
    step for **sub-step z resolution**.

    Vanilla ``z_values[argmax]`` has a hard floor at the grid step
    (~50 µm with 40 steps over a 2 mm scan) — not good enough for
    measuring sphere depths. Fitting a parabola through the argmax
    sample and its two neighbours and returning the analytic
    vertex extends resolution by an order of magnitude whenever
    the landscape is locally smooth, which it is near focus.

    Vertex offset for three equally-spaced samples ``y[-1]``,
    ``y[0]``, ``y[+1]`` measured at step spacing ``h``:

    .. math::

        \\Delta = \\frac{h}{2} \\cdot
            \\frac{y[-1] - y[+1]}{y[-1] - 2\\,y[0] + y[+1]}

    The denominator vanishes on flat regions (triple-equal values)
    and on boundary pixels where we don't have both neighbours —
    we fall back to the grid step for those.
    """
    n_steps, h, w = stack.shape
    if n_steps < 3:
        return z_values[idx]

    step = float(z_values[1] - z_values[0])
    # Only refine interior samples; boundary argmax stays on grid.
    interior = (idx > 0) & (idx < n_steps - 1)

    # Pull neighbour values across the z axis without materialising
    # full-sized tensors.
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y_m = stack[np.clip(idx - 1, 0, n_steps - 1), yy, xx]
    y_0 = stack[idx, yy, xx]
    y_p = stack[np.clip(idx + 1, 0, n_steps - 1), yy, xx]

    denom = y_m - 2.0 * y_0 + y_p
    # Parabola must be strictly concave — for argmax landscapes the
    # denominator is negative; reject near-zero to avoid dividing
    # into noise.
    safe = interior & (np.abs(denom) > 1e-12)

    delta = np.zeros_like(y_0, dtype=np.float64)
    delta[safe] = 0.5 * (y_m[safe] - y_p[safe]) / denom[safe]
    # Clamp into [-1, +1] samples — if parabolic fit pushes past
    # neighbours the quadratic model is a bad fit; discard.
    delta = np.clip(delta, -1.0, 1.0)

    z_map = z_values[idx].astype(np.float64) + step * delta
    return z_map.astype(np.float32)


@dataclass(frozen=True)
class ClusterHeight:
    """Summary stats for one connected high-confidence region of a depth map.

    Attributes
    ----------
    cluster_id
        1-based label from :func:`scipy.ndimage.label`. ``0`` is never
        emitted (it's the background label).
    centroid_yx
        (y, x) pixel centre of the cluster — useful for overlaying
        markers on the phase panel.
    area_px
        Number of pixels in the cluster. Combine with ``pixel_size_m``
        at call-site to get µm² if you need it.
    z_mean_m
        Confidence-weighted mean z inside the cluster. Heavier pixels
        pull the mean; random low-signal pixels contribute less.
    z_std_m
        Confidence-weighted standard deviation of z. Small → the
        cluster sits at one well-defined plane; large → it spans
        depth (stacked cells, tilted substrate).
    mean_confidence
        Average of the depth-map confidence inside the cluster. A
        filter threshold downstream (e.g. "drop clusters with mean
        confidence < 0.3") keeps noise clusters out of the report.
    """
    cluster_id: int
    centroid_yx: tuple
    area_px: int
    z_mean_m: float
    z_std_m: float
    mean_confidence: float


def segment_depth_clusters(
    result: DepthMapResult,
    *,
    confidence_threshold_frac: float = 0.20,
    min_area_px: int = 20,
) -> list[ClusterHeight]:
    """Connected-component analysis of the high-confidence region.

    Pixels whose confidence exceeds ``confidence_threshold_frac`` of
    the map's global max are kept. The survivors are labelled by
    :func:`scipy.ndimage.label` (4-connectivity, SciPy's default),
    components smaller than ``min_area_px`` are dropped, and each
    surviving cluster is summarised into a :class:`ClusterHeight`.

    The confidence weighting on ``z_mean`` and ``z_std`` matters most
    for cell clusters on an uneven substrate — pixels sitting on the
    brightest phase edges dominate, so the reported z reflects the
    "in-focus locus" of the cluster rather than any outlier fringe.
    """
    if not (0.0 <= confidence_threshold_frac <= 1.0):
        raise ValueError(
            f"confidence_threshold_frac must be in [0, 1] "
            f"(got {confidence_threshold_frac})"
        )
    if min_area_px < 1:
        raise ValueError(f"min_area_px must be >= 1 (got {min_area_px})")

    from scipy.ndimage import label

    conf = result.confidence
    z_map = result.z_map
    conf_max = float(conf.max()) if conf.size else 0.0
    if conf_max <= 0.0:
        return []

    cutoff = conf_max * float(confidence_threshold_frac)
    mask = (conf >= cutoff)
    if not np.any(mask):
        return []

    labelled, n = label(mask)
    if n == 0:
        return []

    out: list[ClusterHeight] = []
    for cluster_id in range(1, n + 1):
        sel = (labelled == cluster_id)
        area = int(np.sum(sel))
        if area < min_area_px:
            continue

        # Confidence-weighted z stats.
        w = conf[sel].astype(np.float64)
        z = z_map[sel].astype(np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            z_mean = float(np.mean(z))
            z_std = float(np.std(z))
        else:
            z_mean = float(np.sum(w * z) / w_sum)
            z_var = float(np.sum(w * (z - z_mean) ** 2) / w_sum)
            z_std = float(np.sqrt(max(z_var, 0.0)))

        # Centroid — unweighted; we want the geometric centre of the
        # mask so downstream visual overlays hit the middle of the blob.
        ys, xs = np.where(sel)
        centroid = (float(ys.mean()), float(xs.mean()))

        out.append(ClusterHeight(
            cluster_id=int(cluster_id),
            centroid_yx=centroid,
            area_px=area,
            z_mean_m=z_mean,
            z_std_m=z_std,
            mean_confidence=float(w.mean()),
        ))

    # Largest-first ordering — the biologist wants the dominant cluster
    # at the top of the CSV / table.
    out.sort(key=lambda c: c.area_px, reverse=True)
    return out


def write_cluster_heights_csv(
    path,
    clusters,
    *,
    sample_id: str = "",
    pixel_size_m: Optional[float] = None,
) -> None:
    """Flatten a list of :class:`ClusterHeight` into a CSV.

    Columns: ``sample_id``, ``cluster_id``, ``area_px``, ``area_um2``
    (if ``pixel_size_m`` given), ``centroid_y_px``, ``centroid_x_px``,
    ``z_mean_mm``, ``z_std_mm``, ``mean_confidence``. Same ``sample_id``
    the toolbar carries → round-trips into LIMS.
    """
    import csv
    from pathlib import Path

    cols = [
        "sample_id", "cluster_id", "area_px",
    ]
    if pixel_size_m is not None:
        cols.append("area_um2")
    cols.extend([
        "centroid_y_px", "centroid_x_px",
        "z_mean_mm", "z_std_mm", "mean_confidence",
    ])

    px_um2 = (float(pixel_size_m) * 1e6) ** 2 if pixel_size_m else None

    out = Path(path)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for c in clusters:
            row = {
                "sample_id": sample_id,
                "cluster_id": c.cluster_id,
                "area_px": c.area_px,
                "centroid_y_px": f"{c.centroid_yx[0]:.2f}",
                "centroid_x_px": f"{c.centroid_yx[1]:.2f}",
                "z_mean_mm": f"{c.z_mean_m * 1e3:.4f}",
                "z_std_mm": f"{c.z_std_m * 1e3:.4f}",
                "mean_confidence": f"{c.mean_confidence:.4g}",
            }
            if px_um2 is not None:
                row["area_um2"] = f"{c.area_px * px_um2:.3f}"
            writer.writerow(row)

    _LOG.info("cluster-heights: wrote %s (%d rows)", out, len(clusters))


def mask_low_confidence(
    result: DepthMapResult,
    *,
    threshold_frac: float = 0.10,
    fill_value: float = float("nan"),
) -> np.ndarray:
    """Return a copy of ``result.z_map`` where low-confidence pixels
    are replaced with ``fill_value`` (default NaN).

    ``threshold_frac`` is a fraction of the confidence map's maximum.
    Pixels whose confidence falls below that are masked. NaN is the
    default so downstream code can decide whether to plot, ignore, or
    interpolate them.
    """
    if not (0.0 <= threshold_frac <= 1.0):
        raise ValueError(
            f"threshold_frac must be in [0, 1] (got {threshold_frac})"
        )
    z_map = result.z_map.astype(np.float32, copy=True)
    conf = result.confidence
    cutoff = float(conf.max()) * threshold_frac
    z_map[conf < cutoff] = fill_value
    return z_map


def write_depth_map_npz(
    path, result: DepthMapResult,
    *, sample_id: str = "", app_version: str = "",
) -> None:
    """Save a :class:`DepthMapResult` to a compressed .npz archive.

    Layout — array names are stable so the lab's MATLAB/Python loaders
    can rely on them:

    - ``z_map``: (H, W) float32
    - ``confidence``: (H, W) float32
    - ``z_values_m``: (N_z,) float32
    - ``metric`` / ``window_size`` / ``sample_id`` / ``app_version``:
      stored as 0-dim object arrays so ``np.load`` round-trips them.
    """
    from pathlib import Path
    out = Path(path)
    np.savez_compressed(
        out,
        z_map=result.z_map,
        confidence=result.confidence,
        z_values_m=result.z_values_m,
        metric=np.array(result.metric.name, dtype=object),
        window_size=np.array(result.window_size, dtype=np.int32),
        sample_id=np.array(sample_id or "", dtype=object),
        app_version=np.array(app_version or "", dtype=object),
    )
    _LOG.info("depth-map: wrote %s", out)


def write_depth_map_csv(
    path, result: DepthMapResult,
    *, sample_id: str = "", stride: int = 1,
) -> None:
    """Save ``z_map`` and ``confidence`` as a flat CSV (y, x, z_mm,
    confidence). Useful for quick R/Pandas inspection without reading
    npz. ``stride`` down-samples the grid to keep file sizes sane.
    """
    import csv
    from pathlib import Path
    out = Path(path)
    h, w = result.z_map.shape
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "y_px", "x_px", "z_mm", "confidence"])
        for y in range(0, h, max(1, int(stride))):
            for x in range(0, w, max(1, int(stride))):
                writer.writerow([
                    sample_id,
                    y, x,
                    f"{result.z_map[y, x] * 1e3:.4f}",
                    f"{result.confidence[y, x]:.4g}",
                ])
    _LOG.info("depth-map: wrote %s (stride=%d)", out, stride)


def write_tomography_bundle(
    directory,
    depth_result: DepthMapResult,
    clusters,
    qpi_batch_entries=None,
    *,
    base_name: str = "tomography",
    sample_id: str = "",
    app_version: str = "",
    pixel_size_m: Optional[float] = None,
) -> list:
    """Write every tomography artefact into one directory in one call.

    Files produced (all named ``{base_name}_<suffix>``):

    * ``{base_name}_depth.npz`` — compressed depth archive
      (loss-less, every field of :class:`DepthMapResult`)
    * ``{base_name}_depth.csv`` — flat CSV of ``(y, x, z_mm,
      confidence)`` at stride 4, for quick R/pandas inspection
    * ``{base_name}_clusters.csv`` — one row per cluster
      (``z_mean_mm``, ``z_std_mm``, centroid, area, confidence)
    * ``{base_name}_qpi_batch.csv`` — only if ``qpi_batch_entries``
      is truthy; one row per candidate focus plane with the full QPI
      column set + candidate metadata

    Returns the list of ``Path`` objects actually written — useful
    both for UI status messages and for tests.

    The writer is synchronous and uses each module's dedicated
    exporter (`write_depth_map_npz`, `write_depth_map_csv`,
    `write_cluster_heights_csv`, `core.qpi_batch.write_qpi_batch_csv`)
    so the output format stays identical whether the lab exports one
    artefact or four.
    """
    from pathlib import Path

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list = []

    depth_npz = out_dir / f"{base_name}_depth.npz"
    write_depth_map_npz(
        depth_npz, depth_result,
        sample_id=sample_id, app_version=app_version,
    )
    written.append(depth_npz)

    depth_csv = out_dir / f"{base_name}_depth.csv"
    write_depth_map_csv(
        depth_csv, depth_result,
        sample_id=sample_id, stride=4,
    )
    written.append(depth_csv)

    clusters_csv = out_dir / f"{base_name}_clusters.csv"
    write_cluster_heights_csv(
        clusters_csv, list(clusters or []),
        sample_id=sample_id, pixel_size_m=pixel_size_m,
    )
    written.append(clusters_csv)

    if qpi_batch_entries:
        # Local import keeps core.depth_map free of a qpi_batch import
        # at module load time (avoids a circular-ish coupling).
        from .qpi_batch import write_qpi_batch_csv
        qpi_csv = out_dir / f"{base_name}_qpi_batch.csv"
        write_qpi_batch_csv(
            qpi_csv, list(qpi_batch_entries),
            sample_id=sample_id, app_version=app_version,
        )
        written.append(qpi_csv)

    _LOG.info(
        "tomography-bundle: wrote %d file(s) under %s", len(written), out_dir,
    )
    return written


__all__ = [
    "DepthMapResult",
    "ClusterHeight",
    "compute_depth_map",
    "segment_depth_clusters",
    "mask_low_confidence",
    "write_depth_map_npz",
    "write_depth_map_csv",
    "write_cluster_heights_csv",
    "write_tomography_bundle",
]
