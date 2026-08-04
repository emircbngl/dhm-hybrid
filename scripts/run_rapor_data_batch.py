"""Custom batch runner for ``~/Desktop/rapor data`` (2026-05-04).

The lab session manifest is too irregular for the GUI batch dialog
(some sessions cherry-pick a few files, s1's reference is reused in
s2, s9 has two sub-batches with different references), so this
script encodes the per-session rules directly and runs the full
recon pipeline + rendering offline.

Per hologram we produce:

* ``spectrum.png``   — log-magnitude FFT (off-axis +1 order visible).
* ``amplitude.tiff`` / ``amplitude.png`` — |reference-divided field|.
* ``unwrapped.tiff`` / ``unwrapped.png`` — referenced phase, unwrapped.
* ``surface.png``    — 3D surface plot (Spectral colormap, hocanın stili).
* ``heatmap.png``    — 2D jet heatmap with a µm scalebar.

Then ``report.html`` ties everything together for paper-ready output.

Pipeline mirrors the v2 path with reference division applied AT EACH
trial z (B-051 / B-053 fixes): autofocus's metric judges the same
referenced field that the saved outputs show.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
import numpy as np
import tifffile

# Path setup — script lives under repo/scripts, src lives at repo/src.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.autofocus import FocusMetric  # noqa: E402
from core.autofocus.search_adaptive import adaptive_gradient_search  # noqa: E402
from core.background_phase import subtract_background  # noqa: E402
from core.ingestion import load_any  # noqa: E402
from core.phase_unwrap import unwrap_phase_advanced  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
)
from core.scalebar import compute_scalebar  # noqa: E402
# Shared reference-free pipeline — one implementation for demod / propagate /
# reference-division across the scripts, the DL inference module, and the
# depth/batch paths (2026-07-05 review de-duplicated it into core).
from core.pipelines.reffree_hybrid import (  # noqa: E402
    OpticalConfig,
    demodulate as _core_demodulate,
    propagate_field as _core_propagate_field,
    propagate_referenced as _core_propagate_referenced,
)

_LOG = logging.getLogger("rapor_batch")

# ---------------------------------------------------------------------------
# Optical parameters (set 2026-05-04 by the lab).
# ---------------------------------------------------------------------------
WAVELENGTH_M   = 632.8e-9
CAMERA_PIX_UM  = 4.4
MAGNIFICATION  = 50.0
EFFECTIVE_UM   = CAMERA_PIX_UM / MAGNIFICATION   # 0.088 µm/px at the sample
EFFECTIVE_M    = EFFECTIVE_UM * 1e-6
N_MEDIUM       = 1.337
MASK_RADIUS    = 80
Z_RANGE_MM     = 20.0   # 2026-05-04: ±3 mm saturated 27% of frames at +3.0
Z_STEPS        = 200    # adaptive_gradient max_evaluations
# Display contrast — 1-99 percentile clip on every rendered PNG/TIFF.
# Lab feedback 2026-05-04: default min-max stretch produced "sorunlu"
# (corrupted-looking) outputs because outlier pixels (unwrap phase
# discontinuities, ref-divide singularities) flattened the histogram.
CONTRAST_PCT_LO = 1.0
CONTRAST_PCT_HI = 99.0

# Lab capture directory. Override with DHM_DATA_ROOT; defaults to a
# repo-relative data/rapor so the script is runnable on a fresh clone.
DATA_ROOT = Path(
    os.environ.get(
        "DHM_DATA_ROOT",
        Path(__file__).resolve().parent.parent / "data" / "rapor",
    )
)
OUT_ROOT  = DATA_ROOT / "_output"

# Bundle the optical constants into the shared core pipeline's config.
_OPTICS = OpticalConfig(
    wavelength_m=WAVELENGTH_M, effective_pixel_m=EFFECTIVE_M,
    n_medium=N_MEDIUM, mask_radius=MASK_RADIUS,
    z_range_mm=Z_RANGE_MM, z_steps=Z_STEPS,
)


# ---------------------------------------------------------------------------
# Manifest — one entry per (session, reference) batch. ``files`` is either
# an explicit list or a "before_ref" / "all" rule resolved at run time.
# ---------------------------------------------------------------------------

@dataclass
class SessionPlan:
    label: str
    session_dir: Path
    ref_path: Optional[Path]
    files: list[Path] = field(default_factory=list)


def _files_before_ref(session_dir: Path, ref_name: str) -> list[Path]:
    """All PNG holograms in ``session_dir`` whose name sorts strictly
    before ``ref_name`` (timestamp-encoded filenames sort
    chronologically)."""
    all_files = sorted(p for p in session_dir.glob("*.png"))
    return [p for p in all_files if p.name < ref_name]


def _files_all(session_dir: Path, ref_name: str) -> list[Path]:
    """Every hologram in the session; ref filename excluded if present."""
    return sorted(p for p in session_dir.glob("*.png") if p.name != ref_name)


def _files_explicit(session_dir: Path, names: list[str]) -> list[Path]:
    return [session_dir / n for n in names]


def build_manifest() -> list[SessionPlan]:
    plans: list[SessionPlan] = []

    # s1 — ref 15:00:36; before-ref subset
    s1_dir = DATA_ROOT / "session_01"
    s1_ref = s1_dir / "DHM_2026_04_29_15_00_36.png"
    plans.append(SessionPlan(
        label="s1",
        session_dir=s1_dir,
        ref_path=s1_ref,
        files=_files_before_ref(s1_dir, s1_ref.name),
    ))

    # s2 — uses s1's ref against the (single-file) session_02
    s2_dir = DATA_ROOT / "session_02"
    plans.append(SessionPlan(
        label="s2",
        session_dir=s2_dir,
        ref_path=s1_ref,                           # cross-session reference
        files=sorted(s2_dir.glob("*.png")),
    ))

    # s3 — ref 15:12:28; before-ref subset
    s3_dir = DATA_ROOT / "session_03"
    s3_ref = s3_dir / "DHM_2026_04_29_15_12_28.png"
    plans.append(SessionPlan(
        label="s3",
        session_dir=s3_dir,
        ref_path=s3_ref,
        files=_files_before_ref(s3_dir, s3_ref.name),
    ))

    # s6 — explicit 3-file pick
    s6_dir = DATA_ROOT / "session_06"
    plans.append(SessionPlan(
        label="s6",
        session_dir=s6_dir,
        ref_path=s6_dir / "DHM_2026_04_30_16_36_35.png",
        files=_files_explicit(s6_dir, [
            "DHM_2026_04_30_16_36_25.png",
            "DHM_2026_04_30_16_36_40.png",
            "DHM_2026_04_30_16_37_12.png",
        ]),
    ))

    # s7 — ref 16:55:12; range 16:55:03..16:55:11 inclusive of both ends
    # (ref itself excluded — 16:55:12 is the reference frame)
    s7_dir = DATA_ROOT / "session_07"
    s7_files = []
    for sec in (3, 4, 5, 6, 7, 10, 11):  # 16:55:08-09 missing from disk
        s7_files.append(s7_dir / f"DHM_2026_04_30_16_55_{sec:02d}.png")
    plans.append(SessionPlan(
        label="s7",
        session_dir=s7_dir,
        ref_path=s7_dir / "DHM_2026_04_30_16_55_12.png",
        files=s7_files,
    ))

    # s8 — ref 16:55:30; before-ref subset
    s8_dir = DATA_ROOT / "session_08"
    s8_ref = s8_dir / "DHM_2026_04_30_16_55_30.png"
    plans.append(SessionPlan(
        label="s8",
        session_dir=s8_dir,
        ref_path=s8_ref,
        files=_files_before_ref(s8_dir, s8_ref.name),
    ))

    # s9-A — ref 16:55:35; range 16:55:46..56
    s9_dir = DATA_ROOT / "session_09"
    s9a_files = []
    for sec in range(46, 57):
        p = s9_dir / f"DHM_2026_04_30_16_55_{sec:02d}.png"
        if p.exists():
            s9a_files.append(p)
    plans.append(SessionPlan(
        label="s9_A",
        session_dir=s9_dir,
        ref_path=s9_dir / "DHM_2026_04_30_16_55_35.png",
        files=s9a_files,
    ))

    # s9-B — ref 16:56:20; range 16:56:10..19
    s9b_files = []
    for sec in range(10, 20):
        p = s9_dir / f"DHM_2026_04_30_16_56_{sec:02d}.png"
        if p.exists():
            s9b_files.append(p)
    plans.append(SessionPlan(
        label="s9_B",
        session_dir=s9_dir,
        ref_path=s9_dir / "DHM_2026_04_30_16_56_20.png",
        files=s9b_files,
    ))

    return plans


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _demodulate(raw: np.ndarray, *, mask_radius: int = MASK_RADIUS):
    """Pre-propagation Fourier demodulation. Returns (fc, spectrum_mag).

    Thin wrapper over the shared core pipeline bound to this lab's optics.
    """
    cfg = _OPTICS if mask_radius == MASK_RADIUS else OpticalConfig(
        wavelength_m=WAVELENGTH_M, effective_pixel_m=EFFECTIVE_M,
        n_medium=N_MEDIUM, mask_radius=mask_radius,
        z_range_mm=Z_RANGE_MM, z_steps=Z_STEPS,
    )
    return _core_demodulate(raw, cfg, return_spectrum=True)


def _propagate_referenced(fc: np.ndarray, ref_fc: np.ndarray,
                          z_m: float) -> np.ndarray:
    """Propagate sample and reference to ``z_m`` separately, then divide.

    Propagation does not commute with complex division, so the reference
    must be re-propagated at each trial z (B-053 audit, 2026-04-30).
    """
    return _core_propagate_referenced(fc, ref_fc, z_m, _OPTICS)


def _propagate_reffree(fc: np.ndarray, z_m: float) -> np.ndarray:
    """Propagate sample without a reference. Aberrations / curvature are
    instead handled by the post-unwrap background phase fit in process_one()."""
    return _core_propagate_field(fc, z_m, _OPTICS)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _percentile_limits(arr: np.ndarray,
                       lo: float = CONTRAST_PCT_LO,
                       hi: float = CONTRAST_PCT_HI) -> tuple[float, float]:
    """Robust display range — clips outliers without losing dynamic range.

    Falls back to plain min/max when ``arr`` is constant or all-NaN
    so the matplotlib imshow / TIFF write still gets a valid range.
    """
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0, 1.0
    finite_arr = arr[finite]
    vmin = float(np.percentile(finite_arr, lo))
    vmax = float(np.percentile(finite_arr, hi))
    if vmax - vmin < 1e-12:
        vmin = float(finite_arr.min())
        vmax = float(finite_arr.max())
        if vmax - vmin < 1e-12:
            vmax = vmin + 1e-9
    return vmin, vmax


def _render_spectrum(spectrum_mag: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=110)
    log_mag = np.log10(spectrum_mag + 1.0)
    im = ax.imshow(log_mag, cmap="inferno", aspect="equal")
    ax.set_title("FFT magnitude (log)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_scalebar(ax, spec, *, color="white", y_frac=0.93) -> None:
    """Paint a horizontal µm scalebar on a matplotlib Axes."""
    if spec is None:
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # ax.imshow inverts y by default — `ylim[0] > ylim[1]`.
    width = abs(xlim[1] - xlim[0])
    height = abs(ylim[0] - ylim[1])
    margin = width * 0.04
    x_end = max(xlim) - margin
    x_start = x_end - spec.length_px
    y_pos = max(ylim) - height * (1 - y_frac)
    ax.plot([x_start, x_end], [y_pos, y_pos],
            color=color, linewidth=4, solid_capstyle="butt")
    ax.text((x_start + x_end) / 2, y_pos - height * 0.025, spec.label,
            color=color, ha="center", va="bottom", fontsize=11,
            fontweight="bold")


def _render_amplitude(amp: np.ndarray, path: Path) -> None:
    vmin, vmax = _percentile_limits(amp)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=110)
    im = ax.imshow(amp, cmap="gray", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(f"Amplitude (auto-contrast {CONTRAST_PCT_LO:.0f}-{CONTRAST_PCT_HI:.0f}%)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    spec = compute_scalebar(amp.shape[1], EFFECTIVE_UM)
    _draw_scalebar(ax, spec)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_unwrapped(uw: np.ndarray, path: Path, *, title: str) -> None:
    vmin, vmax = _percentile_limits(uw)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=110)
    im = ax.imshow(uw, cmap="jet", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase (rad)")
    spec = compute_scalebar(uw.shape[1], EFFECTIVE_UM)
    _draw_scalebar(ax, spec)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_surface(uw: np.ndarray, path: Path) -> None:
    """3D surface plot a la o2/myjulia/process_files.jl — Spectral
    colormap, aspect-(8, 8, 1) effect via ``box_aspect``.

    Z-axis range clipped to the same 1-99 percentile as the 2D views
    so a single outlier pixel doesn't crush the surface flat.
    """
    h, w = uw.shape
    # Downsample heavy frames so the matplotlib renderer doesn't choke.
    target = 256
    if max(h, w) > target:
        sy = max(1, h // target)
        sx = max(1, w // target)
        uw_d = uw[::sy, ::sx]
    else:
        uw_d = uw
    h_d, w_d = uw_d.shape

    xs = np.arange(w_d) * EFFECTIVE_UM
    ys = np.arange(h_d) * EFFECTIVE_UM
    X, Y = np.meshgrid(xs, ys)

    vmin, vmax = _percentile_limits(uw)

    fig = plt.figure(figsize=(7.5, 6.0), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    # Clip outliers into the percentile range for the surface — keeps
    # the lab's eye on the bulk topography rather than a single
    # ref-divide singularity that would dominate auto-axis scaling.
    uw_clipped = np.clip(uw_d, vmin, vmax)
    ax.plot_surface(X, Y, uw_clipped, cmap="Spectral", linewidth=0,
                    antialiased=True, rstride=1, cstride=1,
                    vmin=vmin, vmax=vmax)
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("phase (rad)")
    ax.set_title(f"Phase surface (auto-contrast)")
    ax.set_box_aspect((8, 8, 1))   # hocanın aspect=(8,8,1) yapısı
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_heatmap(uw: np.ndarray, path: Path) -> None:
    """2D jet heatmap with µm scalebar — hocanın process_files2.jl
    eşdeğeri."""
    vmin, vmax = _percentile_limits(uw)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=110)
    im = ax.imshow(uw, cmap="jet", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title("Phase Map (rad)")
    ax.set_xlabel("x-pixel")
    ax.set_ylabel("y-pixel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase (rad)")
    spec = compute_scalebar(uw.shape[1], EFFECTIVE_UM)
    _draw_scalebar(ax, spec)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_phase_tiff(arr: np.ndarray, path: Path) -> None:
    """Quantise unwrapped phase to uint16 using the 1-99 percentile
    range — auto contrast for downstream viewers (ImageJ, Fiji).

    Note: unwrapped phase can span far beyond [-π, π] (cumulative
    across the frame), so the canonical ``(angle + π) / (2π)`` map
    used for *wrapped* phase doesn't apply. The rad value isn't
    recoverable from the TIFF; the .npy / live-state path keeps the
    raw radians for downstream maths. Negative phase still preserved
    — only the display range is clipped (see feedback_phase_signs.md)."""
    vmin, vmax = _percentile_limits(arr)
    span = vmax - vmin if vmax > vmin else 1.0
    norm = (arr - vmin) / span
    out = (np.clip(norm, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(path, out)


def _save_amp_tiff(arr: np.ndarray, path: Path) -> None:
    vmin, vmax = _percentile_limits(arr)
    span = vmax - vmin if vmax > vmin else 1.0
    norm = (arr - vmin) / span
    out = (np.clip(norm, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(path, out)


# ---------------------------------------------------------------------------
# Per-hologram processing
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    session: str
    file: str
    out_dir: str
    best_z_mm: float
    best_score: float
    autofocus_runtime_s: float
    total_runtime_s: float
    note: str = ""


def process_one(holo_path: Path, ref_fc: Optional[np.ndarray],
                ref_spectrum: Optional[np.ndarray],
                out_dir: Path, *, session_label: str,
                cancel_flag,
                reference_free: bool = False,
                bg_method: str = "zernike",
                bg_n_terms: int = 15,
                bg_polynomial_order: int = 4) -> FrameRecord:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(load_any(holo_path).array)
    fc, spectrum = _demodulate(raw)
    if not reference_free:
        if ref_fc is None:
            raise RuntimeError("ref_fc is required when reference_free=False")
        if fc.shape != ref_fc.shape:
            raise RuntimeError(
                f"shape mismatch: sample {fc.shape} vs reference "
                f"{ref_fc.shape} — different camera ROI?"
            )

    # Adaptive-step autofocus, reference-aware. Adaptive grows the step
    # in flat regions and shrinks near the peak — covers a wide z range
    # (±20 mm here) without burning the evaluation budget on flat areas.
    z_min_m = -Z_RANGE_MM * 1e-3
    z_max_m = +Z_RANGE_MM * 1e-3
    base = ReconstructionParams(
        wavelength_m=WAVELENGTH_M, pixel_size_m=EFFECTIVE_M,
        z_m=0.0, n=N_MEDIUM,
    )
    af_t0 = time.monotonic()
    res = adaptive_gradient_search(
        field=fc, base_params=base, method=ReconstructionMethod.ASM,
        metric=FocusMetric.LAPLACIAN_VARIANCE,
        z_min_m=z_min_m, z_max_m=z_max_m,
        max_evaluations=Z_STEPS,
        ref_field=None if reference_free else ref_fc,
        cancel_check=cancel_flag,
    )
    af_runtime = time.monotonic() - af_t0
    best_z_m = float(res.best_z_m)
    best_z_mm = best_z_m * 1e3

    # Final propagate at best z. Reference-aware divides by propagated ref;
    # reference-free skips the divide and leans on Zernike/polynomial
    # background subtraction post-unwrap to remove plane curvature, tilt,
    # and higher-order aberrations.
    if reference_free:
        referenced = _propagate_reffree(fc, best_z_m)
    else:
        referenced = _propagate_referenced(fc, ref_fc, best_z_m)
    amp = np.abs(referenced).astype(np.float32)
    phase = np.angle(referenced).astype(np.float32)
    unwrapped = unwrap_phase_advanced(
        phase, complex_field=referenced,
        wavelength_m=WAVELENGTH_M, pixel_size_m=EFFECTIVE_M,
    ).astype(np.float32)

    if reference_free:
        unwrapped, _ = subtract_background(
            unwrapped, amplitude=amp,
            method=bg_method,
            n_terms=bg_n_terms,
            polynomial_order=bg_polynomial_order,
        )
        unwrapped = unwrapped.astype(np.float32)

    # Outputs.
    _render_spectrum(spectrum, out_dir / "spectrum.png")
    _render_amplitude(amp, out_dir / "amplitude.png")
    _save_amp_tiff(amp, out_dir / "amplitude.tiff")
    _render_unwrapped(unwrapped, out_dir / "unwrapped.png",
                      title=f"Unwrapped Δφ — z={best_z_mm:+.3f} mm")
    _save_phase_tiff(unwrapped, out_dir / "unwrapped.tiff")
    np.save(out_dir / "unwrapped.npy", unwrapped.astype(np.float32))
    _render_heatmap(unwrapped, out_dir / "heatmap.png")
    _render_surface(unwrapped, out_dir / "surface.png")

    try:
        rel_out_dir = out_dir.relative_to(out_dir.parents[1])
    except ValueError:
        rel_out_dir = out_dir
    record = FrameRecord(
        session=session_label,
        file=holo_path.name,
        out_dir=str(rel_out_dir),
        best_z_mm=best_z_mm,
        best_score=float(res.best_score),
        autofocus_runtime_s=af_runtime,
        total_runtime_s=time.monotonic() - t0,
    )
    return record


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rapor data — DHM batch report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
         margin: 2rem auto; max-width: 1400px; color: #1c1c1c; }}
  h1 {{ font-weight: 600; border-bottom: 2px solid #333; padding-bottom: .35em; }}
  h2 {{ margin-top: 2.5rem; padding: .35em .6em; background: #eef2f7;
        border-left: 4px solid #2c7be5; }}
  table.frames {{ border-collapse: collapse; width: 100%; margin-top: .8em; }}
  table.frames th, table.frames td {{ border-bottom: 1px solid #d0d7de;
      padding: .35em .6em; font-size: .92em; vertical-align: top; }}
  table.frames th {{ background: #f6f8fa; text-align: left; }}
  .params {{ background: #f6f8fa; padding: .8em 1em; border-radius: 6px;
             font-size: .92em; }}
  .frame {{ margin: 1.4em 0; padding: .8em; border: 1px solid #d0d7de;
            border-radius: 8px; background: #fff; }}
  .frame .hd {{ display: flex; justify-content: space-between;
               align-items: baseline; margin-bottom: .6em; }}
  .frame .name {{ font-family: ui-monospace, monospace; font-weight: 600; }}
  .frame .z {{ color: #2c7be5; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(5, 1fr);
           gap: .6em; }}
  .grid figure {{ margin: 0; }}
  .grid img {{ width: 100%; border-radius: 4px; border: 1px solid #d0d7de; }}
  .grid figcaption {{ text-align: center; font-size: .82em; color: #57606a;
                      padding-top: .25em; }}
  .meta {{ color: #57606a; font-size: .85em; margin-top: .3em; }}
</style>
</head>
<body>
<h1>DHM batch — rapor data ({timestamp})</h1>
<div class="params">
  <strong>Optical parameters</strong>:
  λ = {wl_nm:.1f} nm,
  camera pixel = {cam_um:.2f} µm,
  magnification = {mag:.0f}×,
  effective pixel = {eff_um:.4f} µm,
  n_medium = {n_med:.3f},
  mask radius = {mask} px<br>
  <strong>Autofocus</strong>: Adaptive Gradient (variable step),
  z ∈ [−{zr:.1f}, +{zr:.1f}] mm, max {zsteps} evaluations,
  metric = LAPLACIAN_VARIANCE, reference-aware.<br>
  <strong>Display</strong>: auto-contrast (1-99 percentile clip on every PNG/TIFF).<br>
  <strong>Pipeline</strong>: subtract_mean=True, off-axis demod, ASM,
  per-z reference division (B-053 audit), gradient-integration unwrap.
</div>
{sessions}
</body>
</html>
"""

_FRAME_TEMPLATE = """<div class="frame">
  <div class="hd">
    <span class="name">{name}</span>
    <span class="z">best z = {z:+.4f} mm</span>
  </div>
  <div class="grid">
    <figure><a href="{base}/spectrum.png"><img src="{base}/spectrum.png"></a><figcaption>Spectrum</figcaption></figure>
    <figure><a href="{base}/amplitude.png"><img src="{base}/amplitude.png"></a><figcaption>Amplitude</figcaption></figure>
    <figure><a href="{base}/unwrapped.png"><img src="{base}/unwrapped.png"></a><figcaption>Unwrapped Δφ</figcaption></figure>
    <figure><a href="{base}/heatmap.png"><img src="{base}/heatmap.png"></a><figcaption>2D heatmap</figcaption></figure>
    <figure><a href="{base}/surface.png"><img src="{base}/surface.png"></a><figcaption>3D surface</figcaption></figure>
  </div>
  <div class="meta">
    score = {score:.3g} · autofocus {af_t:.1f}s · total {tot_t:.1f}s
  </div>
</div>"""


def write_html_report(records: list[FrameRecord], plans: list[SessionPlan],
                      out_path: Path) -> None:
    sessions_html = []
    by_session: dict[str, list[FrameRecord]] = {}
    for r in records:
        by_session.setdefault(r.session, []).append(r)

    for plan in plans:
        recs = by_session.get(plan.label, [])
        if not recs:
            continue
        zs = [r.best_z_mm for r in recs]
        z_summary = (f"min {min(zs):+.3f} mm · max {max(zs):+.3f} mm · "
                     f"mean {np.mean(zs):+.3f} mm" if zs else "—")
        if plan.ref_path is not None:
            ref_rel = plan.ref_path.relative_to(DATA_ROOT)
            ref_html = f"reference: <code>{ref_rel}</code>"
        else:
            ref_html = "<em>reference-free</em>"
        sessions_html.append(
            f'<h2>{plan.label} — {ref_html}</h2>'
            f'<div class="meta">{len(recs)} hologram(s) · z-range '
            f'{z_summary}</div>'
        )
        for r in recs:
            sessions_html.append(_FRAME_TEMPLATE.format(
                name=r.file,
                z=r.best_z_mm,
                base=r.out_dir.replace(os.sep, "/"),
                score=r.best_score,
                af_t=r.autofocus_runtime_s,
                tot_t=r.total_runtime_s,
            ))

    html = _HTML_TEMPLATE.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M"),
        wl_nm=WAVELENGTH_M * 1e9,
        cam_um=CAMERA_PIX_UM,
        mag=MAGNIFICATION,
        eff_um=EFFECTIVE_UM,
        n_med=N_MEDIUM,
        mask=MASK_RADIUS,
        zr=Z_RANGE_MM,
        zsteps=Z_STEPS,
        sessions="\n".join(sessions_html),
    )
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(only: Optional[set[str]] = None,
        limit_per_session: Optional[int] = None,
        reference_free: bool = False,
        out_root: Optional[Path] = None,
        bg_method: str = "zernike",
        bg_n_terms: int = 15,
        bg_polynomial_order: int = 4) -> list[FrameRecord]:
    plans = build_manifest()
    if only:
        plans = [p for p in plans if p.label in only]

    out_root = out_root if out_root is not None else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    records: list[FrameRecord] = []
    cancel_flag = lambda: False  # not wired from CLI yet

    total_files = sum(len(p.files) for p in plans)
    if limit_per_session:
        total_files = sum(min(len(p.files), limit_per_session) for p in plans)
    done = 0

    for plan in plans:
        if not reference_free:
            if plan.ref_path is None or not plan.ref_path.exists():
                _LOG.error("[%s] reference missing: %s", plan.label, plan.ref_path)
                continue
        files = plan.files
        if limit_per_session:
            files = files[:limit_per_session]
        if not files:
            _LOG.warning("[%s] no holograms — skipping", plan.label)
            continue

        if reference_free:
            _LOG.info("[%s] %d hologram(s), reference-free (bg=%s)",
                      plan.label, len(files), bg_method)
            ref_fc = None
            ref_spectrum = None
        else:
            _LOG.info("[%s] %d hologram(s), ref=%s",
                      plan.label, len(files), plan.ref_path.name)
            ref_raw = np.asarray(load_any(plan.ref_path).array)
            ref_fc, ref_spectrum = _demodulate(ref_raw)

        for holo in files:
            done += 1
            out_dir = out_root / plan.label / holo.stem
            try:
                rec = process_one(
                    holo, ref_fc, ref_spectrum, out_dir,
                    session_label=plan.label, cancel_flag=cancel_flag,
                    reference_free=reference_free,
                    bg_method=bg_method,
                    bg_n_terms=bg_n_terms,
                    bg_polynomial_order=bg_polynomial_order,
                )
                records.append(rec)
                _LOG.info("  [%d/%d] %s → z=%+.4f mm (%.1fs)",
                          done, total_files, holo.name,
                          rec.best_z_mm, rec.total_runtime_s)
            except Exception as exc:
                _LOG.exception("  [%d/%d] %s FAILED: %s",
                               done, total_files, holo.name, exc)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="rapor data batch runner")
    parser.add_argument("--only", help="Comma-separated session labels "
                        "(e.g. 's1,s2'). Default: all.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap holograms per session — useful for pilot.")
    parser.add_argument("--report", default=None,
                        help="HTML report output path. Defaults to "
                        "<out-root>/report.html.")
    parser.add_argument("--reference-free", action="store_true",
                        help="Skip the reference hologram and remove plane "
                        "curvature/tilt/aberrations via numerical Zernike "
                        "(or polynomial) background fit instead.")
    parser.add_argument("--bg-method", choices=["zernike", "polynomial"],
                        default="zernike",
                        help="Background fit method when --reference-free "
                        "is set. Default: zernike.")
    parser.add_argument("--bg-n-terms", type=int, default=15,
                        help="Zernike Noll terms (default 15 = through tetrafoil).")
    parser.add_argument("--bg-polynomial-order", type=int, default=4,
                        help="Cartesian polynomial total order (default 4).")
    parser.add_argument("--out-root", default=None,
                        help="Override output directory. Reference-free runs "
                        "default to <DATA_ROOT>/_output_reffree/ to avoid "
                        "overwriting the reference-based ground truth.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.out_root is not None:
        out_root = Path(args.out_root)
    elif args.reference_free:
        out_root = DATA_ROOT / "_output_reffree"
    else:
        out_root = OUT_ROOT
    report_path = Path(args.report) if args.report else (out_root / "report.html")

    only = set(args.only.split(",")) if args.only else None
    plans = build_manifest()
    if only:
        plans = [p for p in plans if p.label in only]

    t0 = time.monotonic()
    records = run(
        only=only, limit_per_session=args.limit,
        reference_free=args.reference_free,
        out_root=out_root,
        bg_method=args.bg_method,
        bg_n_terms=args.bg_n_terms,
        bg_polynomial_order=args.bg_polynomial_order,
    )
    elapsed = time.monotonic() - t0

    # Persist machine-readable manifest of what got produced.
    manifest_path = out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(r) for r in records], indent=2),
        encoding="utf-8",
    )

    write_html_report(records, plans, report_path)
    _LOG.info("DONE — %d frames in %.1fs · report: %s",
              len(records), elapsed, report_path)
    _LOG.info("manifest: %s", manifest_path)

    # Quick focus-distance audit so the operator can sanity-check.
    if records:
        zs = np.array([r.best_z_mm for r in records])
        _LOG.info("focus z (mm): min=%+.4f, max=%+.4f, mean=%+.4f, std=%.4f",
                  zs.min(), zs.max(), zs.mean(), zs.std())


if __name__ == "__main__":
    main()
