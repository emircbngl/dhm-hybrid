#!/usr/bin/env python3
"""End-to-end demo on a synthetic hologram — no lab data, known ground truth.

Forward-models an off-axis hologram of a synthetic phase object, reconstructs it
with dhm-hybrid's own pipeline, and reports the error against the ground truth
that produced it. Also renders the figure used in the README.

    python examples/synthetic_demo.py

This is a self-consistency check of the pipeline, NOT a claim about accuracy on
real samples: the same optical model that generates the hologram is inverted to
read it back. It catches sign errors, carrier mis-detection, unwrapping failures
and scaling bugs; it cannot catch an error shared by the forward and inverse
models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug  # noqa: E402
from core.phase_unwrap import UnwrapConfig, UnwrapMethod, unwrap_phase_advanced  # noqa: E402

# ── Optical configuration (a plausible bench, not a specific instrument) ──
WAVELENGTH_M = 632.8e-9      # HeNe
PIXEL_SIZE_M = 3.45e-6       # typical CMOS pitch
N = 512                      # sensor is N x N
CARRIER_CYCLES = 64          # off-axis tilt, in fringes across the frame
BIT_DEPTH = 12
SEED = 7


def ground_truth_phase() -> np.ndarray:
    """A few cell-like spherical caps plus a wedge, in radians."""
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    phase = np.zeros((N, N), dtype=np.float32)

    # (cx, cy, radius_px, peak_rad)
    cells = [
        (170, 190, 46, 2.6),
        (300, 150, 34, 1.9),
        (230, 320, 55, 3.1),
        (360, 340, 28, 1.4),
    ]
    for cx, cy, r, peak in cells:
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inside = rr < r
        cap = np.zeros_like(phase)
        cap[inside] = peak * np.sqrt(np.clip(1.0 - (rr[inside] / r) ** 2, 0, 1))
        phase += cap

    # Gentle sample-slide wedge — the thing background fitting must remove
    phase += 0.35 * (xx / N) + 0.20 * (yy / N)
    return phase.astype(np.float32)


def forward_hologram(phase: np.ndarray) -> np.ndarray:
    """I = |O + R|^2 with a tilted reference, shot noise, 12-bit quantization."""
    rng = np.random.default_rng(SEED + 1)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)

    obj_amp = 0.9 + 0.1 * np.exp(
        -(((xx - N / 2) ** 2 + (yy - N / 2) ** 2) / (2 * (N / 2.2) ** 2))
    )
    obj = obj_amp * np.exp(1j * phase)

    # Reference: tilted plane wave -> off-axis +/-1 orders in the spectrum
    kx = 2 * np.pi * CARRIER_CYCLES / N
    ky = 2 * np.pi * (CARRIER_CYCLES * 0.6) / N
    ref = 1.0 * np.exp(1j * (kx * xx + ky * yy))

    intensity = np.abs(obj + ref) ** 2

    full_scale = 2 ** BIT_DEPTH - 1
    scaled = intensity / intensity.max() * (full_scale * 0.85)
    noisy = rng.poisson(np.clip(scaled, 0, None)).astype(np.float32)
    return np.clip(np.round(noisy), 0, full_scale).astype(np.float32)


def reconstruct(holo: np.ndarray):
    """Run the repo's own off-axis pipeline. Returns (phase, spectrum, center)."""
    params = OffAxisParams(radius=48, dc_exclusion_radius=24, apodization="tukey")
    field, center, spectrum, _mask = extract_complex_field_offaxis_debug(holo, params)

    wrapped = np.angle(field).astype(np.float32)
    cfg = UnwrapConfig(
        method=UnwrapMethod.GRADIENT_INTEGRATION,
        post_bg_remove=True,
        post_bg_order=2,
    )
    unwrapped = unwrap_phase_advanced(wrapped, cfg, complex_field=field)
    return unwrapped, spectrum, center


def detrend(a: np.ndarray, order: int = 2) -> np.ndarray:
    """Remove a low-order polynomial fitted on the background pixels.

    The pipeline removes background tilt/curvature by design, so the ground
    truth must have the same removed before the two can be differenced.
    """
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    x = (xx / a.shape[1]).ravel()
    y = (yy / a.shape[0]).ravel()
    terms = [np.ones_like(x), x, y]
    if order >= 2:
        terms += [x * x, y * y, x * y]
    A = np.stack(terms, axis=1)
    v = a.ravel()
    bg = v <= np.percentile(v, 40)
    coef, *_ = np.linalg.lstsq(A[bg], v[bg], rcond=None)
    return (v - A @ coef).reshape(a.shape)


def render(out: Path, holo, gt, rec, spec, center, rmse) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    err = rec - gt
    cy, cx = center

    for theme in ("dark", "light"):
        if theme == "dark":
            fg, bg, panel = "#e6e6e6", "#0d1117", "#161b22"
        else:
            fg, bg, panel = "#1f2328", "#ffffff", "#f6f8fa"

        fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.7))
        fig.patch.set_facecolor(bg)

        ax = axes[0]
        ax.imshow(holo, cmap="gray")
        ax.set_title("1 - Off-axis hologram\n512x512, 12-bit, shot noise",
                     color=fg, fontsize=11, pad=9)
        zy, zx, zs = 150, 150, 64
        ax.add_patch(Rectangle((zx, zy), zs, zs, ec="#f0883e", fc="none", lw=1.6))
        inset = ax.inset_axes([0.60, 0.03, 0.37, 0.37])
        inset.imshow(holo[zy:zy + zs, zx:zx + zs], cmap="gray")
        inset.set_xticks([]); inset.set_yticks([])
        for s in inset.spines.values():
            s.set_edgecolor("#f0883e"); s.set_linewidth(1.6)

        ax = axes[1]
        ax.imshow(np.log1p(spec), cmap="magma")
        ax.add_patch(Circle((cx, cy), 48, ec="#58a6ff", fc="none", lw=1.8))
        ax.annotate("+1 order\n(auto-detected)", xy=(cx, cy), xytext=(30, 60),
                    color="#58a6ff", fontsize=9.5,
                    arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.4))
        ax.set_title("2 - Angular spectrum\nsideband found, DC rejected",
                     color=fg, fontsize=11, pad=9)

        ax = axes[2]
        im = ax.imshow(rec, cmap="viridis", vmin=-0.2, vmax=3.2)
        ax.set_title("3 - Quantitative phase\nunwrapped, background removed",
                     color=fg, fontsize=11, pad=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("phase (rad)", color=fg, fontsize=9)
        cb.ax.tick_params(colors=fg, labelsize=8)
        cb.outline.set_edgecolor(fg)

        ax = axes[3]
        im = ax.imshow(err, cmap="coolwarm", vmin=-0.35, vmax=0.35)
        ax.set_title(f"4 - Error vs ground truth\nRMSE = {rmse:.3f} rad",
                     color=fg, fontsize=11, pad=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("error (rad)", color=fg, fontsize=9)
        cb.ax.tick_params(colors=fg, labelsize=8)
        cb.outline.set_edgecolor(fg)

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor(panel)
            for s in ax.spines.values():
                s.set_edgecolor(fg); s.set_alpha(0.25)

        fig.suptitle(
            "dhm-hybrid - synthetic hologram in, quantitative phase out.  "
            f"A {gt.max():.3f} rad object recovers to {rec.max():.3f} rad "
            f"({rmse:.3f} rad RMSE).",
            color=fg, fontsize=12.5, y=0.99,
        )
        fig.text(0.5, 0.015,
                 "Fully synthetic: no lab data. Reproduce with  "
                 "python examples/synthetic_demo.py",
                 ha="center", color=fg, fontsize=9, alpha=0.65, family="monospace")
        fig.tight_layout(rect=[0, 0.035, 1, 0.945])

        path = out / f"hero-synthetic-{theme}.png"
        fig.savefig(path, dpi=100, facecolor=bg)
        plt.close(fig)
        print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img", help="figure output directory")
    ap.add_argument("--no-figure", action="store_true", help="numbers only")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    gt = ground_truth_phase()
    holo = forward_hologram(gt)
    rec, spectrum, center = reconstruct(holo)

    gt_d = detrend(gt)
    rec_d = detrend(rec)
    # Global piston is not observable in QPI; align the two backgrounds.
    rec_d -= np.median(rec_d[gt_d < 0.05])
    gt_d -= np.median(gt_d[gt_d < 0.05])

    rmse = float(np.sqrt(np.mean((rec_d - gt_d) ** 2)))

    # The carrier puts the +1 order at a predictable offset from DC. This is an
    # independent check on the forward model, not just on the round trip.
    expected = (N / 2 - CARRIER_CYCLES * 0.6, N / 2 - CARRIER_CYCLES)
    print(f"+1 order: found {center}, carrier predicts "
          f"({expected[0]:.1f}, {expected[1]:.1f})")
    print(f"ground-truth peak : {gt_d.max():.3f} rad")
    print(f"reconstructed peak: {rec_d.max():.3f} rad")
    print(f"RMSE vs ground truth: {rmse:.4f} rad")

    if not args.no_figure:
        render(out, holo, gt_d, rec_d, spectrum, center, rmse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
