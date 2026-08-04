#!/usr/bin/env python3
"""
Enhanced Autofocus Benchmark — Metric Landscape Analysis + Candidate Z Reconstruction.

Module 1: Metric Landscape
  - For each image × backend × method × metric, runs a dense z-sweep and records
    the full score_vs_z array.
  - Analyses each curve: peak count, FWHM of main peak, SNR (peak / noise floor).
  - Plots z-score curves grouped by metric per image.

Module 2: Candidate Z Reconstruction
  - Collects unique best-z values across all metrics for each image.
  - Clusters them and picks up to 6 representative candidate z points.
  - Reconstructs amplitude + phase at each candidate z and saves to disk.

All outputs go to a single timestamped output folder.

Usage:
    python benchmark_landscape.py
    python benchmark_landscape.py --image-dir /path/to/images
"""
import sys, os, time, argparse, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from core.reconstruction import (
    ReconstructionParams, ReconstructionMethod,
    propagate, CachedReconstructor,
)
from core.fft_backend import (
    FFTBackendName, get_best_fft_backend,
    NumpyFFTBackend, ScipyFFTBackend,
)
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.autofocus import (
    FocusMetric, _calc_metric, _is_minimize, _make_fast_evaluator,
    downsample_complex_field,
)

# ─── Optical parameters (50x profile) ───
WAVELENGTH_M = 632.8e-9
PIXEL_SIZE_M = 4.4e-6
MASK_RADIUS  = 80
Z_MIN_M      = 0.0
Z_MAX_M      = 300e-3
LANDSCAPE_STEPS = 151          # dense sweep for landscape (was 51 for speed benchmark)
MAX_CANDIDATE_Z = 6            # max unique z reconstructions per image

ALL_METRICS = list(FocusMetric)
ALL_METHODS = [ReconstructionMethod.ASM, ReconstructionMethod.FRESNEL]
BACKEND_NAMES = [
    FFTBackendName.PYFFTW,
    FFTBackendName.MLX,
    FFTBackendName.SCIPY,
    FFTBackendName.NUMPY,
]


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def try_create_backend(name):
    try:
        b = get_best_fft_backend(prefer=name)
        return b if b.name == name else None
    except Exception:
        return None


def load_image(path):
    from PIL import Image
    img = np.array(Image.open(path)).astype(np.float32)
    if img.ndim == 3:
        img = img[..., 0]
    img = img - float(np.mean(img))
    mx = np.max(np.abs(img))
    if mx > 0:
        img = img / mx
    return img


def extract_field(img, mask_radius=MASK_RADIUS):
    offaxis_params = OffAxisParams(radius=mask_radius)
    fc, _, _, _ = extract_complex_field_offaxis_debug(img, offaxis_params)
    return fc


# ═══════════════════════════════════════════════════════════════
# Module 1: Metric Landscape
# ═══════════════════════════════════════════════════════════════

def compute_landscape(fc, params, method, fft_backend, z_arr):
    """
    Compute ALL metric scores over z_arr in a SINGLE pass of reconstructions.
    Returns dict[FocusMetric] -> np.array of scores.
    """
    import core.fft_backend as fb_mod
    original_fn = fb_mod.get_best_fft_backend
    fb_mod.get_best_fft_backend = lambda prefer=None: fft_backend

    try:
        fft = fft_backend
        recon = CachedReconstructor(fc.shape, method, fft)
        inp = fc.astype(np.complex64, copy=False)
        field_spectrum = fft.fft2(inp)
        if field_spectrum.dtype != np.complex64:
            field_spectrum = field_spectrum.astype(np.complex64)

        scores = {m: [] for m in ALL_METRICS}

        for z in z_arr:
            p = ReconstructionParams(
                wavelength_m=params.wavelength_m,
                pixel_size_m=params.pixel_size_m,
                z_m=float(z), n=params.n,
            )
            result = recon.reconstruct_from_spectrum(field_spectrum, p)
            amp = np.abs(result)
            for m in ALL_METRICS:
                scores[m].append(_calc_metric(amp, m))

        return {m: np.array(v) for m, v in scores.items()}

    finally:
        fb_mod.get_best_fft_backend = original_fn
        import core.reconstruction as recon_mod
        recon_mod._GLOBAL_RECON_CACHE = None


def analyze_landscape(z_arr, scores_arr, metric):
    """
    Analyse a single metric's z-score curve.
    Returns dict with: n_peaks, main_peak_z_mm, fwhm_mm, snr, is_unimodal, peak_type.
    """
    vals = scores_arr.copy()
    minimize = _is_minimize(metric)

    # Normalize to 0-1 (invert if minimize)
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-15:
        return {
            "n_peaks": 0, "main_peak_z_mm": 0.0, "fwhm_mm": 0.0,
            "snr": 0.0, "is_unimodal": False, "peak_type": "flat",
        }

    vals_norm = (vals - vmin) / (vmax - vmin)
    if minimize:
        vals_norm = 1.0 - vals_norm

    # Smooth for peak detection
    sigma = max(1.5, len(vals_norm) / 50.0)
    vals_smooth = gaussian_filter1d(vals_norm, sigma=sigma)

    # Find peaks
    peaks_idx, props = find_peaks(vals_smooth, prominence=0.05, distance=3)
    n_peaks = len(peaks_idx)

    if n_peaks == 0:
        # No peak found — take argmax
        main_idx = int(np.argmax(vals_smooth))
        return {
            "n_peaks": 0, "main_peak_z_mm": float(z_arr[main_idx]) * 1e3,
            "fwhm_mm": 0.0, "snr": 0.0,
            "is_unimodal": False, "peak_type": "plateau",
        }

    # Main peak = highest prominence
    main_peak_order = np.argsort(-props["prominences"])
    main_local_idx = main_peak_order[0]
    main_idx = peaks_idx[main_local_idx]
    main_z_mm = float(z_arr[main_idx]) * 1e3

    # FWHM via scipy peak_widths (at half max)
    try:
        widths, _, _, _ = peak_widths(vals_smooth, [main_idx], rel_height=0.5)
        dz = float(z_arr[1] - z_arr[0]) * 1e3  # mm per step
        fwhm_mm = float(widths[0]) * dz
    except Exception:
        fwhm_mm = 0.0

    # SNR = peak height / noise floor
    peak_height = float(vals_smooth[main_idx])
    # Noise floor = median of bottom 25% of smoothed values
    sorted_vals = np.sort(vals_smooth)
    noise_floor = float(np.median(sorted_vals[:max(1, len(sorted_vals) // 4)]))
    snr = (peak_height - noise_floor) / max(noise_floor, 1e-15) if noise_floor > 1e-15 else peak_height / 1e-10

    # Classify
    if n_peaks == 1 and fwhm_mm > 0:
        peak_type = "clean_unimodal"
    elif n_peaks == 1:
        peak_type = "broad_unimodal"
    elif n_peaks <= 3:
        peak_type = "multimodal_few"
    else:
        peak_type = "multimodal_noisy"

    return {
        "n_peaks": n_peaks,
        "main_peak_z_mm": main_z_mm,
        "fwhm_mm": fwhm_mm,
        "snr": snr,
        "is_unimodal": n_peaks == 1,
        "peak_type": peak_type,
    }


def plot_landscape(z_arr, all_scores, img_name, backend_name, method_name, out_dir):
    """
    Plot z-score curves for all metrics, saved as a single figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_mm = z_arr * 1e3
    n_metrics = len(ALL_METRICS)
    fig, axes = plt.subplots(
        (n_metrics + 1) // 2, 2, figsize=(18, 3.2 * ((n_metrics + 1) // 2)),
        sharex=True,
    )
    axes = axes.flatten()

    for i, metric in enumerate(ALL_METRICS):
        ax = axes[i]
        vals = all_scores[metric]
        minimize = _is_minimize(metric)

        # Normalize
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-15:
            vals_norm = (vals - vmin) / (vmax - vmin)
            if minimize:
                vals_norm = 1.0 - vals_norm
        else:
            vals_norm = np.zeros_like(vals)

        sigma = max(1.5, len(vals_norm) / 50.0)
        vals_smooth = gaussian_filter1d(vals_norm, sigma=sigma)

        ax.plot(z_mm, vals_norm, alpha=0.35, linewidth=0.8, color="steelblue", label="raw (norm)")
        ax.plot(z_mm, vals_smooth, linewidth=1.8, color="darkblue", label="smoothed")

        # Mark peaks
        peaks_idx, props = find_peaks(vals_smooth, prominence=0.05, distance=3)
        if len(peaks_idx) > 0:
            ax.plot(z_mm[peaks_idx], vals_smooth[peaks_idx], "rv", markersize=6)

        info = analyze_landscape(z_arr, vals, metric)
        ax.set_title(
            f"{metric.value}  |  peaks={info['n_peaks']}  "
            f"FWHM={info['fwhm_mm']:.2f}mm  SNR={info['snr']:.1f}  [{info['peak_type']}]",
            fontsize=9, fontweight="bold",
        )
        ax.set_ylabel("score (norm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Remove extra axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    axes[-2].set_xlabel("z (mm)", fontsize=9)
    if len(axes) > n_metrics - 1:
        axes[n_metrics - 1].set_xlabel("z (mm)", fontsize=9)

    stem = Path(img_name).stem
    fig.suptitle(
        f"Metric Landscape — {stem}\n{backend_name} | {method_name}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    fname = f"landscape_{stem}_{backend_name}_{method_name}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ═══════════════════════════════════════════════════════════════
# Module 2: Candidate Z Reconstruction
# ═══════════════════════════════════════════════════════════════

def cluster_z_values(z_mm_list, tol_mm=2.0, max_candidates=MAX_CANDIDATE_Z):
    """
    Cluster z values that are within tol_mm of each other.
    Returns up to max_candidates representative z values (cluster medians), sorted.
    """
    if not z_mm_list:
        return []

    sorted_z = sorted(set(z_mm_list))
    clusters = []
    current_cluster = [sorted_z[0]]

    for z in sorted_z[1:]:
        if z - current_cluster[-1] <= tol_mm:
            current_cluster.append(z)
        else:
            clusters.append(current_cluster)
            current_cluster = [z]
    clusters.append(current_cluster)

    # Sort clusters by size (most common first), then take representatives
    clusters.sort(key=lambda c: -len(c))
    representatives = [float(np.median(c)) for c in clusters[:max_candidates]]
    representatives.sort()
    return representatives


def reconstruct_at_z(fc_full, base_params, method, z_m, fft_backend):
    """
    Reconstruct at a specific z and return (amplitude, phase) as float32 arrays.
    Uses FULL resolution field (not downsampled).
    """
    import core.fft_backend as fb_mod
    original_fn = fb_mod.get_best_fft_backend
    fb_mod.get_best_fft_backend = lambda prefer=None: fft_backend

    try:
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=z_m, n=base_params.n,
        )
        result = propagate(fc_full, params, method, fft=fft_backend, force_python=True)
        amplitude = np.abs(result).astype(np.float32)
        phase = np.angle(result).astype(np.float32)
        return amplitude, phase
    finally:
        fb_mod.get_best_fft_backend = original_fn
        import core.reconstruction as recon_mod
        recon_mod._GLOBAL_RECON_CACHE = None


def save_amplitude_phase(amplitude, phase, img_name, method_name, z_mm, out_dir):
    """Save amplitude and phase images to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = Path(img_name).stem
    z_str = f"{z_mm:.2f}".replace(".", "p")

    # Amplitude
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(amplitude, cmap="gray", aspect="equal")
    ax.set_title(f"Amplitude — z={z_mm:.2f}mm\n{stem} | {method_name}", fontsize=10)
    ax.axis("off")
    amp_fname = f"{stem}_{method_name}_z{z_str}mm_amplitude.png"
    fig.savefig(out_dir / amp_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Phase
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(phase, cmap="twilight", aspect="equal")
    ax.set_title(f"Phase — z={z_mm:.2f}mm\n{stem} | {method_name}", fontsize=10)
    ax.axis("off")
    phase_fname = f"{stem}_{method_name}_z{z_str}mm_phase.png"
    fig.savefig(out_dir / phase_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return amp_fname, phase_fname


def save_candidate_comparison(all_amp_data, img_name, method_name, out_dir):
    """
    Side-by-side amplitude comparison of all candidate z values for one image.
    all_amp_data: list of (z_mm, amplitude_array)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_amp_data)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (z_mm, amp) in zip(axes, all_amp_data):
        ax.imshow(amp, cmap="gray", aspect="equal")
        ax.set_title(f"z = {z_mm:.2f} mm", fontsize=10, fontweight="bold")
        ax.axis("off")

    stem = Path(img_name).stem
    fig.suptitle(
        f"Candidate Z Comparison — {stem} | {method_name}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fname = f"comparison_{stem}_{method_name}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ═══════════════════════════════════════════════════════════════
# Report Writer
# ═══════════════════════════════════════════════════════════════

def write_report(all_landscape_data, all_candidate_data, out_dir, out_path):
    """Write the combined text report."""
    lines = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"{'='*120}")
    lines.append(f"  METRIC LANDSCAPE & CANDIDATE Z RECONSTRUCTION REPORT")
    lines.append(f"  Generated: {ts}")
    lines.append(f"  Z range: {Z_MIN_M*1e3:.1f} – {Z_MAX_M*1e3:.1f} mm | Landscape steps: {LANDSCAPE_STEPS}")
    lines.append(f"  Output dir: {out_dir}")
    lines.append(f"{'='*120}")
    lines.append("")

    # ── Module 1: Landscape Analysis ──
    lines.append(f"{'━'*120}")
    lines.append(f"  MODULE 1: METRIC LANDSCAPE ANALYSIS")
    lines.append(f"{'━'*120}")
    lines.append("")

    for entry in all_landscape_data:
        img = entry["image"]
        backend = entry["backend"]
        method = entry["method"]
        analyses = entry["analyses"]  # dict[metric_name -> analysis dict]
        plot_file = entry.get("plot_file", "")

        lines.append(f"  ┌─ {img} | {backend} | {method}")
        lines.append(f"  │  Plot: {plot_file}")
        lines.append(f"  │")
        lines.append(f"  │  {'Metric':<24} {'Peaks':>6} {'Main Peak Z':>14} {'FWHM (mm)':>12} {'SNR':>10} {'Type':<20}")
        lines.append(f"  │  {'─'*24} {'─'*6} {'─'*14} {'─'*12} {'─'*10} {'─'*20}")

        for metric in ALL_METRICS:
            a = analyses[metric.value]
            lines.append(
                f"  │  {metric.value:<24} {a['n_peaks']:>6} "
                f"{a['main_peak_z_mm']:>12.2f}mm {a['fwhm_mm']:>12.2f} "
                f"{a['snr']:>10.1f} {a['peak_type']:<20}"
            )

        # Summary: best metrics for this combo
        lines.append(f"  │")

        # Rank by SNR (higher = better)
        ranked_snr = sorted(
            [(m, analyses[m.value]["snr"]) for m in ALL_METRICS],
            key=lambda x: -x[1]
        )
        lines.append(f"  │  ── Metric ranking by SNR (higher = cleaner peak) ──")
        for rank, (m, snr) in enumerate(ranked_snr[:5], 1):
            a = analyses[m.value]
            lines.append(
                f"  │    {rank}. {m.value:<24} SNR={snr:>8.1f}  "
                f"peaks={a['n_peaks']}  FWHM={a['fwhm_mm']:.2f}mm  [{a['peak_type']}]"
            )

        # Unimodal metrics
        unimodal = [m for m in ALL_METRICS if analyses[m.value]["is_unimodal"]]
        lines.append(f"  │")
        lines.append(f"  │  Unimodal metrics ({len(unimodal)}/{len(ALL_METRICS)}): "
                     f"{', '.join(m.value for m in unimodal) if unimodal else 'none'}")

        lines.append(f"  └{'─'*118}")
        lines.append("")

    # ── Module 2: Candidate Z Reconstruction ──
    lines.append(f"{'━'*120}")
    lines.append(f"  MODULE 2: CANDIDATE Z RECONSTRUCTION")
    lines.append(f"{'━'*120}")
    lines.append("")

    for entry in all_candidate_data:
        img = entry["image"]
        method = entry["method"]
        candidates = entry["candidates_mm"]
        files = entry["files"]
        comparison = entry.get("comparison_file", "")

        lines.append(f"  ┌─ {img} | {method}")
        lines.append(f"  │  Candidate z values: {len(candidates)}")
        lines.append(f"  │  Comparison plot: {comparison}")
        lines.append(f"  │")
        lines.append(f"  │  {'Z (mm)':>12}   {'Amplitude File':<50} {'Phase File':<50}")
        lines.append(f"  │  {'─'*12}   {'─'*50} {'─'*50}")

        for z_mm, (amp_f, ph_f) in zip(candidates, files):
            lines.append(f"  │  {z_mm:>10.2f}mm   {amp_f:<50} {ph_f:<50}")

        lines.append(f"  └{'─'*118}")
        lines.append("")

    # ── Grand Summary ──
    lines.append(f"{'='*120}")
    lines.append(f"  GRAND LANDSCAPE SUMMARY")
    lines.append(f"{'='*120}")
    lines.append("")

    # Aggregate: which metrics are consistently unimodal across all images?
    metric_unimodal_counts = {m.value: 0 for m in ALL_METRICS}
    metric_total_counts = {m.value: 0 for m in ALL_METRICS}
    metric_snr_sums = {m.value: 0.0 for m in ALL_METRICS}

    for entry in all_landscape_data:
        for m in ALL_METRICS:
            a = entry["analyses"][m.value]
            metric_total_counts[m.value] += 1
            if a["is_unimodal"]:
                metric_unimodal_counts[m.value] += 1
            metric_snr_sums[m.value] += a["snr"]

    lines.append(f"  {'Metric':<24} {'Unimodal Rate':>16} {'Avg SNR':>12} {'Recommendation':<30}")
    lines.append(f"  {'─'*24} {'─'*16} {'─'*12} {'─'*30}")

    for m in ALL_METRICS:
        total = metric_total_counts[m.value]
        uni = metric_unimodal_counts[m.value]
        rate = uni / max(total, 1)
        avg_snr = metric_snr_sums[m.value] / max(total, 1)

        if rate >= 0.8 and avg_snr > 5:
            rec = "★ HIGHLY RECOMMENDED"
        elif rate >= 0.5 and avg_snr > 2:
            rec = "✓ Good"
        elif rate >= 0.3:
            rec = "~ Acceptable"
        else:
            rec = "✗ Unreliable"

        lines.append(
            f"  {m.value:<24} {uni}/{total} ({rate*100:>5.1f}%) {avg_snr:>12.1f} {rec:<30}"
        )

    lines.append("")
    lines.append(f"{'='*120}")
    lines.append(f"  END OF REPORT")
    lines.append(f"{'='*120}")

    report_text = "\n".join(lines)
    out_path.write_text(report_text, encoding="utf-8")
    return report_text


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Enhanced Autofocus Benchmark — Landscape + Candidate Z")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=LANDSCAPE_STEPS,
                        help="Number of z-steps for landscape sweep")
    parser.add_argument("--backend", type=str, default=None,
                        help="Force a single backend (scipy, numpy)")
    parser.add_argument("--method", type=str, default=None,
                        help="Force a single method (ASM, Fresnel)")
    args = parser.parse_args()

    # ── Find images ──
    if args.image_dir:
        img_dir = Path(args.image_dir)
    else:
        main_repo = Path(os.environ.get("DHM_DATA_ROOT", ROOT / "labtest"))
        img_dir = main_repo if main_repo.exists() else ROOT / "labtest"

    if not img_dir.exists():
        print(f"ERROR: Image directory not found: {img_dir}")
        sys.exit(1)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
    if not images:
        print(f"ERROR: No images found in {img_dir}")
        sys.exit(1)

    # ── Output directory ──
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ROOT / f"benchmark_output_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Detect backends ──
    available_backends = []
    for bn in BACKEND_NAMES:
        if args.backend and bn.value != args.backend:
            continue
        b = try_create_backend(bn)
        if b is not None and b.name == bn:
            available_backends.append((bn, b))
            print(f"  [OK] FFT Backend: {bn.value}")
        else:
            print(f"  [--] FFT Backend: {bn.value} — not available")

    if not available_backends:
        print("ERROR: No FFT backends available!")
        sys.exit(1)

    # ── Select methods ──
    methods = ALL_METHODS
    if args.method:
        methods = [m for m in ALL_METHODS if m.value.lower() == args.method.lower()]

    n_steps = args.steps

    n_landscape = len(images) * len(available_backends) * len(methods)
    print(f"\n{'='*80}")
    print(f"ENHANCED AUTOFOCUS BENCHMARK — LANDSCAPE + CANDIDATE Z")
    print(f"{'='*80}")
    print(f"  Images:     {len(images)}")
    print(f"  Backends:   {len(available_backends)}")
    print(f"  Methods:    {len(methods)}")
    print(f"  Landscape steps: {n_steps}")
    print(f"  Total landscape sweeps: {n_landscape}")
    print(f"  Output dir: {out_dir}")
    print(f"{'='*80}\n")

    all_landscape_data = []
    # Per image: collect all best_z values to determine candidates
    image_best_z_collector = {}  # img_name -> list of z_mm

    combo_idx = 0

    for img_path in images:
        img_name = img_path.name
        print(f"\n{'─'*80}")
        print(f"IMAGE: {img_name}")
        print(f"{'─'*80}")

        try:
            img = load_image(img_path)
            fc_full = extract_field(img)
            print(f"  Shape: {fc_full.shape}")
        except Exception as e:
            print(f"  FAILED to load/extract: {e}")
            continue

        base_params = ReconstructionParams(
            wavelength_m=WAVELENGTH_M, pixel_size_m=PIXEL_SIZE_M,
            z_m=0.0, n=1.0,
        )
        fc_ds, params_ds = downsample_complex_field(fc_full, base_params, factor=2)
        print(f"  DS×2 shape: {fc_ds.shape}")

        z_arr = np.linspace(Z_MIN_M, Z_MAX_M, n_steps)

        if img_name not in image_best_z_collector:
            image_best_z_collector[img_name] = []

        for bn_name, fft_backend in available_backends:
            for recon_method in methods:
                combo_idx += 1
                label = f"[{combo_idx}/{n_landscape}] {bn_name.value} | {recon_method.value}"
                print(f"\n  {label} — computing landscape ({n_steps} steps × {len(ALL_METRICS)} metrics)...")

                t0 = time.perf_counter()
                all_scores = compute_landscape(
                    fc_ds, params_ds, recon_method, fft_backend, z_arr
                )
                elapsed = time.perf_counter() - t0
                print(f"    Landscape computed in {elapsed:.1f}s")

                # Analyse each metric
                analyses = {}
                for metric in ALL_METRICS:
                    info = analyze_landscape(z_arr, all_scores[metric], metric)
                    analyses[metric.value] = info

                    # Collect best_z for candidate detection
                    image_best_z_collector[img_name].append(info["main_peak_z_mm"])

                    status = "✓" if info["is_unimodal"] else "~"
                    print(f"    {status} {metric.value:<24} peak_z={info['main_peak_z_mm']:>8.2f}mm  "
                          f"peaks={info['n_peaks']}  FWHM={info['fwhm_mm']:>6.2f}mm  "
                          f"SNR={info['snr']:>6.1f}  [{info['peak_type']}]")

                # Plot
                print(f"    Plotting landscape...")
                plot_file = plot_landscape(
                    z_arr, all_scores, img_name,
                    bn_name.value, recon_method.value, out_dir
                )

                all_landscape_data.append({
                    "image": img_name,
                    "backend": bn_name.value,
                    "method": recon_method.value,
                    "analyses": analyses,
                    "plot_file": plot_file,
                })

    # ═══════════════════════════════════════════════════════════════
    # Module 2: Candidate Z Reconstruction
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"MODULE 2: CANDIDATE Z RECONSTRUCTION")
    print(f"{'='*80}")

    all_candidate_data = []

    # Use first available backend for reconstruction (quality is the same)
    recon_bn_name, recon_fft = available_backends[0]

    for img_path in images:
        img_name = img_path.name
        if img_name not in image_best_z_collector:
            continue

        z_mm_list = image_best_z_collector[img_name]
        candidates_mm = cluster_z_values(z_mm_list, tol_mm=2.0, max_candidates=MAX_CANDIDATE_Z)

        print(f"\n  {img_name}: {len(candidates_mm)} candidate z values: "
              f"{[f'{z:.2f}mm' for z in candidates_mm]}")

        if not candidates_mm:
            continue

        try:
            img = load_image(img_path)
            fc_full = extract_field(img)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

        base_params = ReconstructionParams(
            wavelength_m=WAVELENGTH_M, pixel_size_m=PIXEL_SIZE_M,
            z_m=0.0, n=1.0,
        )

        # Reconstruct with Fresnel (faster for full resolution)
        recon_method = ReconstructionMethod.FRESNEL

        amp_phase_files = []
        amp_data_for_comparison = []

        for z_mm in candidates_mm:
            z_m = z_mm * 1e-3
            print(f"    Reconstructing z={z_mm:.2f}mm ...")
            t0 = time.perf_counter()
            amplitude, phase = reconstruct_at_z(
                fc_full, base_params, recon_method, z_m, recon_fft
            )
            elapsed = time.perf_counter() - t0
            print(f"      Done in {elapsed:.2f}s")

            amp_f, ph_f = save_amplitude_phase(
                amplitude, phase, img_name, recon_method.value, z_mm, out_dir
            )
            amp_phase_files.append((amp_f, ph_f))
            amp_data_for_comparison.append((z_mm, amplitude))

        # Side-by-side comparison
        comp_file = save_candidate_comparison(
            amp_data_for_comparison, img_name, recon_method.value, out_dir
        )

        all_candidate_data.append({
            "image": img_name,
            "method": recon_method.value,
            "candidates_mm": candidates_mm,
            "files": amp_phase_files,
            "comparison_file": comp_file,
        })

    # ═══════════════════════════════════════════════════════════════
    # Write Report
    # ═══════════════════════════════════════════════════════════════
    report_path = out_dir / "landscape_report.txt"
    report_text = write_report(all_landscape_data, all_candidate_data, out_dir, report_path)

    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"  Output dir: {out_dir}")
    print(f"  Report: {report_path}")
    print(f"{'='*80}")
    print(f"\n{report_text}")


if __name__ == "__main__":
    main()
