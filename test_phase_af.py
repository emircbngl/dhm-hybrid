#!/usr/bin/env python3
"""
Phase-based autofocus with object masking + reference subtraction.
1. Reconstruct at z=0 with ref subtraction → detect object
2. Mask the object region
3. Z-scan: propagate sample & ref → divide → phase in ROI → metric
4. Reconstruct at best z, show result
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.ingestion import load_png
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.reconstruction import ReconstructionParams, ReconstructionMethod, propagate, CachedReconstructor
from core.fft_backend import get_best_fft_backend
from core.autofocus import downsample_complex_field

WL = 632.8e-9
PX = 4.4e-6 / 50  # 50x effective

def load_field(path):
    d = load_png(path)
    img = d.array.astype(np.float32)
    if img.ndim == 3: img = np.mean(img, axis=2)
    img -= float(np.mean(img))
    mx = float(np.max(np.abs(img)))
    if mx > 0: img /= mx
    oa = OffAxisParams(radius=80, apodization="tukey", rolloff=0.25)
    fc, _, _, _ = extract_complex_field_offaxis_debug(img, oa)
    return fc


def propagate_with_ref(fc_s, fc_r, z_m, recon, field_spec_s, field_spec_r):
    """Propagate sample & ref to z, return ref-subtracted complex field."""
    params = ReconstructionParams(wavelength_m=WL, pixel_size_m=PX, z_m=z_m, n=1.0)
    result_s = recon.reconstruct_from_spectrum(field_spec_s, params)
    result_r = recon.reconstruct_from_spectrum(field_spec_r, params)
    ref_abs = np.abs(result_r)
    safe = np.where(ref_abs > 1e-10, result_r, np.ones_like(result_r))
    return result_s / safe


def detect_objects(amplitude, phase, n_objects=3):
    """Detect objects from amplitude contrast. Returns list of (cy, cx, radius)."""
    # Use amplitude deviation from median
    amp_med = np.median(amplitude)
    contrast = np.abs(amplitude - amp_med)
    threshold = np.percentile(contrast, 95)
    binary = contrast > threshold
    
    # Clean up
    binary = ndimage.binary_opening(binary, iterations=2)
    binary = ndimage.binary_closing(binary, iterations=3)
    
    labeled, n_labels = ndimage.label(binary)
    
    objects = []
    for i in range(1, n_labels + 1):
        coords = np.argwhere(labeled == i)
        area = len(coords)
        if area < 20:  # skip tiny noise
            continue
        cy, cx = coords.mean(axis=0)
        radius = max(10, np.sqrt(area / np.pi) * 1.5)
        objects.append((cy, cx, radius, area))
    
    # Sort by area (largest first), take top n
    objects.sort(key=lambda x: -x[3])
    return [(cy, cx, r) for cy, cx, r, _ in objects[:n_objects]]


def make_circular_mask(shape, cy, cx, radius, margin=5):
    """Create circular mask with margin around object."""
    ny, nx = shape
    Y, X = np.ogrid[:ny, :nx]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    return r <= (radius + margin)


def phase_tenengrad_roi(complex_field, mask):
    """Phase gradient energy within mask (Tenengrad on phase).
    MINIMIZE: in focus → smooth OPD, defocus → Fresnel rings → high gradient."""
    phase = np.angle(complex_field)
    sy = ndimage.sobel(phase, axis=0, mode='reflect')
    sx = ndimage.sobel(phase, axis=1, mode='reflect')
    grad_mag_sq = sy**2 + sx**2
    return float(np.mean(grad_mag_sq[mask]))


def phase_laplacian_var_roi(complex_field, mask):
    """Phase Laplacian variance within mask.
    MINIMIZE: smooth phase has low Laplacian, Fresnel rings have high."""
    phase = np.angle(complex_field)
    lap = ndimage.laplace(phase)
    return float(np.var(lap[mask]))


def phase_hf_energy_roi(complex_field, mask):
    """High-frequency energy of phase within bounding box of mask.
    MINIMIZE: focused phase is smooth (low HF), defocused has Fresnel rings (high HF)."""
    phase = np.angle(complex_field)
    # Get bounding box of mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    roi = phase[y0:y1+1, x0:x1+1]
    # FFT and measure HF content
    F = np.fft.fftshift(np.fft.fft2(roi))
    mag = np.abs(F)
    ny, nx = roi.shape
    cy, cx = ny // 2, nx // 2
    Y, X = np.ogrid[:ny, :nx]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_max = min(cy, cx)
    hf_mask = r > (r_max * 0.3)
    return float(np.sum(mag[hf_mask]**2))


def amp_flatness_roi(complex_field, mask):
    """Amplitude variance within mask.
    MINIMIZE: for phase objects, amplitude is flat at focus."""
    amp = np.abs(complex_field).astype(np.float64)
    return float(np.var(amp[mask]))


def amp_tenengrad_roi(complex_field, mask):
    """Amplitude Tenengrad within mask. MAXIMIZE for amplitude objects."""
    amp = np.abs(complex_field).astype(np.float64)
    sy = ndimage.sobel(amp, axis=0, mode='reflect')
    sx = ndimage.sobel(amp, axis=1, mode='reflect')
    grad_mag_sq = sy**2 + sx**2
    return float(np.mean(grad_mag_sq[mask]))


# Metrics: (name, function, maximize?)
# Phase metrics: MINIMIZE (focus = smooth phase)
# Amplitude metrics: depends
METRICS = [
    ("PhaseGrad↓",      phase_tenengrad_roi,    False),  # minimize
    ("PhaseLapVar↓",    phase_laplacian_var_roi, False),  # minimize
    ("PhaseHFEnergy↓",  phase_hf_energy_roi,    False),  # minimize
    ("AmpFlat↓",        amp_flatness_roi,       False),  # minimize
    ("AmpTenengrad↑",   amp_tenengrad_roi,      True),   # maximize
]


def run_test(sample_path, ref_path, z_min_mm=-0.1, z_max_mm=0.1, n_steps=201):
    out_dir = ROOT / "test_af_output"
    out_dir.mkdir(exist_ok=True)
    stem = Path(sample_path).stem

    print(f"{'='*70}")
    print(f"  PHASE AUTOFOCUS (masked + ref subtraction)")
    print(f"  Sample: {Path(sample_path).name}")
    print(f"  Ref:    {Path(ref_path).name}")
    print(f"  Z range: [{z_min_mm}, {z_max_mm}] mm, {n_steps} steps")
    print(f"{'='*70}\n")

    # Load fields
    fc_s = load_field(sample_path)
    fc_r = load_field(ref_path)
    print(f"  Field shape: {fc_s.shape}")

    # Setup fast reconstruction
    fft = get_best_fft_backend()
    method = ReconstructionMethod.ASM
    recon = CachedReconstructor(fc_s.shape, method, fft)

    spec_s = fft.fft2(fc_s.astype(np.complex64))
    spec_r = fft.fft2(fc_r.astype(np.complex64))
    if spec_s.dtype != np.complex64: spec_s = spec_s.astype(np.complex64)
    if spec_r.dtype != np.complex64: spec_r = spec_r.astype(np.complex64)

    # Step 1: Reconstruct at z=0 with ref subtraction for object detection
    print("  Step 1: Detecting objects at z=0...")
    field_z0 = propagate_with_ref(fc_s, fc_r, 0.0, recon, spec_s, spec_r)
    amp_z0 = np.abs(field_z0)
    phase_z0 = np.angle(field_z0)

    objects = detect_objects(amp_z0, phase_z0, n_objects=3)
    print(f"  Found {len(objects)} objects:")
    for i, (cy, cx, r) in enumerate(objects):
        print(f"    Object {i+1}: center=({cy:.0f},{cx:.0f}) radius={r:.0f}px")

    if not objects:
        print("  ERROR: No objects detected!")
        return

    # Use largest object
    cy, cx, radius = objects[0]
    mask = make_circular_mask(fc_s.shape, cy, cx, radius, margin=10)
    n_pixels = int(np.sum(mask))
    print(f"\n  Using object 1: center=({cy:.0f},{cx:.0f}) r={radius:.0f}px, mask={n_pixels}px")

    # Step 2: Z-scan with all metrics in masked ROI
    print(f"\n  Step 2: Z-scan ({n_steps} steps)...")
    z_arr = np.linspace(z_min_mm * 1e-3, z_max_mm * 1e-3, n_steps)

    metric_scores = {name: [] for name, _, _ in METRICS}
    t0 = time.perf_counter()

    for z_m in z_arr:
        params = ReconstructionParams(wavelength_m=WL, pixel_size_m=PX, z_m=z_m, n=1.0)
        result_s = recon.reconstruct_from_spectrum(spec_s, params)
        result_r = recon.reconstruct_from_spectrum(spec_r, params)
        ref_abs = np.abs(result_r)
        safe = np.where(ref_abs > 1e-10, result_r, np.ones_like(result_r))
        field = result_s / safe

        for name, fn, _ in METRICS:
            metric_scores[name].append(fn(field, mask))

    elapsed = time.perf_counter() - t0
    print(f"  Z-scan done in {elapsed:.1f}s")

    # Find best z for each metric
    print(f"\n  {'Metric':<22} {'Best z (mm)':>14} {'Best z (µm)':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14}")

    best_z_per_metric = {}
    for name, _, maximize in METRICS:
        scores = np.array(metric_scores[name])
        if maximize:
            best_idx = int(np.argmax(scores))
        else:
            best_idx = int(np.argmin(scores))
        best_z = float(z_arr[best_idx])
        best_z_per_metric[name] = best_z
        print(f"  {name:<22} {best_z*1e3:>+14.4f} {best_z*1e6:>+14.1f}")

    # Step 3: Plot metric landscapes + reconstructions
    print(f"\n  Step 3: Generating plots...")

    # Create figure: metric landscapes + reconstruction at each best z
    unique_z = list(set(round(z * 1e6, 1) for z in best_z_per_metric.values()))
    unique_z.sort()
    n_recon = len(unique_z)

    fig = plt.figure(figsize=(18, 5 * (len(METRICS) + n_recon) // 2 + 2))

    # Top section: metric landscapes
    z_mm = z_arr * 1e3
    for i, (name, _, maximize) in enumerate(METRICS):
        ax = fig.add_subplot(len(METRICS) + n_recon, 2, 2 * i + 1)
        scores = np.array(metric_scores[name])
        # Normalize
        smin, smax = scores.min(), scores.max()
        if smax - smin > 1e-15:
            scores_norm = (scores - smin) / (smax - smin)
        else:
            scores_norm = np.zeros_like(scores)
        ax.plot(z_mm, scores_norm, 'b-', linewidth=1)
        best_z = best_z_per_metric[name]
        ax.axvline(best_z * 1e3, color='r', linestyle='--', linewidth=1)
        ax.set_title(f"{name} ({'max' if maximize else 'min'}) → z={best_z*1e6:+.1f}µm",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("score (norm)")
        ax.grid(True, alpha=0.3)

        # Also show zoomed view
        ax2 = fig.add_subplot(len(METRICS) + n_recon, 2, 2 * i + 2)
        # Zoom to ±20µm around best
        z_center = best_z * 1e3
        zoom_mask = (z_mm > z_center - 0.02) & (z_mm < z_center + 0.02)
        if np.sum(zoom_mask) > 3:
            ax2.plot(z_mm[zoom_mask], scores_norm[zoom_mask], 'b-', linewidth=1.5)
            ax2.axvline(z_center, color='r', linestyle='--')
            ax2.set_title(f"{name} — zoomed ±20µm", fontsize=9)
            ax2.set_xlabel("z (mm)")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.plot(z_mm, scores_norm, 'b-', linewidth=1)
            ax2.set_title(f"{name} — full range", fontsize=9)

    # Bottom section: reconstructions at unique z values
    row_offset = len(METRICS)
    for j, z_um in enumerate(unique_z):
        z_m = z_um * 1e-6
        params = ReconstructionParams(wavelength_m=WL, pixel_size_m=PX, z_m=z_m, n=1.0)
        result_s = recon.reconstruct_from_spectrum(spec_s, params)
        result_r = recon.reconstruct_from_spectrum(spec_r, params)
        ref_abs = np.abs(result_r)
        safe = np.where(ref_abs > 1e-10, result_r, np.ones_like(result_r))
        field = result_s / safe

        amp = np.abs(field)
        phase = np.angle(field)

        # Crop around object for better visibility
        pad = int(radius * 3)
        y0 = max(0, int(cy - pad))
        y1 = min(fc_s.shape[0], int(cy + pad))
        x0 = max(0, int(cx - pad))
        x1 = min(fc_s.shape[1], int(cx + pad))

        ax_a = fig.add_subplot(len(METRICS) + n_recon, 2, 2 * (row_offset + j) + 1)
        vmin_a, vmax_a = np.percentile(amp[y0:y1, x0:x1], [1, 99])
        ax_a.imshow(amp[y0:y1, x0:x1], cmap="gray", vmin=vmin_a, vmax=vmax_a)
        ax_a.set_title(f"Amplitude — z={z_um:+.1f}µm ({z_um/1000:+.4f}mm)", fontsize=10)
        ax_a.axis("off")

        ax_p = fig.add_subplot(len(METRICS) + n_recon, 2, 2 * (row_offset + j) + 2)
        ax_p.imshow(phase[y0:y1, x0:x1], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_p.set_title(f"Phase — z={z_um:+.1f}µm ({z_um/1000:+.4f}mm)", fontsize=10)
        ax_p.axis("off")

    fig.suptitle(f"Phase Autofocus (masked ROI + ref subtraction)\n{stem}\n"
                 f"Object at ({cy:.0f},{cx:.0f}) r={radius:.0f}px",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / f"phase_af_{stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--zmin", type=float, default=-0.1)
    parser.add_argument("--zmax", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=201)
    args = parser.parse_args()
    run_test(args.sample, args.ref, args.zmin, args.zmax, args.steps)
