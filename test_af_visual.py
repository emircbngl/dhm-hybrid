#!/usr/bin/env python3
"""
Visual autofocus test — runs multiple metrics/algos and saves reconstruction images.
Usage: python test_af_visual.py <hologram.png> [--ref <ref.png>] [--wl 632.8] [--px 4.4] [--zmin 0] [--zmax 50]
"""
import sys, argparse, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.ingestion import load_png
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.reconstruction import ReconstructionParams, ReconstructionMethod, propagate
from core.autofocus import (
    FocusMetric, _is_minimize,
    autofocus_zscan, robust_coarse_to_fine_search,
    coarse_to_fine_search, adaptive_distance_search,
    downsample_complex_field,
)


def load_pre(p):
    d = load_png(p)
    img = d.array.astype(np.float32)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    img -= float(np.mean(img))
    mx = float(np.max(np.abs(img)))
    if mx > 0:
        img /= mx
    return img


def run_test(holo_path, ref_path=None, wl_nm=632.8, px_um=4.4, zmin_mm=0.0, zmax_mm=50.0):
    out_dir = ROOT / "test_af_output"
    out_dir.mkdir(exist_ok=True)

    stem = Path(holo_path).stem
    print(f"{'='*70}")
    print(f"  AUTOFOCUS VISUAL TEST: {stem}")
    print(f"  WL={wl_nm}nm  PX={px_um}µm  Z=[{zmin_mm},{zmax_mm}]mm")
    if ref_path:
        print(f"  REF: {Path(ref_path).name}")
    print(f"{'='*70}\n")

    # Load and extract
    img = load_pre(holo_path)
    oa = OffAxisParams(radius=80, apodization="tukey", rolloff=0.25)
    fc, _, _, _ = extract_complex_field_offaxis_debug(img, oa)
    print(f"  Field shape: {fc.shape}")

    wl = wl_nm * 1e-9
    px = px_um * 1e-6
    params = ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=0, n=1.0)

    # Downsample for autofocus
    fc_ds, params_ds = downsample_complex_field(fc, params, factor=2)

    ref_fc_ds = None
    if ref_path:
        ref_img = load_pre(ref_path)
        ref_fc, _, _, _ = extract_complex_field_offaxis_debug(ref_img, oa)
        ref_fc_ds, _ = downsample_complex_field(ref_fc, params, factor=2)

    zmin_m = zmin_mm * 1e-3
    zmax_m = zmax_mm * 1e-3

    # Define test cases
    TESTS = [
        ("Tenengrad / zscan", FocusMetric.TENENGRAD, "zscan", None),
        ("Tenengrad / robust", FocusMetric.TENENGRAD, "robust", None),
        ("PhaseVar / zscan", FocusMetric.PHASE_VARIANCE, "zscan", None),
        ("PhaseVar / robust", FocusMetric.PHASE_VARIANCE, "robust", None),
        ("Laplacian / zscan", FocusMetric.LAPLACIAN_VARIANCE, "zscan", None),
        ("Brenner / zscan", FocusMetric.BRENNER, "zscan", None),
    ]

    if ref_fc_ds is not None:
        TESTS.append(("PhaseVar+Ref / zscan", FocusMetric.PHASE_VARIANCE, "zscan", ref_fc_ds))
        TESTS.append(("PhaseVar+Ref / robust", FocusMetric.PHASE_VARIANCE, "robust", ref_fc_ds))

    z_scan_list = list(np.linspace(zmin_m, zmax_m, 101))

    results = []
    print(f"\n  {'Test':<30} {'Algo':<10} {'best_z (mm)':>14} {'best_z (µm)':>14} {'Time':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*14} {'-'*14} {'-'*8}")

    for name, metric, algo, ref in TESTS:
        t0 = time.perf_counter()
        try:
            if algo == "zscan":
                res = autofocus_zscan(fc_ds, params_ds, z_scan_list,
                                      ReconstructionMethod.ASM, metric, ref_field=ref)
                best_z = res.best_z_m
            elif algo == "robust":
                res = robust_coarse_to_fine_search(
                    fc_ds, params_ds, ReconstructionMethod.ASM, metric,
                    zmin_m, zmax_m, n_coarse=40, ref_field=ref)
                best_z = res.best_z_m
            elif algo == "coarse_fine":
                res = coarse_to_fine_search(
                    fc_ds, params_ds, ReconstructionMethod.ASM, metric,
                    zmin_m, zmax_m, coarse_steps=40, ref_field=ref)
                best_z = res.best_z_m
            elapsed = time.perf_counter() - t0
            z_mm = best_z * 1e3
            z_um = best_z * 1e6
            print(f"  {name:<30} {algo:<10} {z_mm:>+14.4f} {z_um:>+14.1f} {elapsed:>7.2f}s")
            results.append((name, best_z, metric))
        except Exception as ex:
            elapsed = time.perf_counter() - t0
            print(f"  {name:<30} {algo:<10} {'ERROR':>14} {'':>14} {elapsed:>7.2f}s  {ex}")

    # Reconstruct and save images at each unique z
    unique_z = {}
    for name, z, metric in results:
        z_mm_rounded = round(z * 1e3, 3)
        if z_mm_rounded not in unique_z:
            unique_z[z_mm_rounded] = name

    print(f"\n  Reconstructing at {len(unique_z)} unique z values...")

    fig_rows = len(unique_z)
    fig, axes = plt.subplots(fig_rows, 2, figsize=(14, 5 * fig_rows))
    if fig_rows == 1:
        axes = axes[np.newaxis, :]

    for i, (z_mm, label) in enumerate(sorted(unique_z.items())):
        z_m = z_mm * 1e-3
        p = ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=z_m, n=1.0)
        result = propagate(fc, p, ReconstructionMethod.ASM, force_python=True)
        amp = np.abs(result)
        phase = np.angle(result)

        # Reference subtraction for display if available
        if ref_path:
            ref_fc_full, _, _, _ = extract_complex_field_offaxis_debug(load_pre(ref_path), oa)
            ref_result = propagate(ref_fc_full, p, ReconstructionMethod.ASM, force_python=True)
            ref_abs = np.abs(ref_result)
            safe = np.where(ref_abs > 1e-10, ref_result, np.ones_like(ref_result))
            result_sub = result / safe
            phase = np.angle(result_sub)

        ax_amp = axes[i, 0]
        ax_ph = axes[i, 1]

        vmin_a, vmax_a = np.percentile(amp, [1, 99])
        ax_amp.imshow(amp, cmap="gray", vmin=vmin_a, vmax=vmax_a)
        ax_amp.set_title(f"Amplitude — z={z_mm:.3f}mm ({z_mm*1e3:.1f}µm)\n[{label}]", fontsize=10)
        ax_amp.axis("off")

        ax_ph.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_ph.set_title(f"Phase — z={z_mm:.3f}mm ({z_mm*1e3:.1f}µm)\n[{label}]", fontsize=10)
        ax_ph.axis("off")

    fig.suptitle(f"Autofocus Results: {stem}\nWL={wl_nm}nm  PX={px_um}µm", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / f"af_test_{stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    return results, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hologram", type=str)
    parser.add_argument("--ref", type=str, default=None)
    parser.add_argument("--wl", type=float, default=632.8, help="Wavelength (nm)")
    parser.add_argument("--px", type=float, default=4.4, help="Pixel size (µm)")
    parser.add_argument("--zmin", type=float, default=0.0, help="Z min (mm)")
    parser.add_argument("--zmax", type=float, default=50.0, help="Z max (mm)")
    args = parser.parse_args()

    run_test(args.hologram, args.ref, args.wl, args.px, args.zmin, args.zmax)
