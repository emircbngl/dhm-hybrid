#!/usr/bin/env python3
"""
Full Autofocus Benchmark — all images × all algorithms × all metrics × all FFT backends.

Scans every image in the labtest/ folder and produces a comprehensive comparison
table showing best-focus z, timing, and evaluations for every combination.

Usage:
    python full_benchmark.py
    python full_benchmark.py --image-dir /path/to/images
    python full_benchmark.py --quick          # fewer combos for fast check
"""
import sys, os, time, argparse, datetime, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from core.reconstruction import ReconstructionParams, ReconstructionMethod, CachedReconstructor
from core.fft_backend import (
    FFTBackendName, get_best_fft_backend,
    NumpyFFTBackend, ScipyFFTBackend, PyFFTWBackend, MLXFFTBackend,
)
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.autofocus import (
    FocusMetric, autofocus_zscan, coarse_to_fine_search,
    robust_coarse_to_fine_search, adaptive_gradient_search,
    adaptive_ratio_search, adaptive_bracketing_search,
    adaptive_distance_search, downsample_complex_field,
)

# ─── Default optical parameters ───
WAVELENGTH_M = 632.8e-9
PIXEL_SIZE_M = 4.4e-6
MASK_RADIUS  = 80
Z_MIN_M      = 0.0
Z_MAX_M      = 300e-3      # 0–300 mm (wide range for various samples)
STEPS_LINEAR = 51
STEPS_ADAPT  = 100

# ─── All combos ───
ALL_METRICS = list(FocusMetric)

ALL_ALGORITHMS = [
    "Linear Sweep",
    "Robust C2F",
    "Golden Section",
    "Adaptive Gradient",
    "Adaptive Ratio",
    "Adaptive Bracketing",
    "Adaptive Distance",
]

ALL_METHODS = [ReconstructionMethod.ASM, ReconstructionMethod.FRESNEL]

BACKEND_NAMES = [
    FFTBackendName.PYFFTW,
    FFTBackendName.MLX,
    FFTBackendName.SCIPY,
    FFTBackendName.NUMPY,
]


def try_create_backend(name):
    """Try to instantiate an FFT backend, return None if unavailable."""
    try:
        return get_best_fft_backend(prefer=name)
    except Exception:
        return None


def load_image(path):
    """Load image as float32 grayscale."""
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
    """Extract complex field via off-axis processing."""
    offaxis_params = OffAxisParams(radius=mask_radius)
    fc, _, _, _ = extract_complex_field_offaxis_debug(img, offaxis_params)
    return fc


def run_algorithm(algo_name, fc, params, method, metric, fft_backend):
    """Run a single algorithm and return (best_z_m, elapsed_s, n_evals)."""
    # Temporarily override the global FFT backend for CachedReconstructor
    import core.fft_backend as fb_mod
    original_fn = fb_mod.get_best_fft_backend

    def _override(prefer=None):
        return fft_backend
    fb_mod.get_best_fft_backend = _override

    try:
        t0 = time.perf_counter()

        if algo_name == "Linear Sweep":
            z_arr = list(np.linspace(Z_MIN_M, Z_MAX_M, STEPS_LINEAR))
            res = autofocus_zscan(fc, params, z_arr, method, metric)
            best_z = res.best_z_m
            evals = len(res.scores)

        elif algo_name == "Robust C2F":
            res = robust_coarse_to_fine_search(
                fc, params, method, metric, Z_MIN_M, Z_MAX_M,
                n_coarse=STEPS_LINEAR, refine_factor=8, smooth_sigma=1.5,
            )
            best_z = res.best_z_m
            evals = getattr(res, 'total_evaluations', STEPS_LINEAR)

        elif algo_name == "Golden Section":
            res = coarse_to_fine_search(
                fc, params, method, metric, Z_MIN_M, Z_MAX_M,
                coarse_steps=STEPS_LINEAR, fine_tolerance_m=1e-6,
            )
            best_z = res.best_z_m
            evals = getattr(res, 'total_evaluations', STEPS_LINEAR)

        elif algo_name == "Adaptive Gradient":
            res = adaptive_gradient_search(
                fc, params, method, metric, Z_MIN_M, Z_MAX_M,
                max_evaluations=STEPS_ADAPT,
            )
            best_z = res.best_z_m
            evals = res.evaluations

        elif algo_name == "Adaptive Ratio":
            res = adaptive_ratio_search(
                fc, params, method, metric, Z_MIN_M, Z_MAX_M,
                max_evaluations=STEPS_ADAPT,
            )
            best_z = res.best_z_m
            evals = res.evaluations

        elif algo_name == "Adaptive Bracketing":
            res = adaptive_bracketing_search(
                fc, params, method, metric, Z_MIN_M, Z_MAX_M,
                n_refine_levels=3, refine_divisions=6,
                smooth_sigma=1.0, max_evaluations=STEPS_ADAPT,
            )
            best_z = res.best_z_m
            evals = res.evaluations

        elif algo_name == "Adaptive Distance":
            res = adaptive_distance_search(
                fc, params, method, metric, Z_MIN_M, Z_MAX_M,
                max_evaluations=STEPS_ADAPT,
            )
            best_z = res.best_z_m
            evals = res.evaluations

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        elapsed = time.perf_counter() - t0
        return best_z, elapsed, int(evals)

    finally:
        fb_mod.get_best_fft_backend = original_fn
        # Clear cached reconstructor so next backend starts fresh
        import core.reconstruction as recon_mod
        recon_mod._GLOBAL_RECON_CACHE = None


def main():
    parser = argparse.ArgumentParser(description="Full autofocus benchmark")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Folder with hologram images")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer algorithms & metrics")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: benchmark_results_<timestamp>.txt)")
    args = parser.parse_args()

    # Find image directory
    if args.image_dir:
        img_dir = Path(args.image_dir)
    else:
        # Try labtest in main repo
        main_repo = Path.home() / "Documents" / "Windsurf Projects" / "Reconstruction(Mac)" / "Hybrid" / "labtest"
        if main_repo.exists():
            img_dir = main_repo
        else:
            img_dir = ROOT / "labtest"

    if not img_dir.exists():
        print(f"ERROR: Image directory not found: {img_dir}")
        print("Use --image-dir to specify the correct path.")
        sys.exit(1)

    # Collect images
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
    if not images:
        print(f"ERROR: No images found in {img_dir}")
        sys.exit(1)

    # Output file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else ROOT / f"benchmark_results_{ts}.txt"

    # Detect available backends
    available_backends = []
    for bn in BACKEND_NAMES:
        b = try_create_backend(bn)
        if b is not None and b.name == bn:
            available_backends.append((bn, b))
            print(f"  [OK] FFT Backend: {bn.value}")
        else:
            print(f"  [--] FFT Backend: {bn.value} — not available")

    if not available_backends:
        print("ERROR: No FFT backends available!")
        sys.exit(1)

    # Select combos
    metrics = ALL_METRICS
    algorithms = ALL_ALGORITHMS
    methods = ALL_METHODS

    if args.quick:
        metrics = [FocusMetric.TENENGRAD, FocusMetric.LAPLACIAN_VARIANCE, FocusMetric.BRENNER]
        algorithms = ["Linear Sweep", "Robust C2F", "Adaptive Bracketing"]
        methods = [ReconstructionMethod.ASM]

    n_combos = len(images) * len(available_backends) * len(methods) * len(algorithms) * len(metrics)
    print(f"\n{'='*80}")
    print(f"FULL AUTOFOCUS BENCHMARK")
    print(f"{'='*80}")
    print(f"  Images:     {len(images)}")
    print(f"  Backends:   {len(available_backends)} — {', '.join(b[0].value for b in available_backends)}")
    print(f"  Methods:    {len(methods)} — {', '.join(m.value for m in methods)}")
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Metrics:    {len(metrics)}")
    print(f"  Total combos: {n_combos}")
    print(f"  Output:     {out_path}")
    print(f"{'='*80}\n")

    results = []  # list of dicts
    combo_idx = 0

    for img_path in images:
        img_name = img_path.name
        print(f"\n{'─'*80}")
        print(f"IMAGE: {img_name}")
        print(f"{'─'*80}")

        try:
            img = load_image(img_path)
            fc_full = extract_field(img)
            print(f"  Shape: {fc_full.shape}, dtype: {fc_full.dtype}")
        except Exception as e:
            print(f"  FAILED to load/extract: {e}")
            continue

        # DS×2 for speed
        base_params = ReconstructionParams(
            wavelength_m=WAVELENGTH_M, pixel_size_m=PIXEL_SIZE_M,
            z_m=0.0, n=1.0,
        )
        fc_ds, params_ds = downsample_complex_field(fc_full, base_params, factor=2)
        print(f"  DS×2 shape: {fc_ds.shape}")

        for bn_name, fft_backend in available_backends:
            for recon_method in methods:
                for algo in algorithms:
                    for metric in metrics:
                        combo_idx += 1
                        tag = f"[{combo_idx}/{n_combos}]"
                        label = f"{bn_name.value:>7} | {recon_method.value:>7} | {algo:<22} | {metric.value}"

                        try:
                            best_z, elapsed, evals = run_algorithm(
                                algo, fc_ds, params_ds, recon_method, metric, fft_backend
                            )
                            per_eval = (elapsed / evals * 1000) if evals > 0 else 0.0

                            row = {
                                "image": img_name,
                                "backend": bn_name.value,
                                "method": recon_method.value,
                                "algorithm": algo,
                                "metric": metric.value,
                                "best_z_mm": best_z * 1e3,
                                "elapsed_s": elapsed,
                                "evals": evals,
                                "per_eval_ms": per_eval,
                            }
                            results.append(row)

                            print(f"  {tag} {label}  →  z={best_z*1e3:>10.4f}mm  "
                                  f"t={elapsed:>6.2f}s  evals={evals:>4}  "
                                  f"ms/eval={per_eval:>5.1f}")

                        except Exception as e:
                            print(f"  {tag} {label}  →  FAILED: {e}")
                            results.append({
                                "image": img_name,
                                "backend": bn_name.value,
                                "method": recon_method.value,
                                "algorithm": algo,
                                "metric": metric.value,
                                "best_z_mm": None,
                                "elapsed_s": None,
                                "evals": None,
                                "per_eval_ms": None,
                                "error": str(e),
                            })

    # ═══════════════════════════════════════════════════════════════
    # WRITE REPORT
    # ═══════════════════════════════════════════════════════════════
    write_report(results, images, available_backends, methods, algorithms, metrics, out_path)
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE — {len(results)} results written to {out_path}")
    print(f"{'='*80}")


def write_report(results, images, backends, methods, algorithms, metrics, out_path):
    """Write a human-readable benchmark report."""
    lines = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"{'='*120}")
    lines.append(f"  AUTOFOCUS BENCHMARK REPORT")
    lines.append(f"  Generated: {ts}")
    lines.append(f"  Images: {len(images)} | Backends: {len(backends)} | Methods: {len(methods)}")
    lines.append(f"  Algorithms: {len(algorithms)} | Metrics: {len(metrics)}")
    lines.append(f"  Total runs: {len(results)}")
    lines.append(f"  Z range: {Z_MIN_M*1e3:.1f} – {Z_MAX_M*1e3:.1f} mm")
    lines.append(f"  Linear steps: {STEPS_LINEAR} | Adaptive budget: {STEPS_ADAPT}")
    lines.append(f"  Downsampled: ×2 (for speed)")
    lines.append(f"{'='*120}")
    lines.append("")

    # Group results by image
    img_names = sorted(set(r["image"] for r in results))

    for img_name in img_names:
        img_results = [r for r in results if r["image"] == img_name]
        valid = [r for r in img_results if r.get("best_z_mm") is not None]
        failed = [r for r in img_results if r.get("best_z_mm") is None]

        lines.append(f"{'━'*120}")
        lines.append(f"  IMAGE: {img_name}")
        lines.append(f"  Valid: {len(valid)} | Failed: {len(failed)}")
        lines.append(f"{'━'*120}")
        lines.append("")

        if not valid:
            lines.append("  (no valid results)")
            lines.append("")
            continue

        # ── Per-backend summary table ──
        for bn_name, _ in backends:
            bn_results = [r for r in valid if r["backend"] == bn_name.value]
            if not bn_results:
                continue

            lines.append(f"  ┌─ FFT Backend: {bn_name.value.upper()}")
            lines.append(f"  │")

            for recon_method in methods:
                mr = [r for r in bn_results if r["method"] == recon_method.value]
                if not mr:
                    continue

                lines.append(f"  │  Reconstruction: {recon_method.value.upper()}")
                lines.append(f"  │")

                # Header
                hdr = f"  │  {'Algorithm':<24} {'Metric':<22} {'Best Z (mm)':>12} {'Time (s)':>10} {'Evals':>6} {'ms/eval':>8}"
                lines.append(hdr)
                lines.append(f"  │  {'─'*24} {'─'*22} {'─'*12} {'─'*10} {'─'*6} {'─'*8}")

                # Sort by algorithm then metric
                mr_sorted = sorted(mr, key=lambda x: (x["algorithm"], x["metric"]))
                for r in mr_sorted:
                    lines.append(
                        f"  │  {r['algorithm']:<24} {r['metric']:<22} "
                        f"{r['best_z_mm']:>10.4f}mm {r['elapsed_s']:>10.3f} "
                        f"{r['evals']:>6} {r['per_eval_ms']:>7.1f}"
                    )

                lines.append(f"  │")

                # Best per algorithm (consensus z)
                lines.append(f"  │  ── Best per algorithm (most common focus z, lowest time) ──")
                for algo in algorithms:
                    algo_r = [r for r in mr if r["algorithm"] == algo]
                    if not algo_r:
                        continue
                    z_values = [r["best_z_mm"] for r in algo_r]
                    times = [r["elapsed_s"] for r in algo_r]
                    median_z = float(np.median(z_values))
                    best_time = min(times)
                    fastest = min(algo_r, key=lambda x: x["elapsed_s"])
                    lines.append(
                        f"  │    {algo:<24} median_z={median_z:>10.4f}mm  "
                        f"fastest={best_time:.3f}s ({fastest['metric']})"
                    )
                lines.append(f"  │")

            lines.append(f"  └{'─'*118}")
            lines.append("")

        # ── Cross-backend speed comparison ──
        lines.append(f"  ╔{'═'*118}")
        lines.append(f"  ║  SPEED COMPARISON ACROSS BACKENDS (median time per algorithm)")
        lines.append(f"  ╠{'═'*118}")

        backend_names = [bn.value for bn, _ in backends]
        hdr_parts = f"  ║  {'Algorithm':<24}"
        for bn in backend_names:
            hdr_parts += f" {bn:>12}"
        hdr_parts += f" {'Winner':>12}"
        lines.append(hdr_parts)
        lines.append(f"  ║  {'─'*24}" + "".join(f" {'─'*12}" for _ in backend_names) + f" {'─'*12}")

        for algo in algorithms:
            row_parts = f"  ║  {algo:<24}"
            times_by_bn = {}
            for bn in backend_names:
                algo_bn = [r for r in valid if r["image"] == img_name
                           and r["algorithm"] == algo and r["backend"] == bn]
                if algo_bn:
                    med_t = float(np.median([r["elapsed_s"] for r in algo_bn]))
                    times_by_bn[bn] = med_t
                    row_parts += f" {med_t:>10.3f}s"
                else:
                    row_parts += f" {'N/A':>12}"

            if times_by_bn:
                winner = min(times_by_bn, key=times_by_bn.get)
                row_parts += f" {winner:>12}"
            else:
                row_parts += f" {'—':>12}"
            lines.append(row_parts)

        lines.append(f"  ╚{'═'*118}")
        lines.append("")

        # ── Global best focus per image ──
        lines.append(f"  ★ OVERALL BEST FOCUS for {img_name}:")
        # Group by z, find consensus
        z_vals = np.array([r["best_z_mm"] for r in valid])
        median_z = float(np.median(z_vals))
        std_z = float(np.std(z_vals))
        fastest_overall = min(valid, key=lambda x: x["elapsed_s"])
        lines.append(f"    Consensus z (median):  {median_z:.4f} mm  (std: {std_z:.4f} mm)")
        lines.append(f"    Fastest combo:         {fastest_overall['backend']} | "
                     f"{fastest_overall['method']} | {fastest_overall['algorithm']} | "
                     f"{fastest_overall['metric']}  →  z={fastest_overall['best_z_mm']:.4f}mm  "
                     f"t={fastest_overall['elapsed_s']:.3f}s")

        # Find combo closest to median z AND fast
        for r in valid:
            r["_dist"] = abs(r["best_z_mm"] - median_z)
        closest = sorted(valid, key=lambda x: (x["_dist"], x["elapsed_s"]))[:3]
        lines.append(f"    Top 3 closest to consensus:")
        for i, r in enumerate(closest, 1):
            lines.append(
                f"      {i}. {r['backend']:>7} | {r['method']:>7} | {r['algorithm']:<22} | "
                f"{r['metric']:<22} z={r['best_z_mm']:.4f}mm  "
                f"Δ={r['_dist']:.4f}mm  t={r['elapsed_s']:.3f}s"
            )
        lines.append("")

    # ═══ GRAND SUMMARY ═══
    all_valid = [r for r in results if r.get("best_z_mm") is not None]
    lines.append(f"{'='*120}")
    lines.append(f"  GRAND SUMMARY")
    lines.append(f"{'='*120}")
    lines.append(f"  Total runs:    {len(results)}")
    lines.append(f"  Successful:    {len(all_valid)}")
    lines.append(f"  Failed:        {len(results) - len(all_valid)}")
    lines.append("")

    # Speed ranking per backend (overall)
    lines.append(f"  Overall median speed per backend:")
    for bn, _ in backends:
        bn_r = [r for r in all_valid if r["backend"] == bn.value]
        if bn_r:
            med = float(np.median([r["elapsed_s"] for r in bn_r]))
            lines.append(f"    {bn.value:<10} {med:.3f}s median")
    lines.append("")

    # Speed ranking per algorithm (overall)
    lines.append(f"  Overall median speed per algorithm:")
    for algo in algorithms:
        ar = [r for r in all_valid if r["algorithm"] == algo]
        if ar:
            med = float(np.median([r["elapsed_s"] for r in ar]))
            lines.append(f"    {algo:<24} {med:.3f}s median")
    lines.append("")

    # Most reliable metric (lowest z variance across algorithms)
    lines.append(f"  Metric reliability (lower std = more consistent across algorithms):")
    for m in ALL_METRICS:
        mr = [r for r in all_valid if r["metric"] == m.value]
        if len(mr) >= 2:
            std = float(np.std([r["best_z_mm"] for r in mr]))
            lines.append(f"    {m.value:<24} std={std:.4f}mm  (n={len(mr)})")
    lines.append("")
    lines.append(f"{'='*120}")
    lines.append(f"  END OF REPORT")
    lines.append(f"{'='*120}")

    report_text = "\n".join(lines)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport written to: {out_path}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
