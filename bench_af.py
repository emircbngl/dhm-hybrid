#!/usr/bin/env python3
"""
Autofocus benchmark script.
Loads the test hologram with 50x profile parameters,
runs various autofocus configurations, and reports timing + accuracy.

Usage:
    python bench_af.py                  # full benchmark
    python bench_af.py --ref-only       # only find reference z with linear 480-step
    python bench_af.py --quick          # quick comparison (fewer algos)
"""
import sys, time, argparse
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from core.reconstruction import ReconstructionParams, ReconstructionMethod, propagate
from core.fft_backend import get_best_fft_backend
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.autofocus import (
    FocusMetric, autofocus_zscan, coarse_to_fine_search,
    robust_coarse_to_fine_search, adaptive_gradient_search,
    adaptive_ratio_search, adaptive_bracketing_search,
    adaptive_distance_search, downsample_complex_field,
)

# ─── 50x Profile Parameters ───
WAVELENGTH_M = 632.8e-9
PIXEL_SIZE_M = 4.4e-6  # effective
MASK_RADIUS = 80
METHOD = ReconstructionMethod.ASM
Z_MIN_M = 0.0
Z_MAX_M = 120e-3  # 0–120 mm
IMAGE_PATH = ROOT / "labtest" / "ilk_imgbackCCD_24.10mm.png"


def load_and_preprocess():
    """Load image and extract complex field exactly as the GUI does."""
    from PIL import Image
    img = np.array(Image.open(IMAGE_PATH)).astype(np.float32)
    if img.ndim == 3:
        img = img[..., 0]

    # subtract mean (as in 50x profile)
    img = img - float(np.mean(img))

    # normalize
    mx = np.max(np.abs(img))
    if mx > 0:
        img = img / mx

    offaxis_params = OffAxisParams(radius=MASK_RADIUS)
    fc, _, _, _ = extract_complex_field_offaxis_debug(img, offaxis_params)
    return fc


def run_reference(fc, params, steps=480):
    """Run linear sweep with Laplacian Var to get the ground-truth z."""
    metric = FocusMetric.LAPLACIAN_VARIANCE
    z_arr = list(np.linspace(Z_MIN_M, Z_MAX_M, steps))

    print(f"\n{'='*60}")
    print(f"REFERENCE: Linear Sweep, Laplacian Var, {steps} steps")
    print(f"  Z range: {Z_MIN_M*1e3:.1f} – {Z_MAX_M*1e3:.1f} mm")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    res = autofocus_zscan(fc, params, z_arr, METHOD, metric)
    elapsed = time.perf_counter() - t0

    best_score = res.scores.get(res.best_z_m, 0.0)
    print(f"  Best Z  = {res.best_z_m*1e3:.4f} mm")
    print(f"  Score   = {best_score:.6f}")
    print(f"  Time    = {elapsed:.2f} s")
    print(f"  Per eval= {elapsed/steps*1000:.1f} ms")
    return res.best_z_m, elapsed


def run_single(name, func, ref_z):
    """Run a single benchmark and report results."""
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0

    best_z = result.best_z_m
    # Handle different result types
    if hasattr(result, 'evaluations'):
        evals = result.evaluations
    elif hasattr(result, 'total_evaluations'):
        evals = result.total_evaluations
    elif hasattr(result, 'scores'):
        evals = len(result.scores)
    else:
        evals = 0
    error_um = abs(best_z - ref_z) * 1e6  # error in µm

    print(f"  {name:<40} z={best_z*1e3:>10.4f}mm  "
          f"err={error_um:>8.1f}µm  "
          f"evals={str(evals):>5}  "
          f"time={elapsed:>6.2f}s  "
          f"per_eval={elapsed/max(int(evals),1)*1000:>5.1f}ms")
    return best_z, elapsed, int(evals), error_um


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-only", action="store_true", help="Only find reference z")
    parser.add_argument("--quick", action="store_true", help="Quick comparison")
    parser.add_argument("--ref-steps", type=int, default=480, help="Reference linear steps")
    parser.add_argument("--all-metrics", action="store_true", help="Validate DS×2 across all metrics")
    args = parser.parse_args()

    print(f"Loading image: {IMAGE_PATH}")
    fc = load_and_preprocess()
    print(f"  Field shape: {fc.shape}, dtype: {fc.dtype}")

    params = ReconstructionParams(
        wavelength_m=WAVELENGTH_M,
        pixel_size_m=PIXEL_SIZE_M,
        z_m=0.0,
        n=1.0,
    )

    # Check FFT backend
    fft = get_best_fft_backend()
    print(f"  FFT backend: {fft.name}")

    # ─── Reference ───
    ref_z, ref_time = run_reference(fc, params, steps=args.ref_steps)

    if args.ref_only:
        return

    metric = FocusMetric.LAPLACIAN_VARIANCE

    # ─── Benchmark all algorithms ───
    print(f"\n{'='*60}")
    print(f"BENCHMARK (ref z = {ref_z*1e3:.4f} mm)")
    print(f"{'='*60}")

    configs = []

    # Linear sweeps at different step counts
    for steps in [51, 100, 200]:
        z_arr = list(np.linspace(Z_MIN_M, Z_MAX_M, steps))
        configs.append((
            f"Linear {steps}",
            lambda z=z_arr: autofocus_zscan(fc, params, z, METHOD, metric),
        ))

    # Golden Section
    configs.append((
        "Coarse-to-Fine (Golden) 51",
        lambda: coarse_to_fine_search(fc, params, METHOD, metric, Z_MIN_M, Z_MAX_M, coarse_steps=51),
    ))

    # Robust Coarse-to-Fine
    configs.append((
        "Robust C2F 51",
        lambda: robust_coarse_to_fine_search(fc, params, METHOD, metric, Z_MIN_M, Z_MAX_M, n_coarse=51),
    ))

    if not args.quick:
        # Adaptive Gradient
        configs.append((
            "Adaptive Gradient (auto budget)",
            lambda: adaptive_gradient_search(fc, params, METHOD, metric, Z_MIN_M, Z_MAX_M, max_evaluations=100),
        ))

        # Adaptive Ratio
        configs.append((
            "Adaptive Ratio (auto budget)",
            lambda: adaptive_ratio_search(fc, params, METHOD, metric, Z_MIN_M, Z_MAX_M, max_evaluations=100),
        ))

        # Adaptive Bracketing
        configs.append((
            "Adaptive Bracketing (smooth)",
            lambda: adaptive_bracketing_search(
                fc, params, METHOD, metric, Z_MIN_M, Z_MAX_M,
                n_refine_levels=3, refine_divisions=6, smooth_sigma=1.0, max_evaluations=100,
            ),
        ))

        # Adaptive Distance (auto z-range)
        configs.append((
            "Adaptive Distance 100",
            lambda: adaptive_distance_search(
                fc, params, METHOD, metric, Z_MIN_M, Z_MAX_M,
                max_evaluations=100,
            ),
        ))
        configs.append((
            "Adaptive Distance 150",
            lambda: adaptive_distance_search(
                fc, params, METHOD, metric, Z_MIN_M, Z_MAX_M,
                max_evaluations=150,
            ),
        ))

    # ─── Downsampled benchmarks ───
    print(f"\n{'─'*60}")
    print(f"DOWNSAMPLED BENCHMARKS")
    print(f"{'─'*60}")

    for ds_factor in [2, 4]:
        ds_field, ds_params = downsample_complex_field(fc, params, factor=ds_factor)
        print(f"  [DS×{ds_factor}] Shape: {ds_field.shape}, pixel: {ds_params.pixel_size_m*1e6:.2f}µm")

        # Linear sweep at 200 steps (downsampled)
        z_arr_200 = list(np.linspace(Z_MIN_M, Z_MAX_M, 200))
        configs.append((
            f"DS×{ds_factor} Linear 200",
            lambda f=ds_field, p=ds_params: autofocus_zscan(f, p, list(np.linspace(Z_MIN_M, Z_MAX_M, 200)), METHOD, metric),
        ))

        # Adaptive Bracketing (downsampled)
        configs.append((
            f"DS×{ds_factor} Bracketing 100",
            lambda f=ds_field, p=ds_params: adaptive_bracketing_search(
                f, p, METHOD, metric, Z_MIN_M, Z_MAX_M,
                n_refine_levels=3, refine_divisions=6, smooth_sigma=1.0, max_evaluations=100,
            ),
        ))

        # Coarse-to-Fine (downsampled)
        configs.append((
            f"DS×{ds_factor} Golden 51",
            lambda f=ds_field, p=ds_params: coarse_to_fine_search(
                f, p, METHOD, metric, Z_MIN_M, Z_MAX_M, coarse_steps=51,
            ),
        ))

    # ─── Two-stage: DS×2 coarse → full-res refine ───
    print(f"\n{'─'*60}")
    print(f"TWO-STAGE BENCHMARKS (DS×2 coarse → full-res refine)")
    print(f"{'─'*60}")

    ds2_field, ds2_params = downsample_complex_field(fc, params, factor=2)

    def two_stage_golden(coarse_steps=51, fine_steps=51):
        # Stage 1: DS×2 coarse
        r1 = coarse_to_fine_search(
            ds2_field, ds2_params, METHOD, metric, Z_MIN_M, Z_MAX_M,
            coarse_steps=coarse_steps,
        )
        # Stage 2: full-res narrow refine around found z
        margin = (Z_MAX_M - Z_MIN_M) / coarse_steps * 2
        z_lo = max(Z_MIN_M, r1.best_z_m - margin)
        z_hi = min(Z_MAX_M, r1.best_z_m + margin)
        r2 = coarse_to_fine_search(
            fc, params, METHOD, metric, z_lo, z_hi,
            coarse_steps=fine_steps,
        )
        return r2

    def two_stage_bracket(coarse_evals=60, fine_evals=40):
        # Stage 1: DS×2 coarse bracketing
        r1 = adaptive_bracketing_search(
            ds2_field, ds2_params, METHOD, metric, Z_MIN_M, Z_MAX_M,
            n_refine_levels=2, refine_divisions=6, smooth_sigma=1.0,
            max_evaluations=coarse_evals,
        )
        # Stage 2: full-res narrow bracketing
        margin = (Z_MAX_M - Z_MIN_M) / 10 * 0.5
        z_lo = max(Z_MIN_M, r1.best_z_m - margin)
        z_hi = min(Z_MAX_M, r1.best_z_m + margin)
        r2 = adaptive_bracketing_search(
            fc, params, METHOD, metric, z_lo, z_hi,
            n_refine_levels=3, refine_divisions=6, smooth_sigma=1.0,
            max_evaluations=fine_evals,
        )
        return r2

    def two_stage_linear(coarse_steps=100, fine_steps=100):
        # Stage 1: DS×2 linear
        z_arr = list(np.linspace(Z_MIN_M, Z_MAX_M, coarse_steps))
        r1 = autofocus_zscan(ds2_field, ds2_params, z_arr, METHOD, metric)
        # Stage 2: full-res narrow
        step = (Z_MAX_M - Z_MIN_M) / coarse_steps
        z_lo = max(Z_MIN_M, r1.best_z_m - step * 2)
        z_hi = min(Z_MAX_M, r1.best_z_m + step * 2)
        z_arr2 = list(np.linspace(z_lo, z_hi, fine_steps))
        r2 = autofocus_zscan(fc, params, z_arr2, METHOD, metric)
        return r2

    configs.append(("2-Stage Golden (51+51)", lambda: two_stage_golden(51, 51)))
    configs.append(("2-Stage Bracket (60+40)", lambda: two_stage_bracket(60, 40)))
    configs.append(("2-Stage Linear (100+100)", lambda: two_stage_linear(100, 100)))

    results = []
    for name, func in configs:
        try:
            r = run_single(name, func, ref_z)
            results.append((name, *r))
        except Exception as e:
            print(f"  {name:<40} FAILED: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Reference z       = {ref_z*1e3:.4f} mm  ({ref_time:.2f}s, {args.ref_steps} evals)")
    print(f"  Field shape       = {fc.shape}")
    print(f"  Per-eval baseline = {ref_time/args.ref_steps*1000:.1f} ms")
    if results:
        best = min(results, key=lambda x: x[4])  # min error
        fastest = min(results, key=lambda x: x[2])  # min time
        print(f"  Most accurate     = {best[0]} (err={best[4]:.1f}µm)")
        print(f"  Fastest           = {fastest[0]} ({fastest[2]:.2f}s)")

    # ─── Multi-metric validation ───
    if args.all_metrics:
        ds2_f, ds2_p = downsample_complex_field(fc, params, factor=2)
        print(f"\n{'='*60}")
        print("MULTI-METRIC DS×2 VALIDATION (Golden 51)")
        print(f"{'='*60}")
        print(f"  {'Metric':<22} {'Full-res z':>12} {'DS×2 z':>12} {'Δ(µm)':>10} {'Full(s)':>8} {'DS×2(s)':>8} {'Speedup':>8}")
        for fm in FocusMetric:
            # full-res reference
            t0 = time.perf_counter()
            r_full = coarse_to_fine_search(fc, params, METHOD, fm, Z_MIN_M, Z_MAX_M, coarse_steps=51)
            t_full = time.perf_counter() - t0
            # DS×2
            t0 = time.perf_counter()
            r_ds = coarse_to_fine_search(ds2_f, ds2_p, METHOD, fm, Z_MIN_M, Z_MAX_M, coarse_steps=51)
            t_ds = time.perf_counter() - t0
            delta = abs(r_full.best_z_m - r_ds.best_z_m) * 1e6
            speedup = t_full / t_ds if t_ds > 0 else 0
            print(f"  {fm.value:<22} {r_full.best_z_m*1e3:>10.4f}mm {r_ds.best_z_m*1e3:>10.4f}mm {delta:>8.1f} {t_full:>7.2f}s {t_ds:>7.2f}s {speedup:>6.1f}×")


if __name__ == "__main__":
    main()
