"""
Tool: Otomatik Odaklama (Autofocus) Testi
===========================================
Z-scan, Golden Section, Coarse-to-Fine, Robust arama algoritmaları.

Kullanım:
    python -m tools.tool_autofocus
    python -m tools.tool_autofocus --image path/to/hologram.png --z_min -5 --z_max 5
"""
import argparse
import numpy as np

from tools._common import (
    SRC, load_sample, extract_field, section, ok, fail, info, Timer,
    DEFAULT_WAVELENGTH_M, DEFAULT_PIXEL_SIZE_M, DEFAULT_N_MEDIUM
)
import sys
sys.path.insert(0, str(SRC))


def _make_base_params():
    from core.reconstruction import ReconstructionParams
    return ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=0.0,
        n=DEFAULT_N_MEDIUM,
    )


def test_focus_metrics(field, z_m=-1e-3):
    """Evaluate all focus metrics at a fixed z."""
    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
    from core.autofocus import FocusMetric, _calc_metric

    section("1. Odak Metrikleri (tüm metrikler)")

    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=z_m,
        n=DEFAULT_N_MEDIUM,
    )
    recon = np.asarray(propagate(field, params, ReconstructionMethod.ASM))

    info(f"z = {z_m*1e3:.2f} mm'deki metrik değerleri:\n")
    for fm in FocusMetric:
        try:
            score = _calc_metric(recon, fm)
            ok(f"{fm.value:25s}: {score:.6f}")
        except Exception as e:
            fail(f"{fm.value:25s}: HATA — {e}")


def test_zscan(field, z_min_mm, z_max_mm):
    """Test linear Z-scan autofocus."""
    from core.reconstruction import ReconstructionMethod
    from core.autofocus import FocusMetric, autofocus_zscan

    section("2. Z-Scan (Linear Sweep)")

    base = _make_base_params()
    z_values = np.linspace(z_min_mm * 1e-3, z_max_mm * 1e-3, 41).tolist()

    for metric in [FocusMetric.PHASE_VARIANCE, FocusMetric.TENENGRAD, FocusMetric.LAPLACIAN_VARIANCE]:
        with Timer(f"Z-scan ({metric.value})"):
            result = autofocus_zscan(
                field, base, z_values, ReconstructionMethod.ASM, metric
            )
        ok(f"{metric.value:25s}: best_z = {result.best_z_m*1e3:.4f} mm")


def test_golden_section(field, z_min_mm, z_max_mm):
    """Test Golden Section Search."""
    from core.reconstruction import ReconstructionMethod
    from core.autofocus import FocusMetric, golden_section_search

    section("3. Golden Section Search")

    base = _make_base_params()

    for metric in [FocusMetric.PHASE_VARIANCE, FocusMetric.TENENGRAD]:
        with Timer(f"Golden ({metric.value})"):
            result = golden_section_search(
                field, base, ReconstructionMethod.ASM, metric,
                z_min_m=z_min_mm * 1e-3,
                z_max_m=z_max_mm * 1e-3,
                tolerance_m=1e-6,
            )
        ok(f"{metric.value:25s}: best_z = {result.best_z_m*1e3:.4f} mm, "
           f"evals = {result.evaluations}, iters = {result.iterations}")


def test_coarse_to_fine(field, z_min_mm, z_max_mm):
    """Test Coarse-to-Fine hybrid search."""
    from core.reconstruction import ReconstructionMethod
    from core.autofocus import FocusMetric, coarse_to_fine_search

    section("4. Coarse-to-Fine Search")

    base = _make_base_params()

    for metric in [FocusMetric.PHASE_VARIANCE, FocusMetric.TENENGRAD]:
        with Timer(f"C2F ({metric.value})"):
            result = coarse_to_fine_search(
                field, base, ReconstructionMethod.ASM, metric,
                z_min_m=z_min_mm * 1e-3,
                z_max_m=z_max_mm * 1e-3,
                coarse_steps=21,
                fine_tolerance_m=1e-6,
            )
        ok(f"{metric.value:25s}: best_z = {result.best_z_m*1e3:.4f} mm, "
           f"evals = {result.evaluations}")


def test_robust_search(field, z_min_mm, z_max_mm):
    """Test Robust Coarse-to-Fine search (smoothing-based)."""
    from core.reconstruction import ReconstructionMethod
    from core.autofocus import FocusMetric, robust_coarse_to_fine_search

    section("5. Robust Coarse-to-Fine Search")

    base = _make_base_params()

    for metric in [FocusMetric.PHASE_VARIANCE, FocusMetric.TENENGRAD]:
        with Timer(f"Robust ({metric.value})"):
            result = robust_coarse_to_fine_search(
                field, base, ReconstructionMethod.ASM, metric,
                z_min_m=z_min_mm * 1e-3,
                z_max_m=z_max_mm * 1e-3,
                n_coarse=40,
            )
        ok(f"{metric.value:25s}: best_z = {result.best_z_m*1e3:.4f} mm, "
           f"evals = {result.total_evaluations}")


def test_metric_landscape(field, z_min_mm, z_max_mm):
    """Scan metric landscape across all metrics."""
    from core.reconstruction import ReconstructionMethod
    from core.autofocus import scan_metric_landscape, FocusMetric

    section("6. Metrik Manzarası")

    base = _make_base_params()

    with Timer("Landscape scan (tüm metrikler)"):
        result = scan_metric_landscape(
            field, base, ReconstructionMethod.ASM,
            z_min_m=z_min_mm * 1e-3,
            z_max_m=z_max_mm * 1e-3,
            n_steps=41,
        )

    info(f"\n  {'Metrik':25s} | Peak Z (mm) | # Peaks")
    info(f"  {'-'*25}-|-------------|--------")
    for fm in FocusMetric:
        pz = result.peak_z.get(fm, 0)
        np_ = result.n_peaks.get(fm, 0)
        info(f"  {fm.value:25s} | {pz*1e3:+9.4f}   | {np_}")


def test_downsampled_autofocus(field, z_min_mm, z_max_mm):
    """Test autofocus with downsampled field."""
    from core.reconstruction import ReconstructionMethod
    from core.autofocus import (
        FocusMetric, robust_coarse_to_fine_search, downsample_complex_field
    )

    section("7. Downsampled Autofocus")

    base = _make_base_params()

    for factor in [1, 2, 4]:
        ds_field, ds_params = downsample_complex_field(field, base, factor)
        with Timer(f"factor={factor} ({ds_field.shape})"):
            result = robust_coarse_to_fine_search(
                ds_field, ds_params, ReconstructionMethod.ASM,
                FocusMetric.PHASE_VARIANCE,
                z_min_m=z_min_mm * 1e-3,
                z_max_m=z_max_mm * 1e-3,
                n_coarse=30,
            )
        ok(f"factor={factor}: best_z = {result.best_z_m*1e3:.4f} mm, shape={ds_field.shape}")


def main():
    parser = argparse.ArgumentParser(description="Autofocus Test Tool")
    parser.add_argument("--image", help="Hologram dosyası")
    parser.add_argument("--z_min", type=float, default=-5.0, help="Min z (mm)")
    parser.add_argument("--z_max", type=float, default=5.0, help="Max z (mm)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: Otomatik Odaklama (Autofocus)   ║")
    print("╚══════════════════════════════════════════╝")

    img, path = load_sample(args.image)
    info(f"Kaynak: {path.name} ({img.shape})")

    field, center, _, _ = extract_field(img)
    info(f"+1 order: {center}")

    test_focus_metrics(field)
    test_zscan(field, args.z_min, args.z_max)
    test_golden_section(field, args.z_min, args.z_max)
    test_coarse_to_fine(field, args.z_min, args.z_max)
    test_robust_search(field, args.z_min, args.z_max)
    test_metric_landscape(field, args.z_min, args.z_max)
    test_downsampled_autofocus(field, args.z_min, args.z_max)

    print("\n✅ Autofocus testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
