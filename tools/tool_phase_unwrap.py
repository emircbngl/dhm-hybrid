"""
Tool: Faz Açma (Phase Unwrapping) Testi
=========================================
Gradient Integration, TIE, Quality Guided, Least Squares, Goldstein yöntemleri.

Kullanım:
    python -m tools.tool_phase_unwrap
    python -m tools.tool_phase_unwrap --image path/to/hologram.png --z_mm -1.0
"""
import argparse
import numpy as np

from tools._common import (
    SRC, load_sample, extract_field, section, ok, fail, info, Timer,
    DEFAULT_WAVELENGTH_M, DEFAULT_PIXEL_SIZE_M, DEFAULT_N_MEDIUM
)
import sys
sys.path.insert(0, str(SRC))


def _reconstruct_at_z(field, z_mm):
    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=z_mm * 1e-3,
        n=DEFAULT_N_MEDIUM,
    )
    return np.asarray(propagate(field, params, ReconstructionMethod.ASM))


def test_all_unwrap_methods(recon_complex):
    """Test all phase unwrapping methods."""
    from core.phase_unwrap import unwrap_phase_advanced, UnwrapConfig, UnwrapMethod

    section("1. Tüm Faz Açma Yöntemleri")

    wrapped = np.angle(recon_complex)
    info(f"Wrapped phase — min: {wrapped.min():.4f}, max: {wrapped.max():.4f}")
    info(f"Shape: {wrapped.shape}\n")

    methods = [
        UnwrapMethod.GRADIENT_INTEGRATION,
        UnwrapMethod.TIE,
        UnwrapMethod.THIN_SAMPLE,
        UnwrapMethod.LEAST_SQUARES,
        UnwrapMethod.QUALITY_GUIDED,
        UnwrapMethod.GOLDSTEIN,
    ]

    results = {}
    for method in methods:
        config = UnwrapConfig(method=method)
        try:
            with Timer(f"{method.value}"):
                unwrapped = unwrap_phase_advanced(
                    wrapped, config=config,
                    complex_field=recon_complex,
                    wavelength_m=DEFAULT_WAVELENGTH_M,
                    pixel_size_m=DEFAULT_PIXEL_SIZE_M,
                )

            if np.all(np.isfinite(unwrapped)):
                rng = float(unwrapped.max() - unwrapped.min())
                ok(f"{method.value:25s}: range={rng:.2f} rad, "
                   f"mean={unwrapped.mean():.4f}, std={unwrapped.std():.4f}")
                results[method.value] = unwrapped
            else:
                n_bad = int(np.sum(~np.isfinite(unwrapped)))
                fail(f"{method.value:25s}: {n_bad} NaN/Inf değer")
        except Exception as e:
            fail(f"{method.value:25s}: HATA — {e}")

    return results


def test_preprocessing_options(recon_complex):
    """Test pre-processing options (median filter, gaussian smooth)."""
    from core.phase_unwrap import unwrap_phase_advanced, UnwrapConfig, UnwrapMethod

    section("2. Ön-işleme Seçenekleri")

    wrapped = np.angle(recon_complex)
    base_method = UnwrapMethod.QUALITY_GUIDED

    configs = [
        ("Varsayılan", UnwrapConfig(method=base_method)),
        ("Median 3×3", UnwrapConfig(method=base_method, pre_median=True, pre_median_size=3)),
        ("Median 5×5", UnwrapConfig(method=base_method, pre_median=True, pre_median_size=5)),
        ("Gaussian σ=0.5", UnwrapConfig(method=base_method, pre_gaussian=True, pre_gaussian_sigma=0.5)),
        ("Gaussian σ=1.0", UnwrapConfig(method=base_method, pre_gaussian=True, pre_gaussian_sigma=1.0)),
        ("Median + Gaussian", UnwrapConfig(method=base_method, pre_median=True, pre_gaussian=True)),
    ]

    for name, config in configs:
        with Timer(name):
            unwrapped = unwrap_phase_advanced(
                wrapped, config=config,
                complex_field=recon_complex,
                wavelength_m=DEFAULT_WAVELENGTH_M,
                pixel_size_m=DEFAULT_PIXEL_SIZE_M,
            )
        rng = float(unwrapped.max() - unwrapped.min())
        ok(f"{name:20s}: range={rng:.2f} rad, std={unwrapped.std():.4f}")


def test_postprocessing_options(recon_complex):
    """Test post-processing options (outlier clip, bg removal)."""
    from core.phase_unwrap import unwrap_phase_advanced, UnwrapConfig, UnwrapMethod

    section("3. Son-işleme Seçenekleri")

    wrapped = np.angle(recon_complex)

    configs = [
        ("BG yok", UnwrapConfig(method=UnwrapMethod.QUALITY_GUIDED, post_bg_remove=False)),
        ("BG order=1", UnwrapConfig(method=UnwrapMethod.QUALITY_GUIDED, post_bg_remove=True, post_bg_order=1)),
        ("BG order=2", UnwrapConfig(method=UnwrapMethod.QUALITY_GUIDED, post_bg_remove=True, post_bg_order=2)),
        ("BG order=3", UnwrapConfig(method=UnwrapMethod.QUALITY_GUIDED, post_bg_remove=True, post_bg_order=3)),
        ("Outlier clip σ=3", UnwrapConfig(method=UnwrapMethod.QUALITY_GUIDED, post_outlier_clip=True, post_outlier_sigma=3.0)),
    ]

    for name, config in configs:
        with Timer(name):
            unwrapped = unwrap_phase_advanced(
                wrapped, config=config,
                complex_field=recon_complex,
                wavelength_m=DEFAULT_WAVELENGTH_M,
                pixel_size_m=DEFAULT_PIXEL_SIZE_M,
            )
        rng = float(unwrapped.max() - unwrapped.min())
        ok(f"{name:20s}: range={rng:.2f} rad, std={unwrapped.std():.4f}")


def test_method_comparison(recon_complex):
    """Compare unwrap methods by OPD range (useful for bead height estimation)."""
    from core.phase_unwrap import unwrap_phase_advanced, UnwrapConfig, UnwrapMethod

    section("4. OPD Karşılaştırma (yükseklik tahmini)")

    wrapped = np.angle(recon_complex)
    wl = DEFAULT_WAVELENGTH_M

    methods = [
        UnwrapMethod.GRADIENT_INTEGRATION,
        UnwrapMethod.QUALITY_GUIDED,
        UnwrapMethod.LEAST_SQUARES,
        UnwrapMethod.GOLDSTEIN,
    ]

    info(f"\n  {'Yöntem':25s} | Phase Range | OPD Range (nm) | OPD p2p (nm)")
    info(f"  {'-'*25}-|------------|----------------|-------------")

    for method in methods:
        config = UnwrapConfig(method=method, post_bg_remove=True, post_bg_order=2)
        try:
            unwrapped = unwrap_phase_advanced(
                wrapped, config=config,
                complex_field=recon_complex,
                wavelength_m=wl,
                pixel_size_m=DEFAULT_PIXEL_SIZE_M,
            )
            opd_m = unwrapped * wl / (2 * np.pi)
            opd_nm = opd_m * 1e9
            rng = float(unwrapped.max() - unwrapped.min())
            opd_rng = float(opd_nm.max() - opd_nm.min())
            p2p = float(np.percentile(opd_nm, 99) - np.percentile(opd_nm, 1))
            info(f"  {method.value:25s} | {rng:10.2f} | {opd_rng:14.1f} | {p2p:11.1f}")
        except Exception as e:
            info(f"  {method.value:25s} | HATA: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase Unwrap Test Tool")
    parser.add_argument("--image", help="Hologram dosyası")
    parser.add_argument("--z_mm", type=float, default=-1.0, help="Propagasyon mesafesi (mm)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: Faz Açma (Phase Unwrapping)     ║")
    print("╚══════════════════════════════════════════╝")

    img, path = load_sample(args.image)
    info(f"Kaynak: {path.name} ({img.shape})")

    field, center, _, _ = extract_field(img)
    info(f"+1 order: {center}")

    recon_complex = _reconstruct_at_z(field, args.z_mm)
    info(f"Propagasyon: z = {args.z_mm} mm\n")

    test_all_unwrap_methods(recon_complex)
    test_preprocessing_options(recon_complex)
    test_postprocessing_options(recon_complex)
    test_method_comparison(recon_complex)

    print("\n✅ Phase unwrap testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
