"""
Tool: Propagasyon (Reconstruction) Testi
==========================================
ASM ve Fresnel yöntemiyle dalga yayılımı, farklı z mesafeleri.

Kullanım:
    python -m tools.tool_reconstruction
    python -m tools.tool_reconstruction --image path/to/hologram.png --z_mm -1.0
"""
import argparse
import numpy as np

from tools._common import (
    SRC, load_sample, extract_field, section, ok, fail, info, Timer,
    DEFAULT_WAVELENGTH_M, DEFAULT_PIXEL_SIZE_M, DEFAULT_N_MEDIUM, DEFAULT_MASK_RADIUS
)
import sys
sys.path.insert(0, str(SRC))


def test_asm_propagation(field, z_mm):
    """Test ASM propagation at a given z."""
    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod

    section("1. ASM Propagasyon")

    z_m = z_mm * 1e-3
    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=z_m,
        n=DEFAULT_N_MEDIUM,
    )

    with Timer(f"ASM z={z_mm:.3f} mm"):
        result = propagate(field, params, ReconstructionMethod.ASM)

    result = np.asarray(result)
    amp = np.abs(result)
    phase = np.angle(result)

    ok(f"Shape: {result.shape}, dtype: {result.dtype}")
    ok(f"Amplitude — min: {amp.min():.6f}, max: {amp.max():.6f}, mean: {amp.mean():.6f}")
    ok(f"Phase — min: {phase.min():.4f}, max: {phase.max():.4f}")

    if np.all(np.isfinite(result)):
        ok("NaN/Inf yok")
    else:
        fail(f"NaN/Inf var: {np.sum(~np.isfinite(result))}")

    return result


def test_fresnel_propagation(field, z_mm):
    """Test Fresnel propagation at a given z."""
    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod

    section("2. Fresnel Propagasyon")

    z_m = z_mm * 1e-3
    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=z_m,
        n=DEFAULT_N_MEDIUM,
    )

    with Timer(f"Fresnel z={z_mm:.3f} mm"):
        result = propagate(field, params, ReconstructionMethod.FRESNEL)

    result = np.asarray(result)
    amp = np.abs(result)

    ok(f"Shape: {result.shape}, dtype: {result.dtype}")
    ok(f"Amplitude — min: {amp.min():.6f}, max: {amp.max():.6f}, mean: {amp.mean():.6f}")

    return result


def test_z_sweep(field):
    """Test propagation across multiple z values."""
    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod

    section("3. Z Taraması")

    z_values_mm = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
    results = []

    for z_mm in z_values_mm:
        z_m = z_mm * 1e-3
        params = ReconstructionParams(
            wavelength_m=DEFAULT_WAVELENGTH_M,
            pixel_size_m=DEFAULT_PIXEL_SIZE_M,
            z_m=z_m,
            n=DEFAULT_N_MEDIUM,
        )
        with Timer(f"z={z_mm:+.1f} mm"):
            result = propagate(field, params, ReconstructionMethod.ASM)
        amp = np.abs(np.asarray(result))
        results.append((z_mm, float(np.std(amp)), float(np.mean(amp))))

    info("\n  Z (mm)    | Amp Std    | Amp Mean")
    info("  ----------|------------|----------")
    for z, std, mean in results:
        info(f"  {z:+7.2f}   | {std:.6f}  | {mean:.6f}")


def test_cached_reconstructor(field):
    """Test CachedReconstructor performance."""
    from core.reconstruction import CachedReconstructor, ReconstructionParams, ReconstructionMethod
    from core.fft_backend import get_best_fft_backend

    section("4. CachedReconstructor Performans")

    fft = get_best_fft_backend()
    recon = CachedReconstructor(field.shape, ReconstructionMethod.ASM, fft)

    # First call (cold cache)
    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=-1e-3,
        n=DEFAULT_N_MEDIUM,
    )
    with Timer("İlk çağrı (soğuk cache)"):
        _ = recon.reconstruct(field, params)

    # Second call same z (warm cache)
    with Timer("Aynı z (sıcak cache)"):
        _ = recon.reconstruct(field, params)

    # Different z (H recompute, spectrum cached)
    params2 = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=-2e-3,
        n=DEFAULT_N_MEDIUM,
    )
    with Timer("Farklı z (H yeniden hesapla)"):
        _ = recon.reconstruct(field, params2)

    # Batch: 20 different z values
    with Timer("20 farklı z (batch)"):
        for z in np.linspace(-5e-3, 5e-3, 20):
            p = ReconstructionParams(
                wavelength_m=DEFAULT_WAVELENGTH_M,
                pixel_size_m=DEFAULT_PIXEL_SIZE_M,
                z_m=z, n=DEFAULT_N_MEDIUM,
            )
            recon.reconstruct(field, p)
    ok("Batch propagasyon başarılı (20 z)")


def test_asm_vs_fresnel(field):
    """Compare ASM vs Fresnel results."""
    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod

    section("5. ASM vs Fresnel Karşılaştırma")

    z_m = -1e-3
    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=z_m,
        n=DEFAULT_N_MEDIUM,
    )

    r_asm = np.asarray(propagate(field, params, ReconstructionMethod.ASM))
    r_fre = np.asarray(propagate(field, params, ReconstructionMethod.FRESNEL))

    amp_diff = np.abs(np.abs(r_asm) - np.abs(r_fre))
    phase_diff = np.abs(np.angle(r_asm) - np.angle(r_fre))

    ok(f"Amplitude farkı — mean: {amp_diff.mean():.6f}, max: {amp_diff.max():.6f}")
    ok(f"Phase farkı — mean: {phase_diff.mean():.4f}, max: {phase_diff.max():.4f}")

    corr = np.corrcoef(np.abs(r_asm).ravel(), np.abs(r_fre).ravel())[0, 1]
    ok(f"Amplitude korelasyon: {corr:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruction Test Tool")
    parser.add_argument("--image", help="Hologram dosyası")
    parser.add_argument("--z_mm", type=float, default=-1.0, help="Propagasyon mesafesi (mm)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: Propagasyon (Reconstruction)     ║")
    print("╚══════════════════════════════════════════╝")

    img, path = load_sample(args.image)
    info(f"Kaynak: {path.name} ({img.shape})")

    field, center, _, _ = extract_field(img)
    info(f"+1 order: {center}")

    test_asm_propagation(field, args.z_mm)
    test_fresnel_propagation(field, args.z_mm)
    test_z_sweep(field)
    test_cached_reconstructor(field)
    test_asm_vs_fresnel(field)

    print("\n✅ Propagasyon testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
