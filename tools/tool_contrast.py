"""
Tool: Kontrast İyileştirme Testi
==================================
Percentile stretch, CLAHE, histogram eşitleme yöntemleri.

Kullanım:
    python -m tools.tool_contrast
    python -m tools.tool_contrast --image path/to/hologram.png
"""
import argparse
import numpy as np

from tools._common import (
    SRC, load_sample, extract_field, section, ok, fail, info, Timer,
    DEFAULT_WAVELENGTH_M, DEFAULT_PIXEL_SIZE_M, DEFAULT_N_MEDIUM
)
import sys
sys.path.insert(0, str(SRC))


def test_contrast_methods(amplitude):
    """Test all contrast enhancement methods."""
    from core.contrast import apply_contrast, ContrastMethod

    section("1. Kontrast Yöntemleri")

    info(f"Giriş — shape: {amplitude.shape}, range: [{amplitude.min():.4f}, {amplitude.max():.4f}]")

    for method in ContrastMethod:
        with Timer(f"{method.value}"):
            result = apply_contrast(amplitude, method)
        ok(f"{method.value:20s}: range=[{result.min():.4f}, {result.max():.4f}], "
           f"mean={result.mean():.4f}, std={result.std():.4f}")

        # Verify output is in [0, 1]
        if result.min() >= -0.01 and result.max() <= 1.01:
            ok(f"  → [0, 1] aralığında")
        else:
            fail(f"  → [0, 1] aralığı dışında!")


def test_percentile_params(amplitude):
    """Test percentile stretch with different parameters."""
    from core.contrast import apply_contrast, ContrastMethod

    section("2. Percentile Stretch Parametreleri")

    params = [
        (0.5, 99.5),
        (1.0, 99.0),
        (2.0, 98.0),
        (5.0, 95.0),
        (10.0, 90.0),
    ]

    for p_low, p_high in params:
        with Timer(f"p=[{p_low}, {p_high}]"):
            result = apply_contrast(amplitude, ContrastMethod.PERCENTILE,
                                    p_low=p_low, p_high=p_high)
        ok(f"p=[{p_low:4.1f}, {p_high:4.1f}]: std={result.std():.4f}, "
           f"entropy={_entropy(result):.4f}")


def test_clahe_params(amplitude):
    """Test CLAHE with different clip limits and grid sizes."""
    from core.contrast import apply_contrast, ContrastMethod

    section("3. CLAHE Parametreleri")

    configs = [
        (1.0, 8),
        (2.0, 8),
        (4.0, 8),
        (2.0, 4),
        (2.0, 16),
    ]

    for clip, grid in configs:
        with Timer(f"clip={clip}, grid={grid}"):
            result = apply_contrast(amplitude, ContrastMethod.CLAHE,
                                    clahe_clip=clip, clahe_grid=grid)
        ok(f"clip={clip:.1f}, grid={grid:2d}: std={result.std():.4f}, "
           f"entropy={_entropy(result):.4f}")


def test_on_phase(recon_complex):
    """Test contrast enhancement on phase images."""
    from core.contrast import apply_contrast, ContrastMethod

    section("4. Faz Görüntüsünde Kontrast")

    phase = np.angle(recon_complex)
    info(f"Phase range: [{phase.min():.4f}, {phase.max():.4f}]")

    for method in [ContrastMethod.NONE, ContrastMethod.PERCENTILE, ContrastMethod.HISTOGRAM_EQ]:
        with Timer(f"Phase + {method.value}"):
            result = apply_contrast(phase, method)
        ok(f"{method.value:20s}: range=[{result.min():.4f}, {result.max():.4f}], std={result.std():.4f}")


def _entropy(img):
    """Compute image entropy."""
    hist, _ = np.histogram(img.ravel(), bins=256, density=False)
    hist = hist[hist > 0].astype(np.float64)
    p = hist / hist.sum()
    return -float(np.sum(p * np.log2(p)))


def main():
    parser = argparse.ArgumentParser(description="Contrast Enhancement Test Tool")
    parser.add_argument("--image", help="Hologram dosyası")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: Kontrast İyileştirme            ║")
    print("╚══════════════════════════════════════════╝")

    img, path = load_sample(args.image)
    info(f"Kaynak: {path.name} ({img.shape})")

    field, center, _, _ = extract_field(img)

    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=-1e-3,
        n=DEFAULT_N_MEDIUM,
    )
    recon = np.asarray(propagate(field, params, ReconstructionMethod.ASM))
    amplitude = np.abs(recon)

    test_contrast_methods(amplitude)
    test_percentile_params(amplitude)
    test_clahe_params(amplitude)
    test_on_phase(recon)

    print("\n✅ Kontrast testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
