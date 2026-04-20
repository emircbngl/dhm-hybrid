"""
Tool: Off-axis Çıkarım Testi
==============================
Hologramdan +1 order tespiti, spektral maskeleme ve karmaşık alan çıkarımı.

Kullanım:
    python -m tools.tool_offaxis
    python -m tools.tool_offaxis --image path/to/hologram.png --radius 80
"""
import argparse
import numpy as np

from tools._common import (
    SRC, load_sample, extract_field, section, ok, fail, info, Timer,
    DEFAULT_MASK_RADIUS
)
import sys
sys.path.insert(0, str(SRC))


def test_order_detection(img):
    """Test +1 order automatic detection."""
    from core.masking import detect_plus_one_order

    section("1. +1 Order Tespiti")

    with Timer("+1 order detection"):
        center = detect_plus_one_order(img, exclusion_radius=20)

    ok(f"Tespit edilen merkez (y, x): {center}")
    ok(f"Görüntü boyutu: {img.shape}")

    # Sanity check
    ny, nx = img.shape
    if 0 < center[0] < ny and 0 < center[1] < nx:
        ok("Merkez görüntü sınırları içinde")
    else:
        fail("Merkez görüntü dışında!")

    dc_y, dc_x = ny // 2, nx // 2
    dist = np.sqrt((center[0] - dc_y)**2 + (center[1] - dc_x)**2)
    info(f"DC'den uzaklık: {dist:.1f} piksel")

    return center


def test_spectral_mask(img, radius):
    """Test spectral masking with different apodization types."""
    from core.masking import circular_mask

    section("2. Spektral Maske Tipleri")

    center = (img.shape[0] // 4, img.shape[1] // 4)  # Dummy center

    for apod in ["none", "hann", "tukey", "gaussian"]:
        with Timer(f"mask ({apod})"):
            mask = circular_mask(img.shape, center, radius,
                                 apodization=apod, rolloff=0.25)
        nonzero = np.count_nonzero(mask > 0)
        ok(f"{apod:10s}: nonzero={nonzero:,}, max={mask.max():.3f}, min(>0)={mask[mask>0].min():.3f}")


def test_extraction(img, radius):
    """Test full off-axis extraction pipeline."""
    section("3. Karmaşık Alan Çıkarımı")

    with Timer("Off-axis extraction"):
        field, center, spectrum, mask = extract_field(img, radius)

    ok(f"Karmaşık alan shape: {field.shape}, dtype: {field.dtype}")
    ok(f"+1 order merkez: {center}")
    ok(f"Amplitude — min: {np.abs(field).min():.6f}, max: {np.abs(field).max():.6f}")
    ok(f"Phase — min: {np.angle(field).min():.4f} rad, max: {np.angle(field).max():.4f} rad")
    ok(f"Spektrum shape: {spectrum.shape}")
    ok(f"Maske nonzero: {np.count_nonzero(mask > 0):,}")

    # NaN/Inf check
    if np.all(np.isfinite(field)):
        ok("Karmaşık alanda NaN/Inf yok")
    else:
        fail(f"NaN/Inf bulundu: {np.sum(~np.isfinite(field))}")

    return field


def test_different_radii(img):
    """Test extraction with different mask radii."""
    section("4. Farklı Maske Yarıçapları")

    for r in [40, 60, 80, 100, 120]:
        with Timer(f"r={r}"):
            field, center, _, _ = extract_field(img, r)
        amp_std = np.std(np.abs(field))
        ok(f"r={r:3d}: amp_std={amp_std:.6f}, center={center}")


def main():
    parser = argparse.ArgumentParser(description="Off-axis Extraction Test Tool")
    parser.add_argument("--image", help="Hologram dosyası")
    parser.add_argument("--radius", type=int, default=DEFAULT_MASK_RADIUS, help="Maske yarıçapı")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: Off-axis Çıkarım                ║")
    print("╚══════════════════════════════════════════╝")

    img, path = load_sample(args.image)
    info(f"Kaynak: {path.name} ({img.shape})")

    test_order_detection(img)
    test_spectral_mask(img, args.radius)
    field = test_extraction(img, args.radius)
    test_different_radii(img)

    print("\n✅ Off-axis testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
