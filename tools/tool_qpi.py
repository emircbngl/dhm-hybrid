"""
Tool: QPI Pipeline Testi
=========================
OPD, yükseklik, kuru kütle, pürüzlülük, step height hesaplama.

Kullanım:
    python -m tools.tool_qpi
    python -m tools.tool_qpi --image path/to/hologram.png --z_mm -1.0
"""
import argparse
import numpy as np

from tools._common import (
    SRC, load_sample, extract_field, section, ok, fail, info, Timer,
    DEFAULT_WAVELENGTH_M, DEFAULT_PIXEL_SIZE_M, DEFAULT_N_MEDIUM
)
import sys
sys.path.insert(0, str(SRC))


def _get_unwrapped_phase(field, z_mm):
    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
    from core.phase_unwrap import unwrap_phase_advanced

    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=z_mm * 1e-3,
        n=DEFAULT_N_MEDIUM,
    )
    recon = np.asarray(propagate(field, params, ReconstructionMethod.ASM))
    wrapped = np.angle(recon)
    unwrapped = unwrap_phase_advanced(
        wrapped, complex_field=recon,
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
    )
    return unwrapped, recon


def test_opd_computation(unwrapped):
    """Test OPD (Optical Path Difference) computation."""
    from core.qpi import phase_to_opd

    section("1. OPD Hesaplama")

    with Timer("phase_to_opd"):
        opd_m = phase_to_opd(unwrapped, DEFAULT_WAVELENGTH_M)

    opd_nm = opd_m * 1e9
    ok(f"OPD range: {opd_nm.min():.1f} — {opd_nm.max():.1f} nm")
    ok(f"OPD mean: {opd_nm.mean():.1f} nm, std: {opd_nm.std():.1f} nm")
    ok(f"OPD p2p (1-99%): {np.percentile(opd_nm, 99) - np.percentile(opd_nm, 1):.1f} nm")

    return opd_m


def test_height_computation(opd_m):
    """Test height computation with different RI contrasts."""
    from core.qpi import opd_to_height

    section("2. Yükseklik Hesaplama")

    # Typical samples
    samples = [
        ("Polystyrene bead (n=1.59)", 1.59, 1.0),
        ("Glass (n=1.52)", 1.52, 1.0),
        ("Cell (n=1.38) in PBS (n=1.34)", 1.38, 1.34),
        ("PMMA (n=1.49) in air", 1.49, 1.0),
    ]

    for name, n_sample, n_medium in samples:
        try:
            height_m = opd_to_height(opd_m, n_sample, n_medium)
            h_nm = height_m * 1e9
            h_um = height_m * 1e6
            p2p = float(np.percentile(h_nm, 99) - np.percentile(h_nm, 1))
            ok(f"{name:40s}: p2p = {p2p:.1f} nm ({p2p/1000:.3f} µm)")
        except Exception as e:
            fail(f"{name}: {e}")


def test_roughness(opd_m):
    """Test surface roughness computation."""
    from core.qpi import compute_roughness

    section("3. Yüzey Pürüzlülüğü")

    try:
        with Timer("compute_roughness"):
            rough = compute_roughness(opd_m)
        ok(f"Ra = {rough.Ra_nm:.2f} nm")
        ok(f"Rq = {rough.Rq_nm:.2f} nm")
        ok(f"Rz = {rough.Rz_nm:.2f} nm")
    except Exception as e:
        fail(f"Pürüzlülük hesaplanamadı: {e}")


def test_qpi_modes(unwrapped):
    """Test compute_qpi with different modes."""
    from core.qpi import compute_qpi, QPIMode

    section("4. QPI Modları")

    modes = [
        (QPIMode.MICRO_STRUCTURE, 1.59, 1.0, None),
        (QPIMode.BIOLOGICAL_CELL, 1.38, 1.34, None),
    ]

    for mode, n_sample, n_medium, known_h in modes:
        try:
            with Timer(f"{mode.value}"):
                result = compute_qpi(
                    phase_unwrapped=unwrapped,
                    wavelength_m=DEFAULT_WAVELENGTH_M,
                    pixel_size_m=DEFAULT_PIXEL_SIZE_M,
                    mode=mode,
                    n_sample=n_sample,
                    n_medium=n_medium,
                    known_thickness_m=known_h,
                    cell_threshold=0.5,
                    compute_psd=True,
                )
            ok(f"Mode: {mode.value}")
            ok(f"  OPD range: {result.opd_nm.min():.1f} — {result.opd_nm.max():.1f} nm")
            ok(f"  Height range: {result.height_nm.min():.1f} — {result.height_nm.max():.1f} nm")
            if result.roughness:
                ok(f"  Ra = {result.roughness.Ra_nm:.2f} nm, Rq = {result.roughness.Rq_nm:.2f} nm")
            if result.step_height_m is not None:
                ok(f"  Step height = {result.step_height_m * 1e6:.3f} µm")
            if hasattr(result, 'dry_mass_pg') and result.dry_mass_pg is not None:
                ok(f"  Dry mass = {result.dry_mass_pg:.2f} pg")
            if result.psd is not None:
                ok(f"  PSD computed: {len(result.psd.freq_um_inv)} points")
        except Exception as e:
            fail(f"{mode.value}: {e}")


def test_background_correction(unwrapped):
    """Test background phase correction."""
    from core.qpi import correct_background_phase

    section("5. Arka Plan Düzeltme")

    for order in [1, 2, 3]:
        with Timer(f"poly order={order}"):
            corrected = correct_background_phase(unwrapped, order=order)
        rng_before = float(unwrapped.max() - unwrapped.min())
        rng_after = float(corrected.max() - corrected.min())
        ok(f"order={order}: range {rng_before:.2f} → {rng_after:.2f} rad "
           f"({(1 - rng_after/rng_before)*100:+.1f}%)")


def test_reference_subtraction(field, z_mm):
    """Test reference wave subtraction (if ref available)."""
    from core.qpi import subtract_reference_wave

    section("6. Referans Dalga Çıkarma")

    from tools._common import REF_3UM
    if REF_3UM and REF_3UM.exists():
        from tools._common import load_sample as _ls
        ref_img, _ = _ls(str(REF_3UM))
        ref_field, _, _, _ = extract_field(ref_img)

        from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
        params = ReconstructionParams(
            wavelength_m=DEFAULT_WAVELENGTH_M,
            pixel_size_m=DEFAULT_PIXEL_SIZE_M,
            z_m=z_mm * 1e-3,
            n=DEFAULT_N_MEDIUM,
        )
        sample_recon = np.asarray(propagate(field, params, ReconstructionMethod.ASM))
        ref_recon = np.asarray(propagate(ref_field, params, ReconstructionMethod.ASM))

        with Timer("subtract_reference_wave"):
            corrected = subtract_reference_wave(sample_recon, ref_recon)

        ok(f"Referans çıkarma tamamlandı")
        ok(f"Phase std before: {np.std(np.angle(sample_recon)):.4f}")
        ok(f"Phase std after:  {np.std(np.angle(corrected)):.4f}")
    else:
        info("Referans hologram bulunamadı — bu test atlandı")
        info(f"Beklenen konum: {REF_3UM}")


def main():
    parser = argparse.ArgumentParser(description="QPI Pipeline Test Tool")
    parser.add_argument("--image", help="Hologram dosyası")
    parser.add_argument("--z_mm", type=float, default=-1.0, help="Propagasyon mesafesi (mm)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: QPI Pipeline                    ║")
    print("╚══════════════════════════════════════════╝")

    img, path = load_sample(args.image)
    info(f"Kaynak: {path.name} ({img.shape})")

    field, center, _, _ = extract_field(img)
    info(f"+1 order: {center}")

    unwrapped, recon = _get_unwrapped_phase(field, args.z_mm)
    info(f"Propagasyon: z = {args.z_mm} mm")

    opd_m = test_opd_computation(unwrapped)
    test_height_computation(opd_m)
    test_roughness(opd_m)
    test_qpi_modes(unwrapped)
    test_background_correction(unwrapped)
    test_reference_subtraction(field, args.z_mm)

    print("\n✅ QPI testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
