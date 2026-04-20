"""
Tool: Tam Pipeline Testi (Uçtan Uca)
=======================================
Yükleme → Off-axis → Autofocus → Propagasyon → Unwrap → QPI

Kullanım:
    python -m tools.tool_full_pipeline
    python -m tools.tool_full_pipeline --image path/to/hologram.png
"""
import argparse
import numpy as np
import time

from tools._common import (
    ROOT, SRC, load_sample, extract_field, section, ok, fail, info, Timer,
    DEFAULT_WAVELENGTH_M, DEFAULT_PIXEL_SIZE_M, DEFAULT_N_MEDIUM, DEFAULT_MASK_RADIUS
)
import sys
sys.path.insert(0, str(SRC))


def run_full_pipeline(image_path=None, z_min_mm=-5.0, z_max_mm=5.0):
    """Run the complete DHM reconstruction pipeline end-to-end."""

    total_start = time.perf_counter()

    # ── Step 1: Load ──
    section("Step 1: Görüntü Yükleme")
    with Timer("Yükleme"):
        img, path = load_sample(image_path)
    ok(f"Dosya: {path.name}")
    ok(f"Shape: {img.shape}, dtype: {img.dtype}")
    ok(f"Range: [{img.min():.1f}, {img.max():.1f}]")

    # ── Step 2: Off-axis extraction ──
    section("Step 2: Off-axis Çıkarım")
    with Timer("Off-axis"):
        field, center, spectrum, mask = extract_field(img, DEFAULT_MASK_RADIUS)
    ok(f"+1 order merkez: {center}")
    ok(f"Karmaşık alan: {field.shape}, dtype={field.dtype}")

    # ── Step 3: Autofocus ──
    section("Step 3: Otomatik Odaklama")
    from core.reconstruction import ReconstructionParams, ReconstructionMethod
    from core.autofocus import FocusMetric, robust_coarse_to_fine_search

    base = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=0.0,
        n=DEFAULT_N_MEDIUM,
    )

    with Timer("Robust autofocus"):
        af_result = robust_coarse_to_fine_search(
            field, base, ReconstructionMethod.ASM,
            FocusMetric.PHASE_VARIANCE,
            z_min_m=z_min_mm * 1e-3,
            z_max_m=z_max_mm * 1e-3,
            n_coarse=40,
        )
    best_z_mm = af_result.best_z_m * 1e3
    ok(f"En iyi z: {best_z_mm:.4f} mm")
    ok(f"Evaluations: {af_result.total_evaluations}")

    # ── Step 4: Propagation ──
    section("Step 4: Propagasyon")
    from core.reconstruction import propagate

    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=af_result.best_z_m,
        n=DEFAULT_N_MEDIUM,
    )

    with Timer("ASM propagation"):
        recon = np.asarray(propagate(field, params, ReconstructionMethod.ASM))

    amp = np.abs(recon)
    ok(f"Amplitude: min={amp.min():.6f}, max={amp.max():.6f}, mean={amp.mean():.6f}")
    ok(f"NaN/Inf: {np.sum(~np.isfinite(recon))}")

    # ── Step 5: Phase Unwrap ──
    section("Step 5: Faz Açma")
    from core.phase_unwrap import unwrap_phase_advanced, UnwrapConfig, UnwrapMethod

    wrapped = np.angle(recon)
    config = UnwrapConfig(method=UnwrapMethod.QUALITY_GUIDED, post_bg_remove=True, post_bg_order=2)

    with Timer("Phase unwrap (quality_guided)"):
        unwrapped = unwrap_phase_advanced(
            wrapped, config=config,
            complex_field=recon,
            wavelength_m=DEFAULT_WAVELENGTH_M,
            pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        )

    ok(f"Unwrapped range: {unwrapped.min():.4f} — {unwrapped.max():.4f} rad")
    ok(f"NaN/Inf: {np.sum(~np.isfinite(unwrapped))}")

    # ── Step 6: QPI ──
    section("Step 6: QPI Hesaplama")
    from core.qpi import compute_qpi, QPIMode

    with Timer("QPI compute (micro_structure)"):
        qpi_result = compute_qpi(
            phase_unwrapped=unwrapped,
            wavelength_m=DEFAULT_WAVELENGTH_M,
            pixel_size_m=DEFAULT_PIXEL_SIZE_M,
            mode=QPIMode.MICRO_STRUCTURE,
            n_sample=1.59,
            n_medium=DEFAULT_N_MEDIUM,
            cell_threshold=0.5,
            compute_psd=True,
        )

    ok(f"OPD range: {qpi_result.opd_nm.min():.1f} — {qpi_result.opd_nm.max():.1f} nm")
    ok(f"Height range: {qpi_result.height_nm.min():.1f} — {qpi_result.height_nm.max():.1f} nm")
    if qpi_result.roughness:
        ok(f"Ra = {qpi_result.roughness.Ra_nm:.2f} nm, Rq = {qpi_result.roughness.Rq_nm:.2f} nm")
    if qpi_result.step_height_m is not None:
        ok(f"Step height = {qpi_result.step_height_m * 1e6:.3f} µm")
    if qpi_result.psd is not None:
        ok(f"PSD: {len(qpi_result.psd.freq_um_inv)} points")

    # ── Summary ──
    total_time = time.perf_counter() - total_start

    section("ÖZET")
    info(f"Dosya          : {path.name}")
    info(f"Boyut          : {img.shape}")
    info(f"En iyi z       : {best_z_mm:.4f} mm")
    info(f"OPD range      : {qpi_result.opd_nm.max() - qpi_result.opd_nm.min():.1f} nm")
    info(f"Height p2p     : {np.percentile(qpi_result.height_nm, 99) - np.percentile(qpi_result.height_nm, 1):.1f} nm")
    if qpi_result.step_height_m is not None:
        info(f"Step height    : {qpi_result.step_height_m * 1e6:.3f} µm")
    info(f"Toplam süre    : {total_time:.2f} s")

    return {
        "image": path.name,
        "shape": img.shape,
        "best_z_mm": best_z_mm,
        "opd_range_nm": float(qpi_result.opd_nm.max() - qpi_result.opd_nm.min()),
        "total_time_s": total_time,
    }


def run_multi_sample():
    """Run pipeline on all available samples."""
    section("ÇOK-DOSYALI TEST")

    sample_dirs = [ROOT / "labtest", ROOT / "Test Samples"]
    all_samples = []
    for d in sample_dirs:
        if d.exists():
            all_samples.extend(sorted(d.glob("*.png")))

    # Filter out reference images
    samples = [s for s in all_samples if not s.name.startswith("Ref_")]

    if not samples:
        info("Test örneği bulunamadı")
        return

    info(f"Bulunan {len(samples)} örnek\n")
    results = []

    for i, sample in enumerate(samples[:5]):  # Max 5 to save time
        info(f"\n--- [{i+1}/{min(len(samples), 5)}] {sample.name} ---")
        try:
            result = run_full_pipeline(str(sample))
            results.append(result)
        except Exception as e:
            fail(f"{sample.name}: {e}")

    if results:
        section("ÇOK-DOSYA ÖZET")
        info(f"\n  {'Dosya':40s} | {'Z (mm)':>10s} | {'OPD (nm)':>10s} | {'Süre (s)':>10s}")
        info(f"  {'-'*40}-|{'-'*11}-|{'-'*11}-|{'-'*11}")
        for r in results:
            info(f"  {r['image']:40s} | {r['best_z_mm']:10.4f} | {r['opd_range_nm']:10.1f} | {r['total_time_s']:10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Test Tool")
    parser.add_argument("--image", help="Hologram dosyası")
    parser.add_argument("--multi", action="store_true", help="Tüm örneklerde çalıştır")
    parser.add_argument("--z_min", type=float, default=-5.0, help="Min z (mm)")
    parser.add_argument("--z_max", type=float, default=5.0, help="Max z (mm)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: Tam Pipeline (Uçtan Uca)        ║")
    print("╚══════════════════════════════════════════╝")

    if args.multi:
        run_multi_sample()
    else:
        run_full_pipeline(args.image, args.z_min, args.z_max)

    print("\n✅ Full pipeline testi tamamlandı.\n")


if __name__ == "__main__":
    main()
