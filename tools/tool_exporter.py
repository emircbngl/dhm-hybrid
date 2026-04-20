"""
Tool: Dışa Aktarma (Exporter) Testi
======================================
NPY, TIFF, PNG, CSV, MAT formatlarına kaydetme ve doğrulama.

Kullanım:
    python -m tools.tool_exporter
    python -m tools.tool_exporter --image path/to/hologram.png
"""
import argparse
import tempfile
import numpy as np
from pathlib import Path

from tools._common import (
    SRC, load_sample, extract_field, section, ok, fail, info, Timer,
    DEFAULT_WAVELENGTH_M, DEFAULT_PIXEL_SIZE_M, DEFAULT_N_MEDIUM
)
import sys
sys.path.insert(0, str(SRC))


def test_npy_export(data):
    """Test NPY export and round-trip."""
    from core.exporter import save_npy

    section("1. NPY Export")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.npy"
        with Timer("save_npy"):
            save_npy(path, data)
        ok(f"Dosya oluşturuldu: {path.stat().st_size:,} bytes")

        # Round-trip
        loaded = np.load(path)
        diff = np.max(np.abs(loaded - data))
        if diff < 1e-10:
            ok(f"Gidiş-dönüş doğru (fark={diff:.2e})")
        else:
            fail(f"Gidiş-dönüş hatası: fark={diff:.2e}")


def test_tiff_export(data):
    """Test TIFF export."""
    from core.exporter import save_tiff

    section("2. TIFF Export")

    with tempfile.TemporaryDirectory() as tmp:
        # Float32
        path = Path(tmp) / "test_f32.tiff"
        with Timer("save_tiff (float32)"):
            save_tiff(path, data.astype(np.float32))
        ok(f"Float32 TIFF: {path.stat().st_size:,} bytes")

        # Uint16
        norm = ((data - data.min()) / (data.max() - data.min()) * 65535).astype(np.uint16)
        path2 = Path(tmp) / "test_u16.tiff"
        with Timer("save_tiff (uint16)"):
            save_tiff(path2, norm)
        ok(f"Uint16 TIFF: {path2.stat().st_size:,} bytes")

        # Verify readable
        try:
            import tifffile
            loaded = tifffile.imread(path)
            ok(f"TIFF okunabilir: shape={loaded.shape}, dtype={loaded.dtype}")
        except Exception as e:
            fail(f"TIFF okunamadı: {e}")


def test_png_export(data):
    """Test PNG export."""
    from core.exporter import save_png

    section("3. PNG Export")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.png"
        with Timer("save_png"):
            save_png(path, data)
        ok(f"PNG: {path.stat().st_size:,} bytes")

        # Verify
        try:
            from core.ingestion import load_png as _load
            loaded = _load(path)
            ok(f"PNG okunabilir: shape={loaded.array.shape}, dtype={loaded.array.dtype}")
        except Exception as e:
            fail(f"PNG okunamadı: {e}")


def test_csv_export(data):
    """Test CSV export."""
    from core.exporter import save_csv

    section("4. CSV Export")

    small = data[:10, :10]  # Small subset
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.csv"
        with Timer("save_csv"):
            save_csv(path, small)
        ok(f"CSV: {path.stat().st_size:,} bytes")

        # Verify
        loaded = np.loadtxt(str(path), delimiter=",")
        diff = np.max(np.abs(loaded - small))
        ok(f"Gidiş-dönüş farkı: {diff:.2e}")


def test_mat_export(data):
    """Test MATLAB .mat export."""
    from core.exporter import save_mat

    section("5. MAT Export")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.mat"
        try:
            with Timer("save_mat"):
                save_mat(path, {"amplitude": data, "test_param": np.array(42.0)})
            ok(f"MAT: {path.stat().st_size:,} bytes")

            # Verify
            from scipy.io import loadmat
            loaded = loadmat(str(path))
            ok(f"MAT okunabilir: keys={list(loaded.keys())}")
            diff = np.max(np.abs(loaded["amplitude"] - data))
            ok(f"Amplitude farkı: {diff:.2e}")
        except Exception as e:
            fail(f"MAT export/import hatası: {e}")


def test_export_paths():
    """Test ExportPaths utility."""
    from core.exporter import ExportPaths, default_export_root

    section("6. Export Yol Yardımcıları")

    root = default_export_root()
    ok(f"Varsayılan root: {root}")

    with tempfile.TemporaryDirectory() as tmp:
        ep = ExportPaths(root=Path(tmp) / "test_output" / "sub")
        ep.ensure()
        ok(f"Dizin oluşturuldu: {ep.root}")
        ok(f"Dizin var mı: {ep.root.exists()}")


def main():
    parser = argparse.ArgumentParser(description="Exporter Test Tool")
    parser.add_argument("--image", help="Hologram dosyası")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: Dışa Aktarma (Exporter)         ║")
    print("╚══════════════════════════════════════════╝")

    img, path = load_sample(args.image)
    info(f"Kaynak: {path.name} ({img.shape})")

    field, _, _, _ = extract_field(img)

    from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
    params = ReconstructionParams(
        wavelength_m=DEFAULT_WAVELENGTH_M,
        pixel_size_m=DEFAULT_PIXEL_SIZE_M,
        z_m=-1e-3,
        n=DEFAULT_N_MEDIUM,
    )
    recon = np.asarray(propagate(field, params, ReconstructionMethod.ASM))
    amplitude = np.abs(recon)

    test_npy_export(amplitude)
    test_tiff_export(amplitude)
    test_png_export(amplitude)
    test_csv_export(amplitude)
    test_mat_export(amplitude)
    test_export_paths()

    print("\n✅ Exporter testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
