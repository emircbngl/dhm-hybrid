"""
Tool: Görüntü Yükleme (Ingestion) Testi
=========================================
PNG, TIFF, NPY formatlarını yükler, metadata okur, temel istatistikleri gösterir.

Kullanım:
    python -m tools.tool_ingestion
    python -m tools.tool_ingestion --image path/to/image.png
"""
import argparse
import numpy as np
from pathlib import Path

from tools._common import (
    ROOT, SRC, load_sample, find_sample, section, ok, fail, info, Timer
)
import sys
sys.path.insert(0, str(SRC))


def test_load_formats():
    """Test loading different image formats."""
    from core.ingestion import load_any, load_png, load_tiff, load_npy

    section("1. Format Desteği")

    # PNG
    png_files = list(ROOT.glob("labtest/*.png")) + list(ROOT.glob("Test Samples/*.png"))
    if png_files:
        with Timer("PNG load"):
            data = load_png(png_files[0])
        ok(f"PNG: {png_files[0].name} → shape={data.array.shape}, dtype={data.array.dtype}")
    else:
        info("PNG test dosyası bulunamadı")

    # TIFF
    tiff_files = list(ROOT.glob("**/*.tiff")) + list(ROOT.glob("**/*.tif"))
    tiff_files = [f for f in tiff_files if "output" not in str(f) and "tests" not in str(f)]
    if tiff_files:
        with Timer("TIFF load"):
            data = load_tiff(tiff_files[0])
        ok(f"TIFF: {tiff_files[0].name} → shape={data.array.shape}, dtype={data.array.dtype}")
        if data.metadata:
            info(f"  Metadata: {data.metadata}")
    else:
        info("TIFF test dosyası bulunamadı")

    # NPY
    npy_files = list(ROOT.glob("tests/**/*.npy"))
    if npy_files:
        with Timer("NPY load"):
            data = load_npy(npy_files[0])
        ok(f"NPY: {npy_files[0].name} → shape={data.array.shape}, dtype={data.array.dtype}")
    else:
        info("NPY test dosyası bulunamadı")


def test_load_any():
    """Test the universal loader."""
    from core.ingestion import load_any

    section("2. Evrensel Yükleyici (load_any)")

    sample_path = find_sample()
    with Timer("load_any"):
        data = load_any(sample_path)

    arr = np.asarray(data.array)
    ok(f"Dosya: {sample_path.name}")
    ok(f"Shape: {arr.shape}")
    ok(f"Dtype: {arr.dtype}")
    ok(f"Min/Max: {arr.min():.2f} / {arr.max():.2f}")
    ok(f"Mean/Std: {np.mean(arr):.2f} / {np.std(arr):.2f}")

    if data.metadata:
        ok(f"Metadata: {data.metadata}")
    else:
        info("Metadata bulunamadı (beklenen — standart PNG için)")


def test_metadata_reader():
    """Test embedded metadata extraction."""
    from core.metadata_reader import read_embedded_metadata

    section("3. Metadata Reader")

    sample_path = find_sample()
    acq = read_embedded_metadata(sample_path)
    ok(f"wavelength_m: {acq.wavelength_m}")
    ok(f"pixel_size_m: {acq.pixel_size_m}")
    ok(f"magnification: {acq.magnification}")
    ok(f"source: {acq.source}")


def test_error_handling():
    """Test error handling for unsupported formats."""
    from core.ingestion import load_any, UnsupportedFormatError

    section("4. Hata Yönetimi")

    try:
        load_any("/nonexistent/file.xyz")
        fail("UnsupportedFormatError beklendi ama gelmedi")
    except (UnsupportedFormatError, FileNotFoundError, Exception) as e:
        ok(f"Beklenen hata yakalandı: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ingestion Test Tool")
    parser.add_argument("--image", help="Test edilecek görüntü dosyası")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: Görüntü Yükleme (Ingestion)     ║")
    print("╚══════════════════════════════════════════╝")

    test_load_formats()
    test_load_any()
    test_metadata_reader()
    test_error_handling()

    print("\n✅ Ingestion testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
