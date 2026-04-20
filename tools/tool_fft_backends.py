"""
Tool: FFT Backend Performans Testi
====================================
NumPy, SciPy, PyFFTW, MLX backend'lerini karşılaştırır.

Kullanım:
    python -m tools.tool_fft_backends
    python -m tools.tool_fft_backends --size 1024
"""
import argparse
import numpy as np
import time

from tools._common import SRC, section, ok, fail, info, Timer
import sys
sys.path.insert(0, str(SRC))


def test_backend_availability():
    """Check which FFT backends are available."""
    from core.fft_backend import (
        NumpyFFTBackend, ScipyFFTBackend, PyFFTWBackend, MLXFFTBackend,
        get_best_fft_backend
    )

    section("1. Backend Erişilebilirlik")

    backends = {
        "NumPy": NumpyFFTBackend,
        "SciPy": ScipyFFTBackend,
        "PyFFTW": PyFFTWBackend,
        "MLX": MLXFFTBackend,
    }

    available = []
    for name, cls in backends.items():
        try:
            b = cls()
            ok(f"{name:8s}: ✓ erişilebilir")
            available.append((name, b))
        except Exception as e:
            info(f"{name:8s}: ✗ erişilemiyor — {e}")

    best = get_best_fft_backend()
    ok(f"\nVarsayılan backend: {best.name.value}")

    return available


def test_correctness(available, size):
    """Verify all backends produce the same result."""
    section("2. Doğruluk Kontrolü")

    np.random.seed(42)
    data = np.random.randn(size, size).astype(np.complex64)

    ref_fft = None
    ref_name = None

    for name, backend in available:
        result = backend.fft2(data)
        result_np = np.array(result)

        if ref_fft is None:
            ref_fft = result_np
            ref_name = name
            ok(f"{name:8s}: referans olarak ayarlandı")
        else:
            diff = np.max(np.abs(result_np - ref_fft))
            rel_err = diff / np.max(np.abs(ref_fft))
            if rel_err < 1e-4:
                ok(f"{name:8s}: max fark = {diff:.2e}, rel = {rel_err:.2e} (OK)")
            else:
                fail(f"{name:8s}: max fark = {diff:.2e}, rel = {rel_err:.2e} (YÜKSEK!)")

    # Round-trip test (FFT → IFFT = identity)
    info("\nGidiş-dönüş testi (FFT → IFFT):")
    for name, backend in available:
        result = backend.ifft2(backend.fft2(data))
        result_np = np.array(result)
        diff = np.max(np.abs(result_np - data))
        if diff < 1e-4:
            ok(f"{name:8s}: roundtrip farkı = {diff:.2e} (OK)")
        else:
            fail(f"{name:8s}: roundtrip farkı = {diff:.2e} (YÜKSEK!)")


def test_performance(available, size, n_iter=10):
    """Benchmark FFT performance."""
    section("3. Performans Karşılaştırma")

    np.random.seed(42)
    data = np.random.randn(size, size).astype(np.complex64)

    info(f"Boyut: {size}×{size}, İterasyon: {n_iter}\n")
    info(f"  {'Backend':8s} | {'FFT2 (ms)':>10s} | {'IFFT2 (ms)':>10s} | {'Toplam (ms)':>12s}")
    info(f"  {'-'*8}-|{'-'*11}-|{'-'*11}-|{'-'*13}")

    results = []
    for name, backend in available:
        # Warmup
        _ = backend.fft2(data)
        _ = backend.ifft2(backend.fft2(data))

        # FFT benchmark
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = backend.fft2(data)
        fft_time = (time.perf_counter() - t0) / n_iter * 1000

        # IFFT benchmark
        fft_result = backend.fft2(data)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = backend.ifft2(fft_result)
        ifft_time = (time.perf_counter() - t0) / n_iter * 1000

        total = fft_time + ifft_time
        info(f"  {name:8s} | {fft_time:10.2f} | {ifft_time:10.2f} | {total:12.2f}")
        results.append((name, total))

    # Rank
    results.sort(key=lambda x: x[1])
    info(f"\n  Sıralama (hızlıdan yavaşa):")
    for i, (name, total) in enumerate(results):
        speedup = results[-1][1] / total if total > 0 else 0
        info(f"  {i+1}. {name:8s} — {total:.2f} ms ({speedup:.1f}× en yavaşa göre)")


def test_different_sizes(available):
    """Test performance across different array sizes."""
    section("4. Farklı Boyutlarda Performans")

    sizes = [256, 512, 1024, 2048]
    header = f"  {'Size':>6s}"
    for name, _ in available:
        header += f" | {name:>10s}"
    info(header)
    sep = f"  {'-'*6}"
    for _ in available:
        sep += f"-|{'-'*10}"
    info(sep)

    for size in sizes:
        np.random.seed(42)
        data = np.random.randn(size, size).astype(np.complex64)
        row = f"  {size:6d}"

        for name, backend in available:
            try:
                _ = backend.fft2(data)
                t0 = time.perf_counter()
                for _ in range(5):
                    _ = backend.fft2(data)
                t = (time.perf_counter() - t0) / 5 * 1000
                row += f" | {t:8.1f}ms"
            except Exception:
                row += f" | {'HATA':>10s}"
        info(row)


def main():
    parser = argparse.ArgumentParser(description="FFT Backend Test Tool")
    parser.add_argument("--size", type=int, default=512, help="Test array boyutu")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TOOL: FFT Backend Performans          ║")
    print("╚══════════════════════════════════════════╝")

    available = test_backend_availability()
    if not available:
        fail("Hiçbir FFT backend erişilebilir değil!")
        return

    test_correctness(available, args.size)
    test_performance(available, args.size)
    test_different_sizes(available)

    print("\n✅ FFT backend testleri tamamlandı.\n")


if __name__ == "__main__":
    main()
