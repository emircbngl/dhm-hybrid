"""
Tüm test tool'larını sırayla çalıştır.

Kullanım:
    python -m tools.run_all
    python -m tools.run_all --fast  (sadece hızlı testler)
"""
import argparse
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description="Run All Test Tools")
    parser.add_argument("--fast", action="store_true", help="Sadece hızlı testler")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   TÜM TEST TOOL'LARI                    ║")
    print("╚══════════════════════════════════════════╝")

    tools = [
        ("Ingestion", "tools.tool_ingestion"),
        ("Off-axis", "tools.tool_offaxis"),
        ("Reconstruction", "tools.tool_reconstruction"),
        ("Contrast", "tools.tool_contrast"),
        ("Exporter", "tools.tool_exporter"),
        ("FFT Backends", "tools.tool_fft_backends"),
        ("Phase Unwrap", "tools.tool_phase_unwrap"),
    ]

    if not args.fast:
        tools += [
            ("Autofocus", "tools.tool_autofocus"),
            ("QPI", "tools.tool_qpi"),
            ("Full Pipeline", "tools.tool_full_pipeline"),
        ]

    results = []
    total_start = time.perf_counter()

    for name, module_name in tools:
        print(f"\n{'='*60}")
        print(f"  ▶ {name}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        try:
            import importlib
            mod = importlib.import_module(module_name)
            mod.main()
            elapsed = time.perf_counter() - t0
            results.append((name, "✅ BAŞARILI", elapsed))
        except SystemExit:
            elapsed = time.perf_counter() - t0
            results.append((name, "✅ BAŞARILI", elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            traceback.print_exc()
            results.append((name, f"❌ HATA: {e}", elapsed))

    total_time = time.perf_counter() - total_start

    print(f"\n{'='*60}")
    print(f"  SONUÇLAR")
    print(f"{'='*60}")
    print(f"\n  {'Tool':20s} | {'Durum':25s} | {'Süre (s)':>10s}")
    print(f"  {'-'*20}-|{'-'*26}-|{'-'*11}")
    for name, status, elapsed in results:
        print(f"  {name:20s} | {status:25s} | {elapsed:10.2f}")
    print(f"\n  Toplam süre: {total_time:.2f} s")

    failures = sum(1 for _, s, _ in results if "HATA" in s)
    if failures == 0:
        print(f"\n  ✅ Tüm testler başarılı!")
    else:
        print(f"\n  ❌ {failures} test başarısız!")


if __name__ == "__main__":
    main()
