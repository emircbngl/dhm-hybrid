"""Autofocus algorithm benchmark — 6 algorithms × 2 hologram sizes.

Writes a markdown table to stdout, plus a JSON summary the
``test_autofocus_speed_baseline.py`` regression test can read.

Run::

    python scripts/bench_autofocus.py

The 6 algorithms come from ``src/ui2/workers.py`` — ``zscan``,
``coarse_to_fine``, ``robust``, ``adaptive_gradient``,
``adaptive_bracketing``, ``adaptive_distance``. Each is timed on a
synthetic single-sphere hologram (the same ``build_hologram`` the
validation suite uses) at 256×256 (fast sanity) and 512×512
(realistic lab size). 40 steps / max_evaluations across the board
so every algorithm sees the same budget.

Design choice: the benchmark **doesn't** go through
``ScienceDriver`` — ``_prepare_field`` adds TIFF I/O and preprocess
cost that we want to time separately. We run the core search
functions directly on a pre-extracted complex field, which is what
the dispatcher calls under the hood. TIFF I/O and preprocessing
have their own regression test elsewhere; this benchmark isolates
the scan loop cost that the user reported as slow.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core.autofocus import FocusMetric, autofocus_zscan  # noqa: E402
from core.autofocus.search_adaptive import (  # noqa: E402
    adaptive_bracketing_search,
    adaptive_distance_search,
    adaptive_gradient_search,
)
from core.autofocus.search_classic import (  # noqa: E402
    coarse_to_fine_search,
    robust_coarse_to_fine_search,
)
from core.offaxis import OffAxisParams, extract_complex_field_offaxis  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
)
from fixtures.synthetic_hologram import (  # noqa: E402
    HologramConfig,
    SphereSpec,
    build_hologram,
)


def _make_field(shape: Tuple[int, int]) -> Tuple[np.ndarray, ReconstructionParams]:
    """One synthetic-sphere hologram → complex field for the scan."""
    cfg = HologramConfig(
        shape=shape,
        pixel_m=2.5e-6,
        wavelength_m=632.8e-9,
        # carrier in cycles/m — keep within ~20 % of Nyquist so the +1
        # order has headroom for both 256 and 512 grids.
        carrier_freq_m_inv=(50_000.0, 0.0),
    )
    sphere = SphereSpec(
        radius_m=15e-6, z_m=12e-3,
        center_yx_m=(0.0, 0.0),
        n_sphere=1.40, n_medium=1.33,
    )
    holo = build_hologram([sphere], cfg)
    field, _ = extract_complex_field_offaxis(holo, OffAxisParams(radius=40))
    base = ReconstructionParams(
        wavelength_m=cfg.wavelength_m,
        pixel_size_m=cfg.pixel_m,
        z_m=0.0, n=1.33,
    )
    return field, base


def _time(fn: Callable[[], Any]) -> Tuple[float, Any]:
    # Warm once (FT caches, any JIT) and discard — we want steady-state.
    fn()
    t0 = time.monotonic()
    result = fn()
    return (time.monotonic() - t0) * 1000.0, result


def _run_one_shape(shape: Tuple[int, int], n_steps: int = 40) -> List[Dict]:
    field, base = _make_field(shape)
    method = ReconstructionMethod.ASM
    metric = FocusMetric.ENTROPY
    z_min_m, z_max_m = 8e-3, 16e-3  # 8 mm band around truth 12 mm
    rows: List[Dict] = []

    def _zscan():
        zs = list(np.linspace(z_min_m, z_max_m, n_steps))
        return autofocus_zscan(field, base, zs, method, metric)

    def _coarse():
        return coarse_to_fine_search(
            field, base, method, metric,
            z_min_m=z_min_m, z_max_m=z_max_m,
            coarse_steps=n_steps,
            fine_tolerance_m=max(1e-9, (z_max_m - z_min_m) / 1e4),
        )

    def _robust():
        return robust_coarse_to_fine_search(
            field, base, method, metric,
            z_min_m=z_min_m, z_max_m=z_max_m,
            n_coarse=n_steps, refine_factor=8, smooth_sigma=1.5,
        )

    def _adap_grad():
        return adaptive_gradient_search(
            field, base, method, metric,
            z_min_m=z_min_m, z_max_m=z_max_m,
            max_evaluations=n_steps,
        )

    def _adap_brack():
        return adaptive_bracketing_search(
            field, base, method, metric,
            z_min_m=z_min_m, z_max_m=z_max_m,
            n_refine_levels=3, refine_divisions=8,
            smooth_sigma=1.0, max_evaluations=n_steps,
        )

    def _adap_dist():
        return adaptive_distance_search(
            field, base, method, metric,
            initial_range_m=0.5e-3,
            max_range_m=50e-3,
            expand_factor=2.0,
            signal_threshold=0.3,
            max_evaluations=n_steps,
        )

    cases = [
        ("zscan",               _zscan),
        ("coarse_to_fine",      _coarse),
        ("robust",              _robust),
        ("adaptive_gradient",   _adap_grad),
        ("adaptive_bracketing", _adap_brack),
        ("adaptive_distance",   _adap_dist),
    ]
    for name, fn in cases:
        dt_ms, result = _time(fn)
        best_z = float(getattr(result, "best_z_m", 0.0))
        evals = int(getattr(result, "evaluations", n_steps))
        rows.append({
            "algorithm": name,
            "shape": f"{shape[0]}x{shape[1]}",
            "n_steps": n_steps,
            "runtime_ms": round(dt_ms, 1),
            "best_z_mm": round(best_z * 1e3, 3),
            "evaluations": evals,
        })
    return rows


def _bench_zscan_with_backend(shape, n_steps, backend) -> Dict:
    """v2.1.0: time a zscan with a specific FFT backend (numpy
    serial vs torch batched). Returns one row dict."""
    field, base = _make_field(shape)
    method = ReconstructionMethod.ASM
    metric = FocusMetric.ENTROPY
    z_min_m, z_max_m = 8e-3, 16e-3
    zs = list(np.linspace(z_min_m, z_max_m, n_steps))

    # Warm
    autofocus_zscan(field, base, zs, method, metric,
                   batch_backend=backend)
    t0 = time.monotonic()
    res = autofocus_zscan(field, base, zs, method, metric,
                          batch_backend=backend)
    dt_ms = (time.monotonic() - t0) * 1000.0
    return {
        "algorithm": "zscan",
        "shape": f"{shape[0]}x{shape[1]}",
        "n_steps": n_steps,
        "backend": backend.name.value if backend else "default",
        "device": (getattr(backend, "device", "cpu")
                   if backend else "cpu"),
        "runtime_ms": round(dt_ms, 1),
        "best_z_mm": round(float(res.best_z_m) * 1e3, 3),
        "evaluations": n_steps,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes", default="256,512,1024",
        help="Comma-separated list of side sizes (e.g. "
             "'256,512,1024,2048'). Default stops at 1024 because 2048 "
             "adds ~30 s per algorithm.",
    )
    ap.add_argument(
        "--backends", default="default",
        help="Comma-separated FFT backends to compare for zscan-only "
             "head-to-head. 'default' = current best (PyFFTW / SciPy "
             "/ NumPy fallback), 'torch' = TorchFFTBackend (CUDA / "
             "MPS / CPU). Multiple → cartesian over shapes.",
    )
    args = ap.parse_args()
    shapes = [int(s.strip()) for s in args.shapes.split(",") if s.strip()]
    backend_keys = [b.strip() for b in args.backends.split(",")
                    if b.strip()]

    all_rows: List[Dict] = []
    # Per-shape full algorithm sweep (default backend only).
    for n in shapes:
        all_rows.extend(_run_one_shape((n, n)))

    # zscan-only backend comparison (when more than one backend asked).
    if backend_keys != ["default"]:
        from core.fft_backend import FFTBackendName, get_best_fft_backend
        backend_rows: List[Dict] = []
        for key in backend_keys:
            try:
                if key == "default":
                    backend = None
                elif key == "torch":
                    from core.fft_backend import TorchFFTBackend
                    backend = TorchFFTBackend()
                else:
                    backend = get_best_fft_backend(
                        FFTBackendName(key),
                    )
            except Exception as exc:
                print(f"# skipping backend {key}: {exc}")
                continue
            for n in shapes:
                row = _bench_zscan_with_backend(
                    (n, n), n_steps=40, backend=backend,
                )
                backend_rows.append(row)
        if backend_rows:
            print()
            print("# Backend head-to-head — zscan, ENTROPY, 40 steps")
            print()
            print("| shape | backend | device | runtime_ms | best_z_mm |")
            print("|-------|---------|--------|-----------:|----------:|")
            for r in backend_rows:
                print(
                    f"| {r['shape']} | {r['backend']} | "
                    f"{r['device']} | {r['runtime_ms']:.1f} | "
                    f"{r['best_z_mm']:.3f} |"
                )
            all_rows.extend(backend_rows)

    # Markdown table — copy-pastable into todo.md / a report.
    print(f"# Autofocus benchmark — ENTROPY metric, 40 steps, Δn=0.07 sphere\n")
    print("| shape | algorithm | runtime_ms | evals | best_z_mm |")
    print("|-------|-----------|-----------:|------:|----------:|")
    for r in all_rows:
        print(
            f"| {r['shape']} | {r['algorithm']} | "
            f"{r['runtime_ms']:.1f} | {r['evaluations']} | "
            f"{r['best_z_mm']:.3f} |"
        )

    out = ROOT / "tasks" / "bench_autofocus.json"
    out.write_text(json.dumps(all_rows, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
