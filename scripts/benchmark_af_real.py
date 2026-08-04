"""Adaptive-vs-classic autofocus benchmark on REAL lab holograms.

Settles the long-open "adaptive autofocus" backlog item (tasks/todo.md
2026-07-05 note): the six search algorithms were benchmarked only on a
synthetic single-sphere scene (scripts/bench_autofocus.py); their
reliability on real lab data was never measured systematically.

Data: the 2026-04-29 lab sessions under ``~/Desktop/rapor data`` —
one representative (middle) frame per session, with the lab-confirmed
optics from scripts/run_rapor_data_batch.py (632.8 nm, 4.4 µm / 50x =
0.088 µm effective pixel, n=1.337, off-axis mask radius 80).

Ground truth per (frame, metric): a dense 201-step z-scan over ±20 mm
(the lab-confirmed usable range). Each algorithm then runs with the
app-default budget (af_n_steps=40) and is scored on:

* |best_z - dense_truth_z|  (mm)
* evaluations actually spent
* wall time (ms)

plus a per-scene "peak prominence" figure from the dense curve so
flat/ambiguous scenes are visible in the analysis.

Run::

    ./venv/bin/python scripts/benchmark_af_real.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.autofocus import FocusMetric, autofocus_zscan  # noqa: E402
from core.autofocus.metrics import _is_minimize  # noqa: E402
from core.autofocus.search_adaptive import (  # noqa: E402
    adaptive_bracketing_search,
    adaptive_distance_search,
    adaptive_gradient_search,
)
from core.autofocus.search_classic import (  # noqa: E402
    coarse_to_fine_search,
    robust_coarse_to_fine_search,
)
from core.ingestion import load_any  # noqa: E402
from core.offaxis import OffAxisParams, extract_complex_field_offaxis  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
)

# Lab-confirmed optics (scripts/run_rapor_data_batch.py, 2026-05-04).
WAVELENGTH_M = 632.8e-9
EFFECTIVE_M = (4.4 / 50.0) * 1e-6
N_MEDIUM = 1.337
MASK_RADIUS = 80
Z_MIN_M, Z_MAX_M = -20e-3, 20e-3
DENSE_STEPS = 201          # 0.2 mm truth resolution over 40 mm
BUDGET = 40                # app default af_n_steps
METRICS = ("LAPLACIAN_VARIANCE", "ENTROPY")

# Lab session frames. Point DHM_DATA_ROOT at your own capture directory;
# it must contain session_* subdirectories of PNG frames.
DATA_ROOT = Path(
    os.environ.get(
        "DHM_DATA_ROOT",
        Path(__file__).resolve().parent.parent / "data" / "rapor",
    )
)


def _session_frames() -> List[Path]:
    frames: List[Path] = []
    for sess in sorted(DATA_ROOT.glob("session_*")):
        if not sess.is_dir():
            continue
        pngs = sorted(sess.glob("*.png"))
        if pngs:
            frames.append(pngs[len(pngs) // 2])   # middle frame
    return frames


def _prepare_field(path: Path) -> Tuple[np.ndarray, ReconstructionParams]:
    raw = np.asarray(load_any(str(path)).array, dtype=np.float64)
    field, _ = extract_complex_field_offaxis(
        raw, OffAxisParams(radius=MASK_RADIUS))
    base = ReconstructionParams(
        wavelength_m=WAVELENGTH_M, pixel_size_m=EFFECTIVE_M,
        z_m=0.0, n=N_MEDIUM,
    )
    return field, base


def _dense_truth(field, base, metric: FocusMetric) -> Dict[str, Any]:
    zs = list(np.linspace(Z_MIN_M, Z_MAX_M, DENSE_STEPS))
    t0 = time.monotonic()
    res = autofocus_zscan(field, base, zs, ReconstructionMethod.ASM, metric)
    dt = (time.monotonic() - t0) * 1000.0
    # AutoFocusResult.scores is a {z: score} dict — order by z.
    sc = getattr(res, "scores", {}) or {}
    scores = np.asarray([sc[z] for z in sorted(sc)], dtype=np.float64)
    prominence = None
    if scores.size:
        s = scores if not _is_minimize(metric) else -scores
        med = float(np.median(s))
        spread = float(np.percentile(s, 95) - np.percentile(s, 5)) or 1.0
        prominence = (float(np.max(s)) - med) / spread
    return {
        "z_mm": float(res.best_z_m) * 1e3,
        "runtime_ms": dt,
        "prominence": prominence,
    }


def _algorithms(field, base, metric) -> List[Tuple[str, Callable[[], Any]]]:
    m = ReconstructionMethod.ASM

    def _zscan():
        zs = list(np.linspace(Z_MIN_M, Z_MAX_M, BUDGET))
        return autofocus_zscan(field, base, zs, m, metric)

    def _coarse():
        return coarse_to_fine_search(
            field, base, m, metric, z_min_m=Z_MIN_M, z_max_m=Z_MAX_M,
            coarse_steps=BUDGET,
            fine_tolerance_m=max(1e-9, (Z_MAX_M - Z_MIN_M) / 1e4))

    def _robust():
        return robust_coarse_to_fine_search(
            field, base, m, metric, z_min_m=Z_MIN_M, z_max_m=Z_MAX_M,
            n_coarse=BUDGET, refine_factor=8, smooth_sigma=1.5)

    def _grad():
        return adaptive_gradient_search(
            field, base, m, metric, z_min_m=Z_MIN_M, z_max_m=Z_MAX_M,
            max_evaluations=BUDGET)

    def _brack():
        return adaptive_bracketing_search(
            field, base, m, metric, z_min_m=Z_MIN_M, z_max_m=Z_MAX_M,
            n_refine_levels=3, refine_divisions=8, smooth_sigma=1.0,
            max_evaluations=BUDGET)

    def _dist():
        return adaptive_distance_search(
            field, base, m, metric,
            initial_range_m=0.5e-3, max_range_m=50e-3,
            expand_factor=2.0, signal_threshold=0.3,
            max_evaluations=BUDGET)

    return [
        ("zscan", _zscan),
        ("coarse_to_fine", _coarse),
        ("robust", _robust),
        ("adaptive_gradient", _grad),
        ("adaptive_bracketing", _brack),
        ("adaptive_distance", _dist),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = _session_frames()
    if not frames:
        print("no session frames found under", DATA_ROOT)
        raise SystemExit(1)
    print(f"{len(frames)} frames × {len(METRICS)} metrics × 6 algorithms "
          f"(+ dense truth {DENSE_STEPS} steps)")

    results: List[Dict[str, Any]] = []
    for path in frames:
        session = path.parent.name
        print(f"\n== {session} / {path.name}", flush=True)
        field, base = _prepare_field(path)
        for metric_name in METRICS:
            metric = FocusMetric[metric_name]
            truth = _dense_truth(field, base, metric)
            print(f"  [{metric_name}] dense truth z={truth['z_mm']:+.2f} mm "
                  f"(prom={truth['prominence']:.2f}, "
                  f"{truth['runtime_ms']:.0f} ms)", flush=True)
            for name, fn in _algorithms(field, base, metric):
                t0 = time.monotonic()
                try:
                    res = fn()
                    dt = (time.monotonic() - t0) * 1000.0
                    z_mm = float(res.best_z_m) * 1e3
                    evals = int(getattr(res, "evaluations",
                                        getattr(res, "scanned", BUDGET)))
                    err = abs(z_mm - truth["z_mm"])
                    row = {
                        "session": session, "file": path.name,
                        "metric": metric_name, "algorithm": name,
                        "best_z_mm": round(z_mm, 3),
                        "err_mm": round(err, 3),
                        "evaluations": evals,
                        "runtime_ms": round(dt, 1),
                        "truth_z_mm": round(truth["z_mm"], 3),
                        "truth_prominence": round(truth["prominence"], 3),
                    }
                except Exception as exc:  # noqa: BLE001 — record, keep going
                    row = {
                        "session": session, "file": path.name,
                        "metric": metric_name, "algorithm": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results.append(row)
                if "error" in row:
                    print(f"    {name:20s} ERROR {row['error']}", flush=True)
                else:
                    print(f"    {name:20s} z={row['best_z_mm']:+8.2f} mm  "
                          f"err={row['err_mm']:6.2f} mm  "
                          f"evals={row['evaluations']:3d}  "
                          f"{row['runtime_ms']:7.0f} ms", flush=True)

    out = Path(args.out) if args.out else ROOT / "tasks" / "af_real_benchmark.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {len(results)} rows -> {out}")

    # Aggregate: per algorithm × metric — median err, hit-rate (err<=0.5mm),
    # median evals/runtime.
    print("\n=== aggregate (median over scenes) ===")
    print(f"{'algorithm':20s} {'metric':20s} {'med err':>8s} {'hit<=0.5':>9s} "
          f"{'med evals':>9s} {'med ms':>8s}")
    for metric_name in METRICS:
        for name in [a for a, _ in _algorithms(np.zeros((4, 4)), None, None)] \
                if False else ["zscan", "coarse_to_fine", "robust",
                               "adaptive_gradient", "adaptive_bracketing",
                               "adaptive_distance"]:
            rows = [r for r in results
                    if r.get("algorithm") == name
                    and r.get("metric") == metric_name and "err_mm" in r]
            if not rows:
                continue
            errs = sorted(r["err_mm"] for r in rows)
            hit = sum(1 for e in errs if e <= 0.5) / len(errs)
            evs = sorted(r["evaluations"] for r in rows)
            mss = sorted(r["runtime_ms"] for r in rows)
            print(f"{name:20s} {metric_name:20s} "
                  f"{errs[len(errs)//2]:8.2f} {hit:9.0%} "
                  f"{evs[len(evs)//2]:9d} {mss[len(mss)//2]:8.0f}")


if __name__ == "__main__":
    main()
