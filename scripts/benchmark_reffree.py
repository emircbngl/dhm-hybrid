"""Benchmark reference-free reconstruction against the reference-based pipeline.

For every frame in :func:`run_rapor_data_batch.build_manifest`, this script
runs the recon pipeline twice in memory:

* **ref**: classical sample/ref complex division at each trial z (the
  pipeline used for `_output/`).
* **reffree**: skip the reference, fit and subtract a Zernike (or
  polynomial) background after unwrapping.

Both go through the same autofocus, demodulation, propagation, and
unwrap chain; the only difference is whether the reference contributes.

The unwrapped phase arrays are compared per-pixel after a constant
piston shift is removed (a global offset is meaningless for QPI). Output
is a CSV plus a histogram and per-session boxplot, written under
``<DATA_ROOT>/_benchmark_reffree/``.

Run modes:
  python3 scripts/benchmark_reffree.py                   # all sessions, full
  python3 scripts/benchmark_reffree.py --only s1,s7      # subset
  python3 scripts/benchmark_reffree.py --limit 2         # first 2/session
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.ingestion import load_any  # noqa: E402
# The demod / autofocus / propagate / unwrap chain now lives in core so
# there is ONE implementation (2026-07-05 review — it had drifted into
# 3-4 copies). This script keeps its thin ``_demodulate`` / ``reconstruct_
# in_memory`` names (build_track_c_dataset imports them) as wrappers bound
# to the lab optics.
from core.pipelines.reffree_hybrid import (  # noqa: E402
    OpticalConfig,
    demodulate as _core_demodulate,
    piston_align as _piston_align,
    propagate_field as _propagate_at,
    reconstruct_in_memory as _core_reconstruct_in_memory,
)

# Import shared constants and the manifest builder from the batch driver.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_rapor_data_batch import (  # noqa: E402
    CAMERA_PIX_UM,
    DATA_ROOT,
    EFFECTIVE_M,
    EFFECTIVE_UM,
    MAGNIFICATION,
    MASK_RADIUS,
    N_MEDIUM,
    WAVELENGTH_M,
    Z_RANGE_MM,
    Z_STEPS,
    SessionPlan,
    build_manifest,
)

_LOG = logging.getLogger("benchmark_reffree")

OUT_ROOT = DATA_ROOT / "_benchmark_reffree"

# One config built from the lab constants; the core pipeline is pure and
# takes it explicitly.
_OPTICS = OpticalConfig(
    wavelength_m=WAVELENGTH_M, effective_pixel_m=EFFECTIVE_M,
    n_medium=N_MEDIUM, mask_radius=MASK_RADIUS,
    z_range_mm=Z_RANGE_MM, z_steps=Z_STEPS,
)


# ---------------------------------------------------------------------------
# In-memory pipeline — thin wrappers over core.pipelines.reffree_hybrid
# ---------------------------------------------------------------------------

def _demodulate(raw: np.ndarray, mask_radius: int = MASK_RADIUS):
    cfg = _OPTICS if mask_radius == MASK_RADIUS else OpticalConfig(
        wavelength_m=WAVELENGTH_M, effective_pixel_m=EFFECTIVE_M,
        n_medium=N_MEDIUM, mask_radius=mask_radius,
        z_range_mm=Z_RANGE_MM, z_steps=Z_STEPS,
    )
    return _core_demodulate(raw, cfg)


def reconstruct_in_memory(
    holo_fc: np.ndarray,
    ref_fc: Optional[np.ndarray],
    bg_method: str = "zernike",
    bg_n_terms: int = 15,
    bg_polynomial_order: int = 4,
    fixed_z_m: Optional[float] = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    return _core_reconstruct_in_memory(
        holo_fc, ref_fc, _OPTICS,
        bg_method=bg_method, bg_n_terms=bg_n_terms,
        bg_polynomial_order=bg_polynomial_order, fixed_z_m=fixed_z_m,
    )


# ---------------------------------------------------------------------------
# Comparison metric
# ---------------------------------------------------------------------------

def compare(unwrapped_ref: np.ndarray,
            unwrapped_reffree: np.ndarray) -> dict:
    """Per-pixel RMSE / abs error / max diff after piston alignment."""
    aligned, piston = _piston_align(unwrapped_reffree, unwrapped_ref)
    diff = aligned - unwrapped_ref
    finite = np.isfinite(diff)
    diff_f = diff[finite]
    return {
        "piston_shift_rad": piston,
        "rmse_rad": float(np.sqrt(np.mean(diff_f ** 2))) if diff_f.size else float("nan"),
        "mean_abs_rad": float(np.mean(np.abs(diff_f))) if diff_f.size else float("nan"),
        "max_abs_rad": float(np.max(np.abs(diff_f))) if diff_f.size else float("nan"),
        "p50_abs_rad": float(np.percentile(np.abs(diff_f), 50)) if diff_f.size else float("nan"),
        "p95_abs_rad": float(np.percentile(np.abs(diff_f), 95)) if diff_f.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class BenchRow:
    session: str
    file: str
    ref_z_mm: float
    reffree_z_mm: float
    rmse_rad: float
    mean_abs_rad: float
    p50_abs_rad: float
    p95_abs_rad: float
    max_abs_rad: float
    piston_shift_rad: float
    runtime_ref_s: float
    runtime_reffree_s: float


def _load_gt_z_map(manifest_path: Path) -> dict[tuple[str, str], float]:
    """Map (session, file) → best_z_m from the GT manifest.json saved by
    the reference-based batch run. Returns {} if file missing."""
    if not manifest_path.exists():
        return {}
    import json
    with open(manifest_path) as f:
        data = json.load(f)
    return {(r["session"], r["file"]): float(r["best_z_mm"]) * 1e-3
            for r in data}


def _is_z_outlier(z_mm: float, session_zs: list[float],
                  tolerance_mm: float = 5.0) -> bool:
    """A frame's z is an outlier if it sits in the opposite half-space
    from the session median, or more than tolerance_mm away from it."""
    if not session_zs:
        return False
    med = float(np.median(session_zs))
    return abs(z_mm - med) > tolerance_mm or (z_mm * med < 0)


def run_benchmark(only: Optional[set[str]] = None,
                  limit_per_session: Optional[int] = None,
                  bg_method: str = "zernike",
                  bg_n_terms: int = 15,
                  bg_polynomial_order: int = 4,
                  fixed_z_from_gt: bool = False,
                  filter_z_outliers: bool = False,
                  z_outlier_tolerance_mm: float = 5.0,
                  gt_manifest: Optional[Path] = None) -> list[BenchRow]:
    plans = build_manifest()
    if only:
        plans = [p for p in plans if p.label in only]

    gt_z_map: dict[tuple[str, str], float] = {}
    if fixed_z_from_gt or filter_z_outliers:
        manifest_path = gt_manifest or (DATA_ROOT / "_output" / "manifest.json")
        gt_z_map = _load_gt_z_map(manifest_path)
        if not gt_z_map:
            _LOG.warning("GT manifest not found at %s — fixed-z and "
                         "outlier filter disabled", manifest_path)
            fixed_z_from_gt = False
            filter_z_outliers = False
        else:
            _LOG.info("loaded GT z map: %d entries from %s",
                      len(gt_z_map), manifest_path)

    if filter_z_outliers and gt_z_map:
        sess_zs: dict[str, list[float]] = {}
        for (sess, _file), z_m in gt_z_map.items():
            sess_zs.setdefault(sess, []).append(z_m * 1e3)
        outlier_keys = {k for k, z_m in gt_z_map.items()
                        if _is_z_outlier(z_m * 1e3, sess_zs[k[0]],
                                         z_outlier_tolerance_mm)}
        _LOG.info("filtering %d/%d frames as z-outliers (tol=%.1f mm)",
                  len(outlier_keys), len(gt_z_map), z_outlier_tolerance_mm)
    else:
        outlier_keys = set()

    rows: list[BenchRow] = []
    total_after_filter = 0
    for plan in plans:
        files = plan.files[:limit_per_session] if limit_per_session else plan.files
        for holo in files:
            if (plan.label, holo.name) not in outlier_keys:
                total_after_filter += 1
    done = 0

    for plan in plans:
        if plan.ref_path is None or not plan.ref_path.exists():
            _LOG.warning("[%s] no usable reference, skipping for benchmark",
                         plan.label)
            continue
        files = plan.files[:limit_per_session] if limit_per_session else plan.files
        if not files:
            continue

        _LOG.info("[%s] %d frame(s), ref=%s",
                  plan.label, len(files), plan.ref_path.name)
        ref_raw = np.asarray(load_any(plan.ref_path).array)
        ref_fc = _demodulate(ref_raw)

        for holo in files:
            key = (plan.label, holo.name)
            if key in outlier_keys:
                _LOG.info("  skip %s — GT z is an outlier", holo.name)
                continue
            done += 1
            try:
                raw = np.asarray(load_any(holo).array)
                holo_fc = _demodulate(raw)

                # Optionally lock both reconstructions to the GT z, so the
                # comparison measures only the background-subtraction step
                # (not autofocus disagreement).
                fixed_z = gt_z_map.get(key) if fixed_z_from_gt else None

                t0 = time.monotonic()
                u_ref, z_ref, _ = reconstruct_in_memory(
                    holo_fc, ref_fc, fixed_z_m=fixed_z,
                )
                rt_ref = time.monotonic() - t0

                t0 = time.monotonic()
                u_reffree, z_reffree, _ = reconstruct_in_memory(
                    holo_fc, None,
                    bg_method=bg_method,
                    bg_n_terms=bg_n_terms,
                    bg_polynomial_order=bg_polynomial_order,
                    fixed_z_m=fixed_z,
                )
                rt_reffree = time.monotonic() - t0

                metrics = compare(u_ref, u_reffree)
                rows.append(BenchRow(
                    session=plan.label,
                    file=holo.name,
                    ref_z_mm=z_ref * 1e3,
                    reffree_z_mm=z_reffree * 1e3,
                    rmse_rad=metrics["rmse_rad"],
                    mean_abs_rad=metrics["mean_abs_rad"],
                    p50_abs_rad=metrics["p50_abs_rad"],
                    p95_abs_rad=metrics["p95_abs_rad"],
                    max_abs_rad=metrics["max_abs_rad"],
                    piston_shift_rad=metrics["piston_shift_rad"],
                    runtime_ref_s=rt_ref,
                    runtime_reffree_s=rt_reffree,
                ))
                _LOG.info("  [%d/%d] %s rmse=%.3f rad p95=%.3f rad "
                          "z_ref=%+.3f mm z_reffree=%+.3f mm",
                          done, total_after_filter, holo.name,
                          metrics["rmse_rad"], metrics["p95_abs_rad"],
                          z_ref * 1e3, z_reffree * 1e3)
            except Exception as exc:
                _LOG.exception("  [%d/%d] %s FAILED: %s",
                               done, total_after_filter, holo.name, exc)

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(rows: list[BenchRow], path: Path) -> None:
    fields = list(BenchRow.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def write_summary_plot(rows: list[BenchRow], path: Path) -> None:
    rmse = np.array([r.rmse_rad for r in rows])
    sessions = [r.session for r in rows]
    p95 = np.array([r.p95_abs_rad for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=110)

    axes[0].hist(rmse, bins=20, color="#4c72b0", edgecolor="black")
    axes[0].set_xlabel("RMSE (rad)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"RMSE histogram (n={len(rows)})\n"
                      f"median={np.median(rmse):.3f}  p95={np.percentile(rmse, 95):.3f}")
    axes[0].axvline(0.15, color="red", linestyle="--", alpha=0.5,
                    label="Track A target (0.15)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sess_unique = sorted(set(sessions))
    sess_data = [rmse[np.array(sessions) == s] for s in sess_unique]
    axes[1].boxplot(sess_data, tick_labels=sess_unique)
    axes[1].set_ylabel("RMSE (rad)")
    axes[1].set_title("RMSE by session")
    axes[1].axhline(0.15, color="red", linestyle="--", alpha=0.5)
    axes[1].grid(alpha=0.3)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    z_ref = np.array([r.ref_z_mm for r in rows])
    z_reffree = np.array([r.reffree_z_mm for r in rows])
    axes[2].scatter(z_ref, z_reffree, s=18, alpha=0.7, c="#4c72b0")
    lo = float(min(z_ref.min(), z_reffree.min()))
    hi = float(max(z_ref.max(), z_reffree.max()))
    axes[2].plot([lo, hi], [lo, hi], color="red", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("ref-based z (mm)")
    axes[2].set_ylabel("reference-free z (mm)")
    axes[2].set_title("Autofocus z agreement")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",
                        help="Comma-separated session labels (e.g. 's1,s2').")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap frames per session.")
    parser.add_argument("--bg-method", choices=["zernike", "polynomial"],
                        default="zernike")
    parser.add_argument("--bg-n-terms", type=int, default=15)
    parser.add_argument("--bg-polynomial-order", type=int, default=4)
    parser.add_argument("--fixed-z-from-gt", action="store_true",
                        help="Use GT manifest's z for both pipelines instead "
                        "of running autofocus. Isolates the bg-subtraction "
                        "step from autofocus disagreement.")
    parser.add_argument("--filter-z-outliers", action="store_true",
                        help="Drop frames whose GT z is in the opposite "
                        "half-space from the session median or > tolerance "
                        "away from it (the GT autofocus picked wrong z).")
    parser.add_argument("--z-outlier-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--gt-manifest", default=None,
                        help="Path to GT manifest.json. Defaults to "
                        "<DATA_ROOT>/_output/manifest.json.")
    parser.add_argument("--out-dir", default=str(OUT_ROOT))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    only = set(args.only.split(",")) if args.only else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    rows = run_benchmark(
        only=only, limit_per_session=args.limit,
        bg_method=args.bg_method,
        bg_n_terms=args.bg_n_terms,
        bg_polynomial_order=args.bg_polynomial_order,
        fixed_z_from_gt=args.fixed_z_from_gt,
        filter_z_outliers=args.filter_z_outliers,
        z_outlier_tolerance_mm=args.z_outlier_tolerance_mm,
        gt_manifest=Path(args.gt_manifest) if args.gt_manifest else None,
    )
    elapsed = time.monotonic() - t0

    if not rows:
        _LOG.error("no rows produced — nothing to summarise")
        return

    csv_path = out_dir / "benchmark_reffree.csv"
    plot_path = out_dir / "benchmark_reffree.png"
    write_csv(rows, csv_path)
    write_summary_plot(rows, plot_path)

    rmse = np.array([r.rmse_rad for r in rows])
    p95 = np.array([r.p95_abs_rad for r in rows])
    z_diff_mm = np.array([abs(r.ref_z_mm - r.reffree_z_mm) for r in rows])

    _LOG.info("DONE — %d frames in %.1fs", len(rows), elapsed)
    _LOG.info("rmse:        median=%.3f  mean=%.3f  p95=%.3f  max=%.3f rad",
              float(np.median(rmse)), float(np.mean(rmse)),
              float(np.percentile(rmse, 95)), float(rmse.max()))
    _LOG.info("p95 |err|:   median=%.3f  p95=%.3f rad",
              float(np.median(p95)), float(np.percentile(p95, 95)))
    _LOG.info("z disagree:  median=%.3f mm  p95=%.3f mm",
              float(np.median(z_diff_mm)), float(np.percentile(z_diff_mm, 95)))
    _LOG.info("csv: %s  plot: %s", csv_path, plot_path)

    target = 0.15
    pass_rate = float(np.mean(rmse < target))
    _LOG.info("Track A target (rmse<%.2f rad): %.1f%% of frames pass",
              target, pass_rate * 100)


if __name__ == "__main__":
    main()
