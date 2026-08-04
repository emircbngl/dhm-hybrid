"""Build Track C (hybrid CNN) training dataset.

For each non-outlier frame in the GT manifest, generate three arrays:

* ``phi_classical`` — reference-free reconstruction (polynomial bg-fit
  order 5) at the GT's z. Has the structured stripe residual we want
  the CNN to learn to remove.
* ``phi_target`` — reference-based reconstruction at the same z. The
  "clean" ground truth.
* ``residual = phi_classical - phi_target`` (after piston-align) —
  this is what the CNN predicts.

Output layout::

    <DATA_ROOT>/_track_c_dataset/
        index.json              # split metadata + per-frame paths
        s1/DHM_<...>.npz        # arrays for one frame
        s3/DHM_<...>.npz
        ...

Each ``.npz`` carries: ``phi_classical``, ``phi_target``, ``residual``,
``z_m``, ``session``, ``filename`` (string scalars are stored via numpy's
``np.array(..., dtype=str)`` so the file remains self-describing).

Splits:

* ``leave-one-session-out``: every session takes a turn as the held-out
  test split. For now we materialize a single split (session 9_B as
  held-out) plus include per-session arrays so a downstream LOSO loop
  can re-pick the split without regenerating data.

Run::

    python3 scripts/build_track_c_dataset.py \\
        --bg-method polynomial --bg-polynomial-order 5 \\
        --filter-z-outliers --z-outlier-tolerance-mm 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from core.ingestion import load_any  # noqa: E402

# Reuse the well-tested helpers from the benchmark script. They live in
# the same scripts/ directory so the relative import works once the path
# manipulation above is in place.
from benchmark_reffree import (  # noqa: E402
    DATA_ROOT,
    _demodulate,
    _is_z_outlier,
    _load_gt_z_map,
    _piston_align,
    build_manifest,
    reconstruct_in_memory,
)

OUT_ROOT = DATA_ROOT / "_track_c_dataset"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
_LOG = logging.getLogger("build_track_c")


def _save_frame(out_path: Path, *,
                phi_classical: np.ndarray,
                phi_target: np.ndarray,
                residual: np.ndarray,
                z_m: float,
                session: str,
                filename: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        phi_classical=phi_classical.astype(np.float32),
        phi_target=phi_target.astype(np.float32),
        residual=residual.astype(np.float32),
        z_m=np.float32(z_m),
        session=np.array(session, dtype=str),
        filename=np.array(filename, dtype=str),
    )


def _load_synthetic_refs(synth_root: Path,
                         fallback_within_lab: bool = True
                         ) -> dict[str, Path]:
    """Map session label → synthetic-ref image path. Sessions whose
    median synthesis was skipped (too few frames) fall back to a
    sibling session's synthetic ref, since they were captured in the
    same physical setup at the same lab session."""
    manifest_path = synth_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"synthetic-ref manifest missing at {manifest_path}; "
                         f"run scripts/build_synthetic_refs.py first")
    with open(manifest_path) as f:
        data = json.load(f)
    out: dict[str, Path] = {}
    for sess, entry in data.get("sessions", {}).items():
        if entry is None:
            continue
        out[sess] = synth_root.parent / entry["path"]
    if not fallback_within_lab:
        return out
    # Pragmatic fallback: s2 (only 2 frames) reuses s1's synthetic ref.
    # If you ever add more sessions where median synth fails, extend
    # this map (or change to a smarter "nearest-by-timestamp" lookup).
    fallback_map = {"s2": "s1"}
    for missing, donor in fallback_map.items():
        if missing not in out and donor in out:
            out[missing] = out[donor]
            _LOG.info("synthetic-ref fallback: %s ← %s", missing, donor)
    return out


def build_dataset(*,
                  bg_method: str,
                  bg_n_terms: int,
                  bg_polynomial_order: int,
                  filter_z_outliers: bool,
                  z_outlier_tolerance_mm: float,
                  gt_manifest: Optional[Path],
                  out_root: Path,
                  test_session: str = "s9_B",
                  val_session: str = "s9_A",
                  use_synthetic_refs: bool = True,
                  synthetic_ref_root: Optional[Path] = None) -> dict:
    plans = build_manifest()

    manifest_path = gt_manifest or (DATA_ROOT / "_output" / "manifest.json")
    gt_z_map = _load_gt_z_map(manifest_path)
    if not gt_z_map:
        raise SystemExit(f"GT manifest not found at {manifest_path} — cannot "
                         f"build dataset without per-frame z values")

    if use_synthetic_refs:
        synth_root = (synthetic_ref_root
                      or (DATA_ROOT / "_synthetic_refs"))
        synth_refs = _load_synthetic_refs(synth_root)
        _LOG.info("using synthetic refs for %d sessions: %s",
                  len(synth_refs), sorted(synth_refs))
    else:
        synth_refs = {}

    sess_zs: dict[str, list[float]] = {}
    for (sess, _file), z_m in gt_z_map.items():
        sess_zs.setdefault(sess, []).append(z_m * 1e3)

    if filter_z_outliers:
        outlier_keys = {k for k, z_m in gt_z_map.items()
                        if _is_z_outlier(z_m * 1e3, sess_zs[k[0]],
                                         z_outlier_tolerance_mm)}
        _LOG.info("filtering %d/%d frames as z-outliers",
                  len(outlier_keys), len(gt_z_map))
    else:
        outlier_keys = set()

    out_root.mkdir(parents=True, exist_ok=True)

    index: dict = {
        "config": {
            "bg_method": bg_method,
            "bg_n_terms": bg_n_terms,
            "bg_polynomial_order": bg_polynomial_order,
            "filter_z_outliers": filter_z_outliers,
            "z_outlier_tolerance_mm": z_outlier_tolerance_mm,
            "gt_manifest": str(manifest_path),
        },
        "frames": [],
        "splits": {"train": [], "val": [], "test": []},
        "stats": {
            "total_frames": 0,
            "skipped_outliers": 0,
            "skipped_no_ref": 0,
            "skipped_failed": 0,
        },
    }

    t_start = time.monotonic()
    for plan in plans:
        # Pick the reference: synthetic median (clean, recommended) or the
        # original session's reference (user noted it contains bacteria).
        ref_path: Optional[Path] = None
        ref_kind: str = "none"
        if use_synthetic_refs and plan.label in synth_refs:
            ref_path = synth_refs[plan.label]
            ref_kind = "synthetic"
        elif plan.ref_path is not None and plan.ref_path.exists():
            ref_path = plan.ref_path
            ref_kind = "original"
        if ref_path is None:
            _LOG.warning("[%s] no reference of any kind, skipping", plan.label)
            index["stats"]["skipped_no_ref"] += len(plan.files)
            continue
        _LOG.info("[%s] ref = %s (%s)", plan.label, ref_path.name, ref_kind)
        ref_raw = np.asarray(load_any(ref_path).array)
        ref_fc = _demodulate(ref_raw)

        for holo in plan.files:
            key = (plan.label, holo.name)
            if key in outlier_keys:
                index["stats"]["skipped_outliers"] += 1
                continue
            if key not in gt_z_map:
                _LOG.warning("[%s] %s missing from GT manifest, skipping",
                             plan.label, holo.name)
                index["stats"]["skipped_outliers"] += 1
                continue

            z_m = gt_z_map[key]
            try:
                raw = np.asarray(load_any(holo).array)
                holo_fc = _demodulate(raw)

                phi_target, _, _ = reconstruct_in_memory(
                    holo_fc, ref_fc, fixed_z_m=z_m,
                )
                phi_classical, _, _ = reconstruct_in_memory(
                    holo_fc, None,
                    bg_method=bg_method,
                    bg_n_terms=bg_n_terms,
                    bg_polynomial_order=bg_polynomial_order,
                    fixed_z_m=z_m,
                )
                aligned, _ = _piston_align(phi_classical, phi_target)
                residual = aligned - phi_target

                rel_path = Path(plan.label) / (holo.stem + ".npz")
                _save_frame(
                    out_root / rel_path,
                    phi_classical=aligned,
                    phi_target=phi_target,
                    residual=residual,
                    z_m=z_m,
                    session=plan.label,
                    filename=holo.name,
                )

                rmse = float(np.sqrt(np.mean(residual ** 2)))
                p95 = float(np.percentile(np.abs(residual), 95))

                entry = {
                    "session": plan.label,
                    "filename": holo.name,
                    "path": str(rel_path),
                    "z_mm": z_m * 1e3,
                    "residual_rmse_rad": rmse,
                    "residual_p95_abs_rad": p95,
                    "shape": list(residual.shape),
                }
                index["frames"].append(entry)

                if plan.label == test_session:
                    index["splits"]["test"].append(entry["path"])
                elif plan.label == val_session:
                    index["splits"]["val"].append(entry["path"])
                else:
                    index["splits"]["train"].append(entry["path"])

                _LOG.info("  [%s] %s shape=%s z=%+.2fmm rmse=%.3f p95=%.3f",
                          plan.label, holo.name, residual.shape,
                          z_m * 1e3, rmse, p95)
            except Exception as exc:
                _LOG.exception("  [%s] %s FAILED: %s",
                               plan.label, holo.name, exc)
                index["stats"]["skipped_failed"] += 1

    index["stats"]["total_frames"] = len(index["frames"])
    index["stats"]["build_runtime_s"] = time.monotonic() - t_start

    rmses = [f["residual_rmse_rad"] for f in index["frames"]]
    if rmses:
        index["stats"]["residual_rmse_median_rad"] = float(np.median(rmses))
        index["stats"]["residual_rmse_mean_rad"] = float(np.mean(rmses))
        index["stats"]["residual_rmse_max_rad"] = float(np.max(rmses))

    out_root.mkdir(exist_ok=True)
    with open(out_root / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    _LOG.info("DONE — %d frames in %.1fs",
              index["stats"]["total_frames"],
              index["stats"]["build_runtime_s"])
    _LOG.info("splits: train=%d val=%d test=%d",
              len(index["splits"]["train"]),
              len(index["splits"]["val"]),
              len(index["splits"]["test"]))
    if rmses:
        _LOG.info("residual rmse: median=%.3f mean=%.3f max=%.3f rad",
                  index["stats"]["residual_rmse_median_rad"],
                  index["stats"]["residual_rmse_mean_rad"],
                  index["stats"]["residual_rmse_max_rad"])
    _LOG.info("index: %s", out_root / "index.json")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bg-method", choices=["zernike", "polynomial"],
                        default="polynomial")
    parser.add_argument("--bg-n-terms", type=int, default=15)
    parser.add_argument("--bg-polynomial-order", type=int, default=5)
    parser.add_argument("--filter-z-outliers", action="store_true",
                        default=True)
    parser.add_argument("--no-filter-z-outliers", dest="filter_z_outliers",
                        action="store_false")
    parser.add_argument("--z-outlier-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--gt-manifest", default=None,
                        help="Path to GT manifest.json. Defaults to "
                        "<DATA_ROOT>/_output/manifest.json.")
    parser.add_argument("--out-dir", default=str(OUT_ROOT))
    parser.add_argument("--test-session", default="s9_B")
    parser.add_argument("--val-session", default="s9_A")
    parser.add_argument("--use-synthetic-refs", action="store_true",
                        default=True,
                        help="Use temporal-median synthetic refs (clean) "
                        "instead of the saved per-session reference frames "
                        "(which contain bacteria). Default: ON.")
    parser.add_argument("--no-synthetic-refs", dest="use_synthetic_refs",
                        action="store_false")
    parser.add_argument("--synthetic-ref-root", default=None)
    args = parser.parse_args()

    build_dataset(
        bg_method=args.bg_method,
        bg_n_terms=args.bg_n_terms,
        bg_polynomial_order=args.bg_polynomial_order,
        filter_z_outliers=args.filter_z_outliers,
        z_outlier_tolerance_mm=args.z_outlier_tolerance_mm,
        gt_manifest=Path(args.gt_manifest) if args.gt_manifest else None,
        out_root=Path(args.out_dir),
        test_session=args.test_session,
        val_session=args.val_session,
        use_synthetic_refs=args.use_synthetic_refs,
        synthetic_ref_root=(Path(args.synthetic_ref_root)
                            if args.synthetic_ref_root else None),
    )


if __name__ == "__main__":
    main()
