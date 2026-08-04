"""Evaluate Track C on the test split + LOSO cross-validation.

Two reports:

1. Held-out test session (default ``s9_B``) — quantitative metrics +
   per-frame visualization (3-panel: classical, predicted clean, GT
   clean) saved to ``_track_c_eval/<model_dir>/``.
2. Optional LOSO sweep (``--loso``): re-runs the full train+eval cycle
   for every session as held-out, summarized in a CSV. Useful for
   detecting session-specific generalization gaps.

Run::

    python3 scripts/eval_track_c.py --model models/track_c/v0.1/model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from recon_dl.dataset import DatasetConfig, make_loader  # noqa: E402
from recon_dl.unet_lite import UNetLite  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
_LOG = logging.getLogger("eval_track_c")


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _piston_align(arr: np.ndarray, target: np.ndarray) -> np.ndarray:
    return arr - (np.median(arr) - np.median(target))


def evaluate(model_path: Path, data_root: Path, *,
             split: str, crop_size: int, out_dir: Path,
             save_visualizations: int = 6) -> dict:
    device = _pick_device()
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    base_c = args.get("base_channels", 32)
    model = UNetLite(base_c=base_c).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    cfg = DatasetConfig(root=data_root, split=split,
                        crop_size=crop_size, augment=False)
    loader = make_loader(cfg, batch_size=1, shuffle=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    saved = 0

    t_total = time.monotonic()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x_dev = x.to(device)
            t0 = time.monotonic()
            corrected, residual_pred = model(x_dev)
            inf_ms = (time.monotonic() - t0) * 1000.0

            x_np = x.squeeze().numpy()
            y_np = y.squeeze().numpy()
            res_pred = residual_pred.squeeze().cpu().numpy()
            corrected_np = (x_np - res_pred)

            # GT clean = x - target_residual; CNN clean = x - predicted
            gt_clean = x_np - y_np
            cnn_clean = corrected_np

            # Piston-align CNN to GT before measuring error.
            cnn_aligned = _piston_align(cnn_clean, gt_clean)
            err = cnn_aligned - gt_clean

            classical_err = x_np - gt_clean  # = y_np = the target residual
            rmse_classical = float(np.sqrt(np.mean(classical_err ** 2)))
            p95_classical = float(np.percentile(np.abs(classical_err), 95))

            rmse_cnn = float(np.sqrt(np.mean(err ** 2)))
            p95_cnn = float(np.percentile(np.abs(err), 95))

            rows.append({
                "idx": i,
                "rmse_classical_rad": rmse_classical,
                "p95_classical_rad": p95_classical,
                "rmse_cnn_rad": rmse_cnn,
                "p95_cnn_rad": p95_cnn,
                "improvement_rmse": rmse_classical - rmse_cnn,
                "improvement_pct": (rmse_classical - rmse_cnn) /
                max(rmse_classical, 1e-6) * 100.0,
                "inference_ms": inf_ms,
            })

            if saved < save_visualizations:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=110)
                vmax = max(abs(gt_clean.min()), abs(gt_clean.max())) * 0.6
                for ax, arr, title in zip(
                        axes,
                        [x_np, gt_clean, cnn_aligned, err],
                        [f"classical poly5 (rmse={rmse_classical:.2f})",
                         "GT (synthetic-ref clean)",
                         f"CNN clean (rmse={rmse_cnn:.2f})",
                         f"err = CNN − GT (p95={p95_cnn:.2f})"]):
                    cmap = "RdBu_r"
                    if "err" in title:
                        im = ax.imshow(arr, cmap=cmap, vmin=-3, vmax=3)
                    else:
                        im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
                    ax.set_title(title)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_xticks([]); ax.set_yticks([])
                fig.tight_layout()
                fig.savefig(out_dir / f"frame_{i:02d}.png",
                            bbox_inches="tight")
                plt.close(fig)
                saved += 1

            _LOG.info("[%s] frame %02d: classical rmse=%.3f → cnn rmse=%.3f "
                      "(Δ=%+.3f, %.1f%% improve, %.1fms)",
                      split, i, rmse_classical, rmse_cnn,
                      rmse_classical - rmse_cnn,
                      rows[-1]["improvement_pct"], inf_ms)

    total_ms = (time.monotonic() - t_total) * 1000.0
    rmse_classical = np.array([r["rmse_classical_rad"] for r in rows])
    rmse_cnn = np.array([r["rmse_cnn_rad"] for r in rows])
    p95_cnn = np.array([r["p95_cnn_rad"] for r in rows])

    summary = {
        "split": split,
        "model": str(model_path),
        "n_frames": len(rows),
        "rmse_classical": {
            "median": float(np.median(rmse_classical)),
            "mean": float(np.mean(rmse_classical)),
            "max": float(np.max(rmse_classical)),
        },
        "rmse_cnn": {
            "median": float(np.median(rmse_cnn)),
            "mean": float(np.mean(rmse_cnn)),
            "max": float(np.max(rmse_cnn)),
        },
        "p95_cnn": {
            "median": float(np.median(p95_cnn)),
            "p95": float(np.percentile(p95_cnn, 95)),
        },
        "improvement_pct_median": float(np.median(
            [r["improvement_pct"] for r in rows])),
        "inference_ms_mean": float(np.mean(
            [r["inference_ms"] for r in rows])),
        "total_ms": float(total_ms),
        "rows": rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Single overview plot: per-frame bar of classical vs CNN RMSE.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, rmse_classical, 0.4, label="classical poly5", color="orange")
    ax.bar(idx + 0.2, rmse_cnn, 0.4, label="Track C (CNN-corrected)",
           color="steelblue")
    ax.set_xlabel(f"frame ({split})")
    ax.set_ylabel("RMSE vs GT (rad)")
    ax.set_title(f"Track C eval — {split}: "
                 f"median {summary['rmse_classical']['median']:.2f} → "
                 f"{summary['rmse_cnn']['median']:.2f} rad "
                 f"({summary['improvement_pct_median']:.1f}% improve)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", bbox_inches="tight")
    plt.close(fig)

    _LOG.info("[%s] DONE: classical median %.3f → CNN median %.3f rad "
              "(%.1f%% improvement)", split,
              summary["rmse_classical"]["median"],
              summary["rmse_cnn"]["median"],
              summary["improvement_pct_median"])
    _LOG.info("report: %s", out_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root",
                        default=os.environ.get(
                            "DHM_TRACK_C_DATASET",
                            str(REPO_ROOT / "data" / "rapor"
                                / "_track_c_dataset")))
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to _track_c_eval/<model_parent_name>/")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_root = Path(args.data_root)
    if args.out_dir is None:
        eval_root = REPO_ROOT.parent / "Reconstruction(Mac)_track_c_eval"
        eval_root = data_root.parent / "_track_c_eval"
        out_dir = eval_root / model_path.parent.name / args.split
    else:
        out_dir = Path(args.out_dir)

    evaluate(model_path, data_root,
             split=args.split, crop_size=args.crop_size, out_dir=out_dir)


if __name__ == "__main__":
    main()
