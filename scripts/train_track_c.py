"""Train the Track C residual U-Net on the prebuilt dataset.

Usage::

    python3 scripts/train_track_c.py \\
        --epochs 200 --batch-size 2 --lr 3e-4

Outputs to ``models/track_c/v0.1/``:

* ``model.pt`` — best checkpoint (lowest val RMSE)
* ``train_log.json`` — per-epoch train+val loss + val RMSE/p95
* ``config.yaml`` — exact run config for reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_dl.dataset import DatasetConfig, make_loader  # noqa: E402
from recon_dl.losses import CombinedLoss  # noqa: E402
from recon_dl.unet_lite import UNetLite, count_parameters  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
_LOG = logging.getLogger("train_track_c")


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _epoch(model: nn.Module, loader, loss_fn, opt, device,
           train: bool) -> dict:
    model.train(train)
    sum_loss = 0.0
    sum_l1 = 0.0
    sum_tv = 0.0
    sum_pred_rmse = 0.0   # rmse(predicted residual, target residual)
    sum_resid_rmse = 0.0  # rmse(corrected, clean) — the metric we care about
    sum_p95 = 0.0
    n = 0
    for batch_i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            corrected, residual_pred = model(x)
            loss, parts = loss_fn(residual_pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                _LOG.error("non-finite loss at %s batch %d: loss=%s "
                           "(x_range=[%.2f,%.2f] y_range=[%.2f,%.2f] "
                           "pred_range=[%.2f,%.2f] x_nan=%s y_nan=%s "
                           "pred_nan=%s)",
                           "train" if train else "val", batch_i, float(loss),
                           float(x.min()), float(x.max()),
                           float(y.min()), float(y.max()),
                           float(residual_pred.min()),
                           float(residual_pred.max()),
                           bool(torch.isnan(x).any()),
                           bool(torch.isnan(y).any()),
                           bool(torch.isnan(residual_pred).any()))
                continue
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        # diagnostic metrics
        with torch.no_grad():
            err = residual_pred - y
            pred_rmse = torch.sqrt(torch.mean(err * err))
            # corrected = x - pred_residual; clean = x - y; so corrected -
            # clean = y - pred_residual = -err. Same RMSE as pred_rmse,
            # but conceptually clearer.
            resid_after = (corrected - (x - y))
            resid_rmse = torch.sqrt(torch.mean(resid_after * resid_after))
            p95 = torch.quantile(resid_after.abs().reshape(-1), 0.95)

        bs = x.shape[0]
        sum_loss += parts["loss"] * bs
        sum_l1 += parts["l1"] * bs
        sum_tv += parts["tv"] * bs
        sum_pred_rmse += float(pred_rmse) * bs
        sum_resid_rmse += float(resid_rmse) * bs
        sum_p95 += float(p95) * bs
        n += bs
    return {
        "loss": sum_loss / n,
        "l1": sum_l1 / n,
        "tv": sum_tv / n,
        "pred_rmse_rad": sum_pred_rmse / n,
        "corrected_rmse_rad": sum_resid_rmse / n,
        "corrected_p95_rad": sum_p95 / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root",
                        default=os.environ.get(
                            "DHM_TRACK_C_DATASET",
                            str(REPO_ROOT / "data" / "rapor"
                                / "_track_c_dataset")))
    parser.add_argument("--out-dir",
                        default=str(REPO_ROOT / "models" / "track_c" / "v0.1"))
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--tv-weight", type=float, default=0.05)
    parser.add_argument("--piston-jitter-rad", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = _pick_device()
    _LOG.info("device: %s", device)

    train_cfg = DatasetConfig(
        root=Path(args.data_root), split="train",
        crop_size=args.crop_size, augment=True,
        piston_jitter_rad=args.piston_jitter_rad,
    )
    val_cfg = DatasetConfig(
        root=Path(args.data_root), split="val",
        crop_size=args.crop_size, augment=False,
    )
    train_loader = make_loader(train_cfg, batch_size=args.batch_size,
                               num_workers=args.num_workers)
    val_loader = make_loader(val_cfg, batch_size=1,
                             num_workers=args.num_workers, shuffle=False)

    model = UNetLite(base_c=args.base_channels).to(device)
    _LOG.info("model: UNetLite base=%d params=%d", args.base_channels,
              count_parameters(model))

    loss_fn = CombinedLoss(l1_weight=1.0, tv_weight=args.tv_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)

    log: list[dict] = []
    best_val_rmse = float("inf")
    epochs_without_improvement = 0

    t_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        train_stats = _epoch(model, train_loader, loss_fn, opt, device, True)
        val_stats = _epoch(model, val_loader, loss_fn, opt, device, False)
        scheduler.step()

        epoch_time = time.monotonic() - t0
        log.append({
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "lr": opt.param_groups[0]["lr"],
            "time_s": epoch_time,
        })
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)

        improved = val_stats["corrected_rmse_rad"] < best_val_rmse
        if improved:
            best_val_rmse = val_stats["corrected_rmse_rad"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_rmse_rad": best_val_rmse,
                "args": vars(args),
            }, out_dir / "model.pt")
        else:
            epochs_without_improvement += 1

        _LOG.info("ep %3d/%d  train l1=%.3f tv=%.4f rmse=%.3f | "
                  "val rmse=%.3f p95=%.3f%s  (%.1fs, %s)",
                  epoch, args.epochs,
                  train_stats["l1"], train_stats["tv"],
                  train_stats["corrected_rmse_rad"],
                  val_stats["corrected_rmse_rad"],
                  val_stats["corrected_p95_rad"],
                  "  ★" if improved else "",
                  epoch_time,
                  f"lr={opt.param_groups[0]['lr']:.2e}")

        if epochs_without_improvement >= args.early_stop_patience:
            _LOG.info("early stop at epoch %d (no improvement for %d epochs)",
                      epoch, args.early_stop_patience)
            break

    total_time = time.monotonic() - t_start
    _LOG.info("DONE — best val rmse=%.3f rad in %.1fs",
              best_val_rmse, total_time)
    _LOG.info("checkpoint: %s", out_dir / "model.pt")


if __name__ == "__main__":
    main()
