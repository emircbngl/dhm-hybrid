"""PyTorch Dataset for Track C training pairs.

Reads the ``index.json`` produced by ``scripts/build_track_c_dataset.py``
and lazily loads ``.npz`` files. Returns ``(input, target)`` where:

* ``input`` is ``phi_classical`` (poly5 reffree output, with stripes)
* ``target`` is the residual the network should predict

Both are float32 tensors shaped ``(1, H, W)`` after the optional
crop/augment pipeline.

Augmentation strategy (training only):

* Random crop to a fixed size (default 768x768) — handles the native
  1200x1600 frames and gives the loader enough variety to keep the
  small dataset from overfitting.
* Random horizontal/vertical flip — stripe pattern is symmetric enough
  that flips don't break the physics.
* Random 90° rotation — likewise.
* Random piston offset — adds a constant to the input (the residual
  target is invariant to this since target = input - clean and the
  piston cancels). Teaches the net that absolute phase offset is
  irrelevant; only the structured pattern matters.

Validation/test use a deterministic centre crop and no augmentation.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_LOG = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    root: Path
    split: str = "train"             # "train" | "val" | "test"
    crop_size: int = 768
    augment: bool = True
    piston_jitter_rad: float = 1.0   # uniform[-x, x] added to input only


class TrackCDataset(Dataset):
    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        index_path = cfg.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"missing dataset index: {index_path}")
        with open(index_path) as f:
            self.index = json.load(f)
        if cfg.split not in self.index["splits"]:
            raise KeyError(f"unknown split '{cfg.split}'; "
                           f"available: {list(self.index['splits'])}")
        self.paths = [cfg.root / p for p in self.index["splits"][cfg.split]]
        if not self.paths:
            raise RuntimeError(f"split '{cfg.split}' is empty")
        _LOG.info("TrackCDataset[%s] %d frames, crop=%d, augment=%s",
                  cfg.split, len(self.paths), cfg.crop_size, cfg.augment)

    def __len__(self) -> int:
        return len(self.paths)

    def _crop(self, x: np.ndarray, y: np.ndarray,
              top: int, left: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.cfg.crop_size
        return (x[top:top + s, left:left + s],
                y[top:top + s, left:left + s])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        npz = np.load(self.paths[idx])
        phi_classical = npz["phi_classical"].astype(np.float32)
        residual = npz["residual"].astype(np.float32)
        h, w = phi_classical.shape
        s = self.cfg.crop_size
        if h < s or w < s:
            # Pad with edge replication if the frame is smaller than crop.
            pad_h = max(0, s - h)
            pad_w = max(0, s - w)
            phi_classical = np.pad(phi_classical,
                                   ((0, pad_h), (0, pad_w)), mode="edge")
            residual = np.pad(residual,
                              ((0, pad_h), (0, pad_w)), mode="edge")
            h, w = phi_classical.shape

        if self.cfg.augment and self.cfg.split == "train":
            top = random.randint(0, h - s)
            left = random.randint(0, w - s)
            x, y = self._crop(phi_classical, residual, top, left)
            if random.random() < 0.5:
                x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
            if random.random() < 0.5:
                x, y = x[::-1, :].copy(), y[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                x = np.rot90(x, k).copy()
                y = np.rot90(y, k).copy()
            if self.cfg.piston_jitter_rad > 0:
                jitter = random.uniform(
                    -self.cfg.piston_jitter_rad, self.cfg.piston_jitter_rad)
                x = x + jitter
                # residual target unchanged: target = input - clean, so a
                # constant on input cancels in the difference.
        else:
            top = (h - s) // 2
            left = (w - s) // 2
            x, y = self._crop(phi_classical, residual, top, left)

        x_t = torch.from_numpy(x).unsqueeze(0)
        y_t = torch.from_numpy(y).unsqueeze(0)
        return x_t, y_t


def make_loader(cfg: DatasetConfig, *, batch_size: int,
                num_workers: int = 0,
                shuffle: Optional[bool] = None) -> "torch.utils.data.DataLoader":
    if shuffle is None:
        shuffle = (cfg.split == "train")
    ds = TrackCDataset(cfg)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )
