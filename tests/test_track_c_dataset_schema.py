"""Torch-free schema/round-trip tests for the Track C dataset on-disk format.

``scripts/build_track_c_dataset.py`` writes ``index.json`` (see
``build_dataset()``) plus one ``.npz`` per frame (see ``_save_frame()``):

* ``index.json``: ``{"config": {...}, "frames": [...], "splits":
  {"train": [...], "val": [...], "test": [...]}, "stats": {...}}``. Each
  ``splits[<split>]`` entry is a path *relative to the dataset root*
  (``<session>/<stem>.npz``).
* ``<session>/<stem>.npz``: ``phi_classical`` (float32 2D), ``phi_target``
  (float32 2D), ``residual`` (float32 2D, = phi_classical - phi_target
  after piston-align), ``z_m`` (float32 scalar), ``session`` (str
  scalar), ``filename`` (str scalar).

This file only exercises the on-disk *shape* of that contract with pure
numpy/json — no torch import, so it runs in the local venv (which lacks
torch) as well as CI. ``recon_dl.dataset.TrackCDataset`` itself imports
torch at module scope, so the loader-behaviour test lives in
``test_track_c_pipeline.py`` (torch-gated) instead of here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _write_synthetic_track_c_dataset(root: Path, *,
                                     shape: tuple[int, int] = (192, 256),
                                     seed: int = 0) -> None:
    """Build a tiny fixture matching build_track_c_dataset.py's schema.

    Two frames in "s1" (train) and one frame in "s9_A" (val) — enough to
    exercise both non-empty splits without a real dataset on disk.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    frames = [
        ("s1", "DHM_0001.npz", "train"),
        ("s1", "DHM_0002.npz", "train"),
        ("s9_A", "DHM_0100.npz", "val"),
    ]

    index: dict = {
        "config": {
            "bg_method": "polynomial",
            "bg_n_terms": 15,
            "bg_polynomial_order": 5,
            "filter_z_outliers": True,
            "z_outlier_tolerance_mm": 5.0,
            "gt_manifest": str(root / "_output" / "manifest.json"),
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

    for session, filename, split in frames:
        phi_target = (0.3 * rng.standard_normal(shape)).astype(np.float32)
        stripes = (0.1 * np.sin(np.linspace(0, 8 * np.pi, w))
                   .astype(np.float32))
        phi_classical = phi_target + stripes[None, :].repeat(h, axis=0)
        residual = phi_classical - phi_target
        z_m = float(rng.uniform(-0.01, 0.01))

        rel_path = Path(session) / filename
        out_path = root / rel_path
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

        rmse = float(np.sqrt(np.mean(residual ** 2)))
        p95 = float(np.percentile(np.abs(residual), 95))
        entry = {
            "session": session,
            "filename": filename,
            "path": str(rel_path),
            "z_mm": z_m * 1e3,
            "residual_rmse_rad": rmse,
            "residual_p95_abs_rad": p95,
            "shape": list(residual.shape),
        }
        index["frames"].append(entry)
        index["splits"][split].append(entry["path"])

    index["stats"]["total_frames"] = len(index["frames"])
    rmses = [f["residual_rmse_rad"] for f in index["frames"]]
    index["stats"]["residual_rmse_median_rad"] = float(np.median(rmses))
    index["stats"]["residual_rmse_mean_rad"] = float(np.mean(rmses))
    index["stats"]["residual_rmse_max_rad"] = float(np.max(rmses))

    with open(root / "index.json", "w") as f:
        json.dump(index, f, indent=2)


def test_synthetic_fixture_matches_index_schema(tmp_path: Path) -> None:
    _write_synthetic_track_c_dataset(tmp_path)

    index_path = tmp_path / "index.json"
    assert index_path.exists()
    with open(index_path) as f:
        index = json.load(f)

    assert set(index["splits"]) == {"train", "val", "test"}
    assert len(index["splits"]["train"]) == 2
    assert len(index["splits"]["val"]) == 1
    assert len(index["splits"]["test"]) == 0
    assert index["stats"]["total_frames"] == 3

    for split_paths in index["splits"].values():
        for rel in split_paths:
            assert (tmp_path / rel).exists()


def test_synthetic_frame_npz_has_expected_keys_and_dtypes(tmp_path: Path) -> None:
    shape = (96, 128)
    _write_synthetic_track_c_dataset(tmp_path, shape=shape)

    with open(tmp_path / "index.json") as f:
        index = json.load(f)

    rel = index["splits"]["train"][0]
    npz = np.load(tmp_path / rel)

    assert set(npz.files) >= {
        "phi_classical", "phi_target", "residual", "z_m", "session",
        "filename",
    }
    assert npz["phi_classical"].shape == shape
    assert npz["phi_target"].shape == shape
    assert npz["residual"].shape == shape
    assert npz["phi_classical"].dtype == np.float32
    assert npz["phi_target"].dtype == np.float32
    assert npz["residual"].dtype == np.float32
    assert np.isfinite(npz["phi_classical"]).all()
    assert np.isfinite(npz["phi_target"]).all()
    assert np.isfinite(npz["residual"]).all()

    # residual = phi_classical - phi_target (post piston-align), per
    # build_track_c_dataset.py's _save_frame() contract.
    np.testing.assert_allclose(
        npz["residual"], npz["phi_classical"] - npz["phi_target"],
        atol=1e-5,
    )

    assert str(npz["session"]) == "s1"
    assert str(npz["filename"]) == "DHM_0001.npz"
    assert np.asarray(npz["z_m"]).dtype == np.float32


def test_synthetic_fixture_round_trips_through_dataset_config_shape(
    tmp_path: Path,
) -> None:
    """Sanity check the shape contract ``TrackCDataset.__getitem__`` relies
    on (crop_size <= frame shape) without importing torch: the loader pads
    with edge-replication when the frame is smaller than crop_size, so a
    fixture at least as large as the default crop keeps that path
    unexercised and predictable for the torch-gated test."""
    shape = (256, 256)
    _write_synthetic_track_c_dataset(tmp_path, shape=shape)

    with open(tmp_path / "index.json") as f:
        index = json.load(f)
    rel = index["splits"]["train"][0]
    npz = np.load(tmp_path / rel)
    h, w = npz["phi_classical"].shape
    crop_size = 128
    assert h >= crop_size and w >= crop_size
