"""Smoke tests for the Track C reference-free pipeline.

These tests don't require a trained model — they verify the data,
model, and inference glue is wired correctly. The full end-to-end
run is exercised by ``scripts/run_track_c_pipeline.sh``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# importorskip (not a bare import) so environments without torch skip this
# module cleanly instead of failing collection of the whole test run
# (2026-07-05 review: local `pytest tests/` broke on this line).
torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MODEL_PATH = REPO_ROOT / "models" / "track_c" / "v0.1" / "model.pt"


def _write_synthetic_track_c_dataset(root: Path, *,
                                     shape: tuple[int, int] = (192, 256),
                                     seed: int = 0) -> None:
    """Build a tiny fixture matching build_track_c_dataset.py's on-disk
    schema (index.json + per-frame .npz — see that script's ``_save_frame``
    and ``build_dataset`` for the authoritative contract). Kept in sync
    with the identical helper in test_track_c_dataset_schema.py, which
    exercises the same schema without a torch dependency.
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


def test_unet_lite_forward_shape_and_residual_init() -> None:
    from recon_dl.unet_lite import UNetLite, count_parameters

    net = UNetLite(base_c=16)
    n = count_parameters(net)
    assert 100_000 < n < 5_000_000, f"unexpected param count: {n}"
    x = torch.randn(2, 1, 256, 256)
    corrected, residual = net(x)
    assert corrected.shape == x.shape
    assert residual.shape == x.shape
    # Untrained net's final conv is initialized near zero — residual ≈ 0,
    # corrected ≈ input. Tolerate small noise.
    assert residual.abs().max() < 0.5, \
        f"residual init too large: {float(residual.abs().max()):.4f}"


def test_combined_loss_components() -> None:
    from recon_dl.losses import CombinedLoss

    pred = torch.zeros(1, 1, 32, 32)
    target = torch.ones(1, 1, 32, 32) * 2.0
    loss_fn = CombinedLoss(l1_weight=1.0, tv_weight=0.05)
    total, parts = loss_fn(pred, target)
    assert pytest.approx(parts["l1"], abs=1e-5) == 2.0
    assert parts["tv"] == 0.0  # zero pred has no TV
    assert pytest.approx(parts["loss"], abs=1e-5) == 2.0


def test_combined_loss_disabled_terms_follow_pred_device() -> None:
    """Disabled loss terms must be zeros on *pred's* device.

    Regression for the 2026-07-05 review finding: a bare
    ``torch.tensor(0.0)`` lives on CPU, and adding it to an MPS/CUDA
    resident l1 raised "Expected all tensors to be on the same device"
    on every forward — training crashed out of the box on Apple GPU
    (charbonnier_weight defaults to 0.0).
    """
    from recon_dl.losses import CombinedLoss

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    loss_fn = CombinedLoss(l1_weight=1.0, charbonnier_weight=0.0,
                           tv_weight=0.0)  # both extra terms disabled
    for dev in devices:
        pred = torch.zeros(1, 1, 16, 16, device=dev)
        target = torch.ones(1, 1, 16, 16, device=dev)
        total, parts = loss_fn(pred, target)   # pre-fix: RuntimeError on mps/cuda
        assert total.device.type == dev
        assert parts["l1"] == pytest.approx(1.0, abs=1e-5)
        assert parts["charbonnier"] == 0.0
        assert parts["tv"] == 0.0


def test_cosine_window_smooth_corners() -> None:
    from recon_dl.inference import _cosine_window
    win = _cosine_window(64, 64, edge=8)
    assert win.shape == (64, 64)
    assert win[32, 32] == pytest.approx(1.0)
    # Corners should taper toward 0.
    assert win[0, 0] < 0.05
    assert win[-1, -1] < 0.05


def test_dataset_loads_train_split(tmp_path: Path) -> None:
    """Default: exercise the real loader against a small synthetic fixture
    generated on the fly (see ``_write_synthetic_track_c_dataset`` above),
    so this runs on every machine without needing the real Desktop
    dataset. Set DHM_TRACK_C_DATA to point at a real
    ``_track_c_dataset`` root to test against real data instead.
    """
    from recon_dl.dataset import DatasetConfig, TrackCDataset

    env_root = os.environ.get("DHM_TRACK_C_DATA")
    if env_root:
        root = Path(env_root)
        if not (root / "index.json").exists():
            pytest.skip(f"DHM_TRACK_C_DATA set but no index.json under {root}")
    else:
        root = tmp_path / "_track_c_dataset"
        _write_synthetic_track_c_dataset(root, shape=(192, 256))

    cfg = DatasetConfig(root=root, split="train", crop_size=128, augment=False)
    ds = TrackCDataset(cfg)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (1, 128, 128)
    assert y.shape == (1, 128, 128)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()


_DATA_ROOT = Path(
    os.environ.get("DHM_DATA_ROOT", REPO_ROOT / "data" / "rapor")
)
_SYNTH_REF_MANIFEST = _DATA_ROOT / "_synthetic_refs" / "manifest.json"


@pytest.mark.skipif(
    not _SYNTH_REF_MANIFEST.exists(),
    reason="synthetic refs not built yet",
)
def test_synthetic_ref_manifest() -> None:
    import json
    with open(_SYNTH_REF_MANIFEST) as f:
        manifest = json.load(f)
    sessions = manifest.get("sessions", {})
    # We expect at least 5 sessions to have synth refs (s2 may fall back
    # but the rest should have real medians).
    populated = [s for s, e in sessions.items() if e is not None]
    assert len(populated) >= 5, f"only {len(populated)} populated sessions"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="model not trained yet",
)
def test_inference_module_loads_and_runs_one_tile() -> None:
    from recon_dl.inference import ReffreeReconstructor

    recon = ReffreeReconstructor.from_checkpoint(
        MODEL_PATH, tile_size=256, tile_overlap=32,
    )
    # Synthetic input — random noise, not a real hologram. Just verify
    # the tiling + blending math doesn't crash.
    fake_phi = np.random.randn(384, 512).astype(np.float32) * 2.0
    out = recon._cnn_correct(fake_phi)
    assert out.shape == fake_phi.shape
    assert np.isfinite(out).all()
