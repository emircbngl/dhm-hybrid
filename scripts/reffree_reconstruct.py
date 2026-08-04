"""Reference-free DHM reconstruction CLI.

The production-facing command. Given a hologram (PNG/TIFF) and a
checkpoint, produces a clean phase image — no reference hologram
needed at runtime.

Usage::

    python3 scripts/reffree_reconstruct.py \\
        --model models/track_c/v0.1/model.pt \\
        --hologram /path/to/DHM_xxx.png \\
        --out-dir /path/to/output/

Outputs in ``out-dir``:

* ``phase_clean.tif`` — final unwrapped phase (float32, radians)
* ``phase_classical.tif`` — pre-CNN baseline (for comparison/audit)
* ``residual.tif`` — what the CNN subtracted (the structured aberration)
* ``visualization.png`` — 3-panel sanity-check plot
* ``meta.json`` — z, inference time, model path
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_dl.inference import ReffreeReconstructor  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
_LOG = logging.getLogger("reffree_reconstruct")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True,
                        help="Path to trained UNetLite checkpoint (.pt)")
    parser.add_argument("--hologram", required=True,
                        help="Path to a single hologram image (PNG/TIFF)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fixed-z-mm", type=float, default=None,
                        help="Skip autofocus and reconstruct at this z (mm)")
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--tile-overlap", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recon = ReffreeReconstructor.from_checkpoint(
        args.model,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    fixed_z = (args.fixed_z_mm * 1e-3) if args.fixed_z_mm is not None else None
    result = recon.reconstruct(args.hologram, fixed_z_m=fixed_z)

    tifffile.imwrite(out_dir / "phase_clean.tif",
                     result.phase_clean.astype(np.float32))
    tifffile.imwrite(out_dir / "phase_classical.tif",
                     result.phase_classical.astype(np.float32))
    tifffile.imwrite(out_dir / "residual.tif",
                     result.predicted_residual.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=110)
    arrs = [result.phase_classical,
            result.phase_clean,
            result.predicted_residual]
    titles = [f"classical poly5\nstd={float(arrs[0].std()):.2f} rad",
              f"reffree CNN-clean\nstd={float(arrs[1].std()):.2f} rad",
              f"residual subtracted\nstd={float(arrs[2].std()):.2f} rad"]
    vmax = float(np.percentile(np.abs(result.phase_classical), 98))
    for ax, arr, title in zip(axes, arrs, titles):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "visualization.png", bbox_inches="tight")
    plt.close(fig)

    meta = {
        "model": str(args.model),
        "hologram": str(args.hologram),
        "z_mm": result.z_m * 1e3,
        "inference_ms": result.inference_ms,
        "shape": list(result.phase_clean.shape),
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "phase_clean_stats": {
            "min": float(result.phase_clean.min()),
            "max": float(result.phase_clean.max()),
            "mean": float(result.phase_clean.mean()),
            "std": float(result.phase_clean.std()),
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _LOG.info("done — output: %s", out_dir)
    _LOG.info("z=%+.3f mm  inference=%.1f ms  output std=%.2f rad",
              meta["z_mm"], meta["inference_ms"],
              meta["phase_clean_stats"]["std"])


if __name__ == "__main__":
    main()
