"""Build a clean synthetic reference per session via temporal median.

The user has live moving bacteria in the sample, but the "reference"
holograms saved in each session are themselves contaminated — they
contain bacteria too. Reconstructing through `sample / reference`
therefore propagates bacterial structure from the reference into the
"clean" output, polluting the GT we use for Track C training.

Solution: synthesize a clean reference per session by taking the
**temporal median** of every hologram in the session. Per pixel, the
median across N frames cancels random transient signal (moving
bacteria) and preserves stable structure (illumination beam,
fixed-pattern, sensor response, dust). This is a standard live-imaging
trick.

Caveats:

* If a bacterium sits still on the slide for the whole session it will
  survive median (median > 50% of pixels see it). Sparse but-stationary
  contamination is not removed.
* If the session has very few frames (<5) median is unreliable. We
  refuse to synthesize for sessions below ``--min-frames`` (default 4).
* The median is taken on the **raw** intensity image (uint16-ish), not
  on the demodulated complex field. Median commutes with linear
  operations approximately well enough for our purpose; rebuilding the
  field per frame is what gives us a per-frame reconstruction.

Output: ``<DATA_ROOT>/_synthetic_refs/<session>.png`` plus
``<DATA_ROOT>/_synthetic_refs/manifest.json`` mapping session → path.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from core.ingestion import load_any  # noqa: E402
from benchmark_reffree import DATA_ROOT, build_manifest  # noqa: E402

OUT_ROOT = DATA_ROOT / "_synthetic_refs"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
_LOG = logging.getLogger("synthetic_refs")


def synthesize_session_ref(session_files: list[Path],
                           *, min_frames: int = 4) -> "np.ndarray | None":
    if len(session_files) < min_frames:
        _LOG.warning("only %d frames (<%d) — skipping median synth",
                     len(session_files), min_frames)
        return None
    stack = []
    for p in session_files:
        arr = np.asarray(load_any(p).array, dtype=np.float32)
        stack.append(arr)
    stack = np.stack(stack, axis=0)
    _LOG.info("  stack shape %s, value range [%.1f, %.1f]",
              stack.shape, stack.min(), stack.max())
    return np.median(stack, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out-dir", default=str(OUT_ROOT))
    parser.add_argument("--min-frames", type=int, default=4)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    plans = build_manifest()
    manifest: dict = {"sessions": {}}

    for plan in plans:
        # Use the union of plan.files (sample frames) and plan.ref_path
        # — they all came from the same physical session.
        all_files = list(plan.files)
        if plan.ref_path is not None and plan.ref_path not in all_files:
            all_files.append(plan.ref_path)
        if not all_files:
            continue
        _LOG.info("[%s] %d frames", plan.label, len(all_files))
        median_img = synthesize_session_ref(all_files,
                                            min_frames=args.min_frames)
        if median_img is None:
            manifest["sessions"][plan.label] = None
            continue
        out_path = out_root / f"{plan.label}_median_ref.png"
        # Save as uint16 PNG to preserve dynamic range better than uint8.
        # The downstream loader accepts both.
        max_v = float(median_img.max())
        if max_v <= 0:
            _LOG.warning("[%s] median has max=0 — skipping save", plan.label)
            manifest["sessions"][plan.label] = None
            continue
        scaled = np.clip(median_img / max_v * 65535.0, 0, 65535
                         ).astype(np.uint16)
        Image.fromarray(scaled, mode="I;16").save(out_path)
        manifest["sessions"][plan.label] = {
            "path": str(out_path.relative_to(out_root.parent)),
            "n_frames": len(all_files),
            "shape": list(median_img.shape),
            "max_value_uint16_normalized_from": max_v,
        }
        _LOG.info("[%s] saved %s", plan.label, out_path.name)

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    _LOG.info("DONE — manifest: %s", out_root / "manifest.json")


if __name__ == "__main__":
    main()
