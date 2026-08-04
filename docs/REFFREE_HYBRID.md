# Reference-Free DHM Reconstruction (Track C — Hybrid CNN)

End-to-end pipeline that removes the need for a reference hologram at
inference time. A small residual U-Net learns the reproducible
illumination + sensor + carrier-residual aberration the off-axis
demodulation pipeline can't fully cancel, and subtracts it from the
classical polynomial-fit reffree output.

## Why hybrid?

We evaluated three approaches before settling on this one:

| Approach | Median RMSE on test | Verdict |
|---|---|---|
| **Track A** — pure classical (Zernike/poly fit) | ~3.9 rad | 26× target — insufficient |
| **Track B** — pure end-to-end DL | n/a | 63 frames is far too few; needs 5,000+ |
| **Track C** — classical + small CNN residual | see report | The chosen path |

The diff between Track A output and the (synthetic-ref) ground truth
showed **structured stripe patterns** (illumination beam profile +
carrier residual + sensor fixed-pattern), not random speckle. Structured
residuals are a perfect match for a small CNN trained on tens of
examples — it doesn't need to learn physics, it just learns the
session-stable aberration.

See `tasks/track_b_pure_dl_notes.md` for the conditions under which
Track B (pure DL) becomes viable.

## Pipeline overview

### Training time
1. Build a **synthetic clean reference** per session by taking the
   temporal median of every hologram in the session. Moving bacteria
   average out; stable illumination + sensor patterns survive. Output:
   `_synthetic_refs/<session>_median_ref.png`.
2. For every non-outlier frame in the GT manifest, generate a
   training pair:
    * `phi_classical` — polynomial-fit reffree output at the GT z
    * `phi_target` — synthetic-ref-divided reconstruction at same z
    * `residual = phi_classical - phi_target` — what the CNN learns to
      predict
3. Train UNetLite (3.3 M params) with L1 + TV loss, AdamW, cosine LR.

### Runtime
1. Demodulate the off-axis hologram (Fourier sideband, classical).
2. Autofocus (or use a fixed z if known).
3. Propagate at z + unwrap + polynomial bg-fit order 5 → `phi_classical`.
4. Tile `phi_classical` into 768×768 patches with 64-pixel overlap.
5. Run UNetLite on each tile, blend with cosine window → predicted residual.
6. Output `phi_clean = phi_classical - predicted_residual`.

**No reference hologram is read at runtime.**

## Quick start

```bash
# One-command end-to-end (synth refs → dataset → train → eval → demo).
bash scripts/run_track_c_pipeline.sh

# Or step-by-step:
python3 scripts/build_synthetic_refs.py
python3 scripts/build_track_c_dataset.py
python3 scripts/train_track_c.py --epochs 80
python3 scripts/eval_track_c.py --model models/track_c/v0.1/model.pt --split test
python3 scripts/eval_track_c.py --model models/track_c/v0.1/model.pt --split val

# Live use: reconstruct any single hologram with no reference.
python3 scripts/reffree_reconstruct.py \
    --model models/track_c/v0.1/model.pt \
    --hologram /path/to/DHM_xxx.png \
    --out-dir /path/to/output/
```

## File layout

| Path | Purpose |
|---|---|
| `scripts/build_synthetic_refs.py` | Temporal-median ref per session |
| `scripts/build_track_c_dataset.py` | Generate training `.npz` pairs |
| `scripts/train_track_c.py` | Train UNetLite |
| `scripts/eval_track_c.py` | Quantitative + visual evaluation |
| `scripts/reffree_reconstruct.py` | Live single-hologram CLI |
| `scripts/run_track_c_pipeline.sh` | End-to-end orchestrator |
| `src/recon_dl/unet_lite.py` | Model definition |
| `src/recon_dl/dataset.py` | PyTorch Dataset wrapper |
| `src/recon_dl/losses.py` | L1 + TV loss |
| `src/recon_dl/inference.py` | Production reconstructor (tiled, blended) |
| `models/track_c/v0.1/` | Trained checkpoint |
| `_synthetic_refs/` | Generated clean references (under DATA_ROOT) |
| `_track_c_dataset/` | Training `.npz` pairs (under DATA_ROOT) |
| `_track_c_eval/` | Evaluation reports (under DATA_ROOT) |

## Performance targets and current numbers

| Metric | Target | Track A baseline | Track C (this) |
|---|---|---|---|
| Median test RMSE (rad) | ≤ 0.50 | ~2.2 | see `_track_c_eval/v0.1/test/summary.json` |
| Test p95 abs err (rad) | ≤ 1.50 | ~4.4 | see `_track_c_eval/v0.1/test/summary.json` |
| Inference latency (1024², MPS) | < 100 ms | n/a | see `meta.json` from any reffree_reconstruct run |
| LOSO generalization gap | < 2× | n/a | future work |

The baseline (~2.2 rad on the test centre crop) is lower than the full-frame
3.9 rad we saw in the original benchmark because the centre region is
typically cleaner than the edges — most of the structured stripe content
sits in the periphery, where the carrier-frequency leakage is strongest.

## When to retrain

* New optical setup (lens swap, laser change, different camera) — the
  aberration pattern shifts. Retrain.
* New session that systematically fails (visual outlier in
  `eval_track_c.py` reports). Add it to the dataset and retrain.
* Sample type drastically changes (e.g., non-bacterial samples). The
  CNN was trained on backgrounds where bacteria are sparse; if bacteria
  density goes way up, the temporal median may need more frames.

## Pitfalls and notes (lessons learned)

1. **Reference frames in the dataset are themselves contaminated** — they
   contain bacteria. Using them as GT pollutes training. The synthetic
   median ref avoids this by averaging over many frames where bacteria
   move randomly.
2. **Autofocus disagreement inflates RMSE** in benchmarks if both
   pipelines pick their own z. Lock z (via the GT manifest's saved
   value) when comparing reference-free vs reference-based.
3. **GT manifest itself has 10/63 outlier frames** where the saved z is
   in the wrong half-space. The dataset builder filters them out using
   the session median.
4. **Track C is only viable because the residual is structured** — if
   future data shows random/speckle-dominated residuals, switching
   approach (more aug, more capacity, or pure DL with a real data
   campaign) is the right move. See `tasks/track_b_pure_dl_notes.md`.
