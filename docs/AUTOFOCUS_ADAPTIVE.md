# Adaptive Autofocus — Settled (2026-07-06)

Closes the long-open backlog item *"adaptive konseptli algoritmalarımız
vardı, onları oturtamadık tam — bir ara oturtalım"* (tasks/todo.md,
2026-07-05). The six autofocus search algorithms had only ever been
benchmarked on a synthetic single-sphere scene; this settlement measured
them on **real lab data** and pinned defaults from the evidence.

## Method

* **Data:** 9 scenes — the middle frame of each 2026-04-29/30 lab session
  under `~/Desktop/rapor data/session_*` (1600×1200, 16-bit).
* **Optics:** lab-confirmed (scripts/run_rapor_data_batch.py, 2026-05-04):
  632.8 nm, 4.4 µm / 50× = 0.088 µm effective pixel, n = 1.337, off-axis
  mask radius 80.
* **Ground truth per (scene, metric):** dense 201-step z-scan over ±20 mm.
* **Budget:** every algorithm ran with the app default `af_n_steps = 40`.
* **Metrics:** `LAPLACIAN_VARIANCE` (app default) and `ENTROPY`.
* Harness: `scripts/benchmark_af_real.py`; raw rows:
  `tasks/af_real_benchmark.json` (108 rows).

## Results (median over 9 scenes; hit = |z error| ≤ 0.5 mm)

| algorithm | LAPLACIAN err | LAPLACIAN hit | ENTROPY err | ENTROPY hit | med evals | med ms |
|---|---|---|---|---|---|---|
| zscan *(old default)* | 0.65 mm | 33% | 0.71 mm | 44% | 40 | 1.3–2.2 s |
| coarse_to_fine | 0.32 mm | 56% | 0.88 mm | 44% | 57 | 1.9–3.1 s |
| **robust** | **0.07 mm** | **78%** | **0.15 mm** | **67%** | 40 | 2.4–3.9 s |
| adaptive_gradient | 0.11 mm | 67% | 7.24 mm | 22% | 40 | 1.3–2.1 s |
| adaptive_bracketing | 0.03 mm | 78% | 0.10 mm | 56% | 53 | 1.8–2.8 s |
| adaptive_distance | 12.9 mm | 22% | 11.5 mm | 0% | 15–40 | 0.8–1.3 s |

## Decisions (pinned in code)

1. **Default `af_algorithm`: `zscan` → `robust`** (`ReconParams`,
   `core.settings_schema.Ui2State`). At the same 40-eval budget the old
   default was the *least* accurate option (its grid pitch — 1 mm over
   ±20 mm — IS its accuracy floor); `robust` is the only top performer
   on both metrics. The v9 state migration stays frozen at `zscan` —
   existing lab state files keep behaving exactly as before unless the
   operator changes the combo.
2. **Headless/MCP autofocus: `adaptive_gradient` → `robust`**, and its
   default metric aligned to the GUI (`LAPLACIAN_VARIANCE`, was
   `PHASE_VARIANCE`).
3. **Per-algorithm guidance stamped into the UI** — the sidebar/panel
   tips (`ui2.workers.af_algorithm_input_profile`) now carry the
   benchmark verdicts, so the combo tells the operator what the data
   said:
   * `adaptive_bracketing` — accuracy pick on laplacian (0.03 mm/78%);
     budget overrun ~30% (53 evals for a 40 cap).
   * `adaptive_gradient` — fastest accurate option on gradient-friendly
     metrics; **do not pair with ENTROPY** (7.2 mm median error — the
     step-walker stalls on entropy's flat shoulders; same failure mode
     as the 2026-05 lesson).
   * `adaptive_distance` — unreliable on this rig's ±11–18 mm scenes
     (0–22% hit; range expansion stops early at the default
     signal_threshold). Reserve for truly-unknown range, verify result.
4. **No new ReconParams knobs for adaptive internals.** The benchmark
   shows algorithm choice + metric pairing dominate; the auto-derived
   step/range internals were not the bottleneck. The ui3 FocusPanel's
   read-only "Adaptive" info section stands.
5. **`adaptive_steps/` staging folder removed** — the March 2026
   prototype superseded by `core/autofocus/search_adaptive.py`.

## Caveats

* ENTROPY landscapes on these scenes are shallow (dense-curve prominence
  ~0.5 vs ~1.0+ for laplacian); several scenes have entropy optima far
  from the laplacian optimum. The per-metric truth is used for scoring,
  so this is a property of the metric, not the search.
* labtest/ holograms were excluded (optics unrecorded); rerun
  `benchmark_af_real.py` after logging their parameters if needed.

Regression pins: `tests/test_af_settlement.py` (B-095).

## Numerical gradient descent (added after settlement)

`gradient_descent` is a separate, experimental autofocus option; it is not the
older `adaptive_gradient` step walker. It first spends part of its budget on a
global coarse Laplacian-variance landscape, smooths that curve, rejects
range-edge extrema, and brackets the most prominent interior basin. The
caller's selected metric is computed from the same propagated field without
repeating FFT work. Inside that basin it
minimizes a loss derived from the selected focus metric, estimates
`d(loss)/dz` with central finite differences (bounded secants at the basin
limits), and performs projected gradient-descent updates with backtracking.
All coarse, derivative, and line-search propagations count toward
`af_n_steps`.

This method was not part of the 2026-07-06 nine-scene benchmark above, so
`robust` remains the evidence-backed default until the same harness is rerun.

For repeatable synthetic stress testing, run
`python scripts/benchmark_af_synthetic_stress.py --quick`. The factorial mode
uses five independent axes with five levels each: peak-relative Gaussian noise,
absolute distance, object count, object/objectlet radius scale, and axial plane
separation. `--factorial` materialises all 5^5 = 3125 combinations;
`--shard-count N --shard-index I` partitions the same deterministically seeded
matrix without changing a scene's camera noise.

## Object-local multi-focus

`core.depth_map.find_object_focus_planes` is the spatial counterpart to the
whole-frame focus landscape. It reconstructs a z stack once, computes a local
focus volume, detects connected masks from the z-invariant maximum response,
and reduces a separate focus curve inside each mask. `PHASE_VARIANCE` is the
wrap-safe phase/complex-field mode; `LAPLACIAN_VARIANCE` and `TENENGRAD` are
amplitude/visual alternatives. Frame-edge components are rejected as FFT
artefacts, and a three-point parabola refines each interior peak below the scan
step. The benchmark's `object_local` algorithm reports every detected object z,
per-ground-truth-plane error, and plane recall.

The corrected stress generator uses non-overlapping carrier/DC geometry,
explicit +1-order extraction, 12-bit quantisation, and multiplies same-plane
phase screens instead of summing a new incident background per object. It is
still a projected scalar phase-screen fixture—not a volumetric scattering or
physical-metrology reference.

### 3-D persistence successor

`core.depth_map.find_object_focus_planes_3d` avoids the 2-D `max_z` projection
that turned one extreme noise response into a permanent object mask. It builds
phase/complex-variance, amplitude-Laplacian and Tenengrad volumes from each
reconstruction, robustly normalises every channel/plane against its background,
and labels connected components in `(z, y, x)`. Candidates must persist across
neighbouring z planes, stay away from the FFT frame border, remain within a
plausible area fraction, and pass four independent evidence gates: peak
prominence, robust component strength, ROI-to-ring complex contrast SNR, and
complex coherence. Duplicate components are merged in lateral/depth space.

An empty result is intentional abstention (`cannot measure`), not an optimizer
failure. The synthetic benchmark reports both coverage and conditional
accuracy; raising confidence gates must not be described as accuracy progress
without reporting the lost coverage. `object_local_3d` selects this method in
the stress harness. Its thresholds are an algorithmic hypothesis calibrated on
synthetic data, not a verified optical law; real stage-labelled calibration is
required before enabling it as a GUI default.

### Calibrated hybrid operating point

`core.depth_map.find_object_focus_planes_hybrid` keeps the higher-recall 2-D
candidate generator and applies a fixed five-feature logistic objectness model:
focus prominence, log ROI/ring complex contrast, complex coherence, and the two
prominence interactions. The coefficients and normalisation constants are
embedded next to the scorer, so runtime does not depend on scikit-learn. They
were fitted on 125 deterministic 256×256/60-plane factorial scenes and checked
on a disjoint 125-scene split.

On that held-out synthetic split, probability threshold 0.70 gave 41.6%
coverage, 98.1% conditional hit rate and 95.8% conditional plane precision at
0.5 mm tolerance. Threshold 0.80 reached 100% conditional hit but only 34.4%
coverage. These figures demonstrate abstention; they do not establish
real-world accuracy because both splits come from the same simulator family.
The calibration is named synthetic-v1 in code and must be replaced by a
separately versioned, stage-labelled real-data calibration before scientific
deployment.
