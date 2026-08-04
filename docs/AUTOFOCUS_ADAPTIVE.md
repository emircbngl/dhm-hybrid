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
