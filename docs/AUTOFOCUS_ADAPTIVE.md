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

On that held-out synthetic split, at 0.5 mm tolerance:

| threshold | coverage | conditional hit | overall hit |
|---:|---:|---:|---:|
| 0.70 | 41.6% | 98.1% (51/52) | 40.8% |
| **0.75** | **39.2%** | **100% (49/49)** | **39.2%** |
| 0.80 | 34.4% | 100% (43/43) | 34.4% |
| 0.90 | 24.0% | 100% | 24.0% |

**0.75 is the shipped default** because it dominates 0.80 outright — same
conditional accuracy, 4.8 more points of coverage. With n=49 the "100%"
carries a Wilson 95% interval of [92.7%, 100%]; treat it as an operating
point, not a proven accuracy.

These figures demonstrate abstention; they do not establish real-world
accuracy because both splits come from the same simulator family. The
calibration is named synthetic-v1 in code and must be replaced by a
separately versioned, stage-labelled real-data calibration before scientific
deployment.

### What the gate costs

Measured on the same 125 scenes, the calibrated gate is a trade, not a net
gain. It buys precision by discarding scenes:

| method | coverage | conditional hit | overall hit | overall F1 | planes/scene |
|---|---:|---:|---:|---:|---:|
| `object_local` (ungated) | 89.6% | 57.1% | **51.2%** | **36.5%** | 7.83 |
| `object_local_3d` | 64.8% | 40.7% | 26.4% | 16.9% | 1.50 |
| `object_local_hybrid` | 41.6% | 98.1% | 40.8% | 31.4% | 0.61 |

Report overall hit and F1 alongside the conditional numbers. Quoting 98%
conditional hit without the 41.6% coverage overstates the method.

## Fixture characterization — read this before tuning search

`scripts/characterize_focus_fixture.py` measures what the synthetic fixture
itself can resolve, so search scores can be read against a real ceiling rather
than an assumed 0.5 mm tolerance. Raw rows:
`tasks/focus_fixture_characterization.json`; pins:
`tests/test_focus_fixture_characterization.py`.

Full grid, 2026-08-17 — 27 single-sphere scenes (radius 4/10/18 µm, noise
0/0.01/0.06, z 3/12/29 mm) × 5 metrics, 256×256, 401 steps over −5…45 mm:

| metric | global \|err\| | global hit | masked \|err\| | masked hit | masked FWHM |
|---|---:|---:|---:|---:|---:|
| laplacian_variance | 12.13 mm | 11% | 0.875 mm | 22% | 0.375 mm |
| normalized_variance | 13.50 mm | 0% | 0.500 mm | 37% | 3.250 mm |
| phase_variance | 13.50 mm | 0% | 0.500 mm | 37% | 3.250 mm |
| spectral_energy | 26.13 mm | 0% | 0.625 mm | 33% | 3.125 mm |
| tenengrad | 28.38 mm | 7% | 0.625 mm | 37% | 2.125 mm |

Two facts follow, and they reorder the work:

1. **On this fixture the whole-frame focus landscape does not peak at the
   object.** Over the real search bounds every metric's global extremum
   collapses toward z≈0, the camera plane — median error 12–28 mm, 0–11% hit.
   The optimizer is not choosing a wrong extremum; it is finding the correct
   extremum of a landscape whose maximum is in the wrong place. This is what
   the range-edge rejection in `gradient_descent_search` is patching around.
   **Scope correction (2026-08-17):** this is substantially an artefact of the
   ASM fixture, not a general property of off-axis DHM — see the independent
   generator below, where the same measurement gives 1.25–4.6 mm.
2. **The oracle mask is the ceiling, and it is low.** Handed the true object
   footprint, the same metrics reach 0.5–0.875 mm median error but only
   22–37% hit at 0.5 mm. `object_local_hybrid`'s 40.8% overall hit is already
   at that ceiling. No detector, calibration or optimizer improvement moves a
   benchmark past its oracle.

The derived tolerance is **1.625 mm** (half the measured axial response,
floored by the median masked error) against the 0.5 mm currently scored — the
benchmark is roughly 3× tighter than the fixture supports. Loosening it is
not the fix, though: recomputed on the held-out split, moving 0.5 → 1.625 mm
lifts `object_local` from 51.2% to 60.0% but leaves `object_local_hybrid`
unchanged at 40.8%. The hybrid's failures are gross, not marginal, so the
remaining gap is upstream — in the fixture and the metric, not in the
tolerance.

## The independent generator — inverse crime removed

`tests/fixtures/independent_hologram.py` is a second forward model that shares
no propagation code with the reconstruction. Contract pins:
`tests/test_independent_hologram.py`, including an AST check that the module
imports nothing from `core` and uses no FFT — a property no output number would
reveal.

* **Direct Rayleigh-Sommerfeld summation, no FFT.** Every object-plane sample
  is a Huygens source and the field at each sensor pixel is the explicit sum.
  The reconstruction multiplies an angular spectrum by a transfer function;
  there is no shared operator to cancel.
* **Object-plane pitch 1.0 µm against a 2.5 µm sensor pitch**, so grid-aligned
  artefacts cannot round-trip.
* **A detector chain the ASM fixture lacks**: Poisson shot noise (the dominant
  real term — the ASM fixture models only Gaussian read noise), sensor MTF,
  Gaussian illumination falloff, curved reference, low-order phase error,
  optional speckle, full-well saturation, quantisation.

Matched comparison — the same 9 (radius, z) scenes, oracle-masked focus curve,
256×256, 401 steps, ASM fixture at noise=0 versus the independent generator
carrying its full detector chain:

| metric | ASM masked \|err\| / hit | independent masked \|err\| / hit | ASM global \|err\| | independent global \|err\| |
|---|---:|---:|---:|---:|
| laplacian_variance | 0.625 mm / 33% | 0.375 mm / 56% | 3.25 mm | 1.25 mm |
| tenengrad | 0.375 mm / 56% | 0.375 mm / 67% | 11.38 mm | 3.50 mm |
| phase_variance | 0.250 mm / 56% | 0.375 mm / 56% | 11.88 mm | 4.63 mm |
| normalized_variance | 0.250 mm / 56% | 0.375 mm / 56% | 11.88 mm | 4.63 mm |
| spectral_energy | 0.500 mm / 44% | 0.375 mm / 67% | 12.00 mm | 4.63 mm |

Two things follow, and one of them corrects the section above.

**The oracle ceiling survives.** Masked median error is ~0.4 mm and hit@0.5 mm
is 56–67% on the independent generator, against 0.25–0.625 mm and 33–56% on the
ASM fixture — the same order, slightly better, despite the independent scenes
carrying shot noise, MTF, aberration and saturation that the ASM scenes do not.
So the ceiling is a real property of this measurement problem, not a symptom of
ASM inverting ASM. `object_local_hybrid` sitting at 40.8% overall hit remains
near its ceiling, and that conclusion now rests on two independent forward
models rather than one.

**The z≈0 collapse was largely the fixture.** Global error drops from 11–12 mm
to 3.5–4.6 mm (and 3.25 → 1.25 mm for laplacian) once the shared FFT
propagator is gone. The plausible mechanism is wrap-around: the ASM fixture
propagates 12–43 mm across a 640 µm periodic FFT window, so the object's field
re-enters the frame instead of leaving it, and the resulting high-frequency
structure is strongest near z≈0. Direct summation has no periodic window. This
is a hypothesis consistent with the measurement, not a verified derivation.
Whole-frame autofocus is still unreliable on the independent generator —
multi-millimetre error against a ~0.4 mm oracle — so object-local masking is
still the right architecture; it is just not rescuing a landscape that peaks at
the camera.

Remaining shared approximation, stated plainly: both models use a thin
projected phase screen (no refraction inside the sphere, no multiple
scattering, scalar fields). The independent generator is a second, independent
*approximation* whose errors are uncorrelated with the reconstruction's — not a
metrology reference. Real stage-labelled holograms remain the only way to make
an accuracy claim.
