# Reconstruction Accuracy Validation

This note exists because the Lindqvist-lab physicist asked one reasonable
question during the v1.0.1-ux review:

> *"You tested on your own synthetic holograms. How do I know the
> reconstruction is correct?"*

Short answer: we run a triple of regression tests on every commit, and we
have a planned NIH-benchmark validation before v1.1-sci. This document
explains what's checked now, what's not, and where the numbers come
from.

---

## What is checked on every commit

`tests/test_reconstruction_accuracy.py` runs four probes:

### 1. Round-trip phase RMSE — `test_asm_round_trip_phase_rmse`
For a synthetic complex field (Gaussian amplitude × quadratic phase),
propagate forward by `+z` and back by `-z` via the Angular Spectrum
Method. Compare the centre-third of the recovered phase against the
input using the circular distance
`Δφ = angle(e^{i(φ_back − φ_ref)})`.

| `z` (mm) | RMSE tolerance (rad) |
| --- | --- |
| 0.1 | 0.02 |
| 1.0 | 0.02 |
| 10  | 0.15 |
| 44  | 0.15 |

The boundary is loosened past 1 mm because the ASM kernel wraps at the
frame edge — real lab setups crop the outer ring anyway. A regression
doubling the error trips the tolerance.

### 2. Defocus–refocus amplitude — `test_amplitude_survives_defocus_refocus`
At 40 mm working distance (the common DHM regime for the lab's setup),
forward then backward propagation must reproduce the amplitude pattern
with relative L2 error ≤ 10% in the centre crop.

### 3. Pixel-size sweep — `test_round_trip_stable_across_pixel_sizes`
Round-trips at 2.0 µm, 3.45 µm, and 5.5 µm pixel sizes must preserve
total energy within 2%. This guards the `float32` frequency-grid cache
in `CachedReconstructor` against precision regressions.

### 4. Zero-distance identity — `test_asm_zero_z_reproduces_field_exactly`
At `z = 0` the ASM transfer function is identically 1; the output must
equal the input within 1e-4 max absolute error. This catches off-by-one
errors in the FFT shift path.

Run locally: `pytest tests/test_reconstruction_accuracy.py -v`.

---

## Synthetic stress suite

`tests/test_stress_holograms.py` drives the reconstruction / autofocus
pipeline with synthetic sphere holograms whose ground truth is known by
construction. The generator lives in
[tests/fixtures/synthetic_hologram.py](../tests/fixtures/synthetic_hologram.py)
and models each sphere via **volume slicing (split-step)** — the same
scheme the lab's MATLAB reference `sample_vitural_hologram.m` uses. At
the sphere centre the cumulative phase equals the closed-form OPL
`2π · 2r · (n_s − n_m) / λ`; for large spheres (≳ 5 µm radius at
Δn = 0.07) this is materially different from a thin-phase 2D
approximation because multiple 2π wraps stack up.

Coverage:

- **Single-sphere z recovery** — z = 8, 12, 15, 20, 25 mm, ENTROPY
  metric, ±2 mm tolerance.
- **Radius sweep** — r = 10, 20, 30, 45 µm @ z = 10 mm.
- **Lateral offset** — pipeline tolerates off-centre spheres.
- **Multi-sphere scene** — 3 spheres at different z, pipeline returns a
  finite best_z (does not crash).
- **Noise robustness** — 2 %, 5 %, 10 % Gaussian read noise.
- **Volume-slicing / thin-phase consistency** — small sphere case.

### Per-pixel depth map (v1.2-tomo alpha)

`core.depth_map.compute_depth_map()` returns a tomographic depth map:
for every pixel in the frame, the z plane at which local sharpness
(Laplacian variance or Tenengrad over an N×N window) peaks.

Ground-truth validation in `tests/test_depth_map.py`:

| Scene | Sampled pixel | Expected z (mm) | Tolerance |
| --- | --- | --- | --- |
| Single sphere @ z = 12 mm | frame centre | −12 | ±3 mm |
| Two spheres at different z + offsets | each sphere's centre | −true_z | ±4 mm |

Tolerance widens slightly for the multi-sphere case because each
sphere's refocused silhouette carries a lateral diffraction skirt —
the local sharpness peak doesn't always land on the geometric centre.
Runtime on a 256×256 grid with 40 z planes + 5×5 window: ~85 ms on
Apple M1.

Supported metrics (local-form kernels registered): `LAPLACIAN_VARIANCE`
(default — handles phase-edge objects), `TENENGRAD` (amplitude-gradient
fallback). `ENTROPY` is explicitly rejected because the metric has no
meaningful local decomposition — it's a global measure.

Export formats: NPZ (loss-less, all fields) + CSV (flat
``sample_id,y,x,z_mm,confidence`` table for R/MATLAB/pandas). Both
writers live in `core.depth_map` and use the same ``sample_id`` the
toolbar carries for LIMS correlation.

### Per-cluster height segmentation (v1.2-tomo follow-up)

`core.depth_map.segment_depth_clusters()` splits a depth map into
connected high-confidence regions and reports a
confidence-weighted ``(z_mean, z_std)`` for each. Pixels above
``confidence_threshold_frac × conf.max()`` become the foreground;
:func:`scipy.ndimage.label` labels connected components; each
surviving component (≥ ``min_area_px``) becomes one
:class:`ClusterHeight`.

Ground-truth validation in `tests/test_cluster_heights.py` using the
same two-sphere scene as the depth-map tests:

| Check | Condition |
| --- | --- |
| Cluster count | 2 spheres → 2 clusters at threshold 0.4 |
| Centroid location | each cluster's centroid within 20 px of a known sphere |
| z_mean accuracy | each cluster within ±8 mm of a true −z |
| Sort order | largest-area cluster first |
| Empty map | flat confidence → `[]` |
| min_area_px filter | tight value drops small clusters |

Tolerance on ``z_mean`` widens (±8 mm) compared with the overall
depth-map test (±4 mm per pixel) because the confidence-weighted
mean mixes the cluster's core pixels with diffraction skirt
contributions.

CSV columns: ``sample_id``, ``cluster_id``, ``area_px``,
``area_um2`` (optional, when pixel size supplied),
``centroid_y_px``, ``centroid_x_px``, ``z_mean_mm``, ``z_std_mm``,
``mean_confidence`` — stable so the lab's LIMS parser doesn't
rewrite on every release.

### Multi-candidate QPI batch (v1.2-tomo beta)

`core.qpi_batch.run_qpi_for_candidates()` takes a list of
:class:`FocusCandidate`s and runs the standard `compute_qpi` pipeline
once per candidate. Each candidate's ``z_m`` overrides the
reconstruction distance; the same gradient-integration phase unwrap
is used as the app's single-focus QPI so the numbers are directly
comparable.

Ground-truth validation in `tests/test_qpi_batch.py`:

| Scene | Candidates | Check |
| --- | --- | --- |
| Two spheres (r = 20 / 30 µm) at z = 8 / 18 mm | two `FocusCandidate` at the true −z | `opd_range_nm` and/or `total_dry_mass_pg` must **differ** between entries (regression catch) |
| Empty candidate list | — | returns `[]` without raising |
| Row flattening | any result | `qpi_batch_to_rows` attaches `candidate_rank`, `candidate_z_mm`, `candidate_prominence` |
| CSV export | two entries | 2 data rows + header, extra columns present |

CSV schema is the stable `core.qpi_export` column set plus three
trailing batch columns — pandas / R / MATLAB load it without custom
parsing. Runtime: ~10 ms per candidate at 256×256 (reconstruction +
unwrap + compute_qpi).

### Multi-focus candidate discovery (prototype)

`core.autofocus.find_focus_candidates()` extends the single-best-z
autofocus with peak finding on the smoothed metric landscape. Useful
when a scene carries multiple objects at distinct depths.

Characterised behaviour:

| Scene | Best metric | Result |
| --- | --- | --- |
| Single sphere | `ENTROPY` | 1 candidate at true z, ±2 mm |
| Multi-sphere, well-separated z | `LAPLACIAN_VARIANCE` | ≥ 2 candidates, at least 2 within ±3 mm of truth |
| Single sphere + high prominence threshold | any metric | Ghost peaks from diffraction rings are filtered |
| Flat complex field | any metric | Empty list (no raise) |

Physical note: a **global** metric (e.g. ENTROPY over the whole frame)
collapses to a single minimum in multi-sphere scenes because the scene
entropy doesn't drop meaningfully at any individual z — every other
sphere remains defocused. Sharpness metrics like `LAPLACIAN_VARIANCE`
or `TENENGRAD` fire on whichever sphere is momentarily in focus and
therefore expose multiple peaks. UI-side integration (focus-candidate
picker, per-candidate reconstruction) is tracked for v1.1-sci.

---

## What is *not* checked yet

- **NIH benchmark hologram**: the NIST / NIH digital holography test
  set with ground-truth `z` for each frame. Tracked for v1.1-sci.
  Acceptance: recovered `z` within ±0.5 mm on every frame, QPI
  `opd_range_nm` within ±15% of reference.
- **Off-axis pipeline** end-to-end: currently covered by
  `test_reconstruction_worker.py`'s phase checks, but not against a
  published reference hologram.
- **Noise robustness**: added Poisson shot noise and CCD read noise
  sweeps — planned for v1.1-sci.

---

## Numerical model

- Field representation: `complex64` (32-bit real, 32-bit imag).
- Frequency grid: `float32`, cached per `(shape, pixel_size, wavelength,
  n_medium)` tuple. Transfer function `H(z)` cached separately keyed
  by `z`.
- Propagation kernels: Angular Spectrum (evanescent waves clamped to
  zero) and Fresnel. Both documented in
  [reconstruction.py](../src/core/reconstruction.py).
- Phase unwrap: gradient-integration (dual-path, validated 2026-04-04;
  see v1.1 in [.claude/worktrees/VERSIONS.md](../.claude/worktrees/VERSIONS.md)).

---

## Reporting an accuracy regression

If you see reconstruction output drift between versions:

1. Run `pytest tests/test_reconstruction_accuracy.py -v` and attach the
   output.
2. Capture the hologram + parameters that drifted and add a
   reproducible fixture to `tests/conftest.py` under a new
   `synthetic_*` fixture.
3. Pin the test against the previous good release (`git bisect` over
   tagged versions) so the regression commit is identifiable.

See [SECURITY.md](../SECURITY.md) for the disclosure channel if the
regression affects a clinical or regulated workflow.
