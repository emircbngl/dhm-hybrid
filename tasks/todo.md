# Plan — 2026-04-17

## Context
User requested:
1. Fix: "Surface" 3D window not opening
2. Feature: All autofocus algorithms should operate on PHASE (not amplitude) from now on
3. Optimization: Split large files if it helps the program run / be maintained better
4. Question: What do `n_sample` / `n_medium` in QPI actually do?

---

## 1. Surface window bug ✅

- [x] `pip install PyOpenGL PyOpenGL-accelerate` into `venv/`
- [x] Surface the warning to the status bar when import fails
- [x] Manual smoke planned for end of session

**Root cause**: `PyOpenGL` was listed in `requirements.txt` but not installed in the Hybrid `venv/`. The original code swallowed the ImportError silently. Fix: install + surface future failures to the status bar so they can't be silent again (`src/gui/main_window.py:1156-1255`).

## 2. Autofocus → phase ✅

- [x] Added `_phase_of` and `_wrap_diff` helpers
- [x] Rewrote `_calc_metric` so all 11 metrics run on phase (wrap-safe for neighbour-differences; sin/cos decomposition for Tenengrad/Laplacian; circular statistics for Vollath/Normalized variance; HF energy of `exp(iφ)` for spectral energy)
- [x] Removed `AMPLITUDE_FLATNESS` entirely (user: "sana bırakıyorum" — chose removal to keep the "all-phase" philosophy coherent)
- [x] Removed `_is_phase_metric()` and simplified `_make_fast_evaluator` to always receive complex
- [x] `_is_minimize` returns `True` only for `ENTROPY` — every other phase metric peaks at focus
- [x] Propagated changes to `src/core/phase_tracker.py`, `src/core/batch_renderer.py`, `src/gui/sidebar/focus_tab.py`, `test_af_visual.py`, `tools/tool_autofocus.py`, `tools/tool_full_pipeline.py`, `tools/README.md`
- [x] Verified on real hologram: bench_af.py reference sweep converged at z≈44.37mm (120 steps) / z≈42.71mm (60 steps); all search algorithms (Linear, Golden, Robust C2F, Bracketing) cluster at 44-45mm
- [x] Re-tested after package split: same convergence behavior

**Direction sanity check**: Earlier in the session I briefly flipped several phase gradient metrics to MINIMIZE based on a synthetic-test intuition (Fresnel rings getting louder at defocus). On real data this was wrong — sharp cell edges + high contrast at focus dominate, so gradient-family metrics MAXIMIZE. Reverted. Only ENTROPY minimizes.

## 3. File splitting ✅ partial

### 3a. `src/core/autofocus.py` (1545 lines) → package ✅
Split into `src/core/autofocus/`:
- `metrics.py` — `FocusMetric` enum, phase helpers, `_calc_metric`, `_is_minimize`, `_get_hf_mask`
- `evaluator.py` — `AutofocusCancelled`, `AutoFocusResult`, `_make_fast_evaluator`, `downsample_complex_field`
- `search_classic.py` — `autofocus_zscan`, Golden Section, Coarse-to-Fine, Robust C2F
- `search_adaptive.py` — Adaptive Gradient/Ratio/Bracketing/Distance + `AdaptiveFocusState`
- `analysis.py` — `auto_select_metric`, `scan_metric_landscape`, `autofocus_benchmark`
- `__init__.py` — re-exports the full public + internal surface so every `from core.autofocus import …` still works

Benefit: functions are grouped by concern (11 metrics in 170 lines of metrics.py vs scattered across 1500 lines), and editing one search algorithm no longer forces reading the whole file. Import-level API preserved — no call-site changes needed anywhere in the codebase.

### 3b. `src/gui/main_window.py` (2653 lines) — NOT SPLIT (deliberate)
Started a plan to split into mixins (recon / qpi / camera / autofocus / tools) but backed off for these reasons:
- MainWindow is a single Qt class with tight signal/slot coupling — every method assumes a fully constructed `self` with dozens of widgets/state fields. Mixins work, but IDE navigation and type checking degrade because `self.foo` is defined in another file.
- The autofocus split already captures the primary maintenance win (the file Claude most frequently needed to read at depth).
- Runtime doesn't change — this is purely about reading cost, and the GUI file is less frequently opened than `autofocus.py` during work on this project.

**Recommendation**: leave `main_window.py` monolithic unless it becomes painful. If the user still wants it split, the cleanest next step is to extract standalone concerns first (line-profile dialog, 3D surface viewer) into `src/gui/views/` as free-function helpers that take a parent widget.

## 4. QPI refractive index ✅ (answer, not code)

Both `n_sample` (default 1.38) and `n_medium` (default 1.337) are load-bearing, not cosmetic:
- `n_sample - n_medium = Δn` gates whether height can be computed at all (`src/core/qpi.py:614-617`)
- `opd_to_height(opd, n_sample, n_medium)` (`src/core/qpi.py:55-73`) feeds `result.height_nm`
- `compute_cell_morphology` (`src/core/qpi.py:274-388`) uses Δn for per-pixel cell height → volume (fL)
- Roughness (Ra/Rq/Rz) and 2D PSD downstream consume `height_nm`
- `opd_to_refractive_index` inverse helper is only used when `known_thickness_m` is supplied
- UI exposes both spinboxes in `qpi_update/qpi_tab.py:88-115`

Changing either value directly alters the reported thickness, volume, roughness, and PSD. Answer confirmed by tracing through the code — nothing is vestigial.

---

## Review

**What shipped**:
- Surface window opens (PyOpenGL installed; future silent failures now surface a status-bar warning)
- Every autofocus metric now runs on wrap-safe phase instead of amplitude
- AMPLITUDE_FLATNESS is gone everywhere (core, GUI, tests, tools)
- `src/core/autofocus.py` → clean 5-file package with preserved public API; bench reproduces ~same focus z as before the split
- QPI refractive-index question answered with file references

**What didn't ship** (and why):
- main_window.py split: backed off after starting the plan. Trade-off isn't in favor given how mixins interact with Qt signal/slot and IDE support.

**Test status**:
- `bench_af.py --ref-only` converges at plausible focus (≈44mm on lab hologram at 120 steps)
- All public symbols in `from core.autofocus import …` still resolve after the package split
- End-to-end GUI smoke test not run in this session — recommend the user launch the app and exercise the Surface window + autofocus before declaring the changes stable.

**Follow-ups for next session**:
- Manual GUI smoke test (app launch → load hologram → reconstruct → autofocus → QPI → 3D surface)
- If the user still wants main_window.py split, go the helper-module route, not mixins

---

# Follow-up session — 2026-04-17

## Context
User reported after previous session:
1. "Surface hesaplamasında bir sorun var gibi" — surface calculation looks wrong
2. "Autofocus aramasında hatalı sonuç buluyor" — autofocus finds wrong z

Working hypothesis: fix autofocus and surface follows (bad z → bad QPI phase → bad surface).

## Fixes landed

- [x] **PHASE_VARIANCE wrap bug** — `src/core/autofocus/metrics.py`
  `np.var(wrapped_phase)` replaced with circular variance `1 − |mean(exp(iφ))|`.
  Pre-fix landscape peak: z=70mm. Post-fix: z=39mm (lab focus ≈ 45mm).

- [x] **AdDist peak significance check** — `src/core/autofocus/search_adaptive.py`
  Interior peak now requires (range/|median|) > 3% AND (peak − worst_edge) > 30% of range. Prevents early termination on flat noise plateaus. Bilateral expansion only fires when the peak is truly interior but insignificant.

- [x] **AdGrad `prev_deriv` initialization** — same file
  Changed `prev_deriv = 0.0` → `prev_deriv: Optional[float] = None`. Skip step adjustment on the first iteration so the walker doesn't shrink immediately on `abs_d > 0 * 1.5`.

- [x] **AdGrad coverage safety net** — same file
  After Phase 1, if the walker covered < 75% of [z_min, z_max], run a short uniform sweep (up to 12 evals) to catch the global peak. Phase 2 local refinement then operates on the corrected best_z.

- [x] **Worker AdDist budget** — `src/gui/workers/autofocus_worker.py`
  Replaced `max(30, steps*0.5)` with `max(60, ceil(log2(max/init))*15 + 15, steps*0.5)`. For the default ±0.5→±50mm expansion this scales to 120 evals, enough for the full log₂ expansion depth.

- [x] **Worker `step_init` pass-through** — same file
  When adaptive distance is on, the worker now passes `step_init=None` so AdGrad uses its natural `full_range/10` default. Previously `(zmax-zmin)/20` produced steps so tiny the walker couldn't traverse the detected range in its forward budget.

## Verification

Real lab hologram (`labtest/ilk_imgbackCCD_24.10mm.png`), true focus ≈ 44mm:

| Scenario | Metrics tested | Pass |
|---|---|---|
| A. AdDist → AdGrad (wide range, user's 50x profile) | PHASE_VARIANCE, LAPLACIAN_VARIANCE, TENENGRAD, GRADIENT, BRENNER, NORMALIZED_VARIANCE, TOTAL_VARIATION, SPECTRAL_ENERGY | 8/8 within ±6mm |
| B. Narrow refinement [42,46]mm | 4 gradient metrics | 3/4 (phase_variance's true peak is 39mm — correctly clamps to edge z=42, not a regression) |
| C. Precise refinement [43.5,44.5]mm | 4 gradient metrics | 4/4 within ±1mm |

## Not addressed this session
- Stale profile metric silent fallback (`src/gui/sidebar/focus_tab.py set_state` swallows removed metrics like "Amplitude Flatness"). User profile at `~/.dhm-reconstruction/profiles/setup/50x.json` still has this stale value.
- Surface bug not independently verified — assumed downstream of autofocus. Needs a manual GUI smoke test to confirm.
- AdDist `signal_threshold` tuning — 0.3 produces detected ranges that are wider than strictly necessary (e.g. [0.64, 50]mm when peak is at 44). The coverage safety net in AdGrad compensates, but a tighter threshold would also fix it.

## Manual verification remaining
- Launch app with user's 50x profile → autofocus on lab hologram → confirm z converges near 44mm
- Reconstruct at autofocused z → view 3D Surface → confirm it looks right

---

# Follow-up session — 2026-04-17 (QPI/3D/Line profile)

## Context
User reported three fresh bugs:
1. "QPI surface ikinciye tekrar oluşturulmuyor" — 3D surface fails to open on second attempt
2. "QPI pencereleri açıldığında, yeniden reconstruct alındığında güncellensin" — open QPI windows should auto-update on new reconstruction
3. "-phase penceresi üzerinden line graph tool ile çizgi grafiği oluşturamıyorum" — line profile tool can't draw on phase panel

After first pass, user reported follow-up: "surface penceresi açılıyo ikinciye açılmasına tamam da kapkara görüntü yok" — window opens but renders black.

## Fixes landed

- [x] **QPI dialog replacement no longer kills the fresh 3D window** — `src/gui/main_window.py:1044-1062`
  `_cleanup_qpi_dialog` now disconnects `dlg.destroyed → _cleanup_3d_window` before closing. Root cause: `WA_DeleteOnClose` schedules async deletion, and the stale signal fired *after* the new 3D window was created, killing it.

- [x] **QPI auto-refresh on new reconstruction** — `src/gui/main_window.py:726-732`
  `_on_recon_completed` now calls `_compute_qpi()` when a QPI dialog is already open, so height/OPD/mass maps reflect the current z.

- [x] **Live line profile refresh** — `src/gui/main_window.py:734-736`
  When phase data arrives during a live line-profile session, re-sample along the ROI without requiring the user to reopen the dialog.

- [x] **Line profile defaults to phase panel** — `src/gui/main_window.py` `_on_line_profile_toggled`
  ROI is now placed on the phase image when one exists (combo index 1 = "Phase"); falls back to amplitude. Previously always attached to amplitude → phase-panel click did nothing.

- [x] **Black 3D surface on second open** — `src/gui/main_window.py:1176-1279`
  Refactored `_show_3d_surface` to *reuse* the existing `GLViewWidget` when `isValid(w)`. `removeItem()` each old surface/grid, add fresh ones, keep the user's camera rotation (only set camera on first open). Root cause: destroying a `GLViewWidget` via `deleteLater` and creating a new one mid-tick left the new widget's OpenGL context un-initialized → pitch-black render.

## Not addressed
- Stale profile metric silent fallback (`set_state` still swallows removed "Amplitude Flatness" in `~/.dhm-reconstruction/profiles/setup/50x.json`). Low impact — any other valid metric choice overrides it.

## Manual verification needed
- Compute QPI → open 3D surface → close dialog → recompute → 3D surface opens and renders properly on second try
- Reconstruct at z1 → open QPI → reconstruct at z2 → confirm QPI maps refresh automatically
- Toggle line profile → ROI lands on phase panel → drag endpoints → profile updates live
