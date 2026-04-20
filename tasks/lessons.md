# Lessons — DHM Reconstruction (Hybrid)

Running log of course corrections. Every entry should name what went wrong, why, and what rule to apply next time.

---

## 2026-04-17 — Autofocus direction inferred from synthetic ≠ real

**Mistake**: Re-flagged TOTAL_VARIATION, GRADIENT, TENENGRAD, LAPLACIAN_VARIANCE, BRENNER, SPECTRAL_ENERGY as *minimize-at-focus* after watching a synthetic test case (a Gaussian phase bump) and reading `test_phase_af.py`'s Fresnel-ring-based reasoning. Real lab hologram then converged to z ≈ 117 mm instead of the landscape peak at ≈ 45 mm.

**Root cause**: Synthetic "phase bump on flat background" is dominated by defocus-induced Fresnel rings that *add* high-frequency content away from focus. Real DHM samples (cells, beads, stages) have sharp in-focus structure that dominates; at defocus, structure smears and gradients DROP. The directions flip between these two regimes.

**Rule**: Never set `_is_minimize` direction from synthetic reasoning alone. Always confirm on a real hologram with a full landscape scan — if the metric's global peak on the real scan sits at the same z where reconstruction looks focused to the eye, it's a *maximize* metric.

**Current direction convention**: only `ENTROPY` minimizes. Every other phase metric maximizes at focus.

---

## 2026-04-17 — Wrong venv when multiple virtualenvs exist

**Mistake**: Ran `./venv/bin/pip install PyOpenGL ...` from `Hybrid/`, but the shebang in that `pip` was a leftover hardcoded reference to `../Phyton/venv/bin/python`, so the package installed into the wrong env and `import pyqtgraph.opengl` still failed.

**Rule**: When a project lives next to sibling envs (`Hybrid/venv`, `Phyton/venv`, etc.), invoke pip as a module: `venv/bin/python -m pip install …`. That resolves the interpreter from the env's own binary instead of trusting a possibly-stale shebang.

---

## 2026-04-17 — Silent-import warnings swallowed in Qt code

**Mistake**: `_show_3d_surface` caught ImportError on `pyqtgraph.opengl`, logged a warning, and returned. User-facing result: the button did nothing, with no feedback. Diagnosis took far longer than it should have.

**Rule**: In GUI paths, when a missing-optional-dependency error prevents an action, *both* log it AND push a user-visible message (`self.status_bar.show_message(...)` in this codebase). Silent warnings in a Qt app are effectively swallowed — the log pane is usually not open.

---

## 2026-04-17 — Wrapped-phase variance explodes near ±π

**Mistake**: `PHASE_VARIANCE` computed `np.var(wrapped_phase)`. On real holograms the wrap boundary makes the histogram look bimodal at ±π, so the metric peaked at heavy defocus (z≈70mm) instead of the real focus (z≈45mm). Caused AdGrad to converge to the wrong z.

**Rule**: For *any* statistic on wrapped phase, decompose to sin/cos or use circular statistics. The safe variance is `1 − |mean(exp(iφ))|` (circular variance). Same rule applies to gradients (use `_wrap_diff`) and Laplacians (apply to sin/cos separately).

---

## 2026-04-17 — Greedy walker + wide range = local-max trap

**Mistake**: `adaptive_gradient_search` Phase 1 walker shrunk its step on the first rising shoulder and never reached the global peak. Budget exhausted before traversing the full range, so Phase 2 refined around a wrong best_z and couldn't escape.

**Root causes (compounded)**:
1. `prev_deriv = 0.0` initial value tripped the "derivative grew" branch on the very first iteration → step shrank before the walker moved.
2. No coverage check — if the walker only covered 60% of [z_min, z_max], Phase 2's local refinement couldn't rescue.
3. `ad_budget = max(30, steps * 0.5)` in the worker couldn't reach ±50mm from ±0.5mm (log₂(100) ≈ 7 expansions × 15 pts = 105 evals needed).
4. Worker forced `step_init = (zmax-zmin)/20` after AdDist, which made the walker take 20 tiny steps to traverse instead of using the algorithm's natural `full_range/10` default.

**Rule**: A greedy local-search walker must have either (a) a guaranteed-coverage fallback (uniform sweep if the walker didn't traverse the range), or (b) a budget-aware step floor that forces traversal. Pre-validation belongs in the algorithm — don't trust an outer pipeline to supply "good enough" ranges.

**Rule (worker sizing)**: When an expanding-discovery phase feeds downstream refinement, budget it by `ceil(log2(max_range / init_range))` iterations, not a flat `max(30, steps/2)`. That turns the budget into a property of the problem instead of a property of the UI.

---

## 2026-04-17 — Know when NOT to refactor

**Context**: Plan called for splitting `main_window.py` (2653 lines) into mixin classes. Started mapping the sections; backed off before writing code.

**Rule**: Mixin-splitting a single Qt `QMainWindow` class trades a readability win for worse tooling (type checkers can't track cross-mixin `self.x`, IDE jump-to-def breaks, subtle MRO hazards). Only do it when the file is *actively painful to edit*, not merely large. For GUIs, prefer extracting *standalone* subwidgets / free-function helpers into `src/gui/views/` rather than horizontally slicing one class. Independent-function modules (like `src/core/autofocus.py`) split cleanly and are the right candidates; stateful Qt classes usually aren't.

---

## 2026-04-17 — Async `destroyed` signal killed the replacement window

**Mistake**: `_show_qpi_window` connected `dlg.destroyed → _cleanup_3d_window`. When a new reconstruction auto-refreshed QPI, the old dialog was closed (WA_DeleteOnClose) and a new one opened immediately. The old dialog's `destroyed` signal fires *asynchronously* after the new 3D window already exists, so `_cleanup_3d_window()` ran on the fresh widget and tore it down.

**Rule**: When you have a parent→child cleanup wired through a `destroyed` signal, *disconnect it* before you replace the parent programmatically. `WA_DeleteOnClose` means close is not destroy — the destroy fires later, after whatever new object you've created in between. Pattern:
```python
try:
    dlg.destroyed.disconnect(self._cleanup_3d_window)
except (TypeError, RuntimeError):
    pass
dlg.close()
```

---

## 2026-04-17 — Destroying a GLViewWidget and rebuilding it renders black

**Mistake**: `_show_3d_surface` used to `deleteLater()` the old `GLViewWidget` and create a fresh one every call. On the second open the new widget rendered pitch-black even though the data was identical. Camera, grid, items all present — just no visible draw.

**Root cause**: `deleteLater` schedules an async destroy. The OpenGL context teardown for the *old* widget happened on the same GUI tick as the `initializeGL` call for the *new* widget, and Qt didn't give the new context a clean slate.

**Rule**: Prefer *reusing* a `GLViewWidget` (or any QOpenGLWidget) across refreshes. Check `shiboken6.isValid(w)` to tell whether the Python handle still points at a live C++ object; if yes, call `w.removeItem(item)` for each item in `w.items` and re-add fresh items. Only create a new widget if the old one is truly dead (user closed via X with WA_DeleteOnClose, mode switch cleaned it up, etc.). Camera position should only be set on *first* open so the user's rotation survives refresh.

**Tangential rule**: For widgets meant to be long-lived (progressive refinement, live refresh), don't set `WA_DeleteOnClose`. Let `hide()` + reshow handle the open/close cycle; reserve deletion for mode changes and file switches.
