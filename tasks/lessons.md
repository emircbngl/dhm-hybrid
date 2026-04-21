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

---

## 2026-04-20 — `git add -A` pulled in 56k lines of worktree backups

**Mistake**: Asked to save a snapshot, ran `git add -A` after only seeing the first 30 lines of `git status --short` (piped through `head -30`). That sample hid untracked directories — including `.claude/worktrees/v1.0_apr03_base/` (a full copy of an old version) and macOS `.DS_Store` files. Committed 561 files / +56k lines instead of the ~10 real changes.

**Root cause**: Truncated status output + blind wildcard stage. Never enumerated what `-A` would actually add.

**Rule**: Before any `git add -A` or `git add .` on a dirty repo:
1. Run `git status --short` *unfiltered* (or pipe to `| wc -l` first to gauge size).
2. Scan untracked section for unexpected top-level directories (`.claude/`, `worktrees/`, `node_modules/`, anything macOS/IDE-specific).
3. Ensure `.gitignore` covers local-only state *before* staging, not after.
4. Prefer `git add <explicit paths>` when the modified set is small and known.

**Recovery pattern used (non-destructive)**: `git tag -d <tag>` → `git reset --mixed HEAD~1` → fix `.gitignore` → re-stage → re-commit → re-tag. No force-push, no `reset --hard`, working tree preserved.

---

## 2026-04-21 — `QShortcut` moved to `QtGui` in PySide6 6.x; unit tests don't catch it

**Mistake**: `main_window._setup_shortcuts` and `gui/commands_install.install_shortcuts` both imported `QShortcut` from `PySide6.QtWidgets`. All 137 pytest cases passed — they exercised the command *registry* but not the QShortcut *binding* path, because no test instantiated a full `MainWindow`. Manual smoke (`MainWindow()` under offscreen Qt) immediately raised `ImportError: cannot import name 'QShortcut' from PySide6.QtWidgets`.

**Root cause**: PySide6 6.0 moved `QShortcut` from `QtWidgets` to `QtGui`. Widget-level tests that only touch individual widgets never trigger the failing import; only a full main-window boot does.

**Rule**: For any task that touches `main_window.py`, include a headless-Qt smoke that actually runs `MainWindow()` to construction completion. A 5-line smoke (`QApplication([]); MainWindow()`) catches whole classes of "import-at-first-use" regressions that unit tests miss. Add the smoke to the verification checklist, not just the unit tests.

**Qt version note**: for PySide6 6.x use `from PySide6.QtGui import QShortcut, QKeySequence, QAction`. Pinning the import at the Qt-moved site (not at module top-level) limits the blast radius.

---

## 2026-04-21 — `np.bool_` ≠ `bool` in `is` checks

**Mistake**: `has_sufficient_contrast()` in `src/core/autofocus/metrics.py` returned `circ_var >= min_circular_std`, which is a `numpy.bool_` (not Python `bool`). Two pytest cases used `assert has_sufficient_contrast(...) is False` / `is True` — identity check against Python singletons — and failed because `np.False_ is False` → `False`.

**Root cause**: The function's type hint said `-> bool` but the body leaked a numpy scalar. Test authors trusted the annotation and used identity comparison (which is idiomatic and correct for Python booleans).

**Rule**: When a function's return annotation is `bool`, make it a real Python `bool` at the boundary: `return bool(expr)`. Don't let numpy scalars escape typed APIs — they look equal but fail `is` checks and pickle oddly. Same rule applies to `int` / `float` annotations returning `np.int64` / `np.float64`.
