# Changelog

All notable changes to DHM Reconstruction are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), semver.

## [2.0.0] — 2026-08-05: ui3 becomes the v2 frontend; ui2 (Dear PyGui) retired

### Changed
- **ui3 (PySide6 + pyqtgraph) is now the canonical v2 frontend** — built
  from scratch at 1:1 feature parity (`docs/UI3_DESIGN.md` coverage
  matrix), entry `run_ui3.py`.
- The framework-free compute drivers moved `src/ui2/` → `src/core/drivers/`
  (`reconstruction`, `workers`, `camera_feed`). Old `ui2.*` import paths
  remain valid via sys.modules-aliasing shims (same module objects).
- Autofocus default algorithm `zscan` → `robust`, pinned by a 9-scene
  real-lab benchmark (`docs/AUTOFOCUS_ADAPTIVE.md`, B-095). Frozen state
  migrations keep existing installs unchanged.

### Removed
- The Dear PyGui presentation layer (`ui2/app.py`, theme, widgets,
  dialogs, image panel, surface, device/AI panels) and `run_ui2.py`.
  Recoverable from git history. `ui2.state_store` (persisted settings +
  frozen v1..v10 migrations) and the driver shims remain.
- DPG-only tests; the driver/state tests inside mixed `test_ui2_*` files
  were kept (51 tests) — they now exercise `core.drivers.*` through the
  shims.

### Fixed
- 26 bug-registry entries B-072..B-097 from four review rounds the same
  day (ui3 wiring/ownership, MCP parity, observe scalebar/spectrum,
  moved-venv artefacts, relocation regressions). Suite green throughout.

### Track C — reference-free reconstruction (hybrid CNN)

End-to-end pipeline that removes the need for a reference hologram at
inference time. A small residual U-Net learns the reproducible
illumination + sensor + carrier-residual aberration the off-axis
demodulation pipeline can't fully cancel. See
[`docs/REFFREE_HYBRID.md`](docs/REFFREE_HYBRID.md).

#### Added
- `scripts/build_synthetic_refs.py` — temporal-median synthetic
  reference per session (handles bacteria-contaminated saved refs).
- `scripts/build_track_c_dataset.py` — generate `(phi_classical,
  phi_target, residual)` `.npz` triplets with outlier filtering.
- `scripts/train_track_c.py` — UNetLite training with AdamW, cosine
  LR, MPS-aware, NaN-skip guard.
- `scripts/eval_track_c.py` — quantitative + visual evaluation reports.
- `scripts/reffree_reconstruct.py` — production CLI; reconstructs a
  single hologram with no reference at runtime.
- `scripts/run_track_c_pipeline.sh` — one-command end-to-end orchestrator.
- `src/recon_dl/` — `unet_lite`, `dataset`, `losses`, `inference`
  modules. UNetLite ~3.3M params, near-zero residual init for graceful
  degradation.
- `tests/test_track_c_pipeline.py` — 6 smoke tests covering model,
  loss, cosine window, dataset, manifest, inference glue.
- `tasks/track_b_pure_dl_notes.md` — what would be needed to graduate
  to a pure end-to-end DL approach (data, architecture, validation).

#### Findings (recorded in tasks/lessons.md)
- Saved per-session "reference" frames contain bacteria; using them as
  GT pollutes training. Solution: temporal median across all frames in
  a session synthesizes a clean reference (moving bacteria average out).
- Autofocus mismatch between ref-based and ref-free pipelines unfairly
  inflated the original benchmark RMSE by ~25%. Fixed by locking z to
  the GT manifest's value when comparing.
- 10/63 GT manifest frames have outlier z values; the dataset builder
  filters them via session-median half-space heuristic.
- Track A (pure classical Zernike/poly fit) median test RMSE ~2.2 rad
  on centre crop; insufficient for the <0.15 rad target. Track C target
  is ~0.5 rad on centre crop with full LOSO.
- Track B (pure DL) requires 5,000+ frames and 8+ sample types to
  generalize; current 63-frame dataset is far too small.

## [1.0.1-ux] — 2026-04-21

Patch release responding to the Lindqvist-lab pilot's UX feedback.
Two-tier delivery: invisible plumbing (Tier 0) + visible design (Tier 1).

### Added — Tier 0 (plumbing)
- `core/progress.py` — `ProgressEvent`, `Operation`, `PhaseHandle` for
  phase-based progress reporting from workers.
- `core/errors.py` — `ErrorEvent`, `Severity`, `ErrorCenter` pub/sub bus
  with session history and builtin logger/audit sinks.
- `core/settings_schema.py` + `gui/settings_store.py` — typed
  `AppSettings` (recon/autofocus/QPI/IO defaults) with v1 → v2 migrator.
- `gui/commands.py` + `gui/commands_install.py` — `Command` registry,
  single source of truth for ids, shortcuts, and palette visibility.

### Added — Tier 1 (visible)
- **Command palette** (`⌘K`) — fuzzy search across registered commands,
  greys out `when`-blocked entries.
- **Toast + error drawer** — right-top overlay for warn/error events,
  right-docked session error log with collapsible tracebacks.
- **Silent progress line** — 1 px bar in the status bar; hidden under
  500 ms, line-only under 5 s, line + caption + Esc hint beyond.
- **Parameter persistence** — sidebar widgets apply on startup, save on
  successful recon/AF/QPI, `QFileDialog` defaults follow `last_folder`
  / `last_report_folder`.
- **Global Esc** = cancel the active worker (QPI → AF → recon walker),
  silent when nothing is running.

### Changed
- Reconstruction worker refactored into 7 phases with cooperative
  cancellation at each boundary (`OperationCancelled`).
- QPI and autofocus workers emit structured `ErrorEvent`s with
  phase-context fields for the error drawer.
- UI strings: 12 Turkish strings in `main_window.py` translated to
  English and shortened ("Line Profile başlat" → "Line profile", etc.).

### Security / Compliance
- `.pre-commit-config.yaml` + `scripts/check_language.py` — rejects
  Turkish characters inside `src/gui/` so mixed-language drift can't
  re-enter the codebase.

### Tests
- Full suite: **143 passing** (up from 107 in v1.0.0).

---

## [1.0.0] — 2026-04-20

First release targeted at commercial lab deployment.

### Added
- **Versioning & traceability**: `src/__version__.py` with semver, build date, best-effort git SHA. Exposed in About dialog and audit log entries.
- **Audit trail**: per-day JSONL log at `~/.dhm-reconstruction/audit/` capturing reconstruction, autofocus, QPI, and export actions with timestamps + parameters.
- **HTML report export**: one-click report generation embedding reconstruction parameters, phase image, and QPI summary (OPD, height, dry mass, roughness).
- **Right-click context menu** on phase and amplitude panels: Line Profile, 3D Surface, Export view.
- **Generate Report** action in the Tools menu.

### Changed
- Autofocus metrics (`src/core/autofocus/metrics.py`): all 11 metrics now guard against non-finite inputs and return a safe sentinel instead of silently propagating NaN into the search loop.
- `ENTROPY` metric: low-contrast holograms are detected and the search falls back to `PHASE_VARIANCE` with a user-visible warning.
- Error surfacing: reconstruction, autofocus, QPI, and batch workers now post failures to the status bar instead of swallowing silently.
- Metadata parser (`src/core/ingestion.py`): previously silent `except Exception` blocks now log at DEBUG level.
- QPI phase-unwrap fallback (`src/core/qpi.py`): wrap failures log at WARNING and annotate the result flag so downstream code can detect them.

### Fixed
- **Surface window rebuild** rendered pitch-black on second open — GLViewWidget now reused across refreshes, OpenGL context preserved.
- **QPI dialog replacement** killed the fresh 3D window via a stale `destroyed` signal — disconnect-before-close pattern applied.
- **Line profile** couldn't attach to the phase panel — ROI now defaults to phase when a reconstruction exists.
- **QPI auto-refresh** when an open QPI dialog exists and a new reconstruction completes.
- **Autofocus PHASE_VARIANCE** wrap bug: replaced `np.var(wrapped)` with circular variance `1 − |mean(exp(iφ))|`; lab focus now converges correctly.
- **AdGrad walker trap** on wide ranges: `prev_deriv` initialized to `None` so the first iteration doesn't trigger a spurious step shrink; 75% coverage safety net added in Phase 1.
- **Adaptive-distance budget** scales with `ceil(log2(max/init))` expansion depth instead of a flat heuristic.
- **PyOpenGL import failure** is now surfaced to the status bar instead of silently disabling the 3D Surface button.
- **Roughness metric** division-by-zero on flat height maps (`src/core/qpi.py`).
- **Tukey mask** producing rippled output when `rolloff = 0`; now falls back to hard mask below `0.01`.

### Security / Compliance (foundations)
- Audit log is append-only JSONL per day, suitable for downstream SIEM ingestion.
- Version and git SHA emitted in every audit record to pin results to a build.

---

## Unreleased (Roadmap)

See `docs/ROADMAP.md` for Faz 2 / Faz 3 commitments (AI segmentation, licensing framework, aberration compensation).
