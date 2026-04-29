# Roadmap

## Faz 1 — v1.0.0 (2026-04-20, shipped)

Commercial foundations. See `CHANGELOG.md` for the full list.

- Versioning + changelog + About dialog
- Audit trail (JSONL)
- HTML report export
- Core robustness (NaN/edge-case guards, ENTROPY fallback)
- UX polish (right-click context menu, surfaced errors, Generate Report action)

## Faz 2 — v1.1 (planned, 4–6 weeks)

### Report Polish
- PDF output in addition to HTML (WeasyPrint).
- Configurable report templates (lab logo, custom header, per-experiment metadata).
- Batch report generation across a job queue.

## Faz 2.5 — v1.1-sci / v1.2-tomo science patches

### v1.1-sci (shipped, 2026-04-21)
- **Multi-focus candidate picker** (`find_focus_candidates()` +
  `FocusCandidatesDialog`). Landscape peak-finding with prominence
  filtering. One-click *Focus here* drives the reconstruction trigger.

### v1.2-tomo alpha (shipped, 2026-04-21)
- **Per-pixel depth map** (`compute_depth_map()`) — tomographic height
  map using local Laplacian-variance / Tenengrad sharpness. Returns
  `(H, W)` z values + confidence + scanned z axis.
- **Export** — NPZ archive (loss-less) + CSV (R/MATLAB/pandas-friendly
  flat table) via `write_depth_map_npz` / `write_depth_map_csv`.
- Command `tools.compute_depth_map` in ⌘K, gated on a loaded hologram.
- Confidence masking helper (`mask_low_confidence`) — pixels below a
  fraction of peak confidence become NaN.

### v1.2-tomo beta (shipped, 2026-04-21)
- **Multi-candidate QPI batch** (`core.qpi_batch.run_qpi_for_candidates`,
  `write_qpi_batch_csv`) — scans the focus landscape, runs QPI once per
  detected plane, writes a multi-row CSV with the standard QPI columns
  plus ``candidate_rank`` / ``candidate_z_mm`` / ``candidate_prominence``.
- Command `tools.qpi_batch_candidates` in ⌘K, gated on a loaded
  hologram.
- Per-candidate reconstructions use gradient-integration unwrap — same
  path as the single-focus QPI in the app, so the numbers are directly
  comparable.

### v1.2-tomo final (in progress, 2026-04-21)
- **In-app batch review dialog** (shipped) —
  `QPIBatchReviewDialog` lets the operator compare dry mass / area /
  OPD range / step height / circularity across candidate focus
  planes without leaving the app. *Focus here* commits to a row's
  z; *Export CSV…* triggers the host's file-dialog + writer. The
  `tools.qpi_batch_candidates` command now opens the dialog first;
  the CSV is optional on the way out.

### v1.2-tomo final+ (shipped, 2026-04-21)
- **Depth-map overlay** on the phase panel (shipped) —
  `PhasePanel.set_depth_overlay(depth_map, alpha, colormap)` tints
  each pixel by its best-focus z. NaN pixels (low confidence) render
  fully transparent so background doesn't pollute the view.
  Command `tools.toggle_depth_overlay` computes the map, paints the
  overlay, and toggles it off on a second click. The last computed
  result is kept on `self._last_depth_map_result` for follow-on
  features (e.g. confidence-guided height).

### v1.2-tomo follow-up (shipped, 2026-04-21)
- **Confidence-guided cluster heights** — `segment_depth_clusters()`
  performs connected-component labelling over the high-confidence
  region of the depth map and returns `ClusterHeight` objects
  carrying `cluster_id`, `centroid_yx`, `area_px`, `z_mean_m` (with
  confidence weighting), `z_std_m`, `mean_confidence`. CSV export
  via `write_cluster_heights_csv()` includes optional `area_um2`
  when a pixel size is supplied.

### v1.3-polish (in progress, 2026-04-22)

**Cluster centroid markers (shipped).** `PhasePanel.set_cluster_markers()`
/ `clear_cluster_markers()` paint a dot + z-label at each
`ClusterHeight` centroid. The `tools.toggle_depth_overlay` command
now computes clusters alongside the depth tint, so one toggle shows
colour (depth) + markers (which object is where). Toggle-off clears
both.

**Unified "Export tomography bundle" (shipped).**
`write_tomography_bundle(directory, depth, clusters, qpi_batch, …)`
in `core.depth_map` writes every tomography artefact into one
directory with a shared ``base_name`` prefix: depth NPZ + depth CSV
+ cluster CSV + QPI batch CSV (QPI file omitted when the list is
empty). Target directory is auto-created if missing.

The `tools.export_tomography_bundle` command in ⌘K:
1. Reuses the cached depth result + clusters from the last overlay
   toggle if present (no redundant recompute).
2. Scans focus candidates and runs the QPI batch fresh.
3. Asks for an output directory and dumps all artefacts there with
   one timestamped base name.

**Live overlay recompute (shipped).**
`_install_live_overlay_observer()` wires the Focus-tab's
``zscan_min_mm``, ``zscan_max_mm``, and ``metric_combo`` signals to
a 500 ms-debounced ``QTimer``. While the depth overlay is on,
changing any of those parameters schedules a repaint; holding down
a spin-box arrow key coalesces into a single recompute after the
last edit. When the overlay is off the observer is a no-op — no
CPU spent on parameters nobody is looking at. The repaint path is
a shared ``_paint_depth_overlay_fresh()`` helper; toggle and
live-recompute share the same code so the result always matches the
current Focus-tab state.

**v1.3-polish complete.** All three goals (cluster markers, bundle
export, live recompute) shipped.

## Faz 3 — v1.4 UI Redesign (shipped, 2026-04-22)

The v1.x UI grew organically; the stretch goal here is a first-principles
rework driven by the lab pilot's feedback and the Jony Ive / Rams / Ma
constraints that shaped v1.0.1-ux. The tomography, autofocus, and QPI
pipelines stay as-is — this is a **presentation-layer** redesign only.

### Task-oriented workspace modes (alpha shipped)

Three primary modes replace the flat tab list:
- **Acquire** — Camera + Record (Live-mode work)
- **Reconstruct** — Recon + Process (pipeline tuning)
- **Analyse** — Focus + QPI (quantitative measurement)

Implementation:
- `gui.sidebar.sidebar_tabs.WorkflowMode` enum + combo above the tab
  widget. `set_workflow_mode()` filters which tabs are visible using
  `QTabWidget.setTabVisible` (Qt 6.2+).
- Mode persists across sessions via the same QSettings handle the
  window layout uses (`ui/workflow_mode` key).
- Mode switch lands on the first visible tab inside the active set,
  so the user never sees an empty pane.
- Report mode deferred to v1.4-beta (no dedicated Report tab yet —
  Tools menu still covers `tools.generate_report` /
  `tools.export_tomography_bundle`).

### Theme system (alpha shipped)

`gui.theme` ships three themes: **Light**, **Dark**, **System**
(follow platform). Implementation is palette-based: a token dict
per theme populates every `QPalette.ColorRole` we read, and existing
widgets (toast, command palette, error drawer, progress line) already
reference `palette(highlight)` / `palette(mid)` tokens, so the skin
propagates without touching their QSS.

Three ⌘K commands (`view.theme.light / .dark / .system`) switch the
active theme; the choice persists via QSettings (`ui/theme`) and
restores on next launch. On first run with no saved value we leave
the platform palette alone — zero-opinion default.

High-contrast theme deferred to v1.4-beta; needs WCAG-AA contrast
audit for every role we set.

### Redesigned sidebar (pilot shipped)

**Progressive disclosure pilot on ReconTab.** A reusable
`gui.widgets.collapsible_box.CollapsibleBox` hides secondary
parameters behind a single *Advanced ▸* expander. On `ReconTab` the
essentials (Wavelength, Pixel size, z, Reconstruct, Mask radius)
stay visible; Method, FFT Backend, Magnification, and
"Pixel-is-already-effective" collapsed by default. Widget instances
remain bound to `self` so every existing consumer (`get_state` /
`set_state`, persistence, main_window reads) keeps working.

**FocusTab rollout (shipped).** All four expert subgroups (Adaptive
Distance, Adaptive Steps, ROI Tracker, Live Tracking) now live
under one `CollapsibleBox("Advanced focus options")`, collapsed by
default. Individual QGroupBox *activate* checkboxes (``setCheckable``)
stay — this is a visibility refactor, not a semantic one. The
operator sees Algorithm / Metric / z-range / Auto-Focus by default;
expert paths unfold on request.

**QPITab rollout (shipped).** Phase-correction, 3D z-stack, and
display-panel selectors now live under one `CollapsibleBox`
("Advanced QPI options"), collapsed by default. Common flow
(n_sample / n_medium / threshold → Compute) stays clutter-free.

**Inline validation dots (shipped).** `gui.widgets.validation_dot.
ValidationDot` — 12-px coloured circle next to wavelength / pixel
spinboxes on ReconTab. Green = passes
`core.settings_schema.validate`, red = fails (reason in tooltip),
grey = not yet checked. Fires on every `valueChanged`.

**Preset chip row (shipped).** `gui.widgets.preset_chips.
PresetChipRow` sits below the Sample Preset combo on ReconTab.
Exclusive-select chips with a highlighted active state; selection
routes through the existing combo via `_on_preset_chip` so the
preset-apply handler stays the single source of truth.

### Report mode (shipped)

`WorkflowMode.REPORT` lights up a new `ReportTab` that surfaces
every export one-click: HTML/PDF report, QPI CSV, depth map,
tomography bundle, QPI batch. `main_window._bind_report_tab()`
wires the buttons to the existing command handlers — no duplicate
logic, just a task-focused entry surface.

### Accessibility (partial shipped)

Toolbar widgets (mode combo, Load action, Sample ID, Reconstruct)
and the workflow-mode combo carry `accessibleName` +
`statusTip` / `toolTip`. Keyboard: `?` opens the contextual
help overlay; `Esc`, `Ctrl+K`, `Ctrl+R` unchanged. Full
tab-order + focus-ring audit deferred to v1.5.

### Contextual help overlay (shipped)

`gui.widgets.help_overlay.HelpOverlay` walks the main window's
widget tree, collects every visible control's
`accessibleName` + `toolTip`, and renders them in a modeless
dialog. `?` keyboard shortcut opens it; `help.show_overlay` in ⌘K
re-opens anytime. Esc closes.

### Onboarding + help (wizard shipped)

`gui.widgets.onboarding.OnboardingWizard` walks a first-time
operator through five short pages: welcome, load a hologram,
reconstruct + autofocus, command palette + themes, and a close-out
page. The wizard is `QWizard`-based (ModernStyle, no Back on the
first page) and dismissible at any point.

First-run behaviour: on startup the window checks
``ui/onboarding_seen`` in QSettings; if the flag is absent the
wizard auto-opens 400 ms after paint so the main window is visible
behind it. Any exit path (Finish / Cancel / close) flips the flag.
The `help.show_onboarding` command in ⌘K re-opens the wizard on
demand.

Contextual help overlay + palette cheat sheet deferred to
v1.4-beta.

### Accessibility + keyboard
- Full keyboard navigation across the sidebar + toolbar (Tab order
  defined explicitly).
- Screen-reader labels on every icon-only action.
- Focus-ring visibility review (currently invisible in the dark
  theme in several spots).

## Faz 4 — v1.5 Commercial Hardening (planned, 8–10 weeks)

### Optical Correction
- Interactive reference-wave fitting (polynomial tilt/curvature subtraction).
- Aberration compensation (Zernike-basis fit against a flat-field hologram).
- Telecentric correction for off-axis setups with non-parallel reference wave.
- Multi-λ reconstruction with dispersion compensation.

### Throughput
- Parallel z-evaluation in autofocus (target 2.5–3.5× on Apple Silicon).
- GPU-resident FFT pipeline via MLX (end-to-end on device, no host roundtrips).
- Remote batch submission to a compute node.

---

## Out of scope (not planned for v1.x)

- **AI cell segmentation** (Cellpose / PyTorch MPS) — deferred until
  the tomography + UI redesign work proves stable. Non-AI
  segmentation in `core.qpi.segment_cell_phase` stays as the
  production path.
- **Licensing framework** (RSA-signed license files, trial mode,
  floating license server) — on-premise deployment model only, no
  license enforcement in the product.
- **Multi-user + RBAC + data management** (user accounts, audit-log
  viewer, SQLite experiment hierarchy, HDF5 provenance export) —
  single-operator model stays.
- **Windows / Linux builds** — Apple Silicon only for v1.x; revisit
  post-v2.
- **Cloud upload / SaaS mode** — on-premise only.
- **Custom acquisition hardware SDKs** beyond the current NI-IMAQdx
  integration.
