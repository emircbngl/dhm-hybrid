# ui3 — DHM desktop, rebuilt from scratch (2026-07-05)

Toolkit: **PySide6 (Qt 6) + pyqtgraph** (imaging & plots, GPU-capable, in-process) +
`pyqtgraph.opengl` (3D surface) + matplotlib (publication-figure export only).

## Why this toolkit (evaluated, not defaulted)
- **Web/Tauri rejected:** the compute core is numpy/scipy; a web front streams
  1200×1600 complex64 arrays across an IPC boundary every frame — a real
  latency/throughput tax for live reconstruction/autofocus/3D, plus two-process
  packaging on a node-locked macOS deploy. Optimises polish at the cost of the
  app's core competency.
- **Dear PyGui (old ui2) rejected:** the single largest bug-registry phase
  (DPG_PORT, 17 bugs) was macOS silent failures — drop callback, file dialog,
  viewport menu, Latin-1 font, no modals. Qt kills that class outright.
- **PySide6 + pyqtgraph chosen:** the de-facto standard for scientific Python
  desktop imaging (napari, Orange, Spyder). In-process numpy→GPU display of big
  arrays is exactly what DHM needs; native dialogs/docks/keyboard/modals; 3D and
  the AI panel come home natively (no subprocess/event-pump hacks).

`src/ui2/` (Dear PyGui) stays untouched until ui3 reaches parity, so the 100+
ui2 tests and the just-landed reffree/AI work aren't broken mid-flight.

## Design language — "instrument"
A calm, dark scientific-instrument aesthetic. Hierarchy by *scale & space*, not
weight. Accent reserved for **live state** (running, armed, connected). Mono
numerals so readouts align as the operator drives the stage.

Design tokens live in `ui3/design.py` (pure data — Qt-free, testable):
- **Color** (dark default): `bg=#0d1117`, `surface=#151b23`, `surface_high=#1c2330`,
  `border=#2a3441`, `text=#e6edf3`, `text_muted=#8b98a5`, accent `cyan=#3fb6c8`,
  `amber=#e0a458` (armed), `green=#57ab5a` (ok/connected), `red=#e5534b` (error/danger).
  Also `light` and `high_contrast` palettes (WCAG-AA verified — `ui3/wcag.py`).
- **Type:** UI = system sans; **numerals = SF Mono / monospace** everywhere a
  number is read. Scale: 11/13/15/20/28 px.
- **Space:** 4-based scale (4/8/12/16/24/32). Panels breathe; separators only
  between conceptually distinct cards.
- **Radius:** 6 px cards, 4 px controls. 1 px borders, never shadows-as-drama.

## Architecture
```
run_ui3.py → ui3.app.main()
  QApplication (crash handler, theme) → MainWindow (QMainWindow, dock-based)
    design.py    — tokens + qss builder (theme)               [Qt-free tokens]
    wcag.py      — contrast audit of the palettes              [Qt-free]
    state.py     — typed AppState (reuses core.settings_schema) + JSON persist
    bridge.py    — WorkerBridge: QThread pool over core.*, Qt signals out
    viewport.py  — ImageView (pyqtgraph) w/ zoom/pan/colormap/scalebar/ROI/drop
    panels/*     — each dockable feature panel (below)
    dialogs/*    — modal/non-modal dialogs
    widgets/*    — reusable atoms (command palette, toasts, chips, validation dot…)
```
Core is Qt-free and shared with ui2/v1 (no core changes needed). All heavy work
runs on `bridge` worker threads; panels never block the GUI thread. Reference-mode
+ reffree pipeline reuse `core.pipelines.reffree_hybrid` + `core.background_phase`.
AI copilot reuses the proven v1 `core.ai.Agent`/`AIWorker` pattern (native QThread),
not the broken ui2 daemon-thread port. Observation/vision tools via `core.observe`.

## COMPLETE coverage matrix (nothing skipped — mapped 1:1 from ui2)
| Area | ui2 source | ui3 home |
|---|---|---|
| App shell / menu bar / status bar | app.py | main_window.py |
| Command palette (⌘K) | widgets.py | widgets/command_palette.py |
| Toasts / status line | widgets.py | widgets/toast.py |
| Theme (dark/light/midnight/high-contrast) | theme.py | design.py + wcag.py |
| Preset chips + save/replace/delete preset | app.py/widgets.py | panels/recon_panel.py + dialogs/preset_dialogs.py |
| Workflow modes (Acquire/Reconstruct/Analyse/Report) | app.py | main_window.py (mode filter) |
| 4-panel image grid (input/amp/phase/spectrum) zoom/pan/scalebar/drop | image_panel.py | viewport.py (×4 in a grid) |
| Recon params sidebar (λ, px, z, mask, method, backend, mag, effective-px, unwrap, optical mode…) | app.py/reconstruction.py | panels/recon_panel.py |
| Reference mode (off/reference/reference-free + bg method/order/n-terms + CNN gate) | app.py/workers.py | panels/recon_panel.py |
| Load hologram / recent files / drag-drop | app.py | main_window.py + viewport.py |
| Load/clear reference hologram | app.py | panels/recon_panel.py |
| Reconstruct | app.py/reconstruction.py | bridge + panels/recon_panel.py |
| Autofocus + find focus candidates + ROI | app.py/workers.py | panels/focus_panel.py + dialogs/focus_candidates.py |
| QPI (compute) + QPI batch review | app.py/workers.py/dialogs.py | panels/qpi_panel.py + dialogs/qpi_batch.py |
| Depth map + overlay + clear | app.py/workers.py | panels/depth_panel.py |
| 3D surface (depth & phase) | surface.py | dialogs/surface_viewer.py (pyqtgraph.opengl, native) |
| Line profile (enable mode + dialog) | line_profile_state.py/dialogs.py | widgets/line_profile.py + dialogs/line_profile.py |
| Scalebar | image_panel.py + core.scalebar | viewport.py |
| Export: report / QPI CSV / tomography bundle | app.py | panels/report_panel.py |
| Audit viewer | dialogs.py | dialogs/audit_viewer.py |
| Camera feed (start/stop/record) | camera_feed.py | panels/camera_panel.py |
| Device panel (stage HUD/jog/shutter/LED) | device_panel.py | panels/device_panel.py |
| Timelapse / multi-position / tracking | (core) | panels/timelapse_panel.py |
| AI copilot chat | ai_panel*.py/ai_bridge.py | panels/ai_panel.py (+ reuse core.ai) |
| Vision tools surfaced in chat (inspect/render_view) | (new core.observe) | panels/ai_panel.py |
| Onboarding wizard (first-run + manual + reset) | app.py | dialogs/onboarding.py |
| State persistence + migration | state_store.py | state.py (reuses core.settings_schema Ui2State) |
| Responsive layout / panel maximize (⌘1–4/0) | app.py | main_window.py |
| WCAG palette audit | wcag.py | wcag.py |

## Build phases
0. **Spine** (this turn, by hand): design.py, wcag.py, state.py, bridge.py,
   viewport.py, app.py, main_window.py, run_ui3.py — a launchable, themed,
   dockable shell with the working 4-panel imaging viewport + load→reconstruct.
1. **Panels** (workflow, parallel): recon+reference, focus, qpi, depth, report,
   camera, device, timelapse, ai — each dockable, tested headless (offscreen Qt).
2. **Dialogs & widgets** (workflow): surface_viewer, qpi_batch, focus_candidates,
   audit_viewer, onboarding, preset_dialogs, command_palette, toast, line_profile.
3. **Integrate + parity sweep**: wire menus/modes/persistence/palette; offscreen
   smoke of every action; parity checklist against this matrix; review.

## Test strategy
All panels/widgets constructed under `QT_QPA_PLATFORM=offscreen` with a real
`QApplication` (no DPG-style stubbing needed — Qt runs headless natively). Logic
(state, bridge job shaping, param round-trips, wcag) tested Qt-free where possible.
`tests/test_ui3_*.py`. A single `test_ui3_smoke.py` constructs the whole MainWindow
offscreen and drives load→reconstruct→qpi to prove end-to-end wiring.
