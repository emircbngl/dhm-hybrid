"""Central bug-history registry — every regression we've ever had,
organised by phase.

This module is **data-only**. The runner (``scripts/_bug_runner.py``)
and the CLIs (``scripts/check_bugs.py`` + the per-phase wrappers)
import :data:`BUG_REGISTRY` and filter / format from there.

Adding a bug
------------
After every sprint, append entries here with:

* ``bug_id`` — ``B-NNN``, monotonically increasing.
* ``date`` — ``YYYY-MM-DD`` of the fix landing.
* ``topic`` — one-line plain-English (Turkish OK).
* ``phase`` — :class:`Phase` enum value the bug belongs to. New
  versions / sprints earn a new enum member.
* ``status`` — ``'test'`` (auto-runnable), ``'lesson_only'`` (memory
  entry, no surface), or ``'manual'`` (hardware/env-dependent).
* ``test`` — pytest node id when ``status='test'``.
* ``lesson_ref`` — heading reference into ``tasks/lessons.md``.

Adding a phase
--------------
1. New :class:`Phase` enum member with a short kebab-case key.
2. Backfill the ``phase`` field for any newly-discovered bugs.
3. Create ``scripts/check_bugs_phase_<key>.py`` (3-line wrapper).
4. Append the phase to ``Phase`` order so reports stay chronological.

Convention from ``tasks/lessons.md`` (2026-04-27): every shipped
phase earns its own bug-check script. A lab demo asking "is phase
2.0.7 still green?" must finish in seconds, not minutes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional


class Phase(str, Enum):
    """Discrete sprint / version phases. Order matters — chronological,
    oldest first; report ordering reads from this enum."""

    PRE_PILOT = "pre_pilot"
    """Faz 1 / v1.0.0 prep. Early autofocus, Qt UI, PySide6 6.x port quirks."""

    UX_PATCH = "ux_patch_v1_0_1"
    """v1.0.1-ux pilot patch. Tier 0 plumbing + Tier 1 visible design."""

    DPG_PORT = "dpg_port_v2_0_x"
    """v2.0.0 → v2.0.6: Dear PyGui frontend, scientific param port,
    compliance + workflow tools mega sprints."""

    PILOT_PATCHES = "pilot_patches_v2_0_6_post"
    """5-bug acil sprint + end-to-end synthesis tests + multi-focus
    `_make_fast_evaluator` refactor (2026-04-24)."""

    TIMELAPSE_FOUNDATION = "timelapse_v2_0_7"
    """v2.0.7 — current sprint. Session model, headless CLI, multi-user
    profile, preset edit/replace, audit viewer, batch resume."""

    # Future phases — empty registries until those sprints surface bugs.
    # Add to enum when sprint kicks off; the script wrapper can reference
    # the phase before any entries land.
    TRACKING = "tracking_v2_0_8"
    """v2.0.8 — drift correction, per-cell tracking, NIST calibration."""

    PAPER_READY = "paper_ready_v2_0_9"
    """v2.0.9 — vector PDF, Zenodo bundle, line profile ROI, crash
    handler, WCAG-AA."""

    PERF_GPU = "perf_gpu_v2_1_0"
    """v2.1.0 — PyTorch backend, batch FFT, ROI fast-path, headless CI."""

    HARDWARE = "hardware_v2_1_x"
    """v2.1.x — Pylon/IDS/Thorlabs CameraSource, MP4 recording."""

    AI_FAZ_2 = "ai_seg_v3_0"
    """v3.0 / Faz 2 — Cellpose, cell-cycle classifier, onboarding wizard."""


Status = Literal["test", "lesson_only", "manual"]


@dataclass
class BugEntry:
    """One historical bug, with phase + regression check."""
    bug_id: str
    date: str
    topic: str
    phase: Phase
    status: Status = "test"
    test: Optional[str] = None
    lesson_ref: Optional[str] = None


# ---------------------------------------------------------------------------
# THE REGISTRY — every historical bug, oldest to newest. Append-only;
# never re-number, never re-phase a shipped entry (that erases history).
# ---------------------------------------------------------------------------

BUG_REGISTRY: List[BugEntry] = [
    # ── Phase: PRE_PILOT (Faz 1 / v1.0.0 prep) ──────────────────────────
    BugEntry(
        bug_id="B-001", date="2026-04-17",
        phase=Phase.PRE_PILOT,
        topic="Autofocus direction inferred from synthetic ≠ real "
              "(only ENTROPY minimizes at focus)",
        status="test",
        test="tests/test_focus_validation.py::"
             "test_autofocus_single_sphere_recovers_z",
        lesson_ref="2026-04-17 — Autofocus direction inferred",
    ),
    BugEntry(
        bug_id="B-002", date="2026-04-17",
        phase=Phase.PRE_PILOT,
        topic="Wrong venv when multiple virtualenvs exist "
              "(stale pip shebang)",
        status="lesson_only",
        lesson_ref="2026-04-17 — Wrong venv when multiple virtualenvs",
    ),
    BugEntry(
        bug_id="B-003", date="2026-04-17",
        phase=Phase.PRE_PILOT,
        topic="Silent ImportError swallowed in Qt 3D-surface code",
        status="lesson_only",
        lesson_ref="2026-04-17 — Silent-import warnings swallowed",
    ),
    BugEntry(
        bug_id="B-004", date="2026-04-17",
        phase=Phase.PRE_PILOT,
        topic="Wrapped-phase variance explodes near ±π",
        status="test",
        test="tests/test_metrics.py",
        lesson_ref="2026-04-17 — Wrapped-phase variance explodes",
    ),
    BugEntry(
        bug_id="B-005", date="2026-04-17",
        phase=Phase.PRE_PILOT,
        topic="Greedy walker + wide range = local-max trap",
        status="test",
        test="tests/test_cancel_walker.py",
        lesson_ref="2026-04-17 — Greedy walker + wide range",
    ),
    BugEntry(
        bug_id="B-006", date="2026-04-17",
        phase=Phase.PRE_PILOT,
        topic="Async destroyed signal killed replacement window "
              "(Qt 3D viewer)",
        status="lesson_only",
        lesson_ref="2026-04-17 — Async destroyed signal killed replacement",
    ),

    # ── Phase: UX_PATCH (v1.0.1-ux PySide6 6.x quirks) ──────────────────
    BugEntry(
        bug_id="B-007", date="2026-04-21",
        phase=Phase.UX_PATCH,
        topic="QShortcut moved to QtGui in PySide6 6.x; unit tests "
              "didn't catch import",
        status="lesson_only",
        lesson_ref="2026-04-21 — QShortcut moved to QtGui",
    ),
    BugEntry(
        bug_id="B-008", date="2026-04-21",
        phase=Phase.UX_PATCH,
        topic="np.bool_ ≠ bool in `is` checks (mask handling)",
        status="lesson_only",
        lesson_ref="2026-04-21 — np.bool_ ≠ bool in is checks",
    ),

    # ── Phase: DPG_PORT (v2.0.0 → v2.0.6 Dear PyGui frontend) ───────────
    BugEntry(
        bug_id="B-009", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="DPG set_viewport_drop_callback silently dead on macOS — "
              "drop zone capability detect + click fallback",
        status="manual",
        lesson_ref="2026-04-24 — DPG set_viewport_drop_callback dead",
    ),
    BugEntry(
        bug_id="B-010", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="DPG viewport sizing fixed default ≠ each screen — "
              "tier_for_size two-axis logic",
        status="test",
        test="tests/test_ui2_v207.py",
        lesson_ref="2026-04-24 — DPG viewport sizing sabit",
    ),
    BugEntry(
        bug_id="B-011", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="Info text append builds up stale state across runs",
        status="manual",
        test="",
        lesson_ref="2026-04-24 — Info text append birikince stale | RETIRED-UI2 NOTE (2026-07-06): pinning test removed with the Dear PyGui frontend retirement; the fixed DPG code itself was deleted, so this bug can no longer regress. Kept for history.",
    ),
    BugEntry(
        bug_id="B-012", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="v1→v2 frontend port skipped scientific param audit "
              "(magnification, n_sample, n_medium, AF metric)",
        status="test",
        test="tests/test_ui2_scientific_params.py",
        lesson_ref="2026-04-24 — v1→v2 frontend portunda bilimsel",
    ),
    BugEntry(
        bug_id="B-013", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="Eager __init__.py import broke headless test runner",
        status="lesson_only",
        lesson_ref="2026-04-24 — Paket __init__.py'ın eager import'u",
    ),
    BugEntry(
        bug_id="B-014", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="cancel_check optional pattern for cooperative scan abort",
        status="test",
        test="tests/test_cancel_walker.py",
        lesson_ref="2026-04-24 — Core scan fonksiyonuna cancel_check",
    ),
    BugEntry(
        bug_id="B-015", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="find_focus_candidates top candidate 2-3 step deviation "
              "(Fresnel envelope bias)",
        status="test",
        test="tests/test_focus_validation.py::"
             "test_multi_focus_finds_both_peaks_for_two_spheres_at_different_z",
        lesson_ref="2026-04-24 — find_focus_candidates top candidate",
    ),
    BugEntry(
        bug_id="B-016", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="HDF5 group name collision + non-string metadata in "
              "tomography bundle",
        status="test",
        test="tests/test_batch_bundle.py",
        lesson_ref="2026-04-24 — HDF5 group name collision",
    ),
    BugEntry(
        bug_id="B-017", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="Daemon thread stop latency budgets (Esc-to-cancel)",
        status="test",
        test="tests/test_cancel_walker.py",
        lesson_ref="2026-04-24 — Daemon thread stop latency",
    ),
    BugEntry(
        bug_id="B-018", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="DPG mvClickedHandler doesn't accept all widget types "
              "(e.g. child_window) — global mouse_click + hover gate",
        status="manual",
        lesson_ref="2026-04-24 — mvClickedHandler Dear PyGui'de",
    ),
    BugEntry(
        bug_id="B-019", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="Sibling venv ladder (capability probe in launcher)",
        status="manual",
        lesson_ref="2026-04-24 — Sibling venv ladder",
    ),
    BugEntry(
        bug_id="B-020", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="viewport_menu_bar() macOS'ta uygulama içinde görünmüyor — "
              "dpg.window(menubar=True) + inner menu_bar",
        status="manual",
        lesson_ref="2026-04-24 — viewport_menu_bar() macOS'ta",
    ),
    BugEntry(
        bug_id="B-021", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="DPG file_dialog macOS'ta sessizce başarısız — "
              "tkinter / osascript native picker",
        status="manual",
        lesson_ref="2026-04-24 — Dear PyGui file_dialog macOS",
    ),
    BugEntry(
        bug_id="B-022", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="_prepare_field n=1.0 hardcoded — n_medium hiç "
              "propagate'e ulaşmıyordu",
        status="test",
        test="tests/test_ui2_v207.py",
        lesson_ref="2026-04-24 — v1→v2 port audit: _prepare_field n=1.0",
    ),
    BugEntry(
        bug_id="B-023", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="Adaptive autofocus algorithms lost in v1→v2 port — "
              "adaptive_distance/gradient/bracketing geri eklendi",
        status="test",
        test="tests/test_autofocus_worker.py",
        lesson_ref="2026-04-24 — Adaptive autofocus algoritmaları",
    ),
    BugEntry(
        bug_id="B-024", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="compute_depth_map amplitude-only kernel'leri saf-faz "
              "numune için ters çalışıyordu",
        status="test",
        test="tests/test_focus_validation.py::"
             "test_depth_map_localises_single_sphere_to_its_true_z",
        lesson_ref="2026-04-24 — compute_depth_map amplitude-only",
    ),
    BugEntry(
        bug_id="B-025", date="2026-04-24",
        phase=Phase.DPG_PORT,
        topic="Sentetik validasyon testi fiziksel limitlerin "
              "dışında diameter ölçemez",
        status="test",
        test="tests/test_focus_validation.py::"
             "test_end_to_end_autofocus_then_measure_size",
        lesson_ref="2026-04-24 — Sentetik validasyon testi fiziksel",
    ),

    # ── Phase: PILOT_PATCHES (5-bug acil + end-to-end + multi-focus) ────
    BugEntry(
        bug_id="B-026", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="Bug #1: hologram 180° flipped on load — flip_display_v/h "
              "+ View menu toggles",
        status="manual",
        test="",
        lesson_ref="(5-bug sprint) | RETIRED-UI2 NOTE (2026-07-06): pinning test removed with the Dear PyGui frontend retirement; the fixed DPG code itself was deleted, so this bug can no longer regress. Kept for history.",
    ),
    BugEntry(
        bug_id="B-027", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="Bug #2: reference subtract not working (only in Reconstruct "
              "path, not autofocus/multi-focus/QPI/depth) — shared "
              "_preprocess_raw + _extract_field_with_reference helpers",
        status="test",
        test="tests/test_ui2_v207.py",
        lesson_ref="(5-bug sprint)",
    ),
    BugEntry(
        bug_id="B-028", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="Bug #3: first load input panel empty — _push_texture "
              "literal 'tex_input' vs panel_input.tex_tag UUID",
        status="manual",
        test="",
        lesson_ref="(5-bug sprint) | RETIRED-UI2 NOTE (2026-07-06): pinning test removed with the Dear PyGui frontend retirement; the fixed DPG code itself was deleted, so this bug can no longer regress. Kept for history.",
    ),
    BugEntry(
        bug_id="B-029", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="Bug #4: autofocus 'önceden 5 sn'de buluyordu' — bench "
              "showed core fast (2048² = 4.3 sec, matches), FFT cache "
              "intact via _make_fast_evaluator",
        status="test",
        test="tests/test_autofocus_speed_baseline.py",
        lesson_ref="2026-04-24 — Autofocus yavaşladı raporu",
    ),
    BugEntry(
        bug_id="B-030", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="Bug #5: scroll required at 1440×800 — _tier_for_size "
              "two-axis (was width-only), viewport cap 1150",
        status="test",
        test="tests/test_ui2_v207.py",
        lesson_ref="(5-bug sprint)",
    ),
    BugEntry(
        bug_id="B-031", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="NameError raw_disp — flip block must run BEFORE _push_texture "
              "(edit-order mistake)",
        status="manual",
        test="",
        lesson_ref="(5-bug sprint) | RETIRED-UI2 NOTE (2026-07-06): pinning test removed with the Dear PyGui frontend retirement; the fixed DPG code itself was deleted, so this bug can no longer regress. Kept for history.",
    ),
    BugEntry(
        bug_id="B-032", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="End-to-end test eksikliği: lateral + depth correction "
              "autofocus z'sinde değil truth z'sinde ölçülüyordu",
        status="test",
        test="tests/test_focus_validation.py::"
             "test_end_to_end_two_objects_via_multifocus",
        lesson_ref="2026-04-24 — Single-sphere lateral diameter not "
                   "measurable",
    ),
    BugEntry(
        bug_id="B-033", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="Single-sphere synthetic lateral diameter not measurable "
              "post-reconstruction (carrier residual + Δn=0.07 wraps)",
        status="lesson_only",
        lesson_ref="2026-04-24 — Single-sphere lateral diameter not "
                   "measurable",
    ),
    BugEntry(
        bug_id="B-034", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="spectrum.copy() perf prediction was ~4200× off — never "
              "claim a percentage without a measurement",
        status="lesson_only",
        lesson_ref="2026-04-24 — Tahmin yapma, ölç",
    ),
    BugEntry(
        bug_id="B-035", date="2026-04-24",
        phase=Phase.PILOT_PATCHES,
        topic="find_focus_candidates was on propagate() loop instead of "
              "_make_fast_evaluator — multi-focus + 6 autofocus algos "
              "now share one path",
        status="test",
        test="tests/test_autofocus_speed_baseline.py::"
             "test_multifocus_find_candidates_stays_under_ceiling",
        lesson_ref="(multi-focus refactor)",
    ),

    # ── Phase: TIMELAPSE_FOUNDATION (v2.0.7) ────────────────────────────
    BugEntry(
        bug_id="B-036", date="2026-04-27",
        phase=Phase.TIMELAPSE_FOUNDATION,
        topic="Schema v10→v11: Ui2State.user_preset_archive — preset "
              "edit/replace flow, archive previous version (cap 10) "
              "before clobber",
        status="test",
        test="tests/test_ui2_user_presets.py",
        lesson_ref="(v2.0.7 T4 preset edit)",
    ),
    BugEntry(
        bug_id="B-037", date="2026-04-27",
        phase=Phase.TIMELAPSE_FOUNDATION,
        topic="Session model + headless CLI runner + per-frame CSV "
              "+ multi-user profile + audit viewer + batch resume "
              "(v2.0.7 T0/T1/T2/T3/T5/T6)",
        status="test",
        test="tests/test_session.py",
        lesson_ref="(v2.0.7 omurga)",
    ),
    BugEntry(
        bug_id="B-038", date="2026-04-28",
        phase=Phase.TIMELAPSE_FOUNDATION,
        topic="Schema v11→v12 cross-team migration: AI ekibi Qt-side "
              "_migrate_v11_to_v12 ekledi ama JSON-side eksikti — "
              "6 bug registry test'i fail oldu, defensive __new__ "
              "test pattern'i kırıldı",
        status="lesson_only",
        lesson_ref="2026-04-28 — Paralel ekipten gelen schema",
    ),

    # ── Phase: TRACKING (v2.0.8) ────────────────────────────────────────
    BugEntry(
        bug_id="B-039", date="2026-04-27",
        phase=Phase.TRACKING,
        topic="Drift correction + per-cell tracking (Hungarian) "
              "+ NIST bead calibration + multi-line profile",
        status="test",
        test="tests/test_registration.py",
        lesson_ref="(v2.0.8 D1-D4)",
    ),

    # ── Phase: PAPER_READY (v2.0.9) ─────────────────────────────────────
    BugEntry(
        bug_id="B-040", date="2026-04-27",
        phase=Phase.PAPER_READY,
        topic="Vector PDF report + Zenodo bundle + WCAG-AA contrast "
              "audit + crash handler v2 wiring",
        status="test",
        test="tests/test_v209_paper_ready.py",
        lesson_ref="(v2.0.9 P1-P4)",
    ),

    # ── Phase: PERF_GPU (v2.1.0) ────────────────────────────────────────
    BugEntry(
        bug_id="B-041", date="2026-04-28",
        phase=Phase.PERF_GPU,
        topic="Torch FFT backend (lazy, optional) + batched FFT "
              "in evaluator + ROI fast-path + Linux Docker / CI matrix",
        status="test",
        test="tests/test_v210_perf.py",
        lesson_ref="(v2.1.0 G1-G5)",
    ),

    # ── Phase: HARDWARE (v2.1.x) ────────────────────────────────────────
    BugEntry(
        bug_id="B-042", date="2026-04-28",
        phase=Phase.HARDWARE,
        topic="Camera registry (synthetic/mock + Pylon/IDS/Thorlabs "
              "stubs) + MP4 recorder (imageio-ffmpeg) + time-lapse "
              "runner + live/file mode UI ayrımı + lab device "
              "control (core.devices)",
        status="test",
        test="tests/test_v21x_hardware.py",
        lesson_ref="(v2.1.x H1-H6 + v2.1.z devices)",
    ),
    BugEntry(
        bug_id="B-043", date="2026-04-29",
        phase=Phase.HARDWARE,
        topic="Multi-position time-lapse bridge — TimelapseRunner + "
              "orchestrator.run_plan composed for 24-position 12-hour "
              "Karin-pattern session",
        status="test",
        test="tests/test_multi_position_timelapse.py",
        lesson_ref="(v2.1.z follow-up)",
    ),

    # ── AI integration audit (cross-cutting; surface PILOT_PATCHES
    # because that's the active fix sprint phase) ──────────────────────
    BugEntry(
        bug_id="B-044", date="2026-04-29",
        phase=Phase.PILOT_PATCHES,
        topic="AI client.py — non-lazy `import requests` despite "
              "comment claiming lazy; AIPanel.__init__ hard-failed "
              "without the dep. Fixed via _ensure_session() first-use "
              "bootstrap + LLMClientError on missing dep.",
        # Pointed at the panel suite — its tests construct
        # AIPanel/LocalLLMClient and would have caught the eager
        # import failure. test_ai_client.py module-level skips when
        # ``requests`` is absent; the panel suite proves the bigger
        # claim ("panel still constructs even without requests").
        status="test",
        test="tests/test_ai_panel.py",
        lesson_ref="2026-04-29 — Lazy import yorumu ≠ lazy davranış",
    ),
    BugEntry(
        bug_id="B-045", date="2026-04-29",
        phase=Phase.PILOT_PATCHES,
        topic="AIPanel._confirm — QTimer.singleShot post on worker "
              "thread (no event loop) → confirmation dialogs never "
              "fired, 120s timeout silently expired, irreversible "
              "tool gate dead code. Fixed via QMetaObject.invokeMethod + Slot.",
        status="test",
        test="tests/test_ai_panel.py",
        lesson_ref="(AI audit 2026-04-29)",
    ),
    BugEntry(
        bug_id="B-046", date="2026-04-29",
        phase=Phase.PILOT_PATCHES,
        topic="record_timelapse not cancellation-aware — sleep loop "
              "iddia ediyordu ama agent cancel hook scope dışıydı. "
              "ToolContext.is_cancelled callable + frame & sleep-slice "
              "poll eklendi.",
        status="test",
        test="tests/test_ai_device_tools.py::test_record_timelapse_honours_cancel_hook",
        lesson_ref="(AI audit 2026-04-29)",
    ),
    BugEntry(
        bug_id="B-047", date="2026-04-29",
        phase=Phase.PILOT_PATCHES,
        topic="stage_focus_search clamp bound mismatch — "
              "search_range_mm (window width) z_mm (-200..200) "
              "axis bound'u kullanıyordu. NUMERIC_BOUNDS'a doğru "
              "key'ler eklendi, step_mm de clamp ediliyor.",
        status="test",
        test="tests/test_ai_tools.py",
        lesson_ref="(AI audit 2026-04-29)",
    ),
    BugEntry(
        bug_id="B-048", date="2026-04-29",
        phase=Phase.PILOT_PATCHES,
        topic="AI agent APT-uyumlu device tools — 9 yeni tool "
              "(list_devices, shutter_*, led_*, acquire_grid) "
              "core.devices Protocol'üne bağlı, ToolContext'e "
              "is_cancelled/shutter/led/orchestrator hooks eklendi",
        status="test",
        test="tests/test_ai_device_tools.py",
        lesson_ref="(v2.1.z AI device tools)",
    ),
    BugEntry(
        bug_id="B-049", date="2026-04-28",
        phase=Phase.PILOT_PATCHES,
        topic="AI ekibi sample_maps hardcoded path + "
              "_gui_capture_and_process race fix'lerini paralel "
              "commit'le düzeltmiş; audit raporu zaten kapalı "
              "şikayetleri gösteriyordu — bug registry hygiene "
              "gözlem entry'si.",
        status="lesson_only",
        lesson_ref="(AI audit follow-up 2026-04-29)",
    ),
    BugEntry(
        bug_id="B-050", date="2026-04-29",
        phase=Phase.PILOT_PATCHES,
        topic="batch_renderer subtract_mean default'u False'ken "
              "v1 process_tab + v2 ReconParams default'u True; "
              "profilde key yoksa batch DC bias'ı çıkarmadan "
              "demodüle ediyor → ~%50 amplitude scale farkı. "
              "Phase invariant kalsa da metric/dry-mass yanlış. "
              "Fix: default True'ya çekildi + parite testi.",
        status="test",
        test="tests/test_batch_v2_parity.py",
        lesson_ref="(2026-04-29 batch correctness audit)",
    ),
    BugEntry(
        bug_id="B-051", date="2026-04-29",
        phase=Phase.PILOT_PATCHES,
        topic="batch_renderer autofocus/sweep search'leri "
              "reference DIVISION'ı uygulamadan propagate ediyor, "
              "best Z bulununca _apply_ref ile referenced field "
              "kaydediliyor → metric un-referenced field üstünde "
              "optimum buluyor. Fix: v1 main_window _reference_fc'i "
              "cfg'ye geçiyor, batch _apply_ref(z_m=...) ile "
              "referansı hedef z'ye yeniden yayınlıyor + üç "
              "algoritma branch'ı ref_field forward ediyor.",
        status="test",
        test="tests/test_batch_v2_parity.py",
        lesson_ref="(2026-04-29 batch correctness audit)",
    ),
    BugEntry(
        bug_id="B-052", date="2026-04-30",
        phase=Phase.PILOT_PATCHES,
        topic="batch_renderer reference auto-pair — "
              "lab konvansiyonu '<stem>_ref.<ext>' / 'ref_<stem>.<ext>' "
              "(prefix VEYA suffix, case-insensitive, _ veya - separator) "
              "ile her sample'ın paired ref'i sibling olarak dururken "
              "kullanıcı manuel olarak yüklemek zorundaydı. Fix: "
              "_find_reference_for + _is_reference_filename helper'ları, "
              "setup() ref dosyalarını iteration'dan filtreler, "
              "_process_single_job paired ref'i otomatik demodüle eder "
              "ve ref_fc olarak _apply_ref'e geçer. v1 explicit reference "
              "varsa o öncelikli; yoksa auto-pair kicks in.",
        status="test",
        test="tests/test_batch_ref_autopair.py",
        lesson_ref="(2026-04-30 reference auto-pair feature)",
    ),
    BugEntry(
        bug_id="B-054", date="2026-05-04",
        phase=Phase.PILOT_PATCHES,
        topic="UX katmanı: scalebar overlay (faz/amp panel µm cinsinden), "
              "skip-existing batch flag, 3D surface viewer (depth+phase). "
              "core/scalebar.py 1-2-5 ladder + decade boundary log-space "
              "snap; v1 ImagePanel + PhasePanel + v2 ZoomableImagePanel "
              "set_scalebar(pixel_um). batch_renderer _job_already_finished "
              "predicate, BatchRenderDialog 'Skip files...' tik. Yeni "
              "gui/widgets/depth_surface_viewer.py (pyqtgraph.opengl), "
              "v2 menu 'Open 3D surface (depth/phase)'. Negatif faz "
              "asla clip edilmez (feedback memory).",
        status="test",
        test="tests/test_scalebar.py",
        lesson_ref="(2026-05-04 lab UX features)",
    ),
    BugEntry(
        bug_id="B-053", date="2026-04-30",
        phase=Phase.PILOT_PATCHES,
        topic="depth map ref-blind + n hardcoded → \"sıçtı batırdı\". "
              "compute_depth_map ref_field parametresi kabul etmiyordu, "
              "v1 _on_compute_depth_map_triggered _reference_fc'i forward "
              "etmiyordu, v2 ui2/workers run_depth_map _prepare_field "
              "kullanıyordu (pre-propagation division — propagation ile "
              "commute etmediği için yanlış). _prepare_af_field n=1.0 "
              "hardcoded, qpi_tab.n_medium okunmuyordu → su ortamında "
              "depth z %25 yanlış. Fix: compute_depth_map ref_field "
              "kwarg, her z'de hem sample hem ref propagate edilip "
              "bölünüyor. v1 3 callsite + v2 worker forward. "
              "n_medium qpi_tab'tan okunuyor. Boundary saturation "
              ">%50 olduğunda WARNING log'a düşüyor.",
        status="test",
        test="tests/test_depth_map_ref_aware.py",
        lesson_ref="(2026-04-30 depth map correctness audit)",
    ),
    BugEntry(
        bug_id="B-055", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="Batch _do_save used Path.with_suffix on z-decimal base names — "
              "'.5000mm' stripped as extension, same-integer-z sweep slices "
              "collapsed onto one overwriting file (2026-07-05 review).",
        status="test",
        test="tests/test_batch_output_naming.py::test_same_integer_z_values_do_not_collide",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-056", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="Batch auto-pair setup filter silently dropped ref-named samples "
              "(reflow_01.tif) from the job list even with an explicit "
              "reference supplied; now exempt + announced via status "
              "(2026-07-05 review).",
        status="test",
        test="tests/test_batch_output_naming.py::test_explicit_reference_disables_setup_ref_filter",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-057", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="ui2 scalebar fed per-SOURCE-pixel size with the 512-px display "
              "width — bar length/label wrong by src_width/512 in every report; "
              "_display_pixel_um now rescales (2026-07-05 review).",
        status="manual",
        test="",
        lesson_ref="(2026-07-05 code review) | RETIRED-UI2 NOTE (2026-07-06): pinning test removed with the Dear PyGui frontend retirement; the fixed DPG code itself was deleted, so this bug can no longer regress. Kept for history.",
    ),
    BugEntry(
        bug_id="B-058", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="CachedReconstructor spectrum cache keyed on bare id(field) — id "
              "reuse after GC served the previous frame's spectrum for a new "
              "same-shape field (wrong reconstruction in free+realloc loops); "
              "now weakref-identity (2026-07-05, exposed by textbook validation "
              "suite).",
        status="test",
        test="tests/test_reconstruction.py::test_spectrum_cache_survives_id_reuse",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-059", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="v1 depth/tomography/autofocus paths read _reference_fc "
              "unconditionally while reconstruct gated on the 'Enable reference "
              "subtraction' checkbox — unchecking changed the display but depth "
              "kept dividing by the reference; all paths now route through "
              "_active_reference_fc() (2026-07-05 review).",
        status="test",
        test="tests/test_depth_map_ref_aware.py::test_v1_depth_callsites_forward_reference",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-060", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="DevicePanelController.jog unpacked exactly 3 axes inside a bare "
              "except — a 2-axis stage made every jog a silent no-op while the "
              "HUD kept updating; now pads missing axes (2026-07-05 review).",
        status="manual",
        test="",
        lesson_ref="(2026-07-05 code review) | RETIRED-UI2 NOTE (2026-07-06): pinning test removed with the Dear PyGui frontend retirement; the fixed DPG code itself was deleted, so this bug can no longer regress. Kept for history.",
    ),
    BugEntry(
        bug_id="B-061", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="CombinedLoss built disabled terms as CPU torch.tensor(0.0) and "
              "added them to device-resident l1 — MPS/CUDA training crashed on "
              "every forward (charbonnier_weight defaults 0.0); now "
              "pred.new_zeros (2026-07-05 review). Automated test exists "
              "(tests/test_track_c_pipeline.py::"
              "test_combined_loss_disabled_terms_follow_pred_device) but is "
              "torch-gated — neither local venv ships torch, so the runner "
              "can't execute it here; CI-Linux runs it. Marked manual for "
              "local sweeps: retrain smoke on an MPS box must not crash.",
        status="manual",
        test="",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-062", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="v1 live reconstruct hardcoded n_medium=1.0 while autofocus/depth "
              "read the qpi tab (default 1.337) — best-focus z and the "
              "displayed reconstruction diverged by a factor of n; "
              "_build_recon_job now reads the same qpi-tab source. Verify "
              "in-app: set n_medium, run reconstruct + autofocus, z scales "
              "consistently.",
        status="manual",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-063", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="ui2 opened the Qt DepthSurfaceViewer under DearPyGui without any "
              "Qt event loop — window froze (no paint/interaction); render loop "
              "now pumps QApplication.processEvents once the viewer exists. "
              "Verify in-app: open 3D surface from v2, rotate/resize the "
              "window.",
        status="manual",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-064", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="qpi.compute_cell_morphology computed phase as opd_m*(2π) — "
              "dropped the /λ, so CellMorphology.mean_phase_rad/phase_std_rad "
              "were ~1e6× too small and dimensionally wrong (also written to "
              "CSV). Fixed to φ=2π·OPD/λ with a wavelength_m param (2026-07-05 "
              "review, physics_verify-confirmed relation).",
        status="test",
        test="tests/test_qpi.py::test_cell_morphology_phase_stats_have_correct_scale",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-065", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="Layering inversion + reffree duplication: "
              "src/recon_dl/inference.py imported the "
              "demod/autofocus/propagate/unwrap chain from "
              "scripts/benchmark_reffree (shipped code depending on a lab "
              "script) and the reference-division idiom existed in 3-4 copies. "
              "Extracted to core.pipelines.reffree_hybrid with an "
              "OpticalConfig; scripts are thin wrappers, inference takes a "
              "config (2026-07-05 review).",
        status="test",
        test="tests/test_reffree_pipeline.py::test_src_does_not_import_scripts",
        lesson_ref="(2026-07-05 code review)",
    ),
    BugEntry(
        bug_id="B-066", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="dhm-mcp FastMCP bridge registered every tool as a bare **kwargs "
              "handler — FastMCP derives inputSchema from the signature, so all "
              "~40 bridged tools were uncallable over the real protocol (only "
              "'about' worked). Fixed by synthesizing an explicit keyword-only "
              "signature from each ToolSpec JSON schema (2026-07-05 review, "
              "CRITICAL; live-verified against the mcp SDK). Automated test "
              "exists (tests/test_mcp_bridge.py, 3 tests) but is mcp-gated — "
              "the runner venv doesn't ship mcp, so it can't execute here; "
              "verified live with system python3 (mcp 1.x): 3 passed. Marked "
              "manual for local sweeps.",
        status="manual",
        test="",
        lesson_ref="(2026-07-05 AI/MCP review)",
    ),
    BugEntry(
        bug_id="B-067", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="ui2 reference-mode combo and legacy subtract_reference checkbox "
              "could disagree: an explicit 'Off' pick left the legacy flag "
              "armed, effective_reference_mode() re-resolved to 'reference' and "
              "the divide kept happening against the user's choice; now two-way "
              "synced across combo/toggle/load/clear (2026-07-05 review, HIGH). "
              "Also: QPI paths now apply the reference-free background fit "
              "(MED), audit records log the RESOLVED mode (LOW), hydrate drops "
              "un-honourable persisted CNN requests (LOW).",
        status="manual",
        test="",
        lesson_ref="(2026-07-05 ui2 review) | RETIRED-UI2 NOTE (2026-07-06): pinning test removed with the Dear PyGui frontend retirement; the fixed DPG code itself was deleted, so this bug can no longer regress. Kept for history.",
    ),
    BugEntry(
        bug_id="B-068", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AIPanel.__init__ did a synchronous health_check "
              "(requests.get 2s) on the GUI thread in the constructor; repeated "
              "construction raced urllib3 teardown into a segfault. Fixed: no "
              "auto network on build, threaded probe on user action (2026-07-05 "
              "ui3 review CRITICAL).",
        status="test",
        test="tests/test_ui3_ai_panel.py::test_ai_panel_builds_neutral_then_reports_unavailable",
        lesson_ref="(2026-07-05 ui3 review)",
    ),
    BugEntry(
        bug_id="B-069", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 MainWindow.closeEvent only shut down the bridge; Qt doesn't "
              "deliver closeEvent to a dock's content widget, so "
              "camera/ai/timelapse/device panel threads/timers were abandoned "
              "on window close (zombie thread, possibly-invalid TIFF). Fixed: "
              "_shutdown_panels() explicitly stops each panel (shutdown() hook "
              "+ close()) (2026-07-05 ui3 review HIGH).",
        status="test",
        test="tests/test_ui3_spine.py::test_close_shuts_down_panels_without_error",
        lesson_ref="(2026-07-05 ui3 review)",
    ),
    BugEntry(
        bug_id="B-070", date="2026-07-05",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 MainWindow._on_autofocus_done read a nonexistent best_z_mm "
              "(AutofocusResult only has best_z_m, metres); guard never fired "
              "so the app silently reconstructed at the stale z after "
              "autofocus. Fixed: best_z_m*1e3 (2026-07-05 ui3 review, confirmed "
              "bug).",
        status="test",
        test="tests/test_ui3_spine.py::test_autofocus_done_converts_best_z_m_to_mm",
        lesson_ref="(2026-07-05 ui3 review)",
    ),
    BugEntry(
        bug_id="B-071", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 app.py lacked the macOS Sonoma+ paint-engine workaround "
              "(QT_WIDGETS_RHI=0) that v1 main.py has — on the real Cocoa "
              "platform PySide6 6.10 segfaulted on first widget paint (QPainter "
              "engine==0, no Python crash dump), showing an error/blank screen; "
              "offscreen tests never hit it. Fixed: set QT_WIDGETS_RHI=0 + "
              "AA_ShareOpenGLContexts before QApplication; verified "
              "real-platform render (2026-07-06).",
        status="test",
        test="tests/test_ui3_spine.py::test_app_sets_macos_paint_engine_workaround",
        lesson_ref="(2026-07-06 ui3 real-screen)",
    ),
    BugEntry(
        bug_id="B-072", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 MainWindow._on_load_reference called "
              "self._cb_ref_mode.setCurrentText — but that inline combo was "
              "deleted when the dock became ReconPanel, so every reference "
              "load raised AttributeError. Fixed: set params.reference_mode + "
              "subtract_reference and re-hydrate ReconPanel via "
              "_sync_controls_from_params (2026-07-06 ui3 review, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_load_reference_does_not_crash_and_arms_reference",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-073", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AIPanel bridged tools called signal.disconnect() with no "
              "argument in their finally blocks — that severs EVERY slot on "
              "recon_done/autofocus_done/qpi_done/depth_done/focus_candidates_"
              "done, including MainWindow's viewport paint and the rich panels. "
              "After the copilot ran one op the whole app stopped reacting. "
              "Fixed: capture the connect() handle and _disconnect_one() detaches "
              "only that connection (2026-07-06 ui3 review, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_disconnect_one_only_removes_its_own_connection",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-074", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="core.observe.render_view drew the scalebar with the "
              "pre-downsample pixel size but the post-downsample frame width, "
              "so the µm label was wrong by the downsample stride on any image "
              "larger than max_size. Fixed: _downsample returns its stride and "
              "the scalebar uses pixel_size_um*stride (2026-07-06 review, "
              "confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_render_view_scalebar_uses_effective_pixel_size",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-075", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 state._params_to_dict serialised non-scalar ReconParams "
              "fields via str(val), so af_roi (a tuple) persisted as the literal "
              "string '(0.1, 0.2, ...)' and reloaded corrupt. Fixed: store "
              "tuples/lists of scalars as JSON lists, skip ndarray/Path, and "
              "coerce tuple fields back on load (2026-07-06 review, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_af_roi_round_trips_as_tuple",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-076", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 WorkerBridge.busy_changed was a single boolean shared across "
              "two independent executors (recon + science). Two overlapping ops "
              "(e.g. reconstruct + depth map) fought over it — whichever "
              "finished first cleared the busy indicator while the other still "
              "ran. Fixed: reference-count in-flight ops under a lock (_end is "
              "called on a worker thread), signal idle only at zero (2026-07-06 "
              "review, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_busy_changed_is_reference_counted",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-077", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AIPanel._gui_set_reference_mode set reference_mode but left "
              "the legacy subtract_reference flag untouched. effective_reference_"
              "mode() treats mode=='off' + subtract_reference=True as "
              "'reference', so a stale True from an earlier reference load "
              "silently overrode an explicit 'off'/'reference_free' from the "
              "copilot. Fixed: subtract_reference = (mode=='reference') "
              "(2026-07-06 review, confirmed; same class as B-067).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_ai_set_reference_mode_keeps_legacy_flag_in_lockstep",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-078", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="dhm_mcp HeadlessSession.set_recon_param updated recon_params but "
              "did not invalidate the cached derived fields "
              "(demod/recon/phase/depth). A subsequent inspect_* over MCP then "
              "reported the OLD reconstruction as if it reflected the new "
              "params. Fixed: clear the derived caches on param change, same "
              "contract as load_hologram (2026-07-06 review, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_headless_set_recon_param_invalidates_derived_cache",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-079", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="dhm_mcp reference mode was unreachable over MCP: reference_raw "
              "was never populated (no load path), so set_reference_mode('"
              "reference') always failed 'load a reference first'. Fixed: "
              "set_reconstruction_mode accepts an optional reference_path "
              "(validated by validate_path, hologram extensions, restrict_to_"
              "home) that headless loads into reference_raw; ui3 AIPanel honours "
              "it too (2026-07-06 review, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_headless_reference_path_populates_reference_raw",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-080", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="dhm_mcp reference-free path fed subtract_background's two "
              "independent knobs (n_terms for zernike, polynomial_order for "
              "polynomial) from one bg_order key with a shared default — so "
              "polynomial mode with the default bg_order=15 ran a degenerate "
              "order-15 fit. Fixed: route bg_order to the active method's knob "
              "and give the other its own default (2026-07-06 review, "
              "confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_headless_bg_order_defaults_are_method_specific",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-081", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AIPanel._refresh_health overwrote self._health_thread while a "
              "prior probe was still running (health_check blocks up to 2s in "
              "requests.get), orphaning a live QThread parented to the panel — "
              "destroyed with the widget graph on close → 'QThread destroyed "
              "while running' crash; it also blocked the GUI thread on wait(50). "
              "Fixed: track every in-flight probe in a set, disconnect (not "
              "orphan) a superseded one, clean up on finished, and shutdown() "
              "waits on all (2026-07-06 review, confirmed).",
        status="manual",
        test="",
        lesson_ref="(2026-07-06 ui3 review; QThread lifecycle race, verified by "
                   "reasoning + tests/test_ui3_spine.py::test_close_shuts_down_"
                   "panels_without_error)",
    ),
    BugEntry(
        bug_id="B-082", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 double-wiring: MainWindow AND the rich panels both connected "
              "the same bridge result signals. autofocus_done triggered a second "
              "full reconstruct from the shell (double compute per autofocus); "
              "depth_done repainted the shared 'spectrum' viewport twice. Fixed: "
              "the shell drops the auto-reconstruct and only caches depth for "
              "get_field('depth'); DepthPanel/FocusPanel own the paint/readout "
              "(2026-07-06 review, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_mainwindow_depth_done_caches_without_painting",
        lesson_ref="(2026-07-06 ui3 review)",
    ),
    BugEntry(
        bug_id="B-083", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 qpi/candidates leg of the double-wiring (B-082 fixed only "
              "autofocus/depth): qpi_done status written by both QPIPanel and "
              "MainWindow; qpi_batch_done/focus_candidates_done populated the "
              "dialog table twice (dialog's own connection + shell open_with) "
              "and wrote the status 2-3x (panel + dialog + shell). Fixed with "
              "an explicit ownership contract: panels own their domain status "
              "(one-shot QPI now carries dry mass), dialogs own batch/"
              "candidates status + table, the shell only present()s dialogs "
              "and keeps recon paint / z-sync / caches; shell error lambdas "
              "removed (owners exist in panels/dialogs) (2026-07-06 review, "
              "confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_qpi_batch_dialog_populates_once_and_is_presented",
        lesson_ref="(2026-07-06 ui3 review; see also B-082)",
    ),
    BugEntry(
        bug_id="B-084", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="3 crash_handler regression tests false-FAILED since pytest-qt "
              "entered the env with the ui3/PySide6 work: the handler "
              "correctly CHAINS to the previously installed excepthook, which "
              "under the runner is pytest-qt's exception capture (sys hook) / "
              "pytest's threadexception capture (threading hook) — the chained "
              "call was reported as 'Exceptions caught in Qt event loop' and "
              "failed the test even though every assert passed. Product code "
              "correct; fixed in the tests by pinning a benign no-op prior "
              "hook before install (chaining stays covered by "
              "test_install_chains_to_previous_hook). Masked earlier by the "
              "moved-venv stale-__pycache__ artefact that showed phantom "
              "Windsurf paths (venv rebuilt same day, caches purged).",
        status="test",
        test="tests/test_crash_handler.py::test_install_writes_dump_for_main_thread",
        lesson_ref="(2026-07-06 venv rebuild + pytest-qt interaction)",
    ),
    BugEntry(
        bug_id="B-085", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="`from fixtures...` in 6 test modules was collection-ORDER "
              "dependent: tests/ only landed on sys.path when an "
              "alphabetically-earlier module (test_calibration) inserted it. "
              "test_autofocus_speed_baseline collects BEFORE test_calibration "
              "→ permanent collection ERROR (aborted the whole run; was being "
              "--ignore'd); test_depth_map/qpi_batch/cluster_heights/stress_"
              "holograms failed standalone. Fixed: each fixtures importer "
              "inserts tests/ itself (repo's existing test_calibration "
              "pattern) — all six now pass in one run without "
              "test_calibration present (2026-07-06).",
        status="test",
        test="tests/test_autofocus_speed_baseline.py",
        lesson_ref="(2026-07-06 venv rebuild follow-up)",
    ),
    # ---- 2026-07-06 review round #2: adversarial workflow over the
    # ---- round-1 fix diff (4 finder dimensions × 2 refuters each).
    BugEntry(
        bug_id="B-086", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="dhm_mcp: the B-078 cache-invalidation fix was incomplete — "
              "set_reference_mode (mode/bg knobs change what invoke_recon "
              "produces) and invoke_autofocus's direct z write both left "
              "recon_complex/phase_unwrapped/depth cached, so inspect_* served "
              "the OLD reconstruction as current. Fixed: single "
              "_invalidate_derived() helper called from every mutator "
              "(2026-07-06 review #2, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_headless_set_reference_mode_invalidates_derived_cache",
        lesson_ref="(2026-07-06 review #2)",
    ),
    BugEntry(
        bug_id="B-087", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AI snapshot passed the RAW camera pixel_um to the "
              "observation tools while the pipeline works at the effective "
              "pixel (camera/magnification): inspect_phase dry mass inflated "
              "by M² (1600x with a 40x preset), render_view scalebar off by "
              "M. Fixed: _recon_params_dict emits effective_pixel_um() as "
              "pixel_um (2026-07-06 review #2, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_ui3_snapshot_pixel_um_is_effective",
        lesson_ref="(2026-07-06 review #2)",
    ),
    BugEntry(
        bug_id="B-088", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AI bridged tools: the drivers reject overlapping ops "
              "SYNCHRONOUSLY (on_error inline during the bridge call), so the "
              "error was captured before _wait_for_signal started — the "
              "nested event loop then waited the full 60-120s timeout for an "
              "emission that had already happened, stalling the copilot. "
              "Fixed: skip the wait when the result dict is already populated "
              "(all 5 bridged tools) (2026-07-06 review #2, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_ai_tool_does_not_wait_after_synchronous_rejection",
        lesson_ref="(2026-07-06 review #2)",
    ),
    BugEntry(
        bug_id="B-089", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 WorkerBridge: the B-076 refcount emitted busy=False from "
              "the WORKER thread — a queued event that could be delivered "
              "AFTER a newer GUI-thread busy=True (user starts an op before "
              "the stale event drains), clearing the busy UI and re-enabling "
              "buttons mid-op. Fixed: _end only posts an _idle_check signal; "
              "the GUI-thread slot re-reads the counter at delivery time and "
              "drops the stale idle (2026-07-06 review #2, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_stale_idle_from_worker_is_dropped_when_new_op_started",
        lesson_ref="(2026-07-06 review #2)",
    ),
    BugEntry(
        bug_id="B-090", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 MainWindow.set_status auto-toasted ok/warn/error while the "
              "B-083 panel-owned handlers also call ctx.toast — two stacked "
              "identical toasts per event, and 'danger'-level errors were "
              "inconsistently quieter (no auto-toast) than 'error'-level. "
              "Fixed: set_status is status-only; toasts are always explicit "
              "(reffree_note kept its toast via an explicit call) (2026-07-06 "
              "review #2, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_set_status_does_not_auto_toast",
        lesson_ref="(2026-07-06 review #2)",
    ),
    BugEntry(
        bug_id="B-091", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AI set_reconstruction_mode wrote bg_order only to "
              "reffree_bg_order (polynomial's knob) while the pipeline reads "
              "reffree_n_terms for zernike — the AI's bg_order was a silent "
              "no-op in zernike mode and diverged from headless/MCP (which "
              "routes correctly since B-080). Fixed: route by the active "
              "bg_method (2026-07-06 review #2, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_ai_bg_order_routes_to_method_specific_knob",
        lesson_ref="(2026-07-06 review #2; see B-080)",
    ),
    BugEntry(
        bug_id="B-092", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AIPanel._build_tool_context bound self._worker at build "
              "time, but the context is built BEFORE the turn's AIWorker is "
              "constructed — is_cancelled always saw the PREVIOUS worker "
              "(None on first send), so Stop could never cancel a polling "
              "tool. Fixed: the closure reads self._worker dynamically "
              "(2026-07-06 review #2, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_is_cancelled_sees_worker_assigned_after_context_build",
        lesson_ref="(2026-07-06 review #2)",
    ),
    BugEntry(
        bug_id="B-093", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="core.observe.render_view kind='spectrum' skipped the FFT for "
              "complex input (np.abs(fftshift(arr)) on the SPATIAL field) — "
              "rendering quadrant-swapped |field| mislabeled as log|F|. "
              "Complex reconstructions are spatial-domain fields and need "
              "fft2 exactly like real inputs. Fixed: single fft2 path "
              "(2026-07-06 review #2, confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_render_view_spectrum_ffts_complex_fields",
        lesson_ref="(2026-07-06 review #2)",
    ),
    BugEntry(
        bug_id="B-094", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 MainWindow.get_field('phase_unwrapped') silently fell back "
              "to the WRAPPED phase whenever the pipeline hadn't populated "
              "unwrapped_phase (i.e. every off/reference-mode recon — only "
              "the reffree path fills it), corrupting OPD/dry-mass computed "
              "by the AI observation tools. Fixed: unwrap on demand with "
              "core.phase_unwrap.unwrap_phase_advanced (same routine as the "
              "pipeline) and cache on the result; wrapped fallback only if "
              "the unwrap itself fails (logged) (2026-07-06 review #2, "
              "confirmed).",
        status="test",
        test="tests/test_ui3_review_2026_07_06.py::test_get_field_unwraps_wrapped_phase_on_demand",
        lesson_ref="(2026-07-06 review #2)",
    ),
    BugEntry(
        bug_id="B-095", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="Adaptive-autofocus settlement (long-open backlog): the six "
              "search algorithms had never been measured on real lab data. "
              "9-scene real-lab benchmark (scripts/benchmark_af_real.py, "
              "dense 201-step truth, 40-eval budget, laplacian+entropy) "
              "showed the old default zscan was the LEAST accurate option "
              "(33%/44% <=0.5mm hit) while 'robust' tops both metrics "
              "(78%/67%, 0.07/0.15mm median); adaptive_gradient is "
              "unreliable with ENTROPY (7.2mm median — flat-shoulder "
              "stall); adaptive_distance unreliable on this rig (0-22% "
              "hit). Pinned: ReconParams/settings default af_algorithm="
              "'robust' (v9 state migration stays frozen at zscan for "
              "existing files); headless/MCP autofocus switched from "
              "hardcoded adaptive_gradient to robust + GUI-parity default "
              "metric LAPLACIAN_VARIANCE; benchmark verdicts stamped into "
              "af_algorithm_input_profile tips; adaptive_steps/ staging "
              "folder (Mar 2026 prototype) removed. Full write-up: "
              "docs/AUTOFOCUS_ADAPTIVE.md.",
        status="test",
        test="tests/test_af_settlement.py",
        lesson_ref="(2026-07-06 adaptive settlement)",
    ),
    BugEntry(
        bug_id="B-096", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="Driver relocation (ui2 -> core/drivers) moved workers.py one "
              "directory deeper but _REPO_ROOT kept Path(__file__).parents[2] "
              "— it silently resolved to <repo>/src, _TRACK_C_CHECKPOINT "
              "pointed at a nonexistent path, and reffree_cnn_available() "
              "returned False even with a valid checkpoint installed (Track C "
              "CNN greyed out with no error). Caught by the relocation's "
              "adversarial review workflow, runtime-verified. Fixed: "
              "parents[3] + comment; path pinned by test (2026-07-06).",
        status="test",
        test="tests/test_driver_relocation.py::test_repo_root_survived_the_deeper_move",
        lesson_ref="(2026-07-06 relocation review; __file__-relative paths "
                   "must be re-audited on ANY file move)",
    ),
    BugEntry(
        bug_id="B-097", date="2026-07-06",
        phase=Phase.PILOT_PATCHES,
        topic="core/drivers/__init__.py eagerly imported .workers, so ANY "
              "core.drivers submodule import first executed the parent init "
              "and dragged matplotlib + skimage + the whole recon/qpi/depth/"
              "report stack — `import core.drivers.camera_feed` (numpy-only "
              "module) went to ~0.7s and newly required matplotlib; the ui2.* "
              "shims inherited the regression too (old ui2/__init__ was "
              "deliberately lazy). Fixed: PEP 562 lazy __getattr__ re-exports, "
              "same pattern as ui2/__init__ (2026-07-06 relocation review).",
        status="test",
        test="tests/test_driver_relocation.py::test_camera_feed_import_stays_light",
        lesson_ref="(2026-07-06 relocation review)",
    ),
    BugEntry(
        bug_id="B-098", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="Autofocus plain linear z-scan (search_classic.autofocus_zscan) "
              "returned argmax-of-noise with NO signal when the focus-score "
              "landscape was flat or mostly non-finite — the common "
              "mis-parameterised case (wrong pixel size / magnification / "
              "wavelength / z-range / +1-order mask). The flatness CONCEPT "
              "already existed in Hybrid but only inside adaptive_distance_"
              "search, never on the plain path. Surfaced by the 2026-07-08 "
              "3-version audit (ported from the original Phyton app's "
              "autofocus, which raised here). Fixed: focus_landscape_warning() "
              "helper (<3 finite scores, or (max-min)/scale < 1e-3) attaches a "
              "non-fatal AutoFocusResult.warning — best-guess z preserved, "
              "threaded through the worker AutofocusResult to FocusPanel which "
              "shows it as a 'warn' status + ⚠ instead of a confident 'ok'.",
        status="test",
        test="tests/test_autofocus_diagnostic.py::test_zscan_flags_a_degenerate_all_equal_field",
        lesson_ref="(2026-07-08 version audit — do-now salvage from Phyton)",
    ),
    BugEntry(
        bug_id="B-099", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="Manual off-axis +1-order center was accepted by the CORE "
              "(OffAxisParams.center_yx / build_plus_one_order_mask) but "
              "UNREACHABLE: none of the three OffAxisParams call sites "
              "(reconstruct / qpi / autofocus+depth) threaded it from "
              "ReconParams, and ReconParams had no field for it — so on "
              "noisy / multi-order / low-carrier holograms where auto "
              "peak_local_max detection fails, the user had no override "
              "(the original Phyton app's own unfinished NEXT item). Fixed: "
              "ReconParams.offaxis_center + a single _offaxis_params() helper "
              "at all three sites; ui3 spectrum-viewport click handler "
              "(armed via Process ▸ Pick +1 order center, one-shot, crosshair "
              "marker + cursor) writes it and re-reconstructs, with a "
              "Reset-to-auto escape hatch; persisted as a tuple field "
              "(2026-07-08; same 'core-capable-but-unwired' class as B-079).",
        status="test",
        test="tests/test_offaxis_center_pick.py::test_driver_prepare_field_honours_manual_center",
        lesson_ref="(2026-07-08 version-audit backlog item)",
    ),
    BugEntry(
        bug_id="B-100", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="The B-098 flat/non-finite autofocus diagnostic "
              "(focus_landscape_warning) was wired ONLY to the linear zscan; "
              "the driver did warning=getattr(core_result,'warning',None), so "
              "every other algorithm — critically the SETTLED DEFAULT 'robust' "
              "(B-095) plus coarse_to_fine/golden/adaptive — always returned "
              "warning=None. A mis-parameterised run on the default path (wrong "
              "pixel/mag/wavelength, bad z-range, wrong +1-order mask) still "
              "degraded SILENTLY, exactly the failure B-098 meant to end. "
              "Found by verifying (not trusting) the version-audit's 'HIGH "
              "confidence' salvage — the diagnostic already existed but was "
              "dead on the default path. Fixed: _landscape_warning() diagnoses "
              "from each search's retained landscape — robust from its uniform "
              "coarse_z/coarse_scores; golden/coarse_to_fine (no landscape) and "
              "non-uniform adaptive traces intentionally left None to avoid "
              "false positives (2026-07-08).",
        status="test",
        test="tests/test_autofocus_diagnostic.py::test_robust_search_on_degenerate_field_is_flagged_end_to_end",
        lesson_ref="(2026-07-08; 'wired to the non-default path' silent-degrade "
                   "class — a diagnostic only helps on the path users run)",
    ),
    # ---- 2026-07-08 CLAUDE.md-lens review (find→verify→synthesize, 7 agents
    # ---- + 34 verifiers). fix_now batch; flag_for_decision items NOT auto-fixed.
    BugEntry(
        bug_id="B-101", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="SECURITY (path traversal): _tool_map_sample_grid took the LLM's "
              "sample_id verbatim (schema was a bare string) and it became "
              "state_dir/{sample_id}.json in persist_sample_map; an MCP "
              "client/LLM calling map_sample_grid(sample_id='../../../tmp/x') "
              "escaped the state dir and could overwrite any user-writable "
              "*.json — bypassing the validate_path/restrict_to_home guard "
              "load_hologram enforces. Fixed: reject sample_id not matching "
              "[A-Za-z0-9._-]{1,64} (and '.'/'..') before any write "
              "(2026-07-08 review).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_map_sample_grid_rejects_unsafe_sample_id",
        lesson_ref="(2026-07-08; guard present on load path, absent on write path)",
    ),
    BugEntry(
        bug_id="B-102", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 ReconPanel._on_reference_mode_changed showed 'Reference mode "
              "selected but no reference is loaded' (warn) then IMMEDIATELY "
              "overwrote it with set_status('Reference mode: …','info') — a "
              "plain setText replace — so the operator never saw the warning "
              "and Reconstruct came out silently unreferenced. Fixed: the info "
              "line is now the else branch of the pathless-reference check "
              "(2026-07-08 review, silent-degrade cluster).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_recon_panel_reference_warning_not_overwritten",
        lesson_ref="(2026-07-08 review)",
    ),
    BugEntry(
        bug_id="B-103", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="coarse_to_fine_search forwarded cancel_check/ref_field/est_total "
              "to its Golden fine-refinement but NOT roi_bounds — the coarse "
              "sweep used the operator's AF-ROI evaluator, the fine phase "
              "rebuilt a FULL-frame one, so ROI autofocus silently returned "
              "the full-frame optimum (a brighter out-of-ROI particle) instead "
              "of the drawn cell's. golden_section_search already accepts "
              "roi_bounds — pure forwarding oversight. Fixed: forward it "
              "(2026-07-08 review).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_coarse_to_fine_forwards_roi_bounds_to_golden",
        lesson_ref="(2026-07-08 review; guard/feature present on sibling path)",
    ),
    BugEntry(
        bug_id="B-104", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 state._params_to_dict blanket-skipped reference_path (a "
              "Path, 'non-serialisable') while STILL persisting "
              "reference_mode='reference' + subtract_reference=True. On "
              "relaunch reference mode was re-armed with reference_path=None → "
              "the core guard (effective_reference_mode()=='reference' AND "
              "path) fell through to the UNreferenced field → systematically "
              "wrong QPI/OPD presented as reference-corrected, no warning. "
              "Fixed: persist Path fields as str and coerce back on load "
              "(_PATH_PARAM_FIELDS) (2026-07-08 review, silent-degrade cluster).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_reference_path_round_trips_and_does_not_arm_a_pathless_reference",
        lesson_ref="(2026-07-08 review; same 'skip non-serialisable' shape as B-075)",
    ),
    BugEntry(
        bug_id="B-105", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="workers._extract_evaluations guarded only on hasattr(result,"
              "'evaluations'); the DEFAULT robust RobustSearchResult names it "
              "total_evaluations, so AutofocusResult.scanned and the "
              "reproducibility audit reported the coarse n_steps (~half the "
              "true ~72 evals) for the shipped-default algorithm. Fixed: also "
              "honor total_evaluations (2026-07-08 review).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_extract_evaluations_honours_total_evaluations",
        lesson_ref="(2026-07-08 review; only the default path silently understated)",
    ),
    BugEntry(
        bug_id="B-106", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="workers depth-map path wrapped segment_depth_clusters in a bare "
              "`except Exception: clusters=[]` with NO log — a crashing "
              "segmenter was indistinguishable from a genuinely empty scene "
              "(both record clusters:0), destroying the ability to diagnose. "
              "Fixed: _LOG.exception before the empty fallback, matching every "
              "other swallow in the module (2026-07-08 review).",
        status="manual",
        test="",
        lesson_ref="(2026-07-08 review; log-only change, behavior unchanged — "
                   "a swallow that doesn't even log is the worst case)",
    ),
    BugEntry(
        bug_id="B-107", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="dhm_mcp headless recon summary emitted phase_mean_rad/"
              "phase_std_rad while the GUI emits phase_mean/phase_std and "
              "record_timelapse's extractor reads only 'phase_std' — so every "
              "MCP timelapse frame silently lacked the phase-drift signal that "
              "is the tool's stated purpose (the LLM reasoned on absent data "
              "with a success indication). Fixed: emit the unsuffixed aliases "
              "alongside the _rad keys (2026-07-08 review, GUI/MCP contract "
              "drift).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_headless_recon_summary_has_unsuffixed_phase_aliases",
        lesson_ref="(2026-07-08 review; producer/consumer key-name drift across frontends)",
    ),
    # ---- 2026-07-08 review, flag_for_decision items subsequently resolved
    # ---- ('devam'): the ones with an unambiguous 'no silent degrade' answer.
    BugEntry(
        bug_id="B-108", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="qpi.compute_cell_morphology fabricated a hardcoded Δn=0.043 "
              "(the DEFAULT n_sample-n_medium contrast) when n_sample==n_medium, "
              "silently returning invented max/mean height + volume as if "
              "physically real — while the sibling opd_to_height RAISES for the "
              "same Δn≈0 case. Fixed: route height through opd_to_height (single "
              "source of truth) so Δn≈0 raises. Its caller compute_qpi wrapped "
              "segmentation+morphology in a SILENT bare `except Exception` that "
              "degraded dry mass to the whole-field integral with no log; added "
              "a WARNING so a missing-contrast run / real crash is no longer "
              "indistinguishable from a normal cell measurement (2026-07-08 "
              "review; folds in the 'compute_qpi swallows all exceptions' "
              "finding).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_cell_morphology_raises_on_zero_contrast",
        lesson_ref="(2026-07-08; a sibling raised while this one fabricated — "
                   "guards must be consistent across paired helpers)",
    ),
    BugEntry(
        bug_id="B-109", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="camera_feed.AcquisitionThread had no error channel: when "
              "source.start()/grab() blew up mid-acquisition the daemon thread "
              "logged and died, but nothing told the UI — CameraPanel kept the "
              "last frame on screen with Start/Record still armed, a frozen feed "
              "indistinguishable from a live one (and 'recording' apparently "
              "ongoing). Fixed: additive on_error callback fired on start/grab "
              "failure; CameraPanel wires it to a signal + slot that surfaces an "
              "error status/toast and resets the controls to 'stopped' "
              "(2026-07-08 review).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_acquisition_thread_emits_on_error_when_grab_fails",
        lesson_ref="(2026-07-08; a worker thread that can die needs an "
                   "error channel to its owner, not just a log line)",
    ),
    BugEntry(
        bug_id="B-110", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="Reference-division RUNTIME failure (reference file moved/corrupt/"
              "shape-mismatch) was logged then SILENTLY swallowed: "
              "_extract_field_with_reference / _prepare_sample_and_ref_fields "
              "returned the unreferenced field and every pipeline painted a "
              "clean 'success' — the operator got quantitatively wrong "
              "phase/OPD/dry-mass/height with no signal. Third path of the "
              "'reference silently skipped' family (B-102 config, B-104 "
              "persistence, this = runtime). Fixed: both helpers return an "
              "actionable note; threaded through ReconResult.reference_note "
              "(reconstruct), _prepare_field's 4-tuple → AutofocusResult."
              "warning (combined via _join_notes) / QPIOneShotResult / "
              "QPIBatchResultWrap / MultiFocusResult / DepthMapResultWrap "
              ".warning; surfaced as a warn status+toast in the shell recon "
              "handler + qpi/depth/focus panels (2026-07-08 review, "
              "flag_for_decision item done on 'devam').",
        status="test",
        test="tests/test_review_2026_07_08.py::test_extract_field_notes_a_failed_reference_division",
        lesson_ref="(2026-07-08; a graceful degrade must still SURFACE — a "
                   "logged-but-unshown fallback is a silent-degrade to the user)",
    ),
    BugEntry(
        bug_id="B-111", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="autofocus.analysis.auto_select_metric scored a peakless "
              "(monotonic / edge-focus) metric by the curve height "
              "np.max(values_smooth) ≈ 1.0 when find_peaks found no interior "
              "peak — so a metric whose optimum sits AT/OUTSIDE the scanned "
              "range (the LEAST reliable) could beat a genuinely-peaked metric "
              "(prominence < 1), selecting the worst objective. Reliability == "
              "peak prominence and a peakless curve has none: fixed to score 0. "
              "Blast radius verified small — only the v1-GUI opt-in "
              "'auto-select metric' toggle (autofocus_worker.py:175) calls it; "
              "NOT the settled B-095 default path (2026-07-08 review, "
              "flag_for_decision item done on 'devam').",
        status="test",
        test="tests/test_review_2026_07_08.py::test_auto_select_metric_prefers_interior_peak_over_monotonic",
        lesson_ref="(2026-07-08; a no-peak fallback that returns curve height "
                   "inverts the very reliability ranking it's meant to compute)",
    ),
    BugEntry(
        bug_id="B-112", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="observe.inspect_phase_map (AI-vision tool) reported cell_count "
              "as len(unique(mask[mask>0])) over qpi.segment_cell_phase's mask "
              "— but that returns a single-largest-component BINARY (0/255) "
              "mask by design, so the count was structurally 0 or 1: any "
              "multi-cell field of view was reported to the LLM as '1 cell'. "
              "Fixed: an independent threshold + connected-component label in "
              "observe (dropping sub-0.05%-area specks) gives a genuine count; "
              "dry mass now integrates over ALL detected cells. Verified real "
              "(the 'uncertain' finding was correct): segment_cell_phase both "
              "binarises AND keeps-largest (2026-07-08 review).",
        status="test",
        test="tests/test_observe.py::test_inspect_phase_map_segmentation_counts_cells",
        lesson_ref="(2026-07-08; reusing a single-object segmenter for a "
                   "multi-object count silently caps the answer at 1)",
    ),
    BugEntry(
        bug_id="B-113", date="2026-07-08",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 UX/layout pass from real-screen user feedback ('not fluid "
              "like v1, some windows don't open, they overlap so I can't tell "
              "what's where, scroll problems'). Diagnosed by offscreen "
              "introspection (dialogs all CONSTRUCT fine — not a load failure). "
              "Three root causes fixed: (1) NONE of the 8 feature panels were "
              "scroll-wrapped, so 500–740px content clipped in a shorter dock "
              "with no way to reach lower controls → mount_panel now wraps each "
              "in a resizable QScrollArea; (2) every dialog opened via bare "
              "show()+raise()+activateWindow() at Qt's DEFAULT position, so they "
              "stacked on the same spot and behind the main window → shared "
              "present_centered() centres on the parent with a small cascade; "
              "(3) the timelapse dock was claimed by no workflow mode, so it "
              "fell into 'unmanaged → always visible' and leaked into every "
              "mode → added to Acquire (2026-07-08).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_every_feature_panel_is_scroll_wrapped",
        lesson_ref="(2026-07-08; observe > guess — offscreen widget "
                   "introspection pinned the causes without needing screenshots)",
    ),
    BugEntry(
        bug_id="B-114", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 ReconPanel's 'Hologram' status label read ctx."
              "hologram_path() only in its constructor (_refresh_file_label at "
              "build time), so after _load_path loaded a file it still showed "
              "'(no hologram loaded)' right beside the clearly-loaded filename "
              "in the top label — a confusing desync. Found by a systematic "
              "offscreen UI audit (QWidget.grab() PNGs of every mode/panel/"
              "dialog). Fixed: _load_path now calls the panel's "
              "_refresh_file_label() after setting the hologram (2026-07-10 "
              "UI audit).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_loading_a_hologram_syncs_the_recon_panel_label",
        lesson_ref="(2026-07-10; a label that reads a value once at build time "
                   "desyncs the moment that value changes — refresh on change)",
    ),
    BugEntry(
        bug_id="B-115", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 restored QMainWindow.restoreState() unconditionally from "
              "the saved window_state_b64. A saved layout from a DIFFERENT "
              "dock schema (panels added/removed/scroll-wrapped/ui2 retired "
              "since it was written) does not map cleanly, so restoreState "
              "left mismatched docks FLOATING as separate windows overlapping "
              "the central 2x2 image grid — the user's 'windows overlap, I "
              "can't tell what's where' on the REAL screen (invisible offscreen "
              "and to a fresh state; only reproduced by screenshotting the "
              "user's actual restored session). Fixed: a _LAYOUT_VERSION stamp "
              "(Ui3State.layout_version) — the dock layout restores ONLY when "
              "the saved version matches the current schema; window geometry "
              "(size/pos) still always restores. Bump the version on any dock-"
              "structure change (2026-07-10 real-screen audit).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_stale_dock_layout_is_not_restored",
        lesson_ref="(2026-07-10; QMainWindow.saveState/restoreState silently "
                   "corrupts the layout across UI changes — version-gate it)",
    ),
    BugEntry(
        bug_id="B-116", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 had NO affordance to re-home a dock: a user floated a "
              "dock to inspect it (or dragged it) and could not put it back "
              "('soldaki paneli inceliyordum, yerine koyamadım kaldı öyle') — "
              "'Restore grid' (Ctrl+0) only un-maximized the image panels, it "
              "did nothing to docks, and Qt's re-dock-by-drag is not "
              "discoverable. Added _reset_panel_layout(): un-floats + re-docks "
              "every dock to its home area (reconstruct→Left, features→Right "
              "tabified), restores the image grid, re-applies the workflow "
              "mode. Wired to View ▸ Reset panel layout (Ctrl+Shift+0) and the "
              "⌘K command palette (2026-07-10 real-screen user feedback).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_reset_panel_layout_redocks_a_floated_dock",
        lesson_ref="(2026-07-10; any move-out UI affordance needs a matching, "
                   "discoverable put-back — a floatable dock without a reset)",
    ),
    BugEntry(
        bug_id="B-117", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 right-side feature panels were confusing: all 8 docks were "
              "tabified into one group at mount, and _apply_workflow_mode only "
              "toggled setVisible() per mode. Toggling visibility on a "
              "pre-tabified group leaves Qt's tab bar inconsistent — in Analyse "
              "only 'focus' (Autofocus) rendered and qpi/depth/ai were "
              "unreachable, so the user peeled panels back one at a time ('as I "
              "close one, another appears behind, confusing') and the tab bar "
              "sat at the bottom where it was easy to miss. Diagnosed by "
              "screenshotting the real running app. Fixed: _apply_workflow_mode "
              "RE-TABIFIES the visible feature docks into one clean group every "
              "switch (raising the first), and the tab bar moved to the top "
              "(setTabPosition North). Analyse now shows a proper "
              "Autofocus|QPI|Depth|AI-copilot tab bar; verified live "
              "(2026-07-10 real-screen).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_workflow_mode_forms_one_tab_group_from_visible_docks",
        lesson_ref="(2026-07-10; QDockWidget setVisible on a tabified group "
                   "corrupts the tab bar — re-tabify the visible set explicitly)",
    ),
    BugEntry(
        bug_id="B-118", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 QSpinBox/QDoubleSpinBox/QComboBox arrows rendered as blank "
              "light-gray squares — visual noise that made the parameter panels "
              "look cheap/unfinished ('iğrenç'). Root cause: the QSS used the "
              "web CSS border-triangle trick (width:0;height:0 + border colors) "
              "for ::up-arrow/::down-arrow, which Qt's stylesheet engine does "
              "NOT render as a triangle — it draws the native indicator block "
              "instead. Verified empirically in an isolated widget harness that "
              "(a) the border trick draws squares, (b) data: URIs in image:url() "
              "render nothing (Qt's QSS loader treats the URL as a file path), "
              "and (c) a real PNG file renders a crisp triangle. Fixed: "
              "design._ensure_arrow_icons() paints up/down/chevron PNGs in the "
              "palette's muted colour (rendered at 2x logical size for Retina "
              "crispness), caches them under ~/.dhm-reconstruction/icons by "
              "colour+shape-version, and build_qss references them via "
              "image:url(). Only runs when a QApplication exists; headless "
              "callers (tests) degrade to width:0 arrows, never the blank "
              "squares. Verified offscreen (dark + high_contrast) and at 2x DPR.",
        status="test",
        test="tests/test_review_2026_07_08.py::test_arrow_icons_are_generated_and_referenced",
        lesson_ref="(2026-07-10; Qt QSS ignores the CSS border-triangle trick "
                   "and data: URIs for sub-control arrows — use a real PNG file, "
                   "rendered at 2x for HiDPI)",
    ),
    BugEntry(
        bug_id="B-119", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 Advanced group (a checkable QGroupBox) had two defects found "
              "in the multi-agent polish audit: (1) collapsed by default it still "
              "drew its card border around the now-hidden content, showing an "
              "orphaned empty frame on every load; (2) its checkbox indicator was "
              "invisible on the high-contrast theme because the QSS styled "
              "QCheckBox::indicator but not QGroupBox::indicator, so the group's "
              "own checkbox fell back to a native indicator that vanished on the "
              "black background. Fixed: recon_panel sets a 'collapsed' dynamic "
              "property (repolished on toggle) and design.build_qss adds "
              "QGroupBox[collapsed=\"true\"] {border:none;background:transparent} "
              "plus a full QGroupBox::indicator style mirroring the checkbox. "
              "Verified offscreen on dark + high_contrast (indicator now a "
              "white-bordered square, no empty frame, expand still frames the "
              "content).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_advanced_group_starts_collapsed_and_toggles",
        lesson_ref="(2026-07-10; a checkable QGroupBox needs its ::indicator "
                   "styled explicitly and a collapsed-state rule to avoid an "
                   "empty framed box)",
    ),
    BugEntry(
        bug_id="B-120", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 QPI batch tables (the in-dock qpi_panel.batch_table and the "
              "QPIBatchDialog table) used QHeaderView.Stretch, which forces equal "
              "column widths regardless of text — so headers like 'DRY MASS (PG)' "
              "and 'OPD RANGE (NM)' were clipped mid-word into ambiguous labels "
              "('RY MASS', 'IPD RANGE'), misleading in an optics tool. Fixed: "
              "ResizeToContents + stretchLastSection so each column fits its "
              "header and the last absorbs leftover width (horizontal scroll when "
              "the full set overflows). Verified: headers render in full.",
        status="test",
        test="tests/test_review_2026_07_08.py::test_qpi_batch_tables_fit_headers_not_stretch",
        lesson_ref="(2026-07-10; QHeaderView.Stretch clips long headers — use "
                   "ResizeToContents + stretchLastSection for labelled columns)",
    ),
    BugEntry(
        bug_id="B-121", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 error feedback rendered in the wrong colour: recon/qpi/report "
              "panels called ctx.toast(msg, 'danger') on failures, but the "
              "ToastHost level map only knows info/ok/warn/error and "
              "_LEVEL_ROLE.get('danger', 'accent') fell back to the blue 'accent' "
              "(info) role — so 'Reconstruction failed' / 'QPI failed' toasts "
              "showed as info, not red. 'danger' is a ROLE (button/label colour), "
              "not a toast LEVEL. Fixed all six call sites to level 'error' "
              "(which maps to the 'danger' role → red). The audit only flagged "
              "report_panel; grep found the same bug in recon_panel and "
              "qpi_panel.",
        status="test",
        test="tests/test_review_2026_07_08.py::test_error_toasts_use_recognized_level",
        lesson_ref="(2026-07-10; toast LEVEL vocabulary (info/ok/warn/error) is "
                   "not the ROLE vocabulary — a wrong level silently degrades to "
                   "the accent fallback)",
    ),
    BugEntry(
        bug_id="B-122", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 AI copilot health pill (health_label) was created role='muted' "
              "and never changed, so '● connected' / '● unavailable' / "
              "'● checking…' all rendered in the same grey — the ● dot colour "
              "never signalled status. Fixed: design.build_qss gains status-tinted "
              "QLabel[role=ok|warn|danger] roles, and ai_panel routes every state "
              "through a _set_health(text, role) helper that swaps the role and "
              "repolishes (connected→ok/green, unavailable→danger/red, "
              "checking/not-checked→muted).",
        status="test",
        test="tests/test_review_2026_07_08.py::test_ai_health_pill_recolors_by_state",
        lesson_ref="(2026-07-10; a status pill must recolour per state — set the "
                   "role + repolish, don't leave it on the construction-time role)",
    ),
    BugEntry(
        bug_id="B-123", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 miscellaneous polish from the audit, batched: the loaded "
              "filename showed twice in the Reconstruct dock (header caption + the "
              "panel's 'Hologram' group) — removed the header _file_label so it "
              "shows once; the time-lapse dock title said 'Timelapse' while its "
              "heading/status said 'Time-lapse' — unified to 'Time-lapse'; the "
              "reference-path label carried a decorative 📎 emoji clashing with the "
              "mono instrument aesthetic — removed; the central imaging grid's "
              "QSplitter handles were unstyled — added QSplitter::handle to match "
              "the dock separators; the camera Stop button lacked the danger role "
              "its time-lapse twin has — added; the status-bar busy label lacked "
              "the muted role its sibling uses — added.",
        status="test",
        test="tests/test_review_2026_07_08.py::test_no_duplicate_file_label_and_timelapse_named_consistently",
        lesson_ref="(2026-07-10; batch of audit-found label/role/naming "
                   "inconsistencies — one visible display per datum, consistent "
                   "roles across twin controls)",
    ),
    BugEntry(
        bug_id="B-124", date="2026-07-10",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 window opened taller than the display, pushing the "
              "Reconstruct/Autofocus buttons off the bottom edge (user report: "
              "'aşağısı sığmıyor' — the bottom doesn't fit; screenshot showed the "
              "left control panel running past the screen with the action buttons "
              "unreachable and no scrollbar). Two compounding causes: (1) the "
              "control dock set its (very tall) panel widget directly, so the "
              "panel's minimum height became the window's minimum height — the "
              "window could not shrink below the panel and thus never fit a "
              "shorter screen; fixed by scroll-wrapping the control dock in a "
              "QScrollArea (vertical only) like the feature docks, so overflow "
              "scrolls. (2) _restore_geometry replayed a saved geometry with no "
              "screen-fit check, so a window sized on a larger display reopened "
              "off-screen. Per the user's follow-up ('autofit olsun') the window "
              "SIZE no longer restores at all — _autofit_to_screen() sizes it to "
              "the usable QScreen.availableGeometry every launch (width capped at "
              "1920 so it doesn't sprawl on huge monitors; height fills the "
              "space) and runs again once on showEvent when the real screen + "
              "frame margins are known; only the dock LAYOUT is restored. "
              "Together: window always autofits the screen, control dock scrolls, "
              "bottom buttons reachable. Verified: window fits within the usable "
              "screen; a 700px-tall window yields a scrollable control dock.",
        status="test",
        test="tests/test_review_2026_07_08.py::test_window_autofits_screen_and_control_dock_scrolls",
        lesson_ref="(2026-07-10; setting a tall panel directly as a dock's widget "
                   "makes the window un-shrinkable — scroll-wrap it; and autofit "
                   "the window to the current screen instead of restoring a saved "
                   "size that may not fit)",
    ),
    BugEntry(
        bug_id="B-125", date="2026-07-11",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 every feature panel showed its name up to three times stacked "
              "— the tab label (when tabified), the QDockWidget title bar, and a "
              "large in-panel role='heading' QLabel (audit #10/#17/#21) — reading "
              "as cluttered/redundant. The deferred concern was that panels might "
              "double as standalone dialogs where the heading is the only title; "
              "investigation showed no panel is ever mounted as_dock=False and "
              "_panel_action only ever shows docked panels as docks, so every "
              "mounted panel always has a dock title bar. Fixed: mount_panel (and "
              "_build_control_dock) call _hide_panel_heading(widget), which hides "
              "the panel's single role='heading' label — the title bar / tab is "
              "now the one label, and ~40px of vertical space is reclaimed per "
              "panel. Standalone QDialog viewers (surface/qpi_batch/audit/…) are "
              "never routed through mount_panel, so they keep their heading. "
              "Verified offscreen: recon/ai/qpi docks show no heading; the QPI "
              "batch dialog still shows 'QPI batch — candidate comparison'.",
        status="test",
        test="tests/test_review_2026_07_08.py::test_docked_panels_hide_redundant_heading_dialogs_keep_it",
        lesson_ref="(2026-07-11; a panel that is always docked shouldn't repeat "
                   "its dock-title name as a big in-panel heading — hide it at "
                   "mount time, but only for the dock path)",
    ),
    BugEntry(
        bug_id="B-126", date="2026-07-11",
        phase=Phase.PILOT_PATCHES,
        topic="ui3 design-token tokenization of stray magic numbers found by the "
              "audit (#24/#26/#31/#34/#35): (a) input min-height was 22px and "
              "button min-height 20px — an inconsistent pair off the 4-based "
              "Space scale — now both derive from Space.xl (24); (b) the viewport "
              "panel-title QLabel used QFont().setPointSize(10) — a pt magic "
              "number in an otherwise px-based system; measured 10pt==13px here "
              "and switched to setPixelSize(Type.label) so the rendered size is "
              "unchanged; (c) camera_panel and focus_panel root layouts used a "
              "hardcoded margin/spacing of 10 (not on the scale) — now Space.md "
              "(12), matching recon's convention; (d) camera_panel and "
              "focus_panel explicitly right-aligned their form labels while "
              "recon/timelapse used the platform default — dropped the two "
              "overrides so every panel follows the same platform-default "
              "alignment (consistent on macOS and offscreen alike). Verified "
              "offscreen: viewport header stays 13px, inputs read fine at 24px, "
              "no visible regression. Two audit items were deliberately NOT taken: "
              "#28 (form spacing 6->Space.sm 8) would add ~50px to the tall "
              "control panel and undo the B-124 fit work; #25 (unify the 3 "
              "uppercase 'eyebrow' QSS treatments) — the three uses diverge in "
              "padding/border/weight, so a shared rule adds indirection without "
              "clear benefit.",
        status="test",
        test="tests/test_review_2026_07_08.py::test_control_sizes_derive_from_design_tokens",
        lesson_ref="(2026-07-11; measure a pt magic number before swapping to px; "
                   "and a tokenization that grows a panel can conflict with fit "
                   "work — skip it and say why)",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def entries_for_phase(phase: Phase) -> List[BugEntry]:
    """Filter by phase. Empty list is valid — future phases are
    declared in the enum before any bugs land."""
    return [e for e in BUG_REGISTRY if e.phase is phase]


def phase_summary() -> dict:
    """``{phase_value: entry_count}`` — quick at-a-glance count of how
    many bugs each phase holds. Used by the all-phases header."""
    counts: dict = {}
    for e in BUG_REGISTRY:
        counts[e.phase.value] = counts.get(e.phase.value, 0) + 1
    return counts


__all__ = [
    "Phase",
    "BugEntry",
    "BUG_REGISTRY",
    "entries_for_phase",
    "phase_summary",
]
