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
        status="test",
        test="tests/test_ui2_v207.py::test_info_text_shows_optical_mode",
        lesson_ref="2026-04-24 — Info text append birikince stale",
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
        status="test",
        test="tests/test_ui2_v207.py::test_load_hologram_defines_flip_before_push",
        lesson_ref="(5-bug sprint)",
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
        status="test",
        test="tests/test_ui2_v207.py::test_load_hologram_runs_without_nameerror",
        lesson_ref="(5-bug sprint)",
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
        status="test",
        test="tests/test_ui2_v207.py::test_load_hologram_defines_flip_before_push",
        lesson_ref="(5-bug sprint)",
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
