"""Dear PyGui main window for DHM Reconstruction (v2 UI).

Layout:

    ┌───────────────────────────────────────────────────────────────┐
    │ Menu bar:  File  •  Tools  •  Help                            │
    ├────────────────┬──────────────────────────────────────────────┤
    │ Sidebar        │  ┌─────────────┐  ┌─────────────┐            │
    │  · Sample ID   │  │  Input      │  │  Amplitude  │            │
    │  · λ [spin]    │  │             │  │             │            │
    │  · px [spin]   │  └─────────────┘  └─────────────┘            │
    │  · z  [spin]   │  ┌─────────────┐  ┌─────────────┐            │
    │  · mask [spin] │  │  Phase      │  │  Spectrum   │  (future)  │
    │  · Method [c]  │  │             │  │             │            │
    │                │  └─────────────┘  └─────────────┘            │
    │  [Reconstruct] │                                               │
    ├────────────────┴──────────────────────────────────────────────┤
    │ status bar — single line, last operation                      │
    └───────────────────────────────────────────────────────────────┘

The render loop polls a single ``_pending`` mailbox — the
:class:`ReconstructionDriver` fills it from a worker thread. No Qt,
no pyqtgraph; all the heavy-lifting lives in ``core/*``.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from core.settings_schema import AppSettings, Ui2State

# Dear PyGui must be importable — the ``ui2`` package has it as a
# hard dependency. Fail fast with a helpful message if missing.
try:
    import dearpygui.dearpygui as dpg
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "dearpygui is required for the v2 UI. "
        "Install with: pip install dearpygui"
    ) from exc

from .theme import PALETTES, ScientificTheme
from .reconstruction import (
    ReconParams, ReconResult, ReconError, ReconstructionDriver,
)
from .workers import (
    AutofocusResult, BundleExportResult, DepthMapResultWrap,
    MultiFocusResult, QPIBatchResultWrap, QPIOneShotResult,
    ReportExportResult, ScienceDriver,
    af_algorithm_input_profile,
    available_autofocus_algorithms, available_focus_metrics,
    write_qpi_batch_csv, write_qpi_csv,
)


def _available_focus_metric_names() -> list[str]:
    """Small indirection so the sidebar combo stays decoupled from the
    enum — easier to stub in unit tests."""
    try:
        return available_focus_metrics()
    except Exception:
        # Shouldn't happen at runtime, but keeps the sidebar buildable
        # in a test environment where core.autofocus isn't importable.
        return ["LAPLACIAN_VARIANCE"]


def _available_autofocus_algorithms() -> list[str]:
    """Same indirection idea for algorithm combo."""
    try:
        return available_autofocus_algorithms()
    except Exception:
        return ["zscan"]
from .widgets import (
    CommandPalette, PresetChips, ToastStack, show_help_overlay,
)
from .dialogs import (
    show_audit_viewer, show_focus_candidates, show_onboarding,
    show_qpi_batch_review,
)
from .image_panel import ZoomableImagePanel
from .state_store import DebouncedSaver, STATE_PATH, load as load_state
from .surface import open_surface
from .camera_feed import (
    AcquisitionThread, CameraSource, SyntheticCamera, TiffStackRecorder,
)

_LOG = logging.getLogger("ui2")


# ---------------------------------------------------------------------------
# Texture helpers — convert numpy arrays to RGBA for Dear PyGui's texture API
# ---------------------------------------------------------------------------

def _to_rgba(arr: np.ndarray, *, colormap: str = "gray",
             contrast: str = "minmax") -> np.ndarray:
    """Convert a 2-D float array to an (H, W, 4) float32 RGBA in [0, 1].

    ``contrast`` controls the normalisation range:
    * ``"minmax"`` — raw min/max (default, faithful to the data)
    * ``"percentile"`` — clip to the 1st/99th percentile before scaling,
      which keeps reference-divided amplitude images readable when a
      handful of outlier pixels (near zero in the reference) dominate
      the histogram and wash out the rest.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D array, got {a.shape}")
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if contrast == "percentile" and a.size >= 4:
        lo = float(np.percentile(a, 1.0))
        hi = float(np.percentile(a, 99.0))
    else:
        lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        norm = np.zeros_like(a)
    else:
        norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    if colormap == "phase":
        # HSV-like wheel for *wrapped* phase (periodic in [0, 2π]).
        hue = norm
        rgba = np.zeros((*a.shape, 4), dtype=np.float32)
        # Simple sinusoidal mapping — readable, no extra deps.
        rgba[..., 0] = 0.5 + 0.5 * np.cos(2 * np.pi * hue)
        rgba[..., 1] = 0.5 + 0.5 * np.cos(2 * np.pi * (hue - 1/3))
        rgba[..., 2] = 0.5 + 0.5 * np.cos(2 * np.pi * (hue - 2/3))
        rgba[..., 3] = 1.0
        return rgba
    if colormap == "depth":
        # Absolute-value ramp for depth maps (z is physical metres, not
        # periodic — phase wheel would falsely wrap near the extremes).
        # Approximates viridis: blue → teal → green → yellow without
        # pulling matplotlib's LUT into the preview pipeline.
        rgba = np.empty((*a.shape, 4), dtype=np.float32)
        rgba[..., 0] = np.clip(0.267 + 0.748 * norm, 0.0, 1.0)
        rgba[..., 1] = np.clip(0.005 + 0.907 * norm, 0.0, 1.0)
        rgba[..., 2] = np.clip(0.329 + 0.212 * norm - 0.560 * norm ** 2,
                               0.0, 1.0)
        rgba[..., 3] = 1.0
        return rgba
    # Default: plain grayscale.
    rgba = np.empty((*a.shape, 4), dtype=np.float32)
    rgba[..., 0] = norm
    rgba[..., 1] = norm
    rgba[..., 2] = norm
    rgba[..., 3] = 1.0
    return rgba


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class DhmApp:
    """Scientific-tone DHM reconstruction app (Dear PyGui)."""

    TITLE = "DHM Reconstruction — v2 UI"

    # File extensions we'll accept via the file dialog + drag-and-drop.
    _IMAGE_EXTS: tuple[str, ...] = (
        ".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg",
    )
    # How many recent paths the File menu remembers.
    _MAX_RECENT = 6

    # Drop zone height — viewport-level drop is the primary path; the
    # visible strip gives macOS users (where Cocoa binding is broken
    # upstream in Dear PyGui 2.x) a clickable fallback.
    _DROP_ZONE_HEIGHT = 48

    def __init__(self) -> None:
        # --- Hydrate persisted state first — used throughout init -----
        self._settings: AppSettings = load_state()
        ui2 = self._settings.ui2

        self._driver = ReconstructionDriver()
        self._science = ScienceDriver()
        self._toasts = ToastStack()
        self._palette = CommandPalette()

        # Reconstruction parameters — hydrated from disk so the next
        # launch picks up where the operator left off. The live
        # ReconParams lives inside ``_params`` and is the source of
        # truth; ``_settings.ui2`` is a snapshot that we re-sync on
        # every mark_dirty.
        self._params = ReconParams(
            wavelength_nm=float(ui2.wavelength_nm),
            pixel_um=float(ui2.pixel_um),
            z_mm=float(ui2.z_mm),
            mask_radius=int(ui2.mask_radius),
            method=str(ui2.method),
            reference_path=(Path(ui2.reference_path)
                            if ui2.reference_path else None),
            subtract_reference=bool(ui2.subtract_reference),
            magnification=float(ui2.magnification),
            pixel_is_effective=bool(ui2.pixel_is_effective),
            n_sample=float(ui2.n_sample),
            n_medium=float(ui2.n_medium),
            autofocus_metric=str(ui2.autofocus_metric),
            af_z_min_mm=float(ui2.af_z_min_mm),
            af_z_max_mm=float(ui2.af_z_max_mm),
            af_n_steps=int(ui2.af_n_steps),
            subtract_mean=bool(ui2.subtract_mean),
            hann_window=bool(ui2.hann_window),
            fft_backend=str(ui2.fft_backend),
            unwrap_method=str(ui2.unwrap_method),
            optical_mode=str(getattr(ui2, "optical_mode", "transmission")),
            auto_contrast_amplitude=bool(
                getattr(ui2, "auto_contrast_amplitude", True)
            ),
            af_algorithm=str(getattr(ui2, "af_algorithm", "zscan")),
        )
        self._sample_id: str = ui2.sample_id
        self._current_hologram: Optional[Path] = None
        self._recent: list[Path] = [Path(p) for p in ui2.recent
                                    if p]
        self._theme_name: str = ui2.theme or "dark"
        self._workflow_mode: str = ui2.workflow_mode or "Reconstruct"
        self._selected_preset: str = ui2.selected_preset or ""
        self._last_dir: str = ui2.last_dir

        # Drop-and-drop capability: Dear PyGui 2.x ships
        # ``set_viewport_drop_callback`` on every platform but the
        # macOS Cocoa binding doesn't actually emit events. We fall
        # back to the visible drop zone's click-to-browse affordance.
        # ``DHM_FORCE_DROP=1`` re-enables the real callback for users
        # on a future Dear PyGui that has the fix.
        force = os.environ.get("DHM_FORCE_DROP") in ("1", "true", "yes")
        self._drop_supported = force or sys.platform != "darwin"

        # Viewport + preview tiers — computed lazily so the state JSON
        # (which may store a previous session's viewport size) wins
        # when valid, otherwise we probe the screen.
        self._viewport_w, self._viewport_h, self._preview_size = \
            self._compute_initial_viewport()

        # Latest results from each pipeline — cached for export,
        # follow-up runs and depth overlay.
        self._last_recon: Optional[ReconResult] = None
        self._last_autofocus: Optional[AutofocusResult] = None
        self._last_qpi: Optional[QPIOneShotResult] = None
        self._last_qpi_batch: Optional[QPIBatchResultWrap] = None
        self._last_depth: Optional[DepthMapResultWrap] = None

        # Mailbox posted from the worker thread, drained in render loop.
        self._pending_lock = threading.Lock()
        self._pending: Optional[object] = None
        self._pending_status: Optional[str] = None

        # Texture tags — populated in _build_ui.
        self._tex_input: Optional[int] = None
        self._tex_amp: Optional[int] = None
        self._tex_phase: Optional[int] = None
        self._blank = _to_rgba(np.zeros((self._preview_size,
                                         self._preview_size),
                                       dtype=np.float32)).flatten()

        # PresetChips helper — real selection state, not a plain row
        # of buttons. Built inside _build_ui once the parent exists.
        # Labels come from the merged built-in + user preset dict so
        # persisted user presets re-hydrate into chips on startup.
        self._preset_chips = PresetChips(
            labels=list(self._presets().keys()),
            on_select=self._apply_preset,
        )

        # None when all four panels are in the 2×2 grid; otherwise the
        # index (0–3) of the panel currently filling the area.
        self._maximized_panel: Optional[int] = None

        # v2.0.9: display-only image flips. Some TIFF writers (SEM,
        # a handful of microscope cameras) store rows bottom-up, so
        # ``tifffile.imread`` returns an array that shows upside-
        # down once DPG applies the image-series coordinate system.
        # Toggles in the View menu let the user correct orientation
        # without touching the pipeline — ``raw_disp`` is the flipped
        # texture feed, ``raw`` stays the scientific data.
        #
        # Defaults are both True (= 180° rotation). Empirically the
        # lab camera + ``tifffile.imread`` + DPG's ``invert=True``
        # plot axis combination produces an image rotated 180° from
        # the operator's expectation; flipping both axes by default
        # matches the physical sample view out-of-box. The operator
        # can still flip either axis off individually from the View
        # menu — the pipeline math is untouched either way.
        ui2_state = self._settings.ui2
        self._flip_display_v: bool = bool(
            getattr(ui2_state, "flip_display_v", True),
        )
        self._flip_display_h: bool = bool(
            getattr(ui2_state, "flip_display_h", True),
        )

        # Interactive line-draw state — None when no gesture active.
        self._line_mode: bool = False
        self._line_p1: Optional[tuple[float, float]] = None
        self._line_p2: Optional[tuple[float, float]] = None
        self._line_handlers_installed: bool = False

        # Camera live-feed state — None when nothing is running. The
        # driver is picked from a registry (synthetic by default,
        # hardware backends opt-in via env var or future dialog).
        self._camera_thread: Optional[AcquisitionThread] = None
        self._camera_source: Optional[CameraSource] = None
        self._camera_recorder: Optional[TiffStackRecorder] = None
        self._camera_fps: float = 0.0
        # v2.1.x: explicit input mode — "file" | "live" | "timelapse".
        # The status bar prefixes a coloured tag matching the mode so
        # the operator can't mistake a stale frame from a stopped
        # camera for a freshly loaded file. ``_latest_live_frame``
        # caches the last camera frame so reconstruct + snapshot
        # actions can grab it without round-tripping through disk.
        self._input_mode: str = "file"
        self._latest_live_frame: Optional[np.ndarray] = None

        # v2.0.5: bounded error history feeding the "View → Errors"
        # drawer. Keeps the last 64 events so a long session doesn't
        # balloon memory; each entry is (iso_ts, level, message).
        self._error_log: list[tuple[str, str, str]] = []

        # Debounced JSON saver — render loop calls tick() per frame,
        # widgets call mark_dirty() on every meaningful edit.
        self._saver = DebouncedSaver(self._snapshot_state, delay=1.5)

    # ---- lifecycle ------------------------------------------------------

    def run(self) -> int:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
        dpg.create_context()
        ScientificTheme.apply(self._theme_name)
        dpg.create_viewport(
            title=self.TITLE,
            width=self._viewport_w,
            height=self._viewport_h,
        )
        dpg.setup_dearpygui()

        self._build_textures()
        self._build_ui()

        # Reflect hydrated state in every widget we just built.
        self._hydrate_widgets()

        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)

        # Wire the viewport resize handler — responsive preview tiers.
        try:
            dpg.set_viewport_resize_callback(self._on_viewport_resize)
        except Exception:
            _LOG.debug("viewport resize callback unavailable", exc_info=True)

        # UI-side crash wrapper — the core handler in run_ui2.py already
        # writes the JSON dump + audit entry. Layer a toast + error
        # drawer append + synchronous state flush on top so the user
        # sees *something* before the interpreter exits or the main
        # loop recovers.
        self._install_ui_crash_wrapper()

        # First-run welcome — cheap JSON flag in the user home dir so
        # we don't pollute QSettings (the v1 app already owns that).
        self._maybe_show_onboarding()

        try:
            while dpg.is_dearpygui_running():
                self._drain_mailbox()
                self._toasts.tick()
                self._saver.tick()
                dpg.render_dearpygui_frame()
        finally:
            # Synchronous final flush so the user never loses a late
            # edit to a debounce timer that didn't fire.
            try:
                self._capture_viewport_geometry()
                self._saver.flush_now()
            except Exception:
                _LOG.exception("final state save failed")
            # Camera thread is daemon so Python would eventually tear
            # it down, but flushing a half-written TIFF matters.
            try:
                self._on_camera_stop()
            except Exception:
                _LOG.debug("camera shutdown raised", exc_info=True)
            self._driver.shutdown()
            self._science.shutdown()
            dpg.destroy_context()
        return 0

    # ---- Responsive viewport sizing ------------------------------------

    def _compute_initial_viewport(self) -> tuple[int, int, int]:
        """Pick an initial viewport size and preview tier.

        Previously-persisted geometry wins when it looks sane; otherwise
        we probe the screen and pick 90% of it, clamped to a sensible
        range. Tiering now honours both width AND height — before
        v2.0.9 we only checked width, so a short viewport (1440×800
        laptop) still picked the 384 tier and ran into scrollbars
        because two 384-tall panels + toolbar + menu + status > 800.
        """
        ui2 = self._settings.ui2
        # Honour stored geometry if it parses as a real-looking window.
        if (ui2.viewport_w >= 1100 and ui2.viewport_h >= 720
                and ui2.viewport_w <= 3840 and ui2.viewport_h <= 2160):
            w, h = int(ui2.viewport_w), int(ui2.viewport_h)
        else:
            sw, sh = self._screen_size()
            # 90% of screen (up from 85% — gives us the extra
            # breathing room the panels need for tier 384) clamped
            # to an on-screen max. Height cap raised from 1000 to
            # 1150 so tier 384 fits without a scrollbar on 2K
            # monitors; still leaves 100+ px for the dock.
            w = max(1100, min(int(sw * 0.90), 1800))
            h = max(720, min(int(sh * 0.90), 1150))
        return w, h, self._tier_for_size(w, h)

    @staticmethod
    def _tier_for_size(width: int, height: int) -> int:
        """Pick a preview tier that fits in BOTH axes.

        Horizontal invariant: sidebar (320) + 2·(preview+16) ≤ width.
        Vertical invariant: menu (28) + toolbar (40) + drop zone
        (48) + 2·(preview+40) + status (28) ≤ height. The +40 per
        panel covers the title strip + border + separator inside
        the image-panel child window.

        ``_tier_for_width`` used to be width-only — a 1440×800
        laptop would pick 384 by width and then overflow by height
        because 2·424 + 144 = 992 > 800. Now we evaluate both axes
        and return the smaller tier.
        """
        # Width-side cap.
        if width >= 1500:
            w_tier = 512
        elif width >= 1250:
            w_tier = 384
        else:
            w_tier = 288
        # Height-side cap — chrome overhead is 144 px, each panel
        # occupies preview+40, and the 2×2 grid stacks two rows.
        chrome = 144
        usable_h = max(0, height - chrome)
        if usable_h >= 2 * (512 + 40):
            h_tier = 512
        elif usable_h >= 2 * (384 + 40):
            h_tier = 384
        else:
            h_tier = 288
        return min(w_tier, h_tier)

    @classmethod
    def _tier_for_width(cls, width: int) -> int:
        """Width-only legacy path — callers that don't know the
        height (tests, resize callbacks before geometry settles)
        fall back to this. Prefer :meth:`_tier_for_size` everywhere
        you have both dimensions."""
        return cls._tier_for_size(width, 9999)

    @staticmethod
    def _screen_size() -> tuple[int, int]:
        """Best-effort screen dimensions without pulling a GUI toolkit.

        Tk is in the stdlib on every platform we care about; withdraw()
        hides the root window so macOS Cocoa never shows a flicker, and
        destroy() releases it before Dear PyGui takes over."""
        try:
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            size = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()
            return size
        except Exception:
            return (1440, 900)

    def _on_viewport_resize(self, sender=None, app_data=None) -> None:
        """Fired by Dear PyGui on viewport resize. Retier panels if we
        crossed a width boundary; idempotent within a tier so dragging
        doesn't thrash the texture allocation."""
        try:
            w = int(dpg.get_viewport_client_width())
            h = int(dpg.get_viewport_client_height())
        except Exception:
            return
        self._viewport_w = w
        self._viewport_h = h
        new_preview = self._tier_for_size(w, h)
        if new_preview != self._preview_size:
            self._preview_size = new_preview
            for panel in (self.panel_input, self.panel_amp,
                          self.panel_phase):
                try:
                    panel.resize(new_preview)
                except Exception:
                    _LOG.debug("panel resize failed", exc_info=True)
            # Rehydrate the last recon so the user doesn't stare at a
            # blank panel after a resize-triggered texture rebuild.
            if self._last_recon is not None:
                try:
                    self.panel_input.set_image(self._last_recon.input_image)
                    self.panel_amp.set_image(self._last_recon.amplitude)
                    self.panel_phase.set_image(self._last_recon.phase)
                except Exception:
                    _LOG.debug("repaint after resize failed", exc_info=True)
        self._mark_dirty()

    def _capture_viewport_geometry(self) -> None:
        """Stash current viewport dims into the saver snapshot."""
        try:
            self._viewport_w = int(dpg.get_viewport_client_width())
            self._viewport_h = int(dpg.get_viewport_client_height())
        except Exception:
            pass

    # ---- State persistence --------------------------------------------

    def _mark_dirty(self) -> None:
        """Any hydrated widget's callback should call this after
        mutating the live state. Coalesced on a background thread."""
        try:
            self._saver.mark_dirty()
        except Exception:
            _LOG.debug("mark_dirty failed", exc_info=True)

    def _snapshot_state(self) -> AppSettings:
        """Fresh AppSettings reflecting the live UI — saver consumes
        this on every flush."""
        ref_path = (str(self._params.reference_path)
                    if self._params.reference_path else "")
        ui2 = Ui2State(
            viewport_w=int(self._viewport_w),
            viewport_h=int(self._viewport_h),
            theme=str(self._theme_name),
            sample_id=str(self._sample_id),
            workflow_mode=str(self._workflow_mode),
            selected_preset=str(self._selected_preset),
            recent=[str(p) for p in self._recent],
            last_dir=str(self._last_dir),
            last_hologram=(str(self._current_hologram)
                           if self._current_hologram else ""),
            reference_path=ref_path,
            subtract_reference=bool(self._params.subtract_reference),
            wavelength_nm=float(self._params.wavelength_nm),
            pixel_um=float(self._params.pixel_um),
            z_mm=float(self._params.z_mm),
            mask_radius=int(self._params.mask_radius),
            method=str(self._params.method),
            magnification=float(self._params.magnification),
            pixel_is_effective=bool(self._params.pixel_is_effective),
            n_sample=float(self._params.n_sample),
            n_medium=float(self._params.n_medium),
            autofocus_metric=str(self._params.autofocus_metric),
            af_z_min_mm=float(self._params.af_z_min_mm),
            af_z_max_mm=float(self._params.af_z_max_mm),
            af_n_steps=int(self._params.af_n_steps),
            subtract_mean=bool(self._params.subtract_mean),
            hann_window=bool(self._params.hann_window),
            fft_backend=str(self._params.fft_backend),
            unwrap_method=str(self._params.unwrap_method),
            optical_mode=str(self._params.optical_mode),
            auto_contrast_amplitude=bool(self._params.auto_contrast_amplitude),
            af_algorithm=str(self._params.af_algorithm),
            flip_display_v=bool(self._flip_display_v),
            flip_display_h=bool(self._flip_display_h),
        )
        return self._settings.with_ui2(**{
            k: getattr(ui2, k) for k in ui2.__dataclass_fields__
        })

    def _hydrate_widgets(self) -> None:
        """Copy the loaded state onto the freshly-built widgets.

        Runs once after :meth:`_build_ui`. Guarded by ``does_item_exist``
        because the command-palette-only build path (used by tests)
        doesn't create every widget."""
        setters = [
            ("input_wavelength", float(self._params.wavelength_nm)),
            ("input_pixel", float(self._params.pixel_um)),
            ("input_z", float(self._params.z_mm)),
            ("input_mask", int(self._params.mask_radius)),
            ("input_method", str(self._params.method)),
            ("sample_id_input", str(self._sample_id)),
            ("workflow_mode_combo", str(self._workflow_mode)),
            ("subtract_reference_cb", bool(self._params.subtract_reference)),
            ("input_magnification", float(self._params.magnification)),
            ("input_pixel_is_effective",
             bool(self._params.pixel_is_effective)),
            ("input_n_sample", float(self._params.n_sample)),
            ("input_n_medium", float(self._params.n_medium)),
            ("input_af_metric", str(self._params.autofocus_metric)),
            ("input_af_z_min", float(self._params.af_z_min_mm)),
            ("input_af_z_max", float(self._params.af_z_max_mm)),
            ("input_af_n_steps", int(self._params.af_n_steps)),
            ("input_af_algorithm", str(self._params.af_algorithm)),
            ("input_subtract_mean", bool(self._params.subtract_mean)),
            ("input_hann_window", bool(self._params.hann_window)),
            ("input_fft_backend", str(self._params.fft_backend)),
            ("input_unwrap_method", str(self._params.unwrap_method)),
            ("input_optical_mode", str(self._params.optical_mode)),
            ("input_auto_contrast_amp",
             bool(self._params.auto_contrast_amplitude)),
        ]
        for tag, value in setters:
            if dpg.does_item_exist(tag):
                try:
                    dpg.set_value(tag, value)
                except Exception:
                    _LOG.debug("hydrate set_value %s failed", tag,
                               exc_info=True)

        # Reference label + colour.
        if self._params.reference_path is not None and dpg.does_item_exist(
                "reference_path_label"):
            try:
                dpg.set_value("reference_path_label",
                              f"📎 {self._params.reference_path.name}")
                dpg.configure_item("reference_path_label",
                                   color=ScientificTheme.SUCCESS)
            except Exception:
                pass

        # Preset chip state — if a preset was chosen last session.
        if self._selected_preset:
            try:
                self._preset_chips.set_active(self._selected_preset)
            except Exception:
                pass

        # Recent menu.
        try:
            self._rebuild_recent_menu()
        except Exception:
            _LOG.debug("recent menu rebuild failed", exc_info=True)

        # Workflow mode → section visibility.
        try:
            self._apply_workflow_visibility(self._workflow_mode)
        except Exception:
            _LOG.debug("workflow visibility failed", exc_info=True)

        # Reflect the loaded af_algorithm on the step / z-range
        # widgets (label + enabled state) so nothing is mislabelled
        # on first paint.
        try:
            self._apply_af_algorithm_ux()
        except Exception:
            _LOG.debug("af algorithm ux init failed", exc_info=True)

    # ---- UI construction -----------------------------------------------

    def _build_textures(self) -> None:
        """Build zoomable image panels backed by dynamic textures.

        We pre-create three ``ZoomableImagePanel`` instances and
        register their textures inside a single ``texture_registry``
        context. The panels themselves lay out in ``_build_ui``."""
        size = self._preview_size
        self.panel_input = ZoomableImagePanel("Input",
                                              size=size,
                                              colormap="gray")
        self.panel_amp = ZoomableImagePanel("Amplitude",
                                            size=size,
                                            colormap="gray")
        self.panel_phase = ZoomableImagePanel("Phase (wrapped)",
                                              size=size,
                                              colormap="phase")
        with dpg.texture_registry(show=False):
            self.panel_input.register_texture()
            self.panel_amp.register_texture()
            self.panel_phase.register_texture()
        # Legacy tag aliases so existing ``dpg.set_value("tex_input", …)``
        # paths keep working — point at the actual texture tags.
        self._tex_input = self.panel_input.tex_tag
        self._tex_amp = self.panel_amp.tex_tag
        self._tex_phase = self.panel_phase.tex_tag

    # ---- Shortcut labels (Mac-aware) ----------------------------------
    @staticmethod
    def _shortcut(keys: str) -> str:
        """Return the OS-appropriate modifier spelling for ``keys``.

        ``keys`` is the Ctrl form (e.g. ``"Ctrl+O"``); on macOS we swap
        Ctrl for ⌘. Anywhere this is used, the human-readable menu label
        should be the *only* shortcut shown — picking one per platform
        is less confusing than "Ctrl+O / ⌘O" as in v2.0.1."""
        if sys.platform == "darwin":
            return keys.replace("Ctrl+", "⌘")
        return keys

    def _build_ui(self) -> None:
        sc = self._shortcut
        # Menu bar lives *inside* the primary window (``menubar=True``)
        # rather than at viewport level. ``dpg.viewport_menu_bar`` on
        # macOS routes to the OS menu bar, which is only visible when
        # the Python process is frontmost AND bundled as a proper
        # ``.app`` — which we aren't. The window-level menu bar shows
        # up right under the title strip regardless of bundling, so
        # users can always reach File/View/Tools/Help.
        with dpg.window(tag="primary_window",
                        menubar=True,
                        no_title_bar=True, no_move=True,
                        no_resize=True, no_collapse=True,
                        no_scrollbar=True):
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(
                        label=f"Load hologram…     {sc('Ctrl+O')}",
                        callback=self._on_load_clicked,
                    )
                    with dpg.menu(label="Recent", tag="menu_recent"):
                        dpg.add_menu_item(
                            label="(empty)", enabled=False,
                            tag="menu_recent_empty",
                        )
                    dpg.add_menu_item(label="Load reference hologram…",
                                      callback=self._on_load_reference)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Quit",
                                      callback=lambda: dpg.stop_dearpygui())

                with dpg.menu(label="View"):
                    with dpg.menu(label="Theme"):
                        for key in PALETTES.keys():
                            label = key.replace("_", " ").title()
                            dpg.add_menu_item(
                                label=label,
                                callback=lambda s, a, u=key: self._apply_theme(u),
                            )
                    dpg.add_menu_item(
                        label=f"Command palette…     {sc('Ctrl+K')}",
                        callback=self._palette.show,
                    )
                    dpg.add_separator()
                    # v2.0.9: display-only flips for cameras /
                    # scanners whose TIFF rows are stored bottom-up.
                    # Toggles take effect on the next hologram load;
                    # scientific data is never flipped — preview only.
                    dpg.add_menu_item(
                        label="Flip display vertically",
                        check=True,
                        default_value=self._flip_display_v,
                        tag="menu_flip_v",
                        callback=lambda s, a: self._on_flip_toggle(
                            vertical=True, value=bool(a),
                        ),
                    )
                    dpg.add_menu_item(
                        label="Flip display horizontally",
                        check=True,
                        default_value=self._flip_display_h,
                        tag="menu_flip_h",
                        callback=lambda s, a: self._on_flip_toggle(
                            vertical=False, value=bool(a),
                        ),
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(label="Errors & warnings…",
                                      callback=self._show_error_drawer)
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label=f"Maximize Input       {sc('Ctrl+1')}",
                        callback=lambda: self._maximize_panel(0),
                    )
                    dpg.add_menu_item(
                        label=f"Maximize Amplitude   {sc('Ctrl+2')}",
                        callback=lambda: self._maximize_panel(1),
                    )
                    dpg.add_menu_item(
                        label=f"Maximize Phase       {sc('Ctrl+3')}",
                        callback=lambda: self._maximize_panel(2),
                    )
                    dpg.add_menu_item(
                        label=f"Maximize Info        {sc('Ctrl+4')}",
                        callback=lambda: self._maximize_panel(3),
                    )
                    dpg.add_menu_item(
                        label=f"Restore grid         {sc('Ctrl+0')}",
                        callback=self._restore_panel_grid,
                    )

                with dpg.menu(label="Tools"):
                    dpg.add_menu_item(
                        label=f"Reconstruct     {sc('Ctrl+R')}",
                        callback=self._on_reconstruct,
                    )
                    dpg.add_menu_item(
                        label="Autofocus (one-shot)",
                        callback=self._on_autofocus,
                    )
                    dpg.add_menu_item(
                        label="Find multiple focus planes…",
                        callback=self._on_find_focus_candidates,
                    )
                    dpg.add_menu_item(
                        label="Autofocus ROI…",
                        callback=self._show_roi_dialog,
                    )
                    dpg.add_menu_item(
                        label="Clear autofocus ROI",
                        callback=self._clear_af_roi,
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Compute QPI",
                        callback=self._on_compute_qpi,
                    )
                    dpg.add_menu_item(
                        label="Run QPI batch for focus candidates…",
                        callback=self._on_qpi_batch,
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Compute depth map + overlay",
                        callback=self._on_depth_map,
                    )
                    dpg.add_menu_item(
                        label="Clear depth overlay",
                        callback=self._clear_depth_overlay,
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Export HTML report…",
                        callback=self._on_export_report,
                    )
                    dpg.add_menu_item(
                        label="Export QPI CSV…",
                        callback=self._on_export_qpi_csv,
                    )
                    dpg.add_menu_item(
                        label="Export tomography bundle…",
                        callback=self._on_export_bundle,
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Phase line profile (center row)",
                        callback=self._show_line_profile,
                    )
                    dpg.add_menu_item(
                        label="Phase line profile (draw…)",
                        callback=self._enable_line_profile_mode,
                    )
                    dpg.add_menu_item(
                        label="Batch reconstruct directory…",
                        callback=self._on_batch_reconstruct,
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Camera live feed (start)",
                        callback=self._on_camera_start,
                    )
                    dpg.add_menu_item(
                        label="Camera live feed (stop)",
                        callback=self._on_camera_stop,
                    )
                    dpg.add_menu_item(
                        label="Record camera to TIFF stack…",
                        callback=self._on_camera_record,
                    )

                with dpg.menu(label="Help"):
                    dpg.add_menu_item(label="Contextual help (?)",
                                      callback=self._on_help_overlay)
                    dpg.add_menu_item(label="Keyboard shortcuts",
                                      callback=self._show_shortcuts)
                    dpg.add_menu_item(label="Onboarding",
                                      callback=self._show_onboarding_manual)
                    dpg.add_menu_item(label="Reset first-run onboarding",
                                      callback=self._reset_onboarding_flag)
                    dpg.add_separator()
                    # v2.0.7 T5 — audit log browser. Sven's compliance
                    # ask for a usable view on top of the JSONL log.
                    dpg.add_menu_item(label="Show audit log…",
                                      callback=self._show_audit_viewer)
                    dpg.add_separator()
                    dpg.add_menu_item(label=f"State file: {STATE_PATH}",
                                      enabled=False)
                    dpg.add_menu_item(label="About",
                                      callback=self._show_about)
            # Top area grows; the status line sits inside the primary
            # window rather than as a separate QMainWindow-style bar
            # so Dear PyGui's resize semantics can handle everything
            # automatically (``height=-28`` is "parent minus 28 px").
            # ``no_scrollbar`` was True in v2.0.1 — that's the bug that
            # ate content on 1366×768 laptops. Letting it default to
            # False gives us a safety-net scrollbar the responsive tier
            # logic normally prevents from appearing at all.
            with dpg.child_window(tag="content_row",
                                  autosize_x=True, height=-28,
                                  border=False):
              with dpg.group(horizontal=True):
                # ── Sidebar ──────────────────────────────────────────
                with dpg.child_window(width=320, autosize_y=True,
                                      border=True, tag="sidebar"):
                    # Workflow mode — filters which sidebar sections show.
                    with dpg.group(tag="section_workflow"):
                        dpg.add_text("Workflow",
                                     color=ScientificTheme.TEXT_MUTED)
                        dpg.add_combo(
                            items=["Reconstruct", "Analyse", "Report"],
                            default_value=self._workflow_mode, width=-1,
                            tag="workflow_mode_combo",
                            callback=self._on_workflow_changed,
                        )
                        dpg.add_spacer(height=6)
                        dpg.add_separator()

                    with dpg.group(tag="section_sample"):
                        dpg.add_text("Sample",
                                     color=ScientificTheme.TEXT_MUTED)
                        dpg.add_input_text(
                            hint="Sample ID (optional)",
                            width=-1, tag="sample_id_input",
                            default_value=self._sample_id,
                            callback=self._on_sample_id_changed,
                        )
                        dpg.add_spacer(height=4)
                        dpg.add_separator()

                    # Preset chips row — exclusive selection drives
                    # the parameter defaults below. Built via
                    # :class:`PresetChips` so selection state is
                    # visible (selected chip appears disabled-looking).
                    with dpg.group(tag="section_preset"):
                        dpg.add_text("Preset",
                                     color=ScientificTheme.TEXT_MUTED)
                        self._preset_chips.build(dpg.last_item())
                        with dpg.group(horizontal=True):
                            dpg.add_button(
                                label="Save preset…",
                                callback=lambda: self._open_save_preset_dialog(),
                            )
                            dpg.add_button(
                                label="Delete…",
                                callback=lambda: self._open_delete_preset_dialog(),
                            )
                        dpg.add_spacer(height=4)
                        dpg.add_separator()

                    with dpg.group(tag="section_params"):
                        dpg.add_text("Parameters",
                                     color=ScientificTheme.TEXT_MUTED)
                        dpg.add_input_float(
                            label="λ (nm)",
                            default_value=self._params.wavelength_nm,
                            min_value=100.0, max_value=2000.0,
                            min_clamped=True, max_clamped=True,
                            step=10.0, format="%.2f", width=160,
                            tag="input_wavelength",
                            callback=self._on_param_changed,
                        )
                        dpg.add_input_float(
                            label="px (µm)",
                            default_value=self._params.pixel_um,
                            min_value=0.01, max_value=100.0,
                            min_clamped=True, max_clamped=True,
                            step=0.1, format="%.3f", width=160,
                            tag="input_pixel",
                            callback=self._on_param_changed,
                        )
                        # v2.0.3: objective magnification — the v1 field
                        # that never made it into v2.0.2. Combined with
                        # the checkbox below, decides whether the px
                        # value above is treated as camera or sample-
                        # plane pixel size.
                        dpg.add_input_float(
                            label="×M",
                            default_value=self._params.magnification,
                            min_value=0.01, max_value=1000.0,
                            min_clamped=True, max_clamped=True,
                            step=1.0, format="%.2f", width=160,
                            tag="input_magnification",
                            callback=self._on_param_changed,
                        )
                        with dpg.tooltip("input_magnification"):
                            dpg.add_text(
                                "Objective magnification. Effective "
                                "pixel at the sample plane = "
                                "camera_pixel / M when the checkbox "
                                "below is unchecked.",
                                wrap=320,
                            )
                        dpg.add_checkbox(
                            label="Pixel is already effective",
                            default_value=self._params.pixel_is_effective,
                            tag="input_pixel_is_effective",
                            callback=self._on_param_changed,
                        )
                        with dpg.tooltip("input_pixel_is_effective"):
                            dpg.add_text(
                                "Check when the px value above is "
                                "already divided by magnification "
                                "(e.g. you pre-computed it, or the "
                                "setup has no objective).",
                                wrap=320,
                            )
                        dpg.add_input_float(
                            label="z (mm)",
                            default_value=self._params.z_mm,
                            min_value=-1000.0, max_value=1000.0,
                            step=0.5, format="%.3f", width=160,
                            tag="input_z",
                            callback=self._on_param_changed,
                        )
                        dpg.add_input_int(
                            label="mask r (px)",
                            default_value=self._params.mask_radius,
                            min_value=5, max_value=500,
                            min_clamped=True, max_clamped=True,
                            step=5, width=160,
                            tag="input_mask",
                            callback=self._on_param_changed,
                        )
                        dpg.add_combo(
                            items=["ASM", "Fresnel"],
                            default_value=self._params.method,
                            label="method", width=160,
                            tag="input_method",
                            callback=self._on_param_changed,
                        )

                        # Autofocus metric + range + QPI refractive
                        # indices were visible by default in v2.0.5 —
                        # user feedback (2026-04-24): "sidebar dolu,
                        # kafa karıştırıyor". Moved into the Advanced
                        # collapsing block below so the *everyday*
                        # physics (λ / px / z / M / mask / method)
                        # stays inline and everything else is one
                        # click away.
                        dpg.add_spacer(height=6)
                        dpg.add_separator()
                        dpg.add_spacer(height=4)

                    with dpg.group(tag="section_actions"):
                        dpg.add_button(
                            label="Load hologram…",
                            width=-1, height=30,
                            callback=self._on_load_clicked,
                        )
                        dpg.add_button(
                            label="Reconstruct",
                            width=-1, height=34,
                            tag="btn_reconstruct",
                            enabled=False,
                            callback=self._on_reconstruct,
                        )
                        # Tooltip — disabled-reason visible at a hover,
                        # so the user never has to guess why the button
                        # won't fire. Rebound in _load_hologram.
                        with dpg.tooltip("btn_reconstruct",
                                         tag="btn_reconstruct_tip"):
                            dpg.add_text(
                                "Load a hologram first "
                                "(File → Load hologram…).",
                                tag="btn_reconstruct_tip_text",
                            )

                    with dpg.group(tag="section_reference"):
                        dpg.add_spacer(height=6)
                        dpg.add_separator()
                        dpg.add_text("Reference",
                                     color=ScientificTheme.TEXT_MUTED)
                        dpg.add_text("(none)", tag="reference_path_label",
                                     color=ScientificTheme.TEXT_MUTED,
                                     wrap=280)
                        with dpg.group(horizontal=True):
                            dpg.add_button(
                                label="Load ref…",
                                callback=self._on_load_reference,
                            )
                            dpg.add_button(
                                label="Clear",
                                callback=self._on_clear_reference,
                            )
                        dpg.add_checkbox(
                            label="Subtract reference",
                            tag="subtract_reference_cb",
                            callback=lambda s, a: self._on_ref_toggle(a),
                        )

                    # ── Advanced — collapsed by default ─────────────
                    # Everything non-essential lives here so the
                    # main sidebar stays breathable. Expand once to
                    # set per-session values; they persist in state.
                    with dpg.group(tag="section_advanced"):
                        dpg.add_spacer(height=6)
                        dpg.add_separator()
                        with dpg.collapsing_header(
                            label="Advanced",
                            default_open=False,
                            tag="adv_header",
                        ):
                            # Optical path — transmission halves OPD
                            # relative to reflection (single vs double
                            # pass). Picking the wrong mode is a
                            # classic "my cell height is 2× off" bug.
                            dpg.add_text("Optical path",
                                         color=ScientificTheme.TEXT_MUTED)
                            dpg.add_combo(
                                items=["transmission", "reflection"],
                                default_value=self._params.optical_mode,
                                label="mode", width=160,
                                tag="input_optical_mode",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_optical_mode"):
                                dpg.add_text(
                                    "Transmission: light passes through "
                                    "the sample once. Reflection: light "
                                    "traverses the surface height twice "
                                    "— OPD is halved before height "
                                    "conversion (h = OPD/2 in addition "
                                    "to the Δn factor).",
                                    wrap=320,
                                )
                            dpg.add_spacer(height=4)
                            dpg.add_separator()

                            # Autofocus controls — relocated here so
                            # the everyday sidebar stays minimal.
                            dpg.add_text("Autofocus",
                                         color=ScientificTheme.TEXT_MUTED)
                            dpg.add_combo(
                                items=_available_focus_metric_names(),
                                default_value=self._params.autofocus_metric,
                                label="metric", width=160,
                                tag="input_af_metric",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_af_metric"):
                                dpg.add_text(
                                    "Metric used by one-shot and multi-"
                                    "focus scans. Low-contrast bio "
                                    "samples typically work best with "
                                    "gradient metrics; Fresnel-ring-"
                                    "dominated samples prefer "
                                    "LAPLACIAN_VARIANCE.",
                                    wrap=320,
                                )
                            # v2.0.8: autofocus algorithm combo —
                            # restores v1's adaptive/coarse-to-fine
                            # options. ``zscan`` = fixed linear sweep
                            # (v2.0.2 default). Adaptive variants
                            # spend the same step budget more wisely:
                            # large steps in flat regions, tiny steps
                            # near the peak — better z resolution at
                            # the same cost.
                            dpg.add_combo(
                                items=_available_autofocus_algorithms(),
                                default_value=self._params.af_algorithm,
                                label="algorithm", width=160,
                                tag="input_af_algorithm",
                                callback=self._on_af_algorithm_changed,
                            )
                            with dpg.tooltip("input_af_algorithm"):
                                dpg.add_text(
                                    "zscan: linear sweep, uniform "
                                    "grid. "
                                    "coarse_to_fine: coarse grid + "
                                    "Golden-Section refinement. "
                                    "robust: coarse scan smoothed "
                                    "for noisy landscapes. "
                                    "adaptive_gradient: big steps "
                                    "in flat regions, small near "
                                    "the peak — fastest to sub-"
                                    "step accuracy (steps = max "
                                    "evals budget). "
                                    "adaptive_bracketing: nested "
                                    "bracketing — best when you "
                                    "know roughly where focus is "
                                    "(steps = max evals). "
                                    "adaptive_distance: auto-"
                                    "discovers the z range from "
                                    "z = 0 outwards; z min/max "
                                    "widgets are ignored when "
                                    "picked.",
                                    wrap=380,
                                )
                            dpg.add_input_float(
                                label="z min (mm)",
                                default_value=self._params.af_z_min_mm,
                                step=0.5, format="%.2f", width=160,
                                tag="input_af_z_min",
                                callback=self._on_param_changed,
                            )
                            dpg.add_input_float(
                                label="z max (mm)",
                                default_value=self._params.af_z_max_mm,
                                step=0.5, format="%.2f", width=160,
                                tag="input_af_z_max",
                                callback=self._on_param_changed,
                            )
                            dpg.add_input_int(
                                label="steps",
                                default_value=self._params.af_n_steps,
                                min_value=5, max_value=500,
                                min_clamped=True, max_clamped=True,
                                step=5, width=160,
                                tag="input_af_n_steps",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_af_n_steps"):
                                # v2.0.8: tooltip text changes with
                                # the algorithm picker — different
                                # algorithms consume this field
                                # differently (grid count vs budget).
                                dpg.add_text(
                                    "Autofocus scan sample count. "
                                    "More steps = finer z "
                                    "resolution; runtime scales "
                                    "linearly. 40 is a good "
                                    "default.",
                                    tag="input_af_n_steps_tip_text",
                                    wrap=320,
                                )
                            dpg.add_spacer(height=4)
                            dpg.add_separator()

                            # QPI refractive indices.
                            dpg.add_text("QPI refractive indices",
                                         color=ScientificTheme.TEXT_MUTED)
                            dpg.add_input_float(
                                label="n sample",
                                default_value=self._params.n_sample,
                                min_value=1.0, max_value=3.0,
                                min_clamped=True, max_clamped=True,
                                step=0.01, format="%.3f", width=160,
                                tag="input_n_sample",
                                callback=self._on_param_changed,
                            )
                            dpg.add_input_float(
                                label="n medium",
                                default_value=self._params.n_medium,
                                min_value=1.0, max_value=3.0,
                                min_clamped=True, max_clamped=True,
                                step=0.01, format="%.3f", width=160,
                                tag="input_n_medium",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_n_medium"):
                                dpg.add_text(
                                    "Refractive index of the "
                                    "surrounding medium (water ≈ "
                                    "1.337, air ≈ 1.000). Drives QPI "
                                    "cell-height and dry-mass "
                                    "calculations.",
                                    wrap=320,
                                )
                            dpg.add_spacer(height=4)
                            dpg.add_separator()

                            # Pre/post-processing + engine selection.
                            dpg.add_text("Pipeline",
                                         color=ScientificTheme.TEXT_MUTED)
                            dpg.add_checkbox(
                                label="Subtract mean",
                                default_value=self._params.subtract_mean,
                                tag="input_subtract_mean",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_subtract_mean"):
                                dpg.add_text(
                                    "Remove DC bias from the hologram "
                                    "before off-axis demodulation. "
                                    "Usually on — reduces 0-order "
                                    "bleed.",
                                    wrap=320,
                                )
                            dpg.add_checkbox(
                                label="Hann window",
                                default_value=self._params.hann_window,
                                tag="input_hann_window",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_hann_window"):
                                dpg.add_text(
                                    "Taper the raw frame with a 2D "
                                    "Hann envelope so edge "
                                    "discontinuities don't leak into "
                                    "the FFT. Off by default (cost: "
                                    "mild edge dimming).",
                                    wrap=320,
                                )
                            dpg.add_checkbox(
                                label="Auto-contrast amplitude",
                                default_value=self._params.auto_contrast_amplitude,
                                tag="input_auto_contrast_amp",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_auto_contrast_amp"):
                                dpg.add_text(
                                    "Percentile-based stretch on the "
                                    "amplitude panel (1–99%). Keeps "
                                    "reference-divided images "
                                    "readable when outliers dominate "
                                    "the histogram. Does not alter "
                                    "the scientific values, only the "
                                    "on-screen preview.",
                                    wrap=320,
                                )
                            dpg.add_combo(
                                items=["auto", "pyfftw", "mlx",
                                       "scipy", "numpy"],
                                default_value=self._params.fft_backend,
                                label="FFT backend", width=160,
                                tag="input_fft_backend",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_fft_backend"):
                                dpg.add_text(
                                    "‘auto’ picks the best available "
                                    "(MLX on Apple Silicon, PyFFTW "
                                    "elsewhere). Override for "
                                    "debugging.",
                                    wrap=320,
                                )
                            dpg.add_combo(
                                items=[
                                    "GRADIENT_INTEGRATION", "TIE",
                                    "THIN_SAMPLE", "OPTICAL",
                                    "LEAST_SQUARES", "QUALITY_GUIDED",
                                    "GOLDSTEIN",
                                ],
                                default_value=self._params.unwrap_method,
                                label="Unwrap", width=160,
                                tag="input_unwrap_method",
                                callback=self._on_param_changed,
                            )
                            with dpg.tooltip("input_unwrap_method"):
                                dpg.add_text(
                                    "Phase-unwrap algorithm for QPI. "
                                    "Gradient integration is robust "
                                    "for most samples; "
                                    "QUALITY_GUIDED helps on noisy "
                                    "wraps; TIE needs dz set.",
                                    wrap=320,
                                )

                # ── Panels grid ──────────────────────────────────────
                with dpg.child_window(autosize_x=True, autosize_y=True,
                                      border=False):
                    # Quick-access toolbar — sits above the four panels
                    # so the most-used actions are one click away.
                    self._build_quick_toolbar()
                    dpg.add_spacer(height=4)

                    # Drop-zone strip — always visible, clickable on
                    # macOS where the real viewport-drop callback is
                    # a no-op. On other platforms it's an overt hint
                    # that drag-and-drop is supported. Dear PyGui's
                    # ``mvClickedHandler`` isn't applicable to
                    # ``mvChildWindow``; the click handler lives on
                    # the global registry further down and gates on
                    # ``dpg.is_item_hovered("drop_zone")``.
                    with dpg.child_window(
                        tag="drop_zone",
                        height=self._DROP_ZONE_HEIGHT,
                        autosize_x=True,
                        border=True,
                    ):
                        with dpg.group(horizontal=True):
                            dpg.add_text("  ⬇  ",
                                         color=ScientificTheme.ACCENT)
                            dpg.add_text(self._drop_zone_label(),
                                         tag="drop_zone_label",
                                         color=ScientificTheme.TEXT_MUTED)
                    dpg.add_spacer(height=6)

                    with dpg.group(horizontal=True, tag="panel_row_top"):
                        self.panel_input.build("panel_row_top")
                        self.panel_amp.build("panel_row_top")
                    with dpg.group(horizontal=True, tag="panel_row_bot"):
                        self.panel_phase.build("panel_row_bot")
                        with dpg.child_window(
                                width=self._preview_size + 16,
                                height=self._preview_size + 40,
                                border=True,
                                tag="info_panel"):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Info")
                                dpg.add_spacer(width=12)
                                dpg.add_button(
                                    label="Open phase in 3D",
                                    callback=self._on_open_surface,
                                    enabled=False,
                                    tag="btn_surface",
                                )
                                dpg.add_button(
                                    label="Reset zoom",
                                    callback=self._reset_all_zoom,
                                )
                            dpg.add_separator()
                            dpg.add_text("No reconstruction yet.",
                                         tag="info_text",
                                         wrap=self._preview_size)

            # Status line — inside primary_window, always visible at
            # the bottom thanks to the ``height=-28`` reservation on
            # content_row above.
            dpg.add_separator()
            dpg.add_text(self._status_ready_text(),
                         tag="status_text",
                         color=ScientificTheme.TEXT_MUTED)

        # File-open dialog (prebuilt, shown on demand).
        with dpg.file_dialog(
            directory_selector=False, show=False,
            callback=self._on_file_selected,
            tag="file_dialog_open",
            width=720, height=480,
            default_path=self._last_dir or str(Path.home()),
        ):
            dpg.add_file_extension("Images (*.tif *.tiff *.png *.bmp *.jpg){.tif,.tiff,.png,.bmp,.jpg,.jpeg}")
            dpg.add_file_extension(".*")

        # ── Keyboard shortcuts ────────────────────────────────────────
        with dpg.handler_registry():
            dpg.add_key_press_handler(
                dpg.mvKey_O, callback=self._on_key_o,
            )
            dpg.add_key_press_handler(
                dpg.mvKey_R, callback=self._on_key_r,
            )
            dpg.add_key_press_handler(
                dpg.mvKey_K, callback=self._on_key_k,
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Slash, callback=self._on_key_slash,
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Escape,
                callback=self._on_key_escape,
            )
            # Ctrl+1/2/3/4 → maximize individual panel; Ctrl+0 → restore grid.
            # Matches v1 quick-switch semantics so muscle memory carries over.
            for key_const, idx in (
                (dpg.mvKey_1, 0), (dpg.mvKey_2, 1),
                (dpg.mvKey_3, 2), (dpg.mvKey_4, 3),
            ):
                dpg.add_key_press_handler(
                    key_const,
                    callback=lambda s, a, u=idx: self._on_key_number(u),
                )
            dpg.add_key_press_handler(
                dpg.mvKey_0,
                callback=lambda s, a: self._on_key_zero(),
            )
            # Drop-zone click — Dear PyGui's ``mvClickedHandler`` only
            # applies to a handful of widget types (not child_window),
            # so we hook a global left-click handler and gate on
            # ``is_item_hovered``. Matches the visual affordance the
            # drop-zone strip promises on macOS where the real
            # viewport-drop callback is a no-op.
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=lambda s, a: self._on_drop_zone_click(),
            )

        # ── Command palette + registry --------------------------------
        self._palette.build()
        self._register_commands()

        # ── Drag-and-drop — viewport-level ────────────────────────────
        # Only bind when the platform actually emits drop events.
        # See __init__: macOS 2.x reports the symbol but never fires.
        if self._drop_supported:
            try:
                dpg.set_viewport_drop_callback(self._on_viewport_drop)
            except Exception:
                # Dear PyGui too old to expose the API — degrade to
                # click-to-browse via the drop zone.
                self._drop_supported = False
                _LOG.debug("drop callback unavailable", exc_info=True)
                try:
                    dpg.set_value("drop_zone_label",
                                  self._drop_zone_label())
                    dpg.set_value("status_text", self._status_ready_text())
                except Exception:
                    pass

    # _panel legacy helper removed — panels now build themselves via
    # ZoomableImagePanel.build() in _build_ui.

    # ---- callbacks ------------------------------------------------------

    # ---- keyboard shortcuts --------------------------------------------
    def _on_key_o(self, sender, app_data, user_data=None) -> None:
        if self._modifier_down():
            self._on_load_clicked()

    def _on_key_r(self, sender, app_data, user_data=None) -> None:
        if self._modifier_down():
            self._on_reconstruct()

    def _on_key_k(self, sender, app_data, user_data=None) -> None:
        if self._modifier_down():
            self._palette.show()

    def _on_key_number(self, idx: int) -> None:
        """Ctrl+N maximizes panel index N (1=Input, 2=Amplitude,
        3=Phase, 4=Info). Modifier gate mirrors v1's QShortcut filter."""
        if not self._modifier_down():
            return
        self._maximize_panel(idx)

    def _on_key_zero(self) -> None:
        if not self._modifier_down():
            return
        self._restore_panel_grid()

    # ---- Maximize / restore panels ------------------------------------

    def _panel_tags(self) -> list[str]:
        """Wrapping child_window tags in Ctrl+1-4 order. Index 3 is
        the info pane, which doesn't own a ZoomableImagePanel — we
        hardcode its tag so the same maximize flow drives all four."""
        return [
            getattr(self.panel_input, "container_tag", ""),
            getattr(self.panel_amp, "container_tag", ""),
            getattr(self.panel_phase, "container_tag", ""),
            "info_panel",
        ]

    def _maximize_panel(self, idx: int) -> None:
        """Hide the other three panels so the selected one fills the
        grid area. Re-apply the per-panel autosize so it expands to
        fill its row when the sibling's width frees up."""
        tags = self._panel_tags()
        if not (0 <= idx < len(tags)):
            return
        self._maximized_panel = idx
        for i, tag in enumerate(tags):
            if not tag:
                continue
            try:
                dpg.configure_item(tag, show=(i == idx))
            except Exception:
                pass
        # Grow the visible panel to fill what it can.
        visible_tag = tags[idx]
        if visible_tag:
            try:
                dpg.configure_item(visible_tag, width=-1, height=-1)
            except Exception:
                pass
        name_map = ["Input", "Amplitude", "Phase", "Info"]
        self._set_status(f"Maximized: {name_map[idx]}  "
                         f"(Ctrl+0 to restore)",
                         level="info")

    def _restore_panel_grid(self) -> None:
        """Return to the 2×2 panel grid Ctrl+N left behind."""
        tags = self._panel_tags()
        for i, tag in enumerate(tags):
            if not tag:
                continue
            try:
                if i < 3:
                    dpg.configure_item(
                        tag, show=True,
                        width=self._preview_size + 16,
                        height=self._preview_size + 40,
                    )
                else:
                    dpg.configure_item(
                        tag, show=True,
                        width=self._preview_size + 16,
                        height=self._preview_size + 40,
                    )
            except Exception:
                pass
        self._maximized_panel = None
        self._set_status("Grid restored.", level="info")

    def _on_drop_zone_click(self) -> None:
        """Global left-click callback gated on drop-zone hover.
        Dear PyGui child_windows don't accept mvClickedHandler so the
        gate lives here instead of on the widget itself."""
        try:
            if dpg.is_item_hovered("drop_zone"):
                self._on_load_clicked()
        except Exception:
            # Drop zone not built yet (test harness or early teardown)
            # — swallow, nothing meaningful to do.
            pass

    def _on_key_escape(self, sender=None, app_data=None, user_data=None) -> None:
        """Esc sends a cancel request to whichever driver is running.
        If nothing is in flight the key just resets the status line
        — matching v1's walker behaviour."""
        # Exit interactive line-draw mode first — the user doesn't
        # want a cancelled gesture to also cancel a running scan.
        if getattr(self, "_line_mode", False):
            self._line_mode = False
            self._line_p1 = None
            self._line_p2 = None
            self._set_status("Line draw cancelled.", level="info")
            return
        fired = False
        try:
            fired = bool(self._science.cancel()) | fired
        except Exception:
            pass
        try:
            fired = bool(self._driver.cancel()) | fired
        except Exception:
            pass
        if fired:
            self._set_status(
                "Cancel requested — finishing current step…",
                level="warn",
            )
            self._toasts.show("Cancelled.", level="warn", ttl=2.0)
        else:
            self._set_status("Ready.", level="info")

    def _on_key_slash(self, sender, app_data, user_data=None) -> None:
        """``?`` (Shift+/) opens contextual help. No modifier needed —
        widely-recognised convention."""
        try:
            if dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(
                dpg.mvKey_RShift,
            ):
                self._on_help_overlay()
        except Exception:
            pass

    @staticmethod
    def _modifier_down() -> bool:
        """True when either Cmd (macOS) or Ctrl is held.

        Dear PyGui uses ``mvKey_LControl``/``mvKey_RControl`` on all
        platforms and aliases Cmd to Super on macOS via
        ``mvKey_LSuper`` / ``mvKey_RSuper``."""
        try:
            return (
                dpg.is_key_down(dpg.mvKey_LControl)
                or dpg.is_key_down(dpg.mvKey_RControl)
                or dpg.is_key_down(dpg.mvKey_LSuper)
                or dpg.is_key_down(dpg.mvKey_RSuper)
            )
        except Exception:
            return False

    # ---- drag-and-drop --------------------------------------------------
    def _on_viewport_drop(self, sender, app_data, user_data=None) -> None:
        """Callback fired when the user drops one or more files on the
        viewport. Dear PyGui packages the payload as a list of
        absolute paths (or dicts containing ``file_path_name`` on
        older builds); we tolerate both.
        """
        paths: list[str] = []
        if isinstance(app_data, (list, tuple)):
            for entry in app_data:
                if isinstance(entry, str):
                    paths.append(entry)
                elif isinstance(entry, dict):
                    p = entry.get("file_path_name") or entry.get("path")
                    if p:
                        paths.append(p)
        elif isinstance(app_data, str):
            paths.append(app_data)

        if not paths:
            return
        path = Path(paths[0])
        if path.suffix.lower() not in self._IMAGE_EXTS:
            self._set_status(
                f"Dropped file '{path.name}' isn't a supported image "
                f"({', '.join(self._IMAGE_EXTS)}).",
                level="warn",
            )
            return
        self._load_hologram(path)

    def _on_param_changed(self, sender, app_data, user_data=None) -> None:
        # Pull every input value back into ``_params`` — one handler for
        # all of them keeps the widget tree shallow. Any widget that
        # doesn't exist yet (tests or partial builds) falls back to the
        # current ReconParams value via ``_get_or`` below.
        def _get_or(tag: str, default):
            if not dpg.does_item_exist(tag):
                return default
            try:
                return dpg.get_value(tag)
            except Exception:
                return default
        try:
            self._params = ReconParams(
                wavelength_nm=float(_get_or("input_wavelength",
                                            self._params.wavelength_nm)),
                pixel_um=float(_get_or("input_pixel",
                                       self._params.pixel_um)),
                z_mm=float(_get_or("input_z", self._params.z_mm)),
                mask_radius=int(_get_or("input_mask",
                                        self._params.mask_radius)),
                method=str(_get_or("input_method", self._params.method)),
                reference_path=self._params.reference_path,
                subtract_reference=self._params.subtract_reference,
                magnification=float(_get_or("input_magnification",
                                            self._params.magnification)),
                pixel_is_effective=bool(_get_or(
                    "input_pixel_is_effective",
                    self._params.pixel_is_effective)),
                n_sample=float(_get_or("input_n_sample",
                                       self._params.n_sample)),
                n_medium=float(_get_or("input_n_medium",
                                       self._params.n_medium)),
                autofocus_metric=str(_get_or("input_af_metric",
                                             self._params.autofocus_metric)),
                af_z_min_mm=float(_get_or("input_af_z_min",
                                          self._params.af_z_min_mm)),
                af_z_max_mm=float(_get_or("input_af_z_max",
                                          self._params.af_z_max_mm)),
                af_n_steps=int(_get_or("input_af_n_steps",
                                       self._params.af_n_steps)),
                af_algorithm=str(_get_or("input_af_algorithm",
                                         self._params.af_algorithm)),
                subtract_mean=bool(_get_or("input_subtract_mean",
                                           self._params.subtract_mean)),
                hann_window=bool(_get_or("input_hann_window",
                                         self._params.hann_window)),
                fft_backend=str(_get_or("input_fft_backend",
                                        self._params.fft_backend)),
                unwrap_method=str(_get_or("input_unwrap_method",
                                          self._params.unwrap_method)),
                optical_mode=str(_get_or("input_optical_mode",
                                         self._params.optical_mode)),
                auto_contrast_amplitude=bool(_get_or(
                    "input_auto_contrast_amp",
                    self._params.auto_contrast_amplitude,
                )),
                # ROI is set/cleared via its own callback (ROI picker),
                # not the sidebar → preserve through edits.
                af_roi=self._params.af_roi,
            )
            # Nyquist check — warn (don't block) when effective pixel
            # > half the wavelength. Not a strict bound for the ASM
            # kernel, but a useful smell test for a mis-set M.
            eff_um = self._params.effective_pixel_um()
            if eff_um * 1e3 > self._params.wavelength_nm / 2:  # nm vs nm
                self._set_status(
                    f"Heads up: effective px {eff_um:.3f} µm > λ/2. "
                    f"Double-check ×M or 'pixel is effective'.",
                    level="warn",
                )
            self._refresh_info_text()
            self._mark_dirty()
        except Exception:
            _LOG.debug("param sync failed", exc_info=True)

    def _on_flip_toggle(self, *, vertical: bool, value: bool) -> None:
        """View-menu flip toggles. Flipping only affects the preview
        texture — the stored hologram and all downstream math use
        the raw array, so toggling doesn't invalidate cached recon
        results. If a hologram is already loaded we immediately
        repaint the input panel to reflect the new orientation.
        Flags persist to state store so the operator's camera
        orientation preference sticks across restarts."""
        if vertical:
            self._flip_display_v = value
        else:
            self._flip_display_h = value
        self._mark_dirty()
        # Repaint the input panel with the current flip combo, if
        # we have a hologram to show.
        if self._current_hologram is None:
            return
        try:
            from core.ingestion import load_any
            loaded = load_any(self._current_hologram)
            raw = np.asarray(loaded.array, dtype=np.float32)
            if raw.ndim == 3:
                raw = raw[..., 0]
            disp = raw
            if self._flip_display_v:
                disp = np.flipud(disp)
            if self._flip_display_h:
                disp = np.fliplr(disp)
            self._push_texture(
                self.panel_input.tex_tag, disp, colormap="gray",
            )
        except Exception:
            _LOG.debug("flip-toggle repaint failed", exc_info=True)

    def _on_af_algorithm_changed(self, sender, app_data,
                                 user_data=None) -> None:
        """Autofocus algorithm combo — two jobs: (1) route into the
        normal param-changed flow so ``ReconParams.af_algorithm``
        updates, (2) retarget the z-range / steps widgets so their
        labels + enabled state reflect what the chosen algorithm
        actually consumes. v2.0.8 added this because the static
        ``steps`` label read as a fixed grid count no matter which
        algorithm was selected — confusing for adaptive variants
        where the same field is a max-evaluations budget."""
        self._on_param_changed(sender, app_data, user_data)
        self._apply_af_algorithm_ux()

    def _apply_af_algorithm_ux(self) -> None:
        """Reflect the current ``af_algorithm`` on the sidebar:

        * Relabel the steps widget per :func:`af_algorithm_input_profile`
          so adaptive variants read "max evals" while grid methods
          still say "steps".
        * Grey out z_min / z_max when an algorithm ignores them
          (``adaptive_distance`` auto-discovers the range).
        * Swap tooltip text so hovering the widget always says
          what *this* algorithm will do with the value.
        """
        profile = af_algorithm_input_profile(self._params.af_algorithm)
        steps_label = profile.get("steps_label", "steps")
        steps_tip = profile.get("steps_tip", "")
        uses_range = bool(profile.get("uses_z_range", True))
        # Widget tags are emitted by _build_ui; dpg.configure_item is
        # the standard way to mutate properties at runtime.
        if dpg.does_item_exist("input_af_n_steps"):
            try:
                dpg.configure_item("input_af_n_steps", label=steps_label)
            except Exception:
                pass
        if dpg.does_item_exist("input_af_n_steps_tip_text"):
            try:
                dpg.set_value("input_af_n_steps_tip_text",
                              steps_tip or "Autofocus step budget.")
            except Exception:
                pass
        for tag in ("input_af_z_min", "input_af_z_max"):
            if dpg.does_item_exist(tag):
                try:
                    dpg.configure_item(tag, enabled=uses_range)
                except Exception:
                    pass
        # z-range label gets an "(auto)" suffix so the user knows
        # the greyed-out state isn't broken — it's intentional.
        if dpg.does_item_exist("input_af_z_min"):
            try:
                dpg.configure_item(
                    "input_af_z_min",
                    label="z min (mm)" + ("" if uses_range
                                           else "  (auto)"),
                )
            except Exception:
                pass
        if dpg.does_item_exist("input_af_z_max"):
            try:
                dpg.configure_item(
                    "input_af_z_max",
                    label="z max (mm)" + ("" if uses_range
                                           else "  (auto)"),
                )
            except Exception:
                pass

    def _on_sample_id_changed(self, sender, app_data, user_data=None) -> None:
        self._sample_id = str(app_data or "")
        self._mark_dirty()

    # ---- Drop-zone + status bar copy ----------------------------------

    def _drop_zone_label(self) -> str:
        """String shown on the drop strip. Honest about capability —
        macOS Cocoa doesn't emit drop events on Dear PyGui 2.x, so
        we call it out instead of pretending."""
        if self._drop_supported:
            return ("Drop a hologram anywhere in this panel — "
                    "or click here to browse.")
        return ("Click here to load a hologram  ·  "
                "drag-and-drop not supported on this platform.")

    def _status_ready_text(self) -> str:
        if self._drop_supported:
            return ("Ready.   Drag a hologram onto the window "
                    "or click the drop zone above to load it.")
        return ("Ready.   Click the drop zone above, "
                "or File → Load hologram, to open a file.")

    # ---- Workflow section visibility ----------------------------------

    # Tag → minimum set of workflow modes that want to see this section.
    _WORKFLOW_SECTIONS: dict[str, set[str]] = {
        "section_sample": {"Reconstruct", "Analyse", "Report"},
        "section_preset": {"Reconstruct"},
        "section_params": {"Reconstruct", "Analyse"},
        "section_actions": {"Reconstruct", "Analyse", "Report"},
        "section_reference": {"Reconstruct", "Analyse"},
    }

    def _apply_workflow_visibility(self, mode: str) -> None:
        """Show/hide sidebar sections based on the selected workflow
        mode. Safe to call before the widgets exist — we use
        ``does_item_exist`` at every toggle."""
        for tag, visible_for in self._WORKFLOW_SECTIONS.items():
            if not dpg.does_item_exist(tag):
                continue
            try:
                dpg.configure_item(tag, show=(mode in visible_for))
            except Exception:
                pass

    # ---- Recent menu rebuild ------------------------------------------

    def _rebuild_recent_menu(self) -> None:
        if not dpg.does_item_exist("menu_recent"):
            return
        try:
            dpg.delete_item("menu_recent", children_only=True)
        except Exception:
            return
        if not self._recent:
            dpg.add_menu_item(
                label="(empty)", enabled=False, parent="menu_recent",
            )
            return
        for p in self._recent:
            dpg.add_menu_item(
                label=p.name,
                parent="menu_recent",
                callback=lambda s, a, u=p: self._load_hologram(u),
            )

    # ---- Info panel composition --------------------------------------

    def _compose_info_text(self) -> str:
        """Render the info panel from the *live* result cache.

        v2.0.1 appended each pipeline's output onto the existing text,
        so running QPI then depth printed the old text + new text and
        "Clear depth overlay" orphaned a line. Re-composing from cached
        results on every handler call keeps the panel deterministic:
        the content always mirrors what ``self._last_*`` holds."""
        if self._last_recon is None and self._current_hologram is None:
            return "No reconstruction yet."

        lines: list[str] = []
        if self._current_hologram is not None:
            lines.append(f"Hologram: {self._current_hologram.name}")
        if self._last_recon is not None:
            r = self._last_recon
            lines.append(f"Input:     {r.input_image.shape}")
            lines.append(
                f"Amplitude: range [{r.amplitude.min():.3g}, "
                f"{r.amplitude.max():.3g}]"
            )
            lines.append(
                f"Phase:     range [{r.phase.min():.2f}, "
                f"{r.phase.max():.2f}] rad"
            )
            lines.append(f"+1 order:  pixel {r.offaxis_center}")
            lines.append(f"Runtime:   {r.runtime_ms:.0f} ms")
        # v2.0.3: show the pixel-size math the pipeline is actually
        # using — keeps the M vs. effective-pixel relationship honest
        # instead of hiding it in the recon kernel.
        eff_um = self._params.effective_pixel_um()
        if self._params.pixel_is_effective or self._params.magnification == 1:
            lines.append(f"Effective px: {eff_um:.3f} µm "
                         f"(×{self._params.magnification:g})")
        else:
            lines.append(
                f"Effective px: {eff_um:.3f} µm = "
                f"{self._params.pixel_um:.3f} µm / "
                f"×{self._params.magnification:g}"
            )
        lines.append(f"Sample ID: {self._sample_id or '(none)'}")
        # v2.0.7: optical mode transparency — "3 µm object = 6 µm"
        # bug source. Show it so the operator never has to guess
        # whether reflection's ×2 correction is applied.
        lines.append(
            f"Optical:   {self._params.optical_mode}"
        )
        # Reference transparency — the operator should never have to
        # guess whether subtraction is armed.
        if self._params.reference_path is not None:
            ref_state = ("on" if self._params.subtract_reference
                         else "loaded")
            lines.append(
                f"Reference: {self._params.reference_path.name} "
                f"({ref_state})"
            )
        else:
            lines.append("Reference: (none)")
        # Autofocus ROI — local-focus mask status.
        if self._params.af_roi is not None:
            y0, x0, y1, x1 = self._params.af_roi
            lines.append(
                f"AF ROI:    y∈[{y0:.2f},{y1:.2f}] x∈[{x0:.2f},{x1:.2f}]"
            )

        if self._last_qpi is not None:
            q = self._last_qpi.qpi
            qpi_line = f"QPI:       {self._last_qpi.runtime_ms:.0f} ms"
            if q.phase_stats is not None and q.phase_stats.range_nm is not None:
                qpi_line += f" · OPD {q.phase_stats.range_nm:.2f} nm"
            if q.total_dry_mass_pg is not None:
                qpi_line += f" · mass {q.total_dry_mass_pg:.2f} pg"
            if q.step_height_m is not None:
                qpi_line += f" · step {q.step_height_m * 1e9:.2f} nm"
            lines.append(qpi_line)
        if self._last_depth is not None:
            d = self._last_depth
            z_min = d.result.z_map.min() * 1e3
            z_max = d.result.z_map.max() * 1e3
            lines.append(
                f"Depth:     z∈[{z_min:+.2f}, {z_max:+.2f}] mm · "
                f"{len(d.clusters)} cluster(s)"
            )
        return "\n".join(lines)

    def _refresh_info_text(self) -> None:
        """Write the composed info text to the widget — no-op when the
        panel hasn't been built (tests bypass it)."""
        if not dpg.does_item_exist("info_text"):
            return
        try:
            dpg.set_value("info_text", self._compose_info_text())
        except Exception:
            _LOG.debug("info text refresh failed", exc_info=True)

    # ---- Onboarding reset ---------------------------------------------

    def _reset_onboarding_flag(self) -> None:
        """Delete the first-run flag so the next launch shows the
        welcome wizard. Also reopens the wizard immediately for the
        user to walk through without restarting."""
        try:
            if self._ONBOARD_FLAG.exists():
                self._ONBOARD_FLAG.unlink()
        except Exception:
            _LOG.debug("couldn't remove onboarding flag", exc_info=True)
        self._set_status("Onboarding reset — will show on next launch.",
                         level="info")
        self._toasts.show("Onboarding reset.", level="info", ttl=3.0)
        self._show_onboarding_manual()

    # ---- Reconstruct tooltip + enable state --------------------------

    def _refresh_reconstruct_tooltip(self, *, enabled: bool) -> None:
        """Keep the disabled-reason tooltip honest. Called from
        ``_load_hologram`` (enables) and error paths (re-disables)."""
        if not dpg.does_item_exist("btn_reconstruct_tip_text"):
            return
        msg = (f"Run reconstruction ({self._shortcut('Ctrl+R')})"
               if enabled
               else "Load a hologram first (File → Load hologram…).")
        try:
            dpg.set_value("btn_reconstruct_tip_text", msg)
        except Exception:
            pass

    def _on_load_clicked(self) -> None:
        """Open a file picker and hand the result to ``_load_hologram``.

        We prefer :func:`tkinter.filedialog.askopenfilename` — it's the
        native OS picker on every platform (Cocoa on macOS, Win32 on
        Windows, GTK/Qt on Linux). Dear PyGui's built-in ``file_dialog``
        is a custom widget that is notoriously flaky on macOS (callback
        never fires, multi-select behaves oddly, ``file_path_name``
        sometimes empty). We fall back to the DPG dialog only if
        tkinter isn't available."""
        path = self._ask_file_via_tk(
            title="Load hologram",
            filetypes=[
                ("Hologram image", "*.tif *.tiff *.png *.bmp *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
            default_dir=self._last_dir or str(Path.home()),
        )
        if path:
            self._load_hologram(Path(path))
            return
        # tkinter unavailable — fall back to DPG's built-in dialog.
        try:
            if self._last_dir and dpg.does_item_exist("file_dialog_open"):
                dpg.configure_item("file_dialog_open",
                                   default_path=self._last_dir)
        except Exception:
            pass
        try:
            dpg.show_item("file_dialog_open")
        except Exception:
            self._set_status(
                "No file picker available. "
                "Install tkinter or use File → Recent.",
                level="danger",
            )

    def _on_file_selected(self, sender, app_data) -> None:
        """DPG fallback callback. Accept every payload shape DPG is
        known to emit — ``file_path_name`` (common), ``current_path``
        (rare), and the ``selections`` dict (multi-select builds)."""
        sel = (app_data.get("file_path_name")
               or app_data.get("current_path"))
        if not sel:
            selections = app_data.get("selections") or {}
            if selections:
                sel = next(iter(selections.values()))
        if not sel:
            self._set_status(
                "File picker returned nothing to load.", level="warn",
            )
            return
        self._load_hologram(Path(sel))

    @staticmethod
    def _ask_file_via_tk(
        *,
        title: str,
        filetypes: list[tuple[str, str]],
        default_dir: str = "",
        save: bool = False,
        default_filename: str = "",
        ask_directory: bool = False,
    ) -> Optional[str]:
        """Open a native OS file picker. Returns the selected path or
        ``None`` on cancel/unavailable.

        Dispatches by platform:

        * **macOS** → ``osascript`` (AppleScript ``choose file`` /
          ``choose folder`` / ``choose file name``). Runs in its own
          process with its own main thread, so the HIToolbox
          main-thread check (``dispatch_assert_queue``) that
          SIGTRAPs an inline tkinter call from a Dear PyGui callback
          never fires.
        * **Other platforms** → a tiny ``python -c`` subprocess that
          imports ``tkinter.filedialog``. Each subprocess has its own
          fresh main thread, so tkinter initialises cleanly
          regardless of what Dear PyGui is doing in the parent.

        We intentionally don't call tkinter inline: on macOS,
        ``Tk()`` → ``TkpInitKeymapInfo`` → ``TSMGetInputSourceProperty``
        requires the real process main thread, which Dear PyGui's
        callback dispatcher is not.
        """
        import subprocess
        import sys
        if sys.platform == "darwin":
            return DhmApp._ask_file_via_osascript(
                title=title, default_dir=default_dir,
                save=save, default_filename=default_filename,
                ask_directory=ask_directory,
                filetypes=filetypes,
            )
        # Non-macOS — spawn a subprocess that runs tkinter in its own
        # main thread. JSON in, JSON out keeps the protocol boring.
        driver = f"""
import json, sys, tkinter
from tkinter import filedialog
payload = json.loads(sys.stdin.read() or '{{}}')
root = tkinter.Tk()
root.withdraw()
try:
    root.attributes('-topmost', True)
except Exception:
    pass
try:
    if payload['ask_directory']:
        res = filedialog.askdirectory(
            title=payload['title'],
            initialdir=payload['default_dir'] or None,
        )
    elif payload['save']:
        res = filedialog.asksaveasfilename(
            title=payload['title'],
            initialdir=payload['default_dir'] or None,
            initialfile=payload['default_filename'] or None,
            filetypes=[tuple(ft) for ft in payload['filetypes']],
        )
    else:
        res = filedialog.askopenfilename(
            title=payload['title'],
            initialdir=payload['default_dir'] or None,
            filetypes=[tuple(ft) for ft in payload['filetypes']],
        )
finally:
    try:
        root.destroy()
    except Exception:
        pass
sys.stdout.write(res or '')
"""
        import json
        payload = json.dumps({
            "title": title,
            "default_dir": default_dir,
            "default_filename": default_filename,
            "filetypes": filetypes,
            "save": bool(save),
            "ask_directory": bool(ask_directory),
        })
        try:
            r = subprocess.run(
                [sys.executable, "-c", driver],
                input=payload,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except Exception:
            _LOG.exception("tkinter subprocess picker failed")
            return None
        if r.returncode != 0:
            _LOG.debug("tkinter subprocess stderr: %s", r.stderr)
            return None
        return (r.stdout or "").strip() or None

    @staticmethod
    def _ask_file_via_osascript(
        *,
        title: str,
        default_dir: str = "",
        save: bool = False,
        default_filename: str = "",
        ask_directory: bool = False,
        filetypes: Optional[list[tuple[str, str]]] = None,
    ) -> Optional[str]:
        """macOS native picker via AppleScript. Always uses the
        Finder's process main thread — no conflict with whatever
        threads Dear PyGui has running."""
        import subprocess
        # Map GUI filetypes (name, "*.tif *.png") into AppleScript's
        # ``of type {"tif", "png"}`` form. Extensions only, no globs.
        of_type = ""
        if filetypes and not ask_directory:
            exts: list[str] = []
            for _, pattern in filetypes:
                for tok in pattern.split():
                    tok = tok.strip().lstrip("*").lstrip(".")
                    if tok and tok != "*" and tok not in exts:
                        exts.append(tok)
            if exts:
                quoted = ", ".join(f'"{e}"' for e in exts)
                of_type = f" of type {{{quoted}}}"
        safe_title = (title or "").replace('"', '\\"')
        default_location = ""
        if default_dir:
            safe_dir = default_dir.replace('"', '\\"')
            default_location = f' default location POSIX file "{safe_dir}"'
        try:
            if ask_directory:
                script = (
                    f'POSIX path of (choose folder with prompt '
                    f'"{safe_title}"{default_location})'
                )
            elif save:
                safe_name = (default_filename or "").replace('"', '\\"')
                default_name = (f' default name "{safe_name}"'
                                if safe_name else "")
                script = (
                    f'POSIX path of (choose file name with prompt '
                    f'"{safe_title}"{default_location}{default_name})'
                )
            else:
                script = (
                    f'POSIX path of (choose file with prompt '
                    f'"{safe_title}"{default_location}{of_type})'
                )
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=600,
            )
        except Exception:
            _LOG.exception("osascript file picker failed")
            return None
        if r.returncode != 0:
            # User cancelled → osascript returns a -128 error to stderr;
            # that's expected, not a bug, so we don't log at warn level.
            _LOG.debug("osascript returncode=%d stderr=%s",
                       r.returncode, r.stderr)
            return None
        path = (r.stdout or "").strip()
        return path or None

    def _load_hologram(self, path: Path) -> None:
        """Shared entry point — file dialog, drag-drop and Recent menu
        all go through this helper so status / texture / Recent list
        stay in sync."""
        if not path.exists():
            self._set_status(f"File not found: {path}", level="warn")
            return
        # v2.1.x: any explicit file load returns us to "file" mode,
        # even when a camera was previously running. The user's
        # action is unambiguous — they picked a file from disk.
        self._set_input_mode("file")
        self._current_hologram = path
        # Remember the parent directory so the next file dialog opens
        # where the user already is, instead of $HOME.
        try:
            self._last_dir = str(path.parent)
            if dpg.does_item_exist("file_dialog_open"):
                dpg.configure_item("file_dialog_open",
                                   default_path=self._last_dir)
        except Exception:
            pass
        try:
            dpg.configure_item("btn_reconstruct", enabled=True)
        except Exception:
            pass
        self._refresh_reconstruct_tooltip(enabled=True)
        # New hologram wipes derived caches — QPI/depth referred to the
        # previous file. Clear so info panel doesn't show stale metrics.
        self._last_recon = None
        self._last_qpi = None
        self._last_qpi_batch = None
        self._last_depth = None
        try:
            dpg.configure_item("btn_surface", enabled=False)
        except Exception:
            pass
        self._set_status(f"Loaded: {path.name}", level="ok")
        self._remember_recent(path)
        try:
            from core.ingestion import load_any
            loaded = load_any(path)
            raw = np.asarray(loaded.array, dtype=np.float32)
            if raw.ndim == 3:
                raw = raw[..., 0]
            # v2.0.9: honour the user's display-flip preference so
            # holograms from sensors / labs with bottom-up row
            # ordering (some scientific TIFF writers, SEM camera
            # dumps) render the right way up without a rewrite
            # step. Only affects preview; pipeline math uses the
            # raw array untouched. Must happen BEFORE the push.
            if getattr(self, "_flip_display_v", False):
                raw_disp = np.flipud(raw)
            else:
                raw_disp = raw
            if getattr(self, "_flip_display_h", False):
                raw_disp = np.fliplr(raw_disp)
            # ``panel_input.tex_tag`` is a random UUID. Passing a
            # literal "tex_input" string lands in the fallback path
            # which silently no-ops because no such texture exists —
            # the bug that hid the input panel on every load until
            # 2026-04-24. Use the actual tag.
            self._push_texture(
                self.panel_input.tex_tag, raw_disp, colormap="gray",
            )
            # v2.0.3: harvest embedded metadata (magnification,
            # pixel size, wavelength). v1 used this to pre-fill the
            # sidebar — v2 silently ignored it, making users type in
            # values the file already carried.
            self._apply_detected_metadata(loaded.metadata or {})
            self._refresh_info_text()
        except Exception as exc:
            self._set_status(f"Preview failed: {exc}", level="warn")
        self._mark_dirty()

    def _apply_detected_metadata(self, meta: dict) -> None:
        """Pull any ``magnification`` / ``pixel_size_m`` / ``wavelength_m``
        keys off ``meta`` and push them into the sidebar, toasting what
        changed. We're deliberately non-destructive when a key is
        missing — user-entered values stay untouched."""
        changes: list[str] = []
        mag = meta.get("magnification")
        if mag is not None:
            try:
                mag_f = float(mag)
                if mag_f > 0 and abs(mag_f - self._params.magnification) > 1e-6:
                    self._params = ReconParams(**{
                        **self._params.__dict__,
                        "magnification": mag_f,
                        # When metadata provides M, the px value in the
                        # file header is *camera* pixel size — so the
                        # effective flag should flip off.
                        "pixel_is_effective": False,
                    })
                    if dpg.does_item_exist("input_magnification"):
                        dpg.set_value("input_magnification", mag_f)
                    if dpg.does_item_exist("input_pixel_is_effective"):
                        dpg.set_value("input_pixel_is_effective", False)
                    changes.append(f"×M = {mag_f:g}")
            except Exception:
                pass
        px_m = meta.get("pixel_size_m")
        if px_m is not None:
            try:
                px_um = float(px_m) * 1e6
                if px_um > 0 and abs(px_um - self._params.pixel_um) > 1e-3:
                    self._params = ReconParams(**{
                        **self._params.__dict__,
                        "pixel_um": px_um,
                    })
                    if dpg.does_item_exist("input_pixel"):
                        dpg.set_value("input_pixel", px_um)
                    changes.append(f"px = {px_um:.3f} µm")
            except Exception:
                pass
        wl_m = meta.get("wavelength_m")
        if wl_m is not None:
            try:
                wl_nm = float(wl_m) * 1e9
                if wl_nm > 0 and abs(wl_nm - self._params.wavelength_nm) > 1e-3:
                    self._params = ReconParams(**{
                        **self._params.__dict__,
                        "wavelength_nm": wl_nm,
                    })
                    if dpg.does_item_exist("input_wavelength"):
                        dpg.set_value("input_wavelength", wl_nm)
                    changes.append(f"λ = {wl_nm:.1f} nm")
            except Exception:
                pass
        if changes:
            self._toasts.show(
                "Auto-detected from file: " + ", ".join(changes),
                level="info", ttl=5.0,
            )

    def _remember_recent(self, path: Path) -> None:
        """Put ``path`` at the top of the Recent list (de-duplicated)
        and rebuild the File → Recent submenu."""
        try:
            path = path.resolve()
        except Exception:
            pass
        self._recent = [p for p in self._recent if p != path]
        self._recent.insert(0, path)
        self._recent = self._recent[: self._MAX_RECENT]
        self._rebuild_recent_menu()
        self._mark_dirty()

    def _on_reconstruct(self) -> None:
        # v2.1.x: when the camera is running and no file is loaded,
        # snapshot the latest live frame and reconstruct from that.
        # Operator gets immediate "reconstruct what I see" semantics
        # without the manual record→stop→load round-trip.
        if self._current_hologram is None and \
                self._input_mode == "live":
            snap = self._snapshot_live_frame_to_tempfile()
            if snap is not None:
                self._current_hologram = snap
                # Stay in live mode for the status prefix — the
                # snapshot is a derived artefact of the live feed.
                self._set_status(
                    f"Reconstructing live snapshot {snap.name}…",
                    level="info",
                )
        if self._current_hologram is None:
            self._set_status(
                "Load a hologram first (File → Load hologram…).",
                level="warn",
            )
            return
        # Pull latest params before submitting.
        self._on_param_changed(None, None)
        self._set_status("Reconstructing…", level="info")
        try:
            dpg.configure_item("btn_reconstruct", enabled=False)
            dpg.configure_item("btn_surface", enabled=False)
        except Exception:
            pass
        self._refresh_reconstruct_tooltip(enabled=False)
        self._driver.submit(
            self._current_hologram, self._params,
            sample_id=self._sample_id,
            on_result=self._post_result,
            on_error=self._post_error,
        )

    def _show_shortcuts(self) -> None:
        if dpg.does_item_exist("shortcuts_modal"):
            dpg.configure_item("shortcuts_modal", show=True)
            return
        sc = self._shortcut
        drop_help = ("Drops a hologram preview"
                     if self._drop_supported
                     else "Click the drop zone instead (macOS)")
        rows = [
            ("Load hologram…",     sc("Ctrl+O")),
            ("Reconstruct",        sc("Ctrl+R")),
            ("Command palette",    sc("Ctrl+K")),
            ("Contextual help",    "?"),
            ("Clear status",       "Esc"),
            ("Drag a file",        drop_help),
        ]
        with dpg.window(label="Keyboard shortcuts", modal=True,
                        no_resize=True, tag="shortcuts_modal",
                        width=420, height=260, pos=(220, 200)):
            for action, keys in rows:
                with dpg.group(horizontal=True):
                    dpg.add_text(action, color=ScientificTheme.TEXT)
                    dpg.add_spacer(width=8)
                    dpg.add_text(keys,
                                 color=ScientificTheme.TEXT_MUTED)
            dpg.add_spacer(height=8)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item(
                    "shortcuts_modal", show=False,
                ),
            )

    # ---- First-run onboarding ------------------------------------------
    _ONBOARD_FLAG = Path.home() / ".dhm-reconstruction" / "ui2_onboarding.flag"

    def _maybe_show_onboarding(self) -> None:
        try:
            if self._ONBOARD_FLAG.exists():
                return
        except Exception:
            pass

        def _after() -> None:
            try:
                self._ONBOARD_FLAG.parent.mkdir(parents=True, exist_ok=True)
                self._ONBOARD_FLAG.write_text("1")
            except Exception:
                pass

        show_onboarding(on_close=_after)

    # ---- Presets --------------------------------------------------------

    # Built-ins are immutable — Save/Delete UI must refuse to overwrite
    # or remove these names. Keyed separately so the merge order in
    # :meth:`_presets` is deterministic (built-ins first, user after).
    _BUILTIN_PRESET_NAMES: tuple[str, ...] = ("Cell", "Film", "USAF", "Custom")

    def _presets(self) -> dict:
        """Merge built-in presets with the user's saved set. Instance
        method (used to be a static) because user presets come off
        ``self._settings.ui2.user_presets`` — can't be classmethod."""
        built_in = self._builtin_presets()
        # Keep user presets out of the chip row if they collide with a
        # built-in — avoids two chips with the same label rendering
        # over each other. Save dialog already rejects these names.
        user = {
            name: cfg for name, cfg in
            self._settings.ui2.user_presets.items()
            if name not in built_in
        }
        return {**built_in, **user}

    @staticmethod
    def _builtin_presets() -> dict:
        """Name → parameter-defaults mapping. Same values the PySide
        ReconTab carries; kept here so ``ui2`` stays self-contained.

        Magnification defaults match typical DHM lab setups — a biology
        cell on a microscope is 40×, a USAF resolution target calibrates
        at 10×, a macroscopic thin film (coating inspection) has no
        objective (1×). The camera-pixel value stays unchanged, so
        ``pixel_is_effective=False`` lets the pipeline divide by M."""
        return {
            "Cell":   dict(wavelength_nm=632.8, pixel_um=3.45,
                           z_mm=10.0, mask_radius=80, method="ASM",
                           magnification=40.0, pixel_is_effective=False,
                           n_sample=1.38, n_medium=1.337,
                           autofocus_metric="LAPLACIAN_VARIANCE"),
            "Film":   dict(wavelength_nm=532.0, pixel_um=5.5,
                           z_mm=50.0, mask_radius=100, method="Fresnel",
                           magnification=1.0, pixel_is_effective=True,
                           n_sample=1.50, n_medium=1.000,
                           autofocus_metric="LAPLACIAN_VARIANCE"),
            "USAF":   dict(wavelength_nm=632.8, pixel_um=2.2,
                           z_mm=5.0,  mask_radius=60, method="ASM",
                           magnification=10.0, pixel_is_effective=False,
                           n_sample=1.50, n_medium=1.000,
                           autofocus_metric="TENENGRAD"),
            "Custom": dict(wavelength_nm=632.8, pixel_um=5.0,
                           z_mm=10.0, mask_radius=40, method="ASM",
                           magnification=1.0, pixel_is_effective=True,
                           n_sample=1.38, n_medium=1.337,
                           autofocus_metric="LAPLACIAN_VARIANCE"),
        }

    # ---- Save / delete user-defined presets ---------------------------

    def _current_params_as_preset_dict(self) -> dict:
        """Snapshot the live ReconParams into the same shape the
        built-in preset dicts use."""
        return dict(
            wavelength_nm=float(self._params.wavelength_nm),
            pixel_um=float(self._params.pixel_um),
            z_mm=float(self._params.z_mm),
            mask_radius=int(self._params.mask_radius),
            method=str(self._params.method),
            magnification=float(self._params.magnification),
            pixel_is_effective=bool(self._params.pixel_is_effective),
            n_sample=float(self._params.n_sample),
            n_medium=float(self._params.n_medium),
            autofocus_metric=str(self._params.autofocus_metric),
        )

    _SAVE_PRESET_TAG = "save_preset_modal"
    _SAVE_PRESET_INPUT = "save_preset_name"
    _SAVE_PRESET_FEEDBACK = "save_preset_feedback"

    def _open_save_preset_dialog(self) -> None:
        if dpg.does_item_exist(self._SAVE_PRESET_TAG):
            dpg.configure_item(self._SAVE_PRESET_TAG, show=True)
            dpg.set_value(self._SAVE_PRESET_INPUT, "")
            dpg.set_value(self._SAVE_PRESET_FEEDBACK, "")
            dpg.focus_item(self._SAVE_PRESET_INPUT)
            return
        with dpg.window(label="Save preset", modal=True,
                        tag=self._SAVE_PRESET_TAG,
                        no_resize=True, no_collapse=True,
                        width=420, height=200, pos=(240, 200)):
            dpg.add_text(
                "Snapshot the current reconstruction parameters as "
                "a new preset chip. Built-in names "
                "(Cell/Film/USAF/Custom) are reserved.",
                color=ScientificTheme.TEXT_MUTED, wrap=380,
            )
            dpg.add_separator()
            dpg.add_input_text(
                hint="e.g. Lab1 x40",
                width=-1,
                tag=self._SAVE_PRESET_INPUT,
                on_enter=True,
                callback=lambda s, a: self._commit_save_preset(),
            )
            dpg.add_text("", tag=self._SAVE_PRESET_FEEDBACK,
                         color=ScientificTheme.WARN)
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Save",
                    callback=lambda: self._commit_save_preset(),
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.configure_item(
                        self._SAVE_PRESET_TAG, show=False,
                    ),
                )
        dpg.focus_item(self._SAVE_PRESET_INPUT)

    def _commit_save_preset(self) -> None:
        try:
            name = str(dpg.get_value(self._SAVE_PRESET_INPUT) or "").strip()
        except Exception:
            name = ""
        if not name:
            self._set_preset_feedback("Give the preset a name.")
            return
        if name in self._BUILTIN_PRESET_NAMES:
            self._set_preset_feedback(f"'{name}' is a built-in — pick a new name.")
            return
        # v2.0.7: collision branch. Up to v2.0.6 we silently overwrote a
        # user preset of the same name — operator's earlier values were
        # lost without trace, which surfaced in the v2.0.7 backlog as
        # "User preset dialog 'Edit existing'". Now we surface the
        # collision: close the save modal, open a Replace? dialog that
        # archives the previous version before clobbering. Cancel
        # exits without losing the in-flight new values.
        if name in self._settings.ui2.user_presets:
            try:
                dpg.configure_item(self._SAVE_PRESET_TAG, show=False)
            except Exception:
                pass
            self._open_replace_preset_dialog(
                name,
                old_dict=dict(self._settings.ui2.user_presets[name]),
                new_dict=self._current_params_as_preset_dict(),
            )
            return
        self._save_user_preset(name, self._current_params_as_preset_dict())
        try:
            dpg.configure_item(self._SAVE_PRESET_TAG, show=False)
        except Exception:
            pass

    def _save_user_preset(self, name: str, preset: dict,
                          *, archive_previous: Optional[dict] = None) -> None:
        """Single source of truth for writing a user preset.

        Called by both the no-collision save path and the explicit
        Replace flow. ``archive_previous`` is the *old* dict to push
        into ``user_preset_archive[name]`` before storing ``preset``;
        callers pass it when overwriting, leave None for a brand-new
        save. Archive is capped at 10 entries per name (drops oldest)
        to keep state files lean."""
        new_user = {**self._settings.ui2.user_presets, name: preset}
        new_archive = {
            k: list(v) for k, v in
            self._settings.ui2.user_preset_archive.items()
        }
        if archive_previous is not None:
            history = new_archive.get(name, [])
            history = history + [archive_previous]
            # Cap — keep only the most recent 10 versions.
            if len(history) > 10:
                history = history[-10:]
            new_archive[name] = history
        self._settings = self._settings.with_ui2(
            user_presets=new_user,
            user_preset_archive=new_archive,
        )
        self._selected_preset = name
        self._rebuild_preset_chips()
        self._preset_chips.set_active(name)
        verb = "replaced" if archive_previous is not None else "saved"
        self._set_status(f"Preset {verb}: {name}", level="ok")
        self._toasts.show(f"Preset {verb}: {name}",
                          level="ok", ttl=3.0)
        self._mark_dirty()

    _REPLACE_PRESET_TAG = "replace_preset_modal"

    def _open_replace_preset_dialog(self, name: str, *,
                                    old_dict: dict,
                                    new_dict: dict) -> None:
        """Modal asking the operator to confirm overwriting an existing
        user preset. Shows a side-by-side diff of the fields that
        actually differ so the operator sees what they're losing
        before pressing Replace. The previous version is archived
        regardless — Replace is recoverable, not destructive."""
        if dpg.does_item_exist(self._REPLACE_PRESET_TAG):
            dpg.delete_item(self._REPLACE_PRESET_TAG)
        diff_keys = sorted(set(old_dict.keys()) | set(new_dict.keys()))
        with dpg.window(label=f"Replace preset '{name}'?", modal=True,
                        tag=self._REPLACE_PRESET_TAG,
                        no_resize=True, no_collapse=True,
                        width=520, height=360, pos=(220, 180)):
            dpg.add_text(
                f"A user preset named '{name}' already exists. "
                f"Replace will store the new values and move the old "
                f"version to the archive (recoverable, not deleted).",
                color=ScientificTheme.TEXT_MUTED, wrap=480,
            )
            dpg.add_separator()
            dpg.add_text("Differences (old → new):",
                         color=ScientificTheme.TEXT_MUTED)
            with dpg.group(horizontal=False):
                changed = 0
                for k in diff_keys:
                    old_v = old_dict.get(k, "(missing)")
                    new_v = new_dict.get(k, "(missing)")
                    if old_v == new_v:
                        continue
                    changed += 1
                    dpg.add_text(f"  {k}: {old_v} → {new_v}")
                if changed == 0:
                    dpg.add_text("  (no field differs — replace is "
                                 "a no-op, but archive will still "
                                 "record the snapshot)",
                                 color=ScientificTheme.TEXT_MUTED)
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Replace",
                    callback=lambda: self._commit_replace_preset(
                        name, old_dict, new_dict,
                    ),
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.configure_item(
                        self._REPLACE_PRESET_TAG, show=False,
                    ),
                )

    def _commit_replace_preset(self, name: str,
                               old_dict: dict, new_dict: dict) -> None:
        """Replace branch chose by the operator. Archive old, write
        new, close modal."""
        self._save_user_preset(name, new_dict, archive_previous=old_dict)
        try:
            dpg.configure_item(self._REPLACE_PRESET_TAG, show=False)
        except Exception:
            pass

    def _set_preset_feedback(self, msg: str) -> None:
        try:
            dpg.set_value(self._SAVE_PRESET_FEEDBACK, msg)
        except Exception:
            pass

    _DELETE_PRESET_TAG = "delete_preset_modal"

    def _open_delete_preset_dialog(self) -> None:
        names = [n for n in self._settings.ui2.user_presets
                 if n not in self._BUILTIN_PRESET_NAMES]
        if not names:
            self._set_status("No user presets to delete.", level="info")
            return
        if dpg.does_item_exist(self._DELETE_PRESET_TAG):
            dpg.delete_item(self._DELETE_PRESET_TAG)
        with dpg.window(label="Delete preset", modal=True,
                        tag=self._DELETE_PRESET_TAG,
                        no_resize=True, no_collapse=True,
                        width=420, height=260, pos=(240, 200)):
            dpg.add_text(
                "Pick a user-defined preset to remove. Built-in "
                "presets are protected.",
                color=ScientificTheme.TEXT_MUTED, wrap=380,
            )
            dpg.add_separator()
            for name in names:
                with dpg.group(horizontal=True):
                    dpg.add_text(name)
                    dpg.add_spacer(width=8)
                    dpg.add_button(
                        label="Delete",
                        callback=lambda s, a, u=name: self._delete_preset(u),
                    )
            dpg.add_spacer(height=6)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item(
                    self._DELETE_PRESET_TAG, show=False,
                ),
            )

    def _delete_preset(self, name: str) -> bool:
        """Remove a user preset. Returns True if anything was removed.
        Built-in names are silently rejected — the dialog only lists
        deletable names so this is a defence-in-depth guard."""
        if name in self._BUILTIN_PRESET_NAMES:
            return False
        user = dict(self._settings.ui2.user_presets)
        if name not in user:
            return False
        user.pop(name, None)
        self._settings = self._settings.with_ui2(user_presets=user)
        self._rebuild_preset_chips()
        if self._selected_preset == name:
            self._selected_preset = ""
        self._set_status(f"Preset deleted: {name}", level="info")
        self._mark_dirty()
        try:
            dpg.configure_item(self._DELETE_PRESET_TAG, show=False)
        except Exception:
            pass
        return True

    def _rebuild_preset_chips(self) -> None:
        """Re-render the sidebar chip row from the current merged
        preset set — built-ins + user presets."""
        try:
            self._preset_chips.rebuild(list(self._presets().keys()))
        except Exception:
            _LOG.debug("preset chip rebuild failed", exc_info=True)

    def _apply_preset(self, name: str) -> None:
        cfg = self._presets().get(name)
        if not cfg:
            return
        self._params = ReconParams(**{
            **cfg,
            "reference_path": self._params.reference_path,
            "subtract_reference": self._params.subtract_reference,
        })
        for key, tag in (
            ("wavelength_nm", "input_wavelength"),
            ("pixel_um", "input_pixel"),
            ("z_mm", "input_z"),
            ("mask_radius", "input_mask"),
            ("method", "input_method"),
            ("magnification", "input_magnification"),
            ("pixel_is_effective", "input_pixel_is_effective"),
            ("n_sample", "input_n_sample"),
            ("n_medium", "input_n_medium"),
            ("autofocus_metric", "input_af_metric"),
        ):
            try:
                dpg.set_value(tag, cfg[key])
            except Exception:
                pass
        self._selected_preset = name
        self._set_status(f"Preset applied: {name}", level="ok")
        self._toasts.show(f"Preset: {name}", level="info", ttl=2.5)
        self._refresh_info_text()
        self._mark_dirty()

    # ---- Workflow modes -------------------------------------------------
    def _on_workflow_changed(self, sender, app_data, user_data=None) -> None:
        """Apply the chosen workflow — hide/show sidebar sections so the
        mode actually changes what the user sees, not just the toast."""
        mode = str(app_data or "Reconstruct")
        self._workflow_mode = mode
        self._apply_workflow_visibility(mode)
        self._set_status(f"Workflow: {mode}", level="info")
        self._toasts.show(f"Workflow → {mode}", level="info", ttl=2.0)
        self._mark_dirty()

    # ---- Theme switching -----------------------------------------------
    def _apply_theme(self, name: str) -> None:
        try:
            ScientificTheme.apply(name)
            self._theme_name = name
            self._set_status(f"Theme: {name.replace('_', ' ').title()}",
                             level="info")
            self._toasts.show(
                f"Theme changed to “{name.replace('_', ' ').title()}”.",
                level="info", ttl=2.8,
            )
            self._mark_dirty()
        except Exception as exc:
            self._set_status(f"Theme apply failed: {exc}", level="warn")

    # ---- Help surfaces -------------------------------------------------
    def _on_help_overlay(self) -> None:
        show_help_overlay(self._palette)

    def _show_onboarding_manual(self) -> None:
        show_onboarding(on_close=lambda: None)

    def _show_audit_viewer(self) -> None:
        """v2.0.7 T5 — open the audit log browser dialog. Pure
        delegation; data + filter logic lives in
        :mod:`core.audit_viewer`."""
        try:
            show_audit_viewer()
        except Exception as exc:  # pragma: no cover - defensive
            self._set_status(f"Audit viewer failed: {exc}",
                             level="error")

    # ---- Reference hologram -------------------------------------------
    _REF_DIALOG_TAG = "file_dialog_reference"

    def _on_load_reference(self) -> None:
        """Native OS picker first (tkinter), DPG fallback second.
        Same rationale as :meth:`_on_load_clicked` — DPG's custom file
        dialog misbehaves on macOS."""
        path = self._ask_file_via_tk(
            title="Load reference hologram",
            filetypes=[
                ("Reference image", "*.tif *.tiff *.png *.bmp *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
            default_dir=self._last_dir or str(Path.home()),
        )
        if path:
            self._on_reference_selected(None, {"file_path_name": path})
            return
        if dpg.does_item_exist(self._REF_DIALOG_TAG):
            try:
                dpg.configure_item(self._REF_DIALOG_TAG,
                                   default_path=self._last_dir
                                   or str(Path.home()),
                                   show=True)
            except Exception:
                dpg.configure_item(self._REF_DIALOG_TAG, show=True)
            return
        with dpg.file_dialog(
            directory_selector=False, show=True, modal=True,
            callback=self._on_reference_selected,
            tag=self._REF_DIALOG_TAG,
            width=720, height=480,
            default_path=self._last_dir or str(Path.home()),
        ):
            dpg.add_file_extension(
                "Images (*.tif *.tiff *.png *.bmp *.jpg)"
                "{.tif,.tiff,.png,.bmp,.jpg,.jpeg}",
            )
            dpg.add_file_extension(".*")

    def _on_reference_selected(self, sender, app_data) -> None:
        sel = (app_data.get("file_path_name")
               or app_data.get("current_path"))
        if not sel:
            selections = app_data.get("selections") or {}
            if selections:
                sel = next(iter(selections.values()))
        if not sel:
            return
        path = Path(sel)
        if not path.exists():
            self._set_status(f"Reference not found: {path}", level="warn")
            return
        self._params = ReconParams(
            **{**self._params.__dict__,
               "reference_path": path,
               "subtract_reference": True},
        )
        try:
            dpg.set_value("reference_path_label", f"📎 {path.name}")
            dpg.configure_item("reference_path_label",
                               color=ScientificTheme.SUCCESS)
            dpg.set_value("subtract_reference_cb", True)
        except Exception:
            pass
        self._set_status(f"Reference loaded: {path.name}", level="ok")
        self._toasts.show(
            "Reference set — next Reconstruct will divide it out.",
            level="info", ttl=4.0,
        )
        self._refresh_info_text()
        self._mark_dirty()

    def _on_clear_reference(self) -> None:
        self._params = ReconParams(
            **{**self._params.__dict__,
               "reference_path": None,
               "subtract_reference": False},
        )
        try:
            dpg.set_value("reference_path_label", "(none)")
            dpg.configure_item("reference_path_label",
                               color=ScientificTheme.TEXT_MUTED)
            dpg.set_value("subtract_reference_cb", False)
        except Exception:
            pass
        self._set_status("Reference cleared.", level="info")
        self._refresh_info_text()
        self._mark_dirty()

    def _on_ref_toggle(self, checked: bool) -> None:
        self._params = ReconParams(
            **{**self._params.__dict__,
               "subtract_reference": bool(checked)},
        )
        self._refresh_info_text()
        self._mark_dirty()

    # ---- Autofocus (one-shot) -----------------------------------------
    def _on_autofocus(self) -> None:
        if self._current_hologram is None:
            self._set_status("Load a hologram first.", level="warn")
            return
        self._on_param_changed(None, None)
        self._set_status("Running autofocus…", level="info")
        # v2.0.4: scan range + step count now come from ReconParams
        # (sidebar-editable) instead of the v2.0.2 hardcoded -25/+25.
        self._science.run_autofocus(
            self._current_hologram, self._params,
            z_min_mm=float(self._params.af_z_min_mm),
            z_max_mm=float(self._params.af_z_max_mm),
            n_steps=int(self._params.af_n_steps),
            sample_id=self._sample_id,
            on_result=lambda r: self._post_mailbox(("autofocus", r)),
            on_error=lambda e: self._post_mailbox(("error", e)),
        )

    # ---- Multi-focus ---------------------------------------------------
    def _on_find_focus_candidates(self) -> None:
        if self._current_hologram is None:
            self._set_status("Load a hologram first.", level="warn")
            return
        self._on_param_changed(None, None)
        self._set_status("Scanning focus landscape…", level="info")
        # Multi-focus uses a 1.5× step budget relative to one-shot —
        # more steps help the peak finder separate close peaks; user's
        # sidebar step count is still the source of truth.
        self._science.find_focus_candidates(
            self._current_hologram, self._params,
            z_min_mm=float(self._params.af_z_min_mm),
            z_max_mm=float(self._params.af_z_max_mm),
            n_steps=int(round(self._params.af_n_steps * 1.5)),
            sample_id=self._sample_id,
            on_result=lambda r: self._post_mailbox(("multi_focus", r)),
            on_error=lambda e: self._post_mailbox(("error", e)),
        )

    # ---- QPI one-shot --------------------------------------------------
    def _on_compute_qpi(self) -> None:
        if self._current_hologram is None:
            self._set_status("Load a hologram first.", level="warn")
            return
        self._on_param_changed(None, None)
        self._set_status("Computing QPI…", level="info")
        self._science.run_qpi(
            self._current_hologram, self._params,
            z_mm=self._params.z_mm,
            sample_id=self._sample_id,
            on_result=lambda r: self._post_mailbox(("qpi", r)),
            on_error=lambda e: self._post_mailbox(("error", e)),
        )

    # ---- QPI batch -----------------------------------------------------
    def _on_qpi_batch(self) -> None:
        if self._current_hologram is None:
            self._set_status("Load a hologram first.", level="warn")
            return
        self._on_param_changed(None, None)
        self._set_status("Running QPI batch across focus candidates…",
                         level="info")
        self._science.run_qpi_batch(
            self._current_hologram, self._params,
            z_min_mm=float(self._params.af_z_min_mm),
            z_max_mm=float(self._params.af_z_max_mm),
            n_steps=int(round(self._params.af_n_steps * 1.5)),
            sample_id=self._sample_id,
            on_result=lambda r: self._post_mailbox(("qpi_batch", r)),
            on_error=lambda e: self._post_mailbox(("error", e)),
        )

    # ---- Depth map + overlay ------------------------------------------
    def _on_depth_map(self) -> None:
        if self._current_hologram is None:
            self._set_status("Load a hologram first.", level="warn")
            return
        self._on_param_changed(None, None)
        self._set_status("Computing depth map…", level="info")
        self._science.run_depth_map(
            self._current_hologram, self._params,
            z_min_mm=float(self._params.af_z_min_mm),
            z_max_mm=float(self._params.af_z_max_mm),
            n_steps=int(self._params.af_n_steps),
            sample_id=self._sample_id,
            on_result=lambda r: self._post_mailbox(("depth", r)),
            on_error=lambda e: self._post_mailbox(("error", e)),
        )

    def _clear_depth_overlay(self) -> None:
        if self._last_depth is None:
            return
        # Repaint the phase texture from the last recon result to strip
        # the depth tint. No cluster markers to remove — we keep them
        # in the info pane rather than as overlaid graphics.
        if self._last_recon is not None:
            self._push_texture(self.panel_phase.tex_tag,
                               self._last_recon.phase,
                               colormap="phase")
        self._last_depth = None
        self._set_status("Depth overlay cleared.", level="info")
        self._refresh_info_text()

    # ---- Exports -------------------------------------------------------
    def _on_export_report(self) -> None:
        if self._last_recon is None:
            self._set_status("Run reconstruction first.", level="warn")
            return
        default = self._default_export_path("dhm_report", ".html")
        with self._pending_lock:
            self._pending = ("ask_path", {
                "title": "Save HTML report",
                "default": default,
                "suffix": ".html",
                "kind": "report",
            })

    def _on_export_qpi_csv(self) -> None:
        if self._last_qpi is None:
            self._set_status("Compute QPI first.", level="warn")
            return
        default = self._default_export_path("qpi", ".csv")
        with self._pending_lock:
            self._pending = ("ask_path", {
                "title": "Save QPI CSV",
                "default": default,
                "suffix": ".csv",
                "kind": "qpi_csv",
            })

    def _on_export_bundle(self) -> None:
        if self._last_depth is None:
            self._set_status(
                "Compute the depth map first (Tools → Compute depth map).",
                level="warn",
            )
            return
        default = str(Path.home())
        with self._pending_lock:
            self._pending = ("ask_dir", {
                "title": "Select tomography bundle directory",
                "default": default,
                "kind": "bundle",
            })

    def _default_export_path(self, stem: str, suffix: str) -> str:
        from datetime import datetime
        name = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
        base = (self._current_hologram.parent if self._current_hologram
                else Path.home())
        return str(base / name)

    # ---- Command palette registry --------------------------------------
    def _register_commands(self) -> None:
        self._palette.clear()
        r = self._palette.register
        r("file.load", "Load hologram…",
          self._on_load_clicked,
          hint="Open file dialog", category="File")
        r("file.reference", "Load reference hologram…",
          self._on_load_reference,
          hint="Optional reference wave", category="File")
        r("view.palette", "Command palette",
          self._palette.show,
          hint="Open this list (Ctrl+K)", category="View")
        for key in PALETTES.keys():
            label = f"Theme: {key.replace('_', ' ').title()}"
            r(f"view.theme.{key}", label,
              (lambda k=key: (lambda: self._apply_theme(k)))(),
              hint="Palette switch", category="View")
        r("recon.run", "Reconstruct", self._on_reconstruct,
          hint="Ctrl+R", category="Reconstruct")
        r("autofocus.one_shot", "Autofocus (one-shot)",
          self._on_autofocus, hint="Scan + pick best z",
          category="Autofocus")
        r("autofocus.multi", "Find multiple focus planes",
          self._on_find_focus_candidates,
          hint="Landscape peaks", category="Autofocus")
        r("qpi.compute", "Compute QPI", self._on_compute_qpi,
          hint="Dry mass, height, roughness", category="QPI")
        r("qpi.batch", "Run QPI batch for focus candidates",
          self._on_qpi_batch,
          hint="One QPI per candidate", category="QPI")
        r("depth.compute", "Compute depth map + overlay",
          self._on_depth_map,
          hint="Per-pixel best-focus z", category="Depth")
        r("depth.clear", "Clear depth overlay",
          self._clear_depth_overlay,
          hint="Restore plain phase panel", category="Depth")
        r("tools.export_report", "Export HTML report…",
          self._on_export_report, category="Export")
        r("tools.export_qpi_csv", "Export QPI CSV…",
          self._on_export_qpi_csv, category="Export")
        r("tools.export_bundle", "Export tomography bundle…",
          self._on_export_bundle, category="Export")
        r("help.overlay", "Help overlay (?)",
          self._on_help_overlay, category="Help")
        r("help.shortcuts", "Keyboard shortcuts",
          self._show_shortcuts, category="Help")
        r("help.onboarding", "Onboarding",
          self._show_onboarding_manual, category="Help")

    # ---- Mailbox posting helper ---------------------------------------
    def _post_mailbox(self, item: Any) -> None:
        with self._pending_lock:
            self._pending = item

    def _show_about(self) -> None:
        if dpg.does_item_exist("about_modal"):
            dpg.configure_item("about_modal", show=True)
            return
        with dpg.window(label="About", modal=True, no_resize=True,
                        tag="about_modal", width=420, height=180,
                        pos=(200, 200)):
            dpg.add_text("DHM Reconstruction — v2 UI")
            dpg.add_text("Dear PyGui frontend for the same scientific pipeline.",
                         color=ScientificTheme.TEXT_MUTED, wrap=380)
            dpg.add_separator()
            dpg.add_text(
                "Core pipeline: core.reconstruction / core.offaxis / "
                "core.autofocus / core.qpi / core.depth_map",
                wrap=380, color=ScientificTheme.TEXT_MUTED,
            )
            dpg.add_spacer(height=8)
            dpg.add_button(label="Close",
                           callback=lambda: dpg.configure_item(
                               "about_modal", show=False))

    # ---- thread-safe mailbox from the driver ---------------------------

    def _post_result(self, result: ReconResult) -> None:
        with self._pending_lock:
            self._pending = result

    def _post_error(self, err: ReconError) -> None:
        with self._pending_lock:
            self._pending = err

    def _drain_mailbox(self) -> None:
        with self._pending_lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return

        # Legacy reconstruction path — posted as bare objects by
        # ReconstructionDriver (older API).
        if isinstance(pending, ReconResult):
            self._apply_result(pending)
            return
        if isinstance(pending, ReconError):
            self._set_status(f"Reconstruction failed: {pending.message}",
                             level="danger")
            dpg.configure_item("btn_reconstruct", enabled=True)
            return

        # New tagged path: ("kind", payload)
        if isinstance(pending, tuple) and len(pending) == 2:
            kind, payload = pending
            handler = self._mailbox_handlers().get(kind)
            if handler is not None:
                try:
                    handler(payload)
                except Exception:
                    _LOG.exception("mailbox handler for %r failed", kind)

    def _mailbox_handlers(self) -> dict:
        return {
            "error": self._handle_error,
            "autofocus": self._handle_autofocus,
            "multi_focus": self._handle_multi_focus,
            "qpi": self._handle_qpi,
            "qpi_batch": self._handle_qpi_batch,
            "depth": self._handle_depth,
            "report": self._handle_report,
            "qpi_csv": self._handle_qpi_csv_written,
            "qpi_batch_csv": self._handle_qpi_batch_csv_written,
            "bundle": self._handle_bundle,
            "ask_path": self._handle_ask_path,
            "ask_dir": self._handle_ask_dir,
            # Fire-and-forget status updates from background workers.
            "status": lambda msg: self._set_status(str(msg), level="info"),
            # Batch-progress table updates posted by ``_run_batch``.
            "batch_row": self._handle_batch_row,
            "batch_done": self._handle_batch_done,
            # Camera live feed — frame every tick, fps every ~1 s.
            "camera_frame": self._handle_camera_frame,
            "camera_fps": self._handle_camera_fps,
        }

    # ---- individual mailbox handlers ----------------------------------
    def _handle_error(self, err: ReconError) -> None:
        self._set_status(f"Error: {err.message}", level="danger")
        self._toasts.show(err.message, level="danger", ttl=6.0)
        self._append_error("danger", err.message)
        # Reconstruction paths disable btn_reconstruct before starting;
        # re-enable it here so the next attempt is possible even if the
        # worker path blew up.
        try:
            if self._current_hologram is not None:
                dpg.configure_item("btn_reconstruct", enabled=True)
                self._refresh_reconstruct_tooltip(enabled=True)
        except Exception:
            pass

    def _handle_autofocus(self, result: AutofocusResult) -> None:
        self._last_autofocus = result
        z_mm = result.best_z_m * 1e3
        self._params = ReconParams(**{**self._params.__dict__, "z_mm": z_mm})
        try:
            dpg.set_value("input_z", float(z_mm))
        except Exception:
            pass
        msg = (f"Autofocus: best z = {z_mm:+.3f} mm "
               f"({result.runtime_ms:.0f} ms)")
        self._set_status(msg, level="ok")
        self._toasts.show(msg, level="ok")

    def _handle_multi_focus(self, result: MultiFocusResult) -> None:
        n = len(result.candidates)
        self._set_status(
            f"Multi-focus scan: {n} candidate(s) "
            f"({result.runtime_ms:.0f} ms)",
            level="ok",
        )
        show_focus_candidates(
            result.candidates,
            on_focus=lambda z_mm: self._apply_z(z_mm),
        )

    def _apply_z(self, z_mm: float) -> None:
        self._params = ReconParams(**{**self._params.__dict__, "z_mm": z_mm})
        try:
            dpg.set_value("input_z", float(z_mm))
        except Exception:
            pass
        self._on_reconstruct()

    def _handle_qpi(self, result: QPIOneShotResult) -> None:
        self._last_qpi = result
        q = result.qpi
        lines = [f"QPI OK — {result.runtime_ms:.0f} ms"]
        if q.phase_stats is not None and q.phase_stats.range_nm is not None:
            lines.append(f"OPD range: {q.phase_stats.range_nm:.2f} nm")
        if q.total_dry_mass_pg is not None:
            lines.append(f"Dry mass: {q.total_dry_mass_pg:.2f} pg")
        if q.step_height_m is not None:
            lines.append(f"Step: {q.step_height_m * 1e9:.2f} nm")
        msg = " · ".join(lines)
        self._set_status(msg, level="ok")
        self._toasts.show(msg, level="ok", ttl=6.0)
        # Re-render from caches instead of appending — keeps the info
        # panel deterministic even when QPI is re-run on the same recon.
        self._refresh_info_text()

    def _handle_qpi_batch(self, result: QPIBatchResultWrap) -> None:
        self._last_qpi_batch = result
        n = len(result.entries)
        self._set_status(
            f"QPI batch: {n} entry(s) ({result.runtime_ms:.0f} ms)",
            level="ok",
        )
        show_qpi_batch_review(
            result.entries,
            on_focus=lambda z_mm: self._apply_z(z_mm),
            on_export_csv=lambda: self._ask_qpi_batch_csv_path(),
        )

    def _ask_qpi_batch_csv_path(self) -> None:
        default = self._default_export_path("qpi_batch", ".csv")
        with self._pending_lock:
            self._pending = ("ask_path", {
                "title": "Save QPI batch CSV",
                "default": default,
                "suffix": ".csv",
                "kind": "qpi_batch_csv",
            })

    def _handle_depth(self, result: DepthMapResultWrap) -> None:
        self._last_depth = result
        z_min = result.result.z_map.min() * 1e3
        z_max = result.result.z_map.max() * 1e3
        msg = (f"Depth map: z∈[{z_min:+.2f}, {z_max:+.2f}] mm, "
               f"{len(result.clusters)} cluster(s), "
               f"{result.runtime_ms:.0f} ms")
        self._set_status(msg, level="ok")
        self._toasts.show(msg, level="ok", ttl=6.0)
        # Tint the phase panel with a real depth colormap — z is
        # physical metres, so a periodic phase wheel would falsely
        # wrap near the extremes (v2.0.1 bug).
        try:
            self._push_texture(self.panel_phase.tex_tag,
                               result.result.z_map,
                               colormap="depth")
        except Exception:
            _LOG.debug("depth overlay push failed", exc_info=True)
        self._refresh_info_text()

    def _handle_report(self, result: ReportExportResult) -> None:
        msg = f"Report saved: {result.path}"
        self._set_status(msg, level="ok")
        self._toasts.show(msg, level="ok", ttl=5.0)

    def _handle_qpi_csv_written(self, result: Path) -> None:
        msg = f"QPI CSV saved: {result}"
        self._set_status(msg, level="ok")
        self._toasts.show(msg, level="ok", ttl=4.0)

    def _handle_qpi_batch_csv_written(self, result: Path) -> None:
        msg = f"QPI batch CSV saved: {result}"
        self._set_status(msg, level="ok")
        self._toasts.show(msg, level="ok", ttl=4.0)

    def _handle_bundle(self, result: BundleExportResult) -> None:
        msg = f"Bundle saved: {len(result.files)} file(s)"
        self._set_status(msg, level="ok")
        self._toasts.show(msg, level="ok", ttl=5.0)

    def _handle_ask_path(self, info: dict) -> None:
        """Open a save-file dialog; when the user picks a path, actually
        run the export for ``info['kind']``. Uses the native OS save
        picker (tkinter) — DPG's custom dialog eats the callback on
        macOS, so exports silently never landed."""
        kind = info.get("kind")
        default = Path(info.get("default") or "")
        suffix = info.get("suffix") or ""
        tk_picked = self._ask_file_via_tk(
            title=info.get("title") or "Save",
            filetypes=[
                (f"{suffix.lstrip('.').upper() or 'File'}",
                 f"*{suffix}" if suffix else "*.*"),
                ("All files", "*.*"),
            ],
            default_dir=str(default.parent) if default else "",
            save=True,
            default_filename=default.name if default else "",
        )
        if tk_picked:
            out = Path(tk_picked)
            if suffix and out.suffix.lower() != suffix:
                out = out.with_suffix(suffix)
            self._perform_export(kind, out)
            return
        tag = f"save_dialog_{kind}_{id(info)}"

        def _on_picked(sender, app_data):
            sel = (app_data.get("file_path_name")
                   or app_data.get("current_path"))
            if not sel:
                selections = app_data.get("selections") or {}
                if selections:
                    sel = next(iter(selections.values()))
            if not sel:
                return
            out = Path(sel)
            if suffix and out.suffix.lower() != suffix:
                out = out.with_suffix(suffix)
            self._perform_export(kind, out)

        with dpg.file_dialog(
            directory_selector=False, show=True, modal=True,
            callback=_on_picked, tag=tag,
            width=720, height=480,
            default_path=str(default.parent) if default else "",
            default_filename=default.name if default else "",
        ):
            if suffix:
                ext = suffix.lstrip(".")
                dpg.add_file_extension(f".{ext}")
            dpg.add_file_extension(".*")

    def _handle_ask_dir(self, info: dict) -> None:
        kind = info.get("kind")
        tk_picked = self._ask_file_via_tk(
            title=info.get("title") or "Select directory",
            filetypes=[],
            default_dir=str(info.get("default") or ""),
            ask_directory=True,
        )
        if tk_picked:
            self._perform_export(kind, Path(tk_picked))
            return
        tag = f"dir_dialog_{kind}_{id(info)}"

        def _on_picked(sender, app_data):
            sel = (app_data.get("file_path_name")
                   or app_data.get("current_path"))
            if not sel:
                return
            self._perform_export(kind, Path(sel))

        with dpg.file_dialog(
            directory_selector=True, show=True, modal=True,
            callback=_on_picked, tag=tag,
            width=720, height=480,
            default_path=str(info.get("default") or ""),
        ):
            pass

    def _perform_export(self, kind: str, path: Path) -> None:
        if kind == "report":
            if self._last_recon is None:
                return
            self._set_status(f"Writing report to {path.name}…", level="info")
            self._science.export_report(
                path,
                last_recon_params={
                    "wavelength_m": float(self._params.wavelength_nm) * 1e-9,
                    "pixel_size_m": float(self._params.pixel_um) * 1e-6,
                    "z_m": float(self._params.z_mm) * 1e-3,
                    "method": self._params.method,
                    "mask_radius": int(self._params.mask_radius),
                },
                phase_image=self._last_recon.phase,
                amplitude_image=self._last_recon.amplitude,
                sample_id=self._sample_id,
                on_result=lambda r: self._post_mailbox(("report", r)),
                on_error=lambda e: self._post_mailbox(("error", e)),
            )
            return
        if kind == "qpi_csv":
            if self._last_qpi is None:
                return
            try:
                write_qpi_csv(
                    path, self._last_qpi.qpi,
                    sample_id=self._sample_id,
                )
                self._post_mailbox(("qpi_csv", path))
            except Exception as exc:
                self._post_mailbox(("error", ReconError(str(exc))))
            return
        if kind == "qpi_batch_csv":
            if self._last_qpi_batch is None:
                return
            try:
                write_qpi_batch_csv(
                    path, self._last_qpi_batch.entries,
                    sample_id=self._sample_id,
                )
                self._post_mailbox(("qpi_batch_csv", path))
            except Exception as exc:
                self._post_mailbox(("error", ReconError(str(exc))))
            return
        if kind == "bundle":
            if self._last_depth is None:
                return
            entries = (self._last_qpi_batch.entries
                       if self._last_qpi_batch is not None else [])
            self._set_status("Writing tomography bundle…", level="info")
            self._science.export_bundle(
                path,
                depth_result=self._last_depth.result,
                clusters=self._last_depth.clusters,
                qpi_entries=entries,
                sample_id=self._sample_id,
                pixel_size_m=float(self._params.pixel_um) * 1e-6,
                on_result=lambda r: self._post_mailbox(("bundle", r)),
                on_error=lambda e: self._post_mailbox(("error", e)),
            )
            return

    def _apply_result(self, result: ReconResult) -> None:
        self._last_recon = result
        # QPI/depth were computed against the previous recon — clear
        # their caches so the info panel doesn't show stale metrics.
        self._last_qpi = None
        self._last_depth = None
        # Start with btn_surface disabled; only enable *after* the
        # phase texture is successfully painted. v2.0.1 enabled it up
        # front, so if the texture push threw the 3D button pointed at
        # an empty panel.
        surface_ready = False
        try:
            dpg.configure_item("btn_surface", enabled=False)
        except Exception:
            pass
        try:
            self._push_texture(self.panel_input.tex_tag, result.input_image)
            self._push_texture(self.panel_amp.tex_tag, result.amplitude)
            self._push_texture(self.panel_phase.tex_tag, result.phase)
            surface_ready = (result.phase.size > 0)
            self._refresh_info_text()
            self._set_status(
                f"Reconstruction OK — {result.runtime_ms:.0f} ms",
                level="ok",
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Display failed: {exc}", level="danger")
            surface_ready = False
        finally:
            try:
                dpg.configure_item("btn_reconstruct", enabled=True)
                self._refresh_reconstruct_tooltip(enabled=True)
                dpg.configure_item("btn_surface", enabled=surface_ready)
            except Exception:
                pass

    # ---- helpers --------------------------------------------------------

    def _push_texture(
        self, tag: str, arr: np.ndarray, *, colormap: str = "gray",
    ) -> None:
        """Legacy shim: routes to whichever :class:`ZoomableImagePanel`
        owns ``tag``. Keeps existing call sites (set_image on load,
        recon result apply, depth-map overlay) working without
        touching each one.

        v2.0.7: the *amplitude* panel can switch to percentile-based
        contrast so reference-divided images stay readable even when
        zero-reference pixels blow out the histogram. The toggle is
        controlled by ``ReconParams.auto_contrast_amplitude``; phase
        + input keep their usual min/max behaviour so users can read
        physical ranges off the colour bar."""
        panel = self._panel_for_tex(tag)
        if panel is not None:
            # Palette override: amplitude/input are gray, phase uses
            # the phase wheel. Tag match decides which, not the arg —
            # except for depth, which the handler explicitly routes as
            # ``colormap="depth"`` and expects that to win.
            if colormap == "depth":
                panel_colormap = panel.colormap
                panel.colormap = "depth"
                try:
                    panel.set_image(arr)
                finally:
                    panel.colormap = panel_colormap
                return
            contrast = "minmax"
            is_amp_panel = panel is getattr(self, "panel_amp", None)
            if is_amp_panel and self._params.auto_contrast_amplitude:
                contrast = "percentile"
            panel.set_image(arr, contrast=contrast)
            return
        # Fallback — direct texture write.
        from .image_panel import _resize
        resized = _resize(arr, self._preview_size, self._preview_size)
        rgba = _to_rgba(resized, colormap=colormap)
        dpg.set_value(tag, rgba.flatten())

    def _panel_for_tex(self, tag: str):
        for panel in (getattr(self, "panel_input", None),
                      getattr(self, "panel_amp", None),
                      getattr(self, "panel_phase", None)):
            if panel is not None and panel.tex_tag == tag:
                return panel
        return None

    # ---- Quick-access toolbar -----------------------------------------
    def _build_quick_toolbar(self) -> None:
        with dpg.group(horizontal=True, tag="quick_toolbar"):
            items = [
                ("📂  Load", self._on_load_clicked, "Open a hologram (Ctrl+O)"),
                ("⬢  Reconstruct", self._on_reconstruct, "Run reconstruction (Ctrl+R)"),
                ("⊹  Autofocus", self._on_autofocus, "One-shot autofocus"),
                ("∿  Multi-focus", self._on_find_focus_candidates,
                 "Find multiple focus planes"),
                ("⚛  QPI", self._on_compute_qpi, "Compute QPI at current z"),
                ("▥  Depth", self._on_depth_map, "Compute depth map + overlay"),
                ("⇪  Report", self._on_export_report, "Export HTML report"),
                ("⌘  Palette", self._palette.show, "Command palette (Ctrl+K)"),
            ]
            for label, cb, hint in items:
                dpg.add_button(label=label, callback=cb)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(hint)

    # ---- Surface preview ---------------------------------------------
    def _on_open_surface(self) -> None:
        """Phase → 3D surface subprocess."""
        if self._last_recon is None:
            self._toasts.show("Reconstruct first to view phase in 3D.",
                              level="warn")
            return
        phase = self._last_recon.phase
        try:
            open_surface(phase, title=f"Phase — {self._current_hologram.name if self._current_hologram else 'hologram'}")
            self._toasts.show("3D surface window opened.", level="ok",
                              ttl=3.0)
        except Exception as exc:
            self._set_status(f"Surface failed: {exc}", level="danger")

    def _reset_all_zoom(self) -> None:
        for panel in (self.panel_input, self.panel_amp, self.panel_phase):
            try:
                panel.reset_zoom()
            except Exception:
                pass

    # ---- UI-side crash wrapper ----------------------------------------

    def _install_ui_crash_wrapper(self) -> None:
        """Wrap whatever crash handler is currently in ``sys.excepthook``
        with a UI layer: toast + error-drawer entry + state flush, then
        defer to the existing handler so the crash dump + audit record
        still happens. Restore nothing — we stay installed for the
        lifetime of the app."""
        prev = sys.excepthook

        def _ui_hook(exc_type, exc_value, exc_tb):
            # Run the UI bits FIRST so the user sees feedback even if
            # the underlying handler is slow / broken on disk.
            try:
                if not issubclass(exc_type, KeyboardInterrupt):
                    msg = f"{exc_type.__name__}: {exc_value}"
                    self._append_error("danger", msg)
                    try:
                        self._toasts.show(
                            "Crash logged — see View → Errors.",
                            level="danger", ttl=8.0,
                        )
                    except Exception:
                        pass
                    try:
                        self._saver.flush_now()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                prev(exc_type, exc_value, exc_tb)
            except Exception:
                sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = _ui_hook

    # ---- Error drawer ---------------------------------------------------

    _ERROR_LOG_CAP = 64

    def _append_error(self, level: str, message: str) -> None:
        """Add an entry to the bounded error log. Called from
        ``_handle_error`` and status-level warn/danger sites."""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self._error_log.append((ts, level, message))
        if len(self._error_log) > self._ERROR_LOG_CAP:
            self._error_log = self._error_log[-self._ERROR_LOG_CAP:]

    def _show_error_drawer(self) -> None:
        """Open (or refresh) the error-history modal. Shows last 64
        warn+danger events with timestamps; includes a Clear button."""
        tag = "error_drawer"
        if dpg.does_item_exist(tag):
            self._refresh_error_drawer()
            dpg.configure_item(tag, show=True)
            return
        with dpg.window(label="Errors & warnings", modal=True,
                        tag=tag, width=560, height=440,
                        pos=(160, 140), no_collapse=True):
            dpg.add_text(
                "Session error log — most recent at the bottom.",
                color=ScientificTheme.TEXT_MUTED, wrap=540,
            )
            dpg.add_separator()
            with dpg.child_window(tag="error_drawer_list",
                                  autosize_x=True, height=-40,
                                  border=False):
                pass
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Clear",
                    callback=self._clear_error_log,
                )
                dpg.add_button(
                    label="Close",
                    callback=lambda: dpg.configure_item(tag, show=False),
                )
        self._refresh_error_drawer()

    def _refresh_error_drawer(self) -> None:
        if not dpg.does_item_exist("error_drawer_list"):
            return
        try:
            dpg.delete_item("error_drawer_list", children_only=True)
        except Exception:
            return
        if not self._error_log:
            dpg.add_text("(no errors yet — nice)",
                         parent="error_drawer_list",
                         color=ScientificTheme.TEXT_MUTED)
            return
        for ts, level, msg in self._error_log:
            color = (ScientificTheme.DANGER if level == "danger"
                     else ScientificTheme.WARN)
            dpg.add_text(f"[{ts}] {level.upper()}: {msg}",
                         parent="error_drawer_list",
                         color=color, wrap=520)

    def _clear_error_log(self) -> None:
        self._error_log.clear()
        self._refresh_error_drawer()

    # ---- Autofocus ROI (local-focus mask) -------------------------------

    def _show_roi_dialog(self) -> None:
        """Modal to dial in a normalised (y0, x0, y1, x1) ROI in [0, 1].
        When set, autofocus / multi-focus / depth scans restrict their
        metric to that rectangle — useful for scenes with multiple
        objects at different z values ("focus on THIS cell")."""
        tag = "roi_dialog"
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=True)
            return
        existing = self._params.af_roi or (0.25, 0.25, 0.75, 0.75)
        y0, x0, y1, x1 = existing
        with dpg.window(label="Autofocus ROI", modal=True,
                        tag=tag, width=380, height=260,
                        pos=(220, 180), no_collapse=True):
            dpg.add_text(
                "Normalised rectangle (0 = top-left, 1 = bottom-right). "
                "Autofocus / multi-focus / depth scans will only see "
                "pixels inside this region.",
                color=ScientificTheme.TEXT_MUTED, wrap=340,
            )
            dpg.add_separator()
            dpg.add_slider_float(label="y0", default_value=float(y0),
                                 min_value=0.0, max_value=1.0,
                                 format="%.3f", tag="roi_y0")
            dpg.add_slider_float(label="x0", default_value=float(x0),
                                 min_value=0.0, max_value=1.0,
                                 format="%.3f", tag="roi_x0")
            dpg.add_slider_float(label="y1", default_value=float(y1),
                                 min_value=0.0, max_value=1.0,
                                 format="%.3f", tag="roi_y1")
            dpg.add_slider_float(label="x1", default_value=float(x1),
                                 min_value=0.0, max_value=1.0,
                                 format="%.3f", tag="roi_x1")
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Apply",
                               callback=self._apply_roi_from_dialog)
                dpg.add_button(label="Clear",
                               callback=lambda: (self._clear_af_roi(),
                                                 dpg.configure_item(
                                                     tag, show=False)))
                dpg.add_button(
                    label="Close",
                    callback=lambda: dpg.configure_item(tag, show=False),
                )

    def _apply_roi_from_dialog(self) -> None:
        try:
            y0 = float(dpg.get_value("roi_y0"))
            x0 = float(dpg.get_value("roi_x0"))
            y1 = float(dpg.get_value("roi_y1"))
            x1 = float(dpg.get_value("roi_x1"))
        except Exception:
            self._set_status("ROI read failed", level="warn")
            return
        if y1 <= y0 or x1 <= x0:
            self._set_status("ROI has zero area — ignoring.", level="warn")
            return
        self._params = ReconParams(**{
            **self._params.__dict__,
            "af_roi": (y0, x0, y1, x1),
        })
        self._set_status(
            f"Autofocus ROI set: y∈[{y0:.2f},{y1:.2f}] "
            f"x∈[{x0:.2f},{x1:.2f}]",
            level="ok",
        )
        self._toasts.show("Autofocus ROI armed.", level="ok", ttl=3.0)
        self._refresh_info_text()
        self._mark_dirty()
        try:
            dpg.configure_item("roi_dialog", show=False)
        except Exception:
            pass

    def _clear_af_roi(self) -> None:
        if self._params.af_roi is None:
            return
        self._params = ReconParams(**{
            **self._params.__dict__, "af_roi": None,
        })
        self._set_status("Autofocus ROI cleared.", level="info")
        self._refresh_info_text()
        self._mark_dirty()

    # ---- Line profile ---------------------------------------------------

    @staticmethod
    def _sample_line(phase: np.ndarray,
                     p1: tuple[float, float],
                     p2: tuple[float, float],
                     n: int = 512) -> np.ndarray:
        """Bilinear-interpolated sample along the line ``p1 → p2``.

        Points are in plot coordinates (x right, y down) matching the
        image series bounds the phase panel uses. Out-of-bounds pixels
        clamp to the nearest edge so drawing a line that grazes the
        frame boundary doesn't crash with a NaN row.
        """
        from scipy.ndimage import map_coordinates
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        xs = np.linspace(x1, x2, int(max(2, n)))
        ys = np.linspace(y1, y2, int(max(2, n)))
        # map_coordinates expects (row, col) = (y, x) order.
        return map_coordinates(
            phase.astype(np.float64, copy=False),
            [ys, xs], order=1, mode="nearest",
        )

    def _show_line_profile(self) -> None:
        """Pop a dialog showing the horizontal phase profile through
        the image centre. Kept as a fast default; the interactive
        click-drag variant is ``_enable_line_profile_mode``."""
        if self._last_recon is None:
            self._set_status("Reconstruct first to inspect a line profile.",
                             level="warn")
            return
        phase = self._last_recon.phase
        row = int(phase.shape[0] // 2)
        line = phase[row, :].astype("float64").tolist()
        xs = list(range(len(line)))
        self._render_line_profile_window(
            xs, line, label=f"row {row}",
            title=f"Phase profile — row {row}",
        )

    def _render_line_profile_window(
        self,
        xs: list,
        values: list,
        *,
        label: str,
        title: str,
    ) -> None:
        tag = "line_profile_win"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        with dpg.window(label=title,
                        tag=tag, width=640, height=360,
                        pos=(200, 160), no_collapse=True):
            dpg.add_text(
                "Phase values along the selected line. Hover for "
                "values; Esc closes.",
                color=ScientificTheme.TEXT_MUTED, wrap=600,
            )
            with dpg.plot(label="Phase (rad)", height=-1, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, label="sample")
                with dpg.plot_axis(dpg.mvYAxis, label="φ (rad)"):
                    dpg.add_line_series(list(xs), list(values),
                                        label=label)

    def _enable_line_profile_mode(self) -> None:
        """Toggle interactive line-draw mode. Click-drag on the phase
        panel samples a bilinear profile between the endpoints. v1
        offered this via an amplitude-panel right-click; v2 exposes
        it as a dedicated menu item so the gesture is discoverable."""
        if self._last_recon is None:
            self._set_status(
                "Reconstruct first to draw a line on the phase panel.",
                level="warn",
            )
            return
        self._line_mode = True
        self._line_p1 = None
        self._line_p2 = None
        self._set_status(
            "Line draw: click-drag on the Phase panel. Esc to cancel.",
            level="info",
        )
        self._ensure_line_profile_handlers()

    def _ensure_line_profile_handlers(self) -> None:
        """Install mouse handlers once — subsequent calls are no-ops.
        The handlers check ``self._line_mode`` before reacting so
        other panels' mouse input isn't hijacked."""
        if getattr(self, "_line_handlers_installed", False):
            return
        try:
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(
                    button=dpg.mvMouseButton_Left,
                    callback=self._on_phase_mouse_down,
                )
                dpg.add_mouse_release_handler(
                    button=dpg.mvMouseButton_Left,
                    callback=self._on_phase_mouse_release,
                )
            self._line_handlers_installed = True
        except Exception:
            _LOG.debug("line handlers install failed", exc_info=True)

    def _on_phase_mouse_down(self, sender=None, app_data=None) -> None:
        if not getattr(self, "_line_mode", False):
            return
        plot = getattr(self.panel_phase, "plot_tag", None)
        if plot is None or not dpg.is_item_hovered(plot):
            return
        try:
            x, y = dpg.get_plot_mouse_pos()
        except Exception:
            return
        self._line_p1 = (float(x), float(y))
        self._line_p2 = None

    def _on_phase_mouse_release(self, sender=None, app_data=None) -> None:
        if not getattr(self, "_line_mode", False):
            return
        if self._line_p1 is None:
            return
        plot = getattr(self.panel_phase, "plot_tag", None)
        if plot is None:
            return
        try:
            x, y = dpg.get_plot_mouse_pos()
        except Exception:
            return
        self._line_p2 = (float(x), float(y))
        # Exit mode as soon as the user finishes a gesture — safer
        # than leaving it armed and surprising the user on their next
        # pan/zoom.
        self._line_mode = False
        if self._line_p1 == self._line_p2:
            self._set_status("Line too short — pick two distinct points.",
                             level="warn")
            return
        try:
            values = self._sample_line(
                self._last_recon.phase, self._line_p1, self._line_p2,
            )
        except Exception as exc:
            self._set_status(f"Line sample failed: {exc}", level="danger")
            return
        xs = list(range(len(values)))
        label = (f"({self._line_p1[0]:.0f},{self._line_p1[1]:.0f}) → "
                 f"({self._line_p2[0]:.0f},{self._line_p2[1]:.0f})")
        self._render_line_profile_window(
            xs, values.tolist(),
            label=label,
            title=f"Phase profile — {label}",
        )
        self._set_status("Line profile rendered.", level="ok")

    # ---- Batch mode dialog ---------------------------------------------

    # ---- Batch reconstruct (dir → HDF5 bundle + optional PNG) ---------

    _BATCH_DIALOG_TAG = "batch_dialog_modal"

    def _on_batch_reconstruct(self) -> None:
        """Open the directory picker, then hand the file list to the
        batch dialog that exposes output mode selection + progress
        table. Prefers the native OS picker (tkinter) — DPG's
        directory selector is the same widget as its file_dialog and
        fails the same way on macOS."""
        path = self._ask_file_via_tk(
            title="Select a directory of holograms",
            filetypes=[],
            default_dir=self._last_dir or str(Path.home()),
            ask_directory=True,
        )
        if path:
            self._on_batch_dir_picked(None, {"file_path_name": path})
            return
        tag = "batch_dir_dialog"
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=True)
            return
        with dpg.file_dialog(
            directory_selector=True, show=True, modal=True,
            callback=self._on_batch_dir_picked,
            tag=tag, width=720, height=480,
            default_path=self._last_dir or str(Path.home()),
        ):
            pass

    def _on_batch_dir_picked(self, sender, app_data) -> None:
        sel = (app_data.get("file_path_name")
               or app_data.get("current_path"))
        if not sel:
            selections = app_data.get("selections") or {}
            if selections:
                sel = next(iter(selections.values()))
        if not sel:
            return
        dirpath = Path(sel)
        files = sorted(
            p for p in dirpath.iterdir()
            if p.is_file() and p.suffix.lower() in self._IMAGE_EXTS
        )
        if not files:
            self._set_status(
                f"No supported holograms in {dirpath.name}.",
                level="warn",
            )
            return
        self._open_batch_dialog(files, dirpath)

    def _open_batch_dialog(self, files: list[Path], dirpath: Path) -> None:
        """Build or refresh the batch dialog with a per-file progress
        table and an output-mode combo (PNG / HDF5 bundle / both)."""
        self._batch_files = list(files)
        self._batch_dir = dirpath
        if dpg.does_item_exist(self._BATCH_DIALOG_TAG):
            dpg.delete_item(self._BATCH_DIALOG_TAG)
        from datetime import datetime
        default_h5 = dirpath / (
            f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        )
        with dpg.window(label=f"Batch reconstruct — {dirpath.name}",
                        modal=True, tag=self._BATCH_DIALOG_TAG,
                        no_resize=False, no_collapse=True,
                        width=720, height=520, pos=(120, 100)):
            dpg.add_text(
                f"{len(files)} hologram(s) found. Choose output "
                "format and click Start. Esc cancels the run.",
                color=ScientificTheme.TEXT_MUTED, wrap=680,
            )
            dpg.add_separator()
            dpg.add_combo(
                items=["PNG per file", "HDF5 bundle", "Both"],
                default_value="Both",
                label="Output", tag="batch_output_mode",
                width=220,
            )
            dpg.add_input_text(
                label="HDF5 path",
                default_value=str(default_h5),
                tag="batch_h5_path",
                width=-1,
            )
            with dpg.table(header_row=True, tag="batch_table",
                           resizable=True, row_background=True,
                           borders_outerH=True, borders_innerV=True,
                           borders_innerH=True, borders_outerV=True,
                           height=-80):
                dpg.add_table_column(label="#")
                dpg.add_table_column(label="File")
                dpg.add_table_column(label="Status")
                dpg.add_table_column(label="Runtime (ms)")
                for idx, f in enumerate(files, start=1):
                    with dpg.table_row():
                        dpg.add_text(str(idx))
                        dpg.add_text(f.name)
                        dpg.add_text("queued",
                                     tag=self._batch_status_tag(idx),
                                     color=ScientificTheme.TEXT_MUTED)
                        dpg.add_text("—",
                                     tag=self._batch_runtime_tag(idx))
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start",
                    tag="batch_start_btn",
                    callback=lambda: self._start_batch_run(),
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: self._on_key_escape(),
                )
                dpg.add_button(
                    label="Close",
                    callback=lambda: dpg.configure_item(
                        self._BATCH_DIALOG_TAG, show=False,
                    ),
                )

    @staticmethod
    def _batch_status_tag(idx: int) -> str:
        return f"batch_status_{idx}"

    @staticmethod
    def _batch_runtime_tag(idx: int) -> str:
        return f"batch_rt_{idx}"

    def _start_batch_run(self) -> None:
        """Kick off the worker thread. Uses the dialog's live settings
        so the user can pick output mode after the dialog opens."""
        try:
            mode = str(dpg.get_value("batch_output_mode") or "Both")
        except Exception:
            mode = "Both"
        try:
            h5_path = Path(dpg.get_value("batch_h5_path")
                           or "").expanduser()
        except Exception:
            h5_path = None
        try:
            dpg.configure_item("batch_start_btn", enabled=False)
        except Exception:
            pass
        self._run_batch(
            self._batch_files,
            output_mode=mode,
            h5_path=h5_path if mode != "PNG per file" else None,
        )

    def _run_batch(
        self,
        files: list[Path],
        *,
        output_mode: str = "PNG per file",
        h5_path: Optional[Path] = None,
    ) -> None:
        """Background-thread worker. Posts per-file row updates on the
        mailbox so the main loop can paint the table without owning
        any threading details."""
        total = len(files)
        self._set_status(
            f"Batch: starting {total} reconstructions ({output_mode})…",
            level="info",
        )
        self._toasts.show(f"Batch: {total} files queued.",
                          level="info", ttl=3.0)

        want_png = output_mode in ("PNG per file", "Both")
        want_h5 = output_mode in ("HDF5 bundle", "Both")
        if want_h5 and h5_path is None:
            self._set_status("Batch: HDF5 path missing — skipping bundle.",
                             level="warn")
            want_h5 = False

        snapshot_params = self._params
        snapshot_sample = self._sample_id

        def _worker():
            from core.reconstruction import (
                ReconstructionMethod, ReconstructionParams, propagate,
            )
            from core.offaxis import (
                OffAxisParams, extract_complex_field_offaxis,
            )
            from core.ingestion import load_any
            from core.batch_bundle import BatchEntry, write_batch_hdf5
            try:
                import imageio.v2 as imageio
            except Exception:
                imageio = None
            entries: list = []
            ok = 0
            fail = 0
            import time
            for idx, path in enumerate(files, start=1):
                if self._driver._cancel_event.is_set():
                    self._post_mailbox(
                        ("batch_row", (idx, "cancelled", 0.0, "warn")),
                    )
                    break
                t0 = time.monotonic()
                self._post_mailbox(
                    ("batch_row", (idx, "running…", None, "info")),
                )
                try:
                    loaded = load_any(path)
                    raw = np.asarray(loaded.array, dtype=np.float32)
                    if raw.ndim == 3:
                        raw = raw[..., 0]
                    if snapshot_params.subtract_mean:
                        raw = raw - float(np.mean(raw))
                    if snapshot_params.hann_window:
                        wy = np.hanning(raw.shape[0]).astype(np.float32)
                        wx = np.hanning(raw.shape[1]).astype(np.float32)
                        raw = raw * (wy[:, None] * wx[None, :])
                    peak = float(np.max(np.abs(raw)))
                    if peak > 0:
                        raw = raw / peak
                    field, _ = extract_complex_field_offaxis(
                        raw,
                        OffAxisParams(
                            radius=int(snapshot_params.mask_radius),
                        ),
                    )
                    method = (ReconstructionMethod.ASM
                              if snapshot_params.method.upper() == "ASM"
                              else ReconstructionMethod.FRESNEL)
                    p = ReconstructionParams(
                        wavelength_m=float(
                            snapshot_params.wavelength_nm) * 1e-9,
                        pixel_size_m=float(
                            snapshot_params.effective_pixel_um()) * 1e-6,
                        z_m=float(snapshot_params.z_mm) * 1e-3,
                        n=1.0,
                    )
                    recon = propagate(field, p, method)
                    phase = np.angle(recon).astype(np.float32)
                    amp = np.abs(recon).astype(np.float32)
                    rt = (time.monotonic() - t0) * 1000.0

                    if want_png and imageio is not None:
                        imageio.imwrite(
                            path.with_name(f"{path.stem}_phase.png"),
                            self._to_uint8(phase),
                        )
                        imageio.imwrite(
                            path.with_name(f"{path.stem}_amp.png"),
                            self._to_uint8(amp),
                        )
                    if want_h5:
                        entries.append(BatchEntry(
                            source_path=path,
                            phase=phase,
                            amplitude=amp,
                            metadata={
                                "z_mm": float(snapshot_params.z_mm),
                                "wavelength_nm": float(
                                    snapshot_params.wavelength_nm),
                                "magnification": float(
                                    snapshot_params.magnification),
                                "sample_id": snapshot_sample,
                            },
                            runtime_ms=rt,
                        ))
                    ok += 1
                    self._post_mailbox(
                        ("batch_row", (idx, "ok", rt, "ok")),
                    )
                except Exception as exc:
                    _LOG.warning("batch: %s failed: %s", path.name, exc)
                    fail += 1
                    self._post_mailbox(
                        ("batch_row",
                         (idx, f"failed: {exc}", None, "danger")),
                    )

            h5_saved = None
            if want_h5 and entries:
                try:
                    h5_saved = write_batch_hdf5(
                        h5_path, entries,
                        sample_id=snapshot_sample,
                        recon_params={
                            "wavelength_nm": float(
                                snapshot_params.wavelength_nm),
                            "pixel_um": float(snapshot_params.pixel_um),
                            "z_mm": float(snapshot_params.z_mm),
                            "method": str(snapshot_params.method),
                            "magnification": float(
                                snapshot_params.magnification),
                            "pixel_is_effective": bool(
                                snapshot_params.pixel_is_effective),
                        },
                    )
                except Exception as exc:
                    _LOG.warning("batch HDF5 write failed: %s", exc)
                    self._post_mailbox(
                        ("status", f"HDF5 save failed: {exc}"),
                    )

            final = f"Batch done: {ok} OK, {fail} failed (of {total})"
            if h5_saved is not None:
                final += f" · bundle: {h5_saved.name}"
            self._post_mailbox(("status", final))
            self._post_mailbox(("batch_done", None))

        threading.Thread(target=_worker, daemon=True,
                         name="batch-reconstruct").start()

    def _handle_batch_row(self, payload) -> None:
        """Mailbox dispatcher for batch progress rows posted from the
        worker thread. Payload shape: ``(idx, status, runtime_ms, level)``
        where ``runtime_ms`` is ``None`` when still running."""
        try:
            idx, status, rt, level = payload
        except Exception:
            return
        color = {
            "ok": ScientificTheme.SUCCESS,
            "warn": ScientificTheme.WARN,
            "danger": ScientificTheme.DANGER,
            "info": ScientificTheme.TEXT_MUTED,
        }.get(level, ScientificTheme.TEXT_MUTED)
        try:
            dpg.set_value(self._batch_status_tag(idx), status)
            dpg.configure_item(self._batch_status_tag(idx), color=color)
            dpg.set_value(
                self._batch_runtime_tag(idx),
                f"{rt:.0f}" if rt is not None else "—",
            )
        except Exception:
            _LOG.debug("batch row update failed", exc_info=True)

    def _handle_batch_done(self, _payload) -> None:
        try:
            dpg.configure_item("batch_start_btn", enabled=True)
        except Exception:
            pass

    # ---- Camera live feed ---------------------------------------------

    def _build_camera_source(self) -> CameraSource:
        """Pick a backend. macOS dev laptops get the synthetic source;
        future real-hardware support lands by returning a different
        :class:`CameraSource` implementation based on an env var, a
        plugin import probe, or a user-selected radio button."""
        size = min(self._preview_size, 512)
        return SyntheticCamera(size_px=size,
                               target_fps=30.0)

    def _on_camera_start(self) -> None:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._set_status("Camera already running.", level="info")
            return
        try:
            source = self._build_camera_source()
        except Exception as exc:
            self._set_status(f"Camera init failed: {exc}", level="danger")
            return
        self._camera_source = source
        self._camera_thread = AcquisitionThread(
            source=source,
            on_frame=self._post_camera_frame,
            on_fps=self._post_camera_fps,
            recorder=self._camera_recorder,
        )
        self._camera_thread.start()
        # v2.1.x: explicit mode flip so the status prefix +
        # any panel header label switch to "● LIVE".
        self._set_input_mode("live")
        self._set_status(
            f"Camera running ({type(source).__name__}).", level="ok",
        )
        self._toasts.show("Camera live feed started.", level="ok", ttl=2.5)

    def _on_camera_stop(self) -> None:
        thread = self._camera_thread
        if thread is None:
            self._set_status("No camera is running.", level="info")
            return
        thread.stop()
        # Don't join on the UI thread — daemon thread exits on its
        # own, blocking here would freeze the render loop.
        self._camera_thread = None
        self._camera_source = None
        if self._camera_recorder is not None:
            try:
                self._camera_recorder.stop()
            except Exception:
                pass
        self._camera_recorder = None
        # v2.1.x: revert mode. We don't auto-flip back to "file"
        # unless a hologram is actually loaded — a stopped camera
        # with no file means there's nothing reliable on the
        # input panel, and the user should know that. Keep
        # latest_live_frame so a snapshot is still possible right
        # after stop.
        if self._current_hologram is not None:
            self._set_input_mode("file")
        else:
            # No file ever loaded — surface "file" mode anyway so
            # the prefix doesn't pretend the camera is still going.
            self._set_input_mode("file")
        self._set_status("Camera stopped.", level="info")

    def _on_camera_record(self) -> None:
        """Pick a destination TIFF and attach a :class:`TiffStackRecorder`
        to the next acquisition. If the feed is already running we stop
        + restart so the recorder gets every frame from t=0."""
        tag = "camera_record_dialog"
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=True)
            return
        from datetime import datetime
        default = (Path(self._last_dir) if self._last_dir
                   else Path.home()) / (
            f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
        )
        with dpg.file_dialog(
            directory_selector=False, show=True, modal=True,
            callback=self._on_camera_record_path_picked,
            tag=tag, width=720, height=480,
            default_path=str(default.parent),
            default_filename=default.name,
        ):
            dpg.add_file_extension(".tif")
            dpg.add_file_extension(".*")

    def _on_camera_record_path_picked(self, sender, app_data) -> None:
        sel = app_data.get("file_path_name") or app_data.get("current_path")
        if not sel:
            return
        path = Path(sel)
        if path.suffix.lower() not in (".tif", ".tiff"):
            path = path.with_suffix(".tif")
        was_running = (self._camera_thread is not None
                       and self._camera_thread.is_alive())
        if was_running:
            self._on_camera_stop()
        self._camera_recorder = TiffStackRecorder(path)
        try:
            self._camera_recorder.start()
        except Exception as exc:
            self._set_status(f"Recorder init failed: {exc}", level="danger")
            self._camera_recorder = None
            return
        self._set_status(f"Recording to {path.name}…", level="info")
        self._on_camera_start()

    def _post_camera_frame(self, frame: np.ndarray) -> None:
        """Called from the acquisition thread — push frame onto the
        mailbox so the UI repaints on the next render tick."""
        self._post_mailbox(("camera_frame", frame))

    def _post_camera_fps(self, fps: float) -> None:
        self._post_mailbox(("camera_fps", float(fps)))

    def _handle_camera_frame(self, frame: np.ndarray) -> None:
        # v2.1.x: cache the latest live frame so the operator can
        # snapshot it (Tools → Snapshot live frame) or reconstruct
        # against it without waiting for record/stop/load.
        self._latest_live_frame = np.asarray(frame, dtype=np.float32,
                                              copy=True)
        try:
            self.panel_input.set_image(frame)
        except Exception:
            _LOG.debug("camera frame paint failed", exc_info=True)

    def _handle_camera_fps(self, fps: float) -> None:
        self._camera_fps = float(fps)
        rec = ""
        if self._camera_recorder is not None:
            rec = (f" · recording "
                   f"{self._camera_recorder.frames_written} frames")
        try:
            dpg.set_value(
                "status_text",
                f"Camera: {fps:.1f} fps{rec}",
            )
        except Exception:
            pass

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        """Normalise a 2D float array into uint8 for PNG export."""
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-12:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)

    # ---- v2.1.x: explicit input mode tracking --------------------------

    _MODE_PREFIX = {
        "file":      "[FILE]",
        "live":      "[● LIVE]",
        "timelapse": "[● TIMELAPSE]",
    }

    def _set_input_mode(self, mode: str) -> None:
        """Set the operator-visible input mode. Recognised values:
        ``"file"`` (loaded TIFF), ``"live"`` (camera streaming),
        ``"timelapse"`` (interval acquisition).

        The status bar prefix and (where wired) the input panel
        header reflect the active mode so the operator can't
        mistake a stale frame from a stopped camera for a freshly
        loaded file. v2.1.x H6 — surfaced after the lab review on
        2026-04-28.
        """
        if mode not in self._MODE_PREFIX:
            mode = "file"
        # Defensive read — DhmApp instances built via __new__ in
        # tests don't run __init__, so the attribute may not exist
        # when this method is reached for the first time. Default
        # to "file" so the prev/new comparison still works.
        prev = getattr(self, "_input_mode", "file")
        self._input_mode = mode
        # Update the panel header label if it exists. (DPG render
        # may not have built the tag yet during construction.)
        try:
            dpg.set_value("input_mode_label", self._MODE_PREFIX[mode])
        except Exception:
            pass
        if prev != mode:
            self._set_status(
                f"Input mode: {mode}", level="info",
            )

    def _set_status(self, message: str, *, level: str = "info") -> None:
        color = {
            "info":   ScientificTheme.TEXT_MUTED,
            "ok":     ScientificTheme.SUCCESS,
            "warn":   ScientificTheme.WARN,
            "danger": ScientificTheme.DANGER,
        }.get(level, ScientificTheme.TEXT_MUTED)
        # v2.1.x: prefix the status line with the active input mode
        # so a glance at the bar tells the operator whether they
        # are looking at a loaded file, a live stream, or a
        # scheduled time-lapse run.
        prefix = self._MODE_PREFIX.get(
            getattr(self, "_input_mode", "file"), "[FILE]",
        )
        prefixed = f"{prefix}  {message}"
        try:
            dpg.configure_item("status_text", color=color)
            dpg.set_value("status_text", prefixed)
        except Exception:
            pass
        # Feed the error drawer so the warning/danger history survives
        # the status line being overwritten a second later.
        if level in ("warn", "danger"):
            self._append_error(level, message)

    # ---- v2.1.x: snapshot-from-live ------------------------------------

    def _snapshot_live_frame_to_tempfile(self) -> Optional[Path]:
        """Persist the most recent live frame to a temp TIFF and
        return the path. Returns ``None`` when no live frame has
        been received yet (camera just started, no grab completed)."""
        frame = self._latest_live_frame
        if frame is None:
            return None
        from datetime import datetime
        import tempfile
        try:
            import tifffile
        except Exception:
            self._set_status(
                "tifffile missing — can't snapshot live frame.",
                level="danger",
            )
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = Path(tempfile.gettempdir()) / "dhm-snapshots"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"live_snapshot_{ts}.tif"
        u16 = (np.clip(frame, 0.0, 1.0) * 65535.0).astype(np.uint16)
        try:
            tifffile.imwrite(str(path), u16)
        except Exception as exc:
            self._set_status(
                f"snapshot failed: {exc}", level="danger",
            )
            return None
        return path


def _resize_nearest(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Box-filtered nearest-neighbour resample. Keeps the preview cheap —
    accurate inspection goes through the full-resolution pipeline."""
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
    h, w = arr.shape
    if (h, w) == (target_h, target_w):
        return arr.astype(np.float32, copy=False)
    y_idx = (np.linspace(0, h - 1, target_h)).astype(np.int32)
    x_idx = (np.linspace(0, w - 1, target_w)).astype(np.int32)
    return arr[y_idx][:, x_idx].astype(np.float32, copy=False)


def main() -> int:
    # Ensure ``core`` package is importable even if run as a plain script.
    here = Path(__file__).resolve()
    src = here.parent.parent
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    # v2.0.9 P3: install crash handler before any DPG window opens.
    # The UI-side wrapper inside DhmApp chains onto whatever is in
    # sys.excepthook *at construction time*, so installing core's
    # base handler first means crashes flush a dump + the UI shows
    # a toast — both surfaces fire, neither swallows the other.
    try:
        from core.crash_handler import (
            install_crash_handler, install_threading_excepthook,
        )
        install_crash_handler()
        install_threading_excepthook()
    except Exception:
        pass
    return DhmApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
