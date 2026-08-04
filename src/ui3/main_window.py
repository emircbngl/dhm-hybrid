"""MainWindow — the ui3 shell.

A dock-based QMainWindow: a 2×2 imaging grid (input / amplitude / phase /
spectrum) in the centre, a Reconstruct control dock on the left, a menu bar
covering every action from the coverage matrix, a status bar with a busy
indicator, a ⌘K command palette, toasts, theming, workflow modes, and panel
maximize (⌘1–4/0).

Rich feature panels (focus, QPI, depth, reference-free, camera, device, AI,
dialogs) mount into this shell via :meth:`mount_panel`; the spine ships a
working load→reconstruct→autofocus→QPI loop with an inline control dock so the
app is genuinely usable on its own.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget, QFileDialog, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from core.ingestion import load_any
from core.drivers.reconstruction import ReconParams
from ui3.bridge import WorkerBridge
from ui3.design import Space, get_palette
from ui3.state import Ui3State, push_recent
from ui3.viewport import ImagePanel
from ui3.widgets.command_palette import Command, CommandRegistry, CommandPalette
from ui3.widgets.toast import ToastHost

_LOG = logging.getLogger("ui3.main")

WORKFLOW_MODES = ["Acquire", "Reconstruct", "Analyse", "Report"]

# Bump whenever the dock set or a dock's structure changes (panels added/
# removed, scroll-wrapping, objectName changes). A saved window_state from a
# different version is ignored on restore so mismatched docks can't come back
# floating over the central grid (2026-07-10). Current: 3 (per-mode re-tabify
# + top tab position).
_LAYOUT_VERSION = 3


class MainWindow(QMainWindow):
    def __init__(self, state: Optional[Ui3State] = None) -> None:
        super().__init__()
        self._state = state or Ui3State()
        self._palette = get_palette(self._state.theme)
        self._params = self._state.recon_params()
        self._hologram_path: Optional[Path] = None
        self._raw: Optional[np.ndarray] = None
        self._last_recon = None
        self._last_depth = None
        self._docks: Dict[str, QDockWidget] = {}
        self._panels: Dict[str, QWidget] = {}
        self._commands = CommandRegistry()

        self.setWindowTitle("DHM Reconstruction — ui3")
        self.resize(1360, 860)

        self._bridge = WorkerBridge(self)
        self._build_central()
        self._build_statusbar()
        self._toasts = ToastHost(self, self._palette)
        self._build_control_dock()          # rich ReconPanel (source of truth)
        self._mount_feature_panels()        # focus/qpi/depth/camera/device/report/timelapse/ai
        self._build_dialogs()               # surface/qpi_batch/focus_candidates/audit/onboarding
        self._build_menus()
        self._palette_dialog = CommandPalette(self._commands, self)
        self._connect_bridge()
        self._install_shortcuts()
        self._register_commands()

        # Window SIZE autofits the screen every launch (never a stale, too-tall
        # saved size); only the dock LAYOUT is restored.
        self._restore_geometry()
        self._apply_workflow_mode(self._state.workflow_mode)
        self._maybe_onboard()

    # ================================================================
    # Central 2×2 imaging grid
    # ================================================================
    def _build_central(self) -> None:
        self.panel_input = ImagePanel("Input hologram", self._palette,
                                      default_cmap="gray")
        self.panel_amp = ImagePanel("Amplitude", self._palette,
                                    default_cmap="gray")
        self.panel_phase = ImagePanel("Phase", self._palette,
                                      default_cmap="twilight")
        self.panel_spectrum = ImagePanel("Spectrum (log|F|)", self._palette,
                                         default_cmap="viridis")
        self._image_panels = {
            "input": self.panel_input, "amplitude": self.panel_amp,
            "phase": self.panel_phase, "spectrum": self.panel_spectrum,
        }
        for p in self._image_panels.values():
            p.file_dropped.connect(self._load_path)
        # Click the spectrum to override the auto +1-order center.
        self.panel_spectrum.pixel_clicked.connect(self._on_spectrum_center_picked)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.panel_input)
        top.addWidget(self.panel_amp)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self.panel_phase)
        bottom.addWidget(self.panel_spectrum)
        grid = QSplitter(Qt.Vertical)
        grid.addWidget(top)
        grid.addWidget(bottom)
        grid.setSizes([500, 500])
        top.setSizes([500, 500])
        bottom.setSizes([500, 500])
        self._grid = grid
        self._grid_splitters = (top, bottom)
        self.setCentralWidget(grid)

    # ================================================================
    # Left control dock — the rich ReconPanel (single source of truth for
    # reconstruction params; it mutates ``self._params`` via the context).
    # ================================================================
    def _build_control_dock(self) -> None:
        from ui3.panels.recon_panel import ReconPanel
        ctx = self.panel_context()
        self._recon_panel = ReconPanel(ctx)
        # A small header with just the Load button above the panel, so the
        # dock is self-sufficient. The loaded filename is shown once, inside
        # the panel's own "Hologram" group (kept in sync via B-114) — a second
        # header caption here duplicated it within ~110px.
        wrap = QWidget()
        root = QVBoxLayout(wrap)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.sm)
        root.setSpacing(Space.sm)
        load_btn = QPushButton("Load hologram…")
        load_btn.setProperty("role", "primary")
        load_btn.clicked.connect(self._on_load)
        root.addWidget(load_btn)
        root.addWidget(self._recon_panel, 1)

        dock = QDockWidget("Reconstruct", self)
        dock.setObjectName("dock_reconstruct")
        # Scroll-wrap like every feature dock (B-113) — the control panel is
        # tall (params + reference + advanced) and clipped its lower buttons on
        # a short window when set directly (2026-07-10 audit #7).
        scroll = QScrollArea(dock)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # Vertical scroll only — the panel wraps to the dock width, so never a
        # horizontal bar (which the vertical bar's width would otherwise trigger).
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(wrap)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._docks["reconstruct"] = dock
        # Dock title bar already reads "Reconstruct" — drop the panel's own
        # duplicate heading (2026-07-10 UI audit).
        self._hide_panel_heading(self._recon_panel)

    @staticmethod
    def _ref_mode_label(mode: str) -> str:
        return {"off": "Off", "reference": "Reference",
                "reference_free": "Reference-free"}.get(str(mode), "Off")

    @staticmethod
    def _ref_mode_value(label: str) -> str:
        return {"Off": "off", "Reference": "reference",
                "Reference-free": "reference_free"}.get(str(label), "off")

    # ================================================================
    # Menus
    # ================================================================
    def _build_menus(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("&File")
        self._add(m_file, "Load hologram…", self._on_load, "Ctrl+O")
        self._add(m_file, "Load reference hologram…", self._on_load_reference)
        self._recent_menu = m_file.addMenu("Open recent")
        self._rebuild_recent_menu()
        m_file.addSeparator()
        self._add(m_file, "Quit", self.close, "Ctrl+Q")

        m_view = mb.addMenu("&View")
        theme_menu = m_view.addMenu("Theme")
        for name in ("dark", "midnight", "light", "high_contrast"):
            self._add(theme_menu, name.replace("_", " ").title(),
                      lambda n=name: self._apply_theme(n))
        mode_menu = m_view.addMenu("Workflow mode")
        for mode in WORKFLOW_MODES:
            self._add(mode_menu, mode, lambda m=mode: self._apply_workflow_mode(m))
        m_view.addSeparator()
        for i, key in enumerate(("input", "amplitude", "phase", "spectrum"), 1):
            self._add(m_view, f"Maximize panel {i}",
                      lambda k=key: self._maximize_panel(k), f"Ctrl+{i}")
        self._add(m_view, "Restore grid", self._restore_grid, "Ctrl+0")
        self._add(m_view, "Reset panel layout", self._reset_panel_layout,
                  "Ctrl+Shift+0")
        self._add(m_view, "Command palette…", self._open_palette, "Ctrl+K")

        m_process = mb.addMenu("&Process")
        self._add(m_process, "Reconstruct", self._on_reconstruct, "Ctrl+R")
        self._add(m_process, "Autofocus", self._on_autofocus, "Ctrl+F")
        self._add(m_process, "Find focus candidates", self._on_find_candidates)
        m_process.addSeparator()
        # Manual +1-order center: a checkable "arm" toggle, then click the
        # spectrum. Kept together with a reset-to-auto escape hatch.
        self._act_pick_order = QAction("Pick +1 order center (click spectrum)",
                                       self)
        self._act_pick_order.setCheckable(True)
        self._act_pick_order.toggled.connect(self._toggle_pick_order_center)
        m_process.addAction(self._act_pick_order)
        self._add(m_process, "Reset +1 order to auto-detect",
                  self._reset_order_center)

        m_analyse = mb.addMenu("&Analyse")
        self._add(m_analyse, "Compute QPI", self._on_qpi)
        self._add(m_analyse, "QPI batch (per candidate)", self._on_qpi_batch)
        self._add(m_analyse, "Depth map", self._on_depth_map)
        self._add(m_analyse, "3D surface — depth", lambda: self._open_surface("depth"))
        self._add(m_analyse, "3D surface — phase", lambda: self._open_surface("phase"))
        self._add(m_analyse, "Line profile (phase)…", self._open_line_profile)

        m_tools = mb.addMenu("&Tools")
        self._add(m_tools, "AI copilot", lambda: self._panel_action("ai"))
        self._add(m_tools, "Devices", lambda: self._panel_action("device"))
        self._add(m_tools, "Camera", lambda: self._panel_action("camera"))
        self._add(m_tools, "Time-lapse…", lambda: self._panel_action("timelapse"))
        self._add(m_tools, "Export report / CSV / bundle…",
                  lambda: self._panel_action("report"))
        self._add(m_tools, "Audit log…", lambda: self._panel_action("audit"))

        m_help = mb.addMenu("&Help")
        self._add(m_help, "Onboarding", self._show_onboarding)
        self._add(m_help, "About", self._about)

    def _add(self, menu, text, slot, shortcut: str = "") -> QAction:
        act = QAction(text, self)
        act.triggered.connect(lambda _checked=False: slot())
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        menu.addAction(act)
        return act

    # ================================================================
    # Status bar
    # ================================================================
    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self._busy_label = QLabel("")
        self._busy_label.setProperty("role", "muted")
        self._status_label = QLabel("Ready")
        self._status_label.setProperty("role", "muted")
        sb.addWidget(self._status_label, 1)
        sb.addPermanentWidget(self._busy_label)

    def set_status(self, text: str, level: str = "info") -> None:
        # Status ONLY — no auto-toast. Panels/dialogs are self-sufficient
        # units that call ctx.toast explicitly when an event deserves one;
        # auto-toasting ok/warn/error here stacked a second identical toast
        # on top of every panel-owned event (and made 'danger'-level errors
        # inconsistently quieter than 'error'-level ones) (2026-07-06
        # review #2). Callers that want a toast call self.toast/ctx.toast.
        self._status_label.setText(text)

    # ================================================================
    # Bridge wiring
    # ================================================================
    def _connect_bridge(self) -> None:
        """Wire ONLY the shell-owned concerns.

        Ownership contract (2026-07-06 review, B-082/B-083): the rich panels
        and dialogs are self-sufficient units that already connect to the
        bridge for their own domain — status/toast/readout/table. The shell
        must not add a second handler for anything a panel/dialog owns
        (double status, double populate, double compute). The shell owns:
        viewport paint + result caches for get_field(), the z-param write
        after autofocus (single source of truth for self._params), and
        *presenting* result dialogs (panels/dialogs populate, shell shows).
        Error statuses are owned by the panels/dialogs too.
        """
        b = self._bridge
        b.busy_changed.connect(self._on_busy)
        b.recon_done.connect(self._on_recon_done)          # paint + cache + status
        b.autofocus_done.connect(self._on_autofocus_done)  # z write + control sync
        b.focus_candidates_done.connect(self._on_candidates_done)  # present dialog
        b.qpi_batch_done.connect(self._on_qpi_batch_done)          # present dialog
        b.depth_done.connect(self._on_depth_done)                  # cache only

    def _on_busy(self, busy: bool, label: str) -> None:
        self._busy_label.setText(label if busy else "")
        if busy:
            self._status_label.setText(label)

    # ================================================================
    # Actions — load
    # ================================================================
    def _on_load(self) -> None:
        start = self._state.last_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load hologram", start,
            "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg *.npy)")
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            loaded = load_any(Path(path))
            arr = np.asarray(loaded.array, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
        except Exception as exc:
            self.set_status(f"Load failed: {exc}", "error")
            return
        self._hologram_path = Path(path)
        self._raw = arr
        self._state.last_dir = str(Path(path).parent)
        self._state.last_hologram = str(path)
        push_recent(self._state, str(path))
        self._rebuild_recent_menu()
        # Update the ReconPanel's own "Hologram" label — it reads
        # ctx.hologram_path() but only at construction, so after a load it
        # still read "(no hologram loaded)" next to a clearly-loaded file
        # (2026-07-10 UI audit). This is now the single filename display.
        rp = getattr(self, "_recon_panel", None)
        if rp is not None and hasattr(rp, "_refresh_file_label"):
            rp._refresh_file_label()
        self.panel_input.set_image(arr, contrast="percentile")
        # spectrum preview from raw immediately
        self._show_spectrum(arr)
        self.set_status(f"Loaded {Path(path).name}")

    def _on_load_reference(self) -> None:
        start = self._state.last_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference hologram", start,
            "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg *.npy)")
        if not path:
            return
        # Mutate the shared live params and re-hydrate the ReconPanel (the
        # control surface) — NOT a nonexistent inline combo. Keep the legacy
        # subtract_reference flag in lockstep so effective_reference_mode()
        # resolves to "reference" (2026-07-06 review: this used to crash with
        # AttributeError on self._cb_ref_mode, removed when the inline dock
        # was replaced by ReconPanel).
        self._params.reference_path = Path(path)
        self._params.reference_mode = "reference"
        self._params.subtract_reference = True
        self._sync_controls_from_params()
        self._state.last_reference = path
        self.set_status(f"Reference set: {Path(path).name} — Reconstruct will divide it out.")

    # ================================================================
    # Actions — compute
    # ================================================================
    def _params_from_controls(self) -> ReconParams:
        # ReconPanel is the single source of truth: it writes edits straight
        # into ``self._params`` via the context, so this just returns it.
        return self._params

    def _require_hologram(self) -> bool:
        if self._hologram_path is None:
            self.set_status("Load a hologram first.", "warn")
            return False
        return True

    def _on_reconstruct(self) -> None:
        if not self._require_hologram():
            return
        self._bridge.reconstruct(self._hologram_path, self._params_from_controls(),
                                 sample_id=self._state.sample_id)

    def _on_recon_done(self, result) -> None:
        self._last_recon = result
        self.panel_amp.set_image(result.amplitude, contrast="percentile",
                                 pixel_size_um=self._params.effective_pixel_um())
        phase = result.unwrapped_phase if result.unwrapped_phase is not None \
            else result.phase
        self.panel_phase.set_image(phase, contrast="percentile",
                                   pixel_size_um=self._params.effective_pixel_um())
        # Surface any non-fatal note the driver attached — reference_note
        # (reference division FAILED → result is quantitatively unreferenced,
        # B-110) or reffree_note (CNN unavailable etc.). These are mutually
        # exclusive modes but we surface whichever is present; set_status no
        # longer auto-toasts, so a warn note must also toast explicitly (it
        # warns the result is NOT what the user asked for).
        note = (getattr(result, "reference_note", None)
                or getattr(result, "reffree_note", None))
        if note:
            self.set_status(note, "warn")
            self.toast(note, "warn")
        else:
            self.set_status(f"Reconstructed in {result.runtime_ms:.0f} ms")

    def _on_autofocus(self) -> None:
        if not self._require_hologram():
            return
        p = self._params_from_controls()
        self._bridge.autofocus(self._hologram_path, p,
                               z_min_mm=p.af_z_min_mm, z_max_mm=p.af_z_max_mm,
                               n_steps=p.af_n_steps, sample_id=self._state.sample_id)

    def _on_autofocus_done(self, result) -> None:
        # AutofocusResult carries best_z_m (metres) — convert to the mm the
        # UI/params use. (best_z_mm never existed; the old getattr silently
        # no-op'd and reconstructed at the stale z — 2026-07-05 review.)
        best_m = getattr(result, "best_z_m", None)
        if best_m is not None:
            best_mm = float(best_m) * 1e3
            self._params.z_mm = best_mm
            self._sync_controls_from_params()
        # NOTE: no auto-reconstruct and no status here. FocusPanel owns the
        # autofocus status/toast/readout; this shared handler used to run the
        # pipeline a second time and add a third status write (2026-07-06
        # review, B-082/B-083). The shell only writes z + re-hydrates controls.

    def _on_find_candidates(self) -> None:
        if not self._require_hologram():
            return
        p = self._params_from_controls()
        self._bridge.find_focus_candidates(
            self._hologram_path, p, z_min_mm=p.af_z_min_mm,
            z_max_mm=p.af_z_max_mm, sample_id=self._state.sample_id)

    def _on_candidates_done(self, result) -> None:
        # Present-only: the dialog populates its own table + writes the
        # status via its own focus_candidates_done connection; calling
        # open_with(result) here populated the table a second time and
        # duplicated the status (2026-07-06 review, B-083).
        dlg = self._dialogs.get("focus_candidates")
        if dlg is not None and hasattr(dlg, "present"):
            dlg.present()

    def _on_qpi(self) -> None:
        if not self._require_hologram():
            return
        p = self._params_from_controls()
        self._bridge.qpi(self._hologram_path, p, z_mm=p.z_mm,
                        n_sample=p.n_sample, n_medium=p.n_medium,
                        sample_id=self._state.sample_id)

    def _on_qpi_batch(self) -> None:
        if not self._require_hologram():
            return
        p = self._params_from_controls()
        self._bridge.qpi_batch(self._hologram_path, p, z_min_mm=p.af_z_min_mm,
                              z_max_mm=p.af_z_max_mm, n_sample=p.n_sample,
                              n_medium=p.n_medium, sample_id=self._state.sample_id)

    def _on_qpi_batch_done(self, result) -> None:
        # Present-only — see _on_candidates_done (B-083).
        dlg = self._dialogs.get("qpi_batch")
        if dlg is not None and hasattr(dlg, "present"):
            dlg.present()

    def _on_depth_map(self) -> None:
        if not self._require_hologram():
            return
        p = self._params_from_controls()
        self._bridge.depth_map(self._hologram_path, p, z_min_mm=p.af_z_min_mm,
                              z_max_mm=p.af_z_max_mm, sample_id=self._state.sample_id)

    def _on_depth_done(self, result) -> None:
        # Cache only — DepthPanel owns the viewport paint, readout and status
        # (it also connects depth_done). This handler used to repaint the
        # shared 'spectrum' viewport a second time with the same z-map
        # (2026-07-06 review). We still need the cache for get_field('depth')
        # (AI vision + 3D surface viewer read it from the shell).
        self._last_depth = result

    def _show_spectrum(self, arr: np.ndarray) -> None:
        try:
            mag = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
            self.panel_spectrum.set_title("Spectrum (log|F|)")
            self.panel_spectrum.set_colormap("viridis")
            self.panel_spectrum.set_image(np.log1p(mag), contrast="percentile")
        except Exception:
            _LOG.debug("spectrum preview failed", exc_info=True)

    # ----------------------------------------------------------------
    # Manual off-axis +1-order center (click the spectrum)
    # ----------------------------------------------------------------
    def _toggle_pick_order_center(self, on: bool) -> None:
        """Arm/disarm click-to-set the +1-order center on the spectrum.

        The 'spectrum' panel is shared with the depth overlay, so arming
        repaints THIS hologram's log|fftshift(F)| first — that is the exact
        coordinate frame ``extract_complex_field_offaxis`` uses, so a click
        maps 1:1 to ``offaxis_center`` (2026-07-08)."""
        if on and self._raw is None:
            self.set_status("Load a hologram first to pick the +1-order "
                            "center.", "warn")
            self._act_pick_order.setChecked(False)
            return
        if on:
            self._show_spectrum(self._raw)
            if self._params.offaxis_center is not None:
                r, c = self._params.offaxis_center
                self.panel_spectrum.set_marker(int(r), int(c))
            self.panel_spectrum.set_pick_mode(True)
            self.set_status("Click the Spectrum panel to set the +1-order "
                            "center (Process ▸ Reset to return to auto).",
                            "info")
        else:
            self.panel_spectrum.set_pick_mode(False)

    def _on_spectrum_center_picked(self, row: int, col: int) -> None:
        self._params.offaxis_center = (int(row), int(col))
        self.panel_spectrum.set_marker(int(row), int(col))
        self.panel_spectrum.set_pick_mode(False)
        self._act_pick_order.setChecked(False)   # one-shot per arm
        self.set_status(
            f"+1-order center set to (y={row}, x={col}) — reconstructing…",
            "ok")
        self._on_reconstruct()

    def _reset_order_center(self) -> None:
        if self._params.offaxis_center is None:
            self.set_status("+1-order center is already auto-detected.", "info")
            return
        self._params.offaxis_center = None
        self.panel_spectrum.clear_marker()
        self.set_status("+1-order center reset to auto-detect — "
                        "reconstructing…", "ok")
        self._on_reconstruct()

    # ================================================================
    # 3D surface / not-yet-mounted panels
    # ================================================================
    def _open_surface(self, kind: str) -> None:
        viewer = self._dialogs.get("surface")
        if viewer is not None:
            viewer.open_for(self, kind)
        else:
            self.set_status("3D surface viewer unavailable.", "warn")

    def _panel_action(self, key: str, extra=None) -> None:
        """Open a feature dock (raise + focus it) or a dialog."""
        # Docked feature panels: ensure visible + raise.
        dock = self._docks.get(key)
        if dock is not None:
            dock.setVisible(True)
            dock.raise_()
            return
        # Dialogs.
        dlg = self._dialogs.get(key)
        if dlg is not None:
            opener = getattr(dlg, "refresh", None)
            if opener:
                try:
                    opener()
                except Exception:
                    _LOG.debug("%s.refresh failed", key, exc_info=True)
            from ui3.dialogs._present import present_centered
            present_centered(dlg)
            return
        self.set_status(f"'{key}' is not available.", "warn")

    # ================================================================
    # Feature panels + dialogs (built at construction)
    # ================================================================
    def _mount_feature_panels(self) -> None:
        """Instantiate every dockable feature panel against the context."""
        ctx = self.panel_context()
        specs = []
        try:
            from ui3.panels.focus_panel import FocusPanel
            specs.append(("focus", FocusPanel(ctx), "Autofocus"))
        except Exception:
            _LOG.warning("focus panel failed to load", exc_info=True)
        try:
            from ui3.panels.qpi_panel import QPIPanel
            specs.append(("qpi", QPIPanel(ctx), "QPI"))
        except Exception:
            _LOG.warning("qpi panel failed to load", exc_info=True)
        try:
            from ui3.panels.depth_panel import DepthPanel
            dp = DepthPanel(ctx)
            if hasattr(dp, "surface_requested"):
                dp.surface_requested.connect(self._open_surface)
            specs.append(("depth", dp, "Depth"))
        except Exception:
            _LOG.warning("depth panel failed to load", exc_info=True)
        try:
            from ui3.panels.ai_panel import AIPanel
            specs.append(("ai", AIPanel(ctx), "AI copilot"))
        except Exception:
            _LOG.warning("ai panel failed to load", exc_info=True)
        try:
            from ui3.panels.camera_panel import CameraPanel
            specs.append(("camera", CameraPanel(ctx), "Camera"))
        except Exception:
            _LOG.warning("camera panel failed to load", exc_info=True)
        try:
            from ui3.panels.device_panel import DevicePanel
            specs.append(("device", DevicePanel(ctx), "Devices"))
        except Exception:
            _LOG.warning("device panel failed to load", exc_info=True)
        try:
            from ui3.panels.timelapse_panel import TimelapsePanel
            specs.append(("timelapse", TimelapsePanel(ctx), "Time-lapse"))
        except Exception:
            _LOG.warning("timelapse panel failed to load", exc_info=True)
        try:
            from ui3.panels.report_panel import ReportPanel
            specs.append(("report", ReportPanel(ctx), "Export"))
        except Exception:
            _LOG.warning("report panel failed to load", exc_info=True)

        first = None
        for key, widget, title in specs:
            self.mount_panel(key, widget, title=title)
            if first is None:
                first = self._docks[key]
            else:
                self.tabifyDockWidget(first, self._docks[key])
        if first is not None:
            first.raise_()
        # Tabs on TOP (conventional, like a browser) rather than Qt's default
        # bottom — the bottom tab bar was easy to miss, so panels felt like
        # they were hidden behind one another (2026-07-10 real-screen fix).
        self.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.TabPosition.North)
        self.setTabPosition(Qt.LeftDockWidgetArea, QTabWidget.TabPosition.North)

    def _build_dialogs(self) -> None:
        """Instantiate the (non-dock) dialogs, opened via menu/palette."""
        self._dialogs: Dict[str, QWidget] = {}
        ctx = self.panel_context()
        loaders = {
            "surface": ("ui3.dialogs.surface_viewer", "SurfaceViewer"),
            "qpi_batch": ("ui3.dialogs.qpi_batch", "QPIBatchDialog"),
            "focus_candidates": ("ui3.dialogs.focus_candidates", "FocusCandidatesDialog"),
            "audit": ("ui3.dialogs.audit_viewer", "AuditViewerDialog"),
        }
        for key, (mod, cls) in loaders.items():
            try:
                import importlib
                klass = getattr(importlib.import_module(mod), cls)
                self._dialogs[key] = klass(ctx, self)
            except Exception:
                _LOG.warning("dialog %s failed to load", key, exc_info=True)

    def _maybe_onboard(self) -> None:
        if self._state.onboarding_seen:
            return
        try:
            from ui3.dialogs.onboarding import OnboardingDialog
            dlg = OnboardingDialog(self.panel_context(), self)
            dlg.finished_with_choice.connect(self._on_onboarding_choice)
            self._dialogs["onboarding"] = dlg
            dlg.show()
        except Exception:
            _LOG.debug("onboarding unavailable", exc_info=True)

    def _on_onboarding_choice(self, dont_show: bool) -> None:
        self._state.onboarding_seen = bool(dont_show)

    # ================================================================
    # Panel mount API
    # ================================================================
    def mount_panel(self, key: str, widget: QWidget, *,
                    dock_area=Qt.RightDockWidgetArea, title: str = "",
                    as_dock: bool = True) -> None:
        """Register a feature panel/dialog under ``key``. Docks are added to
        the window; dialogs are just registered (opened via menu/palette)."""
        self._panels[key] = widget
        if as_dock:
            dock = QDockWidget(title or key.title(), self)
            dock.setObjectName(f"dock_{key}")
            # Wrap in a scroll area so a panel taller than the dock scrolls
            # instead of clipping its lower controls — several panels need
            # 500–740px and the docks are far shorter when 8 share the tab
            # group (2026-07-08 UX fix). Resizable so the panel still fills
            # the dock width and grows with it; the panel's large
            # minimumSizeHint no longer forces the dock's minimum height.
            scroll = QScrollArea(dock)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setWidget(widget)
            dock.setWidget(scroll)
            self.addDockWidget(dock_area, dock)
            self._docks[key] = dock
            # The dock title bar (and the tab, when tabified) already names the
            # panel, so its large in-panel heading repeated the name 2–3x over.
            # Hide it in dock mode; standalone dialog classes keep their own.
            self._hide_panel_heading(widget)

    @staticmethod
    def _hide_panel_heading(widget: QWidget) -> None:
        """Hide a panel's own role='heading' title when it lives in a dock —
        the dock title bar / tab is the single source of the name (2026-07-10
        UI audit #10/#17/#21). Each feature panel builds exactly one such
        label; standalone QDialog viewers are never routed through here, so
        they keep their heading as their only title."""
        from PySide6.QtWidgets import QLabel
        for lbl in widget.findChildren(QLabel):
            if lbl.property("role") == "heading":
                lbl.setVisible(False)

    @property
    def bridge(self) -> WorkerBridge:
        return self._bridge

    @property
    def params(self) -> ReconParams:
        return self._params

    # -- PanelContext facade -------------------------------------------
    def get_field(self, target: str):
        """Latest numpy array for a target (for observation/vision tools).

        Arrays are assigned wholesale on the GUI thread, never mutated in
        place, so returning the reference is thread-safe."""
        r = self._last_recon
        if target == "raw":
            return self._raw
        if target == "recon_complex":
            amp = getattr(r, "amplitude", None)
            ph = getattr(r, "phase", None)
            if amp is not None and ph is not None:
                return amp * np.exp(1j * ph)
            return None
        if target == "phase_unwrapped":
            up = getattr(r, "unwrapped_phase", None)
            if up is not None:
                return up
            # The pipeline only populates unwrapped_phase on the reffree
            # path; in off/reference modes the old fallback silently handed
            # the WRAPPED phase to the observation tools, corrupting any
            # OPD/dry-mass computed from it (2026-07-06 review #2). Unwrap
            # on demand with the same core routine the pipeline uses and
            # cache it on the result (once per reconstruction).
            ph = getattr(r, "phase", None)
            if ph is None:
                return None
            try:
                from core.phase_unwrap import unwrap_phase_advanced
                amp = getattr(r, "amplitude", None)
                field = (amp * np.exp(1j * ph)) if amp is not None else None
                up = unwrap_phase_advanced(
                    ph, complex_field=field,
                    wavelength_m=float(self._params.wavelength_nm) * 1e-9,
                    pixel_size_m=float(self._params.effective_pixel_um()) * 1e-6,
                ).astype(np.float32)
                r.unwrapped_phase = up
                return up
            except Exception:
                _LOG.warning("on-demand unwrap failed; returning wrapped "
                             "phase", exc_info=True)
                return ph
        if target == "depth":
            d = self._last_depth
            return getattr(getattr(d, "result", d), "z_map", None)
        return None

    def toast(self, text: str, level: str = "info") -> None:
        self._toasts.show_toast(text, level)

    def show_in_panel(self, target: str, array, **kw) -> None:
        panel = self._image_panels.get(target)
        if panel is not None and array is not None:
            panel.set_image(np.asarray(array), **kw)

    def panel_context(self):
        """Build the frozen PanelContext handed to every mounted panel."""
        from ui3.context import PanelContext
        return PanelContext(
            bridge=self._bridge,
            get_params=self._params_from_controls,
            params_changed=self._sync_controls_from_params,
            hologram_path=lambda: self._hologram_path,
            last_recon=lambda: self._last_recon,
            last_depth=lambda: self._last_depth,
            get_field=self.get_field,
            set_status=self.set_status,
            toast=self.toast,
            show_in_panel=self.show_in_panel,
            palette=self._palette,
        )

    def _sync_controls_from_params(self) -> None:
        """Re-hydrate the ReconPanel controls from ``self._params`` — called
        via ``ctx.params_changed`` after any panel mutates the live params
        (e.g. autofocus writing best-z). Guarded inside ReconPanel so it
        never bounces back into a write."""
        panel = getattr(self, "_recon_panel", None)
        if panel is not None and hasattr(panel, "_load_params_into_controls"):
            panel._load_params_into_controls()

    # ================================================================
    # Theme / modes / maximize / palette
    # ================================================================
    def _apply_theme(self, name: str) -> None:
        self._state.theme = name
        self._palette = get_palette(name)
        from PySide6.QtWidgets import QApplication
        from ui3.app import apply_theme
        apply_theme(QApplication.instance(), name)
        for p in self._image_panels.values():
            p.apply_palette(self._palette)
        self._toasts.apply_palette(self._palette)
        self.set_status(f"Theme: {name}")

    def _apply_workflow_mode(self, mode: str) -> None:
        if mode not in WORKFLOW_MODES:
            mode = "Reconstruct"
        self._state.workflow_mode = mode
        # Docks a mode explicitly manages are shown only in that mode; any
        # dock no mode manages (e.g. a future free-floating tool) always
        # shows. The reconstruct dock is visible in every mode.
        # Every feature dock must be claimed by some mode — an unclaimed dock
        # falls into the "unmanaged → always visible" branch below and
        # clutters every mode (timelapse used to leak into Reconstruct/Report
        # until it was added to Acquire, 2026-07-08 UX fix).
        mode_docks = {
            "Acquire": {"camera", "device", "timelapse"},
            "Reconstruct": {"reconstruct"},
            "Analyse": {"reconstruct", "qpi", "depth", "focus", "ai"},
            "Report": {"report", "audit"},
        }
        managed = set().union(*mode_docks.values())
        visible = mode_docks.get(mode, set()) | {"reconstruct"}
        shown_features = []
        for key, dock in self._docks.items():
            vis = key in visible or key not in managed
            dock.setVisible(vis)
            if vis and key != "reconstruct":
                shown_features.append(dock)
        # Re-form ONE clean tab group from the docks this mode shows. Merely
        # toggling setVisible() on a pre-tabified 8-dock group left Qt's tab
        # bar inconsistent — only the front dock rendered and the rest were
        # unreachable ('as I close one, another appears behind, confusing').
        # Re-tabify explicitly on every switch and raise the first so there's
        # always a proper, complete tab bar (2026-07-10 real-screen fix).
        if len(shown_features) > 1:
            anchor = shown_features[0]
            for dock in shown_features[1:]:
                self.tabifyDockWidget(anchor, dock)
            anchor.raise_()
        self.set_status(f"Workflow: {mode}")

    def _maximize_panel(self, key: str) -> None:
        for k, panel in self._image_panels.items():
            panel.setVisible(k == key)

    def _restore_grid(self) -> None:
        for panel in self._image_panels.values():
            panel.setVisible(True)

    def _reset_panel_layout(self) -> None:
        """Put every dock back in its home position — un-floats and re-docks
        anything the user pulled out (floated / dragged elsewhere) and
        couldn't put back, plus un-maximizes the image grid. Recovers the
        window to the clean default layout in one click (2026-07-10)."""
        first = None
        for key, dock in self._docks.items():
            dock.setFloating(False)
            if key == "reconstruct":
                self.addDockWidget(Qt.LeftDockWidgetArea, dock)
            else:
                self.addDockWidget(Qt.RightDockWidgetArea, dock)
                if first is None:
                    first = dock
                else:
                    self.tabifyDockWidget(first, dock)
        self._restore_grid()
        # Re-apply the current workflow mode so per-mode dock visibility is
        # correct after everything was just shown by the re-docking above.
        self._apply_workflow_mode(self._state.workflow_mode)
        self.set_status("Panel layout reset — docks are back home.", "ok")

    def _open_palette(self) -> None:
        self._palette_dialog.open_palette()

    def _register_commands(self) -> None:
        c = self._commands
        c.register(Command("load", "Load hologram…", self._on_load, "File"))
        c.register(Command("reconstruct", "Reconstruct", self._on_reconstruct, "Process"))
        c.register(Command("autofocus", "Autofocus", self._on_autofocus, "Process"))
        c.register(Command("candidates", "Find focus candidates", self._on_find_candidates, "Process"))
        c.register(Command("pick_order", "Pick +1 order center (click spectrum)",
                           lambda: self._act_pick_order.setChecked(True), "Process"))
        c.register(Command("reset_order", "Reset +1 order to auto-detect",
                           self._reset_order_center, "Process"))
        c.register(Command("qpi", "Compute QPI", self._on_qpi, "Analyse"))
        c.register(Command("qpi_batch", "QPI batch", self._on_qpi_batch, "Analyse"))
        c.register(Command("depth", "Depth map", self._on_depth_map, "Analyse"))
        c.register(Command("surface_depth", "3D surface (depth)", lambda: self._open_surface("depth"), "Analyse"))
        c.register(Command("surface_phase", "3D surface (phase)", lambda: self._open_surface("phase"), "Analyse"))
        for name in ("dark", "midnight", "light", "high_contrast"):
            c.register(Command(f"theme_{name}", f"Theme: {name.replace('_', ' ')}",
                               lambda n=name: self._apply_theme(n), "View"))
        for mode in WORKFLOW_MODES:
            c.register(Command(f"mode_{mode}", f"Workflow: {mode}",
                               lambda m=mode: self._apply_workflow_mode(m), "View"))
        c.register(Command("reset_layout", "Reset panel layout (docks back home)",
                           self._reset_panel_layout, "View"))
        for key in ("ai", "device", "camera", "timelapse", "report", "audit"):
            c.register(Command(f"panel_{key}", f"Open {key}",
                               lambda k=key: self._panel_action(k), "Tools"))
        c.register(Command("line_profile", "Line profile (phase)",
                           self._open_line_profile, "Analyse"))
        c.register(Command("onboarding", "Onboarding", self._show_onboarding, "Help"))

    @property
    def commands(self) -> CommandRegistry:
        return self._commands

    def _install_shortcuts(self) -> None:
        # Most shortcuts are on menu actions; palette gets an extra binding.
        pass

    # ================================================================
    # Recent files
    # ================================================================
    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        if not self._state.recent:
            act = QAction("(none)", self)
            act.setEnabled(False)
            self._recent_menu.addAction(act)
            return
        for path in self._state.recent:
            self._add(self._recent_menu, Path(path).name, lambda p=path: self._load_path(p))

    # ================================================================
    # Persistence
    # ================================================================
    def export_state(self) -> Ui3State:
        self._params = self._params_from_controls()
        self._state.set_recon_params(self._params)
        try:
            self._state.window_geometry_b64 = bytes(self.saveGeometry().toBase64()).decode()
            self._state.window_state_b64 = bytes(self.saveState().toBase64()).decode()
            self._state.layout_version = _LAYOUT_VERSION
        except Exception:
            pass
        return self._state

    def _restore_geometry(self) -> None:
        # Size autofits the screen (see _autofit_to_screen) rather than
        # replaying a saved geometry that may not fit the current display.
        self._autofit_to_screen()
        try:
            # Only restore the DOCK LAYOUT when it was saved by this exact
            # dock schema — an older/newer layout leaves mismatched docks
            # floating over the central grid (2026-07-10).
            if (self._state.window_state_b64
                    and self._state.layout_version == _LAYOUT_VERSION):
                from PySide6.QtCore import QByteArray
                self.restoreState(QByteArray.fromBase64(
                    self._state.window_state_b64.encode()))
        except Exception:
            _LOG.debug("dock layout restore failed", exc_info=True)

    def _autofit_to_screen(self) -> None:
        """Size the window to the usable screen so everything fits without any
        manual resizing (the tall control dock scrolls for overflow). Runs on
        startup and once on first show, when the real screen + frame margins
        are known. Width is capped so it doesn't sprawl on a huge monitor;
        height fills the available space (the left panel is tall)."""
        try:
            from PySide6.QtWidgets import QApplication
            screen = self.screen() or QApplication.primaryScreen()
            if screen is None:
                return
            avail = screen.availableGeometry()
            # Frame margins (title bar / borders) not counted by resize();
            # known only once mapped, else 0.
            dw = max(0, self.frameGeometry().width() - self.width())
            dh = max(0, self.frameGeometry().height() - self.height())
            w = max(640, min(1920, avail.width() - dw))
            h = max(480, avail.height() - dh)
            self.resize(w, h)
            self.move(avail.x(), avail.y())
        except Exception:
            _LOG.debug("autofit failed", exc_info=True)

    def _open_line_profile(self) -> None:
        """Attach a click-drag line-profile tool to the phase panel and show
        its plot dialog."""
        if not self.panel_phase.has_image():
            self.set_status("Reconstruct first — nothing to profile.", "warn")
            return
        try:
            from ui3.widgets.line_profile import LineProfileDialog, LineProfileTool
            tool = getattr(self, "_line_tool", None)
            if tool is None:
                tool = LineProfileTool(self.panel_phase)
                self._line_tool = tool
            dlg = LineProfileDialog(tool, self.panel_context(), self)
            from ui3.dialogs._present import present_centered
            present_centered(dlg)
            self.set_status("Line profile: drag on the Phase panel to sample.")
        except Exception as exc:
            self.set_status(f"Line profile unavailable: {exc}", "warn")

    def _show_onboarding(self) -> None:
        try:
            from ui3.dialogs.onboarding import OnboardingDialog
            dlg = self._dialogs.get("onboarding")
            if dlg is None:
                dlg = OnboardingDialog(self.panel_context(), self)
                dlg.finished_with_choice.connect(self._on_onboarding_choice)
                self._dialogs["onboarding"] = dlg
            from ui3.dialogs._present import present_centered
            present_centered(dlg)
        except Exception as exc:
            self.set_status(f"Onboarding unavailable: {exc}", "warn")

    def _about(self) -> None:
        self.set_status("DHM Reconstruction — ui3 (PySide6 + pyqtgraph)")

    # ================================================================
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Autofit once the window is actually mapped: only now are the real
        # screen and the frame (title-bar) margins known, so the fit is exact.
        if not getattr(self, "_autofit_on_show", False):
            self._autofit_on_show = True
            self._autofit_to_screen()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_toasts"):
            self._toasts.resize_to_parent()

    def _shutdown_panels(self) -> None:
        """Stop every panel/dialog that owns a thread or timer before the
        window (and its C++ object graph) is destroyed.

        Qt does NOT deliver closeEvent to a dock's content widget when the
        main window closes, so we tear each panel down explicitly: a
        ``shutdown()`` hook where one exists (ai/timelapse), else ``close()``
        which fires the panel's own closeEvent (camera/device stop their
        thread/timer there). Otherwise a live acquisition/AI/timelapse thread
        would outlive the widgets and hang or segfault on exit."""
        for panel in list(self._panels.values()) + list(
                getattr(self, "_dialogs", {}).values()):
            try:
                shutdown = getattr(panel, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                panel.close()
            except Exception:
                _LOG.debug("panel shutdown failed", exc_info=True)

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self._shutdown_panels()
            self._bridge.shutdown()
        finally:
            super().closeEvent(event)
