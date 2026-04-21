import logging
from datetime import datetime
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QDockWidget,
    QSizePolicy,
    QMenu,
    QFileDialog,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QSettings, QPoint
import pyqtgraph as pg
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

from .toolbar import MainToolbar
from .status_bar import MainStatusBar
from .sidebar.sidebar_tabs import SidebarTabs
from .panels.image_panel import ImagePanel
from .panels.phase_panel import PhasePanel

from .workers.adaptive_focus_worker import AdaptiveFocusWorker
from .workers.recording_worker import RecordingWorker
from .workers.reconstruction_worker import ReconstructionWorker
from .workers.acquisition_worker import AcquisitionWorker
from .widgets.autofocus_overlay import AutofocusOverlay

from core.profile_manager import ProfileManager

try:
    from core.audit import get_audit_log
except Exception:  # pragma: no cover — audit module may not be present yet
    get_audit_log = None  # type: ignore[assignment]


def _nice_scalebar_length(pixel_um: float) -> float:
    """Pick a round scale-bar length (in µm) that spans ~10-20 % of a 1000 px field."""
    target_um = pixel_um * 150  # ~150 px worth
    import math
    exp = math.floor(math.log10(max(target_um, 1e-6)))
    mantissa = target_um / (10 ** exp)
    for nice in [1, 2, 5, 10, 20, 50, 100]:
        if nice >= mantissa:
            return nice * (10 ** exp)
    return round(target_um)


class _ReturnToDockOnClose(QDockWidget):
    """Utility dock that returns to its layout home rather than closing permanently."""
    def closeEvent(self, event):
        event.ignore()
        try:
            self.setFloating(False)
        except Exception:
            pass
        self.show()

class MainWindow(QMainWindow):
    """The root GUI window integrating the toolbar, status bar, sidebar tabs, and image views."""

    def __init__(self):
        super().__init__()
        self._qt_settings = QSettings("DHM", "Reconstruction")
        self.setWindowTitle("DHM Reconstruction (Mac)")
        self.resize(1200, 800)

        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

        # Central Widget (intentionally empty; workspace is composed of dock widgets)
        central_widget = QWidget()
        central_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        central_widget.setMinimumSize(0, 0)
        central_widget.setMaximumSize(0, 0)
        self.setCentralWidget(central_widget)

        self.setAcceptDrops(True)
        pg.setConfigOptions(imageAxisOrder="row-major")

        self._init_ui()

    def _init_ui(self) -> None:
        """Instantiates all discrete UI components and layouts them out."""
        # 1. Toolbar
        self.toolbar = MainToolbar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # 2. Status Bar
        self.status_bar = MainStatusBar(self)
        self.setStatusBar(self.status_bar)

        # 3. Sidebar Tabs
        self._profile_manager = ProfileManager()
        self.sidebar_tabs = SidebarTabs(self)
        self.settings_dock = QDockWidget("Settings", self)
        self.settings_dock.setObjectName("dock_settings")
        self.settings_dock.setWidget(self.sidebar_tabs)
        self.settings_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.settings_dock.setMinimumWidth(380)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.settings_dock)

        # 4. Image Panels
        self.panel_input = ImagePanel()
        self.panel_amp = ImagePanel()
        self.panel_phase = PhasePanel()
        self.panel_spectrum = ImagePanel()

        # 5. Image Grid Central Widget
        from .image_grid import ImageGrid
        self.image_grid = ImageGrid([
            self.panel_input,
            self.panel_amp,
            self.panel_phase,
            self.panel_spectrum
        ])
        
        self.setCentralWidget(self.image_grid)

        self._loaded_path = None
        self._loaded_array = None
        self._loaded_metadata = None

        self._recon_complex = None
        self._recon_complex_raw = None   # pre-reference-subtraction field
        self._phase_unwrapped = None
        self._spectrum_mag = None

        # Reference hologram for phase correction
        self._reference_complex = None   # reconstructed complex field of reference
        self._reference_fc = None        # pre-propagation extracted field (for autofocus)
        self._reference_source = None    # str: filename or "captured"

        # QPI state
        self._qpi_worker = None
        self._qpi_last_result = None
        self._qpi_dialog = None
        self._qpi_3d_window = None

        # Last-known operation state (surfaced in reports / audit log)
        self._last_recon_params: dict | None = None
        self._last_af_result: dict | None = None
        self._last_qpi_result = None  # alias kept for backward clarity

        self._batch_paths = []
        self._batch_out_root = None
        self._batch_running = False

        self._export_crop_roi = None
        self._export_crop_enabled = False
        self._export_crop_source_view = None

        # Tools overlay state
        self._scalebar_items = []     # list of (panel, ScaleBar) tuples
        self._crosshair_items = []    # list of (panel, vLine, hLine, label) tuples
        self._crosshair_proxies = []  # mouse-move proxy connections

        # Line profile tool state
        self._line_profile_roi = None
        self._line_profile_panel = None
        self._line_profile_dlg = None
        self._line_profile_plot = None
        self._line_profile_curve = None
        self._line_profile_stats = None
        self._lp_ch = None  # crosshair measurement state
        self._lp_panel_combo = None  # panel selector combo in line profile dialog

        self._frame_counter = 0
        self._adaptive_state = None

        # ROI Tracker
        from core.phase_tracker import PhaseTracker
        self._phase_tracker = PhaseTracker(roi_size=64)
        self._roi_select_mode = False   # True while waiting for click
        self._roi_rect_item = None      # pyqtgraph ROI rectangle overlay
        self._detected_objects = []     # list of DetectedObject
        self._detected_overlays = []    # list of pyqtgraph items for detected objects
        self._object_select_mode = False  # True after detect, waiting for click
        
        self._adaptive_worker = AdaptiveFocusWorker(self)
        self._adaptive_worker.new_z_discovered.connect(self._on_adaptive_focus_done)
        self._adaptive_worker.error_occurred.connect(lambda e: self.status_bar.show_message(e))

        self._recording_worker = RecordingWorker(self)
        self._recording_worker.finished_recording.connect(
            lambda path: self.status_bar.show_message(f"Video saved: {path}")
        )
        self._recording_worker.frame_written.connect(
            lambda count: self.status_bar.show_message(f"Recording frame {count}")
        )
        self._recording_worker.error_occurred.connect(self._on_recording_worker_error)

        self._recon_worker = ReconstructionWorker()
        self._recon_worker.recon_completed.connect(self._on_recon_completed)
        self._recon_worker.error_occurred.connect(lambda e: self.status_bar.show_message(f"Recon failed: {e}"))
        self._recon_worker.start()

        self._camera = None
        self._acq_worker = None
        self._af_worker = None

        self._af_overlay = AutofocusOverlay(self)
        self._af_overlay.cancel_requested.connect(self._on_af_cancel)

        self._connect_signals()
        self._setup_shortcuts()
        self._setup_dirty_tracking()
        self._init_menus()
        self._init_image_panel_context_menus()
        self._install_error_ui()
        self._load_persisted_settings()
        self.status_bar.show_message("Ready")

        self._refresh_profiles("all")
        
        # Default mode is File — hide Camera tab
        self._on_mode_changed(self.toolbar.mode_combo.currentText())

        # Run startup FFT benchmark
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self._run_fft_benchmark)
        
        # Restore window state
        self._restore_layout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_af_overlay') and self._af_overlay.isVisible():
            self._af_overlay._reposition()

    def closeEvent(self, event):
        """Save settings and cleanly stop all worker threads before shutdown.

        Every QThread-based worker must finish before the QMainWindow is
        destroyed — otherwise Qt aborts with `QThread: Destroyed while thread
        is still running`. We attempt graceful cancellation first (each worker
        advertises some variant of stop/cancel), then fall through to the
        Qt-builtin interrupt + quit, then wait with a bounded timeout. Workers
        that have no cooperative stop path are still given a chance to drain
        via `wait()`; if they still block, we log and move on rather than
        calling `terminate()` (terminating mid-FFT can leave the FFTW plan
        cache in an undefined state).
        """
        self._save_layout()

        # Stop autofocus worker first — its cancel() resolves fastest and the
        # AF overlay widget expects a clean shutdown.
        if self._af_worker and self._af_worker.isRunning():
            try:
                self._af_worker.cancel()
            except Exception:
                log.warning("af_worker cancel() failed", exc_info=True)
            self._af_worker.wait(5000)

        # Reconstruction worker has its own stop()
        if hasattr(self, '_recon_worker'):
            try:
                self._recon_worker.stop()
            except Exception:
                log.warning("recon_worker stop() failed", exc_info=True)
            self._recon_worker.wait(5000)

        # Remaining workers — uniform best-effort cleanup.
        for attr in ("_batch_worker", "_qpi_worker", "_acq_worker",
                     "_recording_worker", "_adaptive_worker"):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            self._stop_worker_safely(worker, attr, timeout_ms=3000)

        try:
            self._on_camera_disconnect()
        except Exception:
            log.warning("camera disconnect on close failed", exc_info=True)

        # Clean up QPI, 3D, and line profile windows
        self._cleanup_qpi_dialog()
        self._cleanup_3d_window()
        self._clear_line_profile()
        super().closeEvent(event)

    def _stop_worker_safely(self, worker, name: str, *, timeout_ms: int = 3000) -> None:
        """Best-effort QThread shutdown: cooperative stop → interrupt → quit → wait.

        Workers expose different stop APIs (stop/cancel/request_stop) or none
        at all. We try each cooperative variant, then fall through to Qt's
        built-in interruption + event-loop quit, then wait up to `timeout_ms`.
        Never calls `terminate()` — forcible termination mid-compute can
        corrupt shared caches (FFTW plans, CachedReconstructor state).
        """
        # 1. Cooperative stop methods the worker may advertise
        for method_name in ("cancel", "stop", "request_stop"):
            method = getattr(worker, method_name, None)
            if callable(method):
                try:
                    method()
                    break
                except Exception:
                    log.warning("%s.%s() raised during close", name, method_name, exc_info=True)
        # 2. Qt-level: ask the thread to interrupt (worker must check
        #    isInterruptionRequested() in its run loop; harmless if it doesn't)
        try:
            worker.requestInterruption()
        except Exception:
            pass
        # 3. End the event loop if one is running
        try:
            worker.quit()
        except Exception:
            pass
        # 4. Bounded wait. If it times out we log and proceed — do NOT
        #    terminate() here.
        try:
            if worker.isRunning():
                finished = worker.wait(timeout_ms)
                if not finished:
                    log.warning(
                        "%s did not finish within %d ms on close; leaving it to Qt",
                        name, timeout_ms,
                    )
        except Exception:
            log.warning("wait() on %s failed", name, exc_info=True)

    def _save_layout(self):
        """Persists window geometry and image grid boundaries to QSettings."""
        self._qt_settings.setValue("window/geometry", self.saveGeometry())
        self._qt_settings.setValue("window/state", self.saveState())
        if hasattr(self, 'image_grid'):
            self._qt_settings.setValue("grid/state", self.image_grid.save_state())

    def _restore_layout(self):
        """Recovers dock positions, window bounds, and grid dimensions."""
        geom = self._qt_settings.value("window/geometry")
        if geom:
            self.restoreGeometry(geom)
            
        state = self._qt_settings.value("window/state")
        if state:
            self.restoreState(state)
            
        if hasattr(self, 'image_grid'):
            grid_state = self._qt_settings.value("grid/state")
            if grid_state:
                self.image_grid.restore_state(grid_state)

    def _setup_shortcuts(self):
        """Install every window-owned keyboard shortcut.

        Sources its shortcut table from the :mod:`gui.commands` registry
        via :mod:`gui.commands_install`. The registry is the single source
        of truth for ids, titles, and key sequences; this method just
        materializes the QShortcut side of it.
        """
        # QShortcut lives in QtGui (PySide6 ≥ 6.0); keep the import local
        # so this module stays side-effect-free on import.
        from PySide6.QtGui import QKeySequence, QShortcut
        from gui.commands_install import (
            install_main_window_commands,
            install_shortcuts,
        )

        install_main_window_commands(self)
        # Keep a reference so the shortcuts aren't GC'd — Qt parents them
        # to ``self`` too, but holding an attribute makes debugging easier.
        self._command_shortcuts = install_shortcuts(self)

        # ⌘K / Ctrl+K opens the command palette. Kept outside the registry
        # because the palette itself isn't a registered command — it's the
        # *launcher* for them, and we don't want it to appear in its own
        # list.
        self._palette_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self._palette_shortcut.setObjectName("sc_open_command_palette")
        self._palette_shortcut.activated.connect(self._open_command_palette)

        # Esc = cancel the active worker. Kept at WindowShortcut context
        # (the default) so modal dialogs and the error drawer keep their
        # own Esc-to-close behaviour — only when the main window is the
        # active focus target does Esc route through the cancel walker.
        self._esc_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._esc_shortcut.setObjectName("sc_cancel_active_worker")
        self._esc_shortcut.activated.connect(self._cancel_active_worker)

    def _open_command_palette(self) -> None:
        """Show the ⌘K palette, lazily creating it on first use."""
        palette = getattr(self, "_command_palette", None)
        if palette is None:
            from gui.widgets.command_palette import CommandPalette
            palette = CommandPalette(self)
            self._command_palette = palette
        palette.show()
        palette.raise_()
        palette.activateWindow()

    def _install_error_ui(self) -> None:
        """Wire the :class:`ToastHost`, :class:`ErrorDrawer`, and
        :class:`ProgressLine` to the error/progress buses.

        The toast host subscribes to the singleton ``ErrorCenter``
        immediately; any worker that emits a structured
        :class:`~core.errors.ErrorEvent` surfaces as a toast. The progress
        line listens to the reconstruction and QPI workers' ``progress``
        signals and stays silent until an operation crosses the
        perceptual-noticeability threshold.
        """
        from gui.commands import Categories, Command, get_registry
        from gui.widgets.error_drawer import ErrorDrawer
        from gui.widgets.progress_line import ProgressLine
        from gui.widgets.toast import ToastHost

        # -- Toast + drawer --------------------------------------------------
        self._toast_host = ToastHost(self)
        self._error_drawer = ErrorDrawer(self)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self._error_drawer,
        )
        self._error_drawer.hide()  # only shown on demand

        self._toast_host.show_drawer_requested.connect(
            self._error_drawer.open_for_event
        )
        self._toast_host.attach()

        # -- Progress line ---------------------------------------------------
        self._progress_line = ProgressLine(self)
        # Mount as a permanent status-bar widget so it sits below the
        # image grid without restructuring the central layout. It renders
        # at height 0 while idle, so no chrome cost.
        self.status_bar.addPermanentWidget(self._progress_line, 1)
        self._progress_line.cancel_requested.connect(self._cancel_active_worker)

        # Wire the reconstruction worker — it already exists at this point.
        try:
            self._recon_worker.progress.connect(self._progress_line.on_progress)
        except AttributeError:
            pass  # worker without new-protocol progress signal (compat)

        # -- Error drawer discovery via ⌘K ----------------------------------
        reg = get_registry()
        if reg.get("help.show_error_log") is None:
            reg.register(Command(
                id="help.show_error_log",
                title="Show error log",
                category=Categories.HELP,
                shortcut="",
                hint="Open the session error drawer",
                callback=lambda: self._error_drawer.toggle(),
            ))

    # ─── Parameter persistence (T1.4) ─────────────────────────────────────
    def _load_persisted_settings(self) -> None:
        """Populate sidebar widgets from disk on startup.

        Runs once, just after UI construction. Invalid on-disk state is
        rejected inside :mod:`gui.settings_store` and we receive defaults
        instead — nothing crashes, the worst-case is that the widgets show
        their factory values.
        """
        try:
            from gui import settings_store
            from gui.persistence import apply_settings
        except Exception:
            log.warning("persistence: import failed, using widget defaults",
                        exc_info=True)
            return
        try:
            self._app_settings = settings_store.load()
            apply_settings(self.sidebar_tabs, self._app_settings)
        except Exception:
            log.warning("persistence: load+apply failed", exc_info=True)
            # Hold a default in memory so save paths still have somewhere to
            # merge I/O history into.
            from core.settings_schema import AppSettings
            self._app_settings = AppSettings.defaults()
        self._sync_toolbar_default_dir()

    def _sync_toolbar_default_dir(self) -> None:
        """Push the latest ``last_folder`` onto the toolbar so the next Load
        dialog opens there."""
        try:
            self.toolbar.default_dir = self._app_settings.io.last_folder or ""
        except Exception:
            log.debug("persistence: toolbar default_dir sync failed",
                      exc_info=True)

    def _persist_current_settings(self) -> None:
        """Snapshot the sidebar and merge with the live I/O history, then save.

        Called after a successful recon/AF/QPI run so the next launch
        reopens in the same state the lab last used. Silently swallows
        storage errors — the app continues; the next save has another
        chance.
        """
        try:
            from gui import settings_store
            from gui.persistence import collect_settings
        except Exception:  # pragma: no cover
            log.debug("persistence: module import failed on save",
                      exc_info=True)
            return
        try:
            io = getattr(self._app_settings, "io", None) if getattr(
                self, "_app_settings", None
            ) is not None else None
            snapshot = collect_settings(self.sidebar_tabs, io=io)
            settings_store.save(snapshot)
            self._app_settings = snapshot
        except Exception:
            log.warning("persistence: save failed", exc_info=True)

    def _update_io_history(self, **updates) -> None:
        """Merge one or more :class:`IODefaults` fields into the live snapshot
        and write it to disk immediately.

        Called whenever the user confirms a file-dialog path — threading
        ``last_folder``/``last_hologram``/``last_report_folder`` back into
        the schema so the next dialog opens where the user left off.
        """
        try:
            from gui import settings_store
            from core.settings_schema import AppSettings, IODefaults
        except Exception:  # pragma: no cover
            return
        try:
            current = getattr(self, "_app_settings", None) or AppSettings.defaults()
            new_io = IODefaults(**{
                **{
                    "last_folder": current.io.last_folder,
                    "last_preset": current.io.last_preset,
                    "last_hologram": current.io.last_hologram,
                    "last_report_folder": current.io.last_report_folder,
                },
                **updates,
            })
            self._app_settings = current.with_io(
                last_folder=new_io.last_folder,
                last_preset=new_io.last_preset,
                last_hologram=new_io.last_hologram,
                last_report_folder=new_io.last_report_folder,
            )
            settings_store.save(self._app_settings)
            self._sync_toolbar_default_dir()
        except Exception:
            log.debug("persistence: io merge failed", exc_info=True)

    def _last_folder_default(self) -> str:
        """Return the last-used folder or empty string (Qt opens $HOME)."""
        try:
            return self._app_settings.io.last_folder or ""
        except AttributeError:
            return ""

    def _cancel_active_worker(self) -> None:
        """Best-effort cancel of whatever worker is live right now.

        The progress line doesn't know which worker produced an event; we
        walk the candidates in priority order (QPI → AF → recon) and call
        each one's cancellation hook if the worker is running. Called from
        two places: the progress line's Esc-hint click, and the global
        :kbd:`Esc` shortcut installed in :meth:`_setup_shortcuts`.

        Silent when no worker is running — Esc with nothing to cancel must
        not flash the status bar or steal focus.
        """
        for attr, method, label in (
            ("_qpi_worker", "cancel", "QPI"),
            ("_af_worker", "cancel", "Autofocus"),
            ("_recon_worker", "stop", "Reconstruction"),
        ):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            try:
                if hasattr(worker, "isRunning") and not worker.isRunning():
                    continue
                getattr(worker, method)()
                # Acknowledge the cancel request so the user sees that Esc
                # landed. The worker's own ``cancelled`` signal will still
                # fire once it actually stops; this is just immediate UI
                # feedback for the keystroke.
                try:
                    self.status_bar.show_message(
                        f"{label}: cancel requested", timeout=4000
                    )
                except Exception:
                    pass
                break
            except Exception:
                continue

    def _reset_layout_to_defaults(self):
        """Resets the layout to factory defaults."""
        self._qt_settings.remove("window/geometry")
        self._qt_settings.remove("window/state")
        self._qt_settings.remove("grid/state")
        
        if hasattr(self, 'image_grid'):
            self.image_grid.restore_grid()
            
        self.settings_dock.show()
        # Reset to Recon tab (index 1 assuming Camera is 0)
        # Using safely by looking it up if possible
        try:
            self.sidebar_tabs.tabs.setCurrentIndex(1)
        except Exception:
            pass
            
        self.status_bar.show_message("Layout reset to defaults. Restart app for full effect.", timeout=3000)

    def _connect_signals(self) -> None:
        """Wires up UI interactions to their handlers."""
        self.toolbar.file_load_requested.connect(self._load_file_path)
        self.toolbar.batch_show_requested.connect(self._show_batch_dock)
        self.toolbar.reconstruct_requested.connect(self._trigger_reconstruction)
        self.toolbar.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        # Profile bindings
        self.toolbar.setup_profile_combo.profile_action.connect(
            lambda act, name: self._on_profile_action("setup", act, name)
        )
        self.toolbar.setup_profile_combo.currentIndexChanged.connect(
            lambda: self._on_profile_selected("setup")
        )
        self.toolbar.cam_profile_combo.profile_action.connect(
            lambda act, name: self._on_profile_action("camera", act, name)
        )
        self.toolbar.cam_profile_combo.currentIndexChanged.connect(
            lambda: self._on_profile_selected("camera")
        )
        self._profile_manager.profiles_changed.connect(self._refresh_profiles)
        
        rtab = self.sidebar_tabs.recon_tab
        rtab.recon_btn.clicked.connect(self._trigger_reconstruction)
        
        rectab = self.sidebar_tabs.record_tab
        rectab.browse_btn.clicked.connect(self._select_video_output)
        rectab.action_btn.clicked.connect(self._on_record_action)
        self.sidebar_tabs.focus_tab.autofocus_btn.clicked.connect(self._autofocus)
        self.sidebar_tabs.focus_tab.benchmark_btn.clicked.connect(self._run_af_benchmark)
        self.sidebar_tabs.focus_tab.diagnostic_btn.clicked.connect(self._run_af_diagnostic)

        camtab = self.sidebar_tabs.camera_tab
        camtab.connect_btn.clicked.connect(self._on_camera_connect)
        camtab.disconnect_btn.clicked.connect(self._on_camera_disconnect)
        camtab.start_acq_btn.clicked.connect(self._on_acq_start)
        camtab.stop_acq_btn.clicked.connect(self._on_acq_stop)

        # Tools
        self.toolbar.crop_toggled.connect(self._on_crop_toggled)
        self.toolbar.scalebar_toggled.connect(self._on_scalebar_toggled)
        self.toolbar.crosshair_toggled.connect(self._on_crosshair_toggled)
        self.toolbar.line_profile_toggled.connect(self._on_line_profile_toggled)
        self.toolbar.export_view_requested.connect(self._on_export_view)

        # Contrast: re-display images when contrast settings change
        self.sidebar_tabs.process_tab.contrast.changed.connect(self._display_recon_images)

        # Reference hologram
        self.sidebar_tabs.process_tab.ref_load_requested.connect(self._load_reference_from_file)
        self.sidebar_tabs.process_tab.ref_capture_requested.connect(self._capture_reference)
        self.sidebar_tabs.process_tab.ref_clear_requested.connect(self._clear_reference)

        # QPI
        self.sidebar_tabs.qpi_tab.compute_requested.connect(self._compute_qpi)
        self.sidebar_tabs.qpi_tab.display_changed.connect(self._on_qpi_display_changed)

        # ROI Tracker
        self.sidebar_tabs.focus_tab.roi_select_btn.toggled.connect(self._on_roi_select_toggled)
        self.sidebar_tabs.focus_tab.detect_objects_btn.clicked.connect(self._on_detect_objects)
        self.sidebar_tabs.focus_tab.roi_clear_btn.clicked.connect(self._reset_roi_state)
        self.sidebar_tabs.focus_tab.roi_tracker_group.toggled.connect(
            lambda checked: None if checked else self._reset_roi_state()
        )

    # ─── Profile Management ───
    def _refresh_profiles(self, profile_type: str = "all"):
        if profile_type in ["setup", "all"]:
            cb = self.toolbar.setup_profile_combo
            cb.blockSignals(True)
            current = cb.currentText()
            cb.clear()
            profs = self._profile_manager.list_profiles("setup")
            cb.addItem("Default")
            cb.addItems(profs)
            if current:
                idx = cb.findText(current)
                if idx >= 0: cb.setCurrentIndex(idx)
            cb.blockSignals(False)
            
        if profile_type in ["camera", "all"]:
            cb = self.toolbar.cam_profile_combo
            cb.blockSignals(True)
            current = cb.currentText()
            cb.clear()
            profs = self._profile_manager.list_profiles("camera")
            cb.addItem("Default")
            cb.addItems(profs)
            if current:
                idx = cb.findText(current)
                if idx >= 0: cb.setCurrentIndex(idx)
            cb.blockSignals(False)

    def _on_profile_action(self, ptype: str, action: str, name: str):
        if action == "save":
            state = self._get_profile_state(ptype)
            self._profile_manager.save_profile(ptype, name, state)
            self.status_bar.show_message(f"Saved {ptype} profile '{name}'")
        elif action == "new":
            state = self._get_profile_state(ptype)
            self._profile_manager.save_profile(ptype, name, state)
            self.status_bar.show_message(f"Created new {ptype} profile '{name}'")
        elif action == "duplicate":
            state = self._profile_manager.load_profile(ptype, name.replace(" Copy", ""))
            if not state:
                state = self._get_profile_state(ptype)
            self._profile_manager.save_profile(ptype, name, state)
        elif action == "delete":
            self._profile_manager.delete_profile(ptype, name)

    def _get_profile_state(self, ptype: str) -> dict:
        if ptype == "setup":
            return {
                "recon": self.sidebar_tabs.recon_tab.get_state(),
                "process": self.sidebar_tabs.process_tab.get_state(),
                "focus": self.sidebar_tabs.focus_tab.get_state()
            }
        elif ptype == "camera":
            return {
                "camera": self.sidebar_tabs.camera_tab.get_state()
            }
        return {}

    def _on_profile_selected(self, ptype: str):
        cb = self.toolbar.setup_profile_combo if ptype == "setup" else self.toolbar.cam_profile_combo
        name = cb.currentText().replace("*", "").strip()
        if name == "Default" or not name:
            return
            
        data = self._profile_manager.load_profile(ptype, name)
        if not data:
            return
            
        if ptype == "setup":
            if "recon" in data: self.sidebar_tabs.recon_tab.set_state(data["recon"])
            if "process" in data: self.sidebar_tabs.process_tab.set_state(data["process"])
            if "focus" in data: self.sidebar_tabs.focus_tab.set_state(data["focus"])
        elif ptype == "camera":
            if "camera" in data: self.sidebar_tabs.camera_tab.set_state(data["camera"])
            
        self.status_bar.show_message(f"Loaded {ptype} profile '{name}'")
        
    def _mark_profile_dirty(self, ptype: str):
        cb = self.toolbar.setup_profile_combo if ptype == "setup" else self.toolbar.cam_profile_combo
        current = cb.currentText()
        if not current.endswith("*") and current and current != "Default":
            cb.setItemText(cb.currentIndex(), current + "*")

    def _setup_dirty_tracking(self):
        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QLineEdit
        
        def hook_widget(w, ptype):
            if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                w.valueChanged.connect(lambda *args: self._mark_profile_dirty(ptype))
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(lambda *args: self._mark_profile_dirty(ptype))
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(lambda *args: self._mark_profile_dirty(ptype))
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(lambda *args: self._mark_profile_dirty(ptype))

        for tab in [self.sidebar_tabs.recon_tab, self.sidebar_tabs.process_tab, self.sidebar_tabs.focus_tab]:
            for wtype in (QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QLineEdit):
                for w in tab.findChildren(wtype):
                    hook_widget(w, "setup")
                
        for wtype in (QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QLineEdit):
            for w in self.sidebar_tabs.camera_tab.findChildren(wtype):
                hook_widget(w, "camera")

    # ─── Batch Rendering ───
    def _show_batch_dock(self) -> None:
        """Opens standalone Batch Rendering dialogue interface."""
        from .dialogs.batch_render_dialog import BatchRenderDialog
        import sys
        
        try:
            from core.batch_renderer import BatchRenderer
        except ImportError:
            import os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from core.batch_renderer import BatchRenderer
            
        if not hasattr(self, '_batch_dialog'):
            self._batch_dialog = BatchRenderDialog(self, self._profile_manager)
            self._batch_worker = BatchRenderer(self)
            
            self._batch_worker.progress.connect(self._batch_dialog.set_progress)
            self._batch_worker.eta_update.connect(self._batch_dialog.set_eta)
            self._batch_worker.status.connect(self._batch_dialog.set_status)
            
            def on_batch_err(e):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self._batch_dialog, "Batch Render Error", str(e))
                self.status_bar.show_message(f"Batch Error: {e}")
            self._batch_worker.error_occurred.connect(on_batch_err)
            
            self._batch_worker.finished_batch.connect(
                lambda: self._batch_dialog.start_btn.setEnabled(True)
            )
            self._batch_worker.finished_batch.connect(
                lambda: self._batch_dialog.cancel_btn.setText("Close")
            )
            
            def on_start(cfg):
                self._batch_dialog.set_eta("--:--")
                # Pass reference complex field to batch if enabled
                ptab = self.sidebar_tabs.process_tab
                if ptab.ref_enable_cb.isChecked() and self._reference_complex is not None:
                    cfg["reference_complex"] = self._reference_complex
                self._batch_worker.setup(cfg, self._profile_manager)
                self._batch_worker.start()
                
            self._batch_dialog.start_batch_requested.connect(on_start)
            self._batch_dialog.rejected.connect(self._batch_worker.stop)

        self._batch_dialog.show()
        self._batch_dialog.raise_()

    # ─── Drag & Drop ───
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        p = Path(urls[0].toLocalFile())
        self._load_file_path(p)

    def _load_file_path(self, path) -> bool:
        """Handles reading image data from disk using core.ingestion."""
        from core.ingestion import load_any, UnsupportedFormatError, OptionalDependencyError
        path = Path(path)  # ensure Path regardless of str or Path input
        
        try:
            loaded = load_any(path)
        except (UnsupportedFormatError, OptionalDependencyError, RuntimeError) as e:
            self.status_bar.show_message(str(e))
            self.status_bar.set_info_text(str(e))
            return False

        self._loaded_path = path
        self._loaded_array = loaded.array
        self._loaded_metadata = loaded.metadata

        self._recon_complex = None
        self._phase_unwrapped = None
        self._spectrum_mag = None

        self._clear_export_crop_roi()

        # Clean up stale analysis state from previous file
        self._clear_line_profile()
        if hasattr(self, 'toolbar') and self.toolbar.action_line_profile.isChecked():
            self.toolbar.action_line_profile.blockSignals(True)
            self.toolbar.action_line_profile.setChecked(False)
            self.toolbar.action_line_profile.blockSignals(False)
        self._cleanup_qpi_dialog()
        self._cleanup_3d_window()
        self._qpi_last_result = None
        self.sidebar_tabs.qpi_tab.clear_stats()
        self.sidebar_tabs.qpi_tab.compute_btn.setEnabled(False)

        self._update_info()

        # Update input image view
        img = self._as_field_2d(self._loaded_array)
        self.panel_input.set_image(img, autoLevels=True)
        self._frame_counter += 1
        
        # Enable processing buttons now that we have data
        self.sidebar_tabs.recon_tab.recon_btn.setEnabled(True)
        self.toolbar.action_reconstruct.setEnabled(True)
        self.sidebar_tabs.focus_tab.autofocus_btn.setEnabled(True)
        self.sidebar_tabs.focus_tab.benchmark_btn.setEnabled(True)
        self.sidebar_tabs.focus_tab.diagnostic_btn.setEnabled(True)

        # Remember the folder the hologram came from so the next "Load"
        # and "Save As" dialogs open next to it.
        try:
            self._update_io_history(
                last_folder=str(Path(path).parent),
                last_hologram=str(path),
            )
        except Exception:
            log.debug("persistence: io history update on load failed",
                      exc_info=True)

        return True

    def _as_field_2d(self, arr) -> np.ndarray:
        import numpy as np
        a = np.asarray(arr)
        if a.ndim == 3:
            a = a[..., 0]
        if a.ndim != 2:
            raise RuntimeError(f"Expected 2D image; got shape {a.shape}")
        
        a = a.astype(np.float32, copy=False)
        if np.max(a) > 0:
            a = a / float(np.max(a))
        return a

    def _clear_export_crop_roi(self) -> None:
        if self._export_crop_roi is not None:
            try:
                if self._export_crop_source_view is not None:
                    self._export_crop_source_view.get_view().removeItem(self._export_crop_roi)
            except Exception:
                pass
        self._export_crop_roi = None
        self._export_crop_source_view = None

    def _update_info(self) -> None:
        if self._loaded_array is None or self._loaded_path is None:
            self.status_bar.set_info_text("No file loaded")
            return
            
        arr = self._loaded_array
        md = self._loaded_metadata or {}
        
        lines = [
            f"Loaded: {self._loaded_path.name} | shape: {arr.shape} | type: {arr.dtype} | min/max: {arr.min()} / {arr.max()}"
        ]
        self.status_bar.set_info_text(" | ".join(lines))

    def _build_recon_job(self) -> dict:
        from core.reconstruction import ReconstructionMethod
        rtab = self.sidebar_tabs.recon_tab
        method = ReconstructionMethod(rtab.method_combo.currentData())
        z_m = float(rtab.z_mm.value()) * 1e-3
        wl = float(rtab.wavelength_nm.value()) * 1e-9
        
        px = float(rtab.pixel_um.value()) * 1e-6
        if not rtab.pixel_is_effective_cb.isChecked():
            mag = float(rtab.magnification.value())
            px = px / (mag if mag > 0 else 1.0)
            
        ptab = self.sidebar_tabs.process_tab
        fft_backend = rtab.fft_backend_combo.currentData()
        
        # Include reference complex field if enabled and loaded
        ref_field = None
        if ptab.ref_enable_cb.isChecked() and self._reference_complex is not None:
            ref_field = self._reference_complex

        return {
            'frame_num': self._frame_counter,
            'image': self._as_field_2d(self._loaded_array),
            'method': method,
            'wavelength': wl,
            'pixel_size': px,
            'z': z_m,
            'n_medium': 1.0,
            'subtract_mean': ptab.subtract_mean_cb.isChecked(),
            'hann_window': ptab.hann_cb.isChecked(),
            'mask_radius': int(rtab.mask_radius.value()),
            'mask_apodization': ptab.mask_apod_combo.currentData(),
            'mask_rolloff': ptab.mask_rolloff.value(),
            'fft_backend': fft_backend,
            'filter_enable': ptab.filter_enable_cb.isChecked(),
            'filter_type': ptab.filter_type_combo.currentData(),
            'filter_cutoff': ptab.filter_cutoff.value(),
            'filter_rolloff': ptab.filter_rolloff.value(),
            'unwrap_config': ptab.get_unwrap_config(),
            'reference_complex': ref_field,
        }

    def _trigger_reconstruction(self) -> None:
        """Packages UI parameters and submits an async job to ReconstructionWorker."""
        if self._loaded_array is None:
            return

        self.sidebar_tabs.recon_tab.recon_btn.setEnabled(False)
        self.toolbar.action_reconstruct.setEnabled(False)
        job = self._build_recon_job()

        # Warn about negative z (back-propagation / evanescent waves)
        z_m = job['z']
        if z_m < 0:
            self.status_bar.show_message(
                f"Warning: z={z_m*1e3:.4f} mm is negative (back-propagation). "
                "Evanescent waves are clamped to zero in ASM."
            )
        else:
            self.status_bar.show_message("Reconstructing...")

        self._recon_worker.submit_job(job)

    def _on_recon_completed(self, result: dict) -> None:
        self._recon_complex = result.get('recon_complex')
        self._recon_complex_raw = result.get('recon_complex_raw')
        self._spectrum_mag = result.get('spectrum_mag')
        self._phase_unwrapped = result.get('phase_unwrapped')

        # Snapshot recon params (for report) + audit record
        try:
            self._last_recon_params = self._collect_recon_params_for_report()
        except Exception:
            self._last_recon_params = None

        try:
            phase_arr = self._phase_unwrapped if self._phase_unwrapped is not None else (
                np.angle(self._recon_complex) if self._recon_complex is not None else None
            )
            result_summary = None
            if phase_arr is not None:
                result_summary = {
                    "shape": list(phase_arr.shape),
                    "duration_s": result.get("duration_s"),
                }
            self._audit(
                action="reconstruct",
                params=self._last_recon_params or {},
                result_summary=result_summary,
            )
        except Exception:
            log.warning("audit log failed for reconstruct", exc_info=True)
        
        if self._spectrum_mag is not None:
            spec_log = np.log1p(self._spectrum_mag)
            self.panel_spectrum.set_image(spec_log, autoLevels=True)
            
        if self._recon_complex is not None:
            self._display_recon_images()
                
            if self._recording_worker.is_recording():
                fdict = {
                    "amplitude": np.abs(self._recon_complex),
                    "phase": np.angle(self._recon_complex) if self._phase_unwrapped is None else self._phase_unwrapped,
                    "input": self._as_field_2d(self._loaded_array) if self._loaded_array is not None else np.zeros((10,10), dtype=np.uint8),
                    "z_mm": self.sidebar_tabs.recon_tab.z_mm.value()
                }
                self._recording_worker.push_frame(fdict)
                
        # Update ROI tracker on amplitude (autofocus mask tracking)
        if self._recon_complex is not None:
            self._update_roi_tracking(np.abs(self._recon_complex))

        self.sidebar_tabs.recon_tab.recon_btn.setEnabled(True)
        self.toolbar.action_reconstruct.setEnabled(True)
        has_phase = self._phase_unwrapped is not None
        self.sidebar_tabs.qpi_tab.compute_btn.setEnabled(has_phase)
        self.sidebar_tabs.focus_tab.detect_objects_btn.setEnabled(has_phase)
        self.status_bar.show_message("Reconstruction complete")
        # Persist the parameters that just produced a successful run —
        # next launch opens on the same values.
        self._persist_current_settings()

        # Auto-refresh open QPI windows using the new reconstruction data.
        # Without this, the QPI dialog and 3D surface keep stale results from
        # the previous z until the user manually clicks "Compute QPI" again.
        if has_phase and self._qpi_dialog is not None:
            from shiboken6 import isValid
            if isValid(self._qpi_dialog):
                self._compute_qpi()

        # Live line profile on phase: update when new phase data arrives.
        if self._line_profile_roi is not None and self._phase_unwrapped is not None:
            self._update_line_profile()

    def _display_recon_images(self) -> None:
        """Display amplitude & phase with optional contrast enhancement."""
        if self._recon_complex is None:
            return
        from core.contrast import apply_contrast
        cfg = self.sidebar_tabs.process_tab.contrast.get_contrast_config()

        amp = np.abs(self._recon_complex)
        amp_display = apply_contrast(amp, cfg["method"],
                                     p_low=cfg["p_low"], p_high=cfg["p_high"],
                                     clahe_clip=cfg["clahe_clip"], clahe_grid=cfg["clahe_grid"])
        self.panel_amp.set_image(amp_display, autoLevels=True)

        self.panel_phase.set_wrapped(np.angle(self._recon_complex), autoLevels=True)
        if self._phase_unwrapped is not None:
            if cfg["apply_phase"]:
                phase_display = apply_contrast(self._phase_unwrapped, cfg["method"],
                                               p_low=cfg["p_low"], p_high=cfg["p_high"],
                                               clahe_clip=cfg["clahe_clip"], clahe_grid=cfg["clahe_grid"])
                self.panel_phase.set_unwrapped(phase_display, autoLevels=True)
            else:
                self.panel_phase.set_unwrapped(self._phase_unwrapped, autoLevels=True)

    # ─── Reference Hologram ─────────────────────────────────────────────

    def _reconstruct_reference(self, img_array: np.ndarray, source_name: str):
        """
        Reconstruct a reference hologram with current parameters and store
        the resulting complex field. This field will be divided out from
        every subsequent sample reconstruction to remove systematic
        aberrations (tilt, curvature, lens errors).

        The principle: E_corrected = E_sample / E_reference
        After division, the phase contains only the sample's OPD.
        """
        from core.reconstruction import ReconstructionMethod
        from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
        from core.reconstruction import propagate, ReconstructionParams

        rtab = self.sidebar_tabs.recon_tab
        ptab = self.sidebar_tabs.process_tab

        # Use same parameters as current reconstruction
        method = ReconstructionMethod(rtab.method_combo.currentData())
        z_m = float(rtab.z_mm.value()) * 1e-3
        wl = float(rtab.wavelength_nm.value()) * 1e-9
        px = float(rtab.pixel_um.value()) * 1e-6
        if not rtab.pixel_is_effective_cb.isChecked():
            mag = float(rtab.magnification.value())
            px = px / (mag if mag > 0 else 1.0)
        mask_radius = int(rtab.mask_radius.value())

        try:
            img = img_array.astype(np.float32)
            if img.ndim == 3:
                img = img[..., 0]

            if ptab.subtract_mean_cb.isChecked():
                img = img - float(np.mean(img))
            if ptab.hann_cb.isChecked():
                wy = np.hanning(img.shape[0]).astype(np.float32)
                wx = np.hanning(img.shape[1]).astype(np.float32)
                img = img * (wy[:, None] * wx[None, :])

            max_val = float(np.max(np.abs(img)))
            if max_val > 0:
                img = img / max_val

            apod = ptab.mask_apod_combo.currentData()
            roll = ptab.mask_rolloff.value()
            offaxis_params = OffAxisParams(radius=mask_radius, apodization=apod, rolloff=roll)
            fc, _, _, _ = extract_complex_field_offaxis_debug(img, offaxis_params)

            recon_params = ReconstructionParams(
                wavelength_m=wl, pixel_size_m=px, z_m=z_m, n=1.0
            )
            ref_complex = propagate(fc, recon_params, method, force_python=True)

            self._reference_fc = np.asarray(fc)  # pre-propagation (for autofocus phase metrics)
            self._reference_complex = np.asarray(ref_complex)
            self._reference_source = source_name

            ptab.ref_enable_cb.setChecked(True)
            ptab.ref_status_label.setText(f"Reference: {source_name}")
            ptab.ref_status_label.setStyleSheet("color: #4CAF50; font-style: normal; font-weight: bold;")
            self.status_bar.show_message(f"Reference loaded: {source_name}")

        except Exception as e:
            self.status_bar.show_message(f"Reference reconstruction failed: {e}")
            import traceback; traceback.print_exc()

    def _load_reference_from_file(self):
        """Open a file dialog and load a reference hologram from disk."""
        from PySide6.QtWidgets import QFileDialog
        from core.ingestion import load_any

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference Hologram", self._last_folder_default(),
            "Images (*.tiff *.tif *.png *.bmp *.jpg *.jpeg);;All (*)"
        )
        if not path:
            return
        self._update_io_history(last_folder=str(Path(path).parent))

        try:
            loaded = load_any(path)
            arr = np.asarray(loaded.array)
            from pathlib import Path
            self._reconstruct_reference(arr, Path(path).name)
        except Exception as e:
            self.status_bar.show_message(f"Failed to load reference: {e}")

    def _capture_reference(self):
        """Use the currently loaded/live image as the reference hologram."""
        if self._loaded_array is None:
            self.status_bar.show_message("No image loaded — cannot capture reference")
            return

        arr = self._as_field_2d(self._loaded_array)
        source = "captured"
        if self._loaded_path is not None:
            source = self._loaded_path.name
        self._reconstruct_reference(arr, f"[captured] {source}")

    def _clear_reference(self):
        """Remove the loaded reference hologram."""
        self._reference_complex = None
        self._reference_fc = None
        self._reference_source = None

        ptab = self.sidebar_tabs.process_tab
        ptab.ref_enable_cb.setChecked(False)
        ptab.ref_status_label.setText("No reference loaded")
        ptab.ref_status_label.setStyleSheet("color: #999; font-style: italic;")
        self.status_bar.show_message("Reference cleared")

    # ─── QPI ──────────────────────────────────────────────────────────────

    def _compute_qpi(self) -> None:
        """Launch QPI analysis on a background thread."""
        if self._recon_complex is None or self._phase_unwrapped is None:
            self.status_bar.show_message("Run reconstruction first")
            return

        from core.qpi import QPIMode
        from gui.workers.qpi_worker import QPIWorker

        qtab = self.sidebar_tabs.qpi_tab
        rtab = self.sidebar_tabs.recon_tab

        wl = float(rtab.wavelength_nm.value()) * 1e-9
        px = self._get_effective_pixel_um() * 1e-6  # µm → m

        known_h = None
        if qtab.known_thickness_cb.isChecked():
            known_h = qtab.known_thickness_nm.value() * 1e-9

        job = {
            "phase_unwrapped": self._phase_unwrapped.copy(),
            "recon_complex": self._recon_complex,
            "recon_complex_raw": self._recon_complex_raw,
            "reference_complex": getattr(self, "_reference_complex", None),
            "wavelength_m": wl,
            "pixel_size_m": px,
            "mode": QPIMode(qtab.mode_combo.currentData()),
            "n_sample": qtab.n_sample.value(),
            "n_medium": qtab.n_medium.value(),
            "alpha_SI": qtab.alpha.value() * 1e-6,
            "known_thickness_m": known_h,
            "cell_threshold": qtab.cell_threshold.value(),
            "compute_psd": qtab.show_psd_cb.isChecked(),
            "bg_correction": qtab.bg_correction_cb.isChecked(),
            "bg_poly_order": qtab.bg_poly_order.value(),
            "ref_subtract": qtab.ref_subtract_cb.isChecked(),
            "unwrap_bg_already_removed": self.sidebar_tabs.process_tab.unwrap_post_bg.isChecked(),
        }

        # Disable button while computing
        qtab.compute_btn.setEnabled(False)
        qtab.compute_btn.setText("Computing...")
        self.status_bar.show_message("Computing QPI...")

        worker = QPIWorker()
        worker.setup(job)
        worker.qpi_completed.connect(self._on_qpi_completed)
        worker.qpi_failed.connect(self._on_qpi_failed)
        worker.finished.connect(worker.deleteLater)
        # Route structured progress through the silent 1px line (Tier 1).
        progress_line = getattr(self, "_progress_line", None)
        if progress_line is not None:
            worker.progress.connect(progress_line.on_progress)
        self._qpi_worker = worker  # prevent GC
        worker.start()

    def _on_qpi_completed(self, result) -> None:
        """Handle QPI worker completion — populate stats and show results."""
        qtab = self.sidebar_tabs.qpi_tab
        qtab.compute_btn.setEnabled(True)
        qtab.compute_btn.setText("Compute QPI")

        # Stash for reports; _qpi_last_result is also set inside _show_qpi_window.
        self._last_qpi_result = result

        # Audit QPI completion
        try:
            summary = {
                "n_sample": float(qtab.n_sample.value()),
                "n_medium": float(qtab.n_medium.value()),
            }
            for attr in ("total_dry_mass_pg", "step_height_m"):
                val = getattr(result, attr, None)
                if val is not None:
                    try:
                        summary[attr] = float(val)
                    except Exception:
                        pass
            ph_stats = getattr(result, "phase_stats", None)
            if ph_stats is not None and getattr(ph_stats, "range_nm", None) is not None:
                try:
                    summary["opd_range_nm"] = float(ph_stats.range_nm)
                except Exception:
                    pass
            self._audit(
                action="qpi",
                params={"n_sample": summary["n_sample"], "n_medium": summary["n_medium"]},
                result_summary=summary,
            )
        except Exception:
            log.warning("audit log failed for qpi", exc_info=True)

        try:
            # Clear stale stats from previous computation before populating
            qtab.clear_stats()

            # Populate stats panels
            if result.phase_stats is not None:
                qtab.set_phase_stats(result.phase_stats)
            if result.cell_morph is not None:
                qtab.set_cell_stats(result.cell_morph)
            if result.roughness is not None:
                qtab.set_roughness_stats(result.roughness)

            # Show QPI maps in a separate popup window
            self._show_qpi_window(result, qtab)

            msg = "QPI complete"
            if result.total_dry_mass_pg is not None and result.total_dry_mass_pg > 0:
                msg += f" — Dry mass: {result.total_dry_mass_pg:.1f} pg"
            if result.roughness is not None:
                msg += f" | Ra: {result.roughness.Ra*1e9:.2f} nm"
            if result.step_height_m is not None:
                msg += f" | Step: {result.step_height_m*1e9:.1f} nm"
            if result.phase_stats is not None:
                msg += f" | OPD range: {result.phase_stats.range_nm:.1f} nm"
            self.status_bar.show_message(msg)
            # Persist the refractive-index values the user just validated.
            self._persist_current_settings()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_bar.show_message(f"QPI display error: {e}")

    def _on_qpi_failed(self, error_msg: str) -> None:
        """Handle QPI worker failure."""
        qtab = self.sidebar_tabs.qpi_tab
        qtab.compute_btn.setEnabled(True)
        qtab.compute_btn.setText("Compute QPI")
        self.status_bar.show_message(f"QPI failed: {error_msg}")

    # ─── QPI Display Helpers ───
    @staticmethod
    def _make_qpi_map_widget(data: np.ndarray, title: str, cmap_name: str = "viridis",
                             pixel_size_um: float = None):
        """Create a titled image panel with colorbar for a QPI map."""
        import pyqtgraph as pg
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
        from PySide6.QtCore import QRectF

        # Sanitize data: replace NaN/Inf with 0
        data = np.array(data, dtype=np.float64, copy=True)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)

        # Title with min/max/mean stats
        dmin, dmax, dmean = float(np.min(data)), float(np.max(data)), float(np.mean(data))
        lbl = QLabel(f"<b>{title}</b>  min={dmin:.2g}  max={dmax:.2g}  mean={dmean:.2g}")
        lbl.setStyleSheet("color: #ddd; font-size: 11px; padding: 2px;")
        vbox.addWidget(lbl)

        gw = pg.GraphicsLayoutWidget()
        plot = gw.addPlot()
        plot.setAspectLocked(True)
        plot.invertY(True)
        plot.hideButtons()

        img = pg.ImageItem(image=data)
        cmap = pg.colormap.get(cmap_name, source="matplotlib")
        if cmap is None:
            cmap = pg.colormap.get(cmap_name, source=None)
        if cmap is None:
            try:
                from matplotlib.cm import get_cmap as _get_cmap
                mpl_cm = _get_cmap(cmap_name)
                positions = np.linspace(0, 1, 256)
                colors = (mpl_cm(positions) * 255).astype(np.uint8)
                cmap = pg.ColorMap(pos=positions, color=colors)
            except Exception:
                cmap = pg.colormap.get("CET-L1")  # built-in fallback
        img.setLookupTable(cmap.getLookupTable(nPts=256))

        # Scale axes to micrometers if pixel size is known
        if pixel_size_um is not None and pixel_size_um > 0:
            ny, nx = data.shape[:2]
            img.setRect(QRectF(0, 0, nx * pixel_size_um, ny * pixel_size_um))
            plot.setLabel("bottom", "µm")
            plot.setLabel("left", "µm")
        else:
            plot.setLabel("bottom", "px")
            plot.setLabel("left", "px")

        plot.addItem(img)

        # Colorbar: handle constant data (min == max)
        cb_min, cb_max = dmin, dmax
        if abs(cb_max - cb_min) < 1e-15:
            cb_min -= 1.0
            cb_max += 1.0

        cbar = pg.ColorBarItem(
            colorMap=cmap,
            values=(cb_min, cb_max),
            interactive=False,
            width=12,
        )
        cbar.setImageItem(img, insert_in=plot)

        vbox.addWidget(gw)
        return container

    def _cleanup_qpi_dialog(self):
        """Safely close the QPI results dialog."""
        dlg = self._qpi_dialog
        self._qpi_dialog = None  # Clear first to prevent re-entrant issues
        if dlg is not None:
            try:
                from shiboken6 import isValid
                if isValid(dlg):
                    # Disconnect the destroyed→_cleanup_3d_window handler before
                    # closing. WA_DeleteOnClose schedules an async delete; without
                    # this disconnect, the stale fire would kill the *next* 3D
                    # window created just after the dialog is replaced.
                    try:
                        dlg.destroyed.disconnect(self._cleanup_3d_window)
                    except (TypeError, RuntimeError):
                        pass
                    dlg.close()
            except Exception:
                pass

    def _show_qpi_window(self, result, qtab) -> None:
        """Open a window showing QPI map results in a proper grid layout."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QWidget
        import pyqtgraph as pg

        # Close previous dialog safely
        self._cleanup_qpi_dialog()

        dlg = QDialog(self)
        dlg.setWindowTitle("QPI Results")
        dlg.resize(1000, 750)
        main_layout = QVBoxLayout(dlg)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Pixel size for axis scaling
        px_um = self._get_effective_pixel_um()

        def _has_data(arr):
            return arr is not None and isinstance(arr, np.ndarray) and arr.size > 0

        # Collect active map panels
        map_panels = []
        if qtab.show_opd_cb.isChecked() and _has_data(result.opd_nm):
            map_panels.append(self._make_qpi_map_widget(result.opd_nm, "OPD (nm)", "viridis", px_um))
        if qtab.show_height_cb.isChecked() and _has_data(result.height_nm):
            map_panels.append(self._make_qpi_map_widget(result.height_nm, "Height (nm)", "viridis", px_um))
        if qtab.show_drymass_cb.isChecked() and _has_data(result.dry_mass_density_pg_um2):
            map_panels.append(self._make_qpi_map_widget(result.dry_mass_density_pg_um2, "Dry mass (pg/µm²)", "inferno", px_um))
        if qtab.show_ri_cb.isChecked() and _has_data(result.ri_map):
            map_panels.append(self._make_qpi_map_widget(result.ri_map, "Refractive Index", "plasma", px_um))

        # Cell mask overlay on OPD
        if _has_data(result.cell_mask) and _has_data(result.opd_nm):
            map_panels.append(self._make_qpi_map_widget(
                result.opd_nm * (result.cell_mask > 0).astype(float),
                "Cell Segmentation (OPD × mask)", "viridis", px_um))

        # Collect plot panels
        plot_panels = []
        if qtab.show_psd_cb.isChecked() and _has_data(result.psd_freq) and _has_data(result.psd_values):
            w = pg.PlotWidget(title="Power Spectral Density")
            w.setLabel("bottom", "Spatial frequency", units="1/m")
            w.setLabel("left", "PSD", units="m³")
            w.setLogMode(x=True, y=True)
            w.showGrid(x=True, y=True, alpha=0.3)
            # Filter out zero/negative values for log scale
            valid = (result.psd_freq > 0) & (result.psd_values > 0)
            if np.any(valid):
                w.plot(result.psd_freq[valid], result.psd_values[valid], pen=pg.mkPen("c", width=2))
            plot_panels.append(w)

        if qtab.show_histogram_cb.isChecked() and _has_data(result.opd_nm):
            w = pg.PlotWidget(title="Phase Histogram")
            w.setLabel("bottom", "OPD (nm)")
            w.setLabel("left", "Count")
            w.showGrid(x=True, y=True, alpha=0.3)
            vals = result.opd_nm.ravel()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                y, x = np.histogram(vals, bins=256)
                w.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True,
                       pen=pg.mkPen("c", width=1.5), brush=(0, 180, 220, 80))
            plot_panels.append(w)

        all_panels = map_panels + plot_panels
        if not all_panels:
            self.status_bar.show_message("QPI: no display panels selected or no data available")
            dlg.deleteLater()
            return

        # Arrange in 2-column grid
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for i, panel in enumerate(all_panels):
            grid.addWidget(panel, i // 2, i % 2)
        main_layout.addWidget(grid_widget)

        # Bottom bar with 3D button
        from PySide6.QtWidgets import QHBoxLayout, QPushButton
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        btn_3d = QPushButton("Show 3D Surface")
        btn_3d.setToolTip("Open interactive 3D surface view of height/OPD data")
        has_surface = (_has_data(result.height_nm) or _has_data(result.opd_nm))
        btn_3d.setEnabled(has_surface)
        btn_3d.clicked.connect(lambda: self._show_3d_surface(result))
        bottom_bar.addWidget(btn_3d)
        main_layout.addLayout(bottom_bar)

        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.destroyed.connect(lambda: setattr(self, '_qpi_dialog', None))
        dlg.destroyed.connect(self._cleanup_3d_window)
        dlg.show()
        self._qpi_dialog = dlg
        self._qpi_last_result = result

        # 3D surface as separate window (auto-open if checkbox is checked)
        if qtab.show_3d_cb.isChecked():
            self._show_3d_surface(result)

    def _cleanup_3d_window(self):
        """Safely close and schedule deletion of the 3D surface window."""
        if self._qpi_3d_window is not None:
            try:
                self._qpi_3d_window.close()
                self._qpi_3d_window.deleteLater()
            except Exception:
                pass
            self._qpi_3d_window = None

    def _show_3d_surface(self, result) -> None:
        """Open (or refresh) the 3D surface window with progressive refinement.

        Reuses the existing GLViewWidget when possible. Creating a fresh
        GLViewWidget while the previous one is still being deleted by Qt caused
        the new window to render pitch-black (OpenGL context torn down mid-init).
        """
        surface_data = result.height_nm if result.height_nm is not None else result.opd_nm
        if surface_data is None or surface_data.size == 0:
            return
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
        except ImportError as e:
            msg = f"3D Surface unavailable — PyOpenGL not installed ({e}). Run: pip install PyOpenGL PyOpenGL-accelerate"
            log.warning(msg)
            self.status_bar.show_message(msg)
            return

        from PySide6.QtCore import QTimer
        from shiboken6 import isValid

        px_um = self._get_effective_pixel_um()
        full_data = surface_data.copy()
        cmap = pg.colormap.get("viridis", source="matplotlib")
        if cmap is None:
            cmap = pg.colormap.get("viridis", source=None)
        if cmap is None:
            cmap = pg.colormap.get("CET-L1")

        full_w_um = full_data.shape[1] * px_um
        full_h_um = full_data.shape[0] * px_um
        xy_extent = max(full_w_um, full_h_um)

        def _build_surface(data, max_dim):
            if data.shape[0] > max_dim or data.shape[1] > max_dim:
                step = max(1, max(data.shape[0], data.shape[1]) // max_dim)
                data = data[::step, ::step]
            ny, nx = data.shape
            eff_px = px_um * (full_data.shape[1] / nx)
            z_range = float(np.ptp(data))
            z_scale = 1.0 if z_range == 0 else xy_extent / (z_range * 3)
            d_min, d_max = float(np.nanmin(data)), float(np.nanmax(data))
            norm = (data - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(data)
            colors = cmap.map(norm.ravel(), mode="float").reshape(ny, nx, 4)
            surface = gl.GLSurfacePlotItem(
                z=data.astype(np.float64),
                colors=colors,
            )
            surface.scale(eff_px, eff_px, z_scale)
            surface.translate(-nx * eff_px / 2, -ny * eff_px / 2, 0)
            return surface

        def _fresh_grid():
            g = gl.GLGridItem()
            g.setSize(full_w_um, full_h_um)
            g.setSpacing(max(1, full_w_um / 10), max(1, full_h_um / 10))
            return g

        w = self._qpi_3d_window
        first_time = w is None or not isValid(w)
        if first_time:
            w = gl.GLViewWidget()
            w.setWindowTitle("QPI — 3D Surface")
            w.resize(700, 550)
            self._qpi_3d_window = w
        else:
            # Reuse the existing widget — remove old items before adding new ones
            for item in list(w.items):
                try:
                    w.removeItem(item)
                except Exception:
                    pass

        current_surface = _build_surface(full_data, max_dim=64)
        grid = _fresh_grid()
        w.addItem(current_surface)
        w.addItem(grid)

        if first_time:
            w.setCameraPosition(distance=xy_extent * 1.5, elevation=30, azimuth=45)
        w.show()
        w.raise_()
        w.activateWindow()

        state = {"current": current_surface, "grid": grid}

        def _refine(max_dim):
            if self._qpi_3d_window is not w or not isValid(w):
                return
            try:
                w.removeItem(state["current"])
                w.removeItem(state["grid"])
            except Exception:
                return
            new_surface = _build_surface(full_data, max_dim=max_dim)
            new_grid = _fresh_grid()
            w.addItem(new_surface)
            w.addItem(new_grid)
            state["current"] = new_surface
            state["grid"] = new_grid

        QTimer.singleShot(150, lambda: _refine(128))
        QTimer.singleShot(500, lambda: _refine(256))

    def _on_qpi_display_changed(self):
        """Re-show QPI window when display checkboxes change."""
        if self._qpi_last_result is None:
            return
        self._cleanup_3d_window()
        self._show_qpi_window(self._qpi_last_result, self.sidebar_tabs.qpi_tab)

    # ─── Camera Live Mode Hooks ───
    def _on_mode_changed(self, mode: str):
        # Clean up analysis state on mode switch
        self._clear_line_profile()
        if hasattr(self, 'toolbar') and self.toolbar.action_line_profile.isChecked():
            self.toolbar.action_line_profile.blockSignals(True)
            self.toolbar.action_line_profile.setChecked(False)
            self.toolbar.action_line_profile.blockSignals(False)

        tabs = self.sidebar_tabs.tabs
        cam_tab = self.sidebar_tabs.camera_tab
        rec_tab = self.sidebar_tabs.record_tab
        if mode == "Live":
            # Show Camera and Record tabs
            if tabs.indexOf(cam_tab) < 0:
                tabs.insertTab(0, cam_tab, "Camera")
            if tabs.indexOf(rec_tab) < 0:
                tabs.addTab(rec_tab, "Record")
            tabs.setCurrentWidget(cam_tab)
            self.toolbar.action_batch.setEnabled(False)
            self.toolbar.action_load.setEnabled(False)
        else:
            self._on_acq_stop()
            self._on_camera_disconnect()
            # Hide Camera and Record tabs — irrelevant in File mode
            for tab in (cam_tab, rec_tab):
                idx = tabs.indexOf(tab)
                if idx >= 0:
                    tabs.removeTab(idx)
            tabs.setCurrentWidget(self.sidebar_tabs.recon_tab)
            self.toolbar.action_batch.setEnabled(True)
            self.toolbar.action_load.setEnabled(True)

    def _on_camera_connect(self):
        camtab = self.sidebar_tabs.camera_tab
        cam_name = camtab.cam_combo.currentData()
        if not cam_name:
            return
            
        from core.camera import NICamera, CameraError
        try:
            self._camera = NICamera(cam_name)
            self._camera.open()
            self._acq_worker = AcquisitionWorker(self._camera)
            self._acq_worker.frame_ready.connect(self._on_frame_ready)
            self._acq_worker.fps_updated.connect(lambda fps: camtab.stats_label.setText(f"Acquisition FPS: {fps:.1f}"))
            self._acq_worker.error_occurred.connect(lambda e: self.status_bar.show_message(f"Acq error: {e}"))
            
            camtab.status_label.setText(f"Status: Connected ({self._camera.resolution[0]}x{self._camera.resolution[1]}, {self._camera.dtype})")
            camtab.connect_btn.setEnabled(False)
            camtab.disconnect_btn.setEnabled(True)
            camtab.start_acq_btn.setEnabled(True)
        except Exception as e:
            self.status_bar.show_message(f"Camera connect failed: {e}")

    def _on_camera_disconnect(self):
        if self._acq_worker and self._acq_worker.running:
            self._on_acq_stop()
        if self._camera:
            self._camera.close()
            self._camera = None
            
        camtab = self.sidebar_tabs.camera_tab
        camtab.status_label.setText("Status: Disconnected")
        camtab.connect_btn.setEnabled(True)
        camtab.disconnect_btn.setEnabled(False)
        camtab.start_acq_btn.setEnabled(False)

    def _on_acq_start(self):
        if not self._acq_worker or not self._camera: return
        camtab = self.sidebar_tabs.camera_tab
        
        try:
            mode = "Triggered" if camtab.mode_combo.currentText() == "Triggered" else "continuous"
            self._camera.configure(
                exposure_us=camtab.exposure_spin.value(),
                gain_db=camtab.gain_spin.value(),
                bit_depth=camtab.bitdepth_combo.currentText().lower()
            )
            self._camera.start_acquisition(mode=mode)
            
            self._acq_worker.target_fps = camtab.fps_spin.value() if camtab.limit_fps_cb.isChecked() else 0
            self._acq_worker.start()
            
            camtab.start_acq_btn.setEnabled(False)
            camtab.stop_acq_btn.setEnabled(True)
        except Exception as e:
            self.status_bar.show_message(f"Start acq failed: {e}")

    def _on_acq_stop(self):
        if self._acq_worker:
            self._acq_worker.stop()
            self._acq_worker.wait()
        if self._camera:
            self._camera.stop_acquisition()
            
        camtab = self.sidebar_tabs.camera_tab
        camtab.start_acq_btn.setEnabled(True)
        camtab.stop_acq_btn.setEnabled(False)

    def _on_frame_ready(self, frame: np.ndarray, frame_num: int):
        self._loaded_array = frame
        self._frame_counter = frame_num
        
        # 1. Update Input Panel immediately
        img = self._as_field_2d(frame)
        self.panel_input.set_image(img, autoLevels=False)
        
        camtab = self.sidebar_tabs.camera_tab
        
        # 2. Adaptive Focus
        ftab = self.sidebar_tabs.focus_tab
        if ftab.adapt_enable_cb.isChecked() and (frame_num % int(ftab.adapt_n_interval.value()) == 0):
            self._trigger_adaptive_focus()
            
        # 3. Reconstruction logic
        if camtab.auto_recon_cb.isChecked():
            skip_n = camtab.recon_skip_spin.value()
            if skip_n > 0 and frame_num % skip_n == 0:
                self._trigger_reconstruction()

    def _prepare_af_field(self):
        """Shared preprocessing for autofocus, benchmark, diagnostic."""
        from core.reconstruction import ReconstructionParams, ReconstructionMethod
        from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug

        rtab = self.sidebar_tabs.recon_tab
        ftab = self.sidebar_tabs.focus_tab
        ptab = self.sidebar_tabs.process_tab

        method = ReconstructionMethod(rtab.method_combo.currentData())
        wl = float(rtab.wavelength_nm.value()) * 1e-9
        px = float(rtab.pixel_um.value()) * 1e-6
        if not rtab.pixel_is_effective_cb.isChecked():
            mag = float(rtab.magnification.value())
            px = px / (mag if mag > 0 else 1.0)

        params = ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=0.0, n=1.0)

        zmin = float(ftab.zscan_min_mm.value()) * 1e-3
        zmax = float(ftab.zscan_max_mm.value()) * 1e-3
        steps = int(ftab.zscan_steps.value())

        img = self._as_field_2d(self._loaded_array)
        if ptab.subtract_mean_cb.isChecked():
            img = img - float(np.mean(img))
        if ptab.hann_cb.isChecked():
            wy = np.hanning(img.shape[0]).astype(np.float32)
            wx = np.hanning(img.shape[1]).astype(np.float32)
            img = img * (wy[:, None] * wx[None, :])
        if np.max(np.abs(img)) > 0:
            img = img / float(np.max(np.abs(img)))

        offaxis_params = OffAxisParams(
            radius=int(rtab.mask_radius.value()),
            apodization=ptab.mask_apod_combo.currentData(),
            rolloff=ptab.mask_rolloff.value(),
        )
        fc, _, _, _ = extract_complex_field_offaxis_debug(img, offaxis_params)

        return fc, params, method, zmin, zmax, steps

    def _autofocus(self) -> None:
        """Initiates a one-shot autofocus scan in a background thread."""
        if self._loaded_array is None:
            return

        ftab = self.sidebar_tabs.focus_tab
        ftab.autofocus_btn.setEnabled(False)

        from core.autofocus import FocusMetric
        from gui.workers.autofocus_worker import AutofocusWorker

        try:
            fc, params, method, zmin, zmax, steps = self._prepare_af_field()
            metric = FocusMetric(ftab.metric_combo.currentData())

            is_auto = ftab.step_init_mm.value() <= ftab.step_init_mm.minimum()
            step_init = None if is_auto else ftab.step_init_mm.value() * 1e-3

            algo = ftab.get_effective_algorithm()
            is_adaptive = algo.startswith("Adaptive Step")
            use_ad = ftab.is_adaptive_distance_enabled()

            # For adaptive algorithms, auto-calculate evaluation budget
            if is_adaptive:
                range_m = abs(zmax - zmin)
                si = step_init if step_init else range_m / 20.0
                steps = max(100, min(500, int(range_m / si * 3)))

            # Increase budget when Distance + refinement are both active
            if use_ad:
                steps = max(steps, 80)

            # ROI bounds — pass to worker so ALL algorithms use ROI-based metric
            roi_bounds = None
            if ftab.roi_tracker_group.isChecked() and self._phase_tracker.active:
                roi_bounds = self._phase_tracker.get_roi_bounds()

            # Pass pre-propagation reference field for phase-based focus metrics
            ref_fc = getattr(self, '_reference_fc', None)

            worker = AutofocusWorker(self)
            worker.configure(
                field=fc, base_params=params, method=method, metric=metric,
                algo=algo,
                z_min_m=zmin, z_max_m=zmax, steps=steps,
                auto_select=ftab.auto_metric_cb.isChecked(),
                step_init=step_init,
                grow_factor=ftab.grow_factor.value(),
                shrink_factor=ftab.shrink_factor.value(),
                refine_levels=ftab.refine_levels.value(),
                refine_divisions=ftab.refine_divisions.value(),
                use_adaptive_distance=use_ad,
                ad_initial_range_m=ftab.ad_initial_range_mm.value() * 1e-3,
                ad_max_range_m=ftab.ad_max_range_mm.value() * 1e-3,
                ad_expand_factor=ftab.ad_expand_factor.value(),
                ad_signal_threshold=ftab.ad_signal_threshold.value(),
                roi_bounds=roi_bounds,
                ref_field=ref_fc,
            )

            worker.progress.connect(self._af_overlay.set_status)
            worker.progress_pct.connect(self._af_overlay.update_progress)
            worker.metric_selected.connect(self._on_af_metric_selected)
            worker.metric_auto_fallback.connect(self._on_af_metric_auto_fallback)
            worker.finished.connect(self._on_af_worker_finished)
            worker.error.connect(self._on_af_worker_error)
            worker.cancelled.connect(self._on_af_worker_cancelled)
            worker.finished.connect(worker.deleteLater)
            worker.error.connect(worker.deleteLater)
            worker.cancelled.connect(worker.deleteLater)

            self._af_worker = worker
            self._af_overlay.show_overlay("Auto-focusing")
            worker.start()

        except Exception as e:
            self.status_bar.show_message(f"Autofocus failed: {e}")
            ftab.autofocus_btn.setEnabled(True)

    def _on_af_cancel(self) -> None:
        if self._af_worker and self._af_worker.isRunning():
            self._af_worker.cancel()

    def _on_af_metric_selected(self, metric_value: str) -> None:
        ftab = self.sidebar_tabs.focus_tab
        for i in range(ftab.metric_combo.count()):
            if ftab.metric_combo.itemData(i) == metric_value:
                ftab.metric_combo.setCurrentIndex(i)
                break

    def _on_af_metric_auto_fallback(self, from_metric: str, to_metric: str) -> None:
        """Worker detected a low-contrast hologram and switched metric pre-flight.

        The user chose (or auto-select picked) ENTROPY, but the hologram lacks
        enough wrapped-phase structure for ENTROPY to converge reliably. The
        worker silently switched to PHASE_VARIANCE for this run. We surface that
        clearly so the lab operator understands why the metric in the report
        differs from what they selected.
        """
        msg = (
            f"Autofocus: {from_metric} unsuitable for low-contrast hologram — "
            f"switched to {to_metric}"
        )
        self.status_bar.show_message(msg, timeout=12000)
        log.info("Autofocus metric auto-fallback: %s → %s", from_metric, to_metric)
        # Reflect the actual metric being used in the focus-tab combo so the
        # user isn't misled about what ran.
        self._on_af_metric_selected(to_metric)

    def _on_af_worker_finished(self, best_z, z_arr, scores, metric_val, elapsed, evals) -> None:
        self._af_overlay.hide_overlay()
        self._af_worker = None
        self._autofocus_table = np.column_stack((z_arr, scores))
        self._autofocus_best_z_m = best_z

        self.status_bar.show_message(
            f"Auto-focus: {elapsed:.2f}s | Z={best_z*1e3:.4f} mm | {metric_val} | {evals} evals"
        )

        # Snapshot AF result (for report) + audit record
        try:
            ftab = self.sidebar_tabs.focus_tab
            algo = ftab.get_effective_algorithm()
        except Exception:
            algo = None
        af_summary = {
            "z_found_m": float(best_z),
            "metric": str(metric_val),
            "algorithm": algo,
            "evaluations": int(evals),
            "elapsed_s": float(elapsed),
        }
        self._last_af_result = af_summary
        self._audit(
            action="autofocus",
            params={"metric": str(metric_val), "algorithm": algo},
            result_summary=af_summary,
        )

        self.sidebar_tabs.recon_tab.z_mm.setValue(best_z * 1e3)
        self.sidebar_tabs.focus_tab.autofocus_btn.setEnabled(True)
        # Save the z-range + metric the user just found useful.
        self._persist_current_settings()
        self._trigger_reconstruction()

    def _on_af_worker_error(self, msg: str) -> None:
        self._af_overlay.hide_overlay()
        self._af_worker = None
        self.status_bar.show_message(f"Autofocus failed: {msg}")
        self.sidebar_tabs.focus_tab.autofocus_btn.setEnabled(True)

    def _on_af_worker_cancelled(self) -> None:
        self._af_overlay.hide_overlay()
        self._af_worker = None
        self.status_bar.show_message("Autofocus cancelled")
        self.sidebar_tabs.focus_tab.autofocus_btn.setEnabled(True)

    # ─── ROI Tracker ───

    def _disconnect_roi_signals(self) -> None:
        """Safely disconnect all ROI-related mouse click signals."""
        vb = self.panel_amp.get_view()
        scene = vb.scene()
        for handler in (self._on_object_click, self._on_roi_click):
            try:
                scene.sigMouseClicked.disconnect(handler)
            except (TypeError, RuntimeError):
                pass

    def _reset_roi_state(self) -> None:
        """Clear all ROI state: overlays, tracker, signals, mode flags."""
        # Disconnect all click handlers
        self._disconnect_roi_signals()
        self._roi_select_mode = False
        self._object_select_mode = False

        # Clear detected object overlays from amplitude panel
        vb = self.panel_amp.get_view()
        for item in self._detected_overlays:
            try:
                vb.removeItem(item)
            except Exception:
                pass
        self._detected_overlays.clear()
        self._detected_objects.clear()

        # Clear ROI rectangle overlay from amplitude panel
        if self._roi_rect_item is not None:
            try:
                vb.removeItem(self._roi_rect_item)
            except Exception:
                pass
            self._roi_rect_item = None

        # Reset tracker
        self._phase_tracker.reset()

        # Reset UI
        ftab = self.sidebar_tabs.focus_tab
        ftab.roi_select_btn.blockSignals(True)
        ftab.roi_select_btn.setChecked(False)
        ftab.roi_select_btn.blockSignals(False)
        ftab.roi_clear_btn.setEnabled(False)
        ftab.roi_status_label.setText("No ROI selected")
        ftab.roi_status_label.setStyleSheet("color: gray; font-style: italic;")

    def _on_detect_objects(self) -> None:
        """Run automatic object detection on the phase image (unwrapped phase)."""
        if self._phase_unwrapped is None:
            self.status_bar.show_message("No phase data — run reconstruction first")
            return

        ftab = self.sidebar_tabs.focus_tab

        from core.phase_tracker import detect_objects
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt

        # Show busy state so user knows detection is running
        ftab.detect_objects_btn.setEnabled(False)
        original_text = ftab.detect_objects_btn.text()
        ftab.detect_objects_btn.setText("Detecting...")
        self.status_bar.show_message("Running phase-based object detection...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        method = ftab.detect_method_combo.currentData()
        sensitivity = ftab.detect_sensitivity.value()
        min_area = ftab.detect_min_area.value()

        try:
            # Use unwrapped phase for detection — phase signatures reveal
            # cells, microspheres, and other phase objects more reliably
            phase_img = self._phase_unwrapped
            objects = detect_objects(
                phase_img, min_area=min_area,
                method=method, sensitivity=sensitivity,
            )
        finally:
            QApplication.restoreOverrideCursor()
            ftab.detect_objects_btn.setText(original_text)
            ftab.detect_objects_btn.setEnabled(True)

        # Clean previous state but keep group enabled
        self._disconnect_roi_signals()
        self._object_select_mode = False
        for item in self._detected_overlays:
            try:
                self.panel_amp.get_view().removeItem(item)
            except Exception:
                pass
        self._detected_overlays.clear()
        if self._roi_rect_item is not None:
            try:
                self.panel_amp.get_view().removeItem(self._roi_rect_item)
            except Exception:
                pass
            self._roi_rect_item = None
        self._phase_tracker.reset()

        # Limit to top 50 objects by area to prevent UI freeze
        MAX_DISPLAY = 50
        if len(objects) > MAX_DISPLAY:
            objects = sorted(objects, key=lambda o: o.area, reverse=True)[:MAX_DISPLAY]

        self._detected_objects = objects

        if not objects:
            ftab.roi_status_label.setText("No objects detected — adjust sensitivity")
            ftab.roi_status_label.setStyleSheet("color: orange;")
            ftab.roi_clear_btn.setEnabled(False)
            self.status_bar.show_message("No objects detected in phase image")
            return

        # Detection from phase, overlay on amplitude (autofocus mask)
        self._draw_detected_overlays()

        ftab.roi_status_label.setText(
            f"{len(objects)} objects — click one on amplitude to select"
        )
        ftab.roi_status_label.setStyleSheet("color: #00bfff; font-weight: bold;")
        ftab.roi_clear_btn.setEnabled(True)
        self.status_bar.show_message(
            f"Detected {len(objects)} phase objects. Click on Amplitude panel to select as autofocus mask."
        )

        # Enter object selection mode — click on amplitude panel
        self._object_select_mode = True
        self.panel_amp.get_view().scene().sigMouseClicked.connect(self._on_object_click)

    def _draw_detected_overlays(self) -> None:
        """Draw numbered bounding boxes for detected objects on amplitude panel (autofocus mask)."""
        vb = self.panel_amp.get_view()
        colors = ['#00ff00', '#ff6600', '#ff00ff', '#ffff00',
                  '#00ffff', '#ff3333', '#66ff66', '#ff66ff']

        for obj in self._detected_objects:
            bb = obj.bbox
            color = colors[(obj.label - 1) % len(colors)]

            rect = pg.QtWidgets.QGraphicsRectItem(
                bb.x0, bb.y0, bb.width, bb.height,
            )
            rect.setPen(pg.mkPen(color, width=2))
            rect.setBrush(pg.mkBrush(None))
            rect.setZValue(100)
            vb.addItem(rect)
            self._detected_overlays.append(rect)

            label = pg.TextItem(
                text=str(obj.label), color=color,
                anchor=(0.5, 1.0),
            )
            label.setPos(obj.cx, bb.y0 - 2)
            label.setZValue(101)
            vb.addItem(label)
            self._detected_overlays.append(label)

    def _on_object_click(self, event) -> None:
        """Handle click on amplitude panel to select the nearest detected object."""
        if not self._object_select_mode or not self._detected_objects:
            return

        vb = self.panel_amp.get_view()
        pos = vb.mapSceneToView(event.scenePos())
        cx, cy = float(pos.x()), float(pos.y())

        # Find nearest object
        best_obj = None
        best_dist = float('inf')
        for obj in self._detected_objects:
            dist = (obj.cx - cx) ** 2 + (obj.cy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_obj = obj

        if best_obj is None:
            return

        # Use the object's bounding box to determine ROI size
        roi_size = max(best_obj.bbox.width, best_obj.bbox.height)
        roi_size = max(16, min(512, ((roi_size + 15) // 16) * 16))

        ftab = self.sidebar_tabs.focus_tab
        ftab.roi_size_spin.setValue(roi_size)
        self._phase_tracker.roi_size = roi_size

        # ROI selected from phase detection, applied on amplitude for autofocus masking
        amp = np.abs(self._recon_complex)
        ok = self._phase_tracker.select_roi(amp, best_obj.cx, best_obj.cy)

        # Exit object selection mode first
        self._object_select_mode = False
        self._disconnect_roi_signals()

        if ok:
            # Clear detection overlays, show ROI rect
            for item in self._detected_overlays:
                try:
                    vb.removeItem(item)
                except Exception:
                    pass
            self._detected_overlays.clear()

            self._draw_roi_overlay()
            ftab.roi_status_label.setText(
                f"#{best_obj.label}  ({int(best_obj.cx)},{int(best_obj.cy)})  "
                f"{roi_size}×{roi_size}px  area={best_obj.area}px"
            )
            ftab.roi_status_label.setStyleSheet("color: green; font-weight: bold;")
            ftab.roi_clear_btn.setEnabled(True)
            self.status_bar.show_message(
                f"Phase object #{best_obj.label} selected — ROI mask ready for autofocus"
            )
        else:
            ftab.roi_status_label.setText("Selection failed")
            ftab.roi_status_label.setStyleSheet("color: red;")

    def _on_roi_select_toggled(self, checked: bool) -> None:
        """Enter/exit manual ROI selection mode on the amplitude panel."""
        # Always disconnect first to prevent stacking
        self._disconnect_roi_signals()
        self._roi_select_mode = checked
        self._object_select_mode = False

        if checked:
            # Clear any detection overlays
            for item in self._detected_overlays:
                try:
                    self.panel_amp.get_view().removeItem(item)
                except Exception:
                    pass
            self._detected_overlays.clear()

            self.status_bar.show_message("Click on the Amplitude image to place ROI (autofocus mask)")
            self.panel_amp.get_view().scene().sigMouseClicked.connect(self._on_roi_click)

    def _on_roi_click(self, event) -> None:
        """Handle mouse click on the amplitude panel to set ROI center (manual mode)."""
        if not self._roi_select_mode:
            return

        vb = self.panel_amp.get_view()
        pos = vb.mapSceneToView(event.scenePos())
        cx, cy = float(pos.x()), float(pos.y())

        ftab = self.sidebar_tabs.focus_tab
        roi_size = ftab.roi_size_spin.value()
        self._phase_tracker.roi_size = roi_size

        ok = False
        if self._recon_complex is not None:
            amp = np.abs(self._recon_complex)
            ok = self._phase_tracker.select_roi(amp, cx, cy)

        if ok:
            self._draw_roi_overlay()
            ftab.roi_status_label.setText(
                f"ROI: ({int(cx)},{int(cy)})  {roi_size}×{roi_size}px"
            )
            ftab.roi_status_label.setStyleSheet("color: green; font-weight: bold;")
            ftab.roi_clear_btn.setEnabled(True)
            self.status_bar.show_message(f"Phase ROI selected at ({int(cx)}, {int(cy)}) — autofocus mask active")
        else:
            ftab.roi_status_label.setText("Selection failed — click inside phase image")
            ftab.roi_status_label.setStyleSheet("color: red;")

        # Uncheck the button (triggers toggled→disconnect)
        ftab.roi_select_btn.setChecked(False)

    def _draw_roi_overlay(self) -> None:
        """Draw/update a lightweight ROI rectangle on the amplitude panel (autofocus mask)."""
        if not self._phase_tracker.active:
            return

        bounds = self._phase_tracker.get_roi_bounds()
        vb = self.panel_amp.get_view()

        # Remove old rect
        if self._roi_rect_item is not None:
            try:
                vb.removeItem(self._roi_rect_item)
            except Exception:
                pass

        # Lightweight QGraphicsRectItem instead of heavy RectROI
        rect = pg.QtWidgets.QGraphicsRectItem(
            bounds.x0, bounds.y0, bounds.width, bounds.height,
        )
        rect.setPen(pg.mkPen('c', width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
        rect.setBrush(pg.mkBrush(None))
        rect.setZValue(99)
        rect.setAcceptedMouseButtons(pg.QtCore.Qt.MouseButton.NoButton)
        vb.addItem(rect)
        self._roi_rect_item = rect

    def _update_roi_tracking(self, amp_image: np.ndarray) -> None:
        """Update ROI tracker with amplitude data (called after reconstruction)."""
        ftab = self.sidebar_tabs.focus_tab
        if not ftab.roi_tracker_group.isChecked():
            return
        if not self._phase_tracker.active:
            return
        if not ftab.roi_track_live_cb.isChecked():
            return

        dx, dy, conf = self._phase_tracker.track(amp_image)
        if conf > 0.15:
            self._draw_roi_overlay()
            ftab.roi_status_label.setText(
                f"Tracking: Δ=({dx:.1f},{dy:.1f}) conf={conf:.2f}"
            )

    def _run_af_benchmark(self) -> None:
        """Runs autofocus benchmark across all algorithm x metric combinations."""
        if self._loaded_array is None:
            return

        self.sidebar_tabs.focus_tab.benchmark_btn.setEnabled(False)
        self.status_bar.show_message("Running AF benchmark...")

        from core.autofocus import autofocus_benchmark

        try:
            fc, params, method, zmin, zmax, steps = self._prepare_af_field()
            result = autofocus_benchmark(fc, params, method, zmin, zmax, steps=steps)

            # Format results as a message box table
            lines = []
            if result.reference_z_m is not None:
                lines.append(f"Reference Z = {result.reference_z_m*1e3:.4f} mm "
                             f"(metric: {result.reference_metric}, {result.reference_elapsed_s:.2f}s)")
                lines.append("")
            lines.append(
                f"{'Algorithm':<30} {'Metric':<20} {'Z (mm)':<12} {'Err(µm)':<10} {'Time(s)':<9} {'Evals':<6} {'ms/eval':<8} {'DS':<4}"
            )
            lines.append("-" * 99)
            for e in result.entries:
                ds_str = f"×{e.downsampled}" if e.downsampled > 1 else ""
                lines.append(
                    f"{e.algorithm:<30} {e.metric.value:<20} {e.best_z_m*1e3:<12.4f} "
                    f"{e.error_um:<10.1f} {e.elapsed_s:<9.3f} {e.evaluations:<6} "
                    f"{e.per_eval_ms:<8.1f} {ds_str:<4}"
                )

            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setWindowTitle("Autofocus Benchmark Results")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText(f"Benchmark complete ({len(result.entries)} runs)")
            msg.setDetailedText("\n".join(lines))
            msg.exec()

            self.status_bar.show_message("AF Benchmark complete")
        except Exception as e:
            self.status_bar.show_message(f"AF Benchmark failed: {e}")
        finally:
            self.sidebar_tabs.focus_tab.benchmark_btn.setEnabled(True)

    def _run_af_diagnostic(self) -> None:
        """Scans all metrics and displays a diagnostic landscape plot."""
        if self._loaded_array is None:
            return

        self.sidebar_tabs.focus_tab.diagnostic_btn.setEnabled(False)
        self.status_bar.show_message("Scanning metric landscape...")

        from core.autofocus import scan_metric_landscape, FocusMetric

        try:
            fc, params, method, zmin, zmax, steps = self._prepare_af_field()
            result = scan_metric_landscape(fc, params, method, zmin, zmax, n_steps=steps)

            # Build a matplotlib dialog
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from PySide6.QtWidgets import QDialog, QVBoxLayout

            n_metrics = len(result.raw_scores)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 2.5 * n_metrics), sharex=True)
            if n_metrics == 1:
                axes = [axes]

            z_mm = result.z_values * 1e3
            for ax, fm in zip(axes, result.raw_scores.keys()):
                raw = result.raw_scores[fm]
                smooth = result.smoothed_scores[fm]
                vmin, vmax = raw.min(), raw.max()
                raw_norm = (raw - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-15 else raw * 0

                ax.plot(z_mm, raw_norm, 'b-', alpha=0.4, linewidth=0.8, label='raw')
                ax.plot(z_mm, smooth, 'r-', linewidth=2, label='smoothed')
                pk_z = result.peak_z[fm] * 1e3
                ax.axvline(pk_z, color='green', linestyle='--', alpha=0.7)
                ax.set_ylabel(fm.value, fontsize=9, fontweight='bold')
                ax.legend(loc='upper right', fontsize=7)

                n_pk = result.n_peaks[fm]
                tag = "UNIMODAL" if n_pk <= 1 else f"{n_pk} peaks"
                color = 'green' if n_pk <= 1 else 'red'
                ax.text(0.02, 0.85, tag, transform=ax.transAxes, fontsize=9,
                        fontweight='bold', color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel('z (mm)', fontsize=10)
            fig.suptitle('Focus Metric Landscape', fontsize=12)
            fig.tight_layout()

            dlg = QDialog(self)
            dlg.setWindowTitle("Metric Landscape Diagnostic")
            dlg.resize(900, 600)
            lay = QVBoxLayout(dlg)
            canvas = FigureCanvasQTAgg(fig)
            lay.addWidget(canvas)
            dlg.show()
            dlg.raise_()

            self.status_bar.show_message("Metric landscape scan complete")
        except Exception as e:
            self.status_bar.show_message(f"Diagnostic failed: {e}")
        finally:
            self.sidebar_tabs.focus_tab.diagnostic_btn.setEnabled(True)

    def _trigger_adaptive_focus(self) -> None:
        if self._loaded_array is None or self._adaptive_worker.isRunning():
            self._finish_batch_step_and_continue()
            return

        from core.autofocus import AdaptiveFocusState, FocusMetric

        fc, params, method, _, _, _ = self._prepare_af_field()

        rtab = self.sidebar_tabs.recon_tab
        ftab = self.sidebar_tabs.focus_tab

        z_m_curr = float(rtab.z_mm.value()) * 1e-3
        search_range_m = float(ftab.adapt_range_mm.value()) * 1e-3
        steps = int(ftab.adapt_steps.value())
        metric = FocusMetric(ftab.metric_combo.currentData())

        if self._adaptive_state is None:
            self._adaptive_state = AdaptiveFocusState(
                current_z_m=z_m_curr,
                search_range_m=search_range_m,
                steps=steps,
                metric=metric,
                algorithm=ftab.get_effective_algorithm()
            )
        else:
            self._adaptive_state.current_z_m = z_m_curr
            self._adaptive_state.search_range_m = search_range_m
            self._adaptive_state.steps = steps
            self._adaptive_state.metric = metric
            self._adaptive_state.algorithm = ftab.get_effective_algorithm()

        self._adaptive_worker.setup(self._adaptive_state, fc, params, method)
        self.status_bar.show_message("Adaptive Autofocus running...")
        self._adaptive_worker.start()

    def _finish_batch_step_and_continue(self) -> None:
        """Placeholder for batch pipeline continuation after adaptive focus."""
        pass

    def _on_adaptive_focus_done(self, best_z_m: float) -> None:
        self.sidebar_tabs.recon_tab.z_mm.setValue(best_z_m * 1e3)
        self.status_bar.show_message(f"Adaptive Tracking Updated: Z={best_z_m*1e3:.4f}mm")
        
        # Resume the pipeline
        try:
            self._trigger_reconstruction()
        except Exception:
            pass
            
        if self._batch_running:
            self._finish_batch_step_and_continue()

    def _export_to_root(self, out_root) -> None:
        """Saves current phase, amplitude, and complex outputs to designated folder layout."""
        if out_root is None or self._loaded_path is None:
            return
        stem = self._loaded_path.stem
        method = self.sidebar_tabs.recon_tab.method_combo.currentText().lower()
        z_mm = float(self.sidebar_tabs.recon_tab.z_mm.value())
        radius = int(self.sidebar_tabs.recon_tab.mask_radius.value())
        
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{stem}_{method}_z{z_mm:.6g}mm_r{radius}px_{ts}"
        
        from pathlib import Path
        run_dir = Path(out_root) / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if self._recon_complex is not None:
            rc = np.asarray(self._recon_complex)
            np.save(run_dir / f"{stem}_Z_{z_mm:.4f}_recon_complex.npy", rc)
            np.save(run_dir / f"{stem}_Z_{z_mm:.4f}_amplitude.npy", np.abs(rc))
            np.save(run_dir / f"{stem}_Z_{z_mm:.4f}_phase.npy", np.angle(rc))

    def _select_video_output(self):
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path
        rectab = self.sidebar_tabs.record_tab
        path = QFileDialog.getExistingDirectory(
            self, "Output Directory", self._last_folder_default()
        )
        if path:
            rectab.out_dir_lbl.setText(str(Path(path).name))
            self._vid_output_path = Path(path)
            self._update_io_history(last_folder=str(Path(path)))

    def _on_record_action(self, checked: bool = False):
        rectab = self.sidebar_tabs.record_tab
        state = rectab.get_state()
        mode = state["mode"]
        
        if mode == "Snapshot":
            self._take_snapshot(state)
            return
            
        if checked:
            if not hasattr(self, "_vid_output_path"):
                self.status_bar.show_message("Please select an output directory first.")
                rectab.action_btn.setChecked(False)
                return
                
            self._recording_worker.setup(
                out_dir=self._vid_output_path,
                template=state["template"],
                fmt=state["format"],
                channel=state["channel"],
                mode=mode,
                vid_max_frames=state["vid_max_frames"],
                tl_interval=state["tl_interval"],
                tl_duration=state["tl_duration"],
                sch_start_delay=state["sch_start_delay"]
            )
            self._recording_worker.start()
            rectab.action_btn.setText(f"Stop {mode}")
        else:
            self._recording_worker.stop_recording()
            rectab.action_btn.setText(f"Start/Schedule {mode}")

    def _take_snapshot(self, state: dict):
        from PySide6.QtWidgets import QFileDialog
        folder = self._last_folder_default()
        default = str(Path(folder) / "snapshot.png") if folder else "snapshot.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", default, "Images (*.png *.tiff *.bmp)"
        )
        if not path:
            return
        self._update_io_history(last_folder=str(Path(path).parent))
            
        pixmap = self.image_grid.grab()
        pixmap.save(path)
        self.status_bar.show_message(f"Snapshot grid saved to {path}")

    # ─── Tools ────────────────────────────────────────────────────────────

    def _on_crop_toggled(self, enabled: bool) -> None:
        """Toggle a draggable crop ROI rectangle on the amplitude panel."""
        if enabled:
            self._clear_export_crop_roi()
            panel = self.panel_amp
            vb = panel.get_view()
            img_item = panel.view.getImageItem()
            if img_item is None or img_item.image is None:
                self.status_bar.show_message("Load an image first")
                self.toolbar.action_crop.setChecked(False)
                return
            h, w = img_item.image.shape[:2]
            roi = pg.RectROI(
                [w * 0.25, h * 0.25], [w * 0.5, h * 0.5],
                pen=pg.mkPen("c", width=2),
                hoverPen=pg.mkPen("y", width=2),
            )
            vb.addItem(roi)
            self._export_crop_roi = roi
            self._export_crop_source_view = panel
            self._export_crop_enabled = True
            self.status_bar.show_message("Crop ROI active — drag to select region")
        else:
            self._clear_export_crop_roi()
            self._export_crop_enabled = False
            self.status_bar.show_message("Crop ROI removed")

    def _on_scalebar_toggled(self, enabled: bool) -> None:
        """Toggle a physical scale bar drawn in image coordinates on recon panels."""
        from PySide6.QtWidgets import QGraphicsRectItem
        from PySide6.QtGui import QBrush, QColor, QPen
        from PySide6.QtCore import QRectF

        if enabled:
            rtab = self.sidebar_tabs.recon_tab
            px_um = float(rtab.pixel_um.value())
            if not rtab.pixel_is_effective_cb.isChecked():
                mag = float(rtab.magnification.value())
                if mag > 0:
                    px_um = px_um / mag

            custom_um = self.toolbar.scalebar_um.value()
            bar_um = custom_um if custom_um > 0 else _nice_scalebar_length(px_um)
            bar_px = bar_um / px_um
            bar_h = max(4, bar_px * 0.06)  # bar thickness
            margin = 12  # px from edge

            label_str = f"{bar_um:.0f} µm" if bar_um >= 1 else f"{bar_um*1000:.0f} nm"

            for panel in [self.panel_amp, self.panel_phase, self.panel_spectrum]:
                try:
                    vb = panel.get_view()
                    img_item = panel.view.getImageItem() if hasattr(panel, 'view') else None
                    if img_item is None:
                        img_item = panel.image_panel.view.getImageItem()
                    if img_item is None or img_item.image is None:
                        continue
                    h, w = img_item.image.shape[:2]

                    # Bar position: bottom-right in image coords
                    x0 = w - margin - bar_px
                    y0 = h - margin - bar_h

                    # Background semi-transparent rect (slightly larger)
                    bg_pad = 4
                    bg = QGraphicsRectItem(QRectF(
                        x0 - bg_pad, y0 - bar_h - bg_pad,
                        bar_px + 2 * bg_pad, bar_h * 2 + 2 * bg_pad,
                    ))
                    bg.setBrush(QBrush(QColor(0, 0, 0, 140)))
                    bg.setPen(QPen(Qt.PenStyle.NoPen))
                    vb.addItem(bg)

                    # White bar
                    bar = QGraphicsRectItem(QRectF(x0, y0, bar_px, bar_h))
                    bar.setBrush(QBrush(QColor(255, 255, 255)))
                    bar.setPen(QPen(Qt.PenStyle.NoPen))
                    vb.addItem(bar)

                    # Label
                    txt = pg.TextItem(label_str, color="w", anchor=(0.5, 1))
                    txt.setFont(pg.QtGui.QFont("Arial", max(8, int(bar_h * 1.5))))
                    txt.setPos(x0 + bar_px / 2, y0 - 2)
                    vb.addItem(txt)

                    self._scalebar_items.append((panel, bg))
                    self._scalebar_items.append((panel, bar))
                    self._scalebar_items.append((panel, txt))
                except Exception:
                    pass
            self.status_bar.show_message(f"Scale bar: {label_str} ({bar_px:.0f} px)")
        else:
            for panel, item in self._scalebar_items:
                try:
                    panel.get_view().removeItem(item)
                except Exception:
                    pass
            self._scalebar_items.clear()
            self.status_bar.show_message("Scale bar removed")

    def _on_crosshair_toggled(self, enabled: bool) -> None:
        """Toggle crosshair lines + coordinate label on all panels."""
        if enabled:
            pen = pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine)
            for panel in [self.panel_input, self.panel_amp, self.panel_phase, self.panel_spectrum]:
                try:
                    vb = panel.get_view()
                    vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
                    hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
                    label = pg.TextItem("", anchor=(0, 1), color="y")
                    label.setFont(pg.QtGui.QFont("Menlo", 9))
                    vb.addItem(vline, ignoreBounds=True)
                    vb.addItem(hline, ignoreBounds=True)
                    vb.addItem(label, ignoreBounds=True)
                    self._crosshair_items.append((panel, vline, hline, label))

                    proxy = pg.SignalProxy(
                        vb.scene().sigMouseMoved,
                        rateLimit=30,
                        slot=lambda pts, p=panel, vl=vline, hl=hline, lb=label: self._update_crosshair(pts, p, vl, hl, lb),
                    )
                    self._crosshair_proxies.append(proxy)
                except Exception:
                    pass
            self.status_bar.show_message("Crosshair active")
        else:
            for panel, vl, hl, lb in self._crosshair_items:
                try:
                    vb = panel.get_view()
                    vb.removeItem(vl)
                    vb.removeItem(hl)
                    vb.removeItem(lb)
                except Exception:
                    pass
            self._crosshair_items.clear()
            self._crosshair_proxies.clear()
            self.status_bar.show_message("Crosshair removed")

    def _update_crosshair(self, pts, panel, vline, hline, label) -> None:
        """Mouse-move callback for crosshair overlay."""
        try:
            pos = pts[0]
            vb = panel.get_view()
            mouse_point = vb.mapSceneToView(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())
            vline.setPos(mouse_point.x())
            hline.setPos(mouse_point.y())

            img_item = panel.view.getImageItem() if hasattr(panel, 'view') else None
            if img_item is None:
                img_item = panel.image_panel.view.getImageItem() if hasattr(panel, 'image_panel') else None
            val_str = ""
            if img_item is not None and img_item.image is not None:
                img = img_item.image
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    val = img[y, x]
                    if np.isscalar(val):
                        val_str = f"  val={val:.4g}"
                    else:
                        val_str = f"  val={val}"
            label.setText(f"({x}, {y}){val_str}")
            label.setPos(mouse_point.x() + 5, mouse_point.y())
        except Exception:
            pass

    # ─── Line Profile Tool ───
    def _on_line_profile_toggled(self, enabled: bool) -> None:
        """Toggle a draggable line segment with live phase-depth profile plot."""
        if enabled:
            # Check that we have phase data for depth measurement
            if self._phase_unwrapped is None:
                self.status_bar.show_message("Run reconstruction first — need phase data")
                self.toolbar.action_line_profile.setChecked(False)
                return

            # Default to phase panel — that's where the measurement lives.
            # Fall back to amplitude only if phase panel isn't showing an image
            # yet (e.g. Wrapped mode with no wrapped data set).
            phase_img = self.panel_phase.image_panel.view.getImageItem()
            if phase_img is not None and phase_img.image is not None:
                panel = self.panel_phase
                img_item = phase_img
                initial_combo_index = 1  # "Phase"
            else:
                panel = self.panel_amp
                img_item = panel.view.getImageItem()
                initial_combo_index = 0  # "Amplitude"

            if img_item is None or img_item.image is None:
                self.status_bar.show_message("Load an image first")
                self.toolbar.action_line_profile.setChecked(False)
                return

            h, w = img_item.image.shape[:2]
            vb = panel.get_view()

            roi = pg.LineSegmentROI(
                positions=[[w * 0.25, h * 0.5], [w * 0.75, h * 0.5]],
                pen=pg.mkPen("m", width=2),
                hoverPen=pg.mkPen("y", width=2),
            )
            vb.addItem(roi)

            self._line_profile_roi = roi
            self._line_profile_panel = panel

            # Create profile plot dialog with crosshair measurement
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox
            dlg = QDialog(self)
            dlg.setWindowTitle("Phase Line Profile — Crosshair Measurement")
            dlg.resize(620, 380)
            dlg_layout = QVBoxLayout(dlg)
            dlg_layout.setContentsMargins(4, 4, 4, 4)

            # Panel selector
            panel_bar = QHBoxLayout()
            panel_bar.addWidget(QLabel("ROI Panel:"))
            panel_combo = QComboBox()
            panel_combo.addItem("Amplitude", "amp")
            panel_combo.addItem("Phase", "phase")
            panel_combo.setCurrentIndex(initial_combo_index)
            panel_combo.currentIndexChanged.connect(self._on_line_profile_panel_changed)
            panel_bar.addWidget(panel_combo)
            panel_bar.addStretch()
            dlg_layout.addLayout(panel_bar)
            self._lp_panel_combo = panel_combo

            pw = pg.PlotWidget()
            pw.setLabel("bottom", "Lateral Distance (µm)")
            pw.setLabel("left", "Depth (µm)")
            pw.showGrid(x=True, y=True, alpha=0.3)
            # Disable default right-click context menu so right-click works for Sel 2
            pw.plotItem.vb.setMenuEnabled(False)
            curve = pw.plot(pen=pg.mkPen("c", width=2))
            stats_text = pg.TextItem("", anchor=(0, 0), color="w")
            stats_text.setPos(0, 0)
            pw.addItem(stats_text, ignoreBounds=True)

            # --- Crosshair elements on the plot ---
            ch_vline1 = pg.InfiniteLine(angle=90, movable=False,
                                        pen=pg.mkPen("#00ff00", width=1, style=2))
            ch_hline1 = pg.InfiniteLine(angle=0, movable=False,
                                        pen=pg.mkPen("#00ff00", width=1, style=2))
            ch_vline2 = pg.InfiniteLine(angle=90, movable=False,
                                        pen=pg.mkPen("#ff4444", width=1, style=2))
            ch_hline2 = pg.InfiniteLine(angle=0, movable=False,
                                        pen=pg.mkPen("#ff4444", width=1, style=2))
            for line in (ch_vline1, ch_hline1, ch_vline2, ch_hline2):
                line.setVisible(False)
                pw.addItem(line, ignoreBounds=True)

            # Marker dots for selections
            sel1_dot = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None),
                                          brush=pg.mkBrush("#00ff00"))
            sel2_dot = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None),
                                          brush=pg.mkBrush("#ff4444"))
            pw.addItem(sel1_dot, ignoreBounds=True)
            pw.addItem(sel2_dot, ignoreBounds=True)

            # Delta label on the plot
            delta_text = pg.TextItem("", anchor=(0.5, 0), color="#ffff00")
            pw.addItem(delta_text, ignoreBounds=True)

            dlg_layout.addWidget(pw)

            # --- Bottom bar: info labels + reset button ---
            bottom_bar = QHBoxLayout()
            sel1_label = QLabel("Sel 1 (Left Click): —")
            sel1_label.setStyleSheet("color: #00ff00; font-weight: bold;")
            sel2_label = QLabel("Sel 2 (Right Click): —")
            sel2_label.setStyleSheet("color: #ff4444; font-weight: bold;")
            delta_label = QLabel("Δ: —")
            delta_label.setStyleSheet("color: #ffff00; font-weight: bold;")
            reset_btn = QPushButton("Reset")
            reset_btn.setFixedWidth(60)

            bottom_bar.addWidget(sel1_label)
            bottom_bar.addWidget(sel2_label)
            bottom_bar.addWidget(delta_label)
            bottom_bar.addStretch()
            bottom_bar.addWidget(reset_btn)
            dlg_layout.addLayout(bottom_bar)

            dlg.show()

            self._line_profile_dlg = dlg
            self._line_profile_plot = pw
            self._line_profile_curve = curve
            self._line_profile_stats = stats_text

            # Crosshair state
            self._lp_ch = {
                "vline1": ch_vline1, "hline1": ch_hline1,
                "vline2": ch_vline2, "hline2": ch_hline2,
                "dot1": sel1_dot, "dot2": sel2_dot,
                "delta_text": delta_text,
                "sel1_label": sel1_label, "sel2_label": sel2_label,
                "delta_label": delta_label,
                "sel1": None,  # (x, y) tuple
                "sel2": None,
            }

            # Connect mouse clicks on the plot
            pw.scene().sigMouseClicked.connect(self._on_line_profile_plot_clicked)
            reset_btn.clicked.connect(self._reset_line_profile_crosshair)

            # Connect live update
            roi.sigRegionChanged.connect(self._update_line_profile)
            # Show initial profile
            self._update_line_profile()

            self.status_bar.show_message("Phase Line Profile — Left click: Sel1, Right click: Sel2")
        else:
            self._clear_line_profile()
            self.status_bar.show_message("Line Profile removed")

    def _get_effective_pixel_um(self) -> float:
        """Return effective pixel size in µm from the Recon tab."""
        rtab = self.sidebar_tabs.recon_tab
        px_um = float(rtab.pixel_um.value())
        if not rtab.pixel_is_effective_cb.isChecked():
            mag = float(rtab.magnification.value())
            px_um = px_um / (mag if mag > 0 else 1.0)
        else:
            # Warn if pixel_is_effective is checked but the pixel size seems
            # too large for the configured magnification (likely a config error).
            mag = float(rtab.magnification.value())
            if mag > 1.0 and px_um > 1.0:
                expected = px_um / mag
                log.warning(
                    "'Pixel is already effective' is checked but "
                    "pixel=%.2fµm with %.0fx magnification seems "
                    "too large. Expected effective pixel ≈ %.3fµm. "
                    "Uncheck the flag if this is a camera pixel size.",
                    px_um, mag, expected
                )
        return px_um

    def _phase_to_height_um(self, phase_rad: np.ndarray) -> np.ndarray:
        """Convert unwrapped phase (radians) to height in µm using current optical params."""
        rtab = self.sidebar_tabs.recon_tab
        wavelength_m = float(rtab.wavelength_nm.value()) * 1e-9

        from core.qpi import phase_to_opd, opd_to_height
        opd_m = phase_to_opd(phase_rad, wavelength_m)

        # Use QPI tab params if available, otherwise reflection mode
        try:
            qtab = self.sidebar_tabs.qpi_tab
            n_sample = float(qtab.n_sample.value())
            n_medium = float(qtab.n_medium.value())
            if abs(n_sample - n_medium) > 1e-6:
                height_m = opd_to_height(opd_m, n_sample, n_medium)
            else:
                height_m = opd_m / 2.0  # reflection mode fallback
        except Exception:
            height_m = opd_m / 2.0  # reflection mode fallback

        return height_m * 1e6  # metres → µm

    def _update_line_profile(self) -> None:
        """Recalculate and redraw the line profile from phase data, converted to depth (µm)."""
        roi = self._line_profile_roi
        panel = self._line_profile_panel
        if roi is None or panel is None:
            return

        # Use stored unwrapped phase data directly
        phase_data = self._phase_unwrapped
        if phase_data is None:
            return

        # Get ROI handle positions in image coordinates
        handles = roi.getSceneHandlePositions()
        if len(handles) < 2:
            return
        vb = panel.get_view()
        p0 = vb.mapSceneToView(handles[0][1])
        p1 = vb.mapSceneToView(handles[1][1])

        start = (int(p0.y()), int(p0.x()))
        end = (int(p1.y()), int(p1.x()))

        from core.qpi import extract_line_profile
        try:
            distances, values = extract_line_profile(phase_data, start, end)
        except Exception:
            return

        # Convert pixel distances to µm (lateral)
        px_um = self._get_effective_pixel_um()
        distances_um = distances * px_um

        # Convert phase values to depth in µm
        height_um = self._phase_to_height_um(values)

        if self._line_profile_curve is not None:
            self._line_profile_curve.setData(distances_um, height_um)

        # Update stats text
        if self._line_profile_stats is not None and len(height_um) > 0:
            vmin = float(np.min(height_um))
            vmax = float(np.max(height_um))
            vmean = float(np.mean(height_um))
            length_um = float(distances_um[-1]) if len(distances_um) > 0 else 0
            self._line_profile_stats.setText(
                f"L={length_um:.1f}µm  min={vmin:.3f}µm  max={vmax:.3f}µm  mean={vmean:.3f}µm"
            )
            # Position stats at top-left of plot
            vr = self._line_profile_plot.viewRange()
            self._line_profile_stats.setPos(vr[0][0], vr[1][1])

    def _on_line_profile_plot_clicked(self, event) -> None:
        """Handle mouse clicks on the line profile plot for crosshair selection."""
        ch = getattr(self, "_lp_ch", None)
        if ch is None or self._line_profile_plot is None:
            return

        pw = self._line_profile_plot
        vb = pw.plotItem.vb
        scene_pos = event.scenePos()
        mouse_point = vb.mapSceneToView(scene_pos)
        x, y = float(mouse_point.x()), float(mouse_point.y())

        from PySide6.QtCore import Qt
        button = event.button()

        if button == Qt.MouseButton.LeftButton:
            # Selection 1 (green)
            ch["sel1"] = (x, y)
            ch["vline1"].setPos(x); ch["hline1"].setPos(y)
            ch["vline1"].setVisible(True); ch["hline1"].setVisible(True)
            ch["dot1"].setData([x], [y])
            ch["sel1_label"].setText(f"Sel 1: ({x:.2f}, {y:.3f}) µm")

        elif button == Qt.MouseButton.RightButton:
            # Selection 2 (red)
            ch["sel2"] = (x, y)
            ch["vline2"].setPos(x); ch["hline2"].setPos(y)
            ch["vline2"].setVisible(True); ch["hline2"].setVisible(True)
            ch["dot2"].setData([x], [y])
            ch["sel2_label"].setText(f"Sel 2: ({x:.2f}, {y:.3f}) µm")

        # Update delta if both selections exist
        if ch["sel1"] is not None and ch["sel2"] is not None:
            dx = abs(ch["sel2"][0] - ch["sel1"][0])
            dy = abs(ch["sel2"][1] - ch["sel1"][1])
            ch["delta_label"].setText(f"Δ: ({dx:.2f}, {dy:.3f}) µm")
            # Position delta text between the two points on the plot
            mx = (ch["sel1"][0] + ch["sel2"][0]) / 2
            my = max(ch["sel1"][1], ch["sel2"][1])
            ch["delta_text"].setText(f"Δx={dx:.2f} µm\nΔy={dy:.3f} µm")
            ch["delta_text"].setPos(mx, my)

    def _reset_line_profile_crosshair(self) -> None:
        """Reset crosshair selections on the line profile plot."""
        ch = getattr(self, "_lp_ch", None)
        if ch is None:
            return
        ch["sel1"] = None
        ch["sel2"] = None
        for key in ("vline1", "hline1", "vline2", "hline2"):
            ch[key].setVisible(False)
        ch["dot1"].setData([], [])
        ch["dot2"].setData([], [])
        ch["delta_text"].setText("")
        ch["sel1_label"].setText("Sel 1 (Left Click): —")
        ch["sel2_label"].setText("Sel 2 (Right Click): —")
        ch["delta_label"].setText("Δ: —")

    def _on_line_profile_panel_changed(self, index: int) -> None:
        """Move line profile ROI between amplitude and phase panels."""
        roi = self._line_profile_roi
        old_panel = self._line_profile_panel
        if roi is None or old_panel is None:
            return

        new_panel = self.panel_amp if index == 0 else self.panel_phase
        if new_panel is old_panel:
            return

        try:
            old_panel.get_view().removeItem(roi)
        except Exception:
            pass
        new_panel.get_view().addItem(roi)
        self._line_profile_panel = new_panel
        self._update_line_profile()

    def _clear_line_profile(self) -> None:
        """Remove line profile ROI and close plot window."""
        if self._line_profile_roi is not None and self._line_profile_panel is not None:
            try:
                self._line_profile_panel.get_view().removeItem(self._line_profile_roi)
            except Exception:
                pass
        if self._line_profile_plot is not None:
            try:
                self._line_profile_plot.scene().sigMouseClicked.disconnect(
                    self._on_line_profile_plot_clicked)
            except Exception:
                pass
        if self._line_profile_dlg is not None:
            self._line_profile_dlg.close()
        self._line_profile_roi = None
        self._line_profile_panel = None
        self._line_profile_dlg = None
        self._line_profile_plot = None
        self._line_profile_curve = None
        self._line_profile_stats = None
        self._lp_ch = None
        self._lp_panel_combo = None

    def _on_export_view(self) -> None:
        """Export the current image grid as a PNG."""
        from PySide6.QtWidgets import QFileDialog
        folder = self._last_folder_default()
        default = str(Path(folder) / "view_export.png") if folder else "view_export.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export View", default,
            "Images (*.png *.jpg *.tiff *.bmp)"
        )
        if not path:
            return
        self._update_io_history(last_folder=str(Path(path).parent))
        pixmap = self.image_grid.grab()
        pixmap.save(path)
        self.status_bar.show_message(f"View exported to {path}")

    # ─── Menus (Help + Tools extensions) ───
    def _init_menus(self) -> None:
        """Build the menu bar with Help + Tools extensions (report action).

        Both actions come from the command registry so the palette, the
        menu, and any future keybinding editor see the same titles and
        callbacks.
        """
        from gui.commands import bind_to_qt, get_registry

        menu_bar = self.menuBar()
        reg = get_registry()

        # ─ Tools menu (report) ─
        tools_menu = menu_bar.addMenu("Tools")

        report_cmd = reg.get("tools.generate_report")
        if report_cmd is not None:
            self.action_generate_report = bind_to_qt(self, report_cmd)
            tools_menu.addAction(self.action_generate_report)
            # Mirror report action on the toolbar for discoverability.
            try:
                self.toolbar.addSeparator()
                self.toolbar.addAction(self.action_generate_report)
            except Exception:
                pass

        # ─ Help menu ─
        help_menu = menu_bar.addMenu("Help")
        about_cmd = reg.get("help.about")
        if about_cmd is not None:
            self.action_about = bind_to_qt(self, about_cmd)
            help_menu.addAction(self.action_about)

    def _show_about_dialog(self) -> None:
        """Show the About dialog (modal)."""
        try:
            from .dialogs.about_dialog import AboutDialog
            dlg = AboutDialog(self)
            dlg.exec()
        except Exception as e:
            log.exception("Failed to show About dialog")
            self.status_bar.show_message(f"About dialog failed: {e}")

    # ─── Image panel context menus ───
    def _init_image_panel_context_menus(self) -> None:
        """Wire right-click context menus on amplitude & phase panels."""
        try:
            self.panel_amp.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.panel_amp.customContextMenuRequested.connect(self._on_amp_panel_context_menu)
        except Exception:
            log.exception("Could not attach context menu to amplitude panel")

        try:
            self.panel_phase.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.panel_phase.customContextMenuRequested.connect(self._on_phase_panel_context_menu)
        except Exception:
            log.exception("Could not attach context menu to phase panel")

    def _on_amp_panel_context_menu(self, pos: QPoint) -> None:
        """Right-click menu on the amplitude panel."""
        menu = QMenu(self.panel_amp)

        act_line = menu.addAction("Line profile")
        act_line.triggered.connect(self._trigger_line_profile_from_menu)

        act_export = menu.addAction("Export image…")
        act_export.triggered.connect(lambda: self._export_panel_image(self.panel_amp, "amplitude"))

        global_pos = self.panel_amp.mapToGlobal(pos)
        menu.exec(global_pos)

    def _on_phase_panel_context_menu(self, pos: QPoint) -> None:
        """Right-click menu on the phase panel."""
        menu = QMenu(self.panel_phase)

        act_line = menu.addAction("Line profile")
        act_line.triggered.connect(self._trigger_line_profile_from_menu)

        act_3d = menu.addAction("3D surface")
        # Only enable 3D if we have reconstruction output with phase data
        act_3d.setEnabled(
            self._qpi_last_result is not None
            or self._phase_unwrapped is not None
        )
        act_3d.triggered.connect(self._show_3d_surface_from_menu)

        act_export = menu.addAction("Export image…")
        act_export.triggered.connect(lambda: self._export_panel_image(self.panel_phase, "phase"))

        global_pos = self.panel_phase.mapToGlobal(pos)
        menu.exec(global_pos)

    def _trigger_line_profile_from_menu(self) -> None:
        """Toggle the line-profile tool ON via the toolbar action."""
        try:
            act = self.toolbar.action_line_profile
            if not act.isChecked():
                act.setChecked(True)  # triggers the toggled signal → _on_line_profile_toggled
            else:
                # Already on — just re-show the dialog if it exists
                if self._line_profile_dlg is not None:
                    self._line_profile_dlg.raise_()
                    self._line_profile_dlg.activateWindow()
        except Exception as e:
            log.exception("Line profile trigger failed")
            self.status_bar.show_message(f"Line profile failed: {e}")

    def _show_3d_surface_from_menu(self) -> None:
        """Open the 3D surface. Prefer the full QPI result; otherwise build a minimal one."""
        if self._qpi_last_result is not None:
            self._show_3d_surface(self._qpi_last_result)
            return

        if self._phase_unwrapped is None:
            self.status_bar.show_message("Run reconstruction (and QPI) first — no phase data")
            return

        # Minimal shim: _show_3d_surface reads .height_nm / .opd_nm. Build a small object.
        try:
            phase = self._phase_unwrapped
            # Convert phase → height in nm using current optical params (µm → nm)
            height_um = self._phase_to_height_um(phase)
            height_nm = (height_um * 1000.0).astype(np.float64)

            class _Shim:
                pass
            shim = _Shim()
            shim.height_nm = height_nm
            shim.opd_nm = None
            self._show_3d_surface(shim)
        except Exception as e:
            log.exception("3D surface from phase failed")
            self.status_bar.show_message(f"3D surface failed: {e}")

    def _export_panel_image(self, panel, label: str) -> None:
        """Save the visible contents of a panel as a PNG."""
        default_name = f"dhm_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        folder = self._last_folder_default()
        default = str(Path(folder) / default_name) if folder else default_name
        path, _ = QFileDialog.getSaveFileName(
            self, "Export image", default,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)"
        )
        if not path:
            return
        self._update_io_history(last_folder=str(Path(path).parent))
        try:
            pixmap = panel.grab()
            ok = pixmap.save(path)
            if not ok:
                raise RuntimeError("QPixmap.save returned False")
            self.status_bar.show_message(f"Image saved: {path}", timeout=6000)
        except Exception as e:
            log.exception("Panel export failed")
            self.status_bar.show_message(f"Image save failed: {e}", timeout=10000)

    # ─── Audit helper ───
    def _audit(self, **kwargs) -> None:
        """Best-effort audit record. Never raises."""
        if get_audit_log is None:
            return
        try:
            get_audit_log().record(**kwargs)
        except Exception:
            log.warning("audit log failed", exc_info=True)

    # ─── Report generation ───
    def _collect_recon_params_for_report(self) -> dict:
        """Gather a JSON-friendly snapshot of current reconstruction settings."""
        rtab = self.sidebar_tabs.recon_tab
        ptab = self.sidebar_tabs.process_tab
        try:
            method = rtab.method_combo.currentData()
            method_str = getattr(method, "name", None) or str(method)
        except Exception:
            method_str = "unknown"

        params = {
            "z_m": float(rtab.z_mm.value()) * 1e-3,
            "wavelength_m": float(rtab.wavelength_nm.value()) * 1e-9,
            "pixel_size_m": self._get_effective_pixel_um() * 1e-6,
            "method": method_str,
            "mask_radius": int(rtab.mask_radius.value()),
        }
        try:
            params["magnification"] = float(rtab.magnification.value())
        except Exception:
            pass
        try:
            params["fft_backend"] = rtab.fft_backend_combo.currentData()
        except Exception:
            pass
        try:
            params["subtract_mean"] = bool(ptab.subtract_mean_cb.isChecked())
            params["hann_window"] = bool(ptab.hann_cb.isChecked())
        except Exception:
            pass
        return params

    def _on_generate_report_triggered(self) -> None:
        """Gather state and generate an HTML report via core.report."""
        default_name = f"dhm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        # Report folder remembered separately from the image folder — the lab
        # typically writes reports to a shared network share, not next to
        # the holograms.
        try:
            default_dir = self._app_settings.io.last_report_folder or ""
        except AttributeError:
            default_dir = ""
        if not default_dir and self._loaded_path is not None:
            try:
                default_dir = str(self._loaded_path.parent)
            except Exception:
                pass
        if not default_dir:
            default_dir = str(Path.home())
        default_path = str(Path(default_dir) / default_name)

        chosen_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save report", default_path, "HTML (*.html);;PDF (*.pdf)"
        )
        if not chosen_path:
            return
        self._update_io_history(last_report_folder=str(Path(chosen_path).parent))

        # Resolve format from the extension first, then the filter. Add the
        # missing extension if the user left it off (Qt does this on Linux but
        # not on macOS, so don't trust the dialog to do it for us).
        ext = Path(chosen_path).suffix.lower()
        if ext not in (".html", ".htm", ".pdf"):
            if "pdf" in (selected_filter or "").lower():
                chosen_path += ".pdf"
                ext = ".pdf"
            else:
                chosen_path += ".html"
                ext = ".html"

        try:
            from core.report import generate_html_report, generate_pdf_report
        except Exception as e:
            log.exception("core.report not available")
            self.status_bar.show_message(
                f"Report failed: report module unavailable ({e})", timeout=10000
            )
            return

        # Current images (2D arrays or None)
        phase_image = None
        amplitude_image = None
        height_image = None
        try:
            if self._recon_complex is not None:
                amplitude_image = np.abs(self._recon_complex)
            if self._phase_unwrapped is not None:
                phase_image = self._phase_unwrapped
            elif self._recon_complex is not None:
                phase_image = np.angle(self._recon_complex)
            if self._phase_unwrapped is not None:
                try:
                    height_image = self._phase_to_height_um(self._phase_unwrapped)
                except Exception:
                    height_image = None
        except Exception:
            log.exception("Failed to snapshot images for report")

        # Use last recon params if available, else build from UI
        recon_params = self._last_recon_params or self._collect_recon_params_for_report()

        # Autofocus summary
        autofocus_summary = self._last_af_result

        # QPI result (raw object; core.report will introspect or skip if None)
        qpi_result = self._last_qpi_result or self._qpi_last_result

        report_fn = generate_pdf_report if ext == ".pdf" else generate_html_report
        fmt_label = "PDF" if ext == ".pdf" else "HTML"

        try:
            path = report_fn(
                output_path=Path(chosen_path),
                title="DHM Reconstruction Report",
                operator=None,  # v1.1 will prompt
                recon_params=recon_params,
                autofocus_summary=autofocus_summary,
                qpi_result=qpi_result,
                phase_image=phase_image,
                amplitude_image=amplitude_image,
                height_image=height_image,
                notes="",
            )
            self.status_bar.show_message(f"{fmt_label} report saved: {path}", timeout=8000)
            self._audit(
                action="export_report",
                params={"path": str(path), "format": fmt_label.lower()},
                result_summary={"has_phase": phase_image is not None,
                                 "has_amplitude": amplitude_image is not None,
                                 "has_height": height_image is not None},
            )
        except Exception as exc:
            log.exception("Report generation failed")
            self.status_bar.show_message(f"Report failed: {exc}", timeout=10000)

    # ─── Worker failure slots (additions) ───
    def _on_recording_worker_error(self, msg: str) -> None:
        """Surface recording worker failures on the status bar."""
        log.error("Recording worker failed: %s", msg)
        self.status_bar.show_message(f"Recording failed: {msg}", timeout=10000)

    def _run_fft_benchmark(self):
        self.status_bar.show_message("Running FFT Benchmark...")
        try:
            import time
            from core.fft_backend import get_best_fft_backend
            fft = get_best_fft_backend()
            
            # Create dummy 1024x1024 array
            test_arr = np.random.rand(1024, 1024).astype(np.complex128)
            
            # Warmup
            fft.fft2(test_arr)
            
            # Benchmark
            t0 = time.perf_counter()
            for _ in range(5):
                res = fft.fft2(test_arr)
                fft.ifft2(res)
                
            self.status_bar.show_message(f"FFT Benchmark: {(time.perf_counter() - t0) * 1000 / 5:.1f} ms/frame")
            
            self.status_bar.show_message(f"Ready | Backend: {fft.name.upper()}")
            
            # Match UI with backend 
            idx = self.sidebar_tabs.recon_tab.fft_backend_combo.findData(fft.name.value)
            if idx >= 0:
                self.sidebar_tabs.recon_tab.fft_backend_combo.setCurrentIndex(idx)
        except Exception as e:
            self.status_bar.show_message(f"Ready | Benchmark failed: {e}")
