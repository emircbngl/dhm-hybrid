from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QDockWidget,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QSettings
import pyqtgraph as pg
import numpy as np
from pathlib import Path

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

from core.contrast import apply_contrast, ContrastMethod
from core.profile_manager import ProfileManager

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
        self._phase_unwrapped = None
        self._spectrum_mag = None

        self._batch_paths = []
        self._batch_out_root = None
        self._batch_running = False

        self._export_crop_roi = None
        self._export_crop_enabled = False
        self._export_crop_source_view = None

        self._frame_counter = 0
        self._adaptive_state = None
        
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
        self.status_bar.show_message("Ready")
        
        self._refresh_profiles("all")
        
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
        """Save settings when the window is closed."""
        self._save_layout()
        # Stop autofocus worker first (must finish before QThread destructor)
        if self._af_worker and self._af_worker.isRunning():
            self._af_worker.cancel()
            self._af_worker.wait(5000)
        if hasattr(self, '_recon_worker'):
            self._recon_worker.stop()
            self._recon_worker.wait(5000)
        try:
            self._on_camera_disconnect()
        except Exception:
            pass
        super().closeEvent(event)

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
        from PySide6.QtGui import QKeySequence, QShortcut
        
        # Maximize panels
        QShortcut(QKeySequence("Ctrl+1"), self).activated.connect(
            lambda: self.image_grid.toggle_maximize(self.panel_input)
        )
        QShortcut(QKeySequence("Ctrl+2"), self).activated.connect(
            lambda: self.image_grid.toggle_maximize(self.panel_amp)
        )
        QShortcut(QKeySequence("Ctrl+3"), self).activated.connect(
            lambda: self.image_grid.toggle_maximize(self.panel_phase)
        )
        QShortcut(QKeySequence("Ctrl+4"), self).activated.connect(
            lambda: self.image_grid.toggle_maximize(self.panel_spectrum)
        )
        
        # Restore grid
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(
            self.image_grid.restore_grid
        )
        
        # Reset layout
        QShortcut(QKeySequence("Ctrl+Shift+R"), self).activated.connect(
            self._reset_layout_to_defaults
        )

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
        self._update_info()

        # Update input image view
        img = self._as_field_2d(self._loaded_array)
        self.panel_input.set_image(img, autoLevels=True)
        self._frame_counter += 1
        
        # Enable processing buttons now that we have data
        self.sidebar_tabs.recon_tab.recon_btn.setEnabled(True)
        self.sidebar_tabs.focus_tab.autofocus_btn.setEnabled(True)
        self.sidebar_tabs.focus_tab.benchmark_btn.setEnabled(True)
        self.sidebar_tabs.focus_tab.diagnostic_btn.setEnabled(True)

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
            'fft_backend': fft_backend,
            'filter_enable': ptab.filter_enable_cb.isChecked(),
            'filter_type': ptab.filter_type_combo.currentData(),
            'filter_cutoff': ptab.filter_cutoff.value(),
            'filter_rolloff': ptab.filter_rolloff.value(),
        }

    def _trigger_reconstruction(self) -> None:
        """Packages UI parameters and submits an async job to ReconstructionWorker."""
        if self._loaded_array is None:
            return

        self.sidebar_tabs.recon_tab.recon_btn.setEnabled(False)
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
        self._spectrum_mag = result.get('spectrum_mag')
        self._phase_unwrapped = result.get('phase_unwrapped')
        
        if self._spectrum_mag is not None:
            spec_log = np.log1p(self._spectrum_mag)
            self.panel_spectrum.set_image(spec_log, autoLevels=True)
            
        if self._recon_complex is not None:
            self._refresh_display_with_contrast()
                
            if self._recording_worker.is_recording():
                fdict = {
                    "amplitude": np.abs(self._recon_complex),
                    "phase": np.angle(self._recon_complex) if self._phase_unwrapped is None else self._phase_unwrapped,
                    "input": self._as_field_2d(self._loaded_array) if self._loaded_array is not None else np.zeros((10,10), dtype=np.uint8),
                    "z_mm": self.sidebar_tabs.recon_tab.z_mm.value()
                }
                self._recording_worker.push_frame(fdict)
                
        self.sidebar_tabs.recon_tab.recon_btn.setEnabled(True)
        self.status_bar.show_message("Reconstruction complete")

    def _get_contrast_config(self) -> dict:
        """Read contrast settings from the UI widget (if it exists)."""
        # Try process_tab.contrast (preferred location)
        ptab = getattr(self.sidebar_tabs, 'process_tab', None)
        ctrl = getattr(ptab, 'contrast', None) if ptab else None
        if ctrl is not None and hasattr(ctrl, 'get_contrast_config'):
            return ctrl.get_contrast_config()
        return {"method": ContrastMethod.NONE}

    def _refresh_display_with_contrast(self) -> None:
        """
        Re-display amplitude (and optionally phase) with current contrast settings.
        Called from _on_recon_completed and when contrast controls change.
        The raw reconstruction data (_recon_complex) is never modified —
        contrast is purely a display transform.
        """
        if self._recon_complex is None:
            return

        cfg = self._get_contrast_config()
        method = cfg.get("method", ContrastMethod.NONE)

        # Amplitude display
        amp_raw = np.abs(self._recon_complex)
        if method != ContrastMethod.NONE:
            amp_display = apply_contrast(
                amp_raw, method,
                p_low=cfg.get("p_low", 1.0),
                p_high=cfg.get("p_high", 99.0),
                clahe_clip=cfg.get("clahe_clip", 2.0),
                clahe_grid=cfg.get("clahe_grid", 8),
            )
            self.panel_amp.set_image(amp_display, autoLevels=False)
        else:
            self.panel_amp.set_image(amp_raw, autoLevels=True)

        # Phase display
        self.panel_phase.set_wrapped(np.angle(self._recon_complex), autoLevels=True)
        if self._phase_unwrapped is not None:
            if cfg.get("apply_phase", False) and method != ContrastMethod.NONE:
                phase_display = apply_contrast(
                    self._phase_unwrapped, method,
                    p_low=cfg.get("p_low", 1.0),
                    p_high=cfg.get("p_high", 99.0),
                    clahe_clip=cfg.get("clahe_clip", 2.0),
                    clahe_grid=cfg.get("clahe_grid", 8),
                )
                self.panel_phase.set_unwrapped(phase_display, autoLevels=False)
            else:
                self.panel_phase.set_unwrapped(self._phase_unwrapped, autoLevels=True)

    # ─── Camera Live Mode Hooks ───
    def _on_mode_changed(self, mode: str):
        if mode == "Live":
            self.sidebar_tabs.tabs.setCurrentWidget(self.sidebar_tabs.camera_tab)
            self.toolbar.action_batch.setEnabled(False)
            self.toolbar.action_load.setEnabled(False)
        else:
            self._on_acq_stop()
            self._on_camera_disconnect()
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

        offaxis_params = OffAxisParams(radius=int(rtab.mask_radius.value()))
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

            worker = AutofocusWorker(self)
            worker.configure(
                field=fc, base_params=params, method=method, metric=metric,
                algo=ftab.algo_combo.currentText(),
                z_min_m=zmin, z_max_m=zmax, steps=steps,
                auto_select=ftab.auto_metric_cb.isChecked(),
                step_init=step_init,
                grow_factor=ftab.grow_factor.value(),
                shrink_factor=ftab.shrink_factor.value(),
                refine_levels=ftab.refine_levels.value(),
                refine_divisions=ftab.refine_divisions.value(),
            )

            worker.progress.connect(self._af_overlay.set_status)
            worker.progress_pct.connect(self._af_overlay.update_progress)
            worker.metric_selected.connect(self._on_af_metric_selected)
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

    def _on_af_worker_finished(self, best_z, z_arr, scores, metric_val, elapsed, evals) -> None:
        self._af_overlay.hide_overlay()
        self._af_worker = None
        self._autofocus_table = np.column_stack((z_arr, scores))
        self._autofocus_best_z_m = best_z

        self.status_bar.show_message(
            f"Auto-focus: {elapsed:.2f}s | Z={best_z*1e3:.4f} mm | {metric_val} | {evals} evals"
        )

        self.sidebar_tabs.recon_tab.z_mm.setValue(best_z * 1e3)
        self.sidebar_tabs.focus_tab.autofocus_btn.setEnabled(True)
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
            lines = [
                f"{'Algorithm':<30} {'Metric':<20} {'Best Z (mm)':<14} {'Time (s)':<10} {'Evals':<6}",
                "-" * 82,
            ]
            for e in result.entries:
                lines.append(
                    f"{e.algorithm:<30} {e.metric.value:<20} {e.best_z_m*1e3:<14.4f} {e.elapsed_s:<10.3f} {e.evaluations:<6}"
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

        # Baseline params wrapper
        params = ReconstructionParams(
            wavelength_m=wl,
            pixel_size_m=px,
            z_m=0.0,
            n=1.0
        )

        img = self._as_field_2d(self._loaded_array)
        if ptab.subtract_mean_cb.isChecked():
            img = img - float(np.mean(img))
        if ptab.hann_cb.isChecked():
            wy = np.hanning(img.shape[0]).astype(np.float32)
            wx = np.hanning(img.shape[1]).astype(np.float32)
            img = img * (wy[:, None] * wx[None, :])
        if np.max(np.abs(img)) > 0:
            img = img / float(np.max(np.abs(img)))

        offaxis_params = OffAxisParams(radius=int(rtab.mask_radius.value()))
        fc, _, _, _ = extract_complex_field_offaxis_debug(img, offaxis_params)

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
                algorithm=ftab.algo_combo.currentText()
            )
        else:
            self._adaptive_state.current_z_m = z_m_curr
            self._adaptive_state.search_range_m = search_range_m
            self._adaptive_state.steps = steps
            self._adaptive_state.metric = metric
            self._adaptive_state.algorithm = ftab.algo_combo.currentText()

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
        path = QFileDialog.getExistingDirectory(self, "Output Directory")
        if path:
            rectab.out_dir_lbl.setText(str(Path(path).name))
            self._vid_output_path = Path(path)

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
        path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "snapshot.png", "Images (*.png *.tiff *.bmp)")
        if not path:
            return
            
        pixmap = self.image_grid.grab()
        pixmap.save(path)
        self.status_bar.show_message(f"Snapshot grid saved to {path}")

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
