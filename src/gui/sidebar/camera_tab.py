from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QComboBox, 
    QSpinBox, QPushButton, QGroupBox, QCheckBox, QHBoxLayout,
    QDoubleSpinBox
)
from core.camera import NICamera, NI_AVAILABLE

class CameraTab(QWidget):
    """Sidebar tab managing live camera connection and acquisition settings."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # ─── Connection ───
        conn_group = QGroupBox("Connection")
        conn_grid = QGridLayout(conn_group)

        self.cam_combo = QComboBox()
        self.refresh_cams_btn = QPushButton("↻")
        self.refresh_cams_btn.setFixedWidth(30)
        
        cam_row_layout = QHBoxLayout()
        cam_row_layout.addWidget(self.cam_combo)
        cam_row_layout.addWidget(self.refresh_cams_btn)
        
        conn_grid.addWidget(QLabel("Camera"), 0, 0)
        conn_grid.addLayout(cam_row_layout, 0, 1)

        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        btn_layout.addWidget(self.connect_btn)
        btn_layout.addWidget(self.disconnect_btn)
        conn_grid.addLayout(btn_layout, 1, 0, 1, 2)

        self.status_label = QLabel("Status: Disconnected")
        conn_grid.addWidget(self.status_label, 2, 0, 1, 2)

        if not NI_AVAILABLE:
            self.status_label.setText("Status: NI drivers missing (File mode only)")
            self.connect_btn.setEnabled(False)
            self.cam_combo.setEnabled(False)

        layout.addWidget(conn_group)

        # ─── Acquisition mode ───
        acq_group = QGroupBox("Acquisition mode")
        acq_grid = QGridLayout(acq_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Continuous", "Triggered"])
        acq_grid.addWidget(QLabel("Mode"), 0, 0)
        acq_grid.addWidget(self.mode_combo, 0, 1)

        self.trigger_combo = QComboBox()
        self.trigger_combo.addItems(["Software", "Hardware"])
        self.trigger_combo.setEnabled(False)
        self.trigger_lbl = QLabel("Trigger src")
        self.trigger_lbl.setEnabled(False)
        acq_grid.addWidget(self.trigger_lbl, 1, 0)
        acq_grid.addWidget(self.trigger_combo, 1, 1)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 200)
        self.fps_spin.setValue(30)
        acq_grid.addWidget(QLabel("FPS target"), 2, 0)
        acq_grid.addWidget(self.fps_spin, 2, 1)

        self.limit_fps_cb = QCheckBox("Limit FPS")
        self.limit_fps_cb.setChecked(True)
        acq_grid.addWidget(self.limit_fps_cb, 3, 0, 1, 2)

        layout.addWidget(acq_group)

        # ─── Camera settings ───
        cam_set_group = QGroupBox("Camera settings")
        cam_set_grid = QGridLayout(cam_set_group)

        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(10, 1_000_000)
        self.exposure_spin.setValue(5000)
        self.exposure_spin.setDecimals(0)
        cam_set_grid.addWidget(QLabel("Exposure (µs)"), 0, 0)
        cam_set_grid.addWidget(self.exposure_spin, 0, 1)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0, 100)
        self.gain_spin.setValue(0.0)
        self.gain_spin.setDecimals(1)
        cam_set_grid.addWidget(QLabel("Gain (dB)"), 1, 0)
        cam_set_grid.addWidget(self.gain_spin, 1, 1)

        self.bitdepth_combo = QComboBox()
        self.bitdepth_combo.addItems(["Mono16", "Mono8"])
        cam_set_grid.addWidget(QLabel("Bit depth"), 2, 0)
        cam_set_grid.addWidget(self.bitdepth_combo, 2, 1)

        layout.addWidget(cam_set_group)

        # ─── Live reconstruction ───
        live_group = QGroupBox("Live reconstruction")
        live_grid = QGridLayout(live_group)

        self.auto_recon_cb = QCheckBox("Auto-reconstruct each frame")
        self.auto_recon_cb.setChecked(True)
        live_grid.addWidget(self.auto_recon_cb, 0, 0, 1, 2)

        self.recon_skip_spin = QSpinBox()
        self.recon_skip_spin.setRange(1, 100)
        self.recon_skip_spin.setValue(1)
        live_grid.addWidget(QLabel("Reconstruct every N frames (1=all)"), 1, 0)
        live_grid.addWidget(self.recon_skip_spin, 1, 1)

        # Stats box
        self.stats_label = QLabel(
            "Acquisition FPS: 0.0\n"
            "Recon FPS: 0.0\n"
            "Dropped: 0\n"
            "Frame: #0"
        )
        self.stats_label.setStyleSheet("background-color: #2b2b2b; padding: 4px; border-radius: 4px;")
        live_grid.addWidget(self.stats_label, 2, 0, 1, 2)

        layout.addWidget(live_group)

        # ─── Start/Stop ───
        ctrl_layout = QHBoxLayout()
        self.start_acq_btn = QPushButton("▶ Start acquisition")
        self.stop_acq_btn = QPushButton("⏹ Stop")
        self.stop_acq_btn.setEnabled(False)
        ctrl_layout.addWidget(self.start_acq_btn)
        ctrl_layout.addWidget(self.stop_acq_btn)
        layout.addLayout(ctrl_layout)

        layout.addStretch()

        self._connect_ui()
        self.refresh_cameras()

    def _connect_ui(self):
        self.refresh_cams_btn.clicked.connect(self.refresh_cameras)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

    def _on_mode_changed(self, mode: str):
        is_trig = (mode == "Triggered")
        self.trigger_combo.setEnabled(is_trig)
        self.trigger_lbl.setEnabled(is_trig)

    def refresh_cameras(self):
        if not NI_AVAILABLE:
            self.cam_combo.clear()
            self.cam_combo.addItem("No NI drivers")
            return
            
        cams = NICamera.enumerate_cameras()
        self.cam_combo.clear()
        if not cams:
            self.cam_combo.addItem("No cameras found")
        else:
            for c in cams:
                self.cam_combo.addItem(f"{c['name']} ({c['model']})", userData=c['name'])

    def get_state(self) -> dict:
        # Note: 'cam_name' isn't explicitly settable if offline, 
        # but we capture it for complete hardware setups.
        return {
            "mode": self.mode_combo.currentText(),
            "trigger": self.trigger_combo.currentText(),
            "fps": self.fps_spin.value(),
            "limit_fps": self.limit_fps_cb.isChecked(),
            "exposure": self.exposure_spin.value(),
            "gain": self.gain_spin.value(),
            "bitdepth": self.bitdepth_combo.currentText(),
            "auto_recon": self.auto_recon_cb.isChecked(),
            "recon_skip": self.recon_skip_spin.value()
        }

    def set_state(self, state: dict):
        if "mode" in state:
            idx = self.mode_combo.findText(state["mode"])
            if idx >= 0: self.mode_combo.setCurrentIndex(idx)
        if "trigger" in state:
            idx = self.trigger_combo.findText(state["trigger"])
            if idx >= 0: self.trigger_combo.setCurrentIndex(idx)
        if "fps" in state:
            self.fps_spin.setValue(state["fps"])
        if "limit_fps" in state:
            self.limit_fps_cb.setChecked(state["limit_fps"])
        if "exposure" in state:
            self.exposure_spin.setValue(state["exposure"])
        if "gain" in state:
            self.gain_spin.setValue(state["gain"])
        if "bitdepth" in state:
            idx = self.bitdepth_combo.findText(state["bitdepth"])
            if idx >= 0: self.bitdepth_combo.setCurrentIndex(idx)
        if "auto_recon" in state:
            self.auto_recon_cb.setChecked(state["auto_recon"])
        if "recon_skip" in state:
            self.recon_skip_spin.setValue(state["recon_skip"])
