from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QComboBox, 
    QSpinBox, QPushButton, QGroupBox, QCheckBox,
    QLineEdit, QStackedWidget, QDoubleSpinBox
)
from PySide6.QtCore import QTimer
import shutil
import os

class RecordTab(QWidget):
    """Sidebar tab managing Video, Timelapse, Scheduled, and Snapshot modes."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # ─── Output Configuration ───
        out_group = QGroupBox("Output Configuration")
        out_grid = QGridLayout(out_group)

        self.out_dir_lbl = QLabel("Not set")
        self.browse_btn = QPushButton("Browse")
        
        out_grid.addWidget(QLabel("Directory"), 0, 0)
        out_grid.addWidget(self.out_dir_lbl, 0, 1)
        out_grid.addWidget(self.browse_btn, 0, 2)

        self.name_template = QLineEdit("{name}_{z}mm_{channel}_{frame:04d}")
        out_grid.addWidget(QLabel("Template"), 1, 0)
        out_grid.addWidget(self.name_template, 1, 1, 1, 2)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["TIFF Stack", "Individual PNGs", "AVI (raw)", "MP4"])
        out_grid.addWidget(QLabel("Format"), 2, 0)
        out_grid.addWidget(self.format_combo, 2, 1, 1, 2)

        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["All", "Amplitude only", "Phase only", "Input only"])
        out_grid.addWidget(QLabel("Channels"), 3, 0)
        out_grid.addWidget(self.channel_combo, 3, 1, 1, 2)
        
        layout.addWidget(out_group)

        # ─── Acquisition Mode ───
        mode_group = QGroupBox("Acquisition Mode")
        mode_grid = QGridLayout(mode_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Snapshot", "Manual Video", "Timelapse", "Scheduled"])
        mode_grid.addWidget(QLabel("Type"), 0, 0)
        mode_grid.addWidget(self.mode_combo, 0, 1)

        self.stacked_params = QStackedWidget()
        
        # 0: Snapshot
        snap_wid = QWidget()
        snap_layout = QGridLayout(snap_wid)
        snap_layout.setContentsMargins(0,0,0,0)
        self.snap_color_cb = QCheckBox("Colorbar")
        self.snap_color_cb.setChecked(True)
        self.snap_scale_cb = QCheckBox("Scale bar")
        snap_layout.addWidget(self.snap_color_cb, 0, 0)
        snap_layout.addWidget(self.snap_scale_cb, 0, 1)

        # 1: Manual Video
        vid_wid = QWidget()
        vid_layout = QGridLayout(vid_wid)
        vid_layout.setContentsMargins(0,0,0,0)
        self.vid_max_frames = QSpinBox()
        self.vid_max_frames.setRange(0, 1_000_000)
        self.vid_max_frames.setValue(0)
        self.vid_max_frames.setSpecialValueText("Unlimited")
        vid_layout.addWidget(QLabel("Stop after (frames)"), 0, 0)
        vid_layout.addWidget(self.vid_max_frames, 0, 1)

        # 2: Timelapse
        tl_wid = QWidget()
        tl_layout = QGridLayout(tl_wid)
        tl_layout.setContentsMargins(0,0,0,0)
        self.tl_interval = QDoubleSpinBox()
        self.tl_interval.setRange(0.1, 3600.0)
        self.tl_interval.setValue(1.0)
        self.tl_interval.setSuffix(" s")
        self.tl_duration = QSpinBox()
        self.tl_duration.setRange(0, 1_000_000)
        self.tl_duration.setValue(60)
        self.tl_duration.setSuffix(" min")
        self.tl_duration.setSpecialValueText("Infinite")
        tl_layout.addWidget(QLabel("Interval"), 0, 0)
        tl_layout.addWidget(self.tl_interval, 0, 1)
        tl_layout.addWidget(QLabel("Duration"), 1, 0)
        tl_layout.addWidget(self.tl_duration, 1, 1)

        # 3: Scheduled
        sch_wid = QWidget()
        sch_layout = QGridLayout(sch_wid)
        sch_layout.setContentsMargins(0,0,0,0)
        self.sch_start_delay = QSpinBox()
        self.sch_start_delay.setRange(0, 10000)
        self.sch_start_delay.setSuffix(" min")
        sch_layout.addWidget(QLabel("Start delay"), 0, 0)
        sch_layout.addWidget(self.sch_start_delay, 0, 1)

        self.stacked_params.addWidget(snap_wid)   # 0
        self.stacked_params.addWidget(vid_wid)    # 1
        self.stacked_params.addWidget(tl_wid)     # 2
        self.stacked_params.addWidget(sch_wid)    # 3

        mode_grid.addWidget(self.stacked_params, 1, 0, 1, 2)
        layout.addWidget(mode_group)

        # ─── Status & Control ───
        self.disk_space_lbl = QLabel("Est. Disk Space: ~120 MB/min")
        self.disk_space_lbl.setStyleSheet("color: #888;")
        layout.addWidget(self.disk_space_lbl)

        self.action_btn = QPushButton("Take Snapshot")
        self.action_btn.setMinimumHeight(40)
        self.action_btn.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.action_btn)

        layout.addStretch()

        self._connect_ui()

    def _connect_ui(self):
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        
        self.disk_timer = QTimer(self)
        self.disk_timer.timeout.connect(self._update_disk_space)
        self.disk_timer.start(5000)  # Verify every 5 seconds
        
        self._current_dir = ""

    def set_output_dir(self, path: str):
        self._current_dir = path
        self.out_dir_lbl.setText(os.path.basename(path) or path)
        self._update_disk_space()

    def _update_disk_space(self):
        if not self._current_dir or not os.path.exists(self._current_dir):
            return
            
        try:
            total, used, free = shutil.disk_usage(self._current_dir)
            free_gb = free / (1024**3)
            
            # Simple heuristic
            fmt = self.format_combo.currentText()
            fps = 30 if self.mode_combo.currentText() == "Manual Video" else 10
            # Assume 1024x1024 frames roughly: PNG ~1MB, TIFF uncompressed ~4MB, MP4 heavily compressed ~0.1MB
            if fmt == "TIFF Stack": rate_mb = 4.0 * fps * 60
            elif fmt == "Individual PNGs": rate_mb = 1.0 * fps * 60
            else: rate_mb = 0.5 * fps * 60 # Video
            
            self.disk_space_lbl.setText(f"Est: ~{rate_mb:.0f} MB/min | Free: {free_gb:.1f} GB")
        except Exception:
            pass

    def _on_mode_changed(self, idx: int):
        self.stacked_params.setCurrentIndex(idx)
        if idx == 0:
            self.action_btn.setText("Take Snapshot")
            self.action_btn.setCheckable(False)
        elif idx == 1:
            self.action_btn.setText("Start Recording")
            self.action_btn.setCheckable(True)
        elif idx == 2:
            self.action_btn.setText("Start Timelapse")
            self.action_btn.setCheckable(True)
        else:
            self.action_btn.setText("Schedule Job")
            self.action_btn.setCheckable(True)

    def get_state(self) -> dict:
        return {
            "template": self.name_template.text(),
            "format": self.format_combo.currentText(),
            "channel": self.channel_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "snap_color": self.snap_color_cb.isChecked(),
            "snap_scale": self.snap_scale_cb.isChecked(),
            "vid_max_frames": self.vid_max_frames.value(),
            "tl_interval": self.tl_interval.value(),
            "tl_duration": self.tl_duration.value(),
            "sch_start_delay": self.sch_start_delay.value()
        }

    def set_state(self, state: dict):
        if "template" in state: self.name_template.setText(state["template"])
        if "format" in state:
            idx = self.format_combo.findText(state["format"])
            if idx >= 0: self.format_combo.setCurrentIndex(idx)
        if "channel" in state:
            idx = self.channel_combo.findText(state["channel"])
            if idx >= 0: self.channel_combo.setCurrentIndex(idx)
        if "mode" in state:
            idx = self.mode_combo.findText(state["mode"])
            if idx >= 0: self.mode_combo.setCurrentIndex(idx)
        if "snap_color" in state: self.snap_color_cb.setChecked(state["snap_color"])
        if "snap_scale" in state: self.snap_scale_cb.setChecked(state["snap_scale"])
        if "vid_max_frames" in state: self.vid_max_frames.setValue(state["vid_max_frames"])
        if "tl_interval" in state: self.tl_interval.setValue(state["tl_interval"])
        if "tl_duration" in state: self.tl_duration.setValue(state["tl_duration"])
        if "sch_start_delay" in state: self.sch_start_delay.setValue(state["sch_start_delay"])
