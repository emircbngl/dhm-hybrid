from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QGroupBox, QListWidget, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QWidget
)
from PySide6.QtCore import Qt, Signal
from pathlib import Path

class BatchRenderDialog(QDialog):
    """
    Standalone dialog configuring and dispatching asynchronous batch reconstruction pipelines.
    Supports Single-Profile iterations and experimental Parameter Sweeping.
    """
    # Emits dict with keys: input_dir, output_dir, mode, profiles, etc.
    start_batch_requested = Signal(dict)

    def __init__(self, parent=None, profile_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Render")
        self.setMinimumWidth(500)
        self._profile_manager = profile_manager
        
        layout = QVBoxLayout(self)

        # ─── I/O Selection ───
        io_group = QGroupBox("Directories")
        io_layout = QVBoxLayout(io_group)
        
        # Input
        in_layout = QHBoxLayout()
        self.in_txt = QLineEdit()
        self.in_txt.setPlaceholderText("Select directory containing raw holograms (.tiff, .png, etc)")
        in_btn = QPushButton("Browse")
        in_btn.clicked.connect(self._browse_in)
        in_layout.addWidget(QLabel("Input:"))
        in_layout.addWidget(self.in_txt)
        in_layout.addWidget(in_btn)
        io_layout.addLayout(in_layout)
        
        # Output
        out_layout = QHBoxLayout()
        self.out_txt = QLineEdit()
        self.out_txt.setPlaceholderText("Select output destination directory")
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(self._browse_out)
        out_layout.addWidget(QLabel("Output:"))
        out_layout.addWidget(self.out_txt)
        out_layout.addWidget(out_btn)
        io_layout.addLayout(out_layout)
        
        layout.addWidget(io_group)

        # ─── Strategy ───
        strat_group = QGroupBox("Execution Strategy")
        strat_layout = QVBoxLayout(strat_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Iterate files with active Profile",
            "Iterate files with selected Profiles",
            "Parameter Sweep (Z-stack)",
            "Auto-Focus (Best Z)"
        ])
        strat_layout.addWidget(self.mode_combo)
        
        # Profile list (only visible in multi-profile mode)
        self.profile_list = QListWidget()
        self.profile_list.setSelectionMode(QListWidget.MultiSelection)
        self.profile_list.setToolTip("Command/Ctrl + Click to select multiple profiles")
        strat_layout.addWidget(self.profile_list)
        
        # Z-Stack parameters (grid layout for cleaner look)
        z_grid = QGridLayout()

        self.z_start = QDoubleSpinBox()
        self.z_start.setRange(-500.0, 500.0)
        self.z_start.setDecimals(3)
        self.z_start.setValue(0.0)
        self.z_start.setSuffix(" mm")
        self.z_end = QDoubleSpinBox()
        self.z_end.setRange(-500.0, 500.0)
        self.z_end.setDecimals(3)
        self.z_end.setValue(1.0)
        self.z_end.setSuffix(" mm")
        self.z_steps = QSpinBox()
        self.z_steps.setRange(2, 1000)
        self.z_steps.setValue(51)

        z_grid.addWidget(QLabel("Start Z:"), 0, 0)
        z_grid.addWidget(self.z_start, 0, 1)
        z_grid.addWidget(QLabel("End Z:"), 0, 2)
        z_grid.addWidget(self.z_end, 0, 3)
        z_grid.addWidget(QLabel("Steps:"), 0, 4)
        z_grid.addWidget(self.z_steps, 0, 5)

        self.focus_metric_combo = QComboBox()
        self.focus_metric_combo.addItems([
            "tenengrad",
            "laplacian_variance",
            "total_variation",
            "gradient",
            "brenner",
            "entropy",
            "vollath_f4",
            "vollath_f5",
            "normalized_variance",
            "spectral_energy",
        ])
        self.focus_metric_combo.setToolTip("Focus quality metric used to score each Z-plane")

        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "Robust Coarse-to-Fine",
            "Golden Section",
            "Coarse-to-Fine (Golden Section)",
            "Linear Sweep (Exhaustive)",
        ])
        self.algo_combo.setToolTip(
            "Algorithm used to search for best focus.\n"
            "Robust and Golden Section are faster but may miss secondary peaks.\n"
            "Linear Sweep evaluates every step (slowest, most thorough)."
        )

        z_grid.addWidget(QLabel("Metric:"), 1, 0)
        z_grid.addWidget(self.focus_metric_combo, 1, 1, 1, 2)
        z_grid.addWidget(QLabel("Algorithm:"), 1, 3)
        z_grid.addWidget(self.algo_combo, 1, 4, 1, 2)

        self.z_container = QWidget()
        self.z_container.setLayout(z_grid)
        strat_layout.addWidget(self.z_container)
        
        layout.addWidget(strat_group)
        
        # ─── Export Metrics ───
        metrics_group = QGroupBox("Export & Reports")
        m_layout = QVBoxLayout(metrics_group)
        self.export_csv = QCheckBox("Export Focus Scores (focus_metrics.csv)")
        self.export_csv.setChecked(True)
        self.export_report = QCheckBox("Generate Reconstruction Report (.txt summary)")
        self.export_report.setChecked(True)
        m_layout.addWidget(self.export_csv)
        m_layout.addWidget(self.export_report)
        layout.addWidget(metrics_group)

        # ─── Outputs to Save ───
        outputs_group = QGroupBox("Outputs to Save")
        out_box = QHBoxLayout(outputs_group)
        self.save_amp = QCheckBox("Amplitude")
        self.save_amp.setChecked(True)
        self.save_pha = QCheckBox("Phase")
        self.save_pha.setChecked(True)
        self.save_unwrapped_pha = QCheckBox("Unwrapped Phase")
        self.save_real = QCheckBox("Real")
        self.save_imag = QCheckBox("Imaginary")
        out_box.addWidget(self.save_amp)
        out_box.addWidget(self.save_pha)
        out_box.addWidget(self.save_unwrapped_pha)
        out_box.addWidget(self.save_real)
        out_box.addWidget(self.save_imag)
        layout.addWidget(outputs_group)

        # ─── Controls ───
        ctrl_layout = QVBoxLayout()
        
        prog_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.eta_lbl = QLabel("ETA: --:--")
        prog_layout.addWidget(self.progress)
        prog_layout.addWidget(self.eta_lbl)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Batch")
        self.start_btn.setStyleSheet("font-weight: bold;")
        self.start_btn.clicked.connect(self._on_start)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setEnabled(True)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        
        ctrl_layout.addLayout(prog_layout)
        ctrl_layout.addLayout(btn_layout)
        
        layout.addLayout(ctrl_layout)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._populate_profiles()
        self._on_mode_changed(0)

    def _browse_in(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path: self.in_txt.setText(path)

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path: self.out_txt.setText(path)

    def _populate_profiles(self):
        self.profile_list.clear()
        if self._profile_manager:
            profs = self._profile_manager.list_profiles("setup")
            self.profile_list.addItems(["Default"] + profs)

    def _on_mode_changed(self, idx: int):
        self.profile_list.setVisible(idx == 1)
        self.z_container.setVisible(idx in (2, 3))

    def set_progress(self, val: int, maximum: int = 100):
        self.progress.setMaximum(maximum)
        self.progress.setValue(val)

    def set_eta(self, eta_str: str):
        self.eta_lbl.setText(f"ETA: {eta_str}")

    def set_status(self, text: str):
        self.setWindowTitle(f"Batch Render - {text}")

    def _on_start(self):
        active_state = {}
        if hasattr(self.parent(), "_get_profile_state"):
            active_state = self.parent()._get_profile_state("setup")
            
        cfg = {
            "input_dir": self.in_txt.text(),
            "output_dir": self.out_txt.text(),
            "mode": self.mode_combo.currentText(),
            "export_csv": self.export_csv.isChecked(),
            "export_report": self.export_report.isChecked(),
            "profiles": [item.text() for item in self.profile_list.selectedItems()],
            "z_start": self.z_start.value(),
            "z_end": self.z_end.value(),
            "z_steps": self.z_steps.value(),
            "focus_metric": self.focus_metric_combo.currentText(),
            "algorithm": self.algo_combo.currentText(),
            "save_amp": self.save_amp.isChecked(),
            "save_pha": self.save_pha.isChecked(),
            "save_unwrapped_pha": self.save_unwrapped_pha.isChecked(),
            "save_real": self.save_real.isChecked(),
            "save_imag": self.save_imag.isChecked(),
            "active_state": active_state
        }
        self.start_batch_requested.emit(cfg)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setText("Stop")
