from pathlib import Path
from PySide6.QtWidgets import (
    QToolBar,
    QToolButton,
    QFileDialog,
    QMessageBox,
    QWidget,
    QComboBox,
    QLabel,
    QLineEdit,
    QMenu,
    QWidgetAction,
    QHBoxLayout,
    QDoubleSpinBox,
    QSizePolicy,
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Signal, Qt
from .profile_combo import ProfileComboBox


class MainToolbar(QToolBar):
    """Top toolbar handling loading, batches, exports, and recording/snapshots."""

    # Emitted when the user asks to load a file
    file_load_requested = Signal(Path)

    # Emitted when batch UI should be shown
    batch_show_requested = Signal()

    # Tools signals
    crop_toggled = Signal(bool)
    scalebar_toggled = Signal(bool)   # enabled/disabled
    crosshair_toggled = Signal(bool)
    line_profile_toggled = Signal(bool)
    export_view_requested = Signal()
    reconstruct_requested = Signal()

    def __init__(self, parent: QWidget = None):
        super().__init__("Main Toolbar", parent)
        self.setObjectName("main_toolbar")

        # Mode switch
        self.addWidget(QLabel(" Mode: "))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["File", "Live"])
        # v1.4 accessibility — every control has a spoken name so
        # screen-readers announce "Data source, combo box, File" etc.
        self.mode_combo.setAccessibleName("Data source mode")
        self.mode_combo.setToolTip(
            "Data source — File loads a hologram from disk, "
            "Live streams from the connected camera."
        )
        self.addWidget(self.mode_combo)

        self.addSeparator()

        # Load File
        self.action_load = QAction("Load File", self)
        self.action_load.setToolTip("Open a hologram image (TIFF, PNG, …)")
        self.action_load.setStatusTip(
            "Load a hologram from disk for reconstruction."
        )
        self.action_load.triggered.connect(self._on_load_clicked)
        self.addAction(self.action_load)

        self.addSeparator()

        # Sample ID — optional tag that lands in every audit record so the
        # lab's LIMS can stitch reconstruction output back to a specimen.
        # Empty string means "not tagged" and is omitted from the audit
        # context automatically.
        self.addWidget(QLabel(" Sample ID: "))
        self.sample_id_edit = QLineEdit()
        self.sample_id_edit.setPlaceholderText("optional (LIMS)")
        self.sample_id_edit.setMaximumWidth(140)
        self.sample_id_edit.setAccessibleName("Sample identifier (LIMS)")
        self.sample_id_edit.setToolTip(
            "Optional sample identifier. Appended to every audit log entry "
            "so external systems (LIMS, ELN) can correlate."
        )
        self.addWidget(self.sample_id_edit)

        self.addSeparator()

        # Profiles
        self.addWidget(QLabel(" Setup Profile: "))
        self.setup_profile_combo = ProfileComboBox()
        self.addWidget(self.setup_profile_combo)

        self.addWidget(QLabel(" Cam Profile: "))
        self.cam_profile_combo = ProfileComboBox()
        self.addWidget(self.cam_profile_combo)

        self.addSeparator()

        # Batch Mode
        self.action_batch = QAction("Batch Mode", self)
        self.action_batch.triggered.connect(self.batch_show_requested.emit)
        self.addAction(self.action_batch)

        self.addSeparator()

        # ── Tools menu ──
        self._tools_menu = QMenu("Tools", self)

        self.action_crop = QAction("Crop ROI", self)
        self.action_crop.setCheckable(True)
        self.action_crop.setToolTip("Toggle crop rectangle on the active panel")
        self.action_crop.toggled.connect(self.crop_toggled.emit)
        self._tools_menu.addAction(self.action_crop)

        self.action_scalebar = QAction("Scale Bar", self)
        self.action_scalebar.setCheckable(True)
        self.action_scalebar.setToolTip("Show / hide scale bar overlay")
        self.action_scalebar.toggled.connect(self.scalebar_toggled.emit)
        self._tools_menu.addAction(self.action_scalebar)

        # Scale bar length widget (inline in menu)
        sb_widget = QWidget()
        sb_layout = QHBoxLayout(sb_widget)
        sb_layout.setContentsMargins(20, 2, 8, 2)
        sb_layout.addWidget(QLabel("Length:"))
        self.scalebar_um = QDoubleSpinBox()
        self.scalebar_um.setRange(0, 10000)
        self.scalebar_um.setDecimals(1)
        self.scalebar_um.setValue(0)
        self.scalebar_um.setSuffix(" µm")
        self.scalebar_um.setSpecialValueText("Auto")
        self.scalebar_um.setToolTip("Scale bar length in µm (0 = auto)")
        self.scalebar_um.valueChanged.connect(self._on_scalebar_length_changed)
        sb_layout.addWidget(self.scalebar_um)
        sb_action = QWidgetAction(self)
        sb_action.setDefaultWidget(sb_widget)
        self._tools_menu.addAction(sb_action)

        self.action_crosshair = QAction("Crosshair", self)
        self.action_crosshair.setCheckable(True)
        self.action_crosshair.setToolTip("Show crosshair with pixel coordinates")
        self.action_crosshair.toggled.connect(self.crosshair_toggled.emit)
        self._tools_menu.addAction(self.action_crosshair)

        self.action_line_profile = QAction("Line Profile", self)
        self.action_line_profile.setCheckable(True)
        self.action_line_profile.setToolTip("Draw a line and see intensity profile along it")
        self.action_line_profile.toggled.connect(self.line_profile_toggled.emit)
        self._tools_menu.addAction(self.action_line_profile)

        self._tools_menu.addSeparator()

        self.action_export_view = QAction("Export Current View...", self)
        self.action_export_view.setToolTip("Export the current grid view as an image")
        self.action_export_view.triggered.connect(self.export_view_requested.emit)
        self._tools_menu.addAction(self.action_export_view)

        tools_btn = QToolButton(self)
        tools_btn.setText("Tools")
        tools_btn.setMenu(self._tools_menu)
        tools_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.addWidget(tools_btn)

        # Spacer pushes Reconstruct to the far right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)

        # Reconstruct (always visible, works from any tab)
        self.action_reconstruct = QAction("⬢ Reconstruct", self)
        self.action_reconstruct.setToolTip("Run reconstruction with current parameters (Ctrl+R)")
        self.action_reconstruct.setStatusTip(
            "Reconstruct the loaded hologram at the current z (Ctrl+R)."
        )
        self.action_reconstruct.setEnabled(False)
        self.action_reconstruct.triggered.connect(self.reconstruct_requested.emit)
        self.addAction(self.action_reconstruct)

    def _on_scalebar_length_changed(self, val: float) -> None:
        """Re-trigger scalebar if it's currently active."""
        if self.action_scalebar.isChecked():
            # Toggle off then on to refresh with new length
            self.action_scalebar.setChecked(False)
            self.action_scalebar.setChecked(True)

    def _on_load_clicked(self) -> None:
        """Open a file dialog and emit the selected path."""
        # ``default_dir`` is set by the main window from persisted I/O history,
        # so the dialog reopens where the user left off. Empty string → $HOME.
        start_dir = getattr(self, "default_dir", "") or ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Hologram",
            start_dir,
            "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.h5 *.nd2 *.czi);;All Files (*)",
        )
        if file_path:
            self.file_load_requested.emit(Path(file_path))
