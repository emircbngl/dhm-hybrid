"""Global stylesheet for the DHM Reconstruction application."""

STYLESHEET = """
/* ─── Base ─── */
QMainWindow, QWidget {
    font-family: "SF Pro Text", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}

/* ─── GroupBox ─── */
QGroupBox {
    font-weight: 600;
    font-size: 12px;
    border: 1px solid palette(mid);
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 6px 6px 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}
QGroupBox::indicator {
    width: 14px;
    height: 14px;
}

/* ─── Tab bar ─── */
QTabWidget::pane {
    border: 1px solid palette(mid);
    border-radius: 4px;
}
QTabBar::tab {
    padding: 5px 12px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 12px;
    font-weight: 500;
}
QTabBar::tab:selected {
    font-weight: 600;
}

/* ─── Buttons ─── */
QPushButton {
    padding: 5px 14px;
    border-radius: 5px;
    font-weight: 500;
    min-height: 24px;
}
QPushButton:disabled {
    opacity: 0.5;
}

/* ─── SpinBoxes ─── */
QDoubleSpinBox, QSpinBox {
    padding: 3px 6px;
    border-radius: 4px;
    border: 1px solid palette(mid);
    min-height: 22px;
}

/* ─── ComboBox ─── */
QComboBox {
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid palette(mid);
    min-height: 22px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}

/* ─── Labels inside grids ─── */
QGridLayout > QLabel {
    font-size: 12px;
}

/* ─── CheckBox ─── */
QCheckBox {
    spacing: 6px;
    font-size: 12px;
}

/* ─── ScrollArea ─── */
QScrollArea {
    border: none;
}

/* ─── Status bar ─── */
QStatusBar {
    font-size: 11px;
}

/* ─── ToolBar ─── */
QToolBar {
    spacing: 4px;
    padding: 2px;
}
"""
