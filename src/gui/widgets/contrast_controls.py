"""
Contrast enhancement controls — embeddable QGroupBox widget.

Drop this into process_tab or any sidebar tab:

    from gui.widgets.contrast_controls import ContrastControls
    self.contrast = ContrastControls()
    layout.addWidget(self.contrast)

Then in main_window, read the settings:

    method, params = self.sidebar_tabs.process_tab.contrast.get_contrast_config()
"""
from PySide6.QtWidgets import (
    QGroupBox,
    QGridLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QWidget,
)
from PySide6.QtCore import Signal

from core.contrast import ContrastMethod


class ContrastControls(QGroupBox):
    """
    Collapsible group with:
      - Enable checkbox
      - Method selector (None / Percentile / CLAHE / Histogram Eq)
      - Method-specific parameters
    
    Emits `changed` signal whenever any setting is modified.
    """

    changed = Signal()

    def __init__(self, parent: QWidget = None):
        super().__init__("Display Contrast", parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.setToolTip(
            "Post-processing contrast enhancement for amplitude display.\n"
            "Does NOT affect reconstruction or autofocus — purely visual."
        )

        g = QGridLayout(self)
        g.setSpacing(5)
        row = 0

        # Method
        self.method_combo = QComboBox()
        self.method_combo.addItem("Percentile Stretch", ContrastMethod.PERCENTILE.value)
        self.method_combo.addItem("CLAHE (Adaptive)", ContrastMethod.CLAHE.value)
        self.method_combo.addItem("Histogram Eq.", ContrastMethod.HISTOGRAM_EQ.value)
        self.method_combo.setToolTip(
            "Percentile: clips outliers, stretches to full range (safest)\n"
            "CLAHE: local adaptive equalization (reveals fine detail)\n"
            "Histogram Eq.: global equalization (aggressive)"
        )
        g.addWidget(QLabel("Method"), row, 0)
        g.addWidget(self.method_combo, row, 1); row += 1

        # Percentile params
        self.p_low = QDoubleSpinBox()
        self.p_low.setRange(0.0, 49.0)
        self.p_low.setDecimals(1)
        self.p_low.setValue(1.0)
        self.p_low.setSuffix(" %")
        self.p_low.setToolTip("Lower percentile clip (default 1%)")
        g.addWidget(QLabel("Clip low"), row, 0)
        g.addWidget(self.p_low, row, 1); row += 1

        self.p_high = QDoubleSpinBox()
        self.p_high.setRange(51.0, 100.0)
        self.p_high.setDecimals(1)
        self.p_high.setValue(99.0)
        self.p_high.setSuffix(" %")
        self.p_high.setToolTip("Upper percentile clip (default 99%)")
        g.addWidget(QLabel("Clip high"), row, 0)
        g.addWidget(self.p_high, row, 1); row += 1

        # CLAHE params
        self.clahe_clip = QDoubleSpinBox()
        self.clahe_clip.setRange(0.5, 20.0)
        self.clahe_clip.setDecimals(1)
        self.clahe_clip.setValue(2.0)
        self.clahe_clip.setToolTip("CLAHE clip limit (2 = moderate, 4+ = strong)")
        g.addWidget(QLabel("CLAHE clip"), row, 0)
        g.addWidget(self.clahe_clip, row, 1); row += 1

        self.clahe_grid = QSpinBox()
        self.clahe_grid.setRange(2, 32)
        self.clahe_grid.setValue(8)
        self.clahe_grid.setToolTip("CLAHE tile grid size (NxN)")
        g.addWidget(QLabel("CLAHE grid"), row, 0)
        g.addWidget(self.clahe_grid, row, 1); row += 1

        # Apply to phase too?
        self.apply_phase_cb = QCheckBox("Also apply to phase")
        self.apply_phase_cb.setToolTip("Apply contrast to unwrapped phase display too")
        g.addWidget(self.apply_phase_cb, row, 0, 1, 2); row += 1

        # Connect signals
        self.toggled.connect(self._emit_changed)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.method_combo.currentIndexChanged.connect(self._emit_changed)
        self.p_low.valueChanged.connect(self._emit_changed)
        self.p_high.valueChanged.connect(self._emit_changed)
        self.clahe_clip.valueChanged.connect(self._emit_changed)
        self.clahe_grid.valueChanged.connect(self._emit_changed)
        self.apply_phase_cb.toggled.connect(self._emit_changed)

        # Initial visibility
        self._on_method_changed(0)

    def _emit_changed(self):
        self.changed.emit()

    def _on_method_changed(self, idx: int):
        """Show/hide method-specific parameters."""
        method_val = self.method_combo.currentData()
        is_pct = (method_val == ContrastMethod.PERCENTILE.value)
        is_clahe = (method_val == ContrastMethod.CLAHE.value)

        # Percentile params
        for w in [self.p_low, self.p_high]:
            w.setVisible(is_pct)
        # Find and toggle labels too
        layout = self.layout()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget():
                w = item.widget()
                if isinstance(w, QLabel):
                    txt = w.text()
                    if txt in ("Clip low", "Clip high"):
                        w.setVisible(is_pct)
                    elif txt in ("CLAHE clip", "CLAHE grid"):
                        w.setVisible(is_clahe)

        # CLAHE params
        self.clahe_clip.setVisible(is_clahe)
        self.clahe_grid.setVisible(is_clahe)

    def get_contrast_config(self) -> dict:
        """Returns current settings as a dict for apply_contrast()."""
        if not self.isChecked():
            return {
                "method": ContrastMethod.NONE,
                "p_low": 1.0,
                "p_high": 99.0,
                "clahe_clip": 2.0,
                "clahe_grid": 8,
                "apply_phase": False,
            }
        return {
            "method": ContrastMethod(self.method_combo.currentData()),
            "p_low": self.p_low.value(),
            "p_high": self.p_high.value(),
            "clahe_clip": self.clahe_clip.value(),
            "clahe_grid": self.clahe_grid.value(),
            "apply_phase": self.apply_phase_cb.isChecked(),
        }

    def get_state(self) -> dict:
        return {
            "enabled": self.isChecked(),
            "method": self.method_combo.currentData(),
            "p_low": self.p_low.value(),
            "p_high": self.p_high.value(),
            "clahe_clip": self.clahe_clip.value(),
            "clahe_grid": self.clahe_grid.value(),
            "apply_phase": self.apply_phase_cb.isChecked(),
        }

    def set_state(self, state: dict):
        if "enabled" in state:
            self.setChecked(state["enabled"])
        if "method" in state:
            for i in range(self.method_combo.count()):
                if self.method_combo.itemData(i) == state["method"]:
                    self.method_combo.setCurrentIndex(i)
                    break
        if "p_low" in state:
            self.p_low.setValue(state["p_low"])
        if "p_high" in state:
            self.p_high.setValue(state["p_high"])
        if "clahe_clip" in state:
            self.clahe_clip.setValue(state["clahe_clip"])
        if "clahe_grid" in state:
            self.clahe_grid.setValue(state["clahe_grid"])
        if "apply_phase" in state:
            self.apply_phase_cb.setChecked(state["apply_phase"])
