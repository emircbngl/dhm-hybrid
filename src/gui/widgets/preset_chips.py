"""Preset chip row (v1.4 UI Redesign).

One toggle button per preset laid out in a horizontal row. Clicking a
chip emits :attr:`preset_selected` with the chip's label; the host
then applies the preset to its widgets (existing ``_apply_preset``
handler on each tab).

Why not keep the QComboBox? Chips are one click instead of two
(open → select), visually show the active choice at a glance, and
read better in the redesigned sidebar. Falls back gracefully when
the host wants to keep the combo box — the chip row is additive,
the combo can stay hidden or coexist.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)


class PresetChipRow(QWidget):
    """Exclusive-select chip row. One preset active at a time."""

    preset_selected = Signal(str)  # chip label

    def __init__(
        self,
        presets: Iterable[str],
        parent: Optional[QWidget] = None,
        *,
        active: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: List[QPushButton] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        for label in presets:
            btn = QPushButton(label, self)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            btn.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed,
            )
            btn.setAccessibleName(f"Preset: {label}")
            # A full QSS override on a native macOS QPushButton triggers
            # a paint-engine crash on Qt 6.8+ (``QPainter::begin: Paint
            # device returned engine == 0``). The button stays in its
            # platform style; ``setCheckable`` already paints a
            # distinguishable pressed state.
            btn.clicked.connect(
                lambda _checked=False, l=label: self._on_clicked(l)
            )
            self._group.addButton(btn)
            self._buttons.append(btn)
            layout.addWidget(btn)
        layout.addStretch(1)

        if active is not None:
            self.setActive(active)
        elif self._buttons:
            self._buttons[0].setChecked(True)

    # ---- public API ----------------------------------------------------

    def presets(self) -> List[str]:
        return [b.text() for b in self._buttons]

    def active(self) -> Optional[str]:
        for b in self._buttons:
            if b.isChecked():
                return b.text()
        return None

    def setActive(self, label: str) -> None:
        """Mark ``label`` as the active preset without re-emitting the
        selection signal (used by hosts that programmatically sync
        the chip row with another control)."""
        for b in self._buttons:
            if b.text() == label:
                b.blockSignals(True)
                try:
                    b.setChecked(True)
                finally:
                    b.blockSignals(False)
                return

    # ---- internal ------------------------------------------------------

    def _on_clicked(self, label: str) -> None:
        self.preset_selected.emit(label)


__all__ = ["PresetChipRow"]
