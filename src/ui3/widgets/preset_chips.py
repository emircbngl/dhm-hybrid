"""PresetChipRow — horizontal exclusive toggle-button row (ui3).

Qt port of ``ui2.widgets.PresetChips`` (a DPG button group with a fake
"disabled = selected" highlight). Here a real ``QButtonGroup`` in exclusive
mode gives genuine toggle-button semantics: one chip stays checked/pressed at
all times once a selection exists, styled via the ``primary`` role token
(same accent used for ``QPushButton[role="primary"]`` in ``ui3.design``) so
the active chip reads as "live" the instrument-aesthetic way, not a generic
grey/blue OS toggle.

``rebuild(names)`` swaps the label set at runtime (e.g. after a preset is
saved/deleted) while preserving the active selection when the new set still
contains it — same behaviour ``PresetChips.rebuild`` documents.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget
from PySide6.QtCore import Signal

from ui3.design import Space


class PresetChipRow(QWidget):
    """A row of exclusive toggle buttons, one per preset name.

    ``selected`` fires with the chip's label whenever the active chip
    changes (including programmatic changes via :meth:`set_active`).
    """

    selected = Signal(str)

    def __init__(self, names: Sequence[str] = (),
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._labels: List[str] = []
        self._buttons: List[QPushButton] = []
        self._active: Optional[str] = None

        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(Space.xs)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonClicked.connect(self._on_clicked)

        self.rebuild(names)

    # ------------------------------------------------------------------
    def rebuild(self, names: Sequence[str]) -> None:
        """Replace the chip set. Preserves the active selection if the
        new set still contains it; otherwise selects the first chip (or
        clears selection if ``names`` is empty)."""
        prev_active = self._active
        for btn in self._buttons:
            self._group.removeButton(btn)
            self._lay.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()
        self._labels = list(names)

        for label in self._labels:
            btn = QPushButton(label)
            btn.setCheckable(True)
            self._group.addButton(btn)
            self._lay.addWidget(btn)
            self._buttons.append(btn)

        if prev_active and prev_active in self._labels:
            self.set_active(prev_active, emit=False)
        elif self._labels:
            self.set_active(self._labels[0], emit=False)
        else:
            self._active = None

    def set_active(self, label: str, *, emit: bool = True) -> None:
        """Programmatically select ``label`` (no-op if not present)."""
        if label not in self._labels:
            return
        idx = self._labels.index(label)
        self._buttons[idx].setChecked(True)
        self._active = label
        if emit:
            self.selected.emit(label)

    def active(self) -> Optional[str]:
        return self._active

    def labels(self) -> List[str]:
        return list(self._labels)

    # ------------------------------------------------------------------
    def _on_clicked(self, btn: QPushButton) -> None:
        idx = self._buttons.index(btn)
        label = self._labels[idx]
        self._active = label
        self.selected.emit(label)


__all__ = ["PresetChipRow"]
