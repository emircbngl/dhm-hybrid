"""Persistent user presets for ui3 — a Qt-free store + two thin dialogs.

``ReconPanel`` (``ui3/panels/recon_panel.py``) currently keeps user presets in
a panel-local ``dict`` — explicitly out of scope for persistence per that
panel's own docstring ("kalici depolama ctx disi ... asma"). ``PresetStore``
is that persistence layer, split out so it can be unit-tested without Qt and
reused by any panel: ``PresetStore.list()`` / ``save(name, params_dict)`` /
``delete(name)`` / ``get(name)``.

Storage: ``~/.dhm-reconstruction/ui3_presets.json`` — a sibling of
``ui3_state.json`` (see ``ui3/state.py``), same atomic tempfile+os.replace
write pattern so a crash mid-write can't corrupt the file.

``SavePresetDialog`` / ``DeletePresetDialog`` are the Qt front ends: simple
``QDialog`` + ``QLineEdit`` / ``QListWidget``. They operate on a
``PresetStore`` instance directly (no ``PanelContext`` dependency — a preset
is a named subset of ``ReconParams`` fields, not a bridge/status concern), so
a hosting panel wires them to its own ``ctx.get_params()`` / ``ctx.toast``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui3.design import Space

# Subset of ReconParams fields a preset captures — mirrors the fields
# ReconPanel._on_save_preset_clicked already hand-picks (physics + a couple
# of workflow knobs), so a preset saved here is drop-in compatible with the
# panel's existing built-in preset dicts.
PRESET_FIELDS = (
    "wavelength_nm", "pixel_um", "z_mm", "mask_radius", "method",
    "magnification", "pixel_is_effective", "n_sample", "n_medium",
    "autofocus_metric",
)


def default_presets_path() -> Path:
    root = Path(os.path.expanduser("~")) / ".dhm-reconstruction"
    return root / "ui3_presets.json"


class PresetStore:
    """Qt-free, dict-backed preset store persisted to a JSON file.

    Every mutating call (``save``/``delete``) writes through to disk
    immediately (atomic tempfile + ``os.replace``, same as
    ``ui3.state.save_state``) so callers never need a separate "flush"
    step. Load failures (missing/corrupt file) fall back to an empty store
    rather than raising — presets are a convenience, never a hard
    dependency for the app to start.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or default_presets_path()
        self._data: Dict[str, Dict[str, Any]] = self._load()

    # -- persistence ------------------------------------------------
    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return {}

    def _flush(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        except OSError:
            pass  # persistence must never crash the app

    # -- public API ---------------------------------------------------
    def list(self) -> List[str]:
        """Preset names, alphabetically."""
        return sorted(self._data.keys())

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        val = self._data.get(str(name))
        return dict(val) if val is not None else None

    def save(self, name: str, params_dict: Dict[str, Any]) -> None:
        """Store ``name`` -> the subset of ``params_dict`` on
        ``PRESET_FIELDS`` (unknown keys are dropped so the file stays a
        stable, portable shape)."""
        name = str(name).strip()
        if not name:
            raise ValueError("preset name must be non-empty")
        clean = {k: params_dict[k] for k in PRESET_FIELDS if k in params_dict}
        self._data[name] = clean
        self._flush()

    def delete(self, name: str) -> bool:
        existed = self._data.pop(str(name), None) is not None
        if existed:
            self._flush()
        return existed

    def reload(self) -> None:
        """Re-read from disk, discarding any in-memory changes not yet
        flushed (there shouldn't be any — save/delete flush immediately)."""
        self._data = self._load()


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

class SavePresetDialog(QDialog):
    """Prompt for a preset name, then save the given params dict into
    ``store``. Warns before silently overwriting an existing name."""

    def __init__(self, store: PresetStore, params_dict: Dict[str, Any],
                 *, reserved_names: tuple = (),
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._store = store
        self._params_dict = params_dict
        self._reserved = tuple(reserved_names)
        self.saved_name: Optional[str] = None

        self.setWindowTitle("Save preset")
        root = QVBoxLayout(self)
        root.setContentsMargins(Space.lg, Space.lg, Space.lg, Space.lg)
        root.setSpacing(Space.sm)

        root.addWidget(QLabel("Preset name:"))
        self.edit_name = QLineEdit()
        self.edit_name.returnPressed.connect(self._on_accept)
        root.addWidget(self.edit_name)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).clicked.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_accept(self) -> None:
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Save preset", "Enter a preset name.")
            return
        if name in self._reserved:
            QMessageBox.warning(
                self, "Save preset",
                f"'{name}' is a built-in name — pick something else.")
            return
        if name in self._store.list():
            reply = QMessageBox.question(
                self, "Replace preset",
                f"A preset named '{name}' already exists. Replace it?")
            if reply != QMessageBox.Yes:
                return
        self._store.save(name, self._params_dict)
        self.saved_name = name
        self.accept()


class DeletePresetDialog(QDialog):
    """List user presets in ``store``; delete the selected one on confirm."""

    def __init__(self, store: PresetStore, *, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._store = store
        self.deleted_name: Optional[str] = None

        self.setWindowTitle("Delete preset")
        root = QVBoxLayout(self)
        root.setContentsMargins(Space.lg, Space.lg, Space.lg, Space.lg)
        root.setSpacing(Space.sm)

        root.addWidget(QLabel("Presets:"))
        self.list_widget = QListWidget()
        self.list_widget.addItems(self._store.list())
        root.addWidget(self.list_widget, 1)

        btn_row = QHBoxLayout()
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setProperty("role", "danger")
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_delete)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    def _on_delete(self) -> None:
        item: Optional[QListWidgetItem] = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "Delete preset", "Select a preset first.")
            return
        name = item.text()
        reply = QMessageBox.question(
            self, "Delete preset", f"Delete preset '{name}'? This cannot be undone.")
        if reply != QMessageBox.Yes:
            return
        self._store.delete(name)
        self.deleted_name = name
        row = self.list_widget.row(item)
        self.list_widget.takeItem(row)


__all__ = [
    "PresetStore",
    "PRESET_FIELDS",
    "SavePresetDialog",
    "DeletePresetDialog",
    "default_presets_path",
]
