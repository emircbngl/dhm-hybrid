"""gui/persistence.py — tabs ↔ AppSettings mapping.

The widgets are mocked with plain objects whose attributes expose
``value()``/``setValue()``/``currentData()``/``findData()``/
``setCurrentIndex()``/``isChecked()``/``setChecked()`` contracts used by
the persistence helpers. This keeps the tests free of Qt while still
exercising every branch.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.settings_schema import (
    AppSettings,
    AutofocusDefaults,
    IODefaults,
    QPIDefaults,
    ReconDefaults,
)
from gui.persistence import apply_settings, collect_settings


# ---- mock widget factories --------------------------------------------------

class _SpinBox:
    """Minimal QDoubleSpinBox / QSpinBox double."""
    def __init__(self, initial: float = 0.0) -> None:
        self._v = initial

    def value(self) -> float:
        return self._v

    def setValue(self, v: float) -> None:
        self._v = v


class _Combo:
    """Minimal QComboBox double with itemData/findData/currentData."""
    def __init__(self, items: list[tuple[str, Any]]) -> None:
        # items is list of (label, data)
        self._items = list(items)
        self._idx = 0

    def findData(self, data: Any) -> int:
        for i, (_, d) in enumerate(self._items):
            if d == data:
                return i
        return -1

    def setCurrentIndex(self, idx: int) -> None:
        if 0 <= idx < len(self._items):
            self._idx = idx

    def currentData(self) -> Any:
        return self._items[self._idx][1]


class _Group:
    """Minimal QGroupBox double with checked state."""
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, v: bool) -> None:
        self._checked = bool(v)


def _make_recon_tab(**kwargs):
    return SimpleNamespace(
        wavelength_nm=_SpinBox(kwargs.get("wavelength_nm", 0.0)),
        pixel_um=_SpinBox(kwargs.get("pixel_um", 0.0)),
        z_mm=_SpinBox(kwargs.get("z_mm", 0.0)),
        mask_radius=_SpinBox(kwargs.get("mask_radius", 0)),
    )


def _make_focus_tab(**kwargs):
    combo_items = [
        ("Phase variance", "PHASE_VARIANCE"),
        ("Amplitude variance", "AMPLITUDE_VARIANCE"),
        ("Entropy", "ENTROPY"),
    ]
    combo = _Combo(combo_items)
    if "metric" in kwargs:
        idx = combo.findData(kwargs["metric"])
        if idx >= 0:
            combo.setCurrentIndex(idx)
    return SimpleNamespace(
        zscan_min_mm=_SpinBox(kwargs.get("zscan_min_mm", 0.0)),
        zscan_max_mm=_SpinBox(kwargs.get("zscan_max_mm", 0.0)),
        metric_combo=combo,
        adapt_step_group=_Group(kwargs.get("adaptive", False)),
    )


def _make_qpi_tab(**kwargs):
    return SimpleNamespace(
        n_sample=_SpinBox(kwargs.get("n_sample", 0.0)),
        n_medium=_SpinBox(kwargs.get("n_medium", 0.0)),
    )


def _tabs(**overrides):
    return SimpleNamespace(
        recon_tab=overrides.get("recon_tab", _make_recon_tab()),
        focus_tab=overrides.get("focus_tab", _make_focus_tab()),
        qpi_tab=overrides.get("qpi_tab", _make_qpi_tab()),
    )


# ---- apply_settings ---------------------------------------------------------

def test_apply_copies_recon_fields_into_widgets():
    tabs = _tabs()
    settings = AppSettings.defaults()
    settings = settings.with_recon(wavelength_nm=488.0, pixel_um=2.4,
                                   z_mm=12.5, mask_radius=64)

    apply_settings(tabs, settings)

    assert tabs.recon_tab.wavelength_nm.value() == 488.0
    assert tabs.recon_tab.pixel_um.value() == 2.4
    assert tabs.recon_tab.z_mm.value() == 12.5
    assert tabs.recon_tab.mask_radius.value() == 64


def test_apply_restores_autofocus_metric_and_adaptive_flag():
    tabs = _tabs()
    settings = AppSettings.defaults().with_autofocus(
        z_min_mm=-5.0, z_max_mm=5.0, metric="ENTROPY", adaptive=True,
    )

    apply_settings(tabs, settings)

    assert tabs.focus_tab.zscan_min_mm.value() == -5.0
    assert tabs.focus_tab.zscan_max_mm.value() == 5.0
    assert tabs.focus_tab.metric_combo.currentData() == "ENTROPY"
    assert tabs.focus_tab.adapt_step_group.isChecked() is True


def test_apply_copies_qpi_refractive_indices():
    tabs = _tabs()
    settings = AppSettings.defaults().with_qpi(
        cell_refractive_index=1.42, medium_refractive_index=1.34,
    )

    apply_settings(tabs, settings)

    assert tabs.qpi_tab.n_sample.value() == pytest.approx(1.42)
    assert tabs.qpi_tab.n_medium.value() == pytest.approx(1.34)


def test_apply_tolerates_missing_widgets():
    """Rename-safe: if a widget is gone, apply just skips it."""
    # Tab with every expected widget *except* z_mm.
    recon_tab = SimpleNamespace(
        wavelength_nm=_SpinBox(),
        pixel_um=_SpinBox(),
        mask_radius=_SpinBox(),
    )
    tabs = _tabs(recon_tab=recon_tab)
    settings = AppSettings.defaults()

    # Must not raise.
    apply_settings(tabs, settings)

    # The widgets that exist still receive their value.
    assert tabs.recon_tab.wavelength_nm.value() == settings.recon.wavelength_nm


def test_apply_with_none_tabs_is_noop():
    # tabs without any of the expected tab attributes
    apply_settings(SimpleNamespace(), AppSettings.defaults())  # no crash


# ---- collect_settings -------------------------------------------------------

def test_collect_snapshots_current_widget_state():
    tabs = _tabs(
        recon_tab=_make_recon_tab(wavelength_nm=405.0, pixel_um=5.5,
                                   z_mm=-2.0, mask_radius=120),
        focus_tab=_make_focus_tab(zscan_min_mm=-10.0, zscan_max_mm=10.0,
                                   metric="AMPLITUDE_VARIANCE",
                                   adaptive=False),
        qpi_tab=_make_qpi_tab(n_sample=1.39, n_medium=1.33),
    )

    snap = collect_settings(tabs)

    assert snap.recon.wavelength_nm == 405.0
    assert snap.recon.pixel_um == 5.5
    assert snap.recon.z_mm == -2.0
    assert snap.recon.mask_radius == 120
    assert snap.autofocus.z_min_mm == -10.0
    assert snap.autofocus.z_max_mm == 10.0
    assert snap.autofocus.metric == "AMPLITUDE_VARIANCE"
    assert snap.autofocus.adaptive is False
    assert snap.qpi.cell_refractive_index == pytest.approx(1.39)
    assert snap.qpi.medium_refractive_index == pytest.approx(1.33)


def test_collect_preserves_supplied_io_history():
    tabs = _tabs()
    io = IODefaults(last_folder="/tmp/x", last_hologram="/tmp/x/frame.tif",
                    last_report_folder="/srv/reports", last_preset="default")

    snap = collect_settings(tabs, io=io)

    assert snap.io.last_folder == "/tmp/x"
    assert snap.io.last_hologram == "/tmp/x/frame.tif"
    assert snap.io.last_report_folder == "/srv/reports"
    assert snap.io.last_preset == "default"


def test_collect_without_tabs_returns_defaults():
    snap = collect_settings(SimpleNamespace())

    defaults = AppSettings.defaults()
    assert snap.recon == defaults.recon
    assert snap.autofocus == defaults.autofocus
    assert snap.qpi == defaults.qpi


# ---- round-trip: apply → collect should recover identical values ------------

def test_apply_then_collect_roundtrip():
    tabs = _tabs()
    original = AppSettings(
        recon=ReconDefaults(wavelength_nm=632.8, pixel_um=3.45, z_mm=44.0,
                            n_medium=1.0, mask_radius=80,
                            subtract_mean=True, hann_window=False),
        autofocus=AutofocusDefaults(z_min_mm=-2.0, z_max_mm=2.0,
                                     metric="PHASE_VARIANCE", adaptive=True),
        qpi=QPIDefaults(cell_refractive_index=1.38,
                        medium_refractive_index=1.335,
                        phase_offset=0.0),
        io=IODefaults(last_folder="/lab/data"),
    )

    apply_settings(tabs, original)
    roundtrip = collect_settings(tabs, io=original.io)

    # The schema has three "bound but not UI-surfaced" fields
    # (n_medium, subtract_mean, hann_window, phase_offset). Those stay at
    # defaults after a round-trip through the widgets. Everything else
    # must survive intact.
    assert roundtrip.recon.wavelength_nm == original.recon.wavelength_nm
    assert roundtrip.recon.pixel_um == original.recon.pixel_um
    assert roundtrip.recon.z_mm == original.recon.z_mm
    assert roundtrip.recon.mask_radius == original.recon.mask_radius
    assert roundtrip.autofocus == original.autofocus
    assert roundtrip.qpi.cell_refractive_index == original.qpi.cell_refractive_index
    assert roundtrip.qpi.medium_refractive_index == original.qpi.medium_refractive_index
    assert roundtrip.io == original.io
