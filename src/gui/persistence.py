"""Bridge between :class:`core.settings_schema.AppSettings` and sidebar UI.

Split out of ``main_window.py`` so the mapping is testable without spinning
up a full :class:`QMainWindow`. The functions here accept the plain sidebar
tabs (``recon_tab``, ``focus_tab``, ``qpi_tab``) and touch *only* the
widgets that exist — missing widgets are ignored, so upstream UI refactors
don't take settings persistence down with them.

Two entry points mirror the :mod:`gui.settings_store` API:

    * :func:`apply_settings(tabs, settings)` — push stored values into the
      visible widgets. Called once at window construction, after
      ``SidebarTabs`` is instantiated.
    * :func:`collect_settings(tabs, *, io=None)` — snapshot the current UI
      state into a new :class:`AppSettings`. Called just before writing to
      disk, so the on-disk state always matches what the user last ran.

I/O defaults (last folder, last hologram, last report folder) don't live on
any tab; they're passed in explicitly via :class:`IODefaults` so the caller
owns that lifecycle.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Optional

from core.settings_schema import (
    AppSettings,
    AutofocusDefaults,
    IODefaults,
    QPIDefaults,
    ReconDefaults,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# apply — settings → widgets
# ---------------------------------------------------------------------------

def apply_settings(tabs: Any, settings: AppSettings) -> None:
    """Copy ``settings`` values into the matching widgets on ``tabs``.

    ``tabs`` is the :class:`SidebarTabs` instance held by the main window.
    Every attribute access is defensive — if a widget was renamed or
    removed upstream, the write is silently skipped and logged at debug
    level. We never want a rogue UI change to block app startup.
    """
    _apply_recon(getattr(tabs, "recon_tab", None), settings.recon)
    _apply_autofocus(getattr(tabs, "focus_tab", None), settings.autofocus)
    _apply_qpi(getattr(tabs, "qpi_tab", None), settings.qpi)


def _apply_recon(tab: Any, recon: ReconDefaults) -> None:
    if tab is None:
        return
    _set_value(tab, "wavelength_nm", recon.wavelength_nm)
    _set_value(tab, "pixel_um", recon.pixel_um)
    _set_value(tab, "z_mm", recon.z_mm)
    _set_value(tab, "mask_radius", recon.mask_radius)
    # ``n_medium``/``subtract_mean``/``hann_window`` aren't surfaced on the
    # recon tab in the current sidebar; they round-trip through QSettings
    # but aren't user-editable yet. Leaving the schema fields in place
    # means adding those controls later costs nothing here.


def _apply_autofocus(tab: Any, af: AutofocusDefaults) -> None:
    if tab is None:
        return
    _set_value(tab, "zscan_min_mm", af.z_min_mm)
    _set_value(tab, "zscan_max_mm", af.z_max_mm)

    # metric_combo uses the FocusMetric enum value as itemData.
    combo = getattr(tab, "metric_combo", None)
    if combo is not None:
        try:
            idx = combo.findData(af.metric)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        except Exception:  # pragma: no cover - defensive
            _LOG.debug("autofocus metric combo not restored", exc_info=True)

    group = getattr(tab, "adapt_step_group", None)
    if group is not None:
        try:
            group.setChecked(bool(af.adaptive))
        except Exception:  # pragma: no cover
            _LOG.debug("autofocus adapt group not restored", exc_info=True)


def _apply_qpi(tab: Any, qpi: QPIDefaults) -> None:
    if tab is None:
        return
    _set_value(tab, "n_sample", qpi.cell_refractive_index)
    _set_value(tab, "n_medium", qpi.medium_refractive_index)
    # ``phase_offset`` isn't a widget on the current QPI tab; left dormant
    # in the schema for the same reason as the recon tab's unbound fields.


# ---------------------------------------------------------------------------
# collect — widgets → settings
# ---------------------------------------------------------------------------

def collect_settings(
    tabs: Any,
    *,
    io: Optional[IODefaults] = None,
) -> AppSettings:
    """Snapshot the current sidebar state into a fresh :class:`AppSettings`.

    Widgets that don't exist fall back to :meth:`AppSettings.defaults`
    values, so a partially-built window still produces a valid payload.
    The ``io`` argument carries the file-dialog history separately; if
    absent, the defaults are used.
    """
    d = AppSettings.defaults()
    recon = _collect_recon(getattr(tabs, "recon_tab", None), d.recon)
    af = _collect_autofocus(getattr(tabs, "focus_tab", None), d.autofocus)
    qpi = _collect_qpi(getattr(tabs, "qpi_tab", None), d.qpi)
    return AppSettings(
        schema_version=d.schema_version,
        recon=recon,
        autofocus=af,
        qpi=qpi,
        io=io if io is not None else d.io,
    )


def _collect_recon(tab: Any, default: ReconDefaults) -> ReconDefaults:
    if tab is None:
        return default
    return replace(
        default,
        wavelength_nm=_get_value(tab, "wavelength_nm", default.wavelength_nm),
        pixel_um=_get_value(tab, "pixel_um", default.pixel_um),
        z_mm=_get_value(tab, "z_mm", default.z_mm),
        mask_radius=int(_get_value(tab, "mask_radius", default.mask_radius)),
    )


def _collect_autofocus(tab: Any, default: AutofocusDefaults) -> AutofocusDefaults:
    if tab is None:
        return default
    metric = default.metric
    combo = getattr(tab, "metric_combo", None)
    if combo is not None:
        try:
            data = combo.currentData()
            if data:
                metric = str(data)
        except Exception:  # pragma: no cover
            pass

    adaptive = default.adaptive
    group = getattr(tab, "adapt_step_group", None)
    if group is not None:
        try:
            adaptive = bool(group.isChecked())
        except Exception:  # pragma: no cover
            pass

    return replace(
        default,
        z_min_mm=_get_value(tab, "zscan_min_mm", default.z_min_mm),
        z_max_mm=_get_value(tab, "zscan_max_mm", default.z_max_mm),
        metric=metric,
        adaptive=adaptive,
    )


def _collect_qpi(tab: Any, default: QPIDefaults) -> QPIDefaults:
    if tab is None:
        return default
    return replace(
        default,
        cell_refractive_index=_get_value(tab, "n_sample",
                                         default.cell_refractive_index),
        medium_refractive_index=_get_value(tab, "n_medium",
                                           default.medium_refractive_index),
    )


# ---------------------------------------------------------------------------
# Widget helpers — QSpinBox/QDoubleSpinBox duck-typing
# ---------------------------------------------------------------------------

def _get_value(tab: Any, name: str, default: float) -> float:
    w = getattr(tab, name, None)
    if w is None:
        return default
    try:
        return float(w.value())
    except Exception:  # pragma: no cover
        _LOG.debug("widget %s value() failed", name, exc_info=True)
        return default


def _set_value(tab: Any, name: str, value: float) -> None:
    w = getattr(tab, name, None)
    if w is None:
        return
    try:
        w.setValue(value)
    except Exception:  # pragma: no cover
        _LOG.debug("widget %s setValue(%r) failed", name, value, exc_info=True)


__all__ = ["apply_settings", "collect_settings"]
