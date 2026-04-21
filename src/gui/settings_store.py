"""Qt-backed persistence for :class:`core.settings_schema.AppSettings`.

Uses ``QSettings("DHM", "Reconstruction")`` as the backend. Window geometry
and dock state are intentionally kept in the same QSettings instance — they
were there in v1.0.0 and nothing in this module touches them, so existing
layout memory survives the v1 → v2 migration untouched.

Entry points:
    * ``load()``  — return an AppSettings; invalid on-disk state falls back
                    to defaults after logging.
    * ``save(s)`` — write an AppSettings to disk.
    * ``update(**kw)`` — convenience: merge a few fields and save.

Migration chain:
    v1 (no schema_version key) → v2 (current): nothing to translate, just
    write the new defaults. Future versions should append to ``_MIGRATIONS``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from PySide6.QtCore import QSettings

from core.settings_schema import (
    SCHEMA_VERSION,
    AppSettings,
    AutofocusDefaults,
    IODefaults,
    QPIDefaults,
    ReconDefaults,
    validate,
)

_LOG = logging.getLogger(__name__)

_ORG = "DHM"
_APP = "Reconstruction"
_VERSION_KEY = "schema/version"

# Prefix for the typed section so we don't collide with v1.0.0's
# "window/geometry", "window/state", "grid/state" keys.
_PREFIX = "v2"


def _qs() -> QSettings:
    # Force IniFormat + UserScope so tests can redirect the path reliably and
    # the on-disk layout is identical across platforms. Native formats (macOS
    # plist, Windows registry) bypass QSettings.setPath and leak between
    # tests; ini is boring and inspectable.
    return QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        _ORG,
        _APP,
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load() -> AppSettings:
    """Return an AppSettings populated from QSettings, or defaults if absent.

    If the stored version is older than :data:`SCHEMA_VERSION`, run the
    registered migrators in order. If validation fails, fall back to
    defaults and log a warning — the lab can continue, and the next save
    will overwrite the garbage.
    """
    qs = _qs()
    stored_version = _coerce_int(qs.value(_VERSION_KEY), default=1)

    if stored_version < SCHEMA_VERSION:
        _migrate(qs, from_version=stored_version)

    settings = _read_typed(qs)
    problems = validate(settings)
    if problems:
        _LOG.warning("settings: validation failed, falling back to defaults: %s",
                     "; ".join(problems))
        return AppSettings.defaults()
    return settings


def _read_typed(qs: QSettings) -> AppSettings:
    """Best-effort typed read. Missing keys fall through to dataclass defaults."""
    d = AppSettings.defaults()

    recon = ReconDefaults(
        wavelength_nm=_coerce_float(qs.value(f"{_PREFIX}/recon/wavelength_nm"),
                                    d.recon.wavelength_nm),
        pixel_um=_coerce_float(qs.value(f"{_PREFIX}/recon/pixel_um"),
                               d.recon.pixel_um),
        z_mm=_coerce_float(qs.value(f"{_PREFIX}/recon/z_mm"),
                           d.recon.z_mm),
        n_medium=_coerce_float(qs.value(f"{_PREFIX}/recon/n_medium"),
                               d.recon.n_medium),
        mask_radius=_coerce_int(qs.value(f"{_PREFIX}/recon/mask_radius"),
                                d.recon.mask_radius),
        subtract_mean=_coerce_bool(qs.value(f"{_PREFIX}/recon/subtract_mean"),
                                   d.recon.subtract_mean),
        hann_window=_coerce_bool(qs.value(f"{_PREFIX}/recon/hann_window"),
                                 d.recon.hann_window),
    )

    af = AutofocusDefaults(
        z_min_mm=_coerce_float(qs.value(f"{_PREFIX}/autofocus/z_min_mm"),
                               d.autofocus.z_min_mm),
        z_max_mm=_coerce_float(qs.value(f"{_PREFIX}/autofocus/z_max_mm"),
                               d.autofocus.z_max_mm),
        metric=_coerce_str(qs.value(f"{_PREFIX}/autofocus/metric"),
                           d.autofocus.metric),
        adaptive=_coerce_bool(qs.value(f"{_PREFIX}/autofocus/adaptive"),
                              d.autofocus.adaptive),
    )

    qpi = QPIDefaults(
        cell_refractive_index=_coerce_float(
            qs.value(f"{_PREFIX}/qpi/cell_refractive_index"),
            d.qpi.cell_refractive_index),
        medium_refractive_index=_coerce_float(
            qs.value(f"{_PREFIX}/qpi/medium_refractive_index"),
            d.qpi.medium_refractive_index),
        phase_offset=_coerce_float(qs.value(f"{_PREFIX}/qpi/phase_offset"),
                                   d.qpi.phase_offset),
    )

    io = IODefaults(
        last_folder=_coerce_str(qs.value(f"{_PREFIX}/io/last_folder"),
                                d.io.last_folder),
        last_preset=_coerce_str(qs.value(f"{_PREFIX}/io/last_preset"),
                                d.io.last_preset),
        last_hologram=_coerce_str(qs.value(f"{_PREFIX}/io/last_hologram"),
                                  d.io.last_hologram),
        last_report_folder=_coerce_str(qs.value(f"{_PREFIX}/io/last_report_folder"),
                                       d.io.last_report_folder),
    )

    return AppSettings(
        schema_version=SCHEMA_VERSION, recon=recon, autofocus=af, qpi=qpi, io=io,
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(settings: AppSettings) -> None:
    """Persist settings. Silent on I/O errors (audit log catches anomalies)."""
    problems = validate(settings)
    if problems:
        _LOG.warning("settings: refusing to persist invalid settings: %s",
                     "; ".join(problems))
        return

    qs = _qs()
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)

    r = settings.recon
    qs.setValue(f"{_PREFIX}/recon/wavelength_nm", float(r.wavelength_nm))
    qs.setValue(f"{_PREFIX}/recon/pixel_um", float(r.pixel_um))
    qs.setValue(f"{_PREFIX}/recon/z_mm", float(r.z_mm))
    qs.setValue(f"{_PREFIX}/recon/n_medium", float(r.n_medium))
    qs.setValue(f"{_PREFIX}/recon/mask_radius", int(r.mask_radius))
    qs.setValue(f"{_PREFIX}/recon/subtract_mean", bool(r.subtract_mean))
    qs.setValue(f"{_PREFIX}/recon/hann_window", bool(r.hann_window))

    a = settings.autofocus
    qs.setValue(f"{_PREFIX}/autofocus/z_min_mm", float(a.z_min_mm))
    qs.setValue(f"{_PREFIX}/autofocus/z_max_mm", float(a.z_max_mm))
    qs.setValue(f"{_PREFIX}/autofocus/metric", str(a.metric))
    qs.setValue(f"{_PREFIX}/autofocus/adaptive", bool(a.adaptive))

    q = settings.qpi
    qs.setValue(f"{_PREFIX}/qpi/cell_refractive_index", float(q.cell_refractive_index))
    qs.setValue(f"{_PREFIX}/qpi/medium_refractive_index", float(q.medium_refractive_index))
    qs.setValue(f"{_PREFIX}/qpi/phase_offset", float(q.phase_offset))

    io = settings.io
    qs.setValue(f"{_PREFIX}/io/last_folder", str(io.last_folder))
    qs.setValue(f"{_PREFIX}/io/last_preset", str(io.last_preset))
    qs.setValue(f"{_PREFIX}/io/last_hologram", str(io.last_hologram))
    qs.setValue(f"{_PREFIX}/io/last_report_folder", str(io.last_report_folder))

    qs.sync()


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_Migrator = Callable[[QSettings], None]


def _migrate_v1_to_v2(qs: QSettings) -> None:
    """v1.0.0 had no typed section. Write v2 defaults next to existing window state."""
    defaults = AppSettings.defaults()
    # We don't call save() here because we don't have an AppSettings to save
    # *except* defaults — and those will be written on the next explicit save().
    # All we do is stamp the new version so subsequent loads don't re-migrate.
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v1 → v2 (defaults used for new typed section)")


_MIGRATIONS: list[tuple[int, int, _Migrator]] = [
    (1, 2, _migrate_v1_to_v2),
]


def _migrate(qs: QSettings, *, from_version: int) -> None:
    current = from_version
    for src, dst, fn in _MIGRATIONS:
        if current == src:
            fn(qs)
            current = dst
    if current < SCHEMA_VERSION:
        _LOG.warning(
            "settings: no migration path from v%d to v%d; using defaults",
            current, SCHEMA_VERSION,
        )


# ---------------------------------------------------------------------------
# QSettings value coercion — QSettings returns strings cross-platform so we
# need explicit parsing. Booleans are the only interesting case.
# ---------------------------------------------------------------------------

def _coerce_float(v: Any, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_int(v: Any, default: int) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    return default


def _coerce_str(v: Any, default: str) -> str:
    if v is None:
        return default
    return str(v)


__all__ = ["load", "save"]
