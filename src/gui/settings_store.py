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
    AIDefaults,
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

    ai = AIDefaults(
        enabled=_coerce_bool(qs.value(f"{_PREFIX}/ai/enabled"), d.ai.enabled),
        endpoint_url=_coerce_str(qs.value(f"{_PREFIX}/ai/endpoint_url"),
                                 d.ai.endpoint_url),
        model_name=_coerce_str(qs.value(f"{_PREFIX}/ai/model_name"),
                               d.ai.model_name),
        temperature=_coerce_float(qs.value(f"{_PREFIX}/ai/temperature"),
                                  d.ai.temperature),
        max_tokens=_coerce_int(qs.value(f"{_PREFIX}/ai/max_tokens"),
                               d.ai.max_tokens),
        max_iterations=_coerce_int(qs.value(f"{_PREFIX}/ai/max_iterations"),
                                   d.ai.max_iterations),
        request_timeout_s=_coerce_float(qs.value(f"{_PREFIX}/ai/request_timeout_s"),
                                        d.ai.request_timeout_s),
        restrict_to_home=_coerce_bool(qs.value(f"{_PREFIX}/ai/restrict_to_home"),
                                      d.ai.restrict_to_home),
        confirm_irreversible=_coerce_bool(qs.value(f"{_PREFIX}/ai/confirm_irreversible"),
                                          d.ai.confirm_irreversible),
        audit_redact_for_llm=_coerce_bool(qs.value(f"{_PREFIX}/ai/audit_redact_for_llm"),
                                          d.ai.audit_redact_for_llm),
    )

    return AppSettings(
        schema_version=SCHEMA_VERSION, recon=recon, autofocus=af, qpi=qpi, io=io,
        ai=ai,
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

    ai = settings.ai
    qs.setValue(f"{_PREFIX}/ai/enabled", bool(ai.enabled))
    qs.setValue(f"{_PREFIX}/ai/endpoint_url", str(ai.endpoint_url))
    qs.setValue(f"{_PREFIX}/ai/model_name", str(ai.model_name))
    qs.setValue(f"{_PREFIX}/ai/temperature", float(ai.temperature))
    qs.setValue(f"{_PREFIX}/ai/max_tokens", int(ai.max_tokens))
    qs.setValue(f"{_PREFIX}/ai/max_iterations", int(ai.max_iterations))
    qs.setValue(f"{_PREFIX}/ai/request_timeout_s", float(ai.request_timeout_s))
    qs.setValue(f"{_PREFIX}/ai/restrict_to_home", bool(ai.restrict_to_home))
    qs.setValue(f"{_PREFIX}/ai/confirm_irreversible", bool(ai.confirm_irreversible))
    qs.setValue(f"{_PREFIX}/ai/audit_redact_for_llm", bool(ai.audit_redact_for_llm))

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


def _migrate_v2_to_v3(qs: QSettings) -> None:
    """v2 → v3 adds Ui2State, which lives in the Dear PyGui frontend's
    own JSON file — Qt doesn't read or write it. Just stamp the version
    so subsequent loads don't keep re-migrating."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v2 → v3 (ui2 state owned by v2 frontend)")


def _migrate_v3_to_v4(qs: QSettings) -> None:
    """v3 → v4 adds magnification fields to Ui2State — same story as
    v2→v3: the Dear PyGui frontend owns that block. No QSettings key
    changes, just stamp the version."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v3 → v4 (ui2 magnification fields)")


def _migrate_v4_to_v5(qs: QSettings) -> None:
    """v4 → v5 adds autofocus z-range fields to Ui2State. Dear PyGui
    owns that section; Qt frontend just stamps the version."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v4 → v5 (ui2 autofocus range fields)")


def _migrate_v5_to_v6(qs: QSettings) -> None:
    """v5 → v6 adds advanced-physics fields to Ui2State. JSON side
    handles the actual defaults; Qt just stamps the version."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v5 → v6 (ui2 advanced physics)")


def _migrate_v6_to_v7(qs: QSettings) -> None:
    """v6 → v7 adds Ui2State.user_presets — Qt frontend doesn't read
    it, just stamp the version."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v6 → v7 (ui2 user_presets)")


def _migrate_v9_to_v10(qs: QSettings) -> None:
    """v9 → v10 adds display-flip fields (v2 UI only). Qt stamps
    the version."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v9 → v10 (ui2 display flips)")


def _migrate_v8_to_v9(qs: QSettings) -> None:
    """v8 → v9 adds ``af_algorithm`` to Ui2State — JSON side handles
    the default; Qt just bumps the version stamp."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v8 → v9 (ui2 af_algorithm)")


def _migrate_v7_to_v8(qs: QSettings) -> None:
    """v7 → v8 adds optical_mode + auto_contrast_amplitude to
    Ui2State. Dear PyGui side owns those fields; Qt only stamps."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v7 → v8 (ui2 optical_mode)")


def _migrate_v10_to_v11(qs: QSettings) -> None:
    """v10 → v11 adds user_preset_archive (clobber recovery list)
    to Ui2State. Dear PyGui side owns those fields; Qt only stamps."""
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v10 → v11 (ui2 user_preset_archive)")


def _migrate_v11_to_v12(qs: QSettings) -> None:
    """v11 → v12 adds AIDefaults — local-LLM assistant configuration.

    Defaults are written lazily on the next ``save()`` (the dataclass
    default factory fills them in); migration only stamps the version
    so subsequent loads stop running this branch.
    """
    qs.setValue(_VERSION_KEY, SCHEMA_VERSION)
    _LOG.info("settings: migrated v11 → v12 (AIDefaults)")


_MIGRATIONS: list[tuple[int, int, _Migrator]] = [
    (1, 2, _migrate_v1_to_v2),
    (2, 3, _migrate_v2_to_v3),
    (3, 4, _migrate_v3_to_v4),
    (4, 5, _migrate_v4_to_v5),
    (5, 6, _migrate_v5_to_v6),
    (6, 7, _migrate_v6_to_v7),
    (7, 8, _migrate_v7_to_v8),
    (8, 9, _migrate_v8_to_v9),
    (9, 10, _migrate_v9_to_v10),
    (10, 11, _migrate_v10_to_v11),
    (11, 12, _migrate_v11_to_v12),
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
