"""Typed app settings schema.

Pure-Python dataclasses. The Qt-backed load/save live in
``src/gui/settings_store.py`` — core stays backend-agnostic and testable.

Versioned on purpose: v1.0.0 shipped QSettings for window geometry only;
v1.0.1 adds reconstruction, autofocus, and I/O defaults so the app stops
forgetting the user between launches. When the schema evolves again the
migrator chain in ``gui/settings_store`` picks it up; ``SCHEMA_VERSION``
is the only knob that moves.

Conventions:
    * All units are explicit in the field name (nm, um, mm).
    * Defaults represent a freshly-unboxed install with no customization.
    * ``last_folder`` / ``last_preset`` are strings, not Paths, because
      QSettings round-trips strings cleanly and paths should be validated
      at point-of-use, not at load time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

# Bumped whenever a field is added, removed, or renamed in a way that
# existing on-disk settings cannot be read as-is. Migrations live in
# gui/settings_store.py::_migrate.
SCHEMA_VERSION = 2


@dataclass
class ReconDefaults:
    """Last-used reconstruction parameters. Recalled on startup."""
    wavelength_nm: float = 632.8
    pixel_um: float = 3.45
    z_mm: float = 44.0
    n_medium: float = 1.0
    mask_radius: int = 80
    subtract_mean: bool = True
    hann_window: bool = False


@dataclass
class AutofocusDefaults:
    """Last-used autofocus parameters."""
    z_min_mm: float = 0.0
    z_max_mm: float = 120.0
    metric: str = "PHASE_VARIANCE"
    adaptive: bool = True


@dataclass
class QPIDefaults:
    """Last-used QPI parameters."""
    cell_refractive_index: float = 1.38
    medium_refractive_index: float = 1.335
    phase_offset: float = 0.0


@dataclass
class IODefaults:
    """Last-used I/O locations. Saved to smooth file dialogs."""
    last_folder: str = ""
    last_preset: str = ""
    last_hologram: str = ""
    last_report_folder: str = ""


@dataclass
class AppSettings:
    """The whole typed payload.

    On-disk layout (QSettings keys) is an implementation detail of
    ``gui/settings_store``; this dataclass is what the rest of the app
    sees.
    """
    schema_version: int = SCHEMA_VERSION
    recon: ReconDefaults = field(default_factory=ReconDefaults)
    autofocus: AutofocusDefaults = field(default_factory=AutofocusDefaults)
    qpi: QPIDefaults = field(default_factory=QPIDefaults)
    io: IODefaults = field(default_factory=IODefaults)

    def as_dict(self) -> dict[str, Any]:
        """Flat dict, suitable for audit-logging or JSON export."""
        return asdict(self)

    @classmethod
    def defaults(cls) -> "AppSettings":
        """Freshly-unboxed state. Handy for tests and a 'Reset' menu item."""
        return cls()

    # ---- shallow updates ---------------------------------------------------

    def with_recon(self, **kw) -> "AppSettings":
        return replace(self, recon=replace(self.recon, **kw))

    def with_autofocus(self, **kw) -> "AppSettings":
        return replace(self, autofocus=replace(self.autofocus, **kw))

    def with_qpi(self, **kw) -> "AppSettings":
        return replace(self, qpi=replace(self.qpi, **kw))

    def with_io(self, **kw) -> "AppSettings":
        return replace(self, io=replace(self.io, **kw))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(settings: AppSettings) -> list[str]:
    """Return a list of human-readable problems, or [] if valid.

    Used both at load time (to reject corrupt on-disk state and fall back
    to defaults) and before submit (to block garbage parameter sets from
    reaching the worker — Rams #4, honest feedback at point of entry).
    """
    problems: list[str] = []

    r = settings.recon
    if r.wavelength_nm <= 0:
        problems.append(f"wavelength_nm must be > 0 (got {r.wavelength_nm})")
    if r.wavelength_nm > 2000:
        problems.append(f"wavelength_nm looks implausible (got {r.wavelength_nm}; max 2000)")
    if r.pixel_um <= 0:
        problems.append(f"pixel_um must be > 0 (got {r.pixel_um})")
    if r.n_medium < 1.0:
        problems.append(f"n_medium must be ≥ 1.0 (got {r.n_medium})")
    if r.mask_radius <= 0:
        problems.append(f"mask_radius must be > 0 (got {r.mask_radius})")

    af = settings.autofocus
    if af.z_min_mm >= af.z_max_mm:
        problems.append(
            f"autofocus.z_min_mm ({af.z_min_mm}) must be < z_max_mm ({af.z_max_mm})"
        )

    q = settings.qpi
    if q.cell_refractive_index <= 0:
        problems.append(
            f"qpi.cell_refractive_index must be > 0 (got {q.cell_refractive_index})"
        )
    if q.medium_refractive_index <= 0:
        problems.append(
            f"qpi.medium_refractive_index must be > 0 (got {q.medium_refractive_index})"
        )
    if q.cell_refractive_index <= q.medium_refractive_index:
        problems.append(
            "qpi.cell_refractive_index must exceed medium_refractive_index "
            f"(got cell={q.cell_refractive_index}, medium={q.medium_refractive_index})"
        )

    return problems


__all__ = [
    "SCHEMA_VERSION",
    "ReconDefaults",
    "AutofocusDefaults",
    "QPIDefaults",
    "IODefaults",
    "AppSettings",
    "validate",
]
