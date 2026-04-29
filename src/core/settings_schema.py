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
# gui/settings_store.py::_migrate (Qt frontend) and
# ui2/state_store.py::_migrate (Dear PyGui frontend).
# v2 → v3: add Ui2State so the Dear PyGui frontend can persist its own
# theme, recent files, workflow mode, and live ReconParams alongside
# the existing Qt-owned recon/autofocus/qpi defaults.
# v3 → v4: add magnification + pixel_is_effective to Ui2State — the
# v1→v2 port accidentally dropped these fields, silently breaking DHM
# physics for microscope setups. Backfill old dumps with M=1, pixel
# assumed effective (matches v2.0.1 behaviour on existing state files).
# v11 → v12: add AIDefaults — local-LLM assistant configuration.
SCHEMA_VERSION = 12


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
class AIDefaults:
    """Local-LLM assistant configuration.

    Defaults target Ollama running on the same machine — the lab's
    air-gapped friendly setup. ``qwen2.5:7b-instruct`` is the default
    because its OpenAI-style tool calling lands cleanly on the first
    try in the >90% range; ``llama3.2`` (3B) is faster to boot but
    misses tool calls more often. Both are surfaced in the Settings
    dialog so the user can swap on demand.
    """
    enabled: bool = True
    endpoint_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    max_tokens: int = 2048
    max_iterations: int = 8
    request_timeout_s: float = 120.0
    # Path-traversal default. True means tool calls that ingest a path
    # (load_hologram) refuse anything outside ``Path.home()``. Lab
    # installations with NAS mounts under ``/data/...`` flip this off.
    restrict_to_home: bool = True
    # Modal confirmation gate for irreversible tool calls (real-stage
    # moves once a hardware driver is connected). v1 has nothing
    # irreversible to confirm; the flag exists so v2 can light up.
    confirm_irreversible: bool = True
    # When True, the audit-tail tool call replaces the ``operator``
    # field with ``<user>`` and trims absolute paths to basenames
    # before passing to the LLM. Belongs in settings (not hardcoded)
    # because some users want the full record for debug.
    audit_redact_for_llm: bool = True


@dataclass
class Ui2State:
    """Dear PyGui (v2) frontend state.

    Everything the v2 app needs to feel continuous across launches:
    window geometry, chosen theme, current workflow mode, recent files,
    the live reconstruction parameters, the reference hologram path,
    and the selected preset.

    Stored as JSON under ``~/.dhm-reconstruction/ui2_state.json`` —
    independent from the Qt tab's QSettings file, same schema.
    """
    # Window geometry — 0 means "recompute from screen size at startup".
    viewport_w: int = 0
    viewport_h: int = 0
    # Appearance
    theme: str = "dark"
    # Sample metadata entered by the operator (free-form string).
    sample_id: str = ""
    # Which workflow tab the user was on.
    workflow_mode: str = "Reconstruct"
    # Preset chip selected (e.g. "Cell", "Film", "USAF", "Custom").
    selected_preset: str = ""
    # File memory — mirrors Qt IODefaults purpose but typed for v2.
    recent: list[str] = field(default_factory=list)
    last_dir: str = ""
    last_hologram: str = ""
    # Reference hologram + whether subtraction is armed.
    reference_path: str = ""
    subtract_reference: bool = False
    # Live reconstruction parameters (subset — v2 holds these in
    # ReconParams; we snapshot them here so the next launch restores).
    wavelength_nm: float = 632.8
    pixel_um: float = 5.0
    z_mm: float = 10.0
    mask_radius: int = 40
    method: str = "ASM"
    # v2.0.3: objective magnification + "is the pixel I typed already
    # the effective one?" flag. Default is pixel_is_effective=True so
    # state files written by v2.0.2 (which never knew M existed) keep
    # behaving exactly as they did on load.
    magnification: float = 1.0
    pixel_is_effective: bool = True
    # v2.0.3: QPI refractive indices — v1's qpi_tab exposed these as
    # user-editable fields. v2.0.2 hardcoded them in workers.py.
    n_sample: float = 1.38
    n_medium: float = 1.337
    # v2.0.3: autofocus metric — v1 has an 11-option combo; v2.0.2
    # silently pinned LAPLACIAN_VARIANCE everywhere.
    autofocus_metric: str = "LAPLACIAN_VARIANCE"
    # v2.0.4: autofocus scan range + step count — exposed in sidebar so
    # small-z (fast DHM) and long-range (tomography) setups don't share
    # the same hardcoded -25/+25 mm assumption.
    af_z_min_mm: float = -25.0
    af_z_max_mm: float = 25.0
    af_n_steps: int = 40
    # v2.0.8: autofocus algorithm — adaptive search restored from v1.
    af_algorithm: str = "zscan"
    # v2.0.9: display-only image flips. 180° default (both axes
    # True) matches the lab camera + DPG axis-inversion empirical
    # result. Persisted here so the operator's preference survives
    # restarts — pipeline math never reads these.
    flip_display_v: bool = True
    flip_display_h: bool = True
    # v2.0.5: advanced pre/post-processing. Exposed in the sidebar's
    # "Advanced" collapsing block; defaults match v1 ReconDefaults /
    # batch_renderer so behaviour of existing state dumps is preserved.
    subtract_mean: bool = True
    hann_window: bool = False
    fft_backend: str = "auto"
    unwrap_method: str = "GRADIENT_INTEGRATION"
    # v2.0.6: user-defined preset dict — maps display name → kwargs dict
    # (same shape the built-in presets use). Merged on top of the
    # hardcoded Cell/Film/USAF/Custom set so user presets appear as
    # extra chips in the sidebar. Persisted here to survive restarts.
    user_presets: dict[str, dict] = field(default_factory=dict)
    # v2.0.7: archive of overwritten user-preset versions. Each save-
    # over-existing append the *previous* dict here under the same key,
    # so the operator can recover or audit a clobber. List preserves
    # insertion order — newest archived first not relied on; consumers
    # iterate by index. Capped at 10 versions per name to keep state
    # files small (oldest dropped on append).
    user_preset_archive: dict[str, list[dict]] = field(default_factory=dict)
    # v2.0.7: optical path (transmission vs reflection) — reflection
    # halves OPD before height conversion. Picking the wrong one is
    # the classic "height is 2× off" bug.
    optical_mode: str = "transmission"
    # Display polish — percentile contrast stretch on amplitude.
    auto_contrast_amplitude: bool = True


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
    ui2: Ui2State = field(default_factory=Ui2State)
    ai: AIDefaults = field(default_factory=AIDefaults)

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

    def with_ui2(self, **kw) -> "AppSettings":
        return replace(self, ui2=replace(self.ui2, **kw))

    def with_ai(self, **kw) -> "AppSettings":
        return replace(self, ai=replace(self.ai, **kw))


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
    "Ui2State",
    "AIDefaults",
    "AppSettings",
    "validate",
]
