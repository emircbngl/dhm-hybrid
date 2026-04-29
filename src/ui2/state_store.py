"""Persistence for the Dear PyGui frontend's :class:`Ui2State`.

JSON-backed, atomic-write, migration-aware. Lives next to the
onboarding flag under ``~/.dhm-reconstruction/ui2_state.json``.

Why a separate file instead of QSettings? The v2 app ships without
Qt — the scientific pipeline is Qt-free and we don't want Dear PyGui
pulling in PySide6 just to persist four dataclasses. QSettings also
collides awkwardly with the v1 frontend's geometry keys; a plain
JSON file is inspectable, diff-able, and the user can delete it
without touching the Qt side.

Schema evolves via the shared :data:`core.settings_schema.SCHEMA_VERSION`
knob. When a field lands that old on-disk JSON wouldn't round-trip,
write a ``_v{n}_to_v{n+1}(raw)`` migrator and append to ``_MIGRATIONS``.
Forward-compatible additions (new field with a default) need no
migration — :func:`_hydrate_ui2` drops them into the dataclass default.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from core.settings_schema import (
    SCHEMA_VERSION,
    AIDefaults,
    AppSettings,
    AutofocusDefaults,
    IODefaults,
    QPIDefaults,
    ReconDefaults,
    Ui2State,
)

_LOG = logging.getLogger(__name__)

# Legacy single-user paths — kept for migration only. New code should
# call :func:`default_state_path` so per-user resolution lands in
# ``~/.dhm-reconstruction/users/<username>/ui2_state.json`` (v2.0.7).
STATE_DIR = Path.home() / ".dhm-reconstruction"
STATE_PATH = STATE_DIR / "ui2_state.json"


def default_state_path() -> Path:
    """Per-user state path, with one-shot legacy migration.

    Resolves :func:`core.user_profile.user_state_path` and migrates
    a pre-existing single-user file the first time it sees one. Call
    every time you'd reach for ``STATE_PATH``; the migration cost is
    a single ``stat`` + maybe one ``shutil.copy2`` on the first run
    and zero thereafter."""
    from core import user_profile  # late import — no circular hazard
    user_profile.migrate_legacy_state_if_needed()
    return user_profile.user_state_path()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(path: Optional[Path] = None) -> AppSettings:
    """Return an AppSettings read from ``path`` (default: per-user
    state from :func:`default_state_path`).

    Missing file → fresh defaults. Corrupt JSON → log + defaults.
    Older schema → run migrators then hydrate. Any missing field in
    the on-disk payload falls through to dataclass defaults, so adding
    fields is always forward-compatible."""
    path = path or default_state_path()
    if not path.exists():
        return AppSettings.defaults()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("ui2 state: couldn't read %s (%s); using defaults",
                     path, exc)
        return AppSettings.defaults()

    raw = _migrate(raw)
    try:
        return _hydrate(raw)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("ui2 state: hydrate failed (%s); using defaults", exc)
        return AppSettings.defaults()


def save(settings: AppSettings, path: Optional[Path] = None) -> None:
    """Persist ``settings`` atomically.

    Writes to a sibling ``.tmp`` file and renames via :func:`os.replace`
    so a crash mid-write never corrupts the live state file."""
    path = path or default_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = asdict(settings)
    # schema_version is the source of truth when we read back — stamp
    # it with the current code version, not whatever was loaded.
    payload["schema_version"] = SCHEMA_VERSION
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              default=_json_default),
                   encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Debounced background save — used by the app's render loop
# ---------------------------------------------------------------------------

class DebouncedSaver:
    """Coalesce rapid ``mark_dirty()`` calls into a single background save.

    The render loop calls :meth:`tick` each frame; once ``delay`` seconds
    have passed since the last ``mark_dirty`` the pending snapshot is
    flushed on a daemon thread so UI rendering never blocks on disk I/O.

    Callers must provide a ``snapshot()`` closure that returns the
    AppSettings to persist — we don't hold a reference to avoid
    copy-on-write issues with the live state."""

    def __init__(self, snapshot: Callable[[], AppSettings],
                 *, delay: float = 1.5) -> None:
        self._snapshot = snapshot
        self._delay = float(delay)
        self._dirty_at: Optional[float] = None
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty_at = time.monotonic()

    def tick(self) -> None:
        with self._lock:
            if self._dirty_at is None:
                return
            if time.monotonic() - self._dirty_at < self._delay:
                return
            self._dirty_at = None
        self._flush_async()

    def flush_now(self) -> None:
        """Synchronous flush — call from the shutdown path so the user
        never loses their last change to a late-frame debounce."""
        with self._lock:
            self._dirty_at = None
        try:
            save(self._snapshot())
            self.last_error = None
        except Exception as exc:  # pragma: no cover - best-effort
            self.last_error = str(exc)
            _LOG.warning("ui2 state: sync save failed: %s", exc)

    def _flush_async(self) -> None:
        def _worker():
            try:
                save(self._snapshot())
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                _LOG.warning("ui2 state: async save failed: %s", exc)

        threading.Thread(target=_worker, daemon=True,
                         name="ui2-state-saver").start()


# ---------------------------------------------------------------------------
# Migration chain — chained transformations on the raw dict
# ---------------------------------------------------------------------------

def _v1_to_v2(raw: dict) -> dict:
    """Stamp version; no field changes — v1 had no ui2 section anyway."""
    raw["schema_version"] = 2
    return raw


def _v2_to_v3(raw: dict) -> dict:
    """v3 introduces the ui2 block. If missing, add an empty one; the
    dataclass defaults take over at hydrate time."""
    if "ui2" not in raw:
        raw["ui2"] = {}
    raw["schema_version"] = 3
    return raw


def _v3_to_v4(raw: dict) -> dict:
    """v4 adds the rest of the scientific parameters the v2 port lost:
    ``magnification``, ``pixel_is_effective``, QPI ``n_sample`` and
    ``n_medium``, and the autofocus ``autofocus_metric`` choice.

    State files written by v2.0.2 didn't know these fields — we backfill
    with the same defaults v2.0.2 hardcoded in workers.py, so loading
    an old state is bit-identical to what v2.0.2 computed. The
    behavioural change only kicks in once the user touches a widget."""
    ui2 = raw.setdefault("ui2", {})
    if not isinstance(ui2, dict):
        raw["ui2"] = {}
        ui2 = raw["ui2"]
    ui2.setdefault("magnification", 1.0)
    ui2.setdefault("pixel_is_effective", True)
    ui2.setdefault("n_sample", 1.38)
    ui2.setdefault("n_medium", 1.337)
    ui2.setdefault("autofocus_metric", "LAPLACIAN_VARIANCE")
    raw["schema_version"] = 4
    return raw


def _v4_to_v5(raw: dict) -> dict:
    """v5 pulls autofocus z_min/z_max/n_steps out of the hardcoded
    workers defaults and into Ui2State so the sidebar can edit them.
    Backfill matches v2.0.2 hardcodes (-25/+25 mm, 40 steps) so old
    dumps produce bit-identical scans."""
    ui2 = raw.setdefault("ui2", {})
    if not isinstance(ui2, dict):
        raw["ui2"] = {}
        ui2 = raw["ui2"]
    ui2.setdefault("af_z_min_mm", -25.0)
    ui2.setdefault("af_z_max_mm", 25.0)
    ui2.setdefault("af_n_steps", 40)
    raw["schema_version"] = 5
    return raw


def _v5_to_v6(raw: dict) -> dict:
    """v6 adds the advanced-physics block (subtract_mean, hann_window,
    fft_backend, unwrap_method). Defaults match v1 ReconDefaults /
    batch_renderer so loading a v5 dump produces bit-identical output
    until the user flips a checkbox."""
    ui2 = raw.setdefault("ui2", {})
    if not isinstance(ui2, dict):
        raw["ui2"] = {}
        ui2 = raw["ui2"]
    ui2.setdefault("subtract_mean", True)
    ui2.setdefault("hann_window", False)
    ui2.setdefault("fft_backend", "auto")
    ui2.setdefault("unwrap_method", "GRADIENT_INTEGRATION")
    raw["schema_version"] = 6
    return raw


def _v6_to_v7(raw: dict) -> dict:
    """v7 introduces ``Ui2State.user_presets`` — empty dict by default.
    State dumps written by v2.0.5 never knew the field existed; the
    backfill is a no-op from a behavioural standpoint (no extra chips
    appear until the user saves one)."""
    ui2 = raw.setdefault("ui2", {})
    if not isinstance(ui2, dict):
        raw["ui2"] = {}
        ui2 = raw["ui2"]
    existing = ui2.get("user_presets")
    if not isinstance(existing, dict):
        ui2["user_presets"] = {}
    raw["schema_version"] = 7
    return raw


def _v7_to_v8(raw: dict) -> dict:
    """v8 adds ``optical_mode`` (transmission/reflection) + the
    amplitude auto-contrast toggle. Backfill matches v1 behaviour:
    transmission mode + auto-contrast on, which is what v2.0.5
    was effectively doing implicitly."""
    ui2 = raw.setdefault("ui2", {})
    if not isinstance(ui2, dict):
        raw["ui2"] = {}
        ui2 = raw["ui2"]
    ui2.setdefault("optical_mode", "transmission")
    ui2.setdefault("auto_contrast_amplitude", True)
    raw["schema_version"] = 8
    return raw


def _v9_to_v10(raw: dict) -> dict:
    """v10 adds ``flip_display_v`` / ``flip_display_h`` toggles.
    Default True / True matches the empirical lab-camera + DPG
    180° discrepancy reported on 2026-04-24 evening."""
    ui2 = raw.setdefault("ui2", {})
    if not isinstance(ui2, dict):
        raw["ui2"] = {}
        ui2 = raw["ui2"]
    ui2.setdefault("flip_display_v", True)
    ui2.setdefault("flip_display_h", True)
    raw["schema_version"] = 10
    return raw


def _v8_to_v9(raw: dict) -> dict:
    """v9 restores v1's adaptive autofocus algorithms. Default
    ``zscan`` keeps existing state files bit-identical — users pick
    adaptive explicitly from the sidebar combo."""
    ui2 = raw.setdefault("ui2", {})
    if not isinstance(ui2, dict):
        raw["ui2"] = {}
        ui2 = raw["ui2"]
    ui2.setdefault("af_algorithm", "zscan")
    raw["schema_version"] = 9
    return raw


def _v10_to_v11(raw: dict) -> dict:
    """v11 introduces ``Ui2State.user_preset_archive`` — empty dict
    by default. v10 dumps lose nothing; the next save-over-existing
    will start populating archive entries."""
    ui2 = raw.setdefault("ui2", {})
    if not isinstance(ui2, dict):
        raw["ui2"] = {}
        ui2 = raw["ui2"]
    ui2.setdefault("user_preset_archive", {})
    raw["schema_version"] = 11
    return raw


def _v11_to_v12(raw: dict) -> dict:
    """v12 introduces top-level ``AIDefaults`` block — local-LLM
    assistant configuration (Ollama endpoint, model, timeouts,
    redaction). v11 dumps backfill with the dataclass defaults
    (``enabled=True``, ``llama3.2``, ``localhost:11434``, …) so
    a fresh launch lights the assistant up; users who don't have
    Ollama running just see the assistant fail gracefully."""
    raw.setdefault("ai", {})
    raw["schema_version"] = 12
    return raw


_MIGRATIONS: list[tuple[int, int, Callable[[dict], dict]]] = [
    (1, 2, _v1_to_v2),
    (2, 3, _v2_to_v3),
    (3, 4, _v3_to_v4),
    (4, 5, _v4_to_v5),
    (5, 6, _v5_to_v6),
    (6, 7, _v6_to_v7),
    (7, 8, _v7_to_v8),
    (8, 9, _v8_to_v9),
    (9, 10, _v9_to_v10),
    (10, 11, _v10_to_v11),
    (11, 12, _v11_to_v12),
]


def _migrate(raw: dict) -> dict:
    """Run every migrator whose source matches the current version."""
    current = int(raw.get("schema_version", 1) or 1)
    for src, dst, fn in _MIGRATIONS:
        if current == src:
            raw = fn(raw)
            current = dst
    if current < SCHEMA_VERSION:
        _LOG.warning(
            "ui2 state: no migration path from v%d to v%d; resetting",
            current, SCHEMA_VERSION,
        )
        return {"schema_version": SCHEMA_VERSION}
    return raw


# ---------------------------------------------------------------------------
# Hydrate — dict → AppSettings, tolerant to missing fields
# ---------------------------------------------------------------------------

def _hydrate(raw: dict) -> AppSettings:
    d = AppSettings.defaults()
    return AppSettings(
        schema_version=SCHEMA_VERSION,
        recon=_hydrate_dc(ReconDefaults, raw.get("recon"), d.recon),
        autofocus=_hydrate_dc(AutofocusDefaults, raw.get("autofocus"),
                              d.autofocus),
        qpi=_hydrate_dc(QPIDefaults, raw.get("qpi"), d.qpi),
        io=_hydrate_dc(IODefaults, raw.get("io"), d.io),
        # v12: AIDefaults — local-LLM assistant config. Same hydrate
        # tolerance as the other dataclass blocks: missing fields fall
        # back to defaults, unknown keys silently dropped.
        ai=_hydrate_dc(AIDefaults, raw.get("ai"), d.ai),
        ui2=_hydrate_ui2(raw.get("ui2"), d.ui2),
    )


def _hydrate_dc(dc_cls, payload: Optional[dict], default):
    """Generic dataclass hydrator. Unknown keys are silently dropped so
    a rollback to an older code version can't crash on forward fields."""
    if not isinstance(payload, dict):
        return default
    kwargs = {}
    for fname, fdef in default.__dataclass_fields__.items():
        if fname in payload:
            try:
                kwargs[fname] = payload[fname]
            except Exception:
                pass
    try:
        return dc_cls(**{**asdict(default), **kwargs})
    except Exception:
        return default


def _hydrate_ui2(payload: Optional[dict], default: Ui2State) -> Ui2State:
    """Same logic as :func:`_hydrate_dc` but narrows list types so
    ``recent`` survives JSON round-trip as a proper ``list[str]``."""
    if not isinstance(payload, dict):
        return default
    merged: dict[str, Any] = asdict(default)
    for fname in default.__dataclass_fields__:
        if fname in payload:
            merged[fname] = payload[fname]
    # Normalise recent → list[str] — JSON gives us list[Any].
    recent = merged.get("recent") or []
    if not isinstance(recent, list):
        recent = []
    merged["recent"] = [str(p) for p in recent]
    # user_presets must be a dict — defend against on-disk corruption
    # (list / scalar) so a malformed JSON doesn't nuke the app.
    user_presets = merged.get("user_presets")
    if not isinstance(user_presets, dict):
        merged["user_presets"] = {}
    else:
        # Each value should itself be a dict — reject anything weirder.
        merged["user_presets"] = {
            str(k): v for k, v in user_presets.items()
            if isinstance(v, dict)
        }
    # user_preset_archive: dict[name -> list[dict]]. Anything else
    # (list at top level, non-list values, non-dict entries) gets
    # normalised to {} on the way in. Same defence as user_presets:
    # a malformed JSON shouldn't nuke the next launch.
    archive = merged.get("user_preset_archive")
    if not isinstance(archive, dict):
        merged["user_preset_archive"] = {}
    else:
        cleaned: dict[str, list[dict]] = {}
        for k, v in archive.items():
            if not isinstance(v, list):
                continue
            cleaned[str(k)] = [item for item in v
                               if isinstance(item, dict)]
        merged["user_preset_archive"] = cleaned
    try:
        return Ui2State(**merged)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """``json.dumps`` fallback: stringify Path-like objects."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON-serialisable")


__all__ = [
    "STATE_PATH", "STATE_DIR",
    "load", "save",
    "DebouncedSaver",
]
