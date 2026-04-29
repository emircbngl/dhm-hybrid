"""Time-series Session model — v2.0.7 foundation for longitudinal
DHM workflows.

Why this exists
---------------
Up to v2.0.6 every pipeline call (autofocus, multi-focus, QPI,
depth map) is single-hologram. Lab feedback (Lindqvist 2026-04-27)
is direct: "we image 30 cells every 5 minutes for 12 hours — your
tool gives me one CSV per hologram, I lose three days a week
stitching them together in Excel."

This module is the data structure that makes longitudinal pipelines
addressable as a single object. A :class:`Session` is:

* a flat ordered list of :class:`HologramFrame` (one per acquired
  image, with a timestamp);
* shared :class:`ReconParams`-shaped defaults that frame-specific
  overrides can mutate;
* operator + sample metadata (audit-grade);
* a sidecar JSON that survives restarts.

It is intentionally storage-agnostic — frames point at file paths,
not loaded arrays, so a 3000-frame session weighs kilobytes on
disk. The CLI session runner (next module up) loads frames on
demand.

Design rules
------------
* No DPG / no Qt — pure data model, pickle-clean.
* No scientific compute — `Session` doesn't autofocus or
  reconstruct; it's a manifest. Compute lives in the CLI / worker
  layer.
* Backward-compatibility: `Session.from_directory(...)` accepts
  glob patterns the lab's existing folder layouts already use, so
  a Karin-style "drop the folder" workflow is the default path.
* Audit semantics: `session.id` is a stable UUID; every frame
  carries a deterministic index so reruns are reproducible.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HologramFrame:
    """One acquired hologram in a session.

    Attributes
    ----------
    path
        Absolute or session-relative path to the TIFF / array on
        disk. The session manifest stores it as-given; the loader
        resolves relative paths against ``Session.root_dir``.
    timestamp_s
        Acquisition time in seconds-since-epoch (UTC). Many lab
        cameras embed a ``DateTime`` EXIF tag we'd parse on import,
        but a plain ``Path.stat().st_mtime`` also works as a
        fallback.
    index
        Position in the session sequence. Stable across reorder /
        resume — once written, never renumbered. The pipeline keys
        per-frame outputs by ``(session_id, index)`` so a partial
        resume can pick up exactly where it left off.
    params_overrides
        Optional dict of per-frame parameter overrides. Any field
        from ``ReconParams`` (wavelength, pixel_um, z_mm, …) can be
        overridden for one frame — useful when a few frames in a
        long session were acquired with a different exposure or
        objective. Defaults: empty dict, frame uses session params
        as-is.
    notes
        Free-form operator note attached to this frame. Lab uses
        these for "cell budded here" / "stage drift" / "auto-focus
        suspicious". Persisted, not interpreted by code.
    """
    path: str
    timestamp_s: float
    index: int
    params_overrides: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """A longitudinal DHM acquisition.

    Carries ordered ``frames`` plus shared metadata. JSON-serialisable
    via :meth:`to_dict` / :meth:`save_json`. Loaders never mutate the
    on-disk file — every save lands a temp + rename for atomicity.

    The session does NOT carry compute results. Per-frame outputs
    (autofocus z, QPI scalars, dry mass, etc.) are written to a
    sibling results directory by the CLI runner — the session is the
    *manifest*, the results are the *evidence*.

    Attributes
    ----------
    id
        Stable UUID hex (32 chars). Generated on creation.
    created_at
        ISO-8601 UTC timestamp string. For audit ordering.
    operator
        Username / lab member identifier. Empty string if unknown
        (multi-user profile lands later in v2.0.7; this field is
        populated by that layer).
    sample_id
        Free-text sample / experiment identifier. e.g.
        "A549_pHluorin_2026-04-27".
    root_dir
        Anchor path for resolving relative ``frame.path``. Defaults
        to the directory the manifest lives in; explicit override
        if the session is moved.
    params
        Default :class:`ReconParams`-shape dict every frame inherits
        unless its ``params_overrides`` says otherwise. Stored as a
        flat dict (not the dataclass) so the manifest survives a
        `ReconParams` schema bump without a migration.
    frames
        Ordered list of :class:`HologramFrame`.
    """
    id: str = ""
    created_at: str = ""
    operator: str = ""
    sample_id: str = ""
    root_dir: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    frames: List[HologramFrame] = field(default_factory=list)

    # ---- factories ------------------------------------------------------

    @classmethod
    def new(cls, *, operator: str = "", sample_id: str = "",
            params: Optional[Dict[str, Any]] = None,
            root_dir: str = "") -> "Session":
        """Fresh session with a new UUID + ISO-8601 timestamp."""
        return cls(
            id=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
            operator=operator,
            sample_id=sample_id,
            root_dir=root_dir,
            params=dict(params or {}),
            frames=[],
        )

    @classmethod
    def from_directory(cls, directory: Path | str, *,
                       glob_pattern: str = "*.tif*",
                       sort_by: str = "filename",
                       operator: str = "",
                       sample_id: str = "",
                       params: Optional[Dict[str, Any]] = None,
                       ) -> "Session":
        """Build a session from every file in ``directory`` matching
        ``glob_pattern``.

        Parameters
        ----------
        directory
            Folder containing the time-series TIFFs.
        glob_pattern
            File matcher. ``*.tif*`` covers ``.tif`` + ``.tiff`` —
            the lab's standard. Pass any ``Path.glob`` pattern.
        sort_by
            One of ``filename`` (lexicographic, default — matches
            the cell counter the lab embeds in filenames),
            ``mtime`` (file modification time — fallback when
            filenames are non-monotonic), ``ctime`` (creation
            time, where the OS supports it).
        operator, sample_id, params
            Forwarded to :meth:`new`.
        """
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"session source is not a directory: {directory}",
            )
        matches = sorted(directory.glob(glob_pattern))
        if sort_by == "filename":
            files = matches
        elif sort_by == "mtime":
            files = sorted(matches, key=lambda p: p.stat().st_mtime)
        elif sort_by == "ctime":
            files = sorted(matches, key=lambda p: p.stat().st_ctime)
        else:
            raise ValueError(f"unknown sort_by={sort_by!r}")

        if not files:
            raise FileNotFoundError(
                f"no files matched {glob_pattern!r} in {directory}",
            )

        sess = cls.new(
            operator=operator, sample_id=sample_id,
            params=params, root_dir=str(directory),
        )
        for i, p in enumerate(files):
            ts = p.stat().st_mtime  # st_mtime is the lab fallback;
            # cameras with EXIF DateTime can override at ingest time.
            sess.frames.append(HologramFrame(
                path=p.name,  # session-relative; root_dir resolves
                timestamp_s=float(ts),
                index=i,
            ))
        return sess

    # ---- mutation -------------------------------------------------------

    def add_frame(self, path: str | Path, *,
                  timestamp_s: Optional[float] = None,
                  notes: str = "",
                  params_overrides: Optional[Dict[str, Any]] = None,
                  ) -> HologramFrame:
        """Append a frame. Timestamp falls back to file mtime if not
        provided. Index is auto-assigned to ``len(frames)``."""
        p = Path(path)
        ts = float(timestamp_s) if timestamp_s is not None else \
            float(p.stat().st_mtime if p.exists() else 0.0)
        frame = HologramFrame(
            path=str(path),
            timestamp_s=ts,
            index=len(self.frames),
            params_overrides=dict(params_overrides or {}),
            notes=notes,
        )
        self.frames.append(frame)
        return frame

    def with_params(self, **kw: Any) -> "Session":
        """Return a copy with merged params dict — handy for
        configuration sweeps. Frames stay shared (HologramFrame is
        frozen)."""
        merged = {**self.params, **kw}
        return replace(self, params=merged)

    # ---- frame access ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def get_frame(self, index: int) -> HologramFrame:
        """Lookup by 0-based session index. ``IndexError`` for OOB."""
        if not 0 <= index < len(self.frames):
            raise IndexError(
                f"frame index {index} out of range (0..{len(self.frames) - 1})",
            )
        return self.frames[index]

    def resolve_frame_path(self, frame: HologramFrame) -> Path:
        """Anchor a frame's relative ``path`` against ``root_dir``.
        Absolute paths pass through unchanged."""
        p = Path(frame.path)
        if p.is_absolute():
            return p
        if self.root_dir:
            return Path(self.root_dir) / p
        return p

    def effective_params_for(self, frame: HologramFrame) -> Dict[str, Any]:
        """Merge session-default params with frame's overrides. Frame
        wins on collision. Returned dict is a fresh shallow copy —
        callers can mutate without touching session state."""
        merged = dict(self.params)
        merged.update(frame.params_overrides)
        return merged

    # ---- signature (for resume) ----------------------------------------

    def signature(self) -> str:
        """Stable hex digest over (params, frame paths + sizes). Two
        runs of the same session get the same signature, used by the
        CLI ``--resume-if-exists`` to skip already-computed frames."""
        import hashlib
        h = hashlib.sha256()
        h.update(json.dumps(self.params, sort_keys=True,
                            default=str).encode())
        for f in self.frames:
            try:
                p = self.resolve_frame_path(f)
                size = p.stat().st_size if p.exists() else -1
            except OSError:
                size = -1
            h.update(f"{f.index}:{f.path}:{size}".encode())
        return h.hexdigest()[:16]

    # ---- serialisation --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict. Frames flattened to plain dicts."""
        return {
            "schema": "dhm.session/1",   # bump when fields evolve
            "id": self.id,
            "created_at": self.created_at,
            "operator": self.operator,
            "sample_id": self.sample_id,
            "root_dir": self.root_dir,
            "params": dict(self.params),
            "frames": [
                {
                    "path": f.path,
                    "timestamp_s": f.timestamp_s,
                    "index": f.index,
                    "params_overrides": dict(f.params_overrides),
                    "notes": f.notes,
                } for f in self.frames
            ],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Session":
        """Reverse of :meth:`to_dict`. Tolerant: missing fields
        fall back to defaults; unknown keys ignored. Schema field
        is consumed for forward-compat warnings but not enforced —
        v2.0.7 is the only schema (1)."""
        if not isinstance(raw, dict):
            raise ValueError("session manifest must be a JSON object")
        frames_raw = raw.get("frames") or []
        frames: List[HologramFrame] = []
        for fr in frames_raw:
            if not isinstance(fr, dict):
                continue
            try:
                frames.append(HologramFrame(
                    path=str(fr.get("path", "")),
                    timestamp_s=float(fr.get("timestamp_s", 0.0)),
                    index=int(fr.get("index", len(frames))),
                    params_overrides=dict(fr.get("params_overrides") or {}),
                    notes=str(fr.get("notes", "")),
                ))
            except (TypeError, ValueError):
                continue
        return cls(
            id=str(raw.get("id", "")),
            created_at=str(raw.get("created_at", "")),
            operator=str(raw.get("operator", "")),
            sample_id=str(raw.get("sample_id", "")),
            root_dir=str(raw.get("root_dir", "")),
            params=dict(raw.get("params") or {}),
            frames=frames,
        )

    def save_json(self, path: Path | str) -> None:
        """Atomic write. Tempfile in the same directory + os.replace
        — survives a power cut without leaving a half-written
        manifest."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Same-dir tempfile guarantees the rename is on the same
        # filesystem (otherwise os.replace can fail across mounts).
        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=str(path.parent),
            prefix=path.stem + ".",
            suffix=".tmp",
            delete=False, encoding="utf-8",
        )
        try:
            json.dump(self.to_dict(), tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, path)
        except Exception:
            try:
                tmp.close()
            except Exception:
                pass
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise

    @classmethod
    def load_json(cls, path: Path | str) -> "Session":
        """Read a manifest from disk. Empty-file / malformed-JSON
        raise ``ValueError`` so the CLI can surface a clear error
        instead of silently using defaults."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"session manifest not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"session manifest is not valid JSON: {path} ({e})",
            ) from e
        return cls.from_dict(raw)


__all__ = [
    "HologramFrame",
    "Session",
]
