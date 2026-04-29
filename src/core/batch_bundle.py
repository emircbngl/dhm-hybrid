"""Batch reconstruction → HDF5 bundle writer (v2.0.6).

Each call to :func:`write_batch_hdf5` collects N reconstructed
holograms into a single ``.h5`` file, so a lab can ship a whole
experiment's worth of phase/amplitude data as one artifact. The
schema is intentionally minimal — one group per source hologram,
two datasets (phase + amplitude) per group, with per-group attrs
holding the source path, z, wavelength, magnification, and runtime.

v1 ships its own tomography bundle writer in ``core.depth_map`` but
that emits a *single* depth volume. v2.0.5's batch path wrote per-file
PNG previews; this module is the bundle upgrade.

Dependency: ``h5py``. Lazy-imported inside the writer so modules that
merely *import* :class:`BatchEntry` don't pay for it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
"""Bumped when a field is renamed or a breaking change lands. The UI
reader passes through whatever ``schema_version`` the on-disk file
declares — older versions are still readable, newer ones trigger a
warning that says "recorded by a newer build, some metadata may be
missing."""


@dataclass
class BatchEntry:
    """One reconstructed hologram ready for the bundle.

    ``phase`` and ``amplitude`` are 2-D float32 arrays (the same shape
    the :mod:`core.reconstruction` kernel produces). ``metadata``
    lands as HDF5 group attrs — keep it JSON-serialisable strings /
    numbers / booleans.
    """
    source_path: Path
    phase: np.ndarray
    amplitude: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_ms: float = 0.0


def _safe_group_key(stem: str, taken: set[str]) -> str:
    """Slugify ``stem`` and append a collision suffix if needed.

    HDF5 group names can contain most characters but ``/`` is a path
    separator, so we replace it and anything else weird with ``_``.
    Duplicate stems (``foo.tif`` + ``foo.png``) get ``foo_01``,
    ``foo_02`` — deterministic, no silent overwrite."""
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem) or "hologram"
    key = base
    counter = 1
    while key in taken:
        counter += 1
        key = f"{base}_{counter:02d}"
    taken.add(key)
    return key


def write_batch_hdf5(
    path: Path,
    entries: Sequence[BatchEntry],
    *,
    sample_id: str = "",
    recon_params: Optional[Dict[str, Any]] = None,
    app_version: str = "unknown",
) -> Path:
    """Write a batch bundle. Returns the output path for chaining."""
    if not entries:
        raise ValueError("write_batch_hdf5: entries is empty")
    import h5py   # lazy — caller can skip bundle path if h5py absent

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["app_version"] = str(app_version)
        f.attrs["sample_id"] = str(sample_id)
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
        f.attrs["n_holograms"] = len(entries)
        if recon_params is not None:
            f.attrs["recon_params_json"] = json.dumps(
                recon_params, default=str,
            )
        grp = f.create_group("holograms")
        for entry in entries:
            key = _safe_group_key(entry.source_path.stem, taken)
            g = grp.create_group(key)
            g.create_dataset(
                "phase",
                data=entry.phase.astype(np.float32, copy=False),
                compression="gzip", compression_opts=4,
            )
            g.create_dataset(
                "amplitude",
                data=entry.amplitude.astype(np.float32, copy=False),
                compression="gzip", compression_opts=4,
            )
            g.attrs["source_path"] = str(entry.source_path)
            g.attrs["runtime_ms"] = float(entry.runtime_ms)
            for mk, mv in entry.metadata.items():
                # HDF5 attrs accept scalars + short strings + numpy
                # scalar types. Stringify anything else so the write
                # doesn't fail halfway through a batch.
                if isinstance(mv, (str, bytes, int, float, bool,
                                   np.integer, np.floating)):
                    g.attrs[mk] = mv
                else:
                    g.attrs[mk] = str(mv)
    return path


def read_batch_hdf5(path: Path) -> List[BatchEntry]:
    """Read back a bundle. Useful for tests + a future "re-open bundle"
    viewer. Keeps the same field layout as the writer produced."""
    import h5py

    path = Path(path)
    entries: List[BatchEntry] = []
    with h5py.File(path, "r") as f:
        grp = f["holograms"]
        for key in grp.keys():
            g = grp[key]
            metadata = {k: g.attrs[k] for k in g.attrs.keys()
                        if k not in ("source_path", "runtime_ms")}
            entries.append(BatchEntry(
                source_path=Path(str(g.attrs["source_path"])),
                phase=np.asarray(g["phase"][...], dtype=np.float32),
                amplitude=np.asarray(g["amplitude"][...], dtype=np.float32),
                metadata={k: _attr_to_py(v) for k, v in metadata.items()},
                runtime_ms=float(g.attrs["runtime_ms"]),
            ))
    return entries


def _attr_to_py(v: Any) -> Any:
    """h5py returns numpy scalars — normalise to builtin types so
    roundtrip tests can use simple ``==`` comparisons."""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except Exception:
            return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


__all__ = [
    "SCHEMA_VERSION",
    "BatchEntry",
    "write_batch_hdf5",
    "read_batch_hdf5",
]
