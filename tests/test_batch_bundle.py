"""HDF5 batch bundle roundtrip + edge-case tests (v2.0.6).

Covers the writer/reader pair in :mod:`core.batch_bundle`:

* Empty input → explicit ValueError (no silent empty file).
* Single-entry roundtrip → phase/amplitude survive, attrs survive.
* Multi-entry order preserved.
* Stem collision → deterministic suffix (``_02``, ``_03`` …), no silent
  overwrite.
* Non-JSON-serialisable metadata values stringified at write time so
  a weird upstream payload can't crash the bundle mid-write.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.batch_bundle import (  # noqa: E402
    SCHEMA_VERSION,
    BatchEntry,
    read_batch_hdf5,
    write_batch_hdf5,
)

h5py = pytest.importorskip("h5py")


def _entry(stem: str, z_mm: float = 5.0) -> BatchEntry:
    phase = np.random.default_rng(42).normal(
        size=(16, 16)).astype(np.float32)
    amp = np.abs(phase) + 0.1
    return BatchEntry(
        source_path=Path(f"/tmp/fake/{stem}.tif"),
        phase=phase,
        amplitude=amp.astype(np.float32),
        metadata={
            "z_mm": float(z_mm),
            "wavelength_nm": 632.8,
            "sample_id": "LIMS-12",
            "flag": True,
        },
        runtime_ms=123.4,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_entries_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_batch_hdf5(tmp_path / "x.h5", entries=[])


def test_single_entry_roundtrip(tmp_path):
    e = _entry("alpha")
    out = write_batch_hdf5(
        tmp_path / "bundle.h5", [e],
        sample_id="S-1", app_version="v2.0.6",
        recon_params={"method": "ASM", "z_mm": 5.0},
    )
    assert out.exists()
    # Root attrs carry the bundle-level metadata we asked for.
    with h5py.File(out, "r") as f:
        assert int(f.attrs["schema_version"]) == SCHEMA_VERSION
        assert str(f.attrs["sample_id"]) == "S-1"
        assert "recon_params_json" in f.attrs
        assert int(f.attrs["n_holograms"]) == 1

    back = read_batch_hdf5(out)
    assert len(back) == 1
    assert back[0].source_path.name == "alpha.tif"
    assert back[0].phase.shape == e.phase.shape
    assert np.allclose(back[0].phase, e.phase)
    assert np.allclose(back[0].amplitude, e.amplitude)
    # Scalar attrs come back as python builtins (see ``_attr_to_py``).
    assert back[0].metadata["z_mm"] == pytest.approx(5.0)
    assert back[0].metadata["sample_id"] == "LIMS-12"
    assert back[0].metadata["flag"] is True
    assert back[0].runtime_ms == pytest.approx(123.4)


def test_multiple_entries_preserve_order(tmp_path):
    entries = [_entry(f"hol_{i:02d}", z_mm=float(i)) for i in range(5)]
    out = write_batch_hdf5(tmp_path / "multi.h5", entries)
    back = read_batch_hdf5(out)
    assert len(back) == 5
    # HDF5 group iteration order is alphabetical by key, which matches
    # our zero-padded stems. Stability is the invariant we care about.
    recovered = [e.source_path.stem for e in back]
    assert recovered == sorted([e.source_path.stem for e in entries])


def test_stem_collision_gets_suffix(tmp_path):
    """Two source files with the same stem (``foo.tif`` + ``foo.png``)
    must land in distinct groups — ``foo``, ``foo_02`` — so neither
    is overwritten."""
    e1 = BatchEntry(
        source_path=Path("/tmp/foo.tif"),
        phase=np.zeros((4, 4), dtype=np.float32),
        amplitude=np.zeros((4, 4), dtype=np.float32),
    )
    e2 = BatchEntry(
        source_path=Path("/tmp/foo.png"),
        phase=np.ones((4, 4), dtype=np.float32),
        amplitude=np.ones((4, 4), dtype=np.float32),
    )
    out = write_batch_hdf5(tmp_path / "collide.h5", [e1, e2])
    with h5py.File(out, "r") as f:
        keys = sorted(f["holograms"].keys())
    assert keys == ["foo", "foo_02"]


def test_non_serialisable_metadata_stringified(tmp_path):
    """A Path or custom object in metadata must not abort the write
    — we stringify so the attr write goes through cleanly."""
    class Weird:
        def __str__(self): return "weird-repr"

    e = _entry("quirky")
    e.metadata["weird"] = Weird()
    e.metadata["path_obj"] = Path("/tmp/extra")
    out = write_batch_hdf5(tmp_path / "quirky.h5", [e])
    back = read_batch_hdf5(out)
    md = back[0].metadata
    assert md["weird"] == "weird-repr"
    assert "extra" in md["path_obj"]


def test_safe_group_key_slugifies():
    """A stem with ``/``, spaces, and punctuation lands as a valid
    HDF5 group name with collisions suffixed."""
    from core.batch_bundle import _safe_group_key
    taken: set = set()
    k1 = _safe_group_key("2026/04 - run#1", taken)
    k2 = _safe_group_key("2026/04 - run#1", taken)   # duplicate
    assert "/" not in k1 and "/" not in k2
    assert k1 != k2
    assert k2.endswith("_02")


def test_writer_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deeper" / "bundle.h5"
    write_batch_hdf5(out, [_entry("solo")])
    assert out.exists()
