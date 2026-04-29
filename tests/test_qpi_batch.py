"""Multi-candidate QPI batch — run QPI at each candidate focus plane.

Ground truth: two spheres at different z + different sizes. If the
batch runs QPI correctly at each sphere's plane, the resulting
``dry_mass_pg`` and ``area_um2`` values must *differ* between entries
(because different refocused silhouettes are segmented).  A regression
that silently reused the same reconstruction would produce two
identical rows — the test fires on that.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from core.autofocus import FocusCandidate
from core.offaxis import OffAxisParams, extract_complex_field_offaxis
from core.qpi_batch import (
    QPIBatchEntry,
    qpi_batch_to_rows,
    run_qpi_for_candidates,
    write_qpi_batch_csv,
)
from core.reconstruction import ReconstructionMethod, ReconstructionParams

from fixtures.synthetic_hologram import (
    HologramConfig,
    SphereSpec,
    build_hologram,
)


_CFG = HologramConfig(
    shape=(256, 256), pixel_m=2.5e-6, wavelength_m=632.8e-9,
    carrier_freq_m_inv=(50_000.0, 0.0),
)


def _scene():
    """Two well-separated spheres at z=8 and z=18 mm, different radii."""
    return [
        SphereSpec(radius_m=20e-6, z_m=8e-3,
                   center_yx_m=(-60e-6, -40e-6)),
        SphereSpec(radius_m=30e-6, z_m=18e-3,
                   center_yx_m=(50e-6, 30e-6)),
    ]


def _field_for_scene():
    hologram = build_hologram(_scene(), _CFG)
    field, _ = extract_complex_field_offaxis(hologram, OffAxisParams(radius=40))
    return field


def _base_params() -> ReconstructionParams:
    return ReconstructionParams(
        wavelength_m=_CFG.wavelength_m,
        pixel_size_m=_CFG.pixel_m, z_m=0.0, n=1.33,
    )


def _ground_truth_candidates():
    return [
        FocusCandidate(z_m=-8e-3, score=0.9, prominence=0.5, rank=0),
        FocusCandidate(z_m=-18e-3, score=0.8, prominence=0.4, rank=1),
    ]


# ---- Empty list -----------------------------------------------------------

def test_empty_candidates_returns_empty_list():
    field = np.ones(_CFG.shape, dtype=np.complex64)
    out = run_qpi_for_candidates(
        field, _base_params(), ReconstructionMethod.ASM, [],
    )
    assert out == []


# ---- Core: per-candidate QPI ----------------------------------------------

def test_run_qpi_for_candidates_produces_one_entry_per_input():
    field = _field_for_scene()
    entries = run_qpi_for_candidates(
        field, _base_params(), ReconstructionMethod.ASM,
        _ground_truth_candidates(),
        n_sample=1.40, n_medium=1.33, compute_psd=False,
    )
    assert len(entries) == 2
    # Candidate order must be preserved.
    assert entries[0].candidate.rank == 0
    assert entries[1].candidate.rank == 1
    # Each entry's QPIResult must be populated — at minimum phase_stats.
    for e in entries:
        assert e.qpi_result.phase_stats is not None


def test_per_candidate_reconstructions_are_distinct():
    """At different focus planes the segmented cell mask area should
    differ (or at worst the dry mass should). A regression that
    silently reused one reconstruction would produce identical
    numbers — fire here if so."""
    field = _field_for_scene()
    entries = run_qpi_for_candidates(
        field, _base_params(), ReconstructionMethod.ASM,
        _ground_truth_candidates(),
        n_sample=1.40, n_medium=1.33, compute_psd=False,
    )
    opd0 = entries[0].qpi_result.phase_stats.range_nm
    opd1 = entries[1].qpi_result.phase_stats.range_nm
    mass0 = entries[0].qpi_result.total_dry_mass_pg or 0.0
    mass1 = entries[1].qpi_result.total_dry_mass_pg or 0.0
    # Either the OPD range or the dry mass must differ — both being
    # identical would mean the batch reused the same reconstruction.
    assert (abs(opd0 - opd1) > 1.0) or (abs(mass0 - mass1) > 0.1), (
        f"QPI outputs are suspiciously identical: "
        f"opd=({opd0:.2f}, {opd1:.2f}), mass=({mass0:.2e}, {mass1:.2e})"
    )


# ---- Row flattening -------------------------------------------------------

def test_qpi_batch_to_rows_attaches_candidate_columns():
    field = _field_for_scene()
    entries = run_qpi_for_candidates(
        field, _base_params(), ReconstructionMethod.ASM,
        _ground_truth_candidates(),
        n_sample=1.40, n_medium=1.33, compute_psd=False,
    )
    rows = qpi_batch_to_rows(entries, sample_id="SPL-BATCH",
                             app_version="1.2.0-beta")
    assert len(rows) == 2
    for i, row in enumerate(rows):
        assert row["sample_id"] == "SPL-BATCH"
        assert row["app_version"] == "1.2.0-beta"
        assert row["candidate_rank"] == i
        # candidate_z_mm is a formatted string (CSV-friendly).
        assert "mm" not in row["candidate_z_mm"]
        assert float(row["candidate_z_mm"]) == pytest.approx(
            entries[i].candidate.z_m * 1e3
        )


# ---- CSV roundtrip --------------------------------------------------------

def test_write_qpi_batch_csv_roundtrip(tmp_path):
    field = _field_for_scene()
    entries = run_qpi_for_candidates(
        field, _base_params(), ReconstructionMethod.ASM,
        _ground_truth_candidates(),
        n_sample=1.40, n_medium=1.33, compute_psd=False,
    )
    out = tmp_path / "batch.csv"
    write_qpi_batch_csv(out, entries,
                        sample_id="SPL-BATCH", app_version="1.2.0-beta")
    with out.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 2
    # Extra columns exist and carry the candidate fields.
    assert rows[0]["candidate_rank"] == "0"
    assert rows[1]["candidate_rank"] == "1"
    assert float(rows[0]["candidate_z_mm"]) == pytest.approx(-8.0)
    assert float(rows[1]["candidate_z_mm"]) == pytest.approx(-18.0)
    # Base columns still present in the header (spot-check a few).
    header = rows[0].keys()
    for key in ("sample_id", "timestamp_utc", "opd_range_nm", "Ra_m"):
        assert key in header


def test_write_qpi_batch_csv_empty_writes_header_only(tmp_path):
    """Zero entries: header row must still appear — downstream scripts
    then see an empty-but-valid CSV."""
    out = tmp_path / "empty.csv"
    write_qpi_batch_csv(out, [], sample_id="SPL-X")
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("sample_id,")
    # Last three extras must be present.
    for col in ("candidate_rank", "candidate_z_mm", "candidate_prominence"):
        assert col in lines[0]
