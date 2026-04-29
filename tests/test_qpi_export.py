"""qpi_export — CSV writer for QPI results.

The module is pure-Python (no pandas, no Qt), so the tests mirror that:
build tiny stand-in objects with the right attributes, write them, and
re-read the CSV with the stdlib ``csv`` module.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional

import pytest

from core.qpi_export import qpi_result_to_row, write_qpi_csv


@dataclass
class _PhaseStats:
    range_nm: float
    mean_nm: float
    std_nm: float


@dataclass
class _Morph:
    area_um2: float
    volume_fL: float
    dry_mass_density_pg_um2: float
    max_height_nm: float
    mean_height_nm: float
    perimeter_um: float
    circularity: float
    aspect_ratio: float
    eccentricity: float
    mean_phase_rad: float
    phase_std_rad: float


@dataclass
class _Roughness:
    Ra: float
    Rq: float
    Rz: float


@dataclass
class _QPIResult:
    phase_stats: Optional[_PhaseStats] = None
    cell_morph: Optional[_Morph] = None
    roughness: Optional[_Roughness] = None
    total_dry_mass_pg: Optional[float] = None
    step_height_m: Optional[float] = None


def _full_result() -> _QPIResult:
    return _QPIResult(
        phase_stats=_PhaseStats(range_nm=420.0, mean_nm=80.0, std_nm=35.0),
        cell_morph=_Morph(
            area_um2=180.0, volume_fL=320.0, dry_mass_density_pg_um2=0.48,
            max_height_nm=5200.0, mean_height_nm=1800.0, perimeter_um=62.0,
            circularity=0.78, aspect_ratio=1.4, eccentricity=0.7,
            mean_phase_rad=0.42, phase_std_rad=0.18,
        ),
        roughness=_Roughness(Ra=1.2e-9, Rq=1.5e-9, Rz=8.0e-9),
        total_dry_mass_pg=155.0,
        step_height_m=1.23e-6,
    )


def test_row_flattens_all_scalar_fields():
    row = qpi_result_to_row(_full_result(), sample_id="SPL-42",
                            app_version="1.0.1-ux")
    assert row["sample_id"] == "SPL-42"
    assert row["app_version"] == "1.0.1-ux"
    assert row["total_dry_mass_pg"] == pytest.approx(155.0)
    # step_height reported in nm (m → *1e9).
    assert row["step_height_nm"] == pytest.approx(1230.0)
    assert row["opd_range_nm"] == pytest.approx(420.0)
    assert row["area_um2"] == pytest.approx(180.0)
    assert row["Ra_m"] == pytest.approx(1.2e-9)


def test_row_is_sparse_when_fields_missing():
    result = _QPIResult(total_dry_mass_pg=42.0)
    row = qpi_result_to_row(result)
    assert row["total_dry_mass_pg"] == pytest.approx(42.0)
    # Missing fields must not appear with a None value — DictWriter then
    # renders them as empty strings. Check by absence.
    assert "area_um2" not in row
    assert "opd_range_nm" not in row
    assert "Ra_m" not in row


def test_row_empty_sample_id_coerces_to_empty_string():
    row = qpi_result_to_row(_QPIResult())
    assert row["sample_id"] == ""
    assert "timestamp_utc" in row


def test_write_creates_header_and_row(tmp_path):
    out = tmp_path / "qpi.csv"
    write_qpi_csv(out, _full_result(), sample_id="SPL-1",
                  app_version="1.0.1-ux")

    with out.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "SPL-1"
    assert row["app_version"] == "1.0.1-ux"
    assert float(row["total_dry_mass_pg"]) == pytest.approx(155.0)
    # Missing fields render as empty strings, not missing columns.
    sparse = _QPIResult()
    out2 = tmp_path / "sparse.csv"
    write_qpi_csv(out2, sparse)
    with out2.open() as fh:
        sparse_row = next(csv.DictReader(fh))
    assert sparse_row["area_um2"] == ""


def test_append_mode_skips_repeat_header(tmp_path):
    out = tmp_path / "batch.csv"
    write_qpi_csv(out, _full_result(), sample_id="SPL-A")
    write_qpi_csv(out, _full_result(), sample_id="SPL-B", append=True)
    write_qpi_csv(out, _full_result(), sample_id="SPL-C", append=True)

    with out.open() as fh:
        lines = fh.readlines()
    # 1 header + 3 data rows.
    assert len(lines) == 4
    # Header appears exactly once.
    assert sum(1 for l in lines if l.startswith("sample_id,")) == 1


def test_append_writes_header_on_empty_file(tmp_path):
    """Appending into an empty file should still emit the header."""
    out = tmp_path / "empty.csv"
    out.touch()
    write_qpi_csv(out, _full_result(), sample_id="SPL-1", append=True)

    with out.open() as fh:
        lines = fh.readlines()
    assert lines[0].startswith("sample_id,")
    assert len(lines) == 2


def test_column_order_is_stable(tmp_path):
    """Downstream LIMS relies on positional column parsing — don't reorder."""
    out = tmp_path / "stable.csv"
    write_qpi_csv(out, _full_result(), sample_id="SPL-X")
    with out.open() as fh:
        header = fh.readline().strip().split(",")
    assert header[:3] == ["sample_id", "timestamp_utc", "app_version"]
    # Cell-morph columns come after the summary scalars.
    assert header.index("area_um2") > header.index("total_dry_mass_pg")
