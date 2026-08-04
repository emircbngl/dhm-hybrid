"""v2.0.9 paper-ready output sprint tests.

Three modules in one file because each is small and they share
no state:

* ``core.pdf_report`` — vector PDF render (P1)
* ``core.zenodo_bundle`` — supplementary zip + checksum (P2)
* ``ui2.wcag`` — WCAG contrast audit on theme palettes (P4)

P3 (crash handler v2 wiring) doesn't need a new test surface —
``test_crash_handler.py`` already exercises the install path.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# P1 — Vector PDF report
# ---------------------------------------------------------------------------

from core.pdf_report import PdfReportData, render_pdf_report  # noqa: E402


def test_pdf_report_minimal_writes_valid_pdf(tmp_path):
    """Bare report with just header + recon params should still
    write a non-empty PDF magic-byte file."""
    spec = PdfReportData(
        sample_id="HeLa_test", operator="karin",
        recon_params={"wavelength_nm": 632.8, "z_mm": 12.4},
    )
    out = render_pdf_report(spec, tmp_path / "r.pdf")
    assert out.exists()
    head = out.read_bytes()[:4]
    assert head == b"%PDF"


def test_pdf_report_full_page_with_images_and_profiles(tmp_path):
    """Report with hologram + phase + amplitude + line profiles +
    calibration footer."""
    rng = np.random.default_rng(0)
    spec = PdfReportData(
        sample_id="A549_2026", operator="emir",
        pixel_size_um=0.5,
        hologram=rng.uniform(0, 1, (64, 64)).astype(np.float32),
        phase=rng.uniform(-np.pi, np.pi, (64, 64)).astype(np.float32),
        amplitude=rng.uniform(0, 1, (64, 64)).astype(np.float32),
        recon_params={"wavelength_nm": 632.8, "z_mm": 12.4,
                      "n_medium": 1.33},
        autofocus_z_mm=12.41,
        qpi_opd_range_nm=412.5,
        qpi_total_dry_mass_pg=88.5,
        calibration_status="green",
        calibration_drift_percent=0.4,
        calibration_timestamp="2026-04-26T08:00:00",
    )
    out = render_pdf_report(spec, tmp_path / "full.pdf")
    assert out.exists()
    # PDFs are 1.4+ — sanity-check size > 1 KB so we know the
    # vector content actually landed.
    assert out.stat().st_size > 1024


def test_pdf_report_creates_parent_dir(tmp_path):
    spec = PdfReportData(sample_id="x")
    target = tmp_path / "nested" / "dir" / "r.pdf"
    out = render_pdf_report(spec, target)
    assert out.exists()


def test_pdf_report_with_line_profiles(tmp_path):
    """Profile overlay path. We construct minimal SampledProfile-
    likes (duck typing) so the test doesn't import line_profile —
    the renderer reads attributes only."""
    from core.line_profile import LineProfile, SampledProfile
    p1 = LineProfile(y0=10, x0=0, y1=10, x1=30, label="row 10")
    sp1 = SampledProfile(
        profile=p1,
        distance_px=np.linspace(0, 30, 31),
        values=np.sin(np.linspace(0, 4 * np.pi, 31)).astype(np.float64),
    )
    spec = PdfReportData(
        sample_id="profile_test", operator="karin",
        phase=np.zeros((64, 64), dtype=np.float32),
        line_profiles=[sp1],
    )
    out = render_pdf_report(spec, tmp_path / "p.pdf")
    assert out.exists()


# ---------------------------------------------------------------------------
# P2 — Zenodo bundle
# ---------------------------------------------------------------------------

from core.zenodo_bundle import (  # noqa: E402
    BundleSpec, build_bundle, bundle_manifest,
)


def _fake_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "figs.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def _fake_csv(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    p.write_text("col1,col2\n1,2\n3,4\n")
    return p


def test_build_bundle_creates_zip(tmp_path):
    spec = BundleSpec(
        sample_id="A549",
        operator="karin",
        figures_pdf=_fake_pdf(tmp_path),
        raw_data_csv=_fake_csv(tmp_path),
        params={"wavelength_nm": 632.8},
        notes="test bundle",
    )
    out = build_bundle(spec, tmp_path / "bundle.zip")
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert {"figures.pdf", "raw_data.csv", "params.json",
            "README.md", "checksum.txt"} <= names


def test_bundle_checksum_lists_every_file(tmp_path):
    """checksum.txt must list every file the zip contains, with a
    real hex SHA-256."""
    spec = BundleSpec(
        sample_id="test",
        figures_pdf=_fake_pdf(tmp_path),
        raw_data_csv=_fake_csv(tmp_path),
    )
    out = build_bundle(spec, tmp_path / "b.zip")
    with zipfile.ZipFile(out) as zf:
        checksum = zf.read("checksum.txt").decode()
    lines = [l for l in checksum.splitlines() if l.strip()]
    # Each line: <64-char hex>  <name>
    assert all(len(l.split("  ")[0]) == 64 for l in lines)
    names = {l.split("  ")[1] for l in lines}
    assert "figures.pdf" in names
    assert "raw_data.csv" in names
    assert "params.json" in names
    assert "README.md" in names


def test_bundle_checksum_matches_actual_files(tmp_path):
    """Checksums in checksum.txt should equal the actual SHA-256
    of the file as stored in the zip."""
    spec = BundleSpec(
        sample_id="test",
        figures_pdf=_fake_pdf(tmp_path),
    )
    out = build_bundle(spec, tmp_path / "b.zip")
    with zipfile.ZipFile(out) as zf:
        checksum_text = zf.read("checksum.txt").decode()
        figures_bytes = zf.read("figures.pdf")
    listed = {
        l.split("  ")[1]: l.split("  ")[0]
        for l in checksum_text.splitlines() if l.strip()
    }
    assert listed["figures.pdf"] == hashlib.sha256(
        figures_bytes,
    ).hexdigest()


def test_bundle_params_json_has_caller_dict(tmp_path):
    spec = BundleSpec(
        sample_id="test",
        figures_pdf=_fake_pdf(tmp_path),
        params={"wavelength_nm": 488.0, "z_mm": 12.4},
    )
    out = build_bundle(spec, tmp_path / "b.zip")
    with zipfile.ZipFile(out) as zf:
        params = json.loads(zf.read("params.json"))
    assert params["wavelength_nm"] == 488.0
    assert params["z_mm"] == 12.4
    assert params["sample_id"] == "test"


def test_bundle_readme_names_sample_and_operator(tmp_path):
    spec = BundleSpec(
        sample_id="HeLa_2026",
        operator="karin",
        figures_pdf=_fake_pdf(tmp_path),
    )
    md = bundle_manifest(spec)
    assert "HeLa_2026" in md
    assert "karin" in md


def test_bundle_extras_attached(tmp_path):
    """Caller can drop extra files (a reproduce script, etc.)
    into the bundle by name."""
    extra = tmp_path / "reproduce.py"
    extra.write_text("# pseudo script\n")
    spec = BundleSpec(
        sample_id="test",
        figures_pdf=_fake_pdf(tmp_path),
        extras={"reproduce.py": extra},
    )
    out = build_bundle(spec, tmp_path / "b.zip")
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "reproduce.py" in names


def test_bundle_missing_pdf_raises(tmp_path):
    spec = BundleSpec(sample_id="test")  # no figures_pdf
    with pytest.raises(ValueError, match="figures_pdf"):
        build_bundle(spec, tmp_path / "b.zip")


def test_bundle_missing_attachment_raises(tmp_path):
    spec = BundleSpec(
        sample_id="test",
        figures_pdf=_fake_pdf(tmp_path),
        raw_data_csv=tmp_path / "no_such.csv",
    )
    with pytest.raises(FileNotFoundError):
        build_bundle(spec, tmp_path / "b.zip")


# ---------------------------------------------------------------------------
# P4 — WCAG contrast section removed 2026-07-06 with the ui2 (Dear PyGui)
# frontend retirement: it audited ui2.theme palettes via ui2.wcag, both
# deleted. The living equivalents are ui3.wcag + ui3.design palettes,
# pinned by tests/test_ui3_spine.py::test_all_palettes_pass_wcag.
# ---------------------------------------------------------------------------
