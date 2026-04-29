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
# P4 — WCAG contrast
# ---------------------------------------------------------------------------

from ui2.theme import PALETTES  # noqa: E402
from ui2.wcag import (  # noqa: E402
    AA_LARGE, AA_NORMAL, audit_palette,
    contrast_ratio, find_aa_failures, relative_luminance,
)


def test_relative_luminance_known_endpoints():
    # Black = 0, white = 1.0.
    assert relative_luminance((0, 0, 0)) == pytest.approx(0.0)
    assert relative_luminance((255, 255, 255)) == pytest.approx(1.0)


def test_contrast_ratio_endpoints():
    # Pure black vs pure white = 21.0, identical = 1.0.
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) \
        == pytest.approx(21.0, rel=1e-3)
    assert contrast_ratio((128, 128, 128), (128, 128, 128)) \
        == pytest.approx(1.0, abs=1e-9)


def test_contrast_ratio_symmetric():
    r1 = contrast_ratio((52, 120, 198), (252, 253, 254))
    r2 = contrast_ratio((252, 253, 254), (52, 120, 198))
    assert r1 == pytest.approx(r2)


def test_contrast_known_aa_pass_pair():
    """Lab-checked: white text on a mid-grey panel passes AA."""
    ratio = contrast_ratio((255, 255, 255), (60, 60, 60))
    assert ratio >= AA_NORMAL


def test_contrast_known_aa_fail_pair():
    ratio = contrast_ratio((180, 180, 180), (160, 160, 160))
    assert ratio < AA_NORMAL


# ---- Theme audits — pin palette compliance ---------------------------

def test_dark_theme_passes_aa_for_main_text_pairs():
    """Body text + critical state colours on panel_bg must pass
    WCAG AA (4.5:1). text_muted is intentionally lower-contrast
    (it's a supporting role, ≥3:1 is the rule for it)."""
    findings = audit_palette(PALETTES["dark"])
    body_text = next(f for f in findings
                     if f.role_fg == "text" and f.role_bg == "panel_bg")
    assert body_text.ratio >= AA_NORMAL


def test_light_theme_passes_aa_for_main_text_pairs():
    findings = audit_palette(PALETTES["light"])
    body_text = next(f for f in findings
                     if f.role_fg == "text" and f.role_bg == "panel_bg")
    assert body_text.ratio >= AA_NORMAL


def test_high_contrast_theme_dominates_dark():
    """The high-contrast palette must score strictly higher than
    dark for body text — that's the whole point of having it."""
    dark_text = next(
        f for f in audit_palette(PALETTES["dark"])
        if f.role_fg == "text" and f.role_bg == "panel_bg"
    )
    hc_text = next(
        f for f in audit_palette(PALETTES["high_contrast"])
        if f.role_fg == "text" and f.role_bg == "panel_bg"
    )
    assert hc_text.ratio > dark_text.ratio


def test_high_contrast_theme_passes_aaa_body_text():
    """high_contrast is the accessibility theme — body text must
    clear AAA (7:1), not just AA. v2.0.6 introduced this palette
    explicitly for low-vision operators; pin the AAA gate."""
    findings = audit_palette(PALETTES["high_contrast"])
    body_text = next(f for f in findings
                     if f.role_fg == "text" and f.role_bg == "panel_bg")
    assert body_text.ratio >= 7.0


def test_state_colours_clear_aa_large_on_panel_bg():
    """success / warn / danger on panel_bg are typically used as
    icon overlays + 14 pt+ semi-bold labels. AA-large (3.0)
    threshold is the right gate."""
    for theme_name in ("dark", "light", "high_contrast"):
        findings = audit_palette(PALETTES[theme_name])
        for role in ("success", "warn", "danger"):
            f = next(x for x in findings
                     if x.role_fg == role and x.role_bg == "panel_bg")
            assert f.passes_aa_large, (
                f"{theme_name}/{role} on panel_bg: ratio={f.ratio:.2f} "
                f"< AA_LARGE ({AA_LARGE}). v2.0.6 lessons.md fixed "
                f"this once; if it regressed, look there for hue + "
                f"luminance targets."
            )


def test_find_aa_failures_returns_only_failing_rows():
    findings = audit_palette(PALETTES["high_contrast"])
    failures = find_aa_failures(findings)
    for f in failures:
        assert not f.passes_aa_normal


def test_audit_palette_custom_pairs_supported():
    """Caller can pass a custom (fg, bg) sequence — useful for
    testing icon colour pairs that aren't in the default set."""
    findings = audit_palette(
        PALETTES["dark"],
        text_pairs=[("text", "border")],
    )
    assert len(findings) == 1
    assert findings[0].role_fg == "text"
    assert findings[0].role_bg == "border"
