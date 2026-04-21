"""HTML & PDF report smoke tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.report import generate_html_report, generate_pdf_report


def _sample_images(shape=(64, 64)) -> dict:
    rng = np.random.default_rng(7)
    return {
        "phase_image": rng.normal(0, 1, shape).astype(np.float32),
        "amplitude_image": rng.uniform(0, 1, shape).astype(np.float32),
        "height_image": rng.normal(0, 50e-9, shape).astype(np.float32),
    }


def _sample_params() -> dict:
    return {
        "wavelength_m": 632.8e-9,
        "pixel_size_m": 5e-6,
        "z_m": 2e-4,
        "method": "asm",
        "n": 1.0,
    }


def test_html_report_generates(tmp_path: Path):
    out = tmp_path / "report.html"
    generate_html_report(
        out,
        title="Validation",
        recon_params=_sample_params(),
        **_sample_images(),
    )
    assert out.exists()
    size = out.stat().st_size
    assert size > 10_000, f"html too small: {size} bytes"
    head = out.read_bytes()[:200].decode("utf-8", errors="replace")
    assert ("<!DOCTYPE html>" in head) or ("<html" in head.lower())


def test_pdf_report_generates(tmp_path: Path):
    out = tmp_path / "report.pdf"
    generate_pdf_report(
        out,
        title="Validation",
        recon_params=_sample_params(),
        **_sample_images(),
    )
    assert out.exists()
    size = out.stat().st_size
    assert size > 50_000, f"pdf too small: {size} bytes"
    with open(out, "rb") as f:
        magic = f.read(5)
    assert magic == b"%PDF-", f"missing PDF magic: {magic!r}"


def test_report_skips_none_sections(tmp_path: Path):
    out = tmp_path / "no_images.html"
    generate_html_report(
        out,
        title="Text only",
        recon_params=_sample_params(),
        phase_image=None,
        amplitude_image=None,
        height_image=None,
    )
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    # Figures block only appears when at least one image is supplied.
    assert "Amplitude" not in body
    assert "Phase (radians)" not in body
