"""Tests for the AI observation / vision layer (core.observe)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import observe


def _synthetic_field(n: int = 128) -> np.ndarray:
    """Reconstructed-like complex field: bump in phase, gaussian amplitude."""
    y, x = np.mgrid[0:n, 0:n]
    r2 = ((x - n / 2) ** 2 + (y - n / 2) ** 2) / (2 * (n / 6) ** 2)
    amp = 0.3 + 0.7 * np.exp(-r2)
    phase = 2.5 * np.exp(-r2)             # a smooth phase bump, < 2π
    return (amp * np.exp(1j * phase)).astype(np.complex64)


# ---------------------------------------------------------------------------
# inspect_reconstruction
# ---------------------------------------------------------------------------

def test_inspect_reconstruction_reports_signal():
    field = _synthetic_field()
    out = observe.inspect_reconstruction(field)
    assert out["ok"] is True
    assert out["shape"] == [128, 128]
    assert out["finite_fraction"] == 1.0
    assert out["amplitude_contrast"] > 0.1        # real object modulation
    assert out["focus_laplacian_variance"] >= 0.0
    assert "signal" in out["verdict"]
    # JSON-friendly: no numpy scalars leak through
    import json
    json.dumps(out)


def test_inspect_reconstruction_flags_empty_field():
    flat = np.ones((32, 32), dtype=np.complex64)   # constant → no contrast
    out = observe.inspect_reconstruction(flat)
    assert out["ok"] is True
    assert out["amplitude_contrast"] == pytest.approx(0.0, abs=1e-6)
    assert "empty" in out["verdict"] or "low amplitude" in out["verdict"]


def test_inspect_reconstruction_flags_nonfinite():
    field = _synthetic_field(32)
    field[0, 0] = np.nan + 1j * np.nan
    out = observe.inspect_reconstruction(field)
    assert out["finite_fraction"] < 1.0
    assert "non-finite" in out["verdict"]


def test_inspect_reconstruction_empty_array():
    out = observe.inspect_reconstruction(np.zeros((0, 0), dtype=np.complex64))
    assert out["ok"] is False


# ---------------------------------------------------------------------------
# inspect_phase_map
# ---------------------------------------------------------------------------

def test_inspect_phase_map_stats_and_opd():
    phase = np.angle(_synthetic_field())     # a real phase map
    out = observe.inspect_phase_map(phase, wavelength_m=632.8e-9,
                                    pixel_size_m=0.088e-6, segment=False)
    assert out["ok"] is True
    assert out["opd_nm"]["count"] == 128 * 128
    # OPD = phase * λ/2π; peak phase ~2.5 rad → OPD ~ 2.5*632.8/(2π) ≈ 252 nm
    assert out["opd_nm"]["max"] == pytest.approx(2.5 * 632.8 / (2 * np.pi),
                                                 rel=0.05)
    assert out["peak_to_valley_rad"] > 2.0


def test_inspect_phase_map_segmentation_counts_cells():
    # Two separated gaussian phase blobs → a GENUINE count reports 2, not the
    # structural 0/1 the old single-largest-binary path gave (B-112).
    n = 128
    y, x = np.mgrid[0:n, 0:n]
    blob1 = 3.0 * np.exp(-(((x - 40) ** 2 + (y - 40) ** 2) / (2 * 8 ** 2)))
    blob2 = 3.0 * np.exp(-(((x - 90) ** 2 + (y - 90) ** 2) / (2 * 8 ** 2)))
    phase = blob1 + blob2
    out = observe.inspect_phase_map(phase, segment=True)
    assert out["ok"] is True
    assert out.get("cell_count", 0) == 2      # both blobs counted
    assert "total_dry_mass_pg" in out


# ---------------------------------------------------------------------------
# inspect_field
# ---------------------------------------------------------------------------

def test_inspect_field_histograms():
    out = observe.inspect_field(_synthetic_field(), bins=16)
    assert out["ok"] is True
    assert len(out["amplitude_hist"]) == 16
    assert len(out["phase_hist"]) == 16
    assert len(out["amplitude_edges"]) == 17
    assert sum(out["amplitude_hist"]) == 128 * 128


# ---------------------------------------------------------------------------
# render_view
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("kind", ["amplitude", "phase", "spectrum", "raw"])
def test_render_view_returns_png(kind):
    png = observe.render_view(_synthetic_field(), kind)
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == _PNG_MAGIC
    assert len(png) > 500          # a real image, not an empty canvas


def test_render_view_depth_from_real_array():
    depth = np.random.default_rng(0).uniform(-2, 2, size=(64, 64))
    png = observe.render_view(depth, "depth", pixel_size_um=0.088)
    assert png[:8] == _PNG_MAGIC


def test_render_view_rejects_bad_kind():
    with pytest.raises(ValueError):
        observe.render_view(_synthetic_field(), "hologram")


def test_render_view_survives_degenerate_input():
    # all-NaN and constant fields must render a blank, not raise.
    nan_field = np.full((32, 32), np.nan, dtype=np.complex64)
    png = observe.render_view(nan_field, "phase")
    assert png[:8] == _PNG_MAGIC
    const = np.ones((32, 32), dtype=np.complex64)
    png2 = observe.render_view(const, "amplitude", pixel_size_um=0.088)
    assert png2[:8] == _PNG_MAGIC


def test_render_view_downsamples_large():
    big = _synthetic_field(4096)
    png = observe.render_view(big, "amplitude", max_size=512)
    assert png[:8] == _PNG_MAGIC     # no OOM / no crash on a big frame
