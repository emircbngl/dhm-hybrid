"""Cluster height extraction — confidence-guided per-cluster z.

Takes a depth map and segments it into connected high-confidence
regions, reporting a ``(z_mean, z_std)`` for each. The test checks
that on a two-sphere scene the segmentation yields two clusters
whose lateral centroids sit near the two spheres and whose z values
correlate with the true planes.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from core.autofocus import FocusMetric
from core.depth_map import (
    ClusterHeight,
    DepthMapResult,
    compute_depth_map,
    segment_depth_clusters,
    write_cluster_heights_csv,
)
from core.offaxis import OffAxisParams, extract_complex_field_offaxis
from core.reconstruction import ReconstructionMethod, ReconstructionParams

# Order-independent `from fixtures...` — see B-085.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from fixtures.synthetic_hologram import (  # noqa: E402
    HologramConfig,
    SphereSpec,
    build_hologram,
)


_CFG = HologramConfig(
    shape=(256, 256), pixel_m=2.5e-6, wavelength_m=632.8e-9,
    carrier_freq_m_inv=(50_000.0, 0.0),
)


def _base_params() -> ReconstructionParams:
    return ReconstructionParams(
        wavelength_m=_CFG.wavelength_m,
        pixel_size_m=_CFG.pixel_m, z_m=0.0, n=1.33,
    )


def _two_sphere_depth_map() -> DepthMapResult:
    """Fixture: two spheres at z = 8 / 18 mm, offsets to opposite corners."""
    spheres = [
        SphereSpec(radius_m=30e-6, z_m=8e-3,  center_yx_m=(-80e-6, -60e-6)),
        SphereSpec(radius_m=30e-6, z_m=18e-3, center_yx_m=( 60e-6,  40e-6)),
    ]
    hologram = build_hologram(spheres, _CFG)
    field, _ = extract_complex_field_offaxis(hologram, OffAxisParams(radius=40))
    return compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-22e-3, z_max_m=-3e-3,
        n_steps=40, window_size=5,
        metric=FocusMetric.LAPLACIAN_VARIANCE,
    )


# ---- Basic segmentation ---------------------------------------------------

def test_segment_two_spheres_yields_two_clusters():
    """At a strict confidence threshold (0.4) the map cleanly splits
    into two clusters — one per sphere."""
    result = _two_sphere_depth_map()
    clusters = segment_depth_clusters(
        result, confidence_threshold_frac=0.4, min_area_px=30,
    )
    assert len(clusters) == 2


def test_cluster_centroids_near_sphere_locations():
    """Sphere 1 at (-80, -60) µm → pixel (96, 104); sphere 2 at
    (60, 40) µm → pixel (152, 144). Each cluster's centroid must sit
    within ~20 px of one of those."""
    result = _two_sphere_depth_map()
    clusters = segment_depth_clusters(
        result, confidence_threshold_frac=0.4, min_area_px=30,
    )
    expected = [(96, 104), (152, 144)]
    for c in clusters:
        y, x = c.centroid_yx
        dists = [np.hypot(y - ey, x - ex) for ey, ex in expected]
        assert min(dists) <= 20, (
            f"cluster {c.cluster_id} centroid=({y:.0f},{x:.0f}) "
            f"far from every sphere: min_dist={min(dists):.1f} px"
        )


def test_cluster_z_values_correlate_with_true_spheres():
    """Each cluster's mean z must land within ±8 mm of one of the
    true −z values. Tolerance is loose on purpose: the farther sphere
    (z=18 mm) carries a wider diffraction skirt and the
    confidence-weighted mean drifts toward the scene's focal centre."""
    result = _two_sphere_depth_map()
    clusters = segment_depth_clusters(
        result, confidence_threshold_frac=0.4, min_area_px=30,
    )
    expected_zs_mm = (-8.0, -18.0)
    for c in clusters:
        z_mm = c.z_mean_m * 1e3
        best = min(abs(z_mm - e) for e in expected_zs_mm)
        assert best <= 8.0, (
            f"cluster {c.cluster_id} z={z_mm:+.2f} mm "
            f"not near any of {expected_zs_mm} (best={best:.1f} mm)"
        )


def test_clusters_sorted_largest_first():
    """Downstream reports present the dominant object first."""
    result = _two_sphere_depth_map()
    clusters = segment_depth_clusters(
        result, confidence_threshold_frac=0.4, min_area_px=30,
    )
    areas = [c.area_px for c in clusters]
    assert areas == sorted(areas, reverse=True)


# ---- Edge cases -----------------------------------------------------------

def test_segment_drops_tiny_clusters():
    """A very high ``min_area_px`` should eliminate clusters that
    squeak past a low-threshold scan."""
    result = _two_sphere_depth_map()
    loose = segment_depth_clusters(
        result, confidence_threshold_frac=0.2, min_area_px=10,
    )
    strict = segment_depth_clusters(
        result, confidence_threshold_frac=0.2, min_area_px=500,
    )
    assert len(strict) <= len(loose)


def test_flat_confidence_returns_empty():
    """If every pixel's confidence is zero there's nothing to segment."""
    flat = DepthMapResult(
        z_map=np.zeros((32, 32), dtype=np.float32),
        confidence=np.zeros((32, 32), dtype=np.float32),
        z_values_m=np.linspace(-10e-3, -1e-3, 5, dtype=np.float32),
        metric=FocusMetric.LAPLACIAN_VARIANCE,
        window_size=5,
    )
    assert segment_depth_clusters(flat) == []


def test_invalid_threshold_raises():
    r = _two_sphere_depth_map()
    with pytest.raises(ValueError):
        segment_depth_clusters(r, confidence_threshold_frac=1.5)


def test_invalid_min_area_raises():
    r = _two_sphere_depth_map()
    with pytest.raises(ValueError):
        segment_depth_clusters(r, min_area_px=0)


# ---- CSV export -----------------------------------------------------------

def test_write_cluster_heights_csv_roundtrip(tmp_path):
    clusters = [
        ClusterHeight(
            cluster_id=1, centroid_yx=(50.0, 80.0),
            area_px=1200, z_mean_m=-10.0e-3, z_std_m=0.5e-3,
            mean_confidence=0.85,
        ),
        ClusterHeight(
            cluster_id=2, centroid_yx=(180.0, 140.0),
            area_px=600, z_mean_m=-18.0e-3, z_std_m=1.2e-3,
            mean_confidence=0.72,
        ),
    ]
    out = tmp_path / "clusters.csv"
    write_cluster_heights_csv(
        out, clusters, sample_id="SPL-X", pixel_size_m=2.5e-6,
    )

    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "SPL-X"
    assert int(rows[0]["cluster_id"]) == 1
    assert float(rows[0]["z_mean_mm"]) == pytest.approx(-10.0)
    assert float(rows[0]["z_std_mm"]) == pytest.approx(0.5, abs=1e-3)
    # Area in µm² should be present because pixel_size_m was supplied.
    assert "area_um2" in rows[0]
    assert float(rows[0]["area_um2"]) == pytest.approx(
        1200 * (2.5e-6 * 1e6) ** 2
    )


def test_write_cluster_heights_csv_without_pixel_size_omits_area_um2(tmp_path):
    clusters = [
        ClusterHeight(
            cluster_id=1, centroid_yx=(10.0, 10.0),
            area_px=100, z_mean_m=-5e-3, z_std_m=0.1e-3,
            mean_confidence=0.9,
        ),
    ]
    out = tmp_path / "clusters_nopx.csv"
    write_cluster_heights_csv(out, clusters, sample_id="SPL-Y")
    header = out.read_text().splitlines()[0]
    assert "area_um2" not in header
    assert "area_px" in header
