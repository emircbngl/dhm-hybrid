"""Depth map — per-pixel best-focus z against synthetic ground truth.

The suite uses the same sphere generator as the stress tests. Ground
truth for depth-map accuracy:

* Lateral location of each sphere → we know where to sample the map.
* True ``z`` of each sphere → we know what value to expect at that
  location.

Tolerances are loose enough to survive scan-step discretisation and
lateral diffraction skirts (the sphere's refocused silhouette spreads
over several pixels). Any regression that halves accuracy trips the
test.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.autofocus import FocusMetric
from core.depth_map import (
    DepthMapResult,
    compute_depth_map,
    mask_low_confidence,
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
    single_sphere_hologram,
)


_CFG = HologramConfig(
    shape=(256, 256), pixel_m=2.5e-6, wavelength_m=632.8e-9,
    carrier_freq_m_inv=(50_000.0, 0.0),
)
_OFFAXIS = OffAxisParams(radius=40)


def _base_params() -> ReconstructionParams:
    return ReconstructionParams(
        wavelength_m=_CFG.wavelength_m,
        pixel_size_m=_CFG.pixel_m, z_m=0.0, n=1.33,
    )


def _extract(hologram: np.ndarray) -> np.ndarray:
    field, _ = extract_complex_field_offaxis(hologram, _OFFAXIS)
    return field


# ---- 1. Output structure ---------------------------------------------------

def test_depth_map_result_shape_and_fields():
    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=12e-3, config=_CFG,
    )
    field = _extract(hologram)
    result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3,
        n_steps=30, window_size=5,
        metric=FocusMetric.LAPLACIAN_VARIANCE,
    )
    assert isinstance(result, DepthMapResult)
    assert result.z_map.shape == _CFG.shape
    assert result.confidence.shape == _CFG.shape
    assert result.z_values_m.shape == (30,)
    assert result.metric is FocusMetric.LAPLACIAN_VARIANCE
    assert result.window_size == 5
    # All z values should lie inside the scan range.
    assert result.z_map.min() >= -20e-3 - 1e-9
    assert result.z_map.max() <= -1e-3 + 1e-9


# ---- 2. Single sphere — depth at sphere location ≈ true z -----------------

def test_depth_map_single_sphere_recovers_z_at_sphere_location():
    """At the sphere's lateral centre the depth map should read
    approximately -true_z within ±3 mm (scan step ~0.5 mm plus a few
    mm of diffraction-skirt bleed)."""
    z_true_mm = 12.0
    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=z_true_mm * 1e-3, config=_CFG,
    )
    field = _extract(hologram)
    result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3, n_steps=40, window_size=5,
    )
    cy, cx = 128, 128
    z_at_center_mm = float(
        np.median(result.z_map[cy - 4:cy + 4, cx - 4:cx + 4])
    ) * 1e3
    assert abs(z_at_center_mm - (-z_true_mm)) <= 3.0, (
        f"depth at centre = {z_at_center_mm:+.2f} mm, want {-z_true_mm:+.2f}"
    )


# ---- 3. Multi-sphere — each sphere's location shows its own z -------------

def test_depth_map_multi_sphere_resolves_per_object_z():
    """Two spheres at different z and different lateral offsets. The
    depth map must read approximately each sphere's true z at its own
    lateral location — this is the whole point of the feature."""
    spheres = [
        SphereSpec(radius_m=30e-6, z_m=8e-3,  center_yx_m=(-80e-6, -60e-6)),
        SphereSpec(radius_m=30e-6, z_m=18e-3, center_yx_m=( 60e-6,  40e-6)),
    ]
    hologram = build_hologram(spheres, _CFG)
    field = _extract(hologram)
    result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-22e-3, z_max_m=-3e-3, n_steps=40, window_size=5,
    )

    cy, cx = 128, 128
    # Convert the sphere's lateral metre offset to pixel coords.
    # pixel_m = 2.5 µm, so 80 µm = 32 px, 60 µm = 24 px, etc.
    s1_yx = (cy - 32, cx - 24)
    s2_yx = (cy + 24, cx + 16)

    def _sample(zmap, y, x, r=4):
        return float(np.median(zmap[y - r:y + r, x - r:x + r])) * 1e3

    z1_mm = _sample(result.z_map, *s1_yx)
    z2_mm = _sample(result.z_map, *s2_yx)

    # ±4 mm absorbs scan-step granularity (~0.5 mm/step at 40 steps over
    # 19 mm) plus the Fresnel diffraction skirt, which spreads an
    # out-of-plane sphere's signal across several pixels — the *peak*
    # of that spread doesn't always land on the geometric centre.
    assert abs(z1_mm - (-8.0)) <= 4.0, (
        f"sphere 1: depth {z1_mm:+.2f} mm, want -8.00"
    )
    assert abs(z2_mm - (-18.0)) <= 4.0, (
        f"sphere 2: depth {z2_mm:+.2f} mm, want -18.00"
    )


# ---- 4. Confidence mask --------------------------------------------------

def test_mask_low_confidence_removes_weak_pixels():
    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=12e-3, config=_CFG,
    )
    field = _extract(hologram)
    result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3, n_steps=30, window_size=5,
    )
    masked = mask_low_confidence(result, threshold_frac=0.3)
    # Some pixels should remain finite (at or near the sphere),
    # and some should be NaN (background, low confidence).
    n_valid = int(np.sum(np.isfinite(masked)))
    n_total = masked.size
    assert 0 < n_valid < n_total, (
        f"mask produced {n_valid}/{n_total} valid pixels "
        f"(expected a mix)"
    )


def test_mask_low_confidence_rejects_bad_threshold():
    result = DepthMapResult(
        z_map=np.zeros((4, 4), dtype=np.float32),
        confidence=np.ones((4, 4), dtype=np.float32),
        z_values_m=np.array([0.0, 1.0], dtype=np.float32),
        metric=FocusMetric.LAPLACIAN_VARIANCE,
        window_size=3,
    )
    with pytest.raises(ValueError):
        mask_low_confidence(result, threshold_frac=1.5)


# ---- 5. Defensive API -----------------------------------------------------

def test_compute_depth_map_rejects_inverted_range():
    field = np.ones(_CFG.shape, dtype=np.complex64)
    with pytest.raises(ValueError):
        compute_depth_map(
            field, _base_params(), ReconstructionMethod.ASM,
            z_min_m=-1e-3, z_max_m=-10e-3, n_steps=10, window_size=5,
        )


def test_compute_depth_map_rejects_tiny_window():
    field = np.ones(_CFG.shape, dtype=np.complex64)
    with pytest.raises(ValueError):
        compute_depth_map(
            field, _base_params(), ReconstructionMethod.ASM,
            z_min_m=-20e-3, z_max_m=-1e-3, n_steps=10, window_size=1,
        )


def test_compute_depth_map_unsupported_metric_raises():
    field = np.ones(_CFG.shape, dtype=np.complex64)
    with pytest.raises(ValueError, match="no local form"):
        compute_depth_map(
            field, _base_params(), ReconstructionMethod.ASM,
            z_min_m=-20e-3, z_max_m=-1e-3, n_steps=10, window_size=5,
            metric=FocusMetric.ENTROPY,
        )


# ---- 6. TENENGRAD local form ----------------------------------------------

def test_depth_map_npz_roundtrip(tmp_path):
    """Writing + re-loading must preserve every field bit-for-bit."""
    from core.depth_map import write_depth_map_npz

    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=12e-3, config=_CFG,
    )
    field = _extract(hologram)
    result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3, n_steps=20, window_size=5,
    )
    out = tmp_path / "depth.npz"
    write_depth_map_npz(out, result,
                        sample_id="SPL-X", app_version="1.2.0-alpha")

    loaded = np.load(out, allow_pickle=True)
    assert np.array_equal(loaded["z_map"], result.z_map)
    assert np.array_equal(loaded["confidence"], result.confidence)
    assert np.array_equal(loaded["z_values_m"], result.z_values_m)
    assert str(loaded["metric"]) == "LAPLACIAN_VARIANCE"
    assert int(loaded["window_size"]) == 5
    assert str(loaded["sample_id"]) == "SPL-X"
    assert str(loaded["app_version"]) == "1.2.0-alpha"


def test_depth_map_csv_has_header_and_stride_reduces_rows(tmp_path):
    """CSV export must be rectangular; a stride > 1 must yield
    proportionally fewer rows."""
    import csv
    from core.depth_map import write_depth_map_csv

    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=12e-3, config=_CFG,
    )
    field = _extract(hologram)
    result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3, n_steps=15, window_size=5,
    )

    dense = tmp_path / "dense.csv"
    sparse = tmp_path / "sparse.csv"
    write_depth_map_csv(dense, result, sample_id="SPL-1", stride=8)
    write_depth_map_csv(sparse, result, sample_id="SPL-1", stride=32)

    def _rows(p):
        with p.open() as fh:
            return list(csv.reader(fh))

    dense_rows = _rows(dense)
    sparse_rows = _rows(sparse)
    assert dense_rows[0] == ["sample_id", "y_px", "x_px", "z_mm", "confidence"]
    # Stride 32 produces ~(32/8)² ≈ 16× fewer rows than stride 8.
    assert len(sparse_rows) < len(dense_rows)
    # First data row must have the sample id.
    assert dense_rows[1][0] == "SPL-1"


def test_write_tomography_bundle_produces_all_artefacts(tmp_path):
    """Bundle writer: depth NPZ + depth CSV + cluster CSV +
    (optional) QPI batch CSV, all in one directory with the same
    base_name prefix."""
    from core.autofocus import FocusCandidate
    from core.depth_map import (
        ClusterHeight,
        compute_depth_map,
        write_tomography_bundle,
    )
    from core.qpi_batch import QPIBatchEntry

    # Build a small depth map from a single-sphere hologram so the
    # test runs fast.
    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=12e-3, config=_CFG,
    )
    field = _extract(hologram)
    depth_result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3, n_steps=12, window_size=5,
    )
    clusters = [
        ClusterHeight(
            cluster_id=1, centroid_yx=(64.0, 64.0),
            area_px=500, z_mean_m=-12e-3, z_std_m=0.3e-3,
            mean_confidence=0.8,
        ),
    ]
    # Fake QPI entries — construct with a minimal QPIResult sibling.
    class _PhaseStats:
        range_nm = 320.0
        mean_nm = 100.0
        std_nm = 40.0
    class _QPIResult:
        total_dry_mass_pg = 25.0
        step_height_m = 1.1e-6
        phase_stats = _PhaseStats()
        cell_morph = None
        roughness = None
    qpi_entries = [
        QPIBatchEntry(
            candidate=FocusCandidate(
                z_m=-12e-3, score=0.9, prominence=0.5, rank=0,
            ),
            qpi_result=_QPIResult(),
        ),
    ]

    written = write_tomography_bundle(
        tmp_path, depth_result, clusters, qpi_entries,
        base_name="bundle", sample_id="SPL-B",
        app_version="1.3-polish", pixel_size_m=2.5e-6,
    )
    assert len(written) == 4
    names = {p.name for p in written}
    assert names == {
        "bundle_depth.npz",
        "bundle_depth.csv",
        "bundle_clusters.csv",
        "bundle_qpi_batch.csv",
    }
    # Every file must actually exist and be non-empty.
    for p in written:
        assert p.exists(), f"{p} not written"
        assert p.stat().st_size > 0, f"{p} is empty"


def test_write_tomography_bundle_skips_qpi_when_empty(tmp_path):
    """Empty QPI list → only three files (no qpi_batch CSV)."""
    from core.depth_map import (
        compute_depth_map,
        write_tomography_bundle,
    )

    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=12e-3, config=_CFG,
    )
    field = _extract(hologram)
    depth_result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3, n_steps=12, window_size=5,
    )
    written = write_tomography_bundle(
        tmp_path, depth_result, [], None,
        base_name="bundle", sample_id="SPL-B",
    )
    assert len(written) == 3
    assert not any(p.name.endswith("qpi_batch.csv") for p in written)


def test_write_tomography_bundle_creates_missing_directory(tmp_path):
    """``directory`` doesn't need to exist — the writer creates it."""
    from core.depth_map import (
        compute_depth_map,
        write_tomography_bundle,
    )

    target = tmp_path / "new_subdir" / "nested"
    assert not target.exists()

    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=12e-3, config=_CFG,
    )
    field = _extract(hologram)
    depth_result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3, n_steps=12, window_size=5,
    )
    written = write_tomography_bundle(
        target, depth_result, [], None,
        base_name="bundle",
    )
    assert target.exists()
    assert all(p.parent == target for p in written)


def test_depth_map_tenengrad_also_produces_valid_output():
    """Both LAPLACIAN_VARIANCE and TENENGRAD have local-form kernels
    registered. TENENGRAD responds to amplitude gradients — exercise
    the alternative path so regressions in its kernel are caught."""
    hologram = single_sphere_hologram(
        radius_m=30e-6, z_m=12e-3, config=_CFG,
    )
    field = _extract(hologram)
    result = compute_depth_map(
        field, _base_params(), ReconstructionMethod.ASM,
        z_min_m=-20e-3, z_max_m=-1e-3,
        n_steps=30, window_size=5,
        metric=FocusMetric.TENENGRAD,
    )
    assert result.metric is FocusMetric.TENENGRAD
    assert np.all(np.isfinite(result.z_map))
    assert np.all(np.isfinite(result.confidence))
