"""Contract tests for the synthetic autofocus stress benchmark."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark_af_synthetic_stress import (
    FACTOR_LEVELS,
    StressScenario,
    _build_field,
    _collapse_nearby_z,
    _plane_detection_scores,
    _sphere,
    factorial_scenarios,
    quick_scenarios,
    run_benchmark,
    scenario_catalog,
)
from core.autofocus import FocusMetric
from core.depth_map import (
    find_object_focus_planes,
    find_object_focus_planes_3d,
    find_object_focus_planes_hybrid,
)
from core.reconstruction import ReconstructionMethod
from fixtures.synthetic_hologram import (
    HologramConfig,
    carrier_offset_px,
    validate_offaxis_sampling,
)


def test_catalog_covers_requested_difficulty_axes():
    scenarios = scenario_catalog()
    axes = {scenario.difficulty_axis for scenario in scenarios}
    assert axes == {
        "noise", "distance", "object_count", "object_size",
        "depth_separation",
    }

    noise_levels = {scenario.noise_sigma_peak for scenario in scenarios
                    if scenario.difficulty_axis == "noise"}
    assert len(noise_levels) >= 4
    assert min(noise_levels) == 0.0
    assert max(noise_levels) >= 0.03

    distances = {scenario.expected_z_m[0] for scenario in scenarios
                 if scenario.difficulty_axis == "distance"}
    assert len(distances) >= 4

    object_counts = {len(scenario.spheres) for scenario in scenarios
                     if scenario.difficulty_axis == "object_count"}
    assert object_counts == {1, 2, 3, 5, 7}

    multi_depth = [scenario for scenario in scenarios
                   if scenario.difficulty_axis == "depth_separation"]
    assert multi_depth
    assert len(multi_depth) == 5
    assert any(len(scenario.expected_z_m) > 1 for scenario in multi_depth)


def test_quick_catalog_has_one_scene_per_axis():
    quick = quick_scenarios()
    assert len(quick) == 5
    assert len({scenario.difficulty_axis for scenario in quick}) == 5


def test_factorial_catalog_contains_every_five_axis_combination():
    scenarios = factorial_scenarios()
    assert len(FACTOR_LEVELS) == 5
    assert all(len(levels) == 5 for levels in FACTOR_LEVELS.values())
    assert len(scenarios) == 5 ** 5
    combos = {scenario.factor_levels for scenario in scenarios}
    assert len(combos) == 5 ** 5
    for axis in range(5):
        assert {combo[axis] for combo in combos} == set(range(5))


def test_plane_scores_are_one_to_one_and_penalize_false_positives():
    precision, recall, f1 = _plane_detection_scores(
        detected_z_m=(10e-3, 10.1e-3, 20e-3, 35e-3),
        expected_z_m=(10e-3, 20e-3),
        tolerance_m=0.5e-3,
    )
    assert precision == 0.5  # two correct among four detections
    assert recall == 1.0     # each truth can be consumed only once
    assert f1 == pytest.approx(2.0 / 3.0)


def test_same_depth_objects_collapse_to_one_focus_plane():
    collapsed = _collapse_nearby_z(
        (12.0e-3, 12.2e-3, 11.9e-3, 18.0e-3), 0.5e-3,
    )
    assert collapsed == pytest.approx((12.0e-3, 18.0e-3))


def test_sampling_guard_rejects_old_dc_overlapping_geometry():
    old = HologramConfig(
        shape=(256, 256),
        pixel_m=2.5e-6,
        carrier_freq_m_inv=(50_000.0, 0.0),
    )
    assert carrier_offset_px(old) == pytest.approx(32.0)
    with pytest.raises(ValueError, match="overlaps DC"):
        validate_offaxis_sampling(old, mask_radius_px=40)


def test_companion_sim_geometry_separates_plus_one_from_dc():
    pixel_m = 2.5e-6
    component = 1.0 / (4.0 * pixel_m * np.sqrt(2.0))
    cfg = HologramConfig(
        shape=(256, 256), pixel_m=pixel_m,
        carrier_freq_m_inv=(component, component),
    )
    assert carrier_offset_px(cfg) == pytest.approx(64.0)
    validate_offaxis_sampling(cfg, mask_radius_px=51, guard_px=1)


def test_benchmark_smoke_runs_gradient_descent_on_generated_hologram():
    scenario = next(s for s in scenario_catalog()
                    if s.name == "noise_0.0000")
    rows = run_benchmark(
        [scenario],
        shape=(128, 128),
        budget=20,
        algorithms=("gradient_descent",),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["algorithm"] == "gradient_descent"
    assert row["difficulty_axis"] == "noise"
    assert row["evaluations"] <= 20
    assert row["search_bounds_mm"] == [-5.0, 45.0]
    assert row["scene_tuned_bounds_mm"] == pytest.approx([6.0, 18.0])
    assert row["nearest_plane_error_mm"] >= 0.0
    assert row["primary_plane_error_mm"] == row["nearest_plane_error_mm"]


def test_phase_local_masks_recover_three_spatial_focus_planes():
    scenario = StressScenario(
        name="three_local_planes",
        difficulty_axis="depth_separation",
        spheres=(
            _sphere(11, 6, -55, -45),
            _sphere(11, 12, 50, 0),
            _sphere(11, 20, -45, 55),
        ),
        expected_z_m=(6e-3, 12e-3, 20e-3),
        primary_z_m=None,
    )
    field, base = _build_field(scenario, (256, 256), seed=123)
    planes = find_object_focus_planes(
        field, base, ReconstructionMethod.ASM, -5e-3, 45e-3,
        n_steps=60, metric=FocusMetric.PHASE_VARIANCE,
        confidence_percentile=95.0,
    )
    found = [plane.z_m for plane in planes[:3]]
    assert len(found) == 3
    for expected in scenario.expected_z_m:
        assert min(abs(z - expected) for z in found) < 0.5e-3
    assert all(0 < p.centroid_yx[0] < 255 for p in planes[:3])
    assert all(0 < p.centroid_yx[1] < 255 for p in planes[:3])


def test_3d_persistence_recovers_clean_planes_without_max_projection_ghosts():
    scenario = StressScenario(
        name="three_3d_planes",
        difficulty_axis="depth_separation",
        spheres=(
            _sphere(11, 6, -55, -45),
            _sphere(11, 12, 50, 0),
            _sphere(11, 20, -45, 55),
        ),
        expected_z_m=(6e-3, 12e-3, 20e-3), primary_z_m=None,
    )
    field, base = _build_field(scenario, (256, 256), seed=123)
    planes = find_object_focus_planes_3d(
        field, base, ReconstructionMethod.ASM, -5e-3, 45e-3, n_steps=60,
    )
    found = [plane.z_m for plane in planes]
    assert len(found) == 3
    for expected in scenario.expected_z_m:
        assert min(abs(z - expected) for z in found) < 0.5e-3
    assert all(plane.z_span_steps >= 2 for plane in planes)
    assert all(plane.confidence >= 0.52 for plane in planes)


def test_calibrated_hybrid_keeps_clean_planes_with_high_probability():
    scenario = StressScenario(
        name="three_hybrid_planes",
        difficulty_axis="depth_separation",
        spheres=(
            _sphere(11, 6, -55, -45),
            _sphere(11, 12, 50, 0),
            _sphere(11, 20, -45, 55),
        ),
        expected_z_m=(6e-3, 12e-3, 20e-3), primary_z_m=None,
    )
    field, base = _build_field(scenario, (256, 256), seed=123)
    planes = find_object_focus_planes_hybrid(
        field, base, ReconstructionMethod.ASM, -5e-3, 45e-3, n_steps=60,
    )
    assert len(planes) == 3
    assert all(plane.confidence >= 0.70 for plane in planes)
    for expected in scenario.expected_z_m:
        assert min(abs(plane.z_m - expected) for plane in planes) < 0.5e-3


def test_calibrated_hybrid_can_abstain_instead_of_forcing_a_plane():
    scenario = StressScenario(
        name="unresolved_hybrid",
        difficulty_axis="object_size",
        spheres=(_sphere(4, 12),),
        expected_z_m=(12e-3,), primary_z_m=12e-3,
    )
    field, base = _build_field(scenario, (128, 128), seed=321)
    assert find_object_focus_planes_hybrid(
        field, base, ReconstructionMethod.ASM, -5e-3, 45e-3,
        n_steps=32, probability_threshold=0.999,
    ) == []
