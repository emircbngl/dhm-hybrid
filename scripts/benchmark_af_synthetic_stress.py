"""Synthetic stress benchmark for autofocus search algorithms.

The benchmark deliberately uses ``tests/fixtures/synthetic_hologram.py`` so
the generated scenes stay identical to the repository's pipeline tests.  That
generator is an algorithmic fixture, not a metrology-grade optical simulator;
results measure search robustness under controlled perturbations rather than
real-world DHM accuracy.

Difficulty axes (five levels each):

* Gaussian read-noise sigma as a fraction of peak hologram intensity;
* object-to-camera propagation distance;
* object count;
* object/objectlet radius scale;
* separation between object depths.

For multi-depth scenes a single autofocus result cannot recover every plane.
We therefore report both distance to the designated primary plane (when one is
defined) and distance to the nearest valid object plane.

Run a compact sweep::

    python scripts/benchmark_af_synthetic_stress.py --quick

Run the full catalogue and save machine-readable rows::

    python scripts/benchmark_af_synthetic_stress.py --out results.json

Run all 5^5 combinations, optionally split into reproducible shards::

    python scripts/benchmark_af_synthetic_stress.py --factorial \
      --shard-count 4 --shard-index 0 --out factorial-0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core.autofocus import FocusMetric, autofocus_zscan  # noqa: E402
from core.autofocus.search_adaptive import (  # noqa: E402
    gradient_descent_search,
)
from core.autofocus.search_classic import (  # noqa: E402
    robust_coarse_to_fine_search,
)
from core.depth_map import find_object_focus_planes  # noqa: E402
from core.offaxis import OffAxisParams, extract_complex_field_offaxis  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
)
from fixtures.synthetic_hologram import (  # noqa: E402
    HologramConfig,
    SphereSpec,
    build_hologram,
    validate_offaxis_sampling,
)


@dataclass(frozen=True)
class StressScenario:
    name: str
    difficulty_axis: str
    spheres: tuple[SphereSpec, ...]
    expected_z_m: tuple[float, ...]
    primary_z_m: float | None
    noise_sigma_peak: float = 0.0
    search_margin_m: float = 6e-3
    # 0-based levels: noise, distance, count, size, depth separation.
    factor_levels: tuple[int, int, int, int, int] | None = None

    @property
    def z_bounds_m(self) -> tuple[float, float]:
        return (
            min(self.expected_z_m) - self.search_margin_m,
            max(self.expected_z_m) + self.search_margin_m,
        )


@dataclass(frozen=True)
class _ObjectLocalResult:
    best_z_m: float
    object_planes: tuple[Any, ...]
    evaluations: int


def _sphere(
    radius_um: float,
    z_mm: float,
    y_um: float = 0.0,
    x_um: float = 0.0,
) -> SphereSpec:
    return SphereSpec(
        radius_m=radius_um * 1e-6,
        z_m=z_mm * 1e-3,
        center_yx_m=(y_um * 1e-6, x_um * 1e-6),
        n_sphere=1.40,
        n_medium=1.33,
    )


def scenario_catalog() -> list[StressScenario]:
    """Five-level, one-factor-at-a-time diagnostic catalogue."""
    scenarios: list[StressScenario] = []

    for noise in (0.0, 0.0025, 0.01, 0.03, 0.06):
        scenarios.append(StressScenario(
            name=f"noise_{noise:.4f}",
            difficulty_axis="noise",
            spheres=(_sphere(15, 12),),
            expected_z_m=(12e-3,),
            primary_z_m=12e-3,
            noise_sigma_peak=noise,
        ))

    for z_mm in (3.0, 8.0, 14.0, 22.0, 29.0):
        scenarios.append(StressScenario(
            name=f"distance_{z_mm:.0f}mm",
            difficulty_axis="distance",
            spheres=(_sphere(15, z_mm),),
            expected_z_m=(z_mm * 1e-3,),
            primary_z_m=z_mm * 1e-3,
        ))

    positions = _object_positions_um()
    for count in (1, 2, 3, 5, 7):
        spheres = tuple(_sphere(10, 12, *positions[i]) for i in range(count))
        scenarios.append(StressScenario(
            name=f"objects_{count}",
            difficulty_axis="object_count",
            spheres=spheres,
            expected_z_m=(12e-3,),
            primary_z_m=12e-3,
        ))

    for radius_um in (4.0, 7.0, 10.0, 14.0, 18.0):
        scenarios.append(StressScenario(
            name=f"size_{radius_um:g}um",
            difficulty_axis="object_size",
            spheres=(_sphere(radius_um, 12),),
            expected_z_m=(12e-3,), primary_z_m=12e-3,
        ))

    for separation_mm in (0.0, 2.0, 5.0, 9.0, 14.0):
        z_mm = (12.0, 12.0 + separation_mm / 2.0, 12.0 + separation_mm)
        scenarios.append(StressScenario(
            name=f"depth_separation_{separation_mm:g}mm",
            difficulty_axis="depth_separation",
            spheres=tuple(
                _sphere(10, z, *positions[i]) for i, z in enumerate(z_mm)
            ),
            expected_z_m=tuple(sorted({z * 1e-3 for z in z_mm})),
            primary_z_m=12e-3,
            search_margin_m=4e-3,
        ))
    return scenarios


def _object_positions_um() -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 0.0), (-60.0, -60.0), (-60.0, 60.0),
        (60.0, -60.0), (60.0, 60.0), (0.0, -90.0), (0.0, 90.0),
    )


FACTOR_LEVELS = {
    "noise_sigma_peak": (0.0, 0.0025, 0.01, 0.03, 0.06),
    "distance_mm": (3.0, 8.0, 14.0, 22.0, 29.0),
    "object_count": (1, 2, 3, 5, 7),
    "object_radius_um": (4.0, 7.0, 10.0, 14.0, 18.0),
    "depth_separation_mm": (0.0, 2.0, 5.0, 9.0, 14.0),
}


def factorial_scenarios() -> list[StressScenario]:
    """Return all 5⁵=3125 combinations of the requested difficulty axes."""
    positions = _object_positions_um()
    scenarios: list[StressScenario] = []
    ranges = [range(5) for _ in range(5)]
    for levels in product(*ranges):
        ni, di, ci, si, zi = levels
        noise = FACTOR_LEVELS["noise_sigma_peak"][ni]
        base_z = FACTOR_LEVELS["distance_mm"][di]
        count = int(FACTOR_LEVELS["object_count"][ci])
        radius = FACTOR_LEVELS["object_radius_um"][si]
        spread = FACTOR_LEVELS["depth_separation_mm"][zi]
        if count == 1:
            z_values = np.array([base_z])
        else:
            z_values = np.linspace(base_z, base_z + spread, count)
        # The first object is the main object. Later objects progressively
        # shrink into objectlets while the size-axis scales the whole family.
        scale = (1.0, 0.80, 0.65, 0.52, 0.44, 0.37, 0.31)
        spheres = tuple(
            _sphere(radius * scale[i], float(z_values[i]), *positions[i])
            for i in range(count)
        )
        expected = tuple(sorted({float(z) * 1e-3 for z in z_values}))
        name = "factorial_" + "_".join(str(level + 1) for level in levels)
        scenarios.append(StressScenario(
            name=name,
            difficulty_axis="full_factorial",
            spheres=spheres,
            expected_z_m=expected,
            primary_z_m=base_z * 1e-3,
            noise_sigma_peak=float(noise),
            search_margin_m=4e-3,
            factor_levels=levels,
        ))
    return scenarios


def quick_scenarios() -> list[StressScenario]:
    """One representative from each axis for fast development checks."""
    wanted = {
        "noise_0.0300",
        "distance_29mm",
        "objects_7",
        "size_4um",
        "depth_separation_14mm",
    }
    return [s for s in scenario_catalog() if s.name in wanted]


def _build_field(
    scenario: StressScenario,
    shape: tuple[int, int],
    seed: int,
) -> tuple[np.ndarray, ReconstructionParams]:
    cfg = HologramConfig(
        shape=shape,
        pixel_m=2.5e-6,
        wavelength_m=632.8e-9,
        # Companion off-axis-hologram-sim geometry: four-pixel fringe
        # period, diagonal carrier, 50x object-wave amplitude scaling.
        carrier_freq_m_inv=(
            1.0 / (4.0 * 2.5e-6 * np.sqrt(2.0)),
            1.0 / (4.0 * 2.5e-6 * np.sqrt(2.0)),
        ),
        noise_sigma=scenario.noise_sigma_peak,
        object_amplitude=1.0 / 50.0,
        bit_depth=12,
        composition_mode="multiplane",
    )
    hologram = build_hologram(
        scenario.spheres,
        cfg,
        rng=np.random.default_rng(seed),
    )
    mask_radius = max(8, int(round(min(shape) / 5)))
    validate_offaxis_sampling(cfg, mask_radius, guard_px=1.0)
    fy, fx = cfg.carrier_freq_m_inv
    plus_one_center = (
        shape[0] // 2 + int(round(fy * shape[0] * cfg.pixel_m)),
        shape[1] // 2 + int(round(fx * shape[1] * cfg.pixel_m)),
    )
    field, _ = extract_complex_field_offaxis(
        hologram,
        OffAxisParams(center_yx=plus_one_center, radius=mask_radius),
    )
    base = ReconstructionParams(
        wavelength_m=cfg.wavelength_m,
        pixel_size_m=cfg.pixel_m,
        z_m=0.0,
        n=1.33,
    )
    return field, base


def _algorithm_factories(
    field: np.ndarray,
    base: ReconstructionParams,
    metric: FocusMetric,
    z_min_m: float,
    z_max_m: float,
    budget: int,
) -> dict[str, Callable[[], Any]]:
    method = ReconstructionMethod.ASM

    def gradient_descent():
        return gradient_descent_search(
            field, base, method, metric,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            max_evaluations=budget,
        )

    def robust():
        return robust_coarse_to_fine_search(
            field, base, method, metric,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            n_coarse=budget,
            refine_factor=8,
            smooth_sigma=1.5,
        )

    def zscan():
        z_values = list(np.linspace(z_min_m, z_max_m, budget))
        return autofocus_zscan(field, base, z_values, method, metric)

    def object_local():
        planes = tuple(find_object_focus_planes(
            field, base, method, z_min_m, z_max_m,
            n_steps=budget, metric=metric,
        ))
        best = planes[0].z_m if planes else float("nan")
        return _ObjectLocalResult(best, planes, budget)

    return {
        "gradient_descent": gradient_descent,
        "robust": robust,
        "zscan": zscan,
        "object_local": object_local,
    }


def _plane_detection_scores(
    detected_z_m: Sequence[float],
    expected_z_m: Sequence[float],
    tolerance_m: float,
) -> tuple[float, float, float]:
    """Greedy one-to-one precision/recall/F1 for unordered z detections."""
    remaining_found = set(range(len(detected_z_m)))
    remaining_truth = set(range(len(expected_z_m)))
    pairs = sorted(
        (abs(detected_z_m[i] - expected_z_m[j]), i, j)
        for i in remaining_found for j in remaining_truth
    )
    true_positive = 0
    for error, i, j in pairs:
        if error > tolerance_m:
            break
        if i in remaining_found and j in remaining_truth:
            remaining_found.remove(i)
            remaining_truth.remove(j)
            true_positive += 1
    precision = true_positive / len(detected_z_m) if detected_z_m else 0.0
    recall = true_positive / len(expected_z_m) if expected_z_m else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0 else 0.0
    )
    return precision, recall, f1


def run_benchmark(
    scenarios: Sequence[StressScenario],
    *,
    shape: tuple[int, int] = (256, 256),
    budget: int = 40,
    metrics: Iterable[FocusMetric] = (FocusMetric.LAPLACIAN_VARIANCE,),
    algorithms: Sequence[str] = ("gradient_descent", "robust"),
    hit_tolerance_m: float = 0.5e-3,
    search_bounds_m: tuple[float, float] = (-5e-3, 45e-3),
) -> list[dict[str, Any]]:
    """Run every scene with one shared, ground-truth-independent z range.

    The fixed default acts as an internal safety envelope: callers do not tune
    bounds per scene, so the distance axis remains a genuinely blind search.
    ``StressScenario.z_bounds_m`` is retained as a diagnostic showing what a
    scene-tuned range would have been, but is not used by the algorithms.
    """
    if search_bounds_m[1] <= search_bounds_m[0]:
        raise ValueError("search_bounds_m must be ordered low-to-high")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        # Stable across sharding/order changes; Python's built-in hash is
        # process-randomised, so use CRC32 for reproducible camera noise.
        seed = 10_000 + int(zlib.crc32(scenario.name.encode("utf-8")))
        field, base = _build_field(scenario, shape, seed=seed)
        z_min_m, z_max_m = search_bounds_m
        for metric in metrics:
            factories = _algorithm_factories(
                field, base, metric, z_min_m, z_max_m, budget,
            )
            unknown = sorted(set(algorithms) - set(factories))
            if unknown:
                raise ValueError(f"unknown algorithms: {', '.join(unknown)}")
            for algorithm in algorithms:
                t0 = time.monotonic()
                result = factories[algorithm]()
                runtime_ms = (time.monotonic() - t0) * 1000.0
                best_z_m = float(result.best_z_m)
                object_planes = tuple(getattr(result, "object_planes", ()))
                detected_z_m = tuple(float(item.z_m) for item in object_planes)
                nearest_error_m = (
                    min(abs(best_z_m - z) for z in scenario.expected_z_m)
                    if np.isfinite(best_z_m) else float("inf")
                )
                primary_error_m = (
                    abs(best_z_m - scenario.primary_z_m)
                    if scenario.primary_z_m is not None
                    and np.isfinite(best_z_m) else None
                )
                truth_errors = [
                    min((abs(found - truth) for found in detected_z_m),
                        default=float("inf"))
                    for truth in scenario.expected_z_m
                ]
                plane_precision, plane_recall, plane_f1 = (
                    _plane_detection_scores(
                        detected_z_m, scenario.expected_z_m, hit_tolerance_m,
                    )
                )
                evaluations = int(getattr(
                    result,
                    "evaluations",
                    getattr(result, "total_evaluations", budget),
                ))
                basin_bounds = getattr(result, "basin_bounds_m", None)
                selected_peak = getattr(result, "selected_peak_z_m", None)
                rows.append({
                    "scenario": scenario.name,
                    "difficulty_axis": scenario.difficulty_axis,
                    "algorithm": algorithm,
                    "metric": metric.name,
                    "shape": list(shape),
                    "budget": budget,
                    "object_count": len(scenario.spheres),
                    "object_radii_um": [s.radius_m * 1e6
                                          for s in scenario.spheres],
                    "expected_z_mm": [z * 1e3 for z in scenario.expected_z_m],
                    "primary_z_mm": (scenario.primary_z_m * 1e3
                                     if scenario.primary_z_m is not None else None),
                    "noise_sigma_peak": scenario.noise_sigma_peak,
                    "factor_levels": (
                        [level + 1 for level in scenario.factor_levels]
                        if scenario.factor_levels is not None else None
                    ),
                    "search_bounds_mm": [z_min_m * 1e3, z_max_m * 1e3],
                    "scene_tuned_bounds_mm": [
                        scenario.z_bounds_m[0] * 1e3,
                        scenario.z_bounds_m[1] * 1e3,
                    ],
                    "best_z_mm": best_z_m * 1e3,
                    "detected_object_z_mm": [z * 1e3 for z in detected_z_m],
                    "detected_object_count": len(detected_z_m),
                    "all_plane_mean_error_mm": (
                        float(np.mean(truth_errors)) * 1e3
                        if truth_errors else None
                    ),
                    "all_plane_max_error_mm": (
                        float(np.max(truth_errors)) * 1e3
                        if truth_errors else None
                    ),
                    "plane_precision": plane_precision,
                    "plane_recall": plane_recall,
                    "plane_f1": plane_f1,
                    "selected_coarse_peak_z_mm": (
                        float(selected_peak) * 1e3
                        if selected_peak is not None else None
                    ),
                    "selected_basin_bounds_mm": (
                        [float(basin_bounds[0]) * 1e3,
                         float(basin_bounds[1]) * 1e3]
                        if basin_bounds is not None else None
                    ),
                    "nearest_plane_error_mm": nearest_error_m * 1e3,
                    "primary_plane_error_mm": (
                        primary_error_m * 1e3
                        if primary_error_m is not None else None
                    ),
                    "hit_any_plane": nearest_error_m <= hit_tolerance_m,
                    "hit_primary_plane": (
                        primary_error_m <= hit_tolerance_m
                        if primary_error_m is not None else None
                    ),
                    "evaluations": evaluations,
                    "runtime_ms": runtime_ms,
                })
    return rows


def _print_rows(rows: Sequence[dict[str, Any]]) -> None:
    print(
        f"{'scenario':34s} {'algorithm':18s} {'metric':20s} "
        f"{'best z':>9s} {'near err':>9s} {'primary':>9s} {'evals':>6s}"
    )
    for row in rows:
        primary = row["primary_plane_error_mm"]
        primary_text = "n/a" if primary is None else f"{primary:.3f}"
        print(
            f"{row['scenario']:34s} {row['algorithm']:18s} "
            f"{row['metric']:20s} {row['best_z_mm']:9.3f} "
            f"{row['nearest_plane_error_mm']:9.3f} "
            f"{primary_text:>9s} {row['evaluations']:6d}"
        )


def _print_aggregate(rows: Sequence[dict[str, Any]]) -> None:
    print("\naggregate by difficulty axis")
    print(f"{'axis':20s} {'algorithm':18s} {'median err':>11s} {'hit rate':>9s}")
    keys = sorted({(r["difficulty_axis"], r["algorithm"]) for r in rows})
    for axis, algorithm in keys:
        selected = [r for r in rows
                    if r["difficulty_axis"] == axis
                    and r["algorithm"] == algorithm]
        errors = [r["nearest_plane_error_mm"] for r in selected]
        hit_rate = sum(bool(r["hit_any_plane"]) for r in selected) / len(selected)
        print(
            f"{axis:20s} {algorithm:18s} "
            f"{float(np.median(errors)):11.3f} {hit_rate:9.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--factorial", action="store_true",
        help="Run the full 5^5=3125 combination matrix.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shape", type=int, default=256)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--z-min-mm", type=float, default=-5.0)
    parser.add_argument("--z-max-mm", type=float, default=45.0)
    parser.add_argument(
        "--metrics",
        default="LAPLACIAN_VARIANCE",
        help="Comma-separated FocusMetric names.",
    )
    parser.add_argument(
        "--algorithms",
        default="gradient_descent,robust",
        help="Comma-separated: gradient_descent, robust, zscan, object_local.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    metrics = tuple(FocusMetric[name.strip().upper()]
                    for name in args.metrics.split(",") if name.strip())
    algorithms = tuple(name.strip() for name in args.algorithms.split(",")
                       if name.strip())
    if args.quick and args.factorial:
        parser.error("--quick and --factorial are mutually exclusive")
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        parser.error("shard index must satisfy 0 <= index < shard count")
    scenarios = (
        factorial_scenarios() if args.factorial
        else quick_scenarios() if args.quick
        else scenario_catalog()
    )
    scenarios = scenarios[args.shard_index::args.shard_count]
    if args.limit is not None:
        scenarios = scenarios[:args.limit]
    rows = run_benchmark(
        scenarios,
        shape=(args.shape, args.shape),
        budget=args.budget,
        metrics=metrics,
        algorithms=algorithms,
        search_bounds_m=(args.z_min_mm * 1e-3, args.z_max_mm * 1e-3),
    )
    _print_rows(rows)
    _print_aggregate(rows)
    if args.out is not None:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
