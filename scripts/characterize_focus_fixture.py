"""Characterise the synthetic fixture's own focus response.

Every autofocus benchmark in this repository scores a search algorithm against
a ``0.5 mm`` hit tolerance.  That number was never derived from a measurement —
it is an assumption.  This script measures what the fixture can actually
support, so the tolerance stops being a guess and search results can be read
against a real ceiling.

For each scene it scans the full search range densely and records, per focus
metric, two curves:

* **global** — the metric over the whole frame, exactly what
  ``autofocus_zscan`` and ``gradient_descent_search`` optimise;
* **oracle-masked** — the metric restricted to the true object footprint.
  No detector can beat this, because it is handed the answer to
  "which pixels are the object?".  It is therefore an upper bound on any
  object-local method (``find_object_focus_planes*``).

Reported per curve:

* ``argmax`` error against the known plane;
* whether the extremum is interior or pinned to a range edge;
* ``fwhm_mm`` — full width at half prominence of the peak.  This is the
  fixture's empirical axial response width; half of it is the smallest
  tolerance a scan can honestly claim.

The generator is a projected thin-phase-screen fixture that calls the same
``core.reconstruction.propagate`` used to reconstruct (an inverse crime), so
these numbers bound *this fixture*, not physical DHM metrology.

Compact run::

    python scripts/characterize_focus_fixture.py --quick

Full grid with machine-readable rows::

    python scripts/characterize_focus_fixture.py --out characterization.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core.autofocus.metrics import FocusMetric, _calc_metric, _is_minimize  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)

# Reused deliberately: sharing the benchmark's scene builder is what keeps the
# characterisation and the scored runs on identical optics.  A local copy would
# silently drift from the thing being characterised.
from benchmark_af_synthetic_stress import (  # noqa: E402
    StressScenario,
    _build_field,
    _sphere,
)

DEFAULT_METRICS: tuple[FocusMetric, ...] = (
    FocusMetric.LAPLACIAN_VARIANCE,
    FocusMetric.TENENGRAD,
    FocusMetric.PHASE_VARIANCE,
    FocusMetric.NORMALIZED_VARIANCE,
    FocusMetric.SPECTRAL_ENERGY,
)


@dataclass(frozen=True)
class CurveStats:
    """Shape of one metric-versus-z curve."""

    argmax_z_m: float
    error_m: float
    edge_pinned: bool
    fwhm_m: float | None
    prominence_ratio: float


def _oriented(values: np.ndarray, metric: FocusMetric) -> np.ndarray:
    """Return a curve whose focus point is always a maximum."""
    return -values if _is_minimize(metric) else values


def _curve_stats(
    z_values: np.ndarray,
    raw: Sequence[float],
    metric: FocusMetric,
    truth_z_m: float,
) -> CurveStats:
    values = _oriented(np.asarray(raw, dtype=np.float64), metric)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=-np.inf)
    peak = int(values.argmax())
    baseline = float(np.percentile(values, 10.0))
    height = float(values[peak]) - baseline
    prominence_ratio = height / max(abs(float(values[peak])), 1e-30)

    fwhm_m: float | None = None
    if height > 0.0:
        half = baseline + 0.5 * height
        left = peak
        while left > 0 and values[left] >= half:
            left -= 1
        right = peak
        last = values.size - 1
        while right < last and values[right] >= half:
            right += 1
        # Only a genuinely interior half-crossing describes a peak width.
        if values[left] < half and values[right] < half:
            fwhm_m = float(z_values[right] - z_values[left])

    return CurveStats(
        argmax_z_m=float(z_values[peak]),
        error_m=float(z_values[peak] - truth_z_m),
        edge_pinned=peak in (0, values.size - 1),
        fwhm_m=fwhm_m,
        prominence_ratio=prominence_ratio,
    )


def _oracle_slice(
    scenario: StressScenario,
    shape: tuple[int, int],
    pixel_m: float,
    margin_px: int = 4,
) -> tuple[slice, slice]:
    """Bounding box of the true object footprint, in pixels.

    This is the oracle: it uses the scene's ground truth, so it is only valid
    as an upper bound, never as a detector.
    """
    ys: list[float] = []
    xs: list[float] = []
    for sphere in scenario.spheres:
        cy, cx = sphere.center_yx_m
        radius_px = sphere.radius_m / pixel_m
        ys.extend([cy / pixel_m - radius_px, cy / pixel_m + radius_px])
        xs.extend([cx / pixel_m - radius_px, cx / pixel_m + radius_px])
    cy0, cx0 = shape[0] / 2.0, shape[1] / 2.0
    y_lo = max(0, int(np.floor(cy0 + min(ys))) - margin_px)
    x_lo = max(0, int(np.floor(cx0 + min(xs))) - margin_px)
    y_hi = min(shape[0], max(int(np.ceil(cy0 + max(ys))) + margin_px, y_lo + 3))
    x_hi = min(shape[1], max(int(np.ceil(cx0 + max(xs))) + margin_px, x_lo + 3))
    return slice(y_lo, y_hi), slice(x_lo, x_hi)


def characterize_scene(
    scenario: StressScenario,
    *,
    shape: tuple[int, int],
    z_min_m: float,
    z_max_m: float,
    steps: int,
    metrics: Sequence[FocusMetric],
    seed: int,
) -> list[dict]:
    """Dense global and oracle-masked focus curves for one scene."""
    field, base = _build_field(scenario, shape, seed=seed)
    roi = _oracle_slice(scenario, shape, base.pixel_size_m)
    z_values = np.linspace(z_min_m, z_max_m, steps)
    truth_z_m = float(scenario.expected_z_m[0])

    global_curves: dict[FocusMetric, list[float]] = {m: [] for m in metrics}
    masked_curves: dict[FocusMetric, list[float]] = {m: [] for m in metrics}
    for z in z_values:
        params = ReconstructionParams(
            wavelength_m=base.wavelength_m,
            pixel_size_m=base.pixel_size_m,
            z_m=float(z),
            n=base.n,
        )
        recon = propagate(field, params, ReconstructionMethod.ASM)
        window = recon[roi]
        for metric in metrics:
            global_curves[metric].append(_calc_metric(recon, metric))
            masked_curves[metric].append(_calc_metric(window, metric))

    rows: list[dict] = []
    for metric in metrics:
        g = _curve_stats(z_values, global_curves[metric], metric, truth_z_m)
        m = _curve_stats(z_values, masked_curves[metric], metric, truth_z_m)
        rows.append({
            "scenario": scenario.name,
            "metric": metric.value,
            "shape": list(shape),
            "steps": steps,
            "scan_step_mm": float(z_values[1] - z_values[0]) * 1e3,
            "truth_z_mm": truth_z_m * 1e3,
            "noise_sigma_peak": scenario.noise_sigma_peak,
            "object_radius_um": scenario.spheres[0].radius_m * 1e6,
            "roi_px": [roi[0].start, roi[0].stop, roi[1].start, roi[1].stop],
            "global_argmax_mm": g.argmax_z_m * 1e3,
            "global_error_mm": g.error_m * 1e3,
            "global_edge_pinned": g.edge_pinned,
            "global_fwhm_mm": None if g.fwhm_m is None else g.fwhm_m * 1e3,
            "masked_argmax_mm": m.argmax_z_m * 1e3,
            "masked_error_mm": m.error_m * 1e3,
            "masked_edge_pinned": m.edge_pinned,
            "masked_fwhm_mm": None if m.fwhm_m is None else m.fwhm_m * 1e3,
            "masked_prominence_ratio": m.prominence_ratio,
        })
    return rows


def characterization_grid(
    radii_um: Sequence[float],
    noises: Sequence[float],
    distances_mm: Sequence[float],
) -> list[StressScenario]:
    """Single-sphere scenes spanning the fixture's difficulty axes.

    One sphere per scene on purpose: the question is what the *metric* does at
    a known plane, so multi-object confusion is deliberately excluded.
    """
    scenarios: list[StressScenario] = []
    for radius_um, noise, z_mm in product(radii_um, noises, distances_mm):
        scenarios.append(StressScenario(
            name=f"char_r{radius_um:g}um_n{noise:.4f}_z{z_mm:g}mm",
            difficulty_axis="characterization",
            spheres=(_sphere(radius_um, z_mm),),
            expected_z_m=(z_mm * 1e-3,),
            primary_z_m=z_mm * 1e-3,
            noise_sigma_peak=float(noise),
        ))
    return scenarios


def summarize(rows: Iterable[dict]) -> dict:
    """Aggregate per metric, and derive a tolerance from the measurement."""
    rows = list(rows)
    per_metric: dict[str, dict] = {}
    for metric in sorted({row["metric"] for row in rows}):
        subset = [row for row in rows if row["metric"] == metric]
        g_err = np.abs([row["global_error_mm"] for row in subset])
        m_err = np.abs([row["masked_error_mm"] for row in subset])
        fwhm = [row["masked_fwhm_mm"] for row in subset
                if row["masked_fwhm_mm"] is not None]
        per_metric[metric] = {
            "scenes": len(subset),
            "global_median_abs_error_mm": float(np.median(g_err)),
            "global_edge_pinned_fraction": float(np.mean(
                [bool(row["global_edge_pinned"]) for row in subset])),
            "global_hit_rate_0p5mm": float(np.mean(g_err <= 0.5)),
            "masked_median_abs_error_mm": float(np.median(m_err)),
            "masked_p90_abs_error_mm": float(np.percentile(m_err, 90.0)),
            "masked_edge_pinned_fraction": float(np.mean(
                [bool(row["masked_edge_pinned"]) for row in subset])),
            "masked_hit_rate_0p5mm": float(np.mean(m_err <= 0.5)),
            "masked_median_fwhm_mm": (
                float(np.median(fwhm)) if fwhm else None),
            # Fraction of scenes where the masked curve has a real interior
            # peak at all.  Scenes without one are unmeasurable, not merely
            # mis-measured — they belong in coverage, never in the tolerance.
            "masked_resolvable_fraction": float(len(fwhm) / len(subset)),
        }

    best = max(
        per_metric.items(),
        key=lambda kv: kv[1]["masked_hit_rate_0p5mm"],
    )
    # Tolerance describes resolution, so it is derived from the TYPICAL scene:
    # half the measured axial response width, floored by the median masked
    # error.  Deliberately not p90 — the upper tail is dominated by scenes
    # that carry no axial signal (sub-resolution objects, swamped SNR), and
    # folding those in would inflate the tolerance to tens of millimetres and
    # make every method look accurate.  Those scenes are reported separately
    # as `masked_resolvable_fraction`.
    best_stats = best[1]
    half_fwhm = (
        0.5 * best_stats["masked_median_fwhm_mm"]
        if best_stats["masked_median_fwhm_mm"] is not None else 0.0
    )
    suggested = max(half_fwhm, best_stats["masked_median_abs_error_mm"])
    return {
        "per_metric": per_metric,
        "best_masked_metric": best[0],
        "suggested_tolerance_mm": suggested,
        "unresolvable_fraction": 1.0 - best_stats["masked_resolvable_fraction"],
        "current_benchmark_tolerance_mm": 0.5,
        "tolerance_is_optimistic": suggested > 0.5,
    }


def _print_report(rows: list[dict], summary: dict) -> None:
    print(f"\nScenes: {len({row['scenario'] for row in rows})}   "
          f"rows: {len(rows)}   "
          f"scan step: {rows[0]['scan_step_mm']:.4f} mm")
    header = (f"{'metric':<20}{'glob|err|':>10}{'glob hit':>10}"
              f"{'mask|err|':>11}{'mask hit':>10}"
              f"{'mask FWHM':>11}{'resolvable':>12}")
    print(header)
    print("-" * len(header))
    for metric, stats in summary["per_metric"].items():
        fwhm = stats["masked_median_fwhm_mm"]
        fwhm_text = "        n/a" if fwhm is None else f"{fwhm:>11.3f}"
        print(
            f"{metric:<20}"
            f"{stats['global_median_abs_error_mm']:>10.3f}"
            f"{stats['global_hit_rate_0p5mm']:>10.0%}"
            f"{stats['masked_median_abs_error_mm']:>11.3f}"
            f"{stats['masked_hit_rate_0p5mm']:>10.0%}"
            f"{fwhm_text}"
            f"{stats['masked_resolvable_fraction']:>12.0%}"
        )
    print(f"\nBest oracle-masked metric: {summary['best_masked_metric']}")
    print(f"Suggested tolerance:       "
          f"{summary['suggested_tolerance_mm']:.3f} mm "
          f"(half the measured axial response, floored by median error)")
    print(f"Benchmark tolerance:       "
          f"{summary['current_benchmark_tolerance_mm']:.3f} mm")
    print(f"Unresolvable scenes:       "
          f"{summary['unresolvable_fraction']:.0%} "
          f"(no interior peak even with the oracle mask — "
          f"these belong in coverage, not accuracy)")
    if summary["tolerance_is_optimistic"]:
        print("\nThe benchmark's 0.5 mm tolerance is TIGHTER than this "
              "fixture's own axial response.\nScores below that width measure "
              "the fixture, not the search algorithm.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Small grid for development checks.")
    parser.add_argument("--shape", type=int, default=256)
    parser.add_argument("--steps", type=int, default=401)
    parser.add_argument("--z-min-mm", type=float, default=-5.0)
    parser.add_argument("--z-max-mm", type=float, default=45.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.quick:
        radii, noises, distances = (10.0,), (0.0,), (3.0, 12.0)
        steps = min(args.steps, 121)
        shape_px = min(args.shape, 128)
    else:
        radii = (4.0, 10.0, 18.0)
        noises = (0.0, 0.01, 0.06)
        distances = (3.0, 12.0, 29.0)
        steps = args.steps
        shape_px = args.shape

    scenarios = characterization_grid(radii, noises, distances)
    shape = (shape_px, shape_px)
    rows: list[dict] = []
    for index, scenario in enumerate(scenarios):
        rows.extend(characterize_scene(
            scenario,
            shape=shape,
            z_min_m=args.z_min_mm * 1e-3,
            z_max_m=args.z_max_mm * 1e-3,
            steps=steps,
            metrics=DEFAULT_METRICS,
            seed=20_000 + index,
        ))

    summary = summarize(rows)
    _print_report(rows, summary)

    if args.out:
        args.out.write_text(json.dumps(
            {"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
