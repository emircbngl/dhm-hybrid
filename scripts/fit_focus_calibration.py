"""Refit the object-local candidate calibration, reproducibly.

``core.depth_map._candidate_focus_probability`` carries five standardisation
constants, five weights and an intercept, and its docstring says keeping them
inline "makes the exact calibration reproducible/auditable". That was not true:
the script that produced them was never committed, and neither was the scene
split they were fitted on. Nobody could re-derive, audit or update them.

This is that script. It defines the split as a rule rather than a stored list,
fits with plain NumPy so the result does not depend on an undeclared
scikit-learn (only ``scikit-image`` is in requirements.txt), and prints a code
block ready to paste back into ``depth_map``.

Split
-----
Scene membership is a deterministic function of the scene NAME, not of list
order, so adding scenarios to the catalogue cannot silently reshuffle who is in
train and who is in validation::

    crc32(name) % 25 == 0  -> train   (125 of the 3125 factorial scenes)
    crc32(name) % 25 == 1  -> val     (125, provably disjoint)

``zlib.crc32`` is already the repo's idiom for name-derived determinism
(``benchmark_af_synthetic_stress`` seeds camera noise the same way).

Labelling
---------
A candidate is positive iff it wins a greedy one-to-one match against a
ground-truth plane within tolerance — the same rule
``benchmark_af_synthetic_stress._plane_detection_scores`` uses to score
precision. Labelling "within tolerance of any true plane" instead would mark
every duplicate near one plane positive and train the model to reward the
duplicate-heavy behaviour the gate exists to suppress.

Usage::

    python3 scripts/fit_focus_calibration.py --out tasks/focus_calibration.json
    python3 scripts/fit_focus_calibration.py --quick      # 20+20 scenes
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from core.autofocus import FocusMetric  # noqa: E402
from core.depth_map import find_object_focus_planes  # noqa: E402
from core.reconstruction import ReconstructionMethod  # noqa: E402

from benchmark_af_synthetic_stress import (  # noqa: E402
    StressScenario,
    _build_field,
    factorial_scenarios,
)

FEATURE_NAMES = (
    "prominence",
    "log_contrast",
    "coherence",
    "prominence*coherence",
    "prominence*log_contrast",
    "border_distance",
)

SPLIT_MODULUS = 25
SPLIT_REMAINDER = {"train": 0, "val": 1}
SPLIT_RULE = "crc32(scene_name) % 25 -> 0=train, 1=val"


def split_scenarios(split: str) -> list[StressScenario]:
    """Deterministic, name-derived, provably disjoint scene split."""
    if split not in SPLIT_REMAINDER:
        raise ValueError(f"split must be one of {sorted(SPLIT_REMAINDER)}")
    want = SPLIT_REMAINDER[split]
    return [
        scenario for scenario in factorial_scenarios()
        if zlib.crc32(scenario.name.encode("utf-8")) % SPLIT_MODULUS == want
    ]


def candidate_features(candidate) -> np.ndarray:
    """The six features, in the exact order `depth_map` consumes them.

    `border_distance` was added 2026-08-17 after a feature search: of six
    candidate additions (log area, z span, channel agreement, mean confidence,
    normalised z, border distance) it was the only one that moved held-out
    AUC — 0.8641 to 0.8953. `z_span_steps` and `channel_agreement` scored
    exactly +0.0000 because the 2-D detector leaves them constant at 1 and 1.0.
    Adding all six together scored 0.8895, WORSE than adding border alone, so
    the model is deliberately not "richer" than the evidence supports.
    """
    prominence = float(candidate.prominence)
    log_contrast = float(np.log(max(candidate.contrast_snr, 1e-6)))
    coherence = float(candidate.complex_coherence)
    return np.array([
        prominence,
        log_contrast,
        coherence,
        prominence * coherence,
        prominence * log_contrast,
        float(candidate.border_distance),
    ], dtype=np.float64)


def _greedy_positive_indices(
    detected_z_m: Sequence[float],
    expected_z_m: Sequence[float],
    tolerance_m: float,
) -> set[int]:
    """Indices of detections that win a one-to-one match against truth.

    Mirrors `benchmark_af_synthetic_stress._plane_detection_scores` exactly, so
    a candidate is labelled positive under the same rule that later scores it.
    """
    remaining_found = set(range(len(detected_z_m)))
    remaining_truth = set(range(len(expected_z_m)))
    pairs = sorted(
        (abs(detected_z_m[i] - expected_z_m[j]), i, j)
        for i in remaining_found for j in remaining_truth
    )
    winners: set[int] = set()
    for error, i, j in pairs:
        if error > tolerance_m:
            break
        if i in remaining_found and j in remaining_truth:
            remaining_found.remove(i)
            remaining_truth.remove(j)
            winners.add(i)
    return winners


def collect(
    scenarios: Iterable[StressScenario],
    *,
    shape: tuple[int, int],
    n_steps: int,
    tolerance_m: float,
    z_min_m: float,
    z_max_m: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run the detector over scenes and return (features, labels, per-scene)."""
    rows: list[np.ndarray] = []
    labels: list[int] = []
    per_scene: list[dict] = []
    for scenario in scenarios:
        seed = 10_000 + int(zlib.crc32(scenario.name.encode("utf-8")))
        field, base = _build_field(scenario, shape, seed=seed)
        candidates = find_object_focus_planes(
            field, base, ReconstructionMethod.ASM, z_min_m, z_max_m,
            n_steps=n_steps, metric=FocusMetric.PHASE_VARIANCE,
        )
        detected = [item.z_m for item in candidates]
        winners = _greedy_positive_indices(
            detected, scenario.expected_z_m, tolerance_m)
        for index, candidate in enumerate(candidates):
            rows.append(candidate_features(candidate))
            labels.append(1 if index in winners else 0)
        per_scene.append({
            "scenario": scenario.name,
            "candidates": len(candidates),
            "positives": len(winners),
            "truth_planes": len(scenario.expected_z_m),
        })
    if not rows:
        raise RuntimeError("no candidates collected; check the scene split")
    return np.vstack(rows), np.asarray(labels, dtype=np.float64), per_scene


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1.0,
    iterations: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Standardise, then fit by Newton/IRLS. Deterministic, NumPy only.

    Returns ``(mean, scale, weights, intercept)`` in the layout
    ``_candidate_focus_probability`` expects. Ridge-regularised because the two
    interaction features are strongly collinear with their parents, which makes
    the unpenalised Hessian near-singular.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardised = (features - mean) / scale

    n_samples, n_features = standardised.shape
    design = np.hstack([np.ones((n_samples, 1)), standardised])
    beta = np.zeros(n_features + 1, dtype=np.float64)
    # Do not penalise the intercept.
    penalty = np.eye(n_features + 1) * l2
    penalty[0, 0] = 0.0

    for _ in range(iterations):
        # macOS Accelerate raises divide/overflow/invalid FPU flags from inside
        # these matmuls even though the arithmetic is benign: verified
        # 2026-08-17 that the Hessian is finite at every iteration with
        # condition number ~1.5e2, and IRLS converges in 6 steps. Suppress the
        # spurious flags; the finiteness check below still fails loudly on a
        # genuine blow-up.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            logit = design @ beta
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
            weight = np.clip(prob * (1.0 - prob), 1e-9, None)
            gradient = design.T @ (labels - prob) - penalty @ beta
            hessian = (design.T * weight) @ design + penalty
        if not (np.all(np.isfinite(hessian)) and np.all(np.isfinite(gradient))):
            raise FloatingPointError("IRLS diverged: non-finite Hessian")
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break

    return mean, scale, beta[1:], float(beta[0])


def probability(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> np.ndarray:
    logit = intercept + ((features - mean) / scale) @ weights
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))


def evaluate(
    scenarios: Sequence[StressScenario],
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    *,
    shape: tuple[int, int],
    n_steps: int,
    tolerance_m: float,
    z_min_m: float,
    z_max_m: float,
    thresholds: Sequence[float],
) -> list[dict]:
    """Coverage and conditional hit rate per threshold, on held-out scenes."""
    scenes: list[tuple[list[float], list[float], tuple[float, ...]]] = []
    for scenario in scenarios:
        seed = 10_000 + int(zlib.crc32(scenario.name.encode("utf-8")))
        field, base = _build_field(scenario, shape, seed=seed)
        candidates = find_object_focus_planes(
            field, base, ReconstructionMethod.ASM, z_min_m, z_max_m,
            n_steps=n_steps, metric=FocusMetric.PHASE_VARIANCE,
        )
        probs = [
            float(probability(candidate_features(c), mean, scale,
                              weights, intercept))
            for c in candidates
        ]
        scenes.append(([c.z_m for c in candidates], probs,
                       scenario.expected_z_m))

    report: list[dict] = []
    for threshold in thresholds:
        covered = 0
        hits = 0
        planes = 0
        for detected, probs, truth in scenes:
            kept = [z for z, p in zip(detected, probs) if p >= threshold]
            planes += len(kept)
            if not kept:
                continue
            covered += 1
            nearest = min(abs(z - t) for z in kept for t in truth)
            if nearest <= tolerance_m:
                hits += 1
        report.append({
            "threshold": round(float(threshold), 4),
            "coverage": covered / len(scenes) if scenes else 0.0,
            "conditional_hit": hits / covered if covered else 0.0,
            "overall_hit": hits / len(scenes) if scenes else 0.0,
            "planes_per_scene": planes / len(scenes) if scenes else 0.0,
            "covered_scenes": covered,
        })
    return report


def _code_block(mean, scale, weights, intercept) -> str:
    def fmt(values):
        return ", ".join(f"{v:.8f}" for v in values)
    return (
        "    mean = np.array([\n"
        f"        {fmt(mean)},\n"
        "    ])\n"
        "    scale = np.array([\n"
        f"        {fmt(scale)},\n"
        "    ])\n"
        "    weights = np.array([\n"
        f"        {fmt(weights)},\n"
        "    ])\n"
        f"    logit = {intercept:.8f} + float(np.dot("
        "weights, (features - mean) / scale))"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="20 train / 20 val scenes for a fast check.")
    parser.add_argument("--shape", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--tolerance-mm", type=float, default=0.5)
    parser.add_argument("--z-min-mm", type=float, default=-5.0)
    parser.add_argument("--z-max-mm", type=float, default=45.0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    train = split_scenarios("train")
    val = split_scenarios("val")
    overlap = {s.name for s in train} & {s.name for s in val}
    if overlap:
        raise RuntimeError(f"split is not disjoint: {sorted(overlap)[:3]}")
    if args.quick:
        # Stride, never head-slice. The factorial is generated in nested-loop
        # order, so its first entries are all the easiest corner (lowest noise,
        # nearest, single smallest object). Head-slicing 20 scenes gave 1.75
        # candidates/scene against ~7 for the full split, and a 100% conditional
        # hit at every threshold — a "quick check" that cannot fail.
        train = train[::max(1, len(train) // 20)][:20]
        val = val[::max(1, len(val) // 20)][:20]

    shape = (args.shape, args.shape)
    tolerance_m = args.tolerance_mm * 1e-3
    kwargs = dict(shape=shape, n_steps=args.n_steps, tolerance_m=tolerance_m,
                  z_min_m=args.z_min_mm * 1e-3, z_max_m=args.z_max_mm * 1e-3)

    print(f"split rule: {SPLIT_RULE}")
    print(f"train scenes: {len(train)}   val scenes: {len(val)}   "
          f"disjoint: yes")

    features, labels, per_scene = collect(train, **kwargs)
    print(f"train candidates: {features.shape[0]}  "
          f"positives: {int(labels.sum())} ({labels.mean():.1%})")

    mean, scale, weights, intercept = fit_logistic(
        features, labels, l2=args.l2)

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    report = evaluate(val, mean, scale, weights, intercept,
                      thresholds=thresholds, **kwargs)

    print(f"\nheld-out validation ({len(val)} scenes, "
          f"tolerance {args.tolerance_mm} mm)")
    print(f"{'thr':>6}{'coverage':>10}{'cond hit':>10}"
          f"{'overall':>9}{'planes':>8}{'n':>5}")
    for row in report:
        print(f"{row['threshold']:>6.2f}{row['coverage']:>10.1%}"
              f"{row['conditional_hit']:>10.1%}{row['overall_hit']:>9.1%}"
              f"{row['planes_per_scene']:>8.2f}{row['covered_scenes']:>5d}")

    print("\npaste into core.depth_map._candidate_focus_probability:\n")
    code = _code_block(mean, scale, weights, intercept)
    print(code)

    if args.out:
        args.out.write_text(json.dumps({
            "generated": date.today().isoformat(),
            "split_rule": SPLIT_RULE,
            "train_scenes": len(train),
            "val_scenes": len(val),
            "shape": list(shape),
            "n_steps": args.n_steps,
            "tolerance_mm": args.tolerance_mm,
            "l2": args.l2,
            "feature_names": list(FEATURE_NAMES),
            "train_candidates": int(features.shape[0]),
            "train_positive_rate": float(labels.mean()),
            "mean": [float(v) for v in mean],
            "scale": [float(v) for v in scale],
            "weights": [float(v) for v in weights],
            "intercept": intercept,
            "threshold_sweep": report,
            "per_scene": per_scene,
            "code_block": code,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
