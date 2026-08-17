"""Pins for the calibration fitting script.

`scripts/fit_focus_calibration.py` exists so the constants in
`core.depth_map._candidate_focus_probability` can be audited and re-derived.
That only holds if two things stay true: the split is a reproducible rule, and
the script computes the same probability the shipped function does. Both are
asserted here, because both fail silently.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from core.autofocus import FocusMetric  # noqa: E402
from core.depth_map import (  # noqa: E402
    ObjectFocusPlane,
    _candidate_focus_probability,
)
from fit_focus_calibration import (  # noqa: E402
    FEATURE_NAMES,
    SPLIT_MODULUS,
    _greedy_positive_indices,
    candidate_features,
    fit_logistic,
    probability,
    split_scenarios,
)

# The constants currently shipped in core.depth_map. Duplicated here on
# purpose: the point of this module is to detect the script and the runtime
# function drifting apart, which a shared import would hide.
SHIPPED_MEAN = np.array([
    0.48911202, 0.54942014, 0.31362574, 0.15469217, 0.28152352, 0.35875535])
SHIPPED_SCALE = np.array([
    0.11759580, 0.27781999, 0.16175545, 0.09644463, 0.19437766, 0.28176341])
SHIPPED_WEIGHTS = np.array([
    0.59382365, -0.24191525, 0.20819830, 0.20494498, 0.40730492, 0.75770116])
SHIPPED_INTERCEPT = -2.46997618


def _plane(prominence, contrast_snr, coherence, z_m=12e-3, border=0.5):
    return ObjectFocusPlane(
        object_id=0,
        centroid_yx=(0.0, 0.0),
        bbox_yx=(0, 0, 1, 1),
        area_px=16,
        z_m=z_m,
        prominence=prominence,
        mean_confidence=0.5,
        metric=FocusMetric.PHASE_VARIANCE,
        contrast_snr=contrast_snr,
        complex_coherence=coherence,
        border_distance=border,
    )


def test_split_is_disjoint_and_name_derived():
    """Membership must follow from the NAME, not from list order.

    Index-based splitting would silently reshuffle train and validation the
    moment a scenario is added to the catalogue, leaking the old validation
    set into training.
    """
    train = split_scenarios("train")
    val = split_scenarios("val")
    assert train and val
    train_names = {s.name for s in train}
    val_names = {s.name for s in val}
    assert not (train_names & val_names), "split is not disjoint"

    for scenario in train:
        assert zlib.crc32(scenario.name.encode("utf-8")) % SPLIT_MODULUS == 0
    for scenario in val:
        assert zlib.crc32(scenario.name.encode("utf-8")) % SPLIT_MODULUS == 1

    assert {s.name for s in split_scenarios("train")} == train_names


def test_script_probability_matches_the_shipped_function():
    """The script must score a candidate exactly as `depth_map` does.

    If the feature order or the sigmoid drift apart, the script would fit and
    report numbers for a model that is not the one gating candidates at
    runtime — the failure this pin exists to prevent.
    """
    for prominence, snr, coherence in [
        (0.75, 3.0, 0.55), (0.40, 1.2, 0.20), (0.95, 8.0, 0.80),
        (0.05, 1e-6, 0.01),
    ]:
        candidate = _plane(prominence, snr, coherence)
        from_script = float(probability(
            candidate_features(candidate), SHIPPED_MEAN, SHIPPED_SCALE,
            SHIPPED_WEIGHTS, SHIPPED_INTERCEPT))
        from_runtime = _candidate_focus_probability(candidate)
        assert from_script == pytest.approx(from_runtime, rel=1e-9, abs=1e-12), (
            f"script and depth_map disagree for "
            f"({prominence}, {snr}, {coherence}): "
            f"{from_script} vs {from_runtime}"
        )


def test_feature_names_document_the_actual_feature_order():
    assert len(FEATURE_NAMES) == 6
    candidate = _plane(0.5, float(np.e), 0.3, border=0.42)
    features = candidate_features(candidate)
    assert features.shape == (6,)
    # log(e) == 1 makes the interaction terms checkable by hand.
    assert features[0] == pytest.approx(0.5)
    assert features[1] == pytest.approx(1.0)
    assert features[2] == pytest.approx(0.3)
    assert features[3] == pytest.approx(0.15)
    assert features[4] == pytest.approx(0.5)
    assert features[5] == pytest.approx(0.42)


def test_border_distance_is_populated_by_the_detector():
    """The feature must come from the detector, not stay at its default.

    `border_distance` defaults to 1.0 so old constructions keep working. If
    `find_object_focus_planes` ever stops populating it, every candidate would
    silently score as a dead-centre object and the sixth feature would become
    a constant — the model would still run, and still be wrong.
    """
    import zlib

    from benchmark_af_synthetic_stress import _build_field
    from core.autofocus import FocusMetric as _FocusMetric
    from core.depth_map import find_object_focus_planes
    from core.reconstruction import ReconstructionMethod

    # Not every scene yields candidates — abstention is a designed outcome —
    # so take the first that does rather than pinning one scene by index.
    candidates: list = []
    for scenario in split_scenarios("val")[:10]:
        field, base = _build_field(
            scenario, (256, 256),
            seed=10_000 + int(zlib.crc32(scenario.name.encode("utf-8"))))
        candidates = find_object_focus_planes(
            field, base, ReconstructionMethod.ASM, -5e-3, 45e-3,
            n_steps=60, metric=_FocusMetric.PHASE_VARIANCE)
        if candidates:
            break
    assert candidates, "no candidates in the first 10 val scenes"
    values = [c.border_distance for c in candidates]
    assert all(0.0 <= v <= 1.0 for v in values), values
    assert any(v != 1.0 for v in values), (
        "every candidate sat at the default 1.0 — the detector is not "
        "populating border_distance"
    )


def test_labelling_is_one_to_one_not_within_tolerance_of_any():
    """Duplicates near one true plane must not all count as positives.

    Labelling "within tolerance of any truth" would teach the model that
    duplicate-heavy output is good — the behaviour the gate exists to suppress.
    """
    detected = [12.0e-3, 12.05e-3, 12.1e-3]  # three detections, one plane
    winners = _greedy_positive_indices(detected, (12.0e-3,), 0.5e-3)
    assert len(winners) == 1, "only one detection may claim a single plane"
    assert winners == {0}, "the closest detection should win the match"


def test_fit_is_deterministic_and_converges():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 5))
    # macOS Accelerate raises divide/overflow/invalid FPU flags even for this
    # matmul — standard normals against a 5-vector, where no value is anywhere
    # near a limit. That is the clearest available proof the flags come from
    # the BLAS backend and not from the arithmetic here or in fit_logistic.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        logits = features @ np.array([1.0, -0.5, 0.25, 0.0, 0.75]) - 0.3
    assert np.all(np.isfinite(logits))
    labels = (rng.uniform(size=400)
              < 1.0 / (1.0 + np.exp(-logits))).astype(float)

    first = fit_logistic(features, labels, l2=1.0)
    second = fit_logistic(features, labels, l2=1.0)
    for a, b in zip(first[:3], second[:3]):
        assert np.array_equal(a, b), "fit is not deterministic"
    assert first[3] == second[3]
    assert np.all(np.isfinite(first[2]))
    # Signs of the strong, well-separated coefficients should be recovered.
    assert first[2][0] > 0 and first[2][1] < 0
