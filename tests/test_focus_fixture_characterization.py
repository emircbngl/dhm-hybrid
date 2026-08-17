"""Regression pins for what the synthetic fixture can actually resolve.

These tests exist to stop a specific mistake from recurring: reading a low
autofocus score as "the search algorithm is bad" when the fixture's own focus
response cannot support the tolerance being scored against.

Measured 2026-08-17 on the full grid (27 single-sphere scenes x 5 metrics,
256x256, 401 steps over -5..45 mm, `scripts/characterize_focus_fixture.py`,
raw rows in `tasks/focus_fixture_characterization.json`):

* global metric median |error| 12.1-28.4 mm, hit@0.5mm 0-11%;
* oracle-masked median |error| 0.5-0.875 mm, hit@0.5mm 22-37%;
* measured axial response (median masked FWHM) 0.375-3.25 mm;
* suggested tolerance 1.625 mm versus the benchmark's 0.5 mm.

The pins below are deliberately loose — they assert the *direction* of these
facts, not the exact numbers, so ordinary numerical drift does not fail CI
while a real change in the fixture's behaviour does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from characterize_focus_fixture import (  # noqa: E402
    characterization_grid,
    characterize_scene,
    summarize,
)
from core.autofocus.metrics import FocusMetric  # noqa: E402

# One clean, comfortably resolvable scene: 10 um sphere, no noise, mid-range.
_SCENE = characterization_grid((10.0,), (0.0,), (12.0,))[0]
_METRICS = (FocusMetric.LAPLACIAN_VARIANCE, FocusMetric.TENENGRAD)


@pytest.fixture(scope="module")
def clean_scene_rows() -> list[dict]:
    return characterize_scene(
        _SCENE,
        shape=(256, 256),
        z_min_m=-5e-3,
        z_max_m=45e-3,
        steps=201,
        metrics=_METRICS,
        seed=20_000,
    )


def test_global_focus_curve_does_not_peak_at_the_object_plane(clean_scene_rows):
    """The whole-frame metric maximum sits near the camera, not the object.

    This is why `gradient_descent_search` needs range-edge rejection and why
    whole-frame autofocus scores so poorly on synthetic scenes: the optimiser
    is finding a real extremum that is in the wrong place.
    """
    for row in clean_scene_rows:
        assert abs(row["global_error_mm"]) > 5.0, (
            f"{row['metric']}: global argmax landed "
            f"{row['global_error_mm']:.3f} mm from truth — if the global "
            f"landscape now peaks at the object, the fixture changed and the "
            f"object-local machinery may no longer be necessary"
        )
        # Specifically: it collapses toward z ~ 0, the camera plane.
        assert abs(row["global_argmax_mm"]) < abs(row["truth_z_mm"]) / 2.0


def test_oracle_mask_recovers_the_object_plane(clean_scene_rows):
    """Restricting the metric to the true footprint fixes the landscape.

    This is the upper bound for every `find_object_focus_planes*` variant —
    no detector can do better than being handed the correct mask.
    """
    for row in clean_scene_rows:
        assert abs(row["masked_error_mm"]) < 2.0, (
            f"{row['metric']}: even the oracle mask missed by "
            f"{row['masked_error_mm']:.3f} mm"
        )
        assert not row["masked_edge_pinned"]
        assert row["masked_fwhm_mm"] is not None


def test_measured_axial_response_is_wider_than_the_scored_tolerance(
    clean_scene_rows,
):
    """The 0.5 mm benchmark tolerance is not backed by the fixture.

    `summarize` derives a tolerance from half the measured axial response.
    On the full grid that came out at 1.625 mm. Scoring below the fixture's
    own response width charges the fixture's bias to the search algorithm.
    """
    summary = summarize(clean_scene_rows)
    assert summary["suggested_tolerance_mm"] > 0.0
    widths = [
        row["masked_fwhm_mm"] for row in clean_scene_rows
        if row["masked_fwhm_mm"] is not None
    ]
    assert widths, "no metric produced a measurable peak width"
    assert float(np.median(widths)) > 0.25, (
        "axial response collapsed below the 0.125 mm scan step — "
        "the scan is now under-sampling the peak"
    )


def test_summarize_separates_unresolvable_scenes_from_accuracy():
    """Scenes with no axial signal must land in coverage, not in tolerance.

    Folding them into the tolerance inflates it to tens of millimetres and
    makes every method look accurate — the exact failure this pin prevents.
    """
    resolvable = {
        "metric": "laplacian_variance",
        "global_error_mm": 12.0, "global_edge_pinned": False,
        "masked_error_mm": 0.4, "masked_edge_pinned": False,
        "masked_fwhm_mm": 1.0,
    }
    hopeless = {
        "metric": "laplacian_variance",
        "global_error_mm": 30.0, "global_edge_pinned": True,
        "masked_error_mm": 28.0, "masked_edge_pinned": True,
        "masked_fwhm_mm": None,
    }
    summary = summarize([resolvable, hopeless])
    stats = summary["per_metric"]["laplacian_variance"]
    assert stats["masked_resolvable_fraction"] == pytest.approx(0.5)
    assert summary["unresolvable_fraction"] == pytest.approx(0.5)
    # The 28 mm outlier must not become the tolerance.
    assert summary["suggested_tolerance_mm"] < 15.0
