"""Core contract for finite-difference gradient-descent autofocus."""
from __future__ import annotations

import numpy as np
import pytest

import core.autofocus.search_adaptive as search_adaptive
from core.autofocus import AutofocusCancelled, FocusMetric
from core.autofocus.evaluator import (
    _make_fast_evaluator,
    _make_multi_metric_evaluator,
)
from core.reconstruction import ReconstructionMethod, ReconstructionParams


_FIELD = np.ones((8, 8), dtype=np.complex64)
_PARAMS = ReconstructionParams(
    wavelength_m=632.8e-9,
    pixel_size_m=2.5e-6,
    z_m=0.0,
)
_METHOD = ReconstructionMethod.ASM


def _install_landscape(monkeypatch, score_function):
    """Replace propagation with a deterministic black-box focus curve."""
    monkeypatch.setattr(
        search_adaptive,
        "_make_fast_evaluator",
        lambda *args, **kwargs: score_function,
    )
    monkeypatch.setattr(
        search_adaptive,
        "_make_multi_metric_evaluator",
        lambda field, params, method, metrics, **kwargs: (
            lambda z: tuple(
                -score_function(z)
                if (item == FocusMetric.LAPLACIAN_VARIANCE
                    and metrics[0] == FocusMetric.ENTROPY)
                else score_function(z)
                for item in metrics
            )
        ),
    )


def test_multi_metric_evaluator_matches_individual_evaluators():
    metrics = (FocusMetric.TENENGRAD, FocusMetric.ENTROPY)
    multi = _make_multi_metric_evaluator(
        _FIELD, _PARAMS, _METHOD, metrics,
    )
    expected = tuple(
        _make_fast_evaluator(_FIELD, _PARAMS, _METHOD, metric)(1e-3)
        for metric in metrics
    )
    assert multi(1e-3) == pytest.approx(expected)


def test_gradient_descent_maximizes_focus_metric_with_finite_differences(
        monkeypatch):
    optimum = 2.3e-3
    _install_landscape(
        monkeypatch,
        lambda z: 1.0 - ((z - optimum) / 4e-3) ** 2,
    )

    result = search_adaptive.gradient_descent_search(
        _FIELD, _PARAMS, _METHOD, FocusMetric.TENENGRAD,
        z_min_m=-10e-3, z_max_m=10e-3,
        max_evaluations=60,
    )

    assert result.method_name == "Global-Bracketed Gradient Descent"
    assert result.best_z_m == pytest.approx(optimum, abs=0.25e-3)
    assert result.evaluations <= 60
    assert len(result.z_history) == result.evaluations
    assert np.all(result.z_history >= -10e-3)
    assert np.all(result.z_history <= 10e-3)
    assert result.coarse_z is not None
    assert result.basin_bounds_m is not None
    assert result.basin_bounds_m[0] <= optimum <= result.basin_bounds_m[1]


def test_gradient_descent_minimizes_entropy(monkeypatch):
    optimum = -1.7e-3
    _install_landscape(
        monkeypatch,
        lambda z: 0.2 + ((z - optimum) / 3e-3) ** 2,
    )

    result = search_adaptive.gradient_descent_search(
        _FIELD, _PARAMS, _METHOD, FocusMetric.ENTROPY,
        z_min_m=-8e-3, z_max_m=8e-3,
        max_evaluations=60,
    )

    assert result.best_z_m == pytest.approx(optimum, abs=0.25e-3)
    assert result.best_score == pytest.approx(0.2, abs=0.01)


def test_gradient_descent_honours_budget_and_cancellation(monkeypatch):
    _install_landscape(monkeypatch, lambda z: -(z - 1e-3) ** 2)
    result = search_adaptive.gradient_descent_search(
        _FIELD, _PARAMS, _METHOD, FocusMetric.TENENGRAD,
        z_min_m=-5e-3, z_max_m=5e-3,
        max_evaluations=17,
    )
    assert result.evaluations <= 17

    with pytest.raises(AutofocusCancelled):
        search_adaptive.gradient_descent_search(
            _FIELD, _PARAMS, _METHOD, FocusMetric.TENENGRAD,
            z_min_m=-5e-3, z_max_m=5e-3,
            max_evaluations=17,
            cancel_check=lambda: True,
        )


def test_global_discovery_rejects_stronger_range_edge_artifact(monkeypatch):
    optimum = 2.0e-3

    def landscape(z):
        focus_peak = np.exp(-((z - optimum) / 1.2e-3) ** 2)
        edge_artifact = 5.0 if z >= 9.9e-3 else 0.0
        return float(focus_peak + edge_artifact)

    _install_landscape(monkeypatch, landscape)
    result = search_adaptive.gradient_descent_search(
        _FIELD, _PARAMS, _METHOD, FocusMetric.TENENGRAD,
        z_min_m=-10e-3, z_max_m=10e-3,
        max_evaluations=60,
    )

    assert result.best_z_m == pytest.approx(optimum, abs=0.35e-3)
    assert result.selected_peak_z_m == pytest.approx(optimum, abs=0.5e-3)
    assert result.basin_bounds_m is not None
    assert result.basin_bounds_m[1] < 10e-3


def test_extra_budget_refines_without_changing_discovery_grid(monkeypatch):
    optimum = 2.0e-3
    _install_landscape(
        monkeypatch,
        lambda z: float(np.exp(-((z - optimum) / 1.5e-3) ** 2)),
    )

    result_40 = search_adaptive.gradient_descent_search(
        _FIELD, _PARAMS, _METHOD, FocusMetric.TENENGRAD,
        z_min_m=-10e-3, z_max_m=10e-3, max_evaluations=40,
    )
    result_80 = search_adaptive.gradient_descent_search(
        _FIELD, _PARAMS, _METHOD, FocusMetric.TENENGRAD,
        z_min_m=-10e-3, z_max_m=10e-3, max_evaluations=80,
    )

    assert len(result_40.coarse_z) == len(result_80.coarse_z) == 24
    assert result_40.selected_peak_z_m == result_80.selected_peak_z_m
    assert abs(result_80.best_z_m - optimum) <= abs(result_40.best_z_m - optimum)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"z_min_m": 1.0, "z_max_m": 1.0},
        {"z_min_m": -1.0, "z_max_m": 1.0, "max_evaluations": 2},
        {"z_min_m": -1.0, "z_max_m": 1.0, "finite_difference_step_m": 0.0},
    ],
)
def test_gradient_descent_rejects_invalid_configuration(monkeypatch, kwargs):
    _install_landscape(monkeypatch, lambda z: z * z)
    with pytest.raises(ValueError):
        search_adaptive.gradient_descent_search(
            _FIELD, _PARAMS, _METHOD, FocusMetric.ENTROPY,
            **kwargs,
        )
