"""Focus-metric validation — finite outputs across all metrics, direction flags."""
from __future__ import annotations

import numpy as np
import pytest

from core.autofocus import FocusMetric, _calc_metric, _is_minimize, has_sufficient_contrast


ALL_METRICS = list(FocusMetric)


def _structured_phase(shape=(64, 64)) -> np.ndarray:
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx].astype(np.float64)
    # Phase ramp + sinusoidal — mix of low and high frequency structure.
    return 0.3 * x / nx + 0.6 * np.sin(2 * np.pi * x / 16.0) + 0.4 * np.cos(2 * np.pi * y / 20.0)


def test_all_metrics_finite_on_structured_field():
    phase = _structured_phase()
    for metric in ALL_METRICS:
        val = _calc_metric(phase, metric)
        assert isinstance(val, float), f"{metric}: not a float ({type(val)!r})"
        assert np.isfinite(val), f"{metric}: non-finite value {val!r}"


def test_all_metrics_finite_on_all_nan():
    phase = np.full((64, 64), np.nan, dtype=np.float64)
    for metric in ALL_METRICS:
        val = _calc_metric(phase, metric)
        assert np.isfinite(val), f"{metric}: returned non-finite {val!r} on all-NaN input"


def test_all_metrics_finite_on_flat_field():
    phase = np.zeros((64, 64), dtype=np.float64)
    for metric in ALL_METRICS:
        val = _calc_metric(phase, metric)
        assert np.isfinite(val), f"{metric}: non-finite on flat field"


def test_has_sufficient_contrast_flat():
    phase = np.zeros((64, 64), dtype=np.float64)
    assert has_sufficient_contrast(phase) is False


def test_has_sufficient_contrast_structured():
    rng = np.random.default_rng(1)
    phase = rng.uniform(-np.pi, np.pi, size=(64, 64))
    assert has_sufficient_contrast(phase) is True


def test_entropy_is_the_only_minimize_metric():
    assert _is_minimize(FocusMetric.ENTROPY) is True
    others = [m for m in ALL_METRICS if m is not FocusMetric.ENTROPY]
    for metric in others:
        assert _is_minimize(metric) is False, f"{metric} should maximize"
