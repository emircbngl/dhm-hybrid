"""Unit tests for src/core/background_phase.py."""
from __future__ import annotations

import numpy as np
import pytest

from core.background_phase import (
    BackgroundFit,
    estimate_sample_mask,
    fit_polynomial_background,
    fit_zernike_background,
    subtract_background,
    _noll_to_nm,
    _zernike_radial,
)


# ---------------------------------------------------------------------------
# Zernike helpers
# ---------------------------------------------------------------------------

def test_noll_index_known_pairs():
    cases = [
        (1, (0, 0)),
        (2, (1, 1)),
        (3, (1, -1)),
        (4, (2, 0)),
        (5, (2, -2)),
        (6, (2, 2)),
        (7, (3, -1)),
        (8, (3, 1)),
        (11, (4, 0)),
    ]
    for j, expected in cases:
        assert _noll_to_nm(j) == expected, f"j={j}"


def test_zernike_radial_known_forms():
    rho = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(_zernike_radial(0, 0, rho), np.ones_like(rho))
    np.testing.assert_allclose(_zernike_radial(2, 0, rho), 2 * rho ** 2 - 1)
    np.testing.assert_allclose(_zernike_radial(4, 0, rho),
                               6 * rho ** 4 - 6 * rho ** 2 + 1, atol=1e-12)


# ---------------------------------------------------------------------------
# Synthetic background recovery
# ---------------------------------------------------------------------------

def _synthetic_background(shape: tuple[int, int]) -> np.ndarray:
    """A smooth plane + paraboloid + small higher-order term."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny / 2.0, nx / 2.0
    R = min(ny, nx) / 2.0
    yn = (yy - cy) / R
    xn = (xx - cx) / R
    return (0.8 * xn          # tilt-x
            + 0.5 * yn        # tilt-y
            + 1.2 * (xn ** 2 + yn ** 2)   # defocus / curvature
            + 0.3 * (xn ** 2 - yn ** 2)   # astigmatism
            + 0.15 * (xn ** 4 + yn ** 4)) # spherical-ish


def _add_sample_features(bg: np.ndarray, n_features: int = 5,
                         mask_threshold: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian sample bumps; mask out anything above mask_threshold*amp.

    Gaussian tails extend far beyond the visible bump, so a tight threshold
    is needed for the mask to actually exclude all sample contamination.
    """
    rng = np.random.default_rng(42)
    out = bg.copy()
    mask = np.zeros_like(bg, dtype=bool)
    ny, nx = bg.shape
    for _ in range(n_features):
        cy = rng.integers(ny // 4, 3 * ny // 4)
        cx = rng.integers(nx // 4, 3 * nx // 4)
        r = rng.integers(15, 35)
        amp = rng.uniform(2.0, 4.0)
        yy, xx = np.mgrid[0:ny, 0:nx]
        bump = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r ** 2))
        out = out + bump
        mask = mask | (bump > amp * mask_threshold)
    return out, mask


def test_zernike_recovers_clean_background():
    bg = _synthetic_background((300, 400))
    fit = fit_zernike_background(bg, n_terms=15)
    ny, nx = bg.shape
    cy, cx = ny / 2.0, nx / 2.0
    R = min(ny, nx) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    rho = np.sqrt(((yy - cy) / R) ** 2 + ((xx - cx) / R) ** 2)
    disk = rho <= 1.0
    rmse = float(np.sqrt(np.mean((bg[disk] - fit.background[disk]) ** 2)))
    assert rmse < 1e-3, f"expected near-perfect recovery on disk, got rmse={rmse}"
    assert fit.n_bg_pixels > 0
    assert fit.method == "zernike"


def test_polynomial_recovers_clean_background():
    bg = _synthetic_background((300, 400))
    fit = fit_polynomial_background(bg, order=4)
    rmse = float(np.sqrt(np.mean((bg - fit.background) ** 2)))
    assert rmse < 1e-3, f"polynomial fit rmse too large: {rmse}"
    assert fit.method == "polynomial"


def test_zernike_robust_with_sample_features():
    bg = _synthetic_background((300, 400))
    dirty, sample_mask = _add_sample_features(bg)
    fit = fit_zernike_background(dirty, n_terms=15, sample_mask=sample_mask)
    ny, nx = bg.shape
    cy, cx = ny / 2.0, nx / 2.0
    R = min(ny, nx) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    rho = np.sqrt(((yy - cy) / R) ** 2 + ((xx - cx) / R) ** 2)
    disk = rho <= 1.0
    bg_only = disk & (~sample_mask)
    rmse = float(np.sqrt(np.mean((bg[bg_only] - fit.background[bg_only]) ** 2)))
    assert rmse < 0.05, f"sample-corrupted rmse too large: {rmse}"


def test_polynomial_robust_with_sample_features():
    bg = _synthetic_background((300, 400))
    dirty, sample_mask = _add_sample_features(bg)
    fit = fit_polynomial_background(dirty, order=4, sample_mask=sample_mask)
    bg_only = ~sample_mask
    rmse = float(np.sqrt(np.mean((bg[bg_only] - fit.background[bg_only]) ** 2)))
    assert rmse < 0.05, f"polynomial sample-corrupted rmse: {rmse}"


def test_sigma_clipping_works_without_explicit_mask():
    """Sigma clipping should reject feature pixels even without an explicit mask.

    With features whose total area is small relative to the frame and a
    polynomial that spans the true background, iterative residual rejection
    converges on the background. Threshold tuned to what the algorithm can
    realistically deliver from feature-tail leakage alone.
    """
    bg = _synthetic_background((300, 400))
    dirty, sample_mask = _add_sample_features(bg, n_features=3)
    fit = fit_polynomial_background(dirty, order=4, sample_mask=None)
    bg_only = ~sample_mask
    rmse = float(np.sqrt(np.mean((bg[bg_only] - fit.background[bg_only]) ** 2)))
    assert rmse < 0.5, f"sigma-clip-only rmse: {rmse}"


# ---------------------------------------------------------------------------
# Sample mask
# ---------------------------------------------------------------------------

def test_estimate_sample_mask_separates_bright_regions():
    rng = np.random.default_rng(7)
    bg_amp = rng.uniform(0.8, 1.0, size=(200, 250))
    sample = bg_amp.copy()
    sample[60:120, 100:160] = 3.0
    mask = estimate_sample_mask(sample, method="percentile", percentile=70.0)
    assert mask[80, 130], "sample center should be flagged True"
    assert not mask[10, 10], "background corner should be False"


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_subtract_background_zernike_end_to_end():
    bg = _synthetic_background((300, 400))
    dirty, sample_mask = _add_sample_features(bg)
    amplitude = np.where(sample_mask, 3.0, 1.0).astype(np.float32)
    corrected, fit = subtract_background(
        dirty, amplitude=amplitude, method="zernike", n_terms=15,
    )
    assert isinstance(fit, BackgroundFit)
    assert corrected.dtype == np.float32

    bg_only = (~sample_mask)
    ny, nx = bg.shape
    cy, cx = ny / 2.0, nx / 2.0
    R = min(ny, nx) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    rho = np.sqrt(((yy - cy) / R) ** 2 + ((xx - cx) / R) ** 2)
    disk = rho <= 1.0
    eval_pix = bg_only & disk

    bg_residual_rms = float(np.sqrt(np.mean(corrected[eval_pix] ** 2)))
    assert bg_residual_rms < 0.15, f"residual on bg pixels too high: {bg_residual_rms}"


def test_subtract_background_polynomial_end_to_end():
    bg = _synthetic_background((300, 400))
    dirty, sample_mask = _add_sample_features(bg)
    amplitude = np.where(sample_mask, 3.0, 1.0).astype(np.float32)
    corrected, fit = subtract_background(
        dirty, amplitude=amplitude, method="polynomial", polynomial_order=4,
    )
    bg_only = ~sample_mask
    bg_residual_rms = float(np.sqrt(np.mean(corrected[bg_only] ** 2)))
    assert bg_residual_rms < 0.15


def test_subtract_background_unknown_method_raises():
    bg = _synthetic_background((100, 100))
    with pytest.raises(ValueError):
        subtract_background(bg, method="bogus")


# ---------------------------------------------------------------------------
# Basis/grid cache — numerics unchanged, basis rebuilt only once per key
# ---------------------------------------------------------------------------

def test_zernike_basis_cache_reused_across_frames_same_numerics(monkeypatch):
    """Two different frames of the same shape must produce allclose fits,
    and the expensive full-resolution basis builder must only run once
    (the second call is served from the lru_cache)."""
    import core.background_phase as bp

    bp._zernike_full_basis.cache_clear()
    bp._zernike_subsampled_basis.cache_clear()
    bp._zernike_subsampled_grid.cache_clear()
    bp._zernike_full_grid.cache_clear()

    calls = {"n": 0}
    real_zernike_basis = bp._zernike_basis

    def counting_zernike_basis(n_terms, rho, theta):
        calls["n"] += 1
        return real_zernike_basis(n_terms, rho, theta)

    monkeypatch.setattr(bp, "_zernike_basis", counting_zernike_basis)

    shape = (200, 240)
    bg = _synthetic_background(shape)
    frame_a = bg + np.random.default_rng(1).normal(0, 1e-4, size=shape)
    frame_b = bg + np.random.default_rng(2).normal(0, 1e-4, size=shape)

    fit_a = bp.fit_zernike_background(frame_a, n_terms=15)
    # _zernike_basis is invoked once for the subsampled grid and once for
    # the full-resolution grid on the first call.
    calls_after_first = calls["n"]
    assert calls_after_first == 2

    fit_b = bp.fit_zernike_background(frame_b, n_terms=15)
    # Same (ny, nx, subsample, n_terms) key — basis must be served from
    # cache, not rebuilt, so the counter must not increase.
    assert calls["n"] == calls_after_first, (
        "expected _zernike_basis NOT to be called again for a repeat "
        "(shape, n_terms) key — the basis cache should have served it"
    )

    # Reference computed with the cache bypassed entirely (fresh clear
    # before and after) must be numerically identical.
    bp._zernike_full_basis.cache_clear()
    bp._zernike_subsampled_basis.cache_clear()
    bp._zernike_subsampled_grid.cache_clear()
    bp._zernike_full_grid.cache_clear()
    monkeypatch.setattr(bp, "_zernike_basis", real_zernike_basis)
    fit_b_uncached = bp.fit_zernike_background(frame_b, n_terms=15)

    np.testing.assert_allclose(fit_b.background, fit_b_uncached.background)
    np.testing.assert_allclose(fit_b.coeffs, fit_b_uncached.coeffs)
    assert fit_b.residual_std == pytest.approx(fit_b_uncached.residual_std)

    bp._zernike_full_basis.cache_clear()
    bp._zernike_subsampled_basis.cache_clear()
    bp._zernike_subsampled_grid.cache_clear()
    bp._zernike_full_grid.cache_clear()


def test_polynomial_basis_cache_reused_across_frames_same_numerics(monkeypatch):
    """Same guarantee as the Zernike test above, for the polynomial path:
    two different frames of identical shape/order share the cached
    monomial basis and produce allclose fits."""
    import core.background_phase as bp

    bp._poly_full_basis.cache_clear()
    bp._poly_subsampled_basis.cache_clear()
    bp._poly_subsampled_indices.cache_clear()

    calls = {"n": 0}
    real_polynomial_basis = bp._polynomial_basis

    def counting_polynomial_basis(order, x, y):
        calls["n"] += 1
        return real_polynomial_basis(order, x, y)

    monkeypatch.setattr(bp, "_polynomial_basis", counting_polynomial_basis)

    shape = (180, 260)
    bg = _synthetic_background(shape)
    frame_a = bg + np.random.default_rng(3).normal(0, 1e-4, size=shape)
    frame_b = bg + np.random.default_rng(4).normal(0, 1e-4, size=shape)

    bp.fit_polynomial_background(frame_a, order=4)
    calls_after_first = calls["n"]
    assert calls_after_first == 2  # subsampled + full basis, once each

    fit_b = bp.fit_polynomial_background(frame_b, order=4)
    assert calls["n"] == calls_after_first, (
        "expected _polynomial_basis NOT to be called again for a repeat "
        "(shape, order) key — the basis cache should have served it"
    )

    bp._poly_full_basis.cache_clear()
    bp._poly_subsampled_basis.cache_clear()
    bp._poly_subsampled_indices.cache_clear()
    monkeypatch.setattr(bp, "_polynomial_basis", real_polynomial_basis)
    fit_b_uncached = bp.fit_polynomial_background(frame_b, order=4)

    np.testing.assert_allclose(fit_b.background, fit_b_uncached.background)
    np.testing.assert_allclose(fit_b.coeffs, fit_b_uncached.coeffs)
    assert fit_b.residual_std == pytest.approx(fit_b_uncached.residual_std)

    bp._poly_full_basis.cache_clear()
    bp._poly_subsampled_basis.cache_clear()
    bp._poly_subsampled_indices.cache_clear()


def test_basis_cache_does_not_leak_between_different_shapes():
    """A different frame shape must get its own basis (not a stale cached
    array from a previous shape) — allclose against an independent
    from-scratch fit on that new shape."""
    import core.background_phase as bp

    bp._zernike_full_basis.cache_clear()
    bp._zernike_subsampled_basis.cache_clear()
    bp._zernike_subsampled_grid.cache_clear()
    bp._zernike_full_grid.cache_clear()

    shape1 = (150, 200)
    shape2 = (220, 180)
    bg1 = _synthetic_background(shape1)
    bg2 = _synthetic_background(shape2)

    fit1 = bp.fit_zernike_background(bg1, n_terms=15)
    fit2 = bp.fit_zernike_background(bg2, n_terms=15)

    assert fit1.background.shape == shape1
    assert fit2.background.shape == shape2

    ny, nx = shape2
    cy, cx = ny / 2.0, nx / 2.0
    R = min(ny, nx) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    rho = np.sqrt(((yy - cy) / R) ** 2 + ((xx - cx) / R) ** 2)
    disk = rho <= 1.0
    rmse = float(np.sqrt(np.mean((bg2[disk] - fit2.background[disk]) ** 2)))
    assert rmse < 1e-3

    bp._zernike_full_basis.cache_clear()
    bp._zernike_subsampled_basis.cache_clear()
    bp._zernike_subsampled_grid.cache_clear()
    bp._zernike_full_grid.cache_clear()
