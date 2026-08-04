"""Reference-free background phase estimation for DHM.

Off-axis DHM normally subtracts background phase (plane curvature, tilt,
optical aberrations) by recording a separate reference hologram and
dividing the propagated complex fields. This module provides numerical
alternatives that do not require a reference acquisition:

* `fit_zernike_background` — Zernike polynomial fit on the inscribed
  unit disk; captures higher-order optical aberrations (spherical,
  coma, trefoil, secondary astigmatism, tetrafoil) beyond what the
  existing order-2 polynomial in :mod:`phase_unwrap` covers.
* `fit_polynomial_background` — extended Cartesian polynomial fit
  (configurable order 1..5), works on the full rectangle without
  corner loss. Mirrors the structure of `_remove_polynomial_bg` but
  exposes a sample mask parameter and higher orders.
* `estimate_sample_mask` — amplitude-driven sample/background
  segmentation; pixels marked True are excluded from the fit.
* `subtract_background` — end-to-end convenience wrapper that picks
  Zernike vs polynomial based on aspect ratio + ROI, fits, returns
  corrected phase + diagnostics.

The fits use iterative sigma-clipping on the residuals so a sample
mask is helpful but not strictly required: pixels whose phase deviates
strongly from the fitted background are dropped between iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sample mask estimation
# ---------------------------------------------------------------------------

def estimate_sample_mask(
    amplitude: np.ndarray,
    method: str = "percentile",
    percentile: float = 60.0,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Return a boolean mask (True = sample, exclude from fit).

    method='percentile' takes pixels above the given percentile of the
    amplitude as sample. method='otsu' uses a single Otsu threshold.

    A small Gaussian blur is applied to suppress salt-and-pepper noise
    before thresholding; the returned mask itself is unsmoothed.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(amplitude, dtype=np.float32)
    if smooth_sigma > 0:
        a = gaussian_filter(a, sigma=smooth_sigma)

    if method == "otsu":
        from skimage.filters import threshold_otsu
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return np.zeros_like(a, dtype=bool)
        thresh = float(threshold_otsu(finite))
    elif method == "percentile":
        thresh = float(np.percentile(a[np.isfinite(a)], percentile))
    else:
        raise ValueError(f"unknown method: {method}")

    return a > thresh


# ---------------------------------------------------------------------------
# Zernike polynomials (Noll indexing)
# ---------------------------------------------------------------------------

def _noll_to_nm(j: int) -> tuple[int, int]:
    """Map Noll single index j (1-based) to (n, m) radial/azimuthal orders."""
    if j < 1:
        raise ValueError("Noll index must be >= 1")
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    j_min = n * (n + 1) // 2 + 1
    k = j - j_min
    if n % 2 == 0:
        m_options = list(range(0, n + 1, 2))
        idx = (k + 1) // 2
    else:
        m_options = list(range(1, n + 1, 2))
        idx = k // 2
    abs_m = m_options[idx]
    if abs_m == 0:
        m = 0
    elif (j % 2) == 0:
        m = abs_m
    else:
        m = -abs_m
    return n, m


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    abs_m = abs(m)
    if (n - abs_m) % 2 != 0:
        return np.zeros_like(rho)
    R = np.zeros_like(rho)
    for k in range((n - abs_m) // 2 + 1):
        coeff = ((-1) ** k * _factorial(n - k)
                 / (_factorial(k)
                    * _factorial((n + abs_m) // 2 - k)
                    * _factorial((n - abs_m) // 2 - k)))
        R = R + coeff * rho ** (n - 2 * k)
    return R


def _factorial(n: int) -> int:
    if n < 0:
        return 0
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def _zernike_basis(n_terms: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Stack of n_terms Zernike polynomials evaluated on (rho, theta).

    Returns array shape (n_terms, *rho.shape). rho is expected to be
    normalised to [0, 1]; values >1 are zeroed implicitly via the
    radial polynomial's compact-support evaluation (the caller masks
    them anyway via the unit-disk indicator).
    """
    out = np.empty((n_terms,) + rho.shape, dtype=np.float64)
    for j in range(1, n_terms + 1):
        n, m = _noll_to_nm(j)
        R = _zernike_radial(n, abs(m), rho)
        if m > 0:
            out[j - 1] = R * np.cos(m * theta)
        elif m < 0:
            out[j - 1] = R * np.sin(abs(m) * theta)
        else:
            out[j - 1] = R
    return out


# ---------------------------------------------------------------------------
# Basis/grid caches
#
# The coordinate grids and evaluated basis matrices built below depend only
# on (ny, nx, subsample, n_terms/order) — never on the phase values
# themselves. For a fixed camera resolution (the overwhelming common case:
# every frame in a recording session shares one shape), rebuilding a
# ~230 MB float64 Zernike basis stack (1200x1600x15) or a polynomial
# monomial matrix from scratch on every single frame is pure waste.
# ``lru_cache`` keyed on the shape-derived scalars lets repeat calls reuse
# the same arrays. maxsize is kept small since each entry can be tens to
# hundreds of MB; a handful of distinct (shape, n_terms/order) combos is
# the realistic ceiling for one session.
#
# IMPORTANT: cached arrays are shared, mutable numpy objects — callers
# must never mutate them in place. The fit functions below only ever read
# from the cached grids/basis (matmul, ravel, boolean compare), so this
# holds today; if that ever changes, copy before mutating.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _zernike_subsampled_grid(ny: int, nx: int, subsample: int):
    """Subsampled (rho, theta, disk-mask) grid for the lstsq fit, cached."""
    cy, cx = ny / 2.0, nx / 2.0
    R = min(ny, nx) / 2.0
    ys = np.arange(0, ny, subsample)
    xs = np.arange(0, nx, subsample)
    yg, xg = np.meshgrid(ys, xs, indexing="ij")
    y_n = (yg - cy) / R
    x_n = (xg - cx) / R
    rho = np.sqrt(x_n ** 2 + y_n ** 2)
    theta = np.arctan2(y_n, x_n)
    disk = rho <= 1.0
    return yg, xg, rho, theta, disk


@lru_cache(maxsize=4)
def _zernike_subsampled_basis(ny: int, nx: int, subsample: int, n_terms: int) -> np.ndarray:
    """Basis matrix (n_pts, n_terms) for the subsampled lstsq fit, cached."""
    _, _, rho, theta, _ = _zernike_subsampled_grid(ny, nx, subsample)
    basis_full = _zernike_basis(n_terms, rho, theta)
    return basis_full.reshape(n_terms, -1).T


@lru_cache(maxsize=4)
def _zernike_full_grid(ny: int, nx: int):
    """Full-resolution (rho, theta, disk-mask) grid for background eval, cached."""
    cy, cx = ny / 2.0, nx / 2.0
    R = min(ny, nx) / 2.0
    y_full, x_full = np.mgrid[0:ny, 0:nx]
    y_n_full = (y_full - cy) / R
    x_n_full = (x_full - cx) / R
    rho_full = np.sqrt(x_n_full ** 2 + y_n_full ** 2)
    theta_full = np.arctan2(y_n_full, x_n_full)
    rho_eval = np.minimum(rho_full, 1.0)
    disk_full = rho_full <= 1.0
    return rho_full, theta_full, rho_eval, disk_full


@lru_cache(maxsize=4)
def _zernike_full_basis(ny: int, nx: int, n_terms: int) -> np.ndarray:
    """Full-resolution evaluated Zernike basis stack (n_terms, ny, nx), cached.

    This is the ~230 MB-class array (float64 @ 1200x1600x15) the cache
    exists for.
    """
    _, theta_full, rho_eval, _ = _zernike_full_grid(ny, nx)
    with np.errstate(all="ignore"):
        return _zernike_basis(n_terms, rho_eval, theta_full)


@lru_cache(maxsize=4)
def _poly_subsampled_basis(ny: int, nx: int, subsample: int, order: int) -> np.ndarray:
    """Subsampled polynomial monomial basis (n_pts, n_terms), cached."""
    cy, cx = ny / 2.0, nx / 2.0
    Rn = ny / 2.0
    Rx = nx / 2.0
    ys = np.arange(0, ny, subsample)
    xs = np.arange(0, nx, subsample)
    yg, xg = np.meshgrid(ys, xs, indexing="ij")
    y_n = (yg - cy) / Rn
    x_n = (xg - cx) / Rx
    return _polynomial_basis(order, x_n.ravel(), y_n.ravel())


@lru_cache(maxsize=4)
def _poly_subsampled_indices(ny: int, nx: int, subsample: int):
    """Row/col index grids matching ``_poly_subsampled_basis``, cached."""
    ys = np.arange(0, ny, subsample)
    xs = np.arange(0, nx, subsample)
    yg, xg = np.meshgrid(ys, xs, indexing="ij")
    return yg, xg


@lru_cache(maxsize=4)
def _poly_full_basis(ny: int, nx: int, order: int) -> np.ndarray:
    """Full-resolution polynomial monomial basis (ny*nx, n_terms), cached."""
    cy, cx = ny / 2.0, nx / 2.0
    Rn = ny / 2.0
    Rx = nx / 2.0
    y_full, x_full = np.mgrid[0:ny, 0:nx]
    y_n_full = (y_full - cy) / Rn
    x_n_full = (x_full - cx) / Rx
    with np.errstate(all="ignore"):
        return _polynomial_basis(order, x_n_full.ravel(), y_n_full.ravel())


# ---------------------------------------------------------------------------
# Fit functions
# ---------------------------------------------------------------------------

@dataclass
class BackgroundFit:
    background: np.ndarray            # fitted background, same shape as input
    coeffs: np.ndarray                # fit coefficients
    residual_std: float               # std of (phase - background) on bg pixels
    n_bg_pixels: int                  # how many pixels were used for the fit
    method: str                       # 'zernike' or 'polynomial'


def fit_zernike_background(
    phase: np.ndarray,
    n_terms: int = 15,
    sample_mask: Optional[np.ndarray] = None,
    sigma_clip_iters: int = 3,
    sigma_clip_factor: float = 2.0,
    subsample: int = 4,
) -> BackgroundFit:
    """Fit Zernike polynomials on the inscribed unit disk.

    Parameters
    ----------
    phase
        Unwrapped phase (radians), 2D array.
    n_terms
        Number of Noll-indexed Zernike modes (15 covers through tetrafoil).
    sample_mask
        True = sample, exclude from fit. Optional; sigma-clipping handles
        outliers automatically.
    sigma_clip_iters
        Iterative residual-based pixel rejection rounds.
    sigma_clip_factor
        Pixels with |residual| > factor * std are dropped each iteration.
    subsample
        Take every k-th pixel for the lstsq (the polynomial is smooth so
        subsampling loses no information). Affects fit speed only.
    """
    ny, nx = phase.shape

    yg, xg, rho, theta, disk = _zernike_subsampled_grid(ny, nx, subsample)
    basis_flat = _zernike_subsampled_basis(ny, nx, subsample, n_terms)

    b = phase[yg, xg].astype(np.float64).ravel()
    keep = disk.ravel() & np.isfinite(b)

    if sample_mask is not None:
        m_sub = sample_mask[yg, xg].ravel()
        keep = keep & (~m_sub)

    coeffs = _iterative_lstsq(
        basis_flat, b, keep,
        sigma_clip_iters=sigma_clip_iters,
        sigma_clip_factor=sigma_clip_factor,
    )

    if coeffs is None:
        return BackgroundFit(
            background=np.zeros_like(phase),
            coeffs=np.zeros(n_terms),
            residual_std=float("nan"),
            n_bg_pixels=0,
            method="zernike",
        )

    _, _, _, disk_full = _zernike_full_grid(ny, nx)
    basis_full_eval = _zernike_full_basis(ny, nx, n_terms)
    with np.errstate(all="ignore"):
        background = np.einsum("k,kij->ij", coeffs, basis_full_eval)
    background = np.nan_to_num(background, nan=0.0, posinf=0.0, neginf=0.0)

    residual = phase - background
    if sample_mask is not None:
        bg_pix = disk_full & (~sample_mask) & np.isfinite(residual)
    else:
        bg_pix = disk_full & np.isfinite(residual)
    residual_std = float(np.std(residual[bg_pix])) if bg_pix.any() else float("nan")

    return BackgroundFit(
        background=background.astype(np.float32),
        coeffs=coeffs,
        residual_std=residual_std,
        n_bg_pixels=int(bg_pix.sum()),
        method="zernike",
    )


def fit_polynomial_background(
    phase: np.ndarray,
    order: int = 4,
    sample_mask: Optional[np.ndarray] = None,
    sigma_clip_iters: int = 3,
    sigma_clip_factor: float = 2.0,
    subsample: int = 4,
) -> BackgroundFit:
    """Fit a Cartesian polynomial of given total order to the phase.

    Order 4 covers tilt + defocus + astigmatism + cubic + quartic
    (spherical-like) terms, which is usually enough for DHM optics.
    Works on the full rectangular frame (no inscribed-disk loss).
    """
    if order < 1 or order > 6:
        raise ValueError("order must be in [1, 6]")

    ny, nx = phase.shape

    yg, xg = _poly_subsampled_indices(ny, nx, subsample)
    basis_flat = _poly_subsampled_basis(ny, nx, subsample, order)

    b = phase[yg, xg].astype(np.float64).ravel()
    keep = np.isfinite(b)
    if sample_mask is not None:
        keep = keep & (~sample_mask[yg, xg].ravel())

    coeffs = _iterative_lstsq(
        basis_flat, b, keep,
        sigma_clip_iters=sigma_clip_iters,
        sigma_clip_factor=sigma_clip_factor,
    )

    n_terms = basis_flat.shape[1]
    if coeffs is None:
        return BackgroundFit(
            background=np.zeros_like(phase),
            coeffs=np.zeros(n_terms),
            residual_std=float("nan"),
            n_bg_pixels=0,
            method="polynomial",
        )

    basis_full = _poly_full_basis(ny, nx, order)
    with np.errstate(all="ignore"):
        background = (basis_full @ coeffs).reshape(ny, nx)
    background = np.nan_to_num(background, nan=0.0, posinf=0.0, neginf=0.0)

    residual = phase - background
    if sample_mask is not None:
        bg_pix = (~sample_mask) & np.isfinite(residual)
    else:
        bg_pix = np.isfinite(residual)
    residual_std = float(np.std(residual[bg_pix])) if bg_pix.any() else float("nan")

    return BackgroundFit(
        background=background.astype(np.float32),
        coeffs=coeffs,
        residual_std=residual_std,
        n_bg_pixels=int(bg_pix.sum()),
        method="polynomial",
    )


def _polynomial_basis(order: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build a Cartesian polynomial basis matrix (n_pts, n_terms).

    Includes all monomials x^i y^j with i+j <= order.
    """
    cols = []
    for total in range(order + 1):
        for i in range(total + 1):
            j = total - i
            cols.append((x ** i) * (y ** j))
    return np.column_stack(cols).astype(np.float64)


def _iterative_lstsq(
    A: np.ndarray,
    b: np.ndarray,
    keep_init: np.ndarray,
    sigma_clip_iters: int,
    sigma_clip_factor: float,
) -> Optional[np.ndarray]:
    """Iterative sigma-clipped least squares. Returns coefficients or None."""
    keep = keep_init.copy()
    coeffs: Optional[np.ndarray] = None

    n_cols = A.shape[1]

    for _ in range(max(1, sigma_clip_iters)):
        if int(keep.sum()) < n_cols + 1:
            break
        try:
            c, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(c)):
            return None
        if np.max(np.abs(c)) > 1e12:
            return None
        coeffs = c

        with np.errstate(all="ignore"):
            residual = b - A @ c
        if not np.all(np.isfinite(residual)):
            return coeffs
        std = np.std(residual[keep])
        if std < 1e-12:
            break
        keep = keep & (np.abs(residual) < sigma_clip_factor * std)

    return coeffs


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------

def subtract_background(
    phase_unwrapped: np.ndarray,
    amplitude: Optional[np.ndarray] = None,
    method: str = "zernike",
    n_terms: int = 15,
    polynomial_order: int = 4,
    use_amplitude_mask: bool = True,
    mask_percentile: float = 60.0,
) -> tuple[np.ndarray, BackgroundFit]:
    """Subtract background phase from an unwrapped phase map.

    Parameters
    ----------
    phase_unwrapped
        Unwrapped phase in radians.
    amplitude
        Optional |complex_field| used to derive a sample mask.
    method
        'zernike' (inscribed disk, higher-order modes) or 'polynomial'
        (full rectangle, total order up to 6).
    n_terms
        Zernike Noll terms (only used when method='zernike').
    polynomial_order
        Total polynomial order (only used when method='polynomial').
    use_amplitude_mask
        If True and amplitude is given, exclude high-amplitude (sample)
        pixels from the fit using `estimate_sample_mask`.
    """
    sample_mask: Optional[np.ndarray] = None
    if use_amplitude_mask and amplitude is not None:
        sample_mask = estimate_sample_mask(amplitude, method="percentile",
                                           percentile=mask_percentile)

    if method == "zernike":
        fit = fit_zernike_background(
            phase_unwrapped, n_terms=n_terms, sample_mask=sample_mask,
        )
    elif method == "polynomial":
        fit = fit_polynomial_background(
            phase_unwrapped, order=polynomial_order, sample_mask=sample_mask,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    corrected = (phase_unwrapped - fit.background).astype(np.float32)
    return corrected, fit
