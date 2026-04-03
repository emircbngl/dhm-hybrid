"""
Phase Unwrapping Module for Digital Holographic Microscopy.

Provides multiple unwrapping algorithms with optional pre/post-processing:
  - Least-Squares (skimage default)
  - Quality-Guided (reliability-based, unwraps high-quality pixels first)
  - Goldstein Branch-Cut (places branch cuts between residues)

Pre-processing:  median filter, Gaussian smooth
Post-processing: outlier clipping, polynomial background removal
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy.ndimage import median_filter, gaussian_filter


# ═══════════════════════════════════════════════════════════════════════
#  Enums & Config
# ═══════════════════════════════════════════════════════════════════════

class UnwrapMethod(str, Enum):
    GRADIENT_INTEGRATION = "gradient_integration"
    TIE = "tie"
    THIN_SAMPLE = "thin_sample"
    OPTICAL = "optical"
    LEAST_SQUARES = "least_squares"
    QUALITY_GUIDED = "quality_guided"
    GOLDSTEIN = "goldstein"


@dataclass
class UnwrapConfig:
    """Full configuration for the unwrapping pipeline."""
    method: UnwrapMethod = UnwrapMethod.GRADIENT_INTEGRATION

    # Pre-processing
    pre_median: bool = False
    pre_median_size: int = 3
    pre_gaussian: bool = False
    pre_gaussian_sigma: float = 0.8

    # Post-processing
    post_outlier_clip: bool = False
    post_outlier_sigma: float = 3.0
    post_bg_remove: bool = True
    post_bg_order: int = 2

    # TIE-specific parameters
    tie_dz_um: float = 1.0           # propagation step δz in micrometres
    tie_regularization: float = 0.01  # Tikhonov reg. for low-freq stability

    # Optical unwrapping parameters
    optical_synth_nm: float = 10.0    # synthetic Δλ offset in nm


# ═══════════════════════════════════════════════════════════════════════
#  Quality Map Computation
# ═══════════════════════════════════════════════════════════════════════

def _phase_derivative_variance(wrapped: np.ndarray, kernel: int = 3) -> np.ndarray:
    """
    Compute Phase Derivative Variance (PDV) quality map.

    Higher values → more reliable pixels.
    PDV measures the local consistency of phase gradients.
    """
    dy = np.diff(wrapped, axis=0)
    dx = np.diff(wrapped, axis=1)

    # Wrap differences to [-pi, pi]
    dy = np.angle(np.exp(1j * dy))
    dx = np.angle(np.exp(1j * dx))

    # Pad to original size
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')

    # Local variance of derivatives → low variance = high quality
    from scipy.ndimage import uniform_filter
    dy_var = uniform_filter(dy ** 2, size=kernel) - uniform_filter(dy, size=kernel) ** 2
    dx_var = uniform_filter(dx ** 2, size=kernel) - uniform_filter(dx, size=kernel) ** 2

    total_var = dy_var + dx_var
    # Invert: high quality where variance is low
    max_var = np.max(total_var)
    if max_var > 0:
        quality = 1.0 - (total_var / max_var)
    else:
        quality = np.ones_like(total_var)

    return quality.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════
#  Unwrapping Algorithms
# ═══════════════════════════════════════════════════════════════════════

def _unwrap_gradient_integration(complex_field: np.ndarray) -> np.ndarray:
    """
    Phase recovery via gradient integration (DCT Poisson solver).

    Gold-standard for DHM. Computes phase gradients directly from E:
        dphi/dx = Im[ (dE/dx) * conj(E) ] / |E|^2
    These gradients never have 2pi jumps. Integrated via Poisson/DCT.
    """
    from scipy.fft import dctn, idctn

    E = complex_field.astype(np.complex128)
    amp2 = np.abs(E) ** 2
    thr = np.percentile(amp2[amp2 > 0], 1) * 0.01 if np.any(amp2 > 0) else 1e-20
    amp2_reg = np.maximum(amp2, thr)

    # Central-difference gradients of E
    dEdx = np.zeros_like(E)
    dEdx[:, 1:-1] = (E[:, 2:] - E[:, :-2]) / 2.0
    dEdx[:, 0] = E[:, 1] - E[:, 0]
    dEdx[:, -1] = E[:, -1] - E[:, -2]

    dEdy = np.zeros_like(E)
    dEdy[1:-1, :] = (E[2:, :] - E[:-2, :]) / 2.0
    dEdy[0, :] = E[1, :] - E[0, :]
    dEdy[-1, :] = E[-1, :] - E[-2, :]

    # Phase gradients (inherently unwrapped)
    dphi_dx = np.imag(dEdx * np.conj(E)) / amp2_reg
    dphi_dy = np.imag(dEdy * np.conj(E)) / amp2_reg

    # Laplacian from gradients
    lap = np.zeros_like(dphi_dx, dtype=np.float64)
    lap[:, 1:-1] += (dphi_dx[:, 2:] - dphi_dx[:, :-2]) / 2.0
    lap[:, 0] += dphi_dx[:, 1] - dphi_dx[:, 0]
    lap[:, -1] += dphi_dx[:, -1] - dphi_dx[:, -2]
    lap[1:-1, :] += (dphi_dy[2:, :] - dphi_dy[:-2, :]) / 2.0
    lap[0, :] += dphi_dy[1, :] - dphi_dy[0, :]
    lap[-1, :] += dphi_dy[-1, :] - dphi_dy[-2, :]

    # Solve Poisson via DCT (Neumann BC)
    ny, nx = lap.shape
    L_hat = dctn(lap, type=2, norm='ortho')

    j_idx = np.arange(ny).reshape(-1, 1)
    i_idx = np.arange(nx).reshape(1, -1)
    denom = 2.0 * (np.cos(np.pi * j_idx / ny) - 1.0) + \
            2.0 * (np.cos(np.pi * i_idx / nx) - 1.0)
    denom[0, 0] = 1.0

    phi_hat = L_hat / denom
    phi_hat[0, 0] = 0.0

    phase = idctn(phi_hat, type=2, norm='ortho')
    return phase.astype(np.float64)


def _solve_poisson_dct(rhs: np.ndarray) -> np.ndarray:
    """Solve ∇²φ = rhs via DCT with Neumann BC.  Shared by several methods."""
    from scipy.fft import dctn, idctn
    ny, nx = rhs.shape
    L_hat = dctn(rhs.astype(np.float64), type=2, norm='ortho')
    j = np.arange(ny).reshape(-1, 1)
    i = np.arange(nx).reshape(1, -1)
    denom = 2.0 * (np.cos(np.pi * j / ny) - 1.0) + \
            2.0 * (np.cos(np.pi * i / nx) - 1.0)
    denom[0, 0] = 1.0
    phi_hat = L_hat / denom
    phi_hat[0, 0] = 0.0
    return idctn(phi_hat, type=2, norm='ortho')


def _unwrap_tie(complex_field: np.ndarray,
                wavelength_m: float,
                pixel_size_m: float,
                dz_m: float = 1e-6,
                regularization: float = 0.01) -> np.ndarray:
    """
    Transport of Intensity Equation (TIE) phase recovery.

    TIE:  -k · ∂I/∂z = ∇·(I₀ ∇φ)

    All gradient/divergence/Poisson operations use pixel-unit coordinates
    (grid spacing = 1).  Physical scaling enters only via  k · dI/dz · dx².

    Two-step auxiliary-variable Poisson solve:
      1) ∇²_pix ψ  = -k · (∂I/∂z) · dx²
      2) ∇²_pix φ  = ∇_pix · (∇_pix ψ / I₀)
    """
    E = complex_field.astype(np.complex128)
    ny, nx = E.shape
    k = 2.0 * np.pi / wavelength_m
    dx = pixel_size_m

    # --- ASM transfer function for ±dz ---
    fy = np.fft.fftfreq(ny, d=dx).astype(np.float64)
    fx = np.fft.fftfreq(nx, d=dx).astype(np.float64)
    FX, FY = np.meshgrid(fx, fy)
    arg = np.maximum(1.0 - wavelength_m ** 2 * (FX ** 2 + FY ** 2), 0.0)
    sqrt_arg = np.sqrt(arg)

    H_pos = np.exp(1j * k * dz_m * sqrt_arg)
    H_neg = np.exp(-1j * k * dz_m * sqrt_arg)

    E_spec = np.fft.fft2(E)
    I_pos = np.abs(np.fft.ifft2(E_spec * H_pos)) ** 2
    I_neg = np.abs(np.fft.ifft2(E_spec * H_neg)) ** 2
    I0 = np.abs(E) ** 2

    dIdz = (I_pos - I_neg) / (2.0 * dz_m)

    I0_reg = np.maximum(I0, np.max(I0) * regularization)

    # Step 1:  ∇²_pix ψ = -k · dI/dz · dx²
    rhs1 = -k * dIdz * (dx ** 2)
    psi = _solve_poisson_dct(rhs1)

    # Step 2:  pixel-unit gradient of ψ
    dpsi_dx = np.zeros_like(psi)
    dpsi_dx[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / 2.0
    dpsi_dx[:, 0] = psi[:, 1] - psi[:, 0]
    dpsi_dx[:, -1] = psi[:, -1] - psi[:, -2]

    dpsi_dy = np.zeros_like(psi)
    dpsi_dy[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / 2.0
    dpsi_dy[0, :] = psi[1, :] - psi[0, :]
    dpsi_dy[-1, :] = psi[-1, :] - psi[-2, :]

    vx = dpsi_dx / I0_reg
    vy = dpsi_dy / I0_reg

    # Pixel-unit divergence of v
    div_v = np.zeros_like(vx)
    div_v[:, 1:-1] += (vx[:, 2:] - vx[:, :-2]) / 2.0
    div_v[:, 0] += vx[:, 1] - vx[:, 0]
    div_v[:, -1] += vx[:, -1] - vx[:, -2]
    div_v[1:-1, :] += (vy[2:, :] - vy[:-2, :]) / 2.0
    div_v[0, :] += vy[1, :] - vy[0, :]
    div_v[-1, :] += vy[-1, :] - vy[-2, :]

    # Step 3:  ∇²_pix φ = div_v
    phase = _solve_poisson_dct(div_v)
    return phase.astype(np.float64)


def _unwrap_thin_sample(complex_field: np.ndarray) -> np.ndarray:
    """
    Thin-sample phase approximation.

    For thin transparent samples where OPD < λ/2 (phase < π),
    the wrapped phase = true phase (no wrapping occurs).

    Steps:
      1. Subtract the mean background phasor to center phase around 0
      2. Use atan2 (exact, not linearised) to extract phase
      3. Since |φ| < π for thin samples, the result is correct without unwrapping

    Very fast — no FFT or iterative solve needed.
    Best for biological cells with small OPD.
    """
    E = complex_field.astype(np.complex128)

    # Remove mean background phasor so sample phase is centered at 0
    E_mean = np.mean(E)
    if np.abs(E_mean) > 0:
        E_demod = E * np.conj(E_mean) / np.abs(E_mean)
    else:
        E_demod = E

    # Exact phase via atan2 — doesn't wrap for |φ| < π
    phase = np.angle(E_demod)

    return phase.astype(np.float64)


def _unwrap_optical(complex_field: np.ndarray,
                    wavelength_m: float,
                    pixel_size_m: float,
                    delta_lambda_m: float = 10e-9) -> np.ndarray:
    """
    Optical (synthetic-wavelength) phase unwrapping.

    Simulates dual-wavelength acquisition by numerical re-propagation:
      - λ₁ = λ  (original)
      - λ₂ = λ + Δλ  (synthetic offset)

    The synthetic wavelength  Λ = λ₁·λ₂ / |λ₁ - λ₂|  is much larger,
    giving an extended unambiguous range.

    Steps:
      1. Original phase φ₁ = angle(E) at wavelength λ₁
      2. Re-propagate to z=0 with λ₁, then forward-propagate with λ₂
         (simulating the same object at a different wavelength)
      3. φ₂ = angle(E₂)
      4. Coarse phase from synthetic wavelength: Φ = φ₁ - φ₂
      5. Use Φ to guide the fine unwrapping of φ₁
    """
    E = complex_field.astype(np.complex128)
    ny, nx = E.shape

    lam1 = wavelength_m
    lam2 = wavelength_m + delta_lambda_m
    lam_synth = (lam1 * lam2) / abs(lam1 - lam2)

    k1 = 2.0 * np.pi / lam1
    k2 = 2.0 * np.pi / lam2

    # Build frequency grids
    dy = dx = pixel_size_m
    fy = np.fft.fftfreq(ny, d=dy).astype(np.float64)
    fx = np.fft.fftfreq(nx, d=dx).astype(np.float64)
    FX, FY = np.meshgrid(fx, fy)
    freq_sq = FX ** 2 + FY ** 2

    # Simulate λ₂ observation: undo λ₁ propagation, redo with λ₂
    # H_back(λ₁) · H_fwd(λ₂) at a small reference distance z_ref
    # z_ref chosen so that the phase difference is meaningful
    z_ref = lam_synth * 0.1  # small fraction of synthetic wavelength

    arg1 = np.maximum(1.0 - lam1 ** 2 * freq_sq, 0.0)
    arg2 = np.maximum(1.0 - lam2 ** 2 * freq_sq, 0.0)

    H_transfer = np.exp(1j * z_ref * (k2 * np.sqrt(arg2) - k1 * np.sqrt(arg1)))

    E_spec = np.fft.fft2(E)
    E2 = np.fft.ifft2(E_spec * H_transfer)

    phi1 = np.angle(E)
    phi2 = np.angle(E2)

    # Synthetic (coarse) phase — unambiguous over Λ
    phi_synth = np.angle(np.exp(1j * (phi1 - phi2)))  # re-wrap to [-π,π]

    # Scale coarse phase to fine wavelength
    phi_coarse = phi_synth * (lam_synth / lam1)

    # Use coarse phase to determine the correct 2π multiple for each pixel
    n_wraps = np.round((phi_coarse - phi1) / (2.0 * np.pi))
    phase = phi1 + 2.0 * np.pi * n_wraps

    # Remove median offset (arbitrary constant from the synthetic approach)
    phase = phase - np.median(phase)

    return phase.astype(np.float64)


def _unwrap_least_squares(wrapped: np.ndarray) -> np.ndarray:
    """Least-squares unwrapping via skimage (baseline method)."""
    from skimage.restoration import unwrap_phase
    return unwrap_phase(wrapped).astype(np.float64)


def _unwrap_quality_guided(wrapped: np.ndarray) -> np.ndarray:
    """
    Quality-guided phase unwrapping with proper heap-based priority queue.

    Seed from highest-quality pixel, dynamically push neighbours into a
    max-heap. Always unwrap the highest-quality frontier pixel next.
    Errors cannot propagate from noisy into clean regions.
    """
    import heapq

    ny, nx = wrapped.shape
    quality = _phase_derivative_variance(wrapped)

    unwrapped = np.zeros_like(wrapped, dtype=np.float64)
    visited = np.zeros((ny, nx), dtype=np.bool_)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    best = np.unravel_index(np.argmax(quality), quality.shape)
    unwrapped[best] = wrapped[best]
    visited[best] = True

    heap = []  # max-heap via negated quality

    def _push(y, x):
        for dy, dx in nbrs:
            ny2, nx2 = y + dy, x + dx
            if 0 <= ny2 < ny and 0 <= nx2 < nx and not visited[ny2, nx2]:
                q = min(quality[y, x], quality[ny2, nx2])
                heapq.heappush(heap, (-q, ny2, nx2, y, x))

    _push(best[0], best[1])

    while heap:
        _, ty, tx, sy, sx = heapq.heappop(heap)
        if visited[ty, tx]:
            continue
        diff = wrapped[ty, tx] - wrapped[sy, sx]
        diff -= 2.0 * np.pi * np.round(diff / (2.0 * np.pi))
        unwrapped[ty, tx] = unwrapped[sy, sx] + diff
        visited[ty, tx] = True
        _push(ty, tx)

    if not np.all(visited):
        uy, ux = np.where(~visited)
        for y, x in zip(uy, ux):
            unwrapped[y, x] = wrapped[y, x]

    return unwrapped


def _unwrap_goldstein(wrapped: np.ndarray) -> np.ndarray:
    """
    Goldstein branch-cut phase unwrapping.

    Strategy:
      1. Detect phase residues (positive and negative)
      2. Connect residues with branch cuts (shortest paths)
      3. Flood-fill unwrap avoiding branch cuts

    Better than least-squares for data with sharp boundaries or defects.
    """
    ny, nx = wrapped.shape

    # ── Step 1: Detect residues ──
    # Compute wrapped phase differences along a 2×2 loop
    def _wrap(x):
        return np.angle(np.exp(1j * x))

    d1 = _wrap(wrapped[:-1, 1:] - wrapped[:-1, :-1])    # right
    d2 = _wrap(wrapped[1:, 1:] - wrapped[:-1, 1:])       # down
    d3 = _wrap(wrapped[1:, :-1] - wrapped[1:, 1:])       # left
    d4 = _wrap(wrapped[:-1, :-1] - wrapped[1:, :-1])     # up

    residue_sum = d1 + d2 + d3 + d4
    # Residues are at positions where sum ≈ ±2π
    residues = np.zeros((ny, nx), dtype=np.int8)
    residues[:-1, :-1] = np.round(residue_sum / (2.0 * np.pi)).astype(np.int8)

    pos_y, pos_x = np.where(residues > 0)
    neg_y, neg_x = np.where(residues < 0)

    # ── Step 2: Place branch cuts ──
    branch_cut = np.zeros((ny, nx), dtype=np.bool_)

    # Connect nearest positive-negative residue pairs with Bresenham lines
    pos_used = np.zeros(len(pos_y), dtype=bool)
    neg_used = np.zeros(len(neg_y), dtype=bool)

    for pi in range(len(pos_y)):
        if pos_used[pi]:
            continue
        best_dist = np.inf
        best_ni = -1
        py, px_ = int(pos_y[pi]), int(pos_x[pi])
        for ni in range(len(neg_y)):
            if neg_used[ni]:
                continue
            d = abs(py - int(neg_y[ni])) + abs(px_ - int(neg_x[ni]))
            if d < best_dist:
                best_dist = d
                best_ni = ni
        if best_ni >= 0 and best_dist < max(ny, nx) // 2:
            pos_used[pi] = True
            neg_used[best_ni] = True
            # Draw branch cut line (Bresenham)
            _draw_line(branch_cut, py, px_, int(neg_y[best_ni]), int(neg_x[best_ni]))

    # Connect unpaired residues to nearest border
    for pi in range(len(pos_y)):
        if not pos_used[pi]:
            _connect_to_border(branch_cut, int(pos_y[pi]), int(pos_x[pi]), ny, nx)
    for ni in range(len(neg_y)):
        if not neg_used[ni]:
            _connect_to_border(branch_cut, int(neg_y[ni]), int(neg_x[ni]), ny, nx)

    # ── Step 3: Flood-fill unwrap avoiding branch cuts ──
    unwrapped = np.zeros_like(wrapped, dtype=np.float64)
    visited = np.zeros((ny, nx), dtype=np.bool_)

    # Start from center
    start_y, start_x = ny // 2, nx // 2
    unwrapped[start_y, start_x] = wrapped[start_y, start_x]
    visited[start_y, start_x] = True

    from collections import deque
    queue = deque()
    queue.append((start_y, start_x))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in neighbors:
            ny2, nx2 = cy + dy, cx + dx
            if 0 <= ny2 < ny and 0 <= nx2 < nx and not visited[ny2, nx2]:
                if branch_cut[ny2, nx2]:
                    continue
                diff = wrapped[ny2, nx2] - wrapped[cy, cx]
                diff = diff - 2.0 * np.pi * np.round(diff / (2.0 * np.pi))
                unwrapped[ny2, nx2] = unwrapped[cy, cx] + diff
                visited[ny2, nx2] = True
                queue.append((ny2, nx2))

    # Fill branch-cut pixels with neighbor average
    if not np.all(visited):
        _fill_unvisited(unwrapped, visited, wrapped)

    return unwrapped


def _draw_line(grid: np.ndarray, y0: int, x0: int, y1: int, x1: int):
    """Bresenham line drawing on boolean grid."""
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < grid.shape[0] and 0 <= x0 < grid.shape[1]:
            grid[y0, x0] = True
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _connect_to_border(grid: np.ndarray, y: int, x: int, ny: int, nx: int):
    """Connect a residue to the nearest image border with a branch cut."""
    dist_top = y
    dist_bot = ny - 1 - y
    dist_left = x
    dist_right = nx - 1 - x
    min_dist = min(dist_top, dist_bot, dist_left, dist_right)

    if min_dist == dist_top:
        _draw_line(grid, y, x, 0, x)
    elif min_dist == dist_bot:
        _draw_line(grid, y, x, ny - 1, x)
    elif min_dist == dist_left:
        _draw_line(grid, y, x, y, 0)
    else:
        _draw_line(grid, y, x, y, nx - 1)


def _fill_unvisited(unwrapped: np.ndarray, visited: np.ndarray, wrapped: np.ndarray):
    """Fill unvisited pixels by averaging visited neighbors, or fallback to wrapped."""
    ny, nx = unwrapped.shape
    unvis_y, unvis_x = np.where(~visited)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y, x in zip(unvis_y, unvis_x):
        vals = []
        for dy, dx in neighbors:
            ny2, nx2 = y + dy, x + dx
            if 0 <= ny2 < ny and 0 <= nx2 < nx and visited[ny2, nx2]:
                diff = wrapped[y, x] - wrapped[ny2, nx2]
                diff = diff - 2.0 * np.pi * np.round(diff / (2.0 * np.pi))
                vals.append(unwrapped[ny2, nx2] + diff)
        if vals:
            unwrapped[y, x] = np.mean(vals)
        else:
            unwrapped[y, x] = wrapped[y, x]
    visited[unvis_y, unvis_x] = True


# ═══════════════════════════════════════════════════════════════════════
#  Pre/Post Processing
# ═══════════════════════════════════════════════════════════════════════

def _preprocess(wrapped: np.ndarray, cfg: UnwrapConfig) -> np.ndarray:
    """Apply pre-processing filters to wrapped phase before unwrapping."""
    phase = wrapped.copy()
    if cfg.pre_median:
        # Median filter on the complex representation to preserve wrapping
        cos_p = median_filter(np.cos(phase), size=cfg.pre_median_size)
        sin_p = median_filter(np.sin(phase), size=cfg.pre_median_size)
        phase = np.arctan2(sin_p, cos_p)
    if cfg.pre_gaussian:
        cos_p = gaussian_filter(np.cos(phase), sigma=cfg.pre_gaussian_sigma)
        sin_p = gaussian_filter(np.sin(phase), sigma=cfg.pre_gaussian_sigma)
        phase = np.arctan2(sin_p, cos_p)
    return phase


def _postprocess(unwrapped: np.ndarray, cfg: UnwrapConfig) -> np.ndarray:
    """Apply post-processing to unwrapped phase."""
    result = unwrapped.copy()

    if cfg.post_outlier_clip:
        mean = np.mean(result)
        std = np.std(result)
        sigma = cfg.post_outlier_sigma
        lo = mean - sigma * std
        hi = mean + sigma * std
        result = np.clip(result, lo, hi)

    if cfg.post_bg_remove:
        result = _remove_polynomial_bg(result, order=cfg.post_bg_order)

    return result


def _remove_polynomial_bg(phase: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Remove polynomial background from unwrapped phase (tilt + curvature).

    Uses iterative sigma-clipping to exclude cell/feature pixels from the
    polynomial fit. This prevents the sample's OPD from biasing the
    background estimate — critical for accurate QPI of biological cells.

    Iteration:
      1. Fit polynomial to all pixels
      2. Compute residuals, mask pixels > 2σ from fit
      3. Re-fit using only background (unmasked) pixels
      4. Repeat until convergence (typically 2-3 iterations)
    """
    ny, nx = phase.shape
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    y_norm = (y_coords.ravel() - ny / 2.0) / (ny / 2.0)
    x_norm = (x_coords.ravel() - nx / 2.0) / (nx / 2.0)

    # Build polynomial basis
    cols = [np.ones_like(x_norm)]  # constant
    if order >= 1:
        cols.extend([x_norm, y_norm])
    if order >= 2:
        cols.extend([x_norm ** 2, y_norm ** 2, x_norm * y_norm])
    if order >= 3:
        cols.extend([
            x_norm ** 3, y_norm ** 3,
            x_norm ** 2 * y_norm, x_norm * y_norm ** 2,
        ])

    A = np.column_stack(cols).astype(np.float64)
    b = phase.ravel().astype(np.float64)

    # Sanitise input — if phase has NaN/Inf the fit will explode
    finite_mask = np.isfinite(b)
    if not np.all(finite_mask):
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    def _safe_lstsq(A_sub, b_sub):
        try:
            c, _, _, _ = np.linalg.lstsq(A_sub, b_sub, rcond=None)
            if not np.all(np.isfinite(c)):
                return None
            # Reject exploding coefficients that will overflow in A @ c
            if np.max(np.abs(c)) > 1e12:
                return None
            return c
        except (np.linalg.LinAlgError, ValueError):
            return None

    n_cols = A.shape[1]

    # Iterative sigma-clipping: exclude sample pixels from background fit
    bg_mask = np.ones(len(b), dtype=bool)
    coeffs = None
    for _iteration in range(3):
        n_bg = int(np.sum(bg_mask))
        if n_bg < n_cols + 1:
            break
        coeffs = _safe_lstsq(A[bg_mask], b[bg_mask])
        if coeffs is None:
            break
        with np.errstate(all='ignore'):
            residuals = b - A @ coeffs
        if not np.all(np.isfinite(residuals)):
            coeffs = None
            break
        res_std = np.std(residuals[bg_mask])
        if res_std < 1e-12:
            break
        bg_mask = np.abs(residuals) < 2.0 * res_std

    # Final fit with cleaned background pixels
    n_bg = int(np.sum(bg_mask))
    if coeffs is not None and n_bg > n_cols + 1:
        final = _safe_lstsq(A[bg_mask], b[bg_mask])
        if final is not None:
            coeffs = final

    if coeffs is None:
        return phase  # fit failed — return unmodified

    with np.errstate(all='ignore'):
        bg = (A @ coeffs).reshape(ny, nx)
    if not np.all(np.isfinite(bg)):
        return phase  # numerical failure — return unmodified

    return phase - bg


# ═══════════════════════════════════════════════════════════════════════
#  Main Public API
# ═══════════════════════════════════════════════════════════════════════

def unwrap_phase_advanced(
    wrapped: np.ndarray,
    config: Optional[UnwrapConfig] = None,
    complex_field: Optional[np.ndarray] = None,
    wavelength_m: Optional[float] = None,
    pixel_size_m: Optional[float] = None,
) -> np.ndarray:
    """
    Full phase unwrapping pipeline: preprocess → unwrap → postprocess.

    Parameters
    ----------
    wrapped : 2D array of wrapped phase values (radians, [-π, π])
    config : UnwrapConfig or None (uses defaults)
    complex_field : optional 2D complex array.  Required for:
                    gradient_integration, TIE, thin_sample, optical.
    wavelength_m : laser wavelength in metres.  Required for TIE, optical.
    pixel_size_m : effective pixel size in metres.  Required for TIE, optical.

    Returns
    -------
    unwrapped : 2D array of unwrapped phase (radians)
    """
    if config is None:
        config = UnwrapConfig()

    if wrapped.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {wrapped.shape}")

    # --- Complex-field-based methods (no traditional unwrapping) ---
    if complex_field is not None:
        if config.method == UnwrapMethod.GRADIENT_INTEGRATION:
            result = _unwrap_gradient_integration(complex_field)
            return _postprocess(result, config).astype(np.float32)

        if config.method == UnwrapMethod.THIN_SAMPLE:
            result = _unwrap_thin_sample(complex_field)
            return _postprocess(result, config).astype(np.float32)

        if config.method == UnwrapMethod.TIE and wavelength_m and pixel_size_m:
            dz = config.tie_dz_um * 1e-6
            result = _unwrap_tie(complex_field, wavelength_m, pixel_size_m,
                                 dz_m=dz, regularization=config.tie_regularization)
            return _postprocess(result, config).astype(np.float32)

        if config.method == UnwrapMethod.OPTICAL and wavelength_m and pixel_size_m:
            dlam = config.optical_synth_nm * 1e-9
            result = _unwrap_optical(complex_field, wavelength_m, pixel_size_m,
                                     delta_lambda_m=dlam)
            return _postprocess(result, config).astype(np.float32)

    # --- Traditional wrapped-phase methods ---
    phase = wrapped.astype(np.float64)
    phase = _preprocess(phase, config)

    if config.method == UnwrapMethod.QUALITY_GUIDED:
        result = _unwrap_quality_guided(phase)
    elif config.method == UnwrapMethod.GOLDSTEIN:
        result = _unwrap_goldstein(phase)
    else:
        result = _unwrap_least_squares(phase)

    result = _postprocess(result, config)
    return result.astype(np.float32)


def get_quality_map(wrapped: np.ndarray) -> np.ndarray:
    """
    Compute and return the phase quality map (0-1).
    Useful for visualization / debugging.
    """
    return _phase_derivative_variance(wrapped.astype(np.float64))
