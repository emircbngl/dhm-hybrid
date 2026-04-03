"""
Quantitative Phase Imaging (QPI) — Parameter Computation Module
================================================================

Unwrapped phase → OPD → {height, dry mass, roughness, RI, ...}

Two usage modes:
  - Micro-structures / thin film : height, roughness (Ra/Rq/Rz), step height, film thickness
  - Biological cells             : dry mass, dry mass density, cell area, volume, growth rate

All functions take unwrapped phase (radians) and optical parameters,
returning physical quantities in SI or commonly-used units.

Display outputs:
  1. OPD map (nm)                — colorbar, min/max/mean overlay
  2. Height / topography (nm)    — 3D surface optional
  3. Dry mass density (pg/μm²)   — for bio, per-pixel mass map
  4. Surface roughness panel     — Ra, Rq, Rz, PSD
  5. Cell morphology stats       — area, volume, dry mass, eccentricity
  6. Refractive index map        — if thickness is known
  7. Phase statistics panel      — σ, range, histogram
  8. Line profile tool           — cross-section at user-drawn line
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. OPTICAL PATH DIFFERENCE (OPD)
# ═══════════════════════════════════════════════════════════════

def phase_to_opd(
    phase_unwrapped: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """
    Convert unwrapped phase (radians) to Optical Path Difference.

    OPD(x,y) = φ(x,y) × λ / (2π)

    Returns: OPD in metres (same unit as wavelength_m).
    """
    return phase_unwrapped * wavelength_m / (2.0 * np.pi)


# ═══════════════════════════════════════════════════════════════
# 2. HEIGHT / TOPOGRAPHY
# ═══════════════════════════════════════════════════════════════

def opd_to_height(
    opd_m: np.ndarray,
    n_sample: float,
    n_medium: float = 1.0,
) -> np.ndarray:
    """
    Convert OPD to physical height (thickness).

    h(x,y) = OPD(x,y) / (n_sample - n_medium)

    Requires knowing the sample's refractive index.
    For reflection DHM: h = OPD / 2  (double pass).

    Returns: height in metres.
    """
    dn = n_sample - n_medium
    if abs(dn) < 1e-12:
        raise ValueError("Δn ≈ 0 — cannot compute height without refractive index contrast")
    return opd_m / dn


def opd_to_height_reflection(opd_m: np.ndarray) -> np.ndarray:
    """
    Height from OPD for reflection-mode DHM.
    h = OPD / 2  (light travels the surface height twice).
    """
    return opd_m / 2.0


# ═══════════════════════════════════════════════════════════════
# 3. REFRACTIVE INDEX MAP
# ═══════════════════════════════════════════════════════════════

def opd_to_refractive_index(
    opd_m: np.ndarray,
    thickness_m: float,
    n_medium: float = 1.0,
) -> np.ndarray:
    """
    Compute local refractive index from OPD (when thickness is known).

    n(x,y) = n_medium + OPD(x,y) / h

    Returns: refractive index map (dimensionless).
    """
    if abs(thickness_m) < 1e-15:
        raise ValueError("Thickness ≈ 0 — cannot compute RI")
    return n_medium + opd_m / thickness_m


# ═══════════════════════════════════════════════════════════════
# 4. DRY MASS (Biological)
# ═══════════════════════════════════════════════════════════════

# Common specific refractive increments (dn/dc) in mL/g:
#   Protein:       0.18 - 0.19
#   DNA:           0.16 - 0.18
#   Lipid:         0.14 - 0.16
#   General cell:  0.18  (default, protein-dominated)
DEFAULT_ALPHA = 0.18e-6  # m³/kg  (= 0.18 mL/g in SI)


def compute_dry_mass_density(
    opd_m: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """
    Dry mass surface density at each pixel.

    σ(x,y) = OPD(x,y) / α

    where α = specific refractive increment (m³/kg).
    Default α = 0.18 mL/g (protein).

    Returns: dry mass density in kg/m² (convert to pg/μm² for display).
    """
    return opd_m / alpha


def compute_dry_mass_total(
    opd_m: np.ndarray,
    pixel_size_m: float,
    alpha: float = DEFAULT_ALPHA,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Total dry mass of a cell or region.

    M = (1/α) × ∫∫ OPD(x,y) dA

    If mask is provided (binary), integration is limited to masked region.

    Returns: dry mass in kg (convert to pg for display: × 1e15).
    """
    pixel_area = pixel_size_m ** 2
    if mask is not None:
        integral = float(np.sum(opd_m[mask > 0])) * pixel_area
    else:
        integral = float(np.sum(opd_m)) * pixel_area
    return integral / alpha


# ═══════════════════════════════════════════════════════════════
# 5. SURFACE ROUGHNESS (Micro-structures)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RoughnessResult:
    """ISO 25178 surface roughness parameters."""
    Ra: float   # Arithmetic mean deviation (Sa for 2D)
    Rq: float   # Root mean square roughness (Sq)
    Rz: float   # Peak-to-valley (Sz)
    Rsk: float  # Skewness (Ssk) — >0 peaks, <0 valleys
    Rku: float  # Kurtosis (Sku) — >3 sharp features
    unit: str = "m"


def compute_roughness(
    height_m: np.ndarray,
    roi: Optional[np.ndarray] = None,
) -> RoughnessResult:
    """
    ISO 25178 surface roughness parameters from height map.

    If roi (binary mask) is provided, only those pixels are used.
    Height should be leveled (plane-subtracted) before calling this.
    """
    if roi is not None:
        h = height_m[roi > 0]
    else:
        h = height_m.ravel()

    h = h - np.mean(h)  # Remove mean (residual tilt)

    ra = float(np.mean(np.abs(h)))
    rq = float(np.sqrt(np.mean(h ** 2)))
    rz = float(np.max(h) - np.min(h))

    if rq > 1e-20:
        rsk = float(np.mean(h ** 3) / rq ** 3)
        rku = float(np.mean(h ** 4) / rq ** 4)
    else:
        rsk = 0.0
        rku = 3.0  # Gaussian

    return RoughnessResult(Ra=ra, Rq=rq, Rz=rz, Rsk=rsk, Rku=rku)


def level_plane(height_m: np.ndarray) -> np.ndarray:
    """
    Remove best-fit plane from height map (tilt correction).
    Essential before roughness calculation.
    """
    ny, nx = height_m.shape
    y, x = np.mgrid[:ny, :nx].astype(np.float64)

    # Least squares: h = a*x + b*y + c
    A = np.column_stack([x.ravel(), y.ravel(), np.ones(nx * ny)])
    h = height_m.ravel().astype(np.float64)
    coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)

    plane = (coeffs[0] * x + coeffs[1] * y + coeffs[2])
    return height_m - plane.astype(height_m.dtype)


# ═══════════════════════════════════════════════════════════════
# 6. CELL MORPHOLOGY (Biological)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CellMorphology:
    """Morphological parameters from phase-based segmentation."""
    area_um2: float              # Projected area (μm²)
    volume_fL: float             # Volume in femtolitres
    dry_mass_pg: float           # Total dry mass (picograms)
    dry_mass_density_pg_um2: float  # Mean dry mass density
    max_height_nm: float         # Maximum height
    mean_height_nm: float        # Mean height within cell
    perimeter_um: float          # Perimeter
    circularity: float           # 4π × area / perimeter²
    aspect_ratio: float          # Major/minor axis
    eccentricity: float          # √(1 - (minor/major)²)
    mean_phase_rad: float        # Mean phase within cell
    phase_std_rad: float         # Phase standard deviation


def segment_cell_phase(
    opd_m: np.ndarray,
    threshold_factor: float = 0.15,
) -> np.ndarray:
    """
    Simple phase/OPD-based cell segmentation.

    Thresholds at `threshold_factor × (max - min) + min` of OPD,
    then cleans with morphological operations.

    Returns: binary mask (uint8, 0/255).
    """
    from scipy.ndimage import binary_fill_holes, binary_opening, label

    mn, mx = opd_m.min(), opd_m.max()
    thresh = mn + threshold_factor * (mx - mn)
    binary = (opd_m > thresh).astype(np.uint8)

    # Clean up
    binary = binary_fill_holes(binary).astype(np.uint8)
    struct = np.ones((3, 3), dtype=np.uint8)
    binary = binary_opening(binary, structure=struct, iterations=2).astype(np.uint8)

    # Keep only largest connected component
    labeled, n_features = label(binary)
    if n_features > 1:
        sizes = [np.sum(labeled == i) for i in range(1, n_features + 1)]
        largest = np.argmax(sizes) + 1
        binary = (labeled == largest).astype(np.uint8)

    return binary * 255


def compute_cell_morphology(
    opd_m: np.ndarray,
    pixel_size_m: float,
    mask: np.ndarray,
    n_sample: float = 1.38,
    n_medium: float = 1.337,
    alpha: float = DEFAULT_ALPHA,
) -> CellMorphology:
    """
    Compute full cell morphology from OPD + binary mask.

    Parameters
    ----------
    opd_m : 2D OPD array in metres
    pixel_size_m : physical pixel size
    mask : binary mask (>0 = cell)
    n_sample : average cell refractive index (default 1.38 for mammalian cells)
    n_medium : medium refractive index (default 1.337 for culture medium)
    alpha : specific refractive increment

    Returns: CellMorphology dataclass with all parameters.
    """
    px = pixel_size_m
    px_um = px * 1e6
    px_area_um2 = px_um ** 2

    cell_mask = mask > 0
    n_pixels = int(np.sum(cell_mask))

    if n_pixels < 4:
        raise ValueError("Cell mask too small (< 4 pixels)")

    opd_cell = opd_m[cell_mask]

    # Area
    area_um2 = n_pixels * px_area_um2

    # Height
    dn = n_sample - n_medium
    if abs(dn) > 1e-12:
        h_m = opd_cell / dn
        h_nm = h_m * 1e9
    else:
        h_nm = opd_cell * 1e9 / 0.043  # fallback dn~0.043
        h_m = h_nm * 1e-9

    max_height_nm = float(np.max(h_nm))
    mean_height_nm = float(np.mean(h_nm))

    # Volume: integral of height over area
    volume_m3 = float(np.sum(h_m)) * (px ** 2)
    volume_fL = volume_m3 * 1e18  # m³ → fL (1 fL = 1e-18 m³)

    # Dry mass
    dry_mass_kg = float(np.sum(opd_cell)) * (px ** 2) / alpha
    dry_mass_pg = dry_mass_kg * 1e15

    # Dry mass density
    dry_mass_density = dry_mass_pg / area_um2 if area_um2 > 0 else 0.0

    # Perimeter (simple: count boundary pixels)
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(cell_mask)
    boundary = cell_mask & ~eroded
    perimeter_pixels = int(np.sum(boundary))
    perimeter_um = perimeter_pixels * px_um

    # Circularity
    if perimeter_um > 0:
        circularity = 4 * np.pi * area_um2 / (perimeter_um ** 2)
    else:
        circularity = 0.0

    # Bounding ellipse (moments-based)
    ys, xs = np.where(cell_mask)
    if len(xs) > 0:
        cx, cy = np.mean(xs), np.mean(ys)
        dx, dy = xs - cx, ys - cy
        Ixx = np.sum(dx ** 2)
        Iyy = np.sum(dy ** 2)
        Ixy = np.sum(dx * dy)
        # Eigenvalues of inertia tensor → semi-axes
        diff = Ixx - Iyy
        s = np.sqrt(diff ** 2 + 4 * Ixy ** 2)
        major = np.sqrt(2 * (Ixx + Iyy + s) / n_pixels) * px_um
        minor = np.sqrt(2 * (Ixx + Iyy - s) / n_pixels) * px_um
        if major > 0:
            aspect_ratio = major / minor if minor > 0 else float('inf')
            eccentricity = np.sqrt(1 - (minor / major) ** 2) if minor <= major else 0.0
        else:
            aspect_ratio = 1.0
            eccentricity = 0.0
    else:
        aspect_ratio = 1.0
        eccentricity = 0.0

    # Phase statistics
    phase_cell = opd_m[cell_mask] * (2 * np.pi)  # back to radians approx
    mean_phase = float(np.mean(phase_cell))
    phase_std = float(np.std(phase_cell))

    return CellMorphology(
        area_um2=area_um2,
        volume_fL=volume_fL,
        dry_mass_pg=dry_mass_pg,
        dry_mass_density_pg_um2=dry_mass_density,
        max_height_nm=max_height_nm,
        mean_height_nm=mean_height_nm,
        perimeter_um=perimeter_um,
        circularity=min(circularity, 1.0),
        aspect_ratio=aspect_ratio,
        eccentricity=eccentricity,
        mean_phase_rad=mean_phase,
        phase_std_rad=phase_std,
    )


# ═══════════════════════════════════════════════════════════════
# 7. GROWTH RATE (Time-lapse)
# ═══════════════════════════════════════════════════════════════

def compute_growth_rate(
    dry_masses_pg: List[float],
    time_points_s: List[float],
) -> Tuple[float, np.ndarray]:
    """
    Linear growth rate from time-series of dry mass.

    Returns:
        rate_pg_per_hour : slope of linear fit
        fitted           : fitted values at each time point
    """
    t = np.array(time_points_s) / 3600.0  # → hours
    m = np.array(dry_masses_pg)

    if len(t) < 2:
        return 0.0, m

    coeffs = np.polyfit(t, m, 1)
    rate = float(coeffs[0])  # pg/hour
    fitted = np.polyval(coeffs, t)

    return rate, fitted


# ═══════════════════════════════════════════════════════════════
# 8. LINE PROFILE
# ═══════════════════════════════════════════════════════════════

def extract_line_profile(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract intensity values along a line on any 2D image.

    Returns:
        distances : array of distances in pixels from start
        values    : interpolated values along the line
    """
    from scipy.ndimage import map_coordinates

    y0, x0 = start
    y1, x1 = end

    length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    t = np.linspace(0, 1, n_points)

    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)

    values = map_coordinates(image, [ys, xs], order=1, mode='nearest')
    distances = t * length

    return distances, values


# ═══════════════════════════════════════════════════════════════
# 9. PHASE STATISTICS
# ═══════════════════════════════════════════════════════════════

@dataclass
class PhaseStatistics:
    mean_rad: float
    std_rad: float
    range_rad: float
    mean_nm: float       # OPD equivalent
    std_nm: float
    range_nm: float
    histogram: Tuple[np.ndarray, np.ndarray]  # (counts, bin_edges)


def compute_phase_statistics(
    phase_unwrapped: np.ndarray,
    wavelength_m: float,
    mask: Optional[np.ndarray] = None,
    n_bins: int = 256,
) -> PhaseStatistics:
    """Phase distribution statistics (optionally within a ROI)."""
    if mask is not None:
        ph = phase_unwrapped[mask > 0]
    else:
        ph = phase_unwrapped.ravel()

    opd = ph * wavelength_m / (2 * np.pi)

    counts, edges = np.histogram(ph, bins=n_bins)

    return PhaseStatistics(
        mean_rad=float(np.mean(ph)),
        std_rad=float(np.std(ph)),
        range_rad=float(np.ptp(ph)),
        mean_nm=float(np.mean(opd)) * 1e9,
        std_nm=float(np.std(opd)) * 1e9,
        range_nm=float(np.ptp(opd)) * 1e9,
        histogram=(counts, edges),
    )


# ═══════════════════════════════════════════════════════════════
# 10. POWER SPECTRAL DENSITY (surface texture)
# ═══════════════════════════════════════════════════════════════

def compute_psd_2d(
    height_m: np.ndarray,
    pixel_size_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radially-averaged 2D Power Spectral Density of height map.
    Useful for surface texture analysis — identifies periodic features.

    Returns:
        freq : spatial frequency array (1/m)
        psd  : power spectral density (m⁴)
    """
    ny, nx = height_m.shape
    h_leveled = level_plane(height_m)

    # 2D FFT
    H = np.fft.fft2(h_leveled)
    psd_2d = (np.abs(H) ** 2) * (pixel_size_m ** 2) / (ny * nx)

    # Radial average
    fy = np.fft.fftfreq(ny, d=pixel_size_m)
    fx = np.fft.fftfreq(nx, d=pixel_size_m)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)

    max_freq = 0.5 / pixel_size_m
    n_bins = min(ny, nx) // 2
    freq_bins = np.linspace(0, max_freq, n_bins + 1)
    freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    psd_radial = np.zeros(n_bins)
    for i in range(n_bins):
        ring = (R >= freq_bins[i]) & (R < freq_bins[i + 1])
        if np.any(ring):
            psd_radial[i] = float(np.mean(psd_2d[ring]))

    return freq_centers, psd_radial


# ═══════════════════════════════════════════════════════════════
# 11. CONVENIENCE: Compute-all wrapper
# ═══════════════════════════════════════════════════════════════

@dataclass
class QPIResult:
    """All QPI outputs bundled together."""
    # Maps (2D arrays)
    opd_m: np.ndarray                             # OPD in metres
    opd_nm: np.ndarray                            # OPD in nm (display)
    height_nm: Optional[np.ndarray] = None        # Physical height (nm)
    ri_map: Optional[np.ndarray] = None           # Refractive index map
    dry_mass_density_pg_um2: Optional[np.ndarray] = None  # Mass density

    # Scalars
    phase_stats: Optional[PhaseStatistics] = None
    roughness: Optional[RoughnessResult] = None
    cell_morph: Optional[CellMorphology] = None
    total_dry_mass_pg: Optional[float] = None

    # Segmentation
    cell_mask: Optional[np.ndarray] = None

    # PSD
    psd_freq: Optional[np.ndarray] = None
    psd_values: Optional[np.ndarray] = None


class QPIMode(str, Enum):
    BIO = "biological"
    MICRO = "micro_structure"
    BOTH = "both"


def compute_qpi(
    phase_unwrapped: np.ndarray,
    wavelength_m: float,
    pixel_size_m: float,
    mode: QPIMode = QPIMode.BOTH,
    n_sample: float = 1.38,
    n_medium: float = 1.337,
    alpha: float = DEFAULT_ALPHA,
    known_thickness_m: Optional[float] = None,
    cell_threshold: float = 0.15,
    compute_psd: bool = True,
) -> QPIResult:
    """
    One-call computation of all QPI parameters.

    Parameters
    ----------
    phase_unwrapped : 2D unwrapped phase (radians)
    wavelength_m    : illumination wavelength
    pixel_size_m    : effective pixel size
    mode            : 'biological', 'micro_structure', or 'both'
    n_sample        : sample refractive index (bio: ~1.38, glass: ~1.52)
    n_medium        : medium refractive index (air: 1.0, culture: 1.337)
    alpha           : specific refractive increment for dry mass
    known_thickness_m : if set, compute RI map instead of height
    cell_threshold  : OPD threshold for cell segmentation (bio mode)
    compute_psd     : whether to compute PSD (slow for large images)
    """
    result = QPIResult(
        opd_m=np.empty(0),
        opd_nm=np.empty(0),
    )

    # OPD
    opd_m = phase_to_opd(phase_unwrapped, wavelength_m)
    result.opd_m = opd_m
    result.opd_nm = opd_m * 1e9

    # Phase stats (always)
    result.phase_stats = compute_phase_statistics(phase_unwrapped, wavelength_m)

    # Height or RI
    dn = n_sample - n_medium
    if known_thickness_m is not None:
        result.ri_map = opd_to_refractive_index(opd_m, known_thickness_m, n_medium)
    elif abs(dn) > 1e-12:
        h_m = opd_to_height(opd_m, n_sample, n_medium)
        result.height_nm = h_m * 1e9

    # Micro-structure mode
    if mode in (QPIMode.MICRO, QPIMode.BOTH):
        if result.height_nm is not None:
            h_leveled = level_plane(result.height_nm * 1e-9)
            result.roughness = compute_roughness(h_leveled)
            if compute_psd:
                result.psd_freq, result.psd_values = compute_psd_2d(
                    h_leveled, pixel_size_m
                )

    # Biological mode
    if mode in (QPIMode.BIO, QPIMode.BOTH):
        # Dry mass density (per pixel)
        density_kg_m2 = compute_dry_mass_density(opd_m, alpha)
        # Convert to pg/μm²: 1 kg/m² = 1e15 pg / 1e12 μm² = 1e3 pg/μm²
        result.dry_mass_density_pg_um2 = density_kg_m2 * 1e3

        # Cell segmentation & morphology
        try:
            mask = segment_cell_phase(opd_m, threshold_factor=cell_threshold)
            result.cell_mask = mask
            result.cell_morph = compute_cell_morphology(
                opd_m, pixel_size_m, mask, n_sample, n_medium, alpha
            )
            result.total_dry_mass_pg = result.cell_morph.dry_mass_pg
        except Exception:
            # Segmentation may fail on non-cell samples
            result.total_dry_mass_pg = compute_dry_mass_total(
                opd_m, pixel_size_m, alpha
            ) * 1e15

    return result
