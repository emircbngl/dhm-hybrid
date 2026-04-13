"""
Phase-correlation based ROI tracker and ROI-focused autofocus for DHM.

Adapted from the Simulation project's phase_tracker.py.
Provides:
  - PhaseTracker: FFT phase-correlation ROI tracking with sub-pixel accuracy
  - ROI-based focus metric evaluation (compute metric only within tracked ROI)
  - ROI autofocus sweep (z-sweep using ROI metric instead of full-frame)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

from .reconstruction import ReconstructionParams, ReconstructionMethod, CachedReconstructor
from .fft_backend import get_best_fft_backend


@dataclass
class ROIBounds:
    """ROI rectangle in pixel coordinates."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    def clamp(self, img_h: int, img_w: int) -> 'ROIBounds':
        return ROIBounds(
            x0=max(0, self.x0),
            y0=max(0, self.y0),
            x1=min(img_w, self.x1),
            y1=min(img_h, self.y1),
        )


class PhaseTracker:
    """
    Phase-correlation based object tracker for DHM phase/amplitude maps.

    Uses FFT-based phase correlation with sub-pixel accuracy
    to track displacement of a selected ROI across frame updates.
    """

    def __init__(self, roi_size: int = 64):
        self.roi_size = int(roi_size)
        self.center_x = 0.0
        self.center_y = 0.0
        self.ref_roi: Optional[np.ndarray] = None
        self.active = False
        self._confidence = 0.0

    def select_roi(self, image: np.ndarray, center_x: float, center_y: float) -> bool:
        """Select a region of interest centered at (center_x, center_y) in pixel coords."""
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.ref_roi = self._extract_roi(image)
        self.active = self.ref_roi is not None
        self._confidence = 1.0 if self.active else 0.0
        return self.active

    def track(self, new_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Track the selected object in a new image.
        Returns (dx, dy, confidence). Updates internal ROI center.
        """
        if not self.active or self.ref_roi is None:
            return 0.0, 0.0, 0.0

        curr_roi = self._extract_roi(new_image)
        if curr_roi is None:
            return 0.0, 0.0, 0.0

        dx, dy, confidence = self._phase_correlate(self.ref_roi, curr_roi)
        self._confidence = confidence

        if confidence > 0.15:
            self.center_x += dx
            self.center_y += dy
            h, w = new_image.shape[:2]
            half = self.roi_size // 2
            self.center_x = float(np.clip(self.center_x, half, w - half - 1))
            self.center_y = float(np.clip(self.center_y, half, h - half - 1))
            self.ref_roi = self._extract_roi(new_image)

        return dx, dy, confidence

    def get_roi_bounds(self) -> ROIBounds:
        """Returns the current ROI rectangle in pixel coords."""
        half = self.roi_size // 2
        cx, cy = int(round(self.center_x)), int(round(self.center_y))
        return ROIBounds(cx - half, cy - half, cx + half, cy + half)

    def _extract_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract ROI patch from image around current center."""
        if image is None or image.ndim < 2:
            return None
        h, w = image.shape[:2]
        half = self.roi_size // 2
        cx, cy = int(round(self.center_x)), int(round(self.center_y))

        x0, x1 = max(0, cx - half), min(w, cx + half)
        y0, y1 = max(0, cy - half), min(h, cy + half)

        if (x1 - x0) < 4 or (y1 - y0) < 4:
            return None

        roi = image[y0:y1, x0:x1].astype(np.float64)
        if roi.shape[0] != self.roi_size or roi.shape[1] != self.roi_size:
            padded = np.zeros((self.roi_size, self.roi_size), dtype=np.float64)
            padded[:roi.shape[0], :roi.shape[1]] = roi
            roi = padded
        return roi

    def _phase_correlate(self, ref: np.ndarray, curr: np.ndarray) -> Tuple[float, float, float]:
        """
        Phase correlation with sub-pixel accuracy (Kuglin & Hines 1975).
        Returns (dx, dy, peak_value) where peak_value is confidence [0,1].
        """
        n = ref.shape[0]
        hann = np.outer(np.hanning(n), np.hanning(ref.shape[1]))
        ref_w = ref * hann
        curr_w = curr * hann

        F1 = np.fft.fft2(ref_w)
        F2 = np.fft.fft2(curr_w)

        cross = F1 * np.conj(F2)
        magnitude = np.abs(cross)
        magnitude[magnitude < 1e-12] = 1e-12
        R = cross / magnitude

        corr = np.fft.ifft2(R).real

        peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
        peak_val = corr[peak_idx]

        dy = float(peak_idx[0])
        dx = float(peak_idx[1])
        ny, nx = corr.shape
        if dy > ny // 2:
            dy -= ny
        if dx > nx // 2:
            dx -= nx

        # Sub-pixel refinement via 5×5 weighted centroid
        py, px = peak_idx
        radius = 2
        total_weight = 0.0
        sx, sy = 0.0, 0.0
        for iy in range(-radius, radius + 1):
            for ix in range(-radius, radius + 1):
                yy = (py + iy) % ny
                xx = (px + ix) % nx
                w = max(corr[yy, xx], 0.0)
                sx += ix * w
                sy += iy * w
                total_weight += w

        if total_weight > 0:
            dx += sx / total_weight
            dy += sy / total_weight

        confidence = float(np.clip(peak_val, 0.0, 1.0))
        return dx, dy, confidence

    def compute_focus_metric(self, image: np.ndarray, metric_name: str = "tenengrad") -> float:
        """
        Compute focus sharpness metric within the tracked ROI.
        Supports: tenengrad, gradient, brenner, laplacian_variance, normalized_variance
        """
        roi = self._extract_roi(image)
        if roi is None:
            return 0.0

        if metric_name == "gradient":
            gy, gx = np.gradient(roi)
            return float(np.sum(gx ** 2 + gy ** 2))
        elif metric_name == "brenner":
            dy = (roi[2:, :] - roi[:-2, :]) ** 2
            dx = (roi[:, 2:] - roi[:, :-2]) ** 2
            return float(np.sum(dy) + np.sum(dx))
        elif metric_name == "laplacian_variance":
            from scipy.ndimage import laplace
            lap = laplace(roi)
            return float(np.var(lap))
        elif metric_name == "normalized_variance":
            mu = float(np.mean(roi))
            return float(np.var(roi)) / mu if mu > 1e-15 else 0.0
        else:  # tenengrad (default)
            from scipy.ndimage import sobel
            sx = sobel(roi, axis=1, mode='reflect')
            sy = sobel(roi, axis=0, mode='reflect')
            return float(np.sum(sx ** 2 + sy ** 2))

    def reset(self):
        """Clear tracking state."""
        self.ref_roi = None
        self.active = False
        self.center_x = 0.0
        self.center_y = 0.0
        self._confidence = 0.0

    @property
    def confidence(self) -> float:
        return self._confidence


# ─────────────────────────────────────────────────────────────
# Automatic object detection
# ─────────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """A detected object with bounding box and centroid."""
    label: int               # unique id
    cx: float                # centroid x (pixels)
    cy: float                # centroid y (pixels)
    bbox: ROIBounds          # bounding box
    area: int                # area in pixels
    mean_intensity: float    # mean intensity inside object

    @property
    def equivalent_radius(self) -> float:
        return float(np.sqrt(self.area / np.pi))


def detect_objects(
    image: np.ndarray,
    min_area: int = 50,
    max_area: int = 0,
    method: str = "adaptive",
    sensitivity: float = 1.0,
    margin: int = 10,
) -> list[DetectedObject]:
    """
    Detect objects (cells, particles, etc.) in a reconstructed amplitude or phase image.

    Parameters
    ----------
    image : 2D ndarray (amplitude or phase)
    min_area : minimum object area in pixels to keep
    max_area : maximum object area (0 = no limit, auto = 25% of image)
    method : "adaptive" (local threshold), "otsu" (global), or "log" (Laplacian of Gaussian blobs)
    sensitivity : multiplier for threshold — higher = more detections
    margin : pixel margin to exclude near image edges

    Returns
    -------
    List of DetectedObject sorted by area (largest first).
    """
    from scipy import ndimage

    if image is None or image.ndim != 2:
        return []

    h, w = image.shape
    img = image.astype(np.float64, copy=True)

    # Normalize to [0, 1]
    vmin, vmax = np.percentile(img, [1, 99])
    if vmax - vmin < 1e-12:
        return []
    img = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)

    if max_area <= 0:
        max_area = int(h * w * 0.25)

    if method == "log":
        # Laplacian-of-Gaussian blob detection
        return _detect_log_blobs(img, h, w, min_area, max_area, sensitivity, margin, image)

    if method == "otsu":
        # Global Otsu threshold — detect both bright and dark outliers
        threshold = _otsu_threshold(img) * sensitivity
        med = np.median(img)
        binary = (img > (med + (threshold - med))) | (img < (med - (threshold - med)))
    else:
        # Adaptive local threshold (default)
        block_size = max(31, min(h, w) // 8) | 1  # ensure odd
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(img, size=block_size, mode='reflect')
        local_std = np.sqrt(
            np.maximum(
                uniform_filter(img ** 2, size=block_size, mode='reflect') - local_mean ** 2,
                0.0,
            )
        )
        # Pixels significantly deviating from local background (both bright AND dark)
        # Phase objects can appear as positive or negative phase shifts
        offset = 0.5 * sensitivity
        deviation = np.abs(img - local_mean)
        binary = deviation > (offset * np.maximum(local_std, 0.02))

    # Clean up with morphological operations
    struct = ndimage.generate_binary_structure(2, 1)
    binary = ndimage.binary_opening(binary, structure=struct, iterations=1)
    binary = ndimage.binary_closing(binary, structure=struct, iterations=1)

    # Exclude edges
    if margin > 0:
        binary[:margin, :] = False
        binary[-margin:, :] = False
        binary[:, :margin] = False
        binary[:, -margin:] = False

    # Connected components
    labeled, n_objects = ndimage.label(binary)
    if n_objects == 0:
        return []

    objects = []
    for i in range(1, n_objects + 1):
        mask_i = labeled == i
        area = int(np.sum(mask_i))
        if area < min_area or area > max_area:
            continue

        ys, xs = np.where(mask_i)
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        bbox = ROIBounds(
            x0=int(xs.min()), y0=int(ys.min()),
            x1=int(xs.max()) + 1, y1=int(ys.max()) + 1,
        )
        mean_int = float(np.mean(image[mask_i]))

        objects.append(DetectedObject(
            label=len(objects) + 1,
            cx=cx, cy=cy, bbox=bbox,
            area=area, mean_intensity=mean_int,
        ))

    # Sort by area descending
    objects.sort(key=lambda o: o.area, reverse=True)
    # Re-label
    for idx, obj in enumerate(objects):
        obj.label = idx + 1

    return objects


def _detect_log_blobs(
    img: np.ndarray, h: int, w: int,
    min_area: int, max_area: int,
    sensitivity: float, margin: int,
    original: np.ndarray,
) -> list[DetectedObject]:
    """Detect objects via Laplacian-of-Gaussian multi-scale blob detection."""
    from scipy.ndimage import gaussian_laplace

    min_r = max(3, int(np.sqrt(min_area / np.pi)))
    max_r = min(min(h, w) // 4, int(np.sqrt(max_area / np.pi)))

    # Multi-scale LoG
    sigmas = np.linspace(min_r / np.sqrt(2), max_r / np.sqrt(2), 10)
    best_response = np.zeros_like(img)
    best_sigma = np.zeros_like(img)

    for s in sigmas:
        log_resp = -gaussian_laplace(img, sigma=s) * s ** 2
        mask = log_resp > best_response
        best_response[mask] = log_resp[mask]
        best_sigma[mask] = s

    # Threshold on response strength
    thresh = np.percentile(best_response[best_response > 0], max(10, 50 - sensitivity * 20)) if np.any(best_response > 0) else 0
    peaks = best_response > thresh

    if margin > 0:
        peaks[:margin, :] = False
        peaks[-margin:, :] = False
        peaks[:, :margin] = False
        peaks[:, -margin:] = False

    from scipy.ndimage import label as nd_label, maximum_filter
    # Non-maximum suppression
    local_max = maximum_filter(best_response, size=max(5, min_r))
    peaks = peaks & (best_response == local_max)

    labeled, n = nd_label(peaks)
    objects = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        s = float(best_sigma[int(round(cy)), int(round(cx))])
        r = int(round(s * np.sqrt(2)))
        area = int(np.pi * r * r)
        if area < min_area or area > max_area:
            continue
        bbox = ROIBounds(
            x0=max(0, int(cx) - r), y0=max(0, int(cy) - r),
            x1=min(w, int(cx) + r), y1=min(h, int(cy) + r),
        )
        mean_int = float(np.mean(original[bbox.y0:bbox.y1, bbox.x0:bbox.x1]))
        objects.append(DetectedObject(
            label=len(objects) + 1,
            cx=cx, cy=cy, bbox=bbox,
            area=area, mean_intensity=mean_int,
        ))

    objects.sort(key=lambda o: o.area, reverse=True)
    for idx, obj in enumerate(objects):
        obj.label = idx + 1
    return objects


def _otsu_threshold(img: np.ndarray) -> float:
    """Simple Otsu threshold for normalized [0,1] image."""
    nbins = 256
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 0.5

    w0 = np.cumsum(hist).astype(np.float64)
    w1 = total - w0
    mu0 = np.cumsum(hist * bin_centers) / np.maximum(w0, 1e-10)
    mu1 = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / np.maximum(w1, 1e-10)

    variance = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.argmax(variance))
    return float(bin_centers[idx])


# ─────────────────────────────────────────────────────────────
# ROI-based autofocus evaluator
# ─────────────────────────────────────────────────────────────

def make_roi_evaluator(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric_name: str,
    roi_bounds: ROIBounds,
):
    """
    Build a fast evaluate(z) closure that computes focus metric
    only within the specified ROI — much faster than full-frame
    and focuses the autofocus on the region of interest.
    """
    from .autofocus import FocusMetric, _calc_metric, _is_phase_metric

    fft = get_best_fft_backend()
    recon = CachedReconstructor(field.shape, method, fft)

    inp = field.astype(np.complex64, copy=False)
    field_spectrum = fft.fft2(inp)
    if field_spectrum.dtype != np.complex64:
        field_spectrum = field_spectrum.astype(np.complex64)

    # Map string metric name to FocusMetric enum
    metric_map = {fm.value: fm for fm in FocusMetric}
    fm = metric_map.get(metric_name, FocusMetric.TENENGRAD)
    phase_metric = _is_phase_metric(fm)

    # Pre-compute clamped ROI bounds
    ny, nx = field.shape
    rb = roi_bounds.clamp(ny, nx)

    def evaluate(z: float) -> float:
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=z, n=base_params.n,
        )
        result = recon.reconstruct_from_spectrum(field_spectrum, params)
        roi = result[rb.y0:rb.y1, rb.x0:rb.x1]
        if phase_metric:
            return _calc_metric(roi, fm)
        return _calc_metric(np.abs(roi), fm)

    return evaluate


def roi_autofocus_sweep(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric_name: str,
    roi_bounds: ROIBounds,
    z_min_m: float,
    z_max_m: float,
    n_steps: int = 51,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Sweep z-positions and find the best focus using ROI-based metric.

    Returns (best_z_m, z_array, score_array).
    """
    from .autofocus import FocusMetric, _is_minimize
    metric_map = {fm.value: fm for fm in FocusMetric}
    fm = metric_map.get(metric_name, FocusMetric.TENENGRAD)
    minimize = _is_minimize(fm)

    evaluate = make_roi_evaluator(field, base_params, method, metric_name, roi_bounds)

    z_values = np.linspace(z_min_m, z_max_m, n_steps)
    scores = np.empty(n_steps, dtype=np.float64)

    best_z = z_values[0]
    best_score = float('inf') if minimize else -float('inf')

    for i, z in enumerate(z_values):
        if cancel_check and cancel_check():
            from .autofocus import AutofocusCancelled
            raise AutofocusCancelled()
        if on_progress:
            on_progress(i, n_steps)
        score = evaluate(z)
        scores[i] = score
        if (minimize and score < best_score) or (not minimize and score > best_score):
            best_score = score
            best_z = z

    return float(best_z), z_values, scores


def roi_autofocus_coarse_fine(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric_name: str,
    roi_bounds: ROIBounds,
    z_min_m: float,
    z_max_m: float,
    coarse_steps: int = 21,
    fine_steps: int = 21,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Two-stage ROI autofocus: coarse sweep → fine sweep around best.
    Returns (best_z_m, z_array, score_array).
    """
    from .autofocus import FocusMetric, _is_minimize
    metric_map = {fm.value: fm for fm in FocusMetric}
    fm = metric_map.get(metric_name, FocusMetric.TENENGRAD)
    minimize = _is_minimize(fm)

    evaluate = make_roi_evaluator(field, base_params, method, metric_name, roi_bounds)
    total = coarse_steps + fine_steps
    evals = 0

    def is_better(new, old):
        return new < old if minimize else new > old

    # Phase 1: Coarse
    z_coarse = np.linspace(z_min_m, z_max_m, coarse_steps)
    s_coarse = np.empty(coarse_steps, dtype=np.float64)
    best_z = z_coarse[0]
    best_score = float('inf') if minimize else -float('inf')

    for i, z in enumerate(z_coarse):
        if cancel_check and cancel_check():
            from .autofocus import AutofocusCancelled
            raise AutofocusCancelled()
        evals += 1
        if on_progress:
            on_progress(evals, total)
        score = evaluate(z)
        s_coarse[i] = score
        if is_better(score, best_score):
            best_score = score
            best_z = z

    # Phase 2: Fine sweep around best
    step = (z_max_m - z_min_m) / max(1, coarse_steps - 1)
    margin = step * 1.5
    fine_min = max(z_min_m, best_z - margin)
    fine_max = min(z_max_m, best_z + margin)
    z_fine = np.linspace(fine_min, fine_max, fine_steps)
    s_fine = np.empty(fine_steps, dtype=np.float64)

    for i, z in enumerate(z_fine):
        if cancel_check and cancel_check():
            from .autofocus import AutofocusCancelled
            raise AutofocusCancelled()
        evals += 1
        if on_progress:
            on_progress(evals, total)
        score = evaluate(z)
        s_fine[i] = score
        if is_better(score, best_score):
            best_score = score
            best_z = z

    all_z = np.concatenate([z_coarse, z_fine])
    all_s = np.concatenate([s_coarse, s_fine])
    return float(best_z), all_z, all_s
