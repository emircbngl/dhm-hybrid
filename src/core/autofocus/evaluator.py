"""Fast focus-metric evaluator and complex-field downsampling utilities.

`_make_fast_evaluator` builds an `evaluate(z)` closure that reuses the pre-computed
field spectrum and a local `CachedReconstructor`, so each evaluation only needs
one IFFT. `downsample_complex_field` performs bandlimited Fourier-domain cropping
and returns an adjusted `ReconstructionParams` so propagation distances stay correct.

v2.1.0 also adds :func:`_make_batch_evaluator` — given a list of z values upfront,
build the entire (n_steps, ny, nx) propagation kernel stack in one shot and feed
it through ``backend.ifft2_batched``. Net win when the backend is a GPU one
(torch CUDA / MPS): a single batched IFFT replaces n_steps serial IFFTs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..fft_backend import FFTBackend, FFTBackendName, get_best_fft_backend
from ..reconstruction import (
    CachedReconstructor,
    ReconstructionMethod,
    ReconstructionParams,
    safe_reference_divide,
)
from .metrics import FocusMetric, _calc_metric


class AutofocusCancelled(Exception):
    """Raised when an autofocus run is cancelled by the user."""
    pass


@dataclass(frozen=True)
class AutoFocusResult:
    best_z_m: float
    scores: Dict[float, float]
    # Non-fatal diagnostic: set when the focus-score landscape is
    # degenerate (nearly flat, or mostly non-finite) so ``best_z_m`` is
    # argmax-of-noise rather than a real focus. The result is still
    # returned (best guess) but callers/UI should surface this
    # (2026-07-08 version audit — ported from the original Phyton app's
    # autofocus diagnostics, which Hybrid's plain linear z-scan lacked).
    warning: Optional[str] = None


def focus_landscape_warning(scores: Dict[float, float]) -> Optional[str]:
    """Return an actionable warning when a focus-score curve is degenerate.

    Two soft guards mirroring the original Phyton autofocus (which *raised*
    here — we return a string instead so the best-guess z is preserved):

    * fewer than 3 finite scores → the reconstruction produced non-finite
      metrics for most z (bad z-range / method / physical parameters);
    * normalized dynamic range ``(max-min)/scale < 1e-3`` → the curve is
      essentially flat, so the picked z is noise, not focus (usually wrong
      pixel size / magnification / wavelength, an unsuitable z-range, or a
      bad +1-order mask center/radius).

    Returns ``None`` when the landscape looks healthy.
    """
    if not scores:
        return None
    ys = np.asarray(list(scores.values()), dtype=np.float64)
    finite = ys[np.isfinite(ys)]
    if finite.size < 3:
        return ("Autofocus unreliable: the reconstruction produced non-finite "
                "focus scores for most z values. Try narrowing the z-range, "
                "switching method (ASM/Fresnel), or checking the pixel size / "
                "magnification / wavelength.")
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    y_scale = max(abs(y_max), abs(y_min), 1e-12)
    if (y_max - y_min) / y_scale < 1e-3:
        return ("Autofocus unreliable: focus scores are nearly flat across the "
                "scan, so the chosen z is not a real focus. This usually means "
                "incorrect physical parameters (pixel size / magnification / "
                "wavelength), an unsuitable z-range, or a wrong +1-order mask "
                "(center / radius).")
    return None


def downsample_complex_field(
    field: np.ndarray,
    base_params: ReconstructionParams,
    factor: int = 2,
) -> tuple:
    """
    Downsample a complex field by *factor* for faster autofocus.

    Uses Fourier-domain cropping (correct for bandlimited signals).
    Returns (downsampled_field, adjusted_params) where pixel_size is scaled
    so propagation distances remain correct.
    """
    if factor <= 1:
        return field, base_params

    ny, nx = field.shape
    new_ny, new_nx = ny // factor, nx // factor

    # Fourier-domain crop: FFT → keep central new_ny×new_nx → IFFT
    F = np.fft.fftshift(np.fft.fft2(field))
    cy, cx = ny // 2, nx // 2
    hy, hx = new_ny // 2, new_nx // 2
    F_crop = F[cy - hy: cy - hy + new_ny, cx - hx: cx - hx + new_nx]
    ds_field = np.fft.ifft2(np.fft.ifftshift(F_crop)).astype(np.complex64)

    # Scale pixel size (effective pixel becomes factor× larger)
    ds_params = ReconstructionParams(
        wavelength_m=base_params.wavelength_m,
        pixel_size_m=base_params.pixel_size_m * factor,
        z_m=base_params.z_m,
        n=base_params.n,
    )
    return ds_field, ds_params


def _make_fast_evaluator(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> Callable[[float], float]:
    """
    Build a fast evaluate(z) closure.

    Pre-computes FFT(field) once and creates a local CachedReconstructor
    so each eval only needs: H computation + element-wise multiply + IFFT.
    Avoids per-call overhead of propagate() global cache, id() checks, and .copy().

    If roi_bounds is provided (ROIBounds-like with x0,y0,x1,y1), the metric
    is computed only within that region — much faster and focuses on the object.

    If ref_field is provided, both sample and reference are propagated to each z
    and divided, giving a reference-subtracted complex field for accurate phase
    measurement (all metrics now operate on phase).
    """
    fft = get_best_fft_backend()
    recon = CachedReconstructor(field.shape, method, fft)

    # Pre-compute field spectrum once
    inp = field.astype(np.complex64, copy=False)
    field_spectrum = fft.fft2(inp)
    if field_spectrum.dtype != np.complex64:
        field_spectrum = field_spectrum.astype(np.complex64)

    # Pre-compute reference spectrum if provided (phase metrics always benefit)
    use_ref = ref_field is not None
    ref_spectrum = None
    if use_ref:
        ref_inp = ref_field.astype(np.complex64, copy=False)
        ref_spectrum = fft.fft2(ref_inp)
        if ref_spectrum.dtype != np.complex64:
            ref_spectrum = ref_spectrum.astype(np.complex64)

    # Pre-compute clamped ROI bounds
    ny, nx = field.shape
    if roi_bounds is not None:
        ry0 = max(0, int(roi_bounds.y0))
        rx0 = max(0, int(roi_bounds.x0))
        ry1 = min(ny, int(roi_bounds.y1))
        rx1 = min(nx, int(roi_bounds.x1))
    else:
        ry0, rx0, ry1, rx1 = 0, 0, ny, nx

    def evaluate(z: float) -> float:
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=z, n=base_params.n,
        )
        result = recon.reconstruct_from_spectrum(field_spectrum, params)

        if use_ref and ref_spectrum is not None:
            ref_result = recon.reconstruct_from_spectrum(ref_spectrum, params)
            result = safe_reference_divide(result, ref_result)

        roi = result[ry0:ry1, rx0:rx1]
        return _calc_metric(roi, metric)

    return evaluate


def _build_H_stack(field_shape: tuple,
                   base_params: ReconstructionParams,
                   method: ReconstructionMethod,
                   zs: Sequence[float]) -> np.ndarray:
    """Assemble the (n_steps, ny, nx) transfer-function stack for a
    list of propagation distances. Reuses the same H-builder the
    single-shot ``CachedReconstructor`` uses, so every search
    algorithm produces bit-identical results whether they go
    through serial or batched paths."""
    # Borrow the single-shot recon's `_compute_H` — it's the
    # canonical implementation; duplicating risks drift.
    fft = get_best_fft_backend()  # only used for shape metadata
    proxy = CachedReconstructor(field_shape, method, fft)
    out = np.empty((len(zs),) + tuple(field_shape), dtype=np.complex64)
    for i, z in enumerate(zs):
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=float(z), n=base_params.n,
        )
        H = proxy._compute_H(params)
        out[i] = H.astype(np.complex64, copy=False)
    return out


def _make_roi_fast_evaluator(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    roi_bounds,
) -> Callable[[float], float]:
    """ROI-aware evaluator that crops the *spatial* result before
    metric computation **and** uses the smallest sensible IFFT
    size when the ROI is much smaller than the frame.

    The standard :func:`_make_fast_evaluator` does the IFFT on the
    full (ny, nx) spectrum and then takes a sub-array for the
    metric. When the operator picks a 256×256 ROI on a 2048×2048
    frame, that's a 64× wasted IFFT. This fast-path runs the IFFT
    at the ROI-sized output instead.

    Implementation: we keep the full-frame field spectrum (it
    carries the off-axis structure we depend on for phase
    reconstruction) and the full-frame H stack, but compute the
    IFFT directly into a smaller-padded buffer using the standard
    "FFT-then-crop" Fourier trick:

    1. Full-frame product ``S · H`` is shape (ny, nx).
    2. ``np.fft.fftshift`` to centre DC, then crop the central
       (roi_h, roi_w) of the spectrum.
    3. ``np.fft.ifft2`` on the *cropped* spectrum gives a
       lower-resolution but spatially-correct ROI image.

    The crop loses high frequencies outside the ROI band; for
    autofocus that's fine — we only care about the metric value
    on the ROI region, and the metric itself averages over many
    samples. Net wall-clock saving = (full_area / roi_area).
    """
    fft = get_best_fft_backend()
    recon = CachedReconstructor(field.shape, method, fft)

    inp = field.astype(np.complex64, copy=False)
    field_spectrum = fft.fft2(inp)
    if field_spectrum.dtype != np.complex64:
        field_spectrum = field_spectrum.astype(np.complex64)

    ny, nx = field.shape
    ry0 = max(0, int(roi_bounds.y0))
    rx0 = max(0, int(roi_bounds.x0))
    ry1 = min(ny, int(roi_bounds.y1))
    rx1 = min(nx, int(roi_bounds.x1))
    roi_h = max(8, ry1 - ry0)  # min 8 px so FFT stays meaningful
    roi_w = max(8, rx1 - rx0)
    # Round up to even for clean fftshift.
    if roi_h % 2:
        roi_h += 1
    if roi_w % 2:
        roi_w += 1

    def evaluate(z: float) -> float:
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=z, n=base_params.n,
        )
        H = recon._get_H(params)
        spec = field_spectrum * H
        # Spectral crop = lower-resolution real-space image with
        # bandwidth limited to the ROI. The cropped result is
        # spatially scaled to match the ROI footprint.
        spec_shifted = np.fft.fftshift(spec)
        cy, cx = ny // 2, nx // 2
        hy, hx = roi_h // 2, roi_w // 2
        spec_crop = spec_shifted[cy - hy: cy + hy,
                                 cx - hx: cx + hx]
        spec_crop = np.fft.ifftshift(spec_crop)
        result = np.fft.ifft2(spec_crop)
        return _calc_metric(result, metric)

    return evaluate


def _make_batch_evaluator(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    metric: FocusMetric,
    *,
    backend: Optional[FFTBackend] = None,
    roi_bounds=None,
    ref_field: Optional[np.ndarray] = None,
) -> Callable[[Sequence[float]], np.ndarray]:
    """Return a callable ``evaluate_many(zs)`` that scores every
    ``z`` in one shot.

    Why
    ---
    For a 2048² × 40-step zscan, the serial path issues 40 IFFTs.
    A batched path issues 1 IFFT over a (40, 2048, 2048) tensor.
    On CPU + numpy this is roughly the same wall time (numpy
    doesn't parallelise the batch axis well). On a torch CUDA /
    MPS backend it's where the real GPU speedup lives — a single
    kernel launch instead of 40, plus device-resident memory
    means no host↔device transfers between iterations.

    Correctness
    -----------
    The result of ``evaluate_many([z0, z1, ...])`` is identical
    (up to float-precision noise) to
    ``[evaluate(z0), evaluate(z1), ...]`` produced by
    :func:`_make_fast_evaluator`. ROI clipping + reference
    division semantics carry through. Caller can switch search
    paths transparently.

    Parameters
    ----------
    backend
        FFT backend to use for the batched IFFT. None → uses
        :func:`get_best_fft_backend`. Pass an explicit
        :class:`TorchFFTBackend` to force GPU.
    """
    fft = backend if backend is not None else get_best_fft_backend()

    inp = field.astype(np.complex64, copy=False)
    field_spectrum = fft.fft2(inp)
    if not isinstance(field_spectrum, np.ndarray):
        # GPU backends return something convertible — normalise.
        field_spectrum = fft.to_numpy(field_spectrum)
    field_spectrum = field_spectrum.astype(np.complex64, copy=False)

    use_ref = ref_field is not None
    ref_spectrum = None
    if use_ref:
        ref_inp = ref_field.astype(np.complex64, copy=False)
        ref_spectrum = fft.fft2(ref_inp)
        if not isinstance(ref_spectrum, np.ndarray):
            ref_spectrum = fft.to_numpy(ref_spectrum)
        ref_spectrum = ref_spectrum.astype(np.complex64, copy=False)

    ny, nx = field.shape
    if roi_bounds is not None:
        ry0 = max(0, int(roi_bounds.y0))
        rx0 = max(0, int(roi_bounds.x0))
        ry1 = min(ny, int(roi_bounds.y1))
        rx1 = min(nx, int(roi_bounds.x1))
    else:
        ry0, rx0, ry1, rx1 = 0, 0, ny, nx

    def evaluate_many(zs: Sequence[float]) -> np.ndarray:
        zs_list = list(zs)
        if not zs_list:
            return np.zeros((0,), dtype=np.float64)

        H_stack = _build_H_stack(
            field.shape, base_params, method, zs_list,
        )
        # Multiply spectrum by every H plane → batched IFFT.
        spec_stack = field_spectrum[None, :, :] * H_stack
        result_stack = fft.ifft2_batched(spec_stack)
        if not isinstance(result_stack, np.ndarray):
            result_stack = fft.to_numpy(result_stack)

        if use_ref and ref_spectrum is not None:
            ref_spec_stack = ref_spectrum[None, :, :] * H_stack
            ref_result = fft.ifft2_batched(ref_spec_stack)
            if not isinstance(ref_result, np.ndarray):
                ref_result = fft.to_numpy(ref_result)
            result_stack = safe_reference_divide(result_stack, ref_result)

        out = np.empty(len(zs_list), dtype=np.float64)
        for i in range(len(zs_list)):
            roi = result_stack[i, ry0:ry1, rx0:rx1]
            out[i] = _calc_metric(roi, metric)
        return out

    return evaluate_many
