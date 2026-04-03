from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from .fft_backend import FFTBackend, get_best_fft_backend


class ReconstructionMethod(str, Enum):
    ASM = "asm"
    FRESNEL = "fresnel"


@dataclass(frozen=True)
class ReconstructionParams:
    wavelength_m: float
    pixel_size_m: float
    z_m: float
    n: float = 1.0


class CachedReconstructor:
    """
    Stateful Python reconstructor for maximum throughput.

    Caches:
      - Frequency grids (FX², FY², sqrt_term) — recomputed only when shape/pixel/wl/n change
      - Transfer function H — recomputed only when z changes
      - Field spectrum FFT(field) — recomputed only when the field data changes

    All internal arrays use complex64 / float32 to halve memory and double throughput.
    """
    def __init__(self, shape: tuple[int, int], method: ReconstructionMethod, fft: FFTBackend):
        self.shape = shape
        self.method = method
        self.fft = fft

        # H (transfer function) cache
        self._cached_H: Optional[np.ndarray] = None
        self._cached_z_m: Optional[float] = None
        self._cached_wavelength: Optional[float] = None
        self._cached_pixel_scale: Optional[float] = None
        self._cached_n: Optional[float] = None

        # Frequency grid cache (depends on shape + pixel + wl + n)
        self._freq_key: Optional[tuple] = None
        self._k: Optional[float] = None
        self._sqrt_term: Optional[np.ndarray] = None   # ASM
        self._freq_sq: Optional[np.ndarray] = None      # Fresnel

        # Field spectrum cache
        self._field_spectrum: Optional[np.ndarray] = None
        self._field_id: Optional[int] = None  # id() of cached field

    # ── frequency grid (reused across z values) ──
    def _ensure_freq_grid(self, params: ReconstructionParams) -> None:
        key = (params.wavelength_m, params.pixel_size_m, params.n)
        if self._freq_key == key:
            return
        ny, nx = self.shape
        dy = dx = np.float32(params.pixel_size_m)
        fy = np.fft.fftfreq(ny, d=dy).astype(np.float32)
        fx = np.fft.fftfreq(nx, d=dx).astype(np.float32)
        FX, FY = np.meshgrid(fx, fy)

        self._k = np.float32(2.0 * np.pi * params.n / params.wavelength_m)

        if self.method == ReconstructionMethod.ASM:
            wl_n = np.float32(params.wavelength_m / params.n)
            term = np.float32(1.0) - (wl_n * FX)**2 - (wl_n * FY)**2
            np.maximum(term, 0.0, out=term)
            np.sqrt(term, out=term)
            self._sqrt_term = term
        else:
            self._freq_sq = FX**2 + FY**2

        self._freq_key = key

    # ── transfer function H for a given z ──
    def _compute_H(self, params: ReconstructionParams) -> np.ndarray:
        self._ensure_freq_grid(params)
        z = np.float32(params.z_m)
        if self.method == ReconstructionMethod.ASM:
            phase = (1j * self._k * z * self._sqrt_term).astype(np.complex64)
            H = np.exp(phase)
            # Clamp evanescent waves: where sqrt_term == 0, H should decay
            # For negative z, evanescent components would grow exponentially
            # Enforce |H| <= 1 to prevent NaN/Inf from evanescent amplification
            H_abs = np.abs(H)
            evanescent_mask = H_abs > 1.0
            if np.any(evanescent_mask):
                H[evanescent_mask] = H[evanescent_mask] / H_abs[evanescent_mask]
        else:
            phasor = np.complex64(np.exp(1j * self._k * z))
            quad = np.float32(-np.pi * params.wavelength_m * params.z_m / params.n)
            H = phasor * np.exp((1j * quad * self._freq_sq).astype(np.complex64))
        # Guard against any residual NaN/Inf in transfer function
        if not np.all(np.isfinite(H)):
            H = np.nan_to_num(H, nan=0.0, posinf=1.0, neginf=-1.0)
        return H

    def _get_H(self, params: ReconstructionParams) -> np.ndarray:
        cache_key = (params.z_m, params.wavelength_m, params.pixel_size_m, params.n)
        current_key = (self._cached_z_m, self._cached_wavelength,
                       self._cached_pixel_scale, self._cached_n)
        if self._cached_H is None or current_key != cache_key:
            self._cached_H = self._compute_H(params)
            self._cached_z_m = params.z_m
            self._cached_wavelength = params.wavelength_m
            self._cached_pixel_scale = params.pixel_size_m
            self._cached_n = params.n
        return self._cached_H

    # ── field spectrum cache ──
    def _get_field_spectrum(self, field: np.ndarray) -> np.ndarray:
        fid = id(field)
        if self._field_spectrum is None or self._field_id != fid:
            inp = field.astype(np.complex64, copy=False)
            self._field_spectrum = self.fft.fft2(inp)
            if self._field_spectrum.dtype != np.complex64:
                self._field_spectrum = self._field_spectrum.astype(np.complex64)
            self._field_id = fid
        return self._field_spectrum

    # ── public API ──
    def reconstruct(self, field: np.ndarray, params: ReconstructionParams) -> np.ndarray:
        H = self._get_H(params)
        spectrum = self._get_field_spectrum(field).copy()  # copy so we don't mutate cache
        spectrum *= H
        result = self.fft.ifft2(spectrum)
        return _sanitize_output(result)

    def reconstruct_from_spectrum(self, field_spectrum: np.ndarray,
                                  params: ReconstructionParams) -> np.ndarray:
        """Reconstruct using a pre-computed field spectrum (skips FFT of field)."""
        H = self._get_H(params)
        buf = field_spectrum * H
        result = self.fft.ifft2(buf)
        return _sanitize_output(result)


def _sanitize_output(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in reconstruction output with zero."""
    if not np.all(np.isfinite(arr)):
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

# Global cache for the static propagate call (when not explicitly instantiating classes)
_GLOBAL_RECON_CACHE: Optional[CachedReconstructor] = None


def propagate(
    field: Any,
    params: ReconstructionParams,
    method: ReconstructionMethod,
    fft: Optional[FFTBackend] = None,
    force_python: bool = True
) -> Any:
    # Use python implementation for fast pipeline
    if force_python:
        global _GLOBAL_RECON_CACHE
        if fft is None:
            fft = get_best_fft_backend()
            
        if (_GLOBAL_RECON_CACHE is None
                or _GLOBAL_RECON_CACHE.shape != field.shape
                or _GLOBAL_RECON_CACHE.method != method
                or _GLOBAL_RECON_CACHE.fft.name != fft.name):
            _GLOBAL_RECON_CACHE = CachedReconstructor(field.shape, method, fft)
        
        return _GLOBAL_RECON_CACHE.reconstruct(field, params)

    raise RuntimeError("Python force failed. Fallback backend removed.")

# Map deprecated individual calls to the wrapper
def propagate_asm(field: Any, params: ReconstructionParams, fft: Optional[FFTBackend] = None) -> Any:
    return propagate(field, params, ReconstructionMethod.ASM, fft)

def propagate_fresnel(field: Any, params: ReconstructionParams, fft: Optional[FFTBackend] = None) -> Any:
    return propagate(field, params, ReconstructionMethod.FRESNEL, fft)
