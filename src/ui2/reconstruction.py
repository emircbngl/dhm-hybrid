"""Thin driver wrapping the existing ``core`` pipeline for the ui2 app.

Dear PyGui is single-threaded at the render layer, so we push heavy
work to ``concurrent.futures.ThreadPoolExecutor`` and post results
back via a lock-protected ``latest`` dict the UI polls in its
render loop. No Qt signals, no QThread — pipeline stays Qt-free.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from core.ingestion import load_any
from core.masking import detect_plus_one_order
from core.offaxis import OffAxisParams, extract_complex_field_offaxis
from core.reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)

_LOG = logging.getLogger(__name__)


@dataclass
class ReconParams:
    """User-editable reconstruction parameters.

    v2.0.3 re-adds the scientific parameters the v1 PySide frontend
    exposed but the v2 port silently dropped. Each field has a direct
    v1 counterpart in ``src/gui/sidebar/*_tab.py``; defaults match v1
    so existing users who don't touch the new widgets get the same
    behaviour they always had."""
    wavelength_nm: float = 632.8
    pixel_um: float = 5.0
    z_mm: float = 10.0
    mask_radius: int = 40
    method: str = "ASM"   # "ASM" or "Fresnel"
    # v2.0.1: optional reference hologram to subtract from the +1
    # order. When ``None`` we're in the standard single-frame path.
    reference_path: Optional[Path] = None
    subtract_reference: bool = False
    # v2.0.3: objective magnification. In DHM the effective pixel at
    # the *sample* plane is ``camera_pixel / M`` — that's what the
    # ASM/Fresnel kernel wants for pixel_size_m. v1 shipped with this
    # field + a "pixel is already effective" checkbox; the v2 port
    # accidentally dropped both, so users on microscope setups got
    # z-propagation off by M² until now.
    magnification: float = 1.0
    pixel_is_effective: bool = True
    # QPI refractive indices — v1's qpi_tab exposes these; v2.0.2
    # hardcoded them in workers.py, so dry-mass / cell-height
    # calculations were stuck on water+typical-cell even when the
    # operator's medium was different.
    n_sample: float = 1.38
    n_medium: float = 1.337
    # Autofocus metric — v1 exposes 11 metrics via a combo; v2.0.2
    # was stuck on LAPLACIAN_VARIANCE, which fails on low-contrast
    # bio samples where gradient metrics are the standard choice.
    autofocus_metric: str = "LAPLACIAN_VARIANCE"
    # v2.0.4: autofocus scan range + step count — v1 exposes these on
    # focus_tab; v2.0.2 hardcoded -25/+25 mm with 40 steps, which is
    # wrong for both fast DHM (small-z) and long-range tomography.
    af_z_min_mm: float = -25.0
    af_z_max_mm: float = 25.0
    af_n_steps: int = 40
    # v2.0.8: autofocus search algorithm. v1 shipped six algorithms
    # on focus_tab; v2 only ran the fixed-step "zscan" linear sweep.
    # Adaptive variants spend the step budget smarter — big steps in
    # flat regions, small steps near the peak — so user can reach
    # finer z resolution with the same ``af_n_steps`` budget.
    # Options: "zscan" (linear sweep, v2.0.2 default),
    # "coarse_to_fine" (Golden-Section refinement of coarse pass),
    # "robust" (coarse scan + Gaussian-smoothed fine), or
    # "adaptive_gradient" (gradient-driven adaptive step size),
    # "adaptive_bracketing" (nested bracketing with refinement).
    af_algorithm: str = "zscan"
    # v2.0.5: advanced pre/post-processing.
    # ``subtract_mean`` removes DC bias from the hologram before the
    # off-axis mask — reduces the 0th-order bleed into the +1 order.
    # ``hann_window`` applies a 2D Hann envelope to the raw frame —
    # suppresses edge discontinuities that leak into the FFT.
    # ``fft_backend`` picks the engine used by propagate; "auto"
    # delegates to ``core.fft_backend.get_best_fft_backend``.
    # ``unwrap_method`` drives ``core.phase_unwrap`` for QPI runs.
    subtract_mean: bool = True
    hann_window: bool = False
    fft_backend: str = "auto"       # "auto" | "pyfftw" | "mlx" | "scipy" | "numpy"
    unwrap_method: str = "GRADIENT_INTEGRATION"
    # v2.0.7: optical path — "transmission" (single pass) or
    # "reflection" (double pass, OPD halves before height
    # conversion). Picking the wrong one was the "3 µm object
    # measures 6 µm" bug reported on 2026-04-24.
    optical_mode: str = "transmission"
    # Percentile-based contrast stretch on the amplitude preview.
    # Compensates for reference-division dynamic-range collapse —
    # display only, scientific values untouched.
    auto_contrast_amplitude: bool = True
    # v2.0.7: autofocus ROI — ``None`` = whole frame (current
    # behaviour), otherwise a normalised (y0, x0, y1, x1) tuple in
    # [0, 1] coordinates. Mask multiplies the demodulated field so
    # the metric only "sees" the ROI region.
    af_roi: Optional[Tuple[float, float, float, float]] = None

    def effective_pixel_um(self) -> float:
        """Camera vs. effective pixel size at the sample plane.

        When ``pixel_is_effective`` is True the user already did the
        division (legacy behaviour + single-frame / direct-imaging
        setups where M = 1). Otherwise we divide by magnification here
        so the physics layer never sees a camera-pixel value."""
        if self.pixel_is_effective or self.magnification <= 0:
            return float(self.pixel_um)
        return float(self.pixel_um) / float(self.magnification)


@dataclass
class ReconResult:
    """One successful reconstruction — three images + metadata."""
    input_image: np.ndarray       # raw hologram (H, W) float32
    amplitude: np.ndarray         # |E| reconstructed
    phase: np.ndarray             # angle(E) in radians, wrapped
    offaxis_center: tuple         # (y, x) of detected +1 order
    runtime_ms: float


@dataclass
class ReconError:
    message: str


class ReconstructionDriver:
    """Serial runner — one reconstruction in flight at a time.

    Callers push a job via :meth:`submit`. When the worker finishes,
    either ``on_result(ReconResult)`` or ``on_error(ReconError)`` is
    invoked from the **worker thread** — the UI layer must marshal
    back to the main thread via its own polling loop.

    v2.0.4 adds ``cancel()``: a single propagate + off-axis demodulation
    can't be interrupted mid-call (no scan loop to poll), but once the
    user hits Esc the driver marks the job cancelled and discards the
    result in the dispatcher so the UI paints "Cancelled" cleanly.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="recon",
        )
        self._inflight: Optional[Future] = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

    # ---- public API -----------------------------------------------------

    def submit(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        sample_id: str = "",
        on_result: Callable[[ReconResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        with self._lock:
            if self._inflight is not None and not self._inflight.done():
                on_error(ReconError("A reconstruction is already running."))
                return
            self._cancel_event.clear()
            self._inflight = self._executor.submit(
                self._run, hologram_path, params, sample_id,
            )
            self._inflight.add_done_callback(
                lambda fut: self._dispatch(fut, on_result, on_error),
            )

    def cancel(self) -> bool:
        with self._lock:
            if self._inflight is not None and not self._inflight.done():
                self._cancel_event.set()
                return True
        return False

    def shutdown(self) -> None:
        self._cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _run(path: Path, params: ReconParams,
             sample_id: str = "") -> ReconResult:
        import time
        t0 = time.monotonic()

        loaded = load_any(path)
        raw = np.asarray(loaded.array, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        if raw.size == 0:
            raise RuntimeError("Loaded hologram is empty.")
        # v2.0.9: preprocessing + reference subtract now live in a
        # shared helper (``ui2.workers``) so every pipeline path
        # behaves identically. Before this refactor only the
        # Reconstruct button ran reference-division; autofocus/
        # QPI/depth silently ignored the toggle.
        from .workers import _preprocess_raw, _extract_field_with_reference
        raw = _preprocess_raw(raw, params)
        offaxis = OffAxisParams(radius=int(params.mask_radius))
        field, center = _extract_field_with_reference(raw, params, offaxis)

        method = (ReconstructionMethod.ASM
                  if params.method.upper() == "ASM"
                  else ReconstructionMethod.FRESNEL)
        # ``effective_pixel_um()`` handles the camera_pixel / M division
        # so the kernel always gets pixel-at-sample-plane.
        p = ReconstructionParams(
            wavelength_m=float(params.wavelength_nm) * 1e-9,
            pixel_size_m=float(params.effective_pixel_um()) * 1e-6,
            z_m=float(params.z_mm) * 1e-3,
            # v2.0.8: propagate in the actual medium's index, not air.
            n=float(params.n_medium),
        )
        recon = propagate(field, p, method)
        amplitude = np.abs(recon).astype(np.float32)
        phase = np.angle(recon).astype(np.float32)
        runtime_ms = (time.monotonic() - t0) * 1000.0

        # Audit record — best-effort, imported lazily so the driver
        # stays importable in headless environments where ~/ isn't
        # writable (CI containers, tests without monkeypatching).
        try:
            from core.audit import get_audit_log
            get_audit_log().record(
                action="reconstruction",
                params={
                    "hologram": str(path),
                    "sample_id": sample_id,
                    "wavelength_nm": float(params.wavelength_nm),
                    "pixel_um": float(params.pixel_um),
                    "magnification": float(params.magnification),
                    "pixel_is_effective": bool(params.pixel_is_effective),
                    "effective_pixel_um": float(params.effective_pixel_um()),
                    "z_mm": float(params.z_mm),
                    "mask_radius": int(params.mask_radius),
                    "method": str(params.method),
                    "reference_path": (str(params.reference_path)
                                       if params.reference_path else ""),
                    "subtract_reference": bool(params.subtract_reference),
                },
                result_summary={
                    "offaxis_center": [int(v) for v in center],
                    "runtime_ms": runtime_ms,
                    "phase_range_rad": [float(phase.min()), float(phase.max())],
                },
            )
        except Exception:
            _LOG.warning("audit write for reconstruction failed",
                         exc_info=True)

        return ReconResult(
            input_image=raw,
            amplitude=amplitude,
            phase=phase,
            offaxis_center=tuple(int(v) for v in center),
            runtime_ms=runtime_ms,
        )

    def _dispatch(
        self,
        fut: Future,
        on_result: Callable[[ReconResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        cancelled = self._cancel_event.is_set()
        try:
            result = fut.result()
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("recon failed")
            on_error(ReconError(str(exc) or repr(exc)))
            return
        if cancelled:
            # Result landed after Esc — surface cancel instead of OK.
            on_error(ReconError("Cancelled."))
            return
        on_result(result)


__all__ = [
    "ReconParams", "ReconResult", "ReconError", "ReconstructionDriver",
]
