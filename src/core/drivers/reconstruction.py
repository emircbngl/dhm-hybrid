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
    # Manual off-axis +1-order center (y, x) in the FFT-SHIFTED spectrum,
    # ``None`` = auto-detect via skimage.peak_local_max. Escape hatch for
    # noisy / multi-order / low-carrier holograms where auto-detection
    # fails; set from ui3 by clicking the Spectrum viewport (2026-07-08).
    offaxis_center: Optional[Tuple[int, int]] = None
    # v2.0.1: optional reference hologram to subtract from the +1
    # order. When ``None`` we're in the standard single-frame path.
    reference_path: Optional[Path] = None
    subtract_reference: bool = False
    # Phase 3 (2026-07-05, AI_VISION_MCP_PLAN.md): three-way reference
    # mode — "off" (no background handling), "reference" (division by
    # a loaded reference hologram, the ``subtract_reference`` path
    # above), "reference_free" (numerical background-phase fit via
    # ``core.background_phase.subtract_background`` — no reference
    # hologram required). ``subtract_reference`` is kept as-is for
    # back-compat (old state files, the existing checkbox widget);
    # :meth:`effective_reference_mode` is the single place that
    # reconciles the two so every consumer agrees on which mode is
    # actually active.
    reference_mode: str = "off"   # "off" | "reference" | "reference_free"
    # Reference-free background-fit knobs — mirror
    # ``core.background_phase.subtract_background``'s parameters.
    reffree_bg_method: str = "zernike"     # "zernike" | "polynomial"
    reffree_bg_order: int = 4              # polynomial_order when method="polynomial"
    reffree_n_terms: int = 15              # Zernike Noll terms when method="zernike"
    # Optional Track C CNN residual correction on top of the classical
    # background fit. Gated at the call site on torch being importable
    # AND a trained checkpoint existing — see ``ui2.workers``.
    reffree_cnn: bool = False
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
    #
    # Default settled 2026-07-06 on the 9-scene real-lab benchmark
    # (scripts/benchmark_af_real.py → tasks/af_real_benchmark.json):
    # at the same 40-eval budget "robust" hits ≤0.5 mm in 78%/67% of
    # scenes (laplacian/entropy) vs zscan's 33%/44%, and is the only
    # top performer consistent across BOTH metrics. Old persisted
    # states keep their stored/migrated value (see state_store v9).
    af_algorithm: str = "robust"
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

    def effective_reference_mode(self) -> str:
        """Resolve the single reference mode every pipeline path should use.

        ``reference_mode`` is the source of truth going forward, but
        state files (and any code path) written before Phase 3 only
        know ``subtract_reference``. Back-compat rule: if the mode is
        still at its default ("off") and the legacy flag is armed
        (``subtract_reference=True``), treat it as "reference" — the
        old checkbox behaviour. Any explicit ``reference_mode`` value
        (including an explicit "off" the user picked *after* also
        toggling the legacy checkbox — can't happen via the UI, but
        matters for hand-built ReconParams in tests/tools) wins once
        it differs from the default only via this one heuristic: we
        can't distinguish "user explicitly chose off" from "field was
        never touched" since both are the string "off". That's fine —
        it matches the pre-Phase-3 behaviour exactly (no reference_mode
        existed, so the checkbox is what decided it)."""
        mode = str(self.reference_mode or "off").lower()
        if mode == "off" and self.subtract_reference:
            return "reference"
        return mode


@dataclass
class ReconResult:
    """One successful reconstruction — three images + metadata."""
    input_image: np.ndarray       # raw hologram (H, W) float32
    amplitude: np.ndarray         # |E| reconstructed
    phase: np.ndarray             # angle(E) in radians, wrapped
    offaxis_center: tuple         # (y, x) of detected +1 order
    runtime_ms: float
    # Phase 3: populated only for reference_mode == "reference_free",
    # where we already have to unwrap + background-fit the phase to
    # do the reffree correction. ``None`` everywhere else (off/reference)
    # — consumers that want an unwrapped view fall back to ``phase``
    # (wrapped) via ``getattr(result, "unwrapped_phase", None)``.
    unwrapped_phase: Optional[np.ndarray] = None
    # Phase 3: non-fatal reffree status the UI should surface (e.g.
    # "CNN correction requested but unavailable — used classical fit
    # only"). ``None`` when there's nothing worth telling the operator.
    # Never blocks the reconstruction — always paired with a usable
    # ``unwrapped_phase``/``phase``.
    reffree_note: Optional[str] = None
    # 2026-07-08 (B-110): non-fatal warning when reference mode was armed but
    # the reference DIVISION failed and we fell back to an unreferenced field.
    # The reconstruction still succeeds (usable images) but is quantitatively
    # unreferenced — the UI must surface this, not paint a clean "success".
    reference_note: Optional[str] = None


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
        from .workers import (
            _preprocess_raw, _extract_field_with_reference, _offaxis_params,
        )
        raw = _preprocess_raw(raw, params)
        offaxis = _offaxis_params(params)
        field, center, reference_note = _extract_field_with_reference(
            raw, params, offaxis)

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

        # Phase 3: reference-free background correction. Requires an
        # unwrap step the wrapped ``phase`` above doesn't have — done
        # here (not shared with QPI's unwrap) because the Reconstruct
        # button has never unwrapped before and the QPI worker's unwrap
        # additionally halves for reflection mode, which is QPI-specific
        # physics we don't want to duplicate/entangle here.
        unwrapped_phase: Optional[np.ndarray] = None
        reffree_note: Optional[str] = None
        if params.effective_reference_mode() == "reference_free":
            from .workers import _apply_reffree_correction
            unwrapped_phase, reffree_note = _apply_reffree_correction(
                phase, amplitude, recon, p, params,
            )

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
                    "reference_mode": params.effective_reference_mode(),
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
            unwrapped_phase=unwrapped_phase,
            reffree_note=reffree_note,
            reference_note=reference_note,
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
