"""Reconstruction worker — phased, cancellable, progress-instrumented.

v1.0.1 rewrite. The v1.0.0 body was a monolithic ``_process`` with no
progress emission and no interruption checkpoints, which made "is it
frozen or working?" and "cancel this run" both architecturally impossible.

This version:
    * Breaks ``_process`` into seven named phases via
      :class:`core.progress.Operation`.
    * Checks ``isInterruptionRequested()`` at each phase boundary so
      ``QThread.requestInterruption()`` actually stops work within ~one
      phase (<200 ms on typical hardware).
    * Emits a ``progress`` signal carrying :class:`ProgressEvent` that
      the UI can render as a quiet progress line.
    * Emits a ``error_event`` signal carrying :class:`ErrorEvent` with
      structured cause/action/context. The legacy ``error_occurred(str)``
      signal is kept for backward compatibility — its cause is the raw
      ``str(exc)``.

The signal compatibility matters because the UI binds lambdas to
``error_occurred`` in several places; we migrate callsites separately
rather than racing signal changes with UI refactors.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from core.errors import ErrorEvent, Severity
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.phase_unwrap import UnwrapConfig, unwrap_phase_advanced
from core.progress import Operation, OperationCancelled, ProgressEvent
from core.reconstruction import ReconstructionMethod, ReconstructionParams, propagate

log = logging.getLogger(__name__)


_PHASES = (
    "preprocess",
    "offaxis",
    "freqfilter",
    "propagate",
    "refsub",
    "finitecheck",
    "unwrap",
)


class ReconstructionWorker(QThread):
    """Dedicated thread for reconstruction.

    Maintains a single-frame "latest-wins" queue so a fast camera does not
    lag the worker: if a new job arrives while the previous is queued, the
    previous is discarded.
    """

    # Back-compat: v1.0.0 shape, str message. UI still uses this in places.
    error_occurred = Signal(str)
    # New: structured ErrorEvent (core.errors).
    error_event = Signal(object)
    # New: ProgressEvent from core.progress.
    progress = Signal(object)
    # Result payload — unchanged.
    recon_completed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self._mutex = QMutex()
        self._next_job = None

    # ---- public API --------------------------------------------------------

    def submit_job(self, params: dict):
        """Queue a job. Latest-wins semantics: unclaimed previous job drops."""
        with QMutexLocker(self._mutex):
            self._next_job = params

    def stop(self):
        """Request shutdown. Honours cooperative cancellation — the current
        job's next phase boundary observes the interruption and exits."""
        self.running = False
        self.requestInterruption()

    # ---- Qt thread entry point --------------------------------------------

    def run(self):
        self.running = True
        while self.running and not self.isInterruptionRequested():
            job = None
            with QMutexLocker(self._mutex):
                if self._next_job is not None:
                    job = self._next_job
                    self._next_job = None

            if job is None:
                time.sleep(0.005)
                continue

            try:
                result = self._process(job)
            except OperationCancelled:
                # Cancel is not an error. Progress event already flagged it.
                log.info("[RECON] cancelled cooperatively")
                continue
            except Exception as exc:  # noqa: BLE001
                self._emit_error(exc, job)
                continue

            if result is not None:
                self.recon_completed.emit(result)

    # ---- pipeline ----------------------------------------------------------

    def _process(self, job: dict) -> Optional[dict]:
        """Run the seven-phase reconstruction. Returns the result dict, or
        None if interruption was observed before the final phase."""
        t_total = time.perf_counter()
        op = Operation(kind="recon", phases=_PHASES)
        op.on_progress = self.progress.emit

        log.info("[RECON] Start — shape=%s z=%.4f mm method=%s",
                 job["image"].shape, job["z"] * 1e3, job["method"])

        # --- phase 1: preprocess -------------------------------------------
        with op.phase("preprocess"):
            self._check_cancel()
            img = self._preprocess(job)

        # --- phase 2: off-axis extraction ----------------------------------
        with op.phase("offaxis"):
            self._check_cancel()
            fc, spec = self._offaxis(img, job)

        # --- phase 3: optional frequency filter ----------------------------
        with op.phase("freqfilter"):
            self._check_cancel()
            if job.get("filter_enable", False):
                fc = self._apply_freq_filter(
                    fc,
                    filter_type=job.get("filter_type", "lowpass"),
                    cutoff=job.get("filter_cutoff", 0.15),
                    rolloff=job.get("filter_rolloff", 0.02),
                )

        # --- phase 4: angular spectrum propagation -------------------------
        with op.phase("propagate"):
            self._check_cancel()
            recon_complex = self._propagate(fc, job)

        # --- phase 5: reference subtraction --------------------------------
        with op.phase("refsub"):
            self._check_cancel()
            recon_complex_raw = recon_complex.copy()
            recon_complex = self._subtract_reference(recon_complex, job)

        # --- phase 6: finite-value guard -----------------------------------
        with op.phase("finitecheck"):
            self._check_cancel()
            recon_complex = self._sanitize_finite(recon_complex)

        # --- phase 7: phase unwrap -----------------------------------------
        with op.phase("unwrap"):
            self._check_cancel()
            phase_unwrapped = self._unwrap(recon_complex, job)

        op.finish(message=f"done in {time.perf_counter() - t_total:.2f}s")

        return {
            "frame_num": job["frame_num"],
            "recon_complex": recon_complex,
            "recon_complex_raw": recon_complex_raw,
            "spectrum_mag": spec,
            "phase_unwrapped": phase_unwrapped,
        }

    # ---- phase implementations --------------------------------------------

    def _preprocess(self, job: dict) -> np.ndarray:
        t0 = time.perf_counter()
        img = job["image"]
        if job.get("subtract_mean", False):
            img = img - float(np.mean(img))
        if job.get("hann_window", False):
            wy = np.hanning(img.shape[0]).astype(np.float32)
            wx = np.hanning(img.shape[1]).astype(np.float32)
            img = img * (wy[:, None] * wx[None, :])
        peak = float(np.max(np.abs(img)))
        if peak > 0:
            img = img / peak
        log.info("[RECON]  preprocess: %.3fs", time.perf_counter() - t0)
        return img

    def _offaxis(self, img: np.ndarray, job: dict):
        t0 = time.perf_counter()
        offaxis_params = OffAxisParams(
            radius=job.get("mask_radius", 80),
            apodization=job.get("mask_apodization", "tukey"),
            rolloff=job.get("mask_rolloff", 0.25),
        )
        fc, _center, spec, _mask = extract_complex_field_offaxis_debug(
            img, offaxis_params,
        )
        log.info("[RECON]  offaxis:    %.3fs", time.perf_counter() - t0)
        return fc, spec

    def _propagate(self, fc: np.ndarray, job: dict) -> np.ndarray:
        t0 = time.perf_counter()
        recon_params = ReconstructionParams(
            wavelength_m=job["wavelength"],
            pixel_size_m=job["pixel_size"],
            z_m=job["z"],
            n=job.get("n_medium", 1.0),
        )
        fft_name = job.get("fft_backend")
        fft = None
        if fft_name:
            from core.fft_backend import FFTBackendName, get_best_fft_backend
            try:
                fft = get_best_fft_backend(prefer=FFTBackendName(fft_name))
            except Exception as exc:
                log.warning("FFT backend '%s' unavailable, using default: %s",
                            fft_name, exc)
        out = np.asarray(propagate(fc, recon_params, job["method"], fft=fft))
        log.info("[RECON]  propagate:  %.3fs", time.perf_counter() - t0)
        return out

    def _subtract_reference(self, recon_complex: np.ndarray,
                            job: dict) -> np.ndarray:
        t0 = time.perf_counter()
        ref_complex = job.get("reference_complex")
        if ref_complex is None:
            return recon_complex
        from core.reconstruction import safe_reference_divide
        out = safe_reference_divide(recon_complex, ref_complex)
        log.info("[RECON]  ref_sub:    %.3fs", time.perf_counter() - t0)
        return out

    def _sanitize_finite(self, recon_complex: np.ndarray) -> np.ndarray:
        if np.all(np.isfinite(recon_complex)):
            return recon_complex
        n_bad = int(np.sum(~np.isfinite(recon_complex)))
        log.warning("Reconstruction contains %d non-finite values — sanitizing",
                    n_bad)
        return np.nan_to_num(recon_complex, nan=0.0, posinf=0.0, neginf=0.0)

    def _unwrap(self, recon_complex: np.ndarray, job: dict) -> np.ndarray:
        t0 = time.perf_counter()
        wrapped = np.angle(recon_complex)
        if not np.all(np.isfinite(wrapped)):
            log.warning("Wrapped phase has non-finite values — skipping unwrap")
            return np.nan_to_num(wrapped, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            unwrap_cfg = job.get("unwrap_config", None)
            log.info("[RECON]  unwrap method: %s",
                     unwrap_cfg.method if unwrap_cfg else "default")
            phase = unwrap_phase_advanced(
                wrapped,
                config=unwrap_cfg,
                complex_field=recon_complex,
                wavelength_m=job.get("wavelength"),
                pixel_size_m=job.get("pixel_size"),
            )
            if not np.all(np.isfinite(phase)):
                n_bad = int(np.sum(~np.isfinite(phase)))
                log.warning(
                    "Phase unwrap produced %d non-finite values — "
                    "falling back to wrapped",
                    n_bad,
                )
                return wrapped.copy()
            log.info("[RECON]  unwrap:     %.3fs", time.perf_counter() - t0)
            return phase
        except Exception as exc:  # noqa: BLE001 — unwrap is best-effort
            log.warning("Phase unwrap failed (%s) — using wrapped phase", exc)
            return wrapped

    # ---- cancellation & errors --------------------------------------------

    def _check_cancel(self) -> None:
        if self.isInterruptionRequested() or not self.running:
            raise OperationCancelled()

    def _emit_error(self, exc: BaseException, job: dict) -> None:
        """Emit both the structured event (new) and the legacy string (old)."""
        log.exception("Reconstruction failed")
        context = {
            "z_mm": float(job.get("z", 0.0)) * 1e3,
            "wavelength_nm": float(job.get("wavelength", 0.0)) * 1e9,
            "pixel_um": float(job.get("pixel_size", 0.0)) * 1e6,
            "method": str(job.get("method", "")),
            "frame_num": job.get("frame_num"),
        }
        event = ErrorEvent.from_exception(
            exc,
            title="Reconstruction failed",
            action=_suggest_action(exc, job),
            context=context,
            severity=Severity.ERROR,
        )
        self.error_event.emit(event)
        # Legacy signal — plain string for the old status-bar lambdas.
        self.error_occurred.emit(str(exc))

    # ---- optional frequency filter ----------------------------------------

    @staticmethod
    def _apply_freq_filter(field: np.ndarray, filter_type: str,
                           cutoff: float, rolloff: float) -> np.ndarray:
        """Smooth low- or high-pass with a cosine (Tukey-style) rolloff."""
        ny, nx = field.shape
        fy = np.fft.fftfreq(ny)
        fx = np.fft.fftfreq(nx)
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX ** 2 + FY ** 2)

        half_rolloff = max(rolloff, 1e-8)
        if filter_type == "highpass":
            mask = np.where(
                R < cutoff - half_rolloff, 0.0,
                np.where(
                    R > cutoff + half_rolloff, 1.0,
                    0.5 * (1.0 + np.cos(np.pi * (R - cutoff - half_rolloff)
                                        / (2 * half_rolloff))),
                ),
            )
        else:  # lowpass
            mask = np.where(
                R < cutoff - half_rolloff, 1.0,
                np.where(
                    R > cutoff + half_rolloff, 0.0,
                    0.5 * (1.0 + np.cos(np.pi * (R - cutoff + half_rolloff)
                                        / (2 * half_rolloff))),
                ),
            )

        spectrum = np.fft.fft2(field)
        spectrum *= mask.astype(spectrum.dtype)
        return np.fft.ifft2(spectrum)


# ---------------------------------------------------------------------------
# Error → action copy: one sentence, honest, user-correctable where possible.
# Rams #4 — good design makes a product understandable.
# ---------------------------------------------------------------------------

def _suggest_action(exc: BaseException, job: dict) -> str:
    """Return a one-sentence action hint, or '' if none is obvious."""
    msg = str(exc).lower()
    if "wavelength" in msg or job.get("wavelength", 0) <= 0:
        return "Set a positive wavelength in the Parameters panel."
    if "pixel" in msg or job.get("pixel_size", 0) <= 0:
        return "Set a positive pixel size in the Parameters panel."
    if "mask" in msg and "radius" in msg:
        return "Increase the off-axis mask radius."
    if "memory" in msg or isinstance(exc, MemoryError):
        return "Close other applications or use a smaller image crop."
    if "fft" in msg:
        return "Try a different FFT backend in Preferences."
    return ""
