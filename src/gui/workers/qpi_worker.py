"""QPI worker — phased, cancellable, progress-instrumented.

v1.0.1 rewrite. The v1.0.0 body was a single ``run()`` with no progress,
no cancellation, and a lossy string error. This version decomposes into
four phases and emits the same :class:`ProgressEvent` / :class:`ErrorEvent`
shapes as :mod:`gui.workers.reconstruction_worker`, so the UI can bind to
one protocol instead of three bespoke ones.
"""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.errors import ErrorEvent, Severity
from core.progress import Operation, OperationCancelled

log = logging.getLogger(__name__)


_PHASES = ("validate", "refsub", "bgcorrect", "compute")


class QPIWorker(QThread):
    """Background thread for QPI computation — keeps the UI responsive."""

    # Legacy signals (retained for v1.0.0 call-sites)
    qpi_completed = Signal(object)   # QPIResult
    qpi_failed = Signal(str)
    # New structured signals
    error_event = Signal(object)     # ErrorEvent
    progress = Signal(object)        # ProgressEvent

    def __init__(self):
        super().__init__()
        self._job: dict | None = None

    # ---- public API --------------------------------------------------------

    def setup(self, job: dict) -> None:
        """Configure the job. See v1.0.0 docstring for expected keys."""
        self._job = job

    def cancel(self) -> None:
        """Request cooperative cancellation. The running phase observes it
        on its next checkpoint and raises OperationCancelled."""
        self.requestInterruption()

    # ---- Qt entry point ----------------------------------------------------

    def run(self) -> None:
        job = self._job
        if job is None:
            self._emit_error(
                ValueError("No QPI job configured"),
                job=None,
                title="QPI failed",
                action="Reload the hologram and retry.",
            )
            return

        try:
            result = self._process(job)
        except OperationCancelled:
            log.info("[QPI] cancelled cooperatively")
            return
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc, job=job)
            return

        if result is not None:
            self.qpi_completed.emit(result)

    # ---- pipeline ----------------------------------------------------------

    def _process(self, job: dict):
        op = Operation(kind="qpi", phases=_PHASES)
        op.on_progress = self.progress.emit

        # Lazy imports — avoid paying init cost unless a QPI actually runs.
        from core.phase_unwrap import unwrap_phase_advanced
        from core.qpi import (
            compute_qpi,
            correct_background_phase,
            subtract_reference_wave,
        )

        # --- phase 1: validate / sanitize input ----------------------------
        with op.phase("validate"):
            self._check_cancel()
            phase_input = self._validate_phase(job["phase_unwrapped"])

        # --- phase 2: optional reference-wave subtraction ------------------
        with op.phase("refsub"):
            self._check_cancel()
            if job.get("ref_subtract") and job.get("reference_complex") is not None:
                recon = job.get("recon_complex_raw") or job.get("recon_complex")
                if recon is not None:
                    try:
                        corrected = subtract_reference_wave(
                            recon, job["reference_complex"],
                        )
                        phase_input = np.angle(corrected)
                        phase_input = unwrap_phase_advanced(
                            phase_input,
                            complex_field=corrected,
                            wavelength_m=job["wavelength_m"],
                            pixel_size_m=job["pixel_size_m"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("[QPI] reference subtraction failed: %s", exc)

        # --- phase 3: optional background polynomial correction ------------
        with op.phase("bgcorrect"):
            self._check_cancel()
            if job.get("bg_correction") and not job.get(
                    "unwrap_bg_already_removed", False):
                try:
                    phase_input = correct_background_phase(
                        phase_input,
                        order=job.get("bg_poly_order", 2),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("[QPI] background correction failed: %s", exc)

        # --- phase 4: QPI computation --------------------------------------
        with op.phase("compute"):
            self._check_cancel()
            result = compute_qpi(
                phase_unwrapped=phase_input,
                wavelength_m=job["wavelength_m"],
                pixel_size_m=job["pixel_size_m"],
                mode=job["mode"],
                n_sample=job["n_sample"],
                n_medium=job["n_medium"],
                alpha=job["alpha_SI"],
                known_thickness_m=job.get("known_thickness_m"),
                cell_threshold=job["cell_threshold"],
                compute_psd=job["compute_psd"],
            )

        op.finish()
        return result

    # ---- helpers -----------------------------------------------------------

    def _validate_phase(self, raw: np.ndarray) -> np.ndarray:
        phase = raw.copy()
        if not np.all(np.isfinite(phase)):
            n_bad = int(np.sum(~np.isfinite(phase)))
            log.warning("[QPI] sanitizing %d non-finite values in phase input",
                        n_bad)
            phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
        return phase

    def _check_cancel(self) -> None:
        if self.isInterruptionRequested():
            raise OperationCancelled()

    def _emit_error(self, exc: BaseException, *,
                    job: dict | None,
                    title: str = "QPI failed",
                    action: str = "") -> None:
        log.exception("[QPI] worker failed")
        context: dict = {}
        if job is not None:
            context = {
                "wavelength_nm": float(job.get("wavelength_m", 0.0)) * 1e9,
                "pixel_um": float(job.get("pixel_size_m", 0.0)) * 1e6,
                "mode": str(job.get("mode", "")),
                "n_sample": float(job.get("n_sample", 0.0)),
                "n_medium": float(job.get("n_medium", 0.0)),
            }
        event = ErrorEvent.from_exception(
            exc,
            title=title,
            action=action or _suggest_action(exc, job or {}),
            context=context,
            severity=Severity.ERROR,
        )
        self.error_event.emit(event)
        self.qpi_failed.emit(f"{type(exc).__name__}: {exc}")


def _suggest_action(exc: BaseException, job: dict) -> str:
    """One-sentence, user-correctable hint — Rams #4."""
    msg = str(exc).lower()
    if isinstance(exc, MemoryError) or "memory" in msg:
        return "Close other applications or use a smaller image crop."
    if "n_sample" in msg or job.get("n_sample", 0) <= 0:
        return "Set a positive sample refractive index in the QPI panel."
    if "n_medium" in msg or job.get("n_medium", 0) <= 0:
        return "Set a positive medium refractive index in the QPI panel."
    if "phase" in msg and "finite" in msg:
        return "Re-run reconstruction; the phase input contains non-finite values."
    return ""
