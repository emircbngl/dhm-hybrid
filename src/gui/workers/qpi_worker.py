import numpy as np
from PySide6.QtCore import QThread, Signal


class QPIWorker(QThread):
    """Background thread for QPI computation — keeps UI responsive."""

    qpi_completed = Signal(object)  # QPIResult
    qpi_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._job = None

    def setup(self, job: dict):
        """
        Prepare a QPI job.

        job keys:
            phase_unwrapped : np.ndarray
            recon_complex   : np.ndarray (optional, for ref subtraction)
            reference_complex : np.ndarray or None
            wavelength_m    : float
            pixel_size_m    : float
            mode            : QPIMode
            n_sample        : float
            n_medium        : float
            alpha_SI        : float
            known_thickness_m : float or None
            cell_threshold  : float
            compute_psd     : bool
            bg_correction   : bool
            bg_poly_order   : int
            ref_subtract    : bool
        """
        self._job = job

    def run(self):
        job = self._job
        if job is None:
            self.qpi_failed.emit("No QPI job configured")
            return

        try:
            from core.qpi import compute_qpi, correct_background_phase, subtract_reference_wave
            from core.phase_unwrap import unwrap_phase_advanced

            phase_input = job["phase_unwrapped"].copy()

            # Validate input
            if not np.all(np.isfinite(phase_input)):
                n_bad = int(np.sum(~np.isfinite(phase_input)))
                print(f"QPI: sanitizing {n_bad} non-finite values in phase input")
                phase_input = np.nan_to_num(phase_input, nan=0.0, posinf=0.0, neginf=0.0)

            # Reference wave subtraction (if enabled)
            if job.get("ref_subtract") and job.get("reference_complex") is not None:
                recon = job.get("recon_complex")
                if recon is not None:
                    try:
                        corrected = subtract_reference_wave(recon, job["reference_complex"])
                        phase_input = np.angle(corrected)
                        phase_input = unwrap_phase_advanced(
                            phase_input,
                            complex_field=corrected,
                            wavelength_m=job["wavelength_m"],
                            pixel_size_m=job["pixel_size_m"],
                        )
                    except Exception as e:
                        print(f"WARNING: Reference subtraction failed: {e}")

            # Background phase correction (once, after ref subtraction)
            if job.get("bg_correction"):
                try:
                    order = job.get("bg_poly_order", 2)
                    phase_input = correct_background_phase(phase_input, order=order)
                except Exception as e:
                    print(f"WARNING: Background correction failed: {e}")

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

            self.qpi_completed.emit(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.qpi_failed.emit(f"{type(e).__name__}: {e}")
