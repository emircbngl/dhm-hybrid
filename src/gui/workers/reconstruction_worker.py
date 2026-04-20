import logging
import numpy as np
import time
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.phase_unwrap import unwrap_phase_advanced, UnwrapConfig

log = logging.getLogger(__name__)

class ReconstructionWorker(QThread):
    """
    Dedicated thread for reconstruction. 
    It maintains a single-frame "latest-wins" queue to avoid lagging behind camera.
    """
    # Emits dict with keys: 'frame_num', 'recon_complex', 'spectrum_mag', 'phase_unwrapped'
    recon_completed = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self._mutex = QMutex()
        self._next_job = None

    def submit_job(self, params: dict):
        """
        params format:
        {
            'frame_num': int,
            'image': np.ndarray,
            'method': ReconstructionMethod,
            'wavelength': float,
            'pixel_size': float,
            'z': float,
            'n_medium': float,
            'subtract_mean': bool,
            'hann_window': bool,
            'mask_radius': int
        }
        """
        with QMutexLocker(self._mutex):
            self._next_job = params

    def run(self):
        self.running = True
        while self.running:
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
                self.recon_completed.emit(result)
            except Exception as e:
                log.exception("Reconstruction failed")
                self.error_occurred.emit(str(e))

    def _process(self, job: dict) -> dict:
        t_total = time.perf_counter()
        img = job['image']
        log.info("[RECON] Start — image %s, z=%.4f mm, method=%s",
                 img.shape, job['z']*1e3, job['method'])
        
        # Preprocessing
        t0 = time.perf_counter()
        if job.get('subtract_mean', False):
            img = img - float(np.mean(img))
        if job.get('hann_window', False):
            wy = np.hanning(img.shape[0]).astype(np.float32)
            wx = np.hanning(img.shape[1]).astype(np.float32)
            img = img * (wy[:, None] * wx[None, :])
            
        if np.max(np.abs(img)) > 0:
            img = img / float(np.max(np.abs(img)))
        log.info("[RECON]  preprocess: %.3fs", time.perf_counter() - t0)

        # 1. Offaxis extraction (with soft-edge spectral mask)
        t0 = time.perf_counter()
        offaxis_params = OffAxisParams(
            radius=job.get('mask_radius', 80),
            apodization=job.get('mask_apodization', 'tukey'),
            rolloff=job.get('mask_rolloff', 0.25),
        )
        fc, center, spec, mask = extract_complex_field_offaxis_debug(img, offaxis_params)
        log.info("[RECON]  offaxis:    %.3fs", time.perf_counter() - t0)
        
        # 1.5. Frequency domain filtering (if enabled)
        if job.get('filter_enable', False):
            fc = self._apply_freq_filter(
                fc,
                filter_type=job.get('filter_type', 'lowpass'),
                cutoff=job.get('filter_cutoff', 0.15),
                rolloff=job.get('filter_rolloff', 0.02),
            )

        # 2. Propagation
        t0 = time.perf_counter()
        recon_params = ReconstructionParams(
            wavelength_m=job['wavelength'],
            pixel_size_m=job['pixel_size'],
            z_m=job['z'],
            n=job.get('n_medium', 1.0)
        )
        
        # FFT override
        fft_name = job.get('fft_backend')
        fft = None
        if fft_name:
            from core.fft_backend import get_best_fft_backend, FFTBackendName
            try:
                fft = get_best_fft_backend(prefer=FFTBackendName(fft_name))
            except Exception as e:
                log.warning("FFT backend '%s' unavailable, using default: %s", fft_name, e)
                
        recon_complex = np.asarray(propagate(fc, recon_params, job['method'], fft=fft))
        log.info("[RECON]  propagate:  %.3fs", time.perf_counter() - t0)

        # Store raw (pre-reference) complex field for QPI use
        t0 = time.perf_counter()
        recon_complex_raw = recon_complex.copy()

        # Reference hologram subtraction (complex division)
        ref_complex = job.get('reference_complex')
        if ref_complex is not None:
            ref_abs = np.abs(ref_complex)
            safe_ref = np.where(ref_abs > 1e-10, ref_complex,
                                np.ones_like(ref_complex))
            recon_complex = recon_complex / safe_ref

        log.info("[RECON]  ref_sub:    %.3fs", time.perf_counter() - t0)

        # Validate reconstruction output
        if not np.all(np.isfinite(recon_complex)):
            n_bad = int(np.sum(~np.isfinite(recon_complex)))
            log.warning("Reconstruction contains %d non-finite values — sanitizing", n_bad)
            recon_complex = np.nan_to_num(recon_complex, nan=0.0, posinf=0.0, neginf=0.0)

        phase_unwrapped = None
        t0 = time.perf_counter()
        try:
            wrapped = np.angle(recon_complex)
            if np.all(np.isfinite(wrapped)):
                unwrap_cfg = job.get('unwrap_config', None)
                log.info("[RECON]  unwrap method: %s", unwrap_cfg.method if unwrap_cfg else 'default')
                phase_unwrapped = unwrap_phase_advanced(
                    wrapped, config=unwrap_cfg, complex_field=recon_complex,
                    wavelength_m=job.get('wavelength'),
                    pixel_size_m=job.get('pixel_size'),
                )
                # Validate unwrapped phase
                if not np.all(np.isfinite(phase_unwrapped)):
                    n_bad = int(np.sum(~np.isfinite(phase_unwrapped)))
                    log.warning("Phase unwrapping produced %d non-finite values — falling back to wrapped phase", n_bad)
                    phase_unwrapped = wrapped.copy()
            else:
                log.warning("Wrapped phase contains non-finite values — skipping unwrap")
                phase_unwrapped = np.nan_to_num(wrapped, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            log.warning("Phase unwrapping failed (%s) — using wrapped phase", e)
            phase_unwrapped = np.angle(recon_complex)
        log.info("[RECON]  unwrap:     %.3fs", time.perf_counter() - t0)
        log.info("[RECON]  TOTAL:      %.3fs", time.perf_counter() - t_total)

        return {
            'frame_num': job['frame_num'],
            'recon_complex': recon_complex,
            'recon_complex_raw': recon_complex_raw,
            'spectrum_mag': spec,
            'phase_unwrapped': phase_unwrapped
        }

    @staticmethod
    def _apply_freq_filter(field: np.ndarray, filter_type: str, cutoff: float, rolloff: float) -> np.ndarray:
        """Apply a frequency domain low-pass or high-pass filter with smooth Tukey rolloff."""
        ny, nx = field.shape
        fy = np.fft.fftfreq(ny)
        fx = np.fft.fftfreq(nx)
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)  # normalized frequency radius [0, 0.5]

        # Build smooth transition mask using a cosine (Tukey-style) rolloff
        half_rolloff = max(rolloff, 1e-8)
        if filter_type == 'highpass':
            mask = np.where(
                R < cutoff - half_rolloff, 0.0,
                np.where(
                    R > cutoff + half_rolloff, 1.0,
                    0.5 * (1.0 + np.cos(np.pi * (R - cutoff - half_rolloff) / (2 * half_rolloff)))
                )
            )
        else:  # lowpass
            mask = np.where(
                R < cutoff - half_rolloff, 1.0,
                np.where(
                    R > cutoff + half_rolloff, 0.0,
                    0.5 * (1.0 + np.cos(np.pi * (R - cutoff + half_rolloff) / (2 * half_rolloff)))
                )
            )

        spectrum = np.fft.fft2(field)
        spectrum *= mask.astype(spectrum.dtype)
        return np.fft.ifft2(spectrum)

    def stop(self):
        self.running = False
