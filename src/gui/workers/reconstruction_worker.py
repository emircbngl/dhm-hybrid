import numpy as np
import time
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.phase_unwrap import unwrap_phase_advanced, UnwrapConfig

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
        print("ReconWorker thread running")
        while self.running:
            job = None
            with QMutexLocker(self._mutex):
                if self._next_job is not None:
                    job = self._next_job
                    self._next_job = None

            if job is None:
                time.sleep(0.005)
                continue

            print("ReconWorker found job, starting _process")
            try:
                result = self._process(job)
                print("ReconWorker finished _process, emitting completed")
                self.recon_completed.emit(result)
            except Exception as e:
                import traceback
                print("EXCEPTION IN RECON WORKER:")
                traceback.print_exc()
                self.error_occurred.emit(str(e))

    def _process(self, job: dict) -> dict:
        img = job['image']
        
        # Preprocessing
        if job.get('subtract_mean', False):
            img = img - float(np.mean(img))
        if job.get('hann_window', False):
            wy = np.hanning(img.shape[0]).astype(np.float32)
            wx = np.hanning(img.shape[1]).astype(np.float32)
            img = img * (wy[:, None] * wx[None, :])
            
        if np.max(np.abs(img)) > 0:
            img = img / float(np.max(np.abs(img)))

        # 1. Offaxis extraction (with soft-edge spectral mask)
        offaxis_params = OffAxisParams(
            radius=job.get('mask_radius', 80),
            apodization=job.get('mask_apodization', 'tukey'),
            rolloff=job.get('mask_rolloff', 0.25),
        )
        fc, center, spec, mask = extract_complex_field_offaxis_debug(img, offaxis_params)
        
        # 1.5. Frequency domain filtering (if enabled)
        if job.get('filter_enable', False):
            fc = self._apply_freq_filter(
                fc,
                filter_type=job.get('filter_type', 'lowpass'),
                cutoff=job.get('filter_cutoff', 0.15),
                rolloff=job.get('filter_rolloff', 0.02),
            )

        # 2. Propagation
        print("ReconWorker: Starting Propagation")
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
            except Exception:
                pass
                
        print("ReconWorker: Calling propagate")
        recon_complex = np.asarray(propagate(fc, recon_params, job['method'], fft=fft))
        print("ReconWorker: Propagate finished")

        # Store raw (pre-reference) complex field for QPI use
        recon_complex_raw = recon_complex.copy()

        # Reference hologram subtraction (complex division)
        ref_complex = job.get('reference_complex')
        if ref_complex is not None:
            print("ReconWorker: Applying reference subtraction")
            ref_abs = np.abs(ref_complex)
            safe_ref = np.where(ref_abs > 1e-10, ref_complex,
                                np.ones_like(ref_complex))
            recon_complex = recon_complex / safe_ref

        # Validate reconstruction output
        if not np.all(np.isfinite(recon_complex)):
            n_bad = int(np.sum(~np.isfinite(recon_complex)))
            print(f"WARNING: Reconstruction contains {n_bad} non-finite values — sanitizing")
            recon_complex = np.nan_to_num(recon_complex, nan=0.0, posinf=0.0, neginf=0.0)

        phase_unwrapped = None
        try:
            wrapped = np.angle(recon_complex)
            if np.all(np.isfinite(wrapped)):
                unwrap_cfg = job.get('unwrap_config', None)
                phase_unwrapped = unwrap_phase_advanced(
                    wrapped, config=unwrap_cfg, complex_field=recon_complex,
                    wavelength_m=job.get('wavelength'),
                    pixel_size_m=job.get('pixel_size'),
                )
                # Validate unwrapped phase
                if not np.all(np.isfinite(phase_unwrapped)):
                    n_bad = int(np.sum(~np.isfinite(phase_unwrapped)))
                    print(f"WARNING: Phase unwrapping produced {n_bad} non-finite values — falling back to wrapped phase")
                    phase_unwrapped = wrapped.copy()
            else:
                print("WARNING: Wrapped phase contains non-finite values — skipping unwrap")
                phase_unwrapped = np.nan_to_num(wrapped, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"WARNING: Phase unwrapping failed ({e}) — using wrapped phase")
            phase_unwrapped = np.angle(recon_complex)

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
