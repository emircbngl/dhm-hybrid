from PySide6.QtCore import QThread, Signal
import time
from pathlib import Path
import os
import sys
import csv
import numpy as np
import tifffile
from core.phase_unwrap import unwrap_phase_advanced
from core.ingestion import load_any
from core.reconstruction import propagate, ReconstructionParams, ReconstructionMethod
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.autofocus import FocusMetric, _calc_metric, _is_minimize

def _save_normalized(path: Path, arr: np.ndarray, is_phase: bool = False):
    if is_phase:
        norm = (arr + np.pi) / (2 * np.pi)
    else:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            norm = np.zeros_like(arr)
        else:
            norm = (arr - mn) / (mx - mn)
    
    out_arr = (np.clip(norm, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(path, out_arr)

class BatchJob:
    def __init__(self, in_file: Path, out_file: Path, params: dict):
        self.in_file = in_file
        self.out_file = out_file
        self.params = params

class BatchRenderer(QThread):
    progress = Signal(int, int) # current, maximum
    eta_update = Signal(str)
    status = Signal(str)
    finished_batch = Signal()
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs = []
        self._is_running = False
        self.config = {}
        self._profile_manager = None

    def setup(self, cfg: dict, profile_manager):
        self.config = cfg
        self._profile_manager = profile_manager
        
        # Build jobs based on mode
        in_dir = Path(cfg.get("input_dir", ""))
        out_dir = Path(cfg.get("output_dir", ""))
        
        if not in_dir.exists():
            return
            
        os.makedirs(out_dir, exist_ok=True)
            
        exts = ("*.tiff", "*.tif", "*.png", "*.bmp", "*.jpg", "*.jpeg")
        files = sorted(
            [f for ext in exts for f in in_dir.glob(ext)],
            key=lambda p: p.name
        )
        
        self.jobs = []
        mode = cfg.get("mode", "")
        
        if mode == "Iterate files with active Profile":
            for f in files:
                self.jobs.append(BatchJob(f, out_dir / f.name, {"profile": "Active"}))
                
        elif mode == "Iterate files with selected Profiles":
            for prof in cfg.get("profiles", []):
                prof_dir = out_dir / prof
                os.makedirs(prof_dir, exist_ok=True)
                for f in files:
                    self.jobs.append(BatchJob(f, prof_dir / f.name, {"profile": prof}))
                    
        elif mode in ("Parameter Sweep (Z-stack)", "Auto-Focus (Best Z)"):
            z_s = cfg.get("z_start", 0.0)
            z_e = cfg.get("z_end", 1.0)
            z_steps = cfg.get("z_steps", 21)
            
            if z_steps > 1:
                z_vals = np.linspace(z_s, z_e, z_steps)
                for f in files:
                    out_dir_sweep = out_dir / ("Autofocus_Best" if mode == "Auto-Focus (Best Z)" else "Z_Sweep")
                    os.makedirs(out_dir_sweep, exist_ok=True)
                    self.jobs.append(BatchJob(f, out_dir_sweep / f.name, {
                        "mode": "sweep" if mode == "Parameter Sweep (Z-stack)" else "autofocus",
                        "z_vals": z_vals,
                        "focus_metric": cfg.get("focus_metric", "laplacian_variance"),
                        "algorithm": cfg.get("algorithm", "Linear Sweep (Exhaustive)"),
                        "profile": "Active"
                    }))

    def run(self):
        self._is_running = True
        total = len(self.jobs)

        if total == 0:
            self.status.emit("Error: No valid images found or directories invalid.")
            self.finished_batch.emit()
            return

        start_time = time.time()
        error_count = 0

        csv_path = Path(self.config["output_dir"]) / "focus_metrics.csv"
        csv_file = None
        writer = None
        if self.config.get("export_csv", True):
            csv_file = open(csv_path, 'w', newline='')
            writer = csv.writer(csv_file)
            writer.writerow(["Filename", "Profile", "Z_offset_mm", "FocusScore"])

        try:
            for i, job in enumerate(self.jobs):
                if not self._is_running:
                    break

                self.status.emit(f"Processing ({i+1}/{total}): {job.in_file.name}")
                self.progress.emit(i + 1, total)

                try:
                    self._process_single_job(job, writer)
                except Exception as e:
                    error_count += 1
                    self.error_occurred.emit(f"{job.in_file.name}: {e}")
                    continue

                # Calculate ETA
                elapsed = time.time() - start_time
                completed = i + 1
                avg_time = elapsed / completed
                eta_seconds = int((total - completed) * avg_time)
                m, s = divmod(eta_seconds, 60)
                h, m = divmod(m, 60)
                eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                self.eta_update.emit(eta_str)
        finally:
            if csv_file:
                csv_file.close()

        if self.config.get("export_report", True):
            report_path = Path(self.config["output_dir"]) / "reconstruction_report.txt"
            elapsed_total = time.time() - start_time
            with open(report_path, 'w') as f:
                f.write("Batch Reconstruction Report\n")
                f.write("===========================\n")
                f.write(f"Total Jobs Processed : {total}\n")
                f.write(f"Errors / Skipped     : {error_count}\n")
                f.write(f"Total Time Taken     : {elapsed_total:.2f} seconds\n")
                f.write(f"Mode Used            : {self.config.get('mode', 'Unknown')}\n")
                if total > 0:
                    f.write(f"Average Time / Image : {(elapsed_total/total):.3f} seconds\n")

        self.status.emit("Complete" if self._is_running else "Cancelled")
        self._is_running = False
        self.finished_batch.emit()

    def _process_single_job(self, job: BatchJob, writer):
        """Process one batch job (single file reconstruction or sweep)."""
        loaded = load_any(job.in_file)
        arr = np.asarray(loaded.array)
        if arr.ndim == 3:
            arr = arr[..., 0]

        # Extract Profile State
        prof = job.params.get("profile", "Active")
        if prof == "Active":
            recon_state = self.config.get("active_state", {}).get("recon", {})
            process_state = self.config.get("active_state", {}).get("process", {})
        else:
            st = self._profile_manager.load_profile("setup", prof)
            recon_state = st.get("recon", {}) if st else {}
            process_state = st.get("process", {}) if st else {}

        # Parameters Mapping
        wl = float(recon_state.get("wavelength", 632.8)) * 1e-9
        is_eff = recon_state.get("pixel_is_effective", True)
        px = float(recon_state.get("pixel_size", 3.45)) * 1e-6
        if not is_eff:
            mag = float(recon_state.get("magnification", 10.0))
            px = px / (mag if mag > 0 else 1.0)
        n_medium = float(recon_state.get("n", recon_state.get("refractive_index", 1.0)))
        mask_radius = int(recon_state.get("mask_radius", 80))
        method_str = recon_state.get("method", "ASM")
        method = ReconstructionMethod.ASM if method_str == "ASM" else ReconstructionMethod.FRESNEL

        mode_type = job.params.get("mode", "single")
        save_amp = self.config.get("save_amp", True)
        save_pha = self.config.get("save_pha", True)
        save_unwrapped_pha = self.config.get("save_unwrapped_pha", False)
        save_real = self.config.get("save_real", False)
        save_imag = self.config.get("save_imag", False)

        def _do_save(base_path, cfield):
            if save_amp:
                _save_normalized(base_path.with_suffix(".amp.tiff"), np.abs(cfield), False)
            if save_pha:
                _save_normalized(base_path.with_suffix(".pha.tiff"), np.angle(cfield), True)
            if save_unwrapped_pha:
                _save_normalized(base_path.with_suffix(".unwrapped.tiff"),
                    unwrap_phase_advanced(np.angle(cfield), complex_field=cfield,
                                         wavelength_m=wl, pixel_size_m=px), True)
            if save_real:
                _save_normalized(base_path.with_suffix(".real.tiff"), np.real(cfield), False)
            if save_imag:
                _save_normalized(base_path.with_suffix(".imag.tiff"), np.imag(cfield), False)

        # Pre-processing & Offaxis Extraction
        img = arr.astype(np.float32)
        subtract_mean = process_state.get("subtract_mean", False)
        hann_window = process_state.get("hann_window", False)
        if subtract_mean:
            img = img - float(np.mean(img))
        if hann_window:
            wy = np.hanning(img.shape[0]).astype(np.float32)
            wx = np.hanning(img.shape[1]).astype(np.float32)
            img = img * (wy[:, None] * wx[None, :])
        max_val = float(np.max(np.abs(img)))
        img = img / max(max_val, 1e-12)

        mask_apod = process_state.get("mask_apodization", "tukey")
        mask_roll = float(process_state.get("mask_rolloff", 0.25))
        offaxis_params = OffAxisParams(radius=mask_radius, apodization=mask_apod, rolloff=mask_roll)
        fc, _, _, _ = extract_complex_field_offaxis_debug(img, offaxis_params)

        # Reference complex field for phase correction (complex division)
        ref_complex = self.config.get("reference_complex", None)

        def _apply_ref(cfield):
            if ref_complex is not None:
                ref_abs = np.abs(ref_complex)
                safe_ref = np.where(ref_abs > 1e-10, ref_complex,
                                    np.ones_like(ref_complex))
                return cfield / safe_ref
            return cfield

        if mode_type in ("sweep", "autofocus"):
            self._process_sweep_or_autofocus(
                job, fc, wl, px, n_medium, method, prof, mode_type, writer, _do_save, _apply_ref
            )
        else:
            # Normal Single/Multi Profile Logic
            z_val = job.params.get("z_mm", float(recon_state.get("z_mm", 0.0)))
            z_m = z_val * 1e-3
            recon_params = ReconstructionParams(
                wavelength_m=wl, pixel_size_m=px, z_m=z_m, n=n_medium
            )
            complex_field = _apply_ref(propagate(fc, recon_params, method, force_python=True))

            base_stem = job.out_file.stem
            out_base = job.out_file.with_name(f"{base_stem}_Z_{z_val:.4f}mm")
            _do_save(out_base, complex_field)

            if writer:
                score = float(np.var(np.abs(complex_field)))
                writer.writerow([job.in_file.name, prof, f"{z_val:.4f}", f"{score:.2f}"])

    def _process_sweep_or_autofocus(self, job, fc, wl, px, n_medium, method,
                                     prof, mode_type, writer, _do_save, _apply_ref):
        """Handle Z-sweep and autofocus modes for a single file."""
        z_vals = job.params.get("z_vals", [0.0])
        algorithm = job.params.get("algorithm", "Linear Sweep (Exhaustive)")
        metric_val = job.params.get("focus_metric", "laplacian_variance")
        try:
            metric = FocusMetric(metric_val)
        except ValueError:
            metric = FocusMetric.LAPLACIAN_VARIANCE

        best_score = float('inf') if _is_minimize(metric) else -float('inf')
        best_complex = None
        best_z = 0.0

        zmin_m = float(min(z_vals)) * 1e-3
        zmax_m = float(max(z_vals)) * 1e-3
        recon_base = ReconstructionParams(
            wavelength_m=wl, pixel_size_m=px, z_m=0.0, n=n_medium
        )

        if mode_type == "autofocus" and "Robust" in algorithm:
            from core.autofocus import robust_coarse_to_fine_search
            res = robust_coarse_to_fine_search(
                field=fc, base_params=recon_base, method=method, metric=metric,
                z_min_m=zmin_m, z_max_m=zmax_m,
                n_coarse=max(20, len(z_vals)),
            )
            best_z = res.best_z_m * 1e3
            best_complex = _apply_ref(propagate(
                fc, ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=res.best_z_m, n=n_medium),
                method, force_python=True,
            ))
            if writer:
                writer.writerow([job.in_file.name, prof, f"{best_z:.4f}", f"{res.best_score:.2f}"])
            out_base = job.out_file.with_name(f"{job.out_file.stem}_Best_Z_{best_z:.4f}mm")
            _do_save(out_base, best_complex)

        elif mode_type == "autofocus" and "Golden" in algorithm:
            from core.autofocus import golden_section_search
            res = golden_section_search(
                field=fc, base_params=recon_base, method=method, metric=metric,
                z_min_m=zmin_m, z_max_m=zmax_m,
                tolerance_m=1e-6,
            )
            best_z = res.best_z_m * 1e3
            best_complex = _apply_ref(propagate(
                fc, ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=res.best_z_m, n=n_medium),
                method, force_python=True,
            ))
            if writer:
                writer.writerow([job.in_file.name, prof, f"{best_z:.4f}", f"{res.best_score:.2f}"])
            out_base = job.out_file.with_name(f"{job.out_file.stem}_Best_Z_{best_z:.4f}mm")
            _do_save(out_base, best_complex)

        elif mode_type == "autofocus" and "Coarse-to-Fine" in algorithm:
            from core.autofocus import coarse_to_fine_search
            res = coarse_to_fine_search(
                field=fc, base_params=recon_base, method=method, metric=metric,
                z_min_m=zmin_m, z_max_m=zmax_m,
                coarse_steps=len(z_vals), fine_tolerance_m=1e-6,
            )
            best_z = res.best_z_m * 1e3
            best_complex = _apply_ref(propagate(
                fc, ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=res.best_z_m, n=n_medium),
                method, force_python=True,
            ))
            if writer:
                writer.writerow([job.in_file.name, prof, f"{best_z:.4f}", f"{res.best_score:.2f}"])
            out_base = job.out_file.with_name(f"{job.out_file.stem}_Best_Z_{best_z:.4f}mm")
            _do_save(out_base, best_complex)

        else:
            # Linear Sweep (Exhaustive) — evaluate every Z
            for z_val in z_vals:
                if not self._is_running:
                    break

                self.status.emit(f"Sweeping: {job.in_file.name} (Z = {z_val:.3f} mm)")
                z_m = z_val * 1e-3
                recon_params = ReconstructionParams(
                    wavelength_m=wl, pixel_size_m=px, z_m=z_m, n=n_medium
                )
                complex_field = _apply_ref(propagate(fc, recon_params, method, force_python=True))

                try:
                    score = _calc_metric(complex_field, metric)
                except Exception:
                    score = float(np.var(np.angle(complex_field)))

                if writer:
                    writer.writerow([job.in_file.name, prof, f"{z_val:.4f}", f"{score:.2f}"])

                minimize = _is_minimize(metric)
                better = (score < best_score) if minimize else (score > best_score)
                if better:
                    best_score = score
                    best_complex = complex_field.copy()
                    best_z = z_val

                if mode_type == "sweep":
                    out_base = job.out_file.with_name(f"{job.out_file.stem}_Z_{z_val:.4f}mm")
                    _do_save(out_base, complex_field)

            if mode_type == "autofocus" and best_complex is not None and self._is_running:
                out_base = job.out_file.with_name(f"{job.out_file.stem}_Best_Z_{best_z:.4f}mm")
                _do_save(out_base, best_complex)

    def stop(self):
        self._is_running = False
