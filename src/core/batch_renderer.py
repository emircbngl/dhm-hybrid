from PySide6.QtCore import QThread, Signal
import time
from pathlib import Path
import os
import sys
import csv
from typing import Optional
import numpy as np
import tifffile
from core.phase_unwrap import unwrap_phase_advanced
from core.ingestion import load_any
from core.reconstruction import (
    propagate,
    ReconstructionParams,
    ReconstructionMethod,
    safe_reference_divide,
)
from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug
from core.autofocus import FocusMetric, _calc_metric, _is_minimize


# ---------------------------------------------------------------------------
# Reference hologram auto-pairing.
#
# Lab convention: each sample recording <name>.<ext> ships with a paired
# reference acquired on the same setup *without* the sample. Two naming
# patterns are common in the field:
#
#   * suffix:  ``<name>_ref.<ext>`` / ``<name>-ref.<ext>``
#   * prefix:  ``ref_<name>.<ext>`` / ``ref-<name>.<ext>``
#
# Case-insensitive — labs mix ``Ref``, ``REF``, ``ref`` interchangeably.
#
# The renderer uses these helpers to:
#   1. Skip reference files when iterating an input directory (they're not
#      samples; reconstructing them produces meaningless output).
#   2. Auto-discover the paired reference for each sample at job time when
#      the v1 GUI hasn't supplied an explicit ``reference_fc``/``reference_complex``.
# ---------------------------------------------------------------------------

_REF_PREFIXES = ("ref_", "ref-")
_REF_SUFFIXES = ("_ref", "-ref")


def _is_reference_filename(path: Path) -> bool:
    """True if the file's stem looks like a reference recording."""
    stem = path.stem.lower()
    if any(stem.startswith(p) for p in _REF_PREFIXES):
        return True
    if any(stem.endswith(s) for s in _REF_SUFFIXES):
        return True
    return False


def _find_reference_for(sample_path: Path) -> Optional[Path]:
    """Find the paired reference hologram for ``sample_path``.

    Returns the first existing file matching either the suffix or prefix
    convention, in case-variant order (lower → upper → title). ``None``
    when nothing pairs — caller can fall back to its other reference
    sources.

    The function does NOT recurse; the reference must sit alongside the
    sample. Lab-data hierarchies that bury references under
    ``./refs/`` are out of scope until someone asks for it.
    """
    if _is_reference_filename(sample_path):
        # Don't try to ref a ref — we'd recurse and pollute the output dir.
        return None

    parent = sample_path.parent
    stem = sample_path.stem
    suffix = sample_path.suffix

    case_variants = ("ref", "REF", "Ref")
    for ref in case_variants:
        for sep in ("_", "-"):
            for candidate in (
                parent / f"{stem}{sep}{ref}{suffix}",
                parent / f"{ref}{sep}{stem}{suffix}",
            ):
                if candidate.exists():
                    return candidate
    return None


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
        # Reference hologram files (``<stem>_ref.<ext>`` / ``ref_<stem>.<ext>``)
        # live alongside samples in the same input directory. When the user
        # opts in to auto-pairing we drop them from the job list so the
        # renderer only reconstructs samples; the paired reference is
        # fetched per-job by ``_find_reference_for``. When auto-pair is
        # off the user gets the legacy behaviour: every recognised image
        # in the folder becomes a job.
        auto_pair = bool(cfg.get("auto_pair_reference", True))
        # When the caller supplies an explicit reference, per-job auto-pair
        # is skipped anyway (see _process_single_job), so filtering
        # ref-named files would only silently drop legitimate samples
        # (e.g. "reflow_01.tif", "blood_ref.tif") from the job list.
        has_explicit_ref = (
            cfg.get("reference_fc") is not None
            or cfg.get("reference_complex") is not None
        )
        all_files = [f for ext in exts for f in in_dir.glob(ext)]
        if auto_pair and not has_explicit_ref:
            dropped = sorted(
                f.name for f in all_files if _is_reference_filename(f)
            )
            all_files = [f for f in all_files if not _is_reference_filename(f)]
            if dropped:
                shown = ", ".join(dropped[:8]) + ("…" if len(dropped) > 8 else "")
                self.status.emit(
                    f"Auto-pair: {len(dropped)} reference-named file(s) "
                    f"excluded from jobs: {shown}"
                )
        files = sorted(all_files, key=lambda p: p.name)
        
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

        skip_existing = bool(self.config.get("skip_existing", False))
        skipped_count = 0

        try:
            for i, job in enumerate(self.jobs):
                if not self._is_running:
                    break

                if skip_existing and self._job_already_finished(job):
                    skipped_count += 1
                    self.status.emit(
                        f"Skipping ({i+1}/{total}): {job.in_file.name} — "
                        "output exists"
                    )
                    self.progress.emit(i + 1, total)
                    continue

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
                f.write(f"Errors               : {error_count}\n")
                f.write(f"Skipped (existing)   : {skipped_count}\n")
                f.write(f"Total Time Taken     : {elapsed_total:.2f} seconds\n")
                f.write(f"Mode Used            : {self.config.get('mode', 'Unknown')}\n")
                if total > 0:
                    f.write(f"Average Time / Image : {(elapsed_total/total):.3f} seconds\n")

        self.status.emit("Complete" if self._is_running else "Cancelled")
        self._is_running = False
        self.finished_batch.emit()

    def _job_already_finished(self, job: BatchJob) -> bool:
        """Heuristic: every save_* output the user asked for already
        sits next to ``job.out_file``.

        We can't enumerate sweep TIFFs ahead of time (z values are
        decided per-job inside ``_process_single_job``), so for sweep
        / autofocus modes we treat the *first* matching glob as
        sufficient — finishing a sweep produces dozens of files all
        named ``<stem>_Z_*.amp.tiff``, so finding any one is a
        reliable "already ran" signal.
        """
        save_amp = self.config.get("save_amp", True)
        save_pha = self.config.get("save_pha", True)
        save_unwrapped_pha = self.config.get("save_unwrapped_pha", False)
        save_real = self.config.get("save_real", False)
        save_imag = self.config.get("save_imag", False)
        wanted_suffixes = []
        if save_amp:
            wanted_suffixes.append(".amp.tiff")
        if save_pha:
            wanted_suffixes.append(".pha.tiff")
        if save_unwrapped_pha:
            wanted_suffixes.append(".unwrapped.tiff")
        if save_real:
            wanted_suffixes.append(".real.tiff")
        if save_imag:
            wanted_suffixes.append(".imag.tiff")
        if not wanted_suffixes:
            return False

        mode_type = job.params.get("mode", "single")
        out_dir = job.out_file.parent
        if mode_type in ("sweep", "autofocus"):
            # Sweep / autofocus emit ``<stem>_Z_*<suffix>`` or
            # ``<stem>_Best_Z_*<suffix>`` — any glob hit means the job ran.
            stem = job.out_file.stem
            for suffix in wanted_suffixes:
                hits = list(out_dir.glob(f"{stem}_*Z_*{suffix}"))
                if not hits:
                    return False
            return True
        # Single-frame mode — output named ``<stem>_Z_<z>mm<suffix>``.
        stem = job.out_file.stem
        for suffix in wanted_suffixes:
            hits = list(out_dir.glob(f"{stem}_Z_*{suffix}"))
            if not hits:
                return False
        return True

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
            # APPEND the channel suffix instead of Path.with_suffix():
            # base names carry a decimal z ("sample_Z_0.5000mm"), and
            # with_suffix treats ".5000mm" as the extension and strips it —
            # collapsing every same-integer-z slice of a sweep onto one
            # filename that silently overwrites.
            def _out(suffix):
                return base_path.with_name(base_path.name + suffix)
            if save_amp:
                _save_normalized(_out(".amp.tiff"), np.abs(cfield), False)
            if save_pha:
                _save_normalized(_out(".pha.tiff"), np.angle(cfield), True)
            if save_unwrapped_pha:
                _save_normalized(_out(".unwrapped.tiff"),
                    unwrap_phase_advanced(np.angle(cfield), complex_field=cfield,
                                         wavelength_m=wl, pixel_size_m=px), True)
            if save_real:
                _save_normalized(_out(".real.tiff"), np.real(cfield), False)
            if save_imag:
                _save_normalized(_out(".imag.tiff"), np.imag(cfield), False)

        # Pre-processing & Offaxis Extraction
        # subtract_mean defaults to True to match v1's process_tab default
        # (process_tab.py:44 — checked by default) and v2 ReconParams default
        # (ui2/reconstruction.py:92). Older batch sessions defaulted to False,
        # which caused amplitude-scale drift between batch and the live
        # reconstruct path on the same hologram (verified 2026-04-29).
        subtract_mean = process_state.get("subtract_mean", True)
        hann_window = process_state.get("hann_window", False)
        mask_apod = process_state.get("mask_apodization", "tukey")
        mask_roll = float(process_state.get("mask_rolloff", 0.25))
        offaxis_params = OffAxisParams(radius=mask_radius, apodization=mask_apod, rolloff=mask_roll)

        def _demodulate(raw: np.ndarray) -> np.ndarray:
            """Sample → Fourier ``fc``, applying the same prep as the live path."""
            x = raw.astype(np.float32, copy=True)
            if subtract_mean:
                x = x - float(np.mean(x))
            if hann_window:
                wy = np.hanning(x.shape[0]).astype(np.float32)
                wx = np.hanning(x.shape[1]).astype(np.float32)
                x = x * (wy[:, None] * wx[None, :])
            peak = float(np.max(np.abs(x)))
            x = x / max(peak, 1e-12)
            f, _, _, _ = extract_complex_field_offaxis_debug(x, offaxis_params)
            return f

        fc = _demodulate(arr)

        # Reference handling. Three sources, in priority order:
        #   1. ``reference_fc``      — explicit pre-propagation reference
        #      from v1 GUI's "Load reference" workflow. Best: lets
        #      autofocus / sweep re-propagate to each trial z.
        #   2. ``reference_complex`` — propagated to user's nominal z.
        #      Legacy fallback for callers that don't supply ``reference_fc``.
        #   3. Auto-paired sibling — ``<stem>_ref.<ext>`` /
        #      ``ref_<stem>.<ext>`` next to the sample. Demodulated here
        #      with the same prep so it's compatible with case (1).
        #      Lab convention; user opts in by simply naming files this way.
        ref_fc = self.config.get("reference_fc", None)
        ref_complex = self.config.get("reference_complex", None)
        auto_pair_enabled = bool(self.config.get("auto_pair_reference", True))
        if auto_pair_enabled and ref_fc is None and ref_complex is None:
            paired = _find_reference_for(job.in_file)
            if paired is not None:
                try:
                    ref_loaded = load_any(paired)
                    ref_arr = np.asarray(ref_loaded.array)
                    if ref_arr.ndim == 3:
                        ref_arr = ref_arr[..., 0]
                    ref_fc = _demodulate(ref_arr)
                    self.status.emit(
                        f"Auto-paired reference: {paired.name} → {job.in_file.name}"
                    )
                except Exception as exc:
                    # Don't fail the whole job over a malformed sibling;
                    # just continue without reference division.
                    self.status.emit(
                        f"Reference '{paired.name}' load failed ({exc}); "
                        "continuing without reference division."
                    )
                    ref_fc = None

        def _apply_ref(cfield, *, z_m: float | None = None):
            """Divide cfield by the reference at the same propagation z.

            Prefers ``reference_fc`` (re-propagate to ``z_m``) so the
            autofocus / sweep best-Z and the saved field both see the
            same reference. Falls back to the legacy fixed
            ``reference_complex`` when ``reference_fc`` isn't supplied
            or ``z_m`` is None.
            """
            if ref_fc is not None and z_m is not None:
                ref_params = ReconstructionParams(
                    wavelength_m=wl, pixel_size_m=px, z_m=float(z_m), n=n_medium
                )
                ref_at_z = propagate(ref_fc, ref_params, method, force_python=True)
                return safe_reference_divide(cfield, ref_at_z)
            if ref_complex is not None:
                return safe_reference_divide(cfield, ref_complex)
            return cfield

        if mode_type in ("sweep", "autofocus"):
            self._process_sweep_or_autofocus(
                job, fc, wl, px, n_medium, method, prof, mode_type, writer,
                _do_save, _apply_ref, ref_fc=ref_fc,
            )
        else:
            # Normal Single/Multi Profile Logic
            z_val = job.params.get("z_mm", float(recon_state.get("z_mm", 0.0)))
            z_m = z_val * 1e-3
            recon_params = ReconstructionParams(
                wavelength_m=wl, pixel_size_m=px, z_m=z_m, n=n_medium
            )
            complex_field = _apply_ref(
                propagate(fc, recon_params, method, force_python=True), z_m=z_m
            )

            base_stem = job.out_file.stem
            out_base = job.out_file.with_name(f"{base_stem}_Z_{z_val:.4f}mm")
            _do_save(out_base, complex_field)

            if writer:
                score = float(np.var(np.abs(complex_field)))
                writer.writerow([job.in_file.name, prof, f"{z_val:.4f}", f"{score:.2f}"])

    def _process_sweep_or_autofocus(self, job, fc, wl, px, n_medium, method,
                                     prof, mode_type, writer, _do_save, _apply_ref,
                                     *, ref_fc=None):
        """Handle Z-sweep and autofocus modes for a single file.

        ``ref_fc`` is the demodulated reference (pre-propagation). When
        supplied, the autofocus search divides the sample by the
        reference at every trial z — so the metric judges the same
        field the user gets back. Without it, the search runs on the
        un-referenced sample (legacy v1 behaviour) and the best Z may
        diverge from what the operator sees.
        """
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
                ref_field=ref_fc,
            )
            best_z = res.best_z_m * 1e3
            best_complex = _apply_ref(propagate(
                fc, ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=res.best_z_m, n=n_medium),
                method, force_python=True,
            ), z_m=res.best_z_m)
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
                ref_field=ref_fc,
            )
            best_z = res.best_z_m * 1e3
            best_complex = _apply_ref(propagate(
                fc, ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=res.best_z_m, n=n_medium),
                method, force_python=True,
            ), z_m=res.best_z_m)
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
                ref_field=ref_fc,
            )
            best_z = res.best_z_m * 1e3
            best_complex = _apply_ref(propagate(
                fc, ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=res.best_z_m, n=n_medium),
                method, force_python=True,
            ), z_m=res.best_z_m)
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
                complex_field = _apply_ref(
                    propagate(fc, recon_params, method, force_python=True), z_m=z_m
                )

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
