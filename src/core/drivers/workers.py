"""Thread-based workers for autofocus / QPI / depth map / report
export. Mirrors ``src/ui2/reconstruction.py``'s pattern — each worker
takes a job + two callbacks and runs on a dedicated executor so the
DPG render loop stays responsive.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from core.audit import get_audit_log
from core.autofocus import (
    AutofocusCancelled,
    FocusMetric,
    autofocus_zscan,
    find_focus_candidates,
)
from core.autofocus.search_adaptive import (
    adaptive_bracketing_search,
    adaptive_distance_search,
    adaptive_gradient_search,
)
from core.autofocus.evaluator import focus_landscape_warning
from core.autofocus.search_classic import (
    coarse_to_fine_search,
    robust_coarse_to_fine_search,
)
from core.background_phase import subtract_background
from core.depth_map import (
    ClusterHeight,
    DepthMapResult,
    compute_depth_map,
    segment_depth_clusters,
    write_tomography_bundle,
)
from core.ingestion import load_any
from core.offaxis import OffAxisParams, extract_complex_field_offaxis
from core.phase_unwrap import UnwrapConfig, UnwrapMethod, unwrap_phase_advanced
from core.qpi import QPIMode, QPIResult, compute_qpi
from core.qpi_batch import QPIBatchEntry, run_qpi_for_candidates, write_qpi_batch_csv
from core.qpi_export import write_qpi_csv
from core.reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)
from core.report import generate_html_report

from .reconstruction import ReconError, ReconParams

_LOG = logging.getLogger(__name__)


def _metric_from_params(params: ReconParams) -> FocusMetric:
    """Look up the FocusMetric enum from the string stored on ReconParams.

    Falls back to LAPLACIAN_VARIANCE (the v2.0.2 default) if the name
    doesn't match — keeps old state files working."""
    name = str(getattr(params, "autofocus_metric",
                       "LAPLACIAN_VARIANCE") or "").upper()
    try:
        return FocusMetric[name]
    except KeyError:
        return FocusMetric.LAPLACIAN_VARIANCE


def available_focus_metrics() -> list[str]:
    """Every FocusMetric name, in enum declaration order. Used by the
    sidebar combo so the UI stays in sync with core automatically."""
    return [m.name for m in FocusMetric]


# Canonical list of autofocus search algorithms the sidebar exposes.
# Keys are user-visible strings; the values are the legacy v1 names
# used internally (kept here to avoid another round of string refactor
# when porting further v1 algorithms).
_AUTOFOCUS_ALGORITHMS: list[str] = [
    "zscan",               # fixed-step linear sweep — simplest, baseline
    "coarse_to_fine",      # coarse grid → Golden-Section refinement
    "robust",              # coarse scan + Gaussian-smoothed fine refine
    "adaptive_gradient",   # variable step driven by metric derivative
    "adaptive_bracketing", # nested bracketing + divisions refinement
    "adaptive_distance",   # auto-discovers the z-range from scratch
]

# Which UI fields each algorithm actually consumes. The sidebar
# greys out / relabels anything an algorithm ignores so the user
# can see at a glance what the input actually does. Used by the
# sidebar layer via :func:`af_algorithm_input_profile` — keeps the
# UI / dispatch contract in one place so drift is impossible.
_AF_ALGO_PROFILES: dict[str, dict] = {
    # steps_tip verdict lines: 9-scene real-lab benchmark, 2026-07-06
    # (scripts/benchmark_af_real.py; median |z err| / ≤0.5 mm hit-rate
    # at the same 40-eval budget, laplacian & entropy metrics).
    "zscan": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "steps",
        "steps_tip": (
            "Number of z samples on the uniform grid. Real-data "
            "benchmark: least accurate at 40 steps (33%/44% hit) — "
            "grid resolution IS the accuracy; prefer 'robust'."
        ),
    },
    "coarse_to_fine": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "coarse steps",
        "steps_tip": (
            "Coarse-grid sample count. After the coarse pass the "
            "algorithm refines with Golden-Section search — no "
            "extra budget needed."
        ),
    },
    "robust": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "coarse steps",
        "steps_tip": (
            "Coarse-grid sample count before Gaussian smoothing "
            "+ 8× refine pass. Real-data benchmark: DEFAULT — the "
            "only top performer on both metrics (78%/67% hit, "
            "0.07/0.15 mm median error)."
        ),
    },
    "adaptive_gradient": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "max evals",
        "steps_tip": (
            "Evaluation budget — the algorithm adapts step size "
            "itself (big in flat regions, tiny near peak). Unlike "
            "the fixed grid, this is a cap, not a sample count. "
            "Real-data benchmark: fastest accurate option on "
            "gradient-friendly metrics (laplacian: 0.11 mm, 67%) "
            "but UNRELIABLE with ENTROPY (7.2 mm median) — its "
            "walker stalls on entropy's flat shoulders."
        ),
    },
    "adaptive_bracketing": {
        "uses_z_range": True,
        "uses_steps": True,
        "steps_label": "max evals",
        "steps_tip": (
            "Evaluation budget across the nested bracketing + "
            "refinement levels (3 levels × 8 divisions + refine). "
            "Cap, not fixed sample count. Real-data benchmark: "
            "best laplacian accuracy (0.03 mm, 78%) — typically "
            "overruns the cap ~30% (53 evals for 40)."
        ),
    },
    "adaptive_distance": {
        "uses_z_range": False,
        "uses_steps": True,
        "steps_label": "max evals",
        "steps_tip": (
            "Evaluation budget. Adaptive-distance starts with a "
            "small range around z = 0 and expands until the peak "
            "is interior — user-supplied z_min / z_max are "
            "ignored. Real-data benchmark: UNRELIABLE on this "
            "rig's ±11–18 mm scenes (0–22% hit; expansion stops "
            "early) — use only when the range is truly unknown, "
            "and verify the result."
        ),
    },
}


def af_algorithm_input_profile(algo: str) -> dict:
    """Return which sidebar inputs a given algorithm actually uses.

    Keys:

    * ``uses_z_range`` — whether ``af_z_min_mm`` / ``af_z_max_mm``
      influence the run (False for ``adaptive_distance``).
    * ``uses_steps`` — whether ``af_n_steps`` matters at all.
    * ``steps_label`` — the label the sidebar should display
      (e.g. "max evals" for adaptive variants, "steps" for grid
      methods).
    * ``steps_tip`` — per-algorithm tooltip copy.
    """
    return dict(_AF_ALGO_PROFILES.get(
        str(algo or "").lower(),
        _AF_ALGO_PROFILES["zscan"],
    ))


def available_autofocus_algorithms() -> list[str]:
    """Names of autofocus search algorithms the sidebar can pick
    from. Kept as a free function so the UI layer can build the
    combo without importing the dispatch details."""
    return list(_AUTOFOCUS_ALGORITHMS)


def _extract_best_z(result: Any) -> float:
    """Pull ``best_z_m`` out of any of the five result dataclasses
    core returns — autofocus_zscan returns ``AutoFocusResult``,
    coarse/robust return their own, adaptive variants return
    ``AdaptiveStepResult`` / ``RobustResult``. They all carry
    ``best_z_m`` as the first field by convention."""
    return float(getattr(result, "best_z_m"))


def _extract_best_score(result: Any) -> float:
    """Best-score extractor with graceful fallbacks — some result
    dataclasses expose it as ``best_score`` directly, autofocus_zscan
    carries it in its ``scores`` dict keyed by z."""
    if hasattr(result, "best_score"):
        return float(result.best_score)
    if hasattr(result, "scores"):
        scores = result.scores
        best_z = float(getattr(result, "best_z_m"))
        return float(scores.get(best_z, 0.0))
    return 0.0


def _extract_evaluations(result: Any, fallback: int) -> int:
    """Count the number of z evaluations consumed. Adaptive results
    expose it as ``evaluations``; zscan we compute from the supplied
    z list length upstream and pass via ``fallback``."""
    if hasattr(result, "evaluations"):
        return int(result.evaluations)
    # The default 'robust' path (RobustSearchResult) names it differently —
    # without this it silently reported the coarse n_steps (~half its true
    # cost) in the UI and the reproducibility audit (2026-07-08 review).
    if hasattr(result, "total_evaluations"):
        return int(result.total_evaluations)
    return int(fallback)


def _join_notes(*notes: Optional[str]) -> Optional[str]:
    """Combine several optional non-fatal notes into one, dropping None/empty.
    Returns None when nothing is worth telling the operator so neither a
    reference-fallback (B-110) nor a flat-landscape (B-100) note masks the
    other."""
    present = [n for n in notes if n]
    if not present:
        return None
    return "  |  ".join(present)


def _landscape_warning(result: Any) -> Optional[str]:
    """Actionable flat / non-finite focus-landscape warning for whichever
    search ran (2026-07-08).

    The linear zscan already diagnoses its full landscape and exposes it as
    ``AutoFocusResult.warning``. The SETTLED DEFAULT ``robust`` search left the
    warning at ``None`` — a silent-degrade gap: on a mis-parameterised run
    (wrong pixel/mag/wavelength, bad z-range, wrong +1-order mask) it returned
    the z of pure noise with no signal to the user. Its uniform coarse sweep
    (``coarse_z``/``coarse_scores``) is exactly the dense landscape
    ``focus_landscape_warning`` expects, so diagnose from that.

    Left undiagnosed (return ``None``): golden / coarse_to_fine keep only the
    best point (no landscape), and the adaptive traces are non-uniform search
    paths whose scores cluster near the peak — feeding them to the flat-range
    heuristic would false-positive, so they are intentionally not checked."""
    if hasattr(result, "warning"):
        return result.warning  # linear zscan — already diagnosed
    zs = getattr(result, "coarse_z", None)
    fs = getattr(result, "coarse_scores", None)
    if zs is None or fs is None:
        return None
    scores = {float(z): float(s)
              for z, s in zip(np.asarray(zs).ravel(), np.asarray(fs).ravel())}
    return focus_landscape_warning(scores)


def _params_for_audit(params: ReconParams, **extra: Any) -> dict:
    """Flatten ReconParams into a JSON-friendly dict for audit logging.

    Mirrors the v1 ``main_window._audit`` spirit: every field that
    influenced the result lands in the record so a future operator can
    reproduce the run from the log alone. Extra kwargs (z_range,
    sample_id, hologram path, …) get merged on top."""
    d: dict[str, Any] = {
        "wavelength_nm": float(params.wavelength_nm),
        "pixel_um": float(params.pixel_um),
        "z_mm": float(params.z_mm),
        "mask_radius": int(params.mask_radius),
        "method": str(params.method),
        "magnification": float(params.magnification),
        "pixel_is_effective": bool(params.pixel_is_effective),
        "effective_pixel_um": float(params.effective_pixel_um()),
        "n_sample": float(params.n_sample),
        "n_medium": float(params.n_medium),
        "autofocus_metric": str(params.autofocus_metric),
        "af_z_min_mm": float(params.af_z_min_mm),
        "af_z_max_mm": float(params.af_z_max_mm),
        "af_n_steps": int(params.af_n_steps),
        "reference_path": (str(params.reference_path)
                           if params.reference_path else ""),
        "subtract_reference": bool(params.subtract_reference),
        # The RESOLVED mode, not just the raw legacy flag — an auditor
        # must be able to tell which reference behaviour actually gated
        # the run (2026-07-05 review).
        "reference_mode": params.effective_reference_mode(),
    }
    if d["reference_mode"] == "reference_free":
        d["reffree_bg_method"] = str(params.reffree_bg_method)
        d["reffree_bg_order"] = int(params.reffree_bg_order)
        d["reffree_cnn"] = bool(params.reffree_cnn)
    for k, v in extra.items():
        if v is None:
            continue
        d[k] = str(v) if isinstance(v, Path) else v
    return d


def _emit_audit(action: str, params: dict,
                result_summary: Optional[dict] = None) -> None:
    """Best-effort audit write — never raises, never blocks science."""
    try:
        get_audit_log().record(
            action=action, params=params, result_summary=result_summary,
        )
    except Exception:
        _LOG.warning("audit write failed for %s", action, exc_info=True)


# ---------------------------------------------------------------------------
# Shared helpers — a demodulated complex field any worker starts from.
# ---------------------------------------------------------------------------

def _offaxis_params(params: ReconParams) -> OffAxisParams:
    """Build :class:`OffAxisParams` from :class:`ReconParams`.

    Single source of truth for the off-axis config so every pipeline path
    (reconstruct / qpi / autofocus / depth) honours the same manual
    +1-order center. ``offaxis_center`` may arrive as a list after a JSON
    state round-trip — coerce to an int tuple; ``None`` = auto-detect
    (2026-07-08: manual center was accepted by the core but never plumbed
    from ReconParams into any of the three OffAxisParams call sites)."""
    center = params.offaxis_center
    if center is not None:
        center = (int(center[0]), int(center[1]))
    return OffAxisParams(radius=int(params.mask_radius), center_yx=center)


def _preprocess_raw(raw: np.ndarray, params: ReconParams) -> np.ndarray:
    """Pre-offaxis preprocessing shared by every pipeline path.

    Mirrors ``core.batch_renderer``'s branch order:

    1. Subtract the image mean when the user has armed DC removal
       (reduces the 0-order bleed into the +1 order after FFT).
    2. Apply a 2-D Hann taper when requested (suppresses edge
       discontinuities that leak into the FFT as wide sinc wings).
    3. Normalise by the absolute peak so downstream math stays in
       [-1, 1].

    ``ReconstructionDriver._run`` and ``ScienceDriver._prepare_field``
    used to duplicate this logic — they drifted in v2.0.5 (only
    the Reconstruct button got subtract_mean / hann wiring, every
    other pipeline ran raw normalisation). Bug report 2026-04-24:
    "reference subtract isn't working" was half-caused by that
    skew, so it's extracted here as a single source of truth.
    """
    raw = np.asarray(raw, dtype=np.float32, copy=False)
    if getattr(params, "subtract_mean", True):
        raw = raw - float(np.mean(raw))
    if getattr(params, "hann_window", False):
        wy = np.hanning(raw.shape[0]).astype(np.float32)
        wx = np.hanning(raw.shape[1]).astype(np.float32)
        raw = raw * (wy[:, None] * wx[None, :])
    peak = float(np.max(np.abs(raw)))
    if peak > 0:
        raw = raw / peak
    return raw


def _extract_field_with_reference(
    raw: np.ndarray,
    params: ReconParams,
    offaxis: OffAxisParams,
) -> Tuple[np.ndarray, Tuple[int, int], Optional[str]]:
    """Off-axis demodulation + optional reference-division.

    Returns ``(field, +1_order_centre, note)``. ``+1_order_centre`` is
    the ``(y, x)`` pixel the demodulator masked out of the sample
    hologram — downstream diagnostics still want to see it even
    when reference division is on. ``note`` is ``None`` on success and
    an actionable message when reference mode was armed but the division
    FAILED (reference file moved/corrupt/shape-mismatch) and we fell back
    to the unreferenced field — the caller must surface it so the user
    isn't handed a silently-unreferenced (quantitatively wrong) result
    that looks successful (2026-07-08 review, B-110).

    ``subtract_reference=True`` loads a second hologram (the
    "reference" — sample-free background), demodulates *it* with
    the same mask, and divides the two complex fields. Division
    (not subtraction) because the reference carries the
    illumination profile multiplicatively; dividing cancels
    illumination non-uniformity and leaves the sample-induced
    perturbation behind.

    Up to v2.0.8 this logic existed only in
    ``ReconstructionDriver._run`` — autofocus, multi-focus, QPI
    and depth map all ignored ``subtract_reference`` because
    ``_prepare_field`` skipped it. User noticed with a real lab
    hologram. Now sits on the shared prep path so every pipeline
    respects the reference toggle identically.

    Phase 3: gated on :meth:`ReconParams.effective_reference_mode`
    rather than the raw ``subtract_reference`` flag — when the user
    has picked "reference_free" mode, a stale/loaded reference path
    must NOT get divided in here (that division would compete with
    the numerical background fit applied after unwrap).
    """
    field, center = extract_complex_field_offaxis(raw, offaxis)
    center_t = tuple(int(v) for v in center)
    if not (params.effective_reference_mode() == "reference"
            and params.reference_path):
        return field, center_t, None
    try:
        ref_loaded = load_any(params.reference_path)
        ref_raw = np.asarray(ref_loaded.array, dtype=np.float32)
        if ref_raw.ndim == 3:
            ref_raw = ref_raw[..., 0]
        ref_raw = _preprocess_raw(ref_raw, params)
        ref_field, _ = extract_complex_field_offaxis(ref_raw, offaxis)
        eps = 1e-9
        return field / (ref_field + eps), center_t, None
    except Exception as exc:
        _LOG.warning("reference subtract failed (%s) — "
                     "falling back to unreferenced field", exc)
        return field, center_t, (
            f"Reference division FAILED ({type(exc).__name__}: {exc}); this "
            "result is UNREFERENCED — illumination non-uniformity was not "
            "removed, so quantitative phase/OPD/dry-mass are unreliable. "
            "Check the reference file path/format.")


# ---------------------------------------------------------------------------
# Reference-free background correction (Phase 3, AI_VISION_MCP_PLAN.md)
# ---------------------------------------------------------------------------

# Repo root: src/core/drivers/workers.py -> drivers -> core -> src -> <repo>.
# parents[3], NOT [2] — the 2026-07-06 relocation moved this file one level
# deeper and the stale index silently pointed _TRACK_C_CHECKPOINT at
# <repo>/src/models/..., disabling the Track C CNN (B-096).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRACK_C_CHECKPOINT = _REPO_ROOT / "models" / "track_c" / "v0.1" / "model.pt"


def reffree_cnn_available(checkpoint: Optional[Path] = None) -> bool:
    """True when the Track C CNN residual corrector can actually run.

    Two conditions, both required: torch importable in the current
    interpreter (the shipped venv doesn't have it — GPU/MPS envs do),
    and a trained checkpoint on disk. The sidebar uses this to grey
    out the ``reffree_cnn`` checkbox with an explanatory tooltip
    instead of silently no-op'ing when the user checks it.
    """
    ckpt = Path(checkpoint) if checkpoint is not None else _TRACK_C_CHECKPOINT
    if not ckpt.exists():
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _apply_reffree_correction(
    phase_wrapped: np.ndarray,
    amplitude: np.ndarray,
    complex_field: np.ndarray,
    recon_params: ReconstructionParams,
    params: ReconParams,
) -> Tuple[np.ndarray, Optional[str]]:
    """Unwrap + numerical background-fit for reference-free mode.

    Runs ``core.phase_unwrap.unwrap_phase_advanced`` (same method the
    user picked for QPI — ``params.unwrap_method``) then
    ``core.background_phase.subtract_background`` with the reffree
    knobs on ``params``. When ``params.reffree_cnn`` is armed AND
    :func:`reffree_cnn_available` says a checkpoint + torch are both
    present, additionally runs the Track C CNN residual corrector on
    top of the classical fit.

    Returns ``(corrected_phase, note)``. ``note`` is ``None`` when
    everything ran as requested, otherwise a short human-readable
    string the UI surfaces via ``_set_status`` — e.g. the CNN toggle
    was armed but unavailable, or a fit step failed and we degraded.

    Never raises: unwrap/bg-fit/CNN failures are logged and degrade to
    the best result obtained so far (wrapped phase in the worst case)
    so a bad fit never blocks the operator from seeing *a*
    reconstruction.
    """
    method_name = str(params.unwrap_method or "").upper()
    try:
        unwrap_cfg = UnwrapConfig(method=UnwrapMethod[method_name])
    except KeyError:
        unwrap_cfg = UnwrapConfig(method=UnwrapMethod.GRADIENT_INTEGRATION)
    try:
        unwrapped = unwrap_phase_advanced(
            phase_wrapped,
            unwrap_cfg,
            complex_field=complex_field,
            wavelength_m=recon_params.wavelength_m,
            pixel_size_m=recon_params.pixel_size_m,
        )
    except Exception:
        _LOG.warning("reffree: unwrap failed, using wrapped phase",
                     exc_info=True)
        return phase_wrapped, "Reffree: unwrap failed — showing wrapped phase."

    return _reffree_correct_unwrapped(unwrapped, amplitude, params)


def _reffree_correct_unwrapped(
    unwrapped: np.ndarray,
    amplitude: np.ndarray,
    params: ReconParams,
) -> Tuple[np.ndarray, Optional[str]]:
    """Background-fit (+ optional CNN) on an ALREADY-unwrapped phase.

    Split out of :func:`_apply_reffree_correction` so pipelines that run
    their own unwrap (QPI does — with its own config and reflection
    handling) can apply the reference-free correction without a second
    unwrap pass. Same never-raise contract: returns
    ``(corrected_phase, note)``.
    """
    try:
        corrected, _fit = subtract_background(
            unwrapped,
            amplitude=amplitude,
            method=str(params.reffree_bg_method or "zernike"),
            n_terms=int(params.reffree_n_terms),
            polynomial_order=int(params.reffree_bg_order),
        )
    except Exception:
        _LOG.warning("reffree: background fit failed, using unwrapped "
                     "phase uncorrected", exc_info=True)
        return unwrapped, ("Reffree: background fit failed — showing "
                          "unwrapped phase uncorrected.")

    if not params.reffree_cnn:
        return corrected, None
    if not reffree_cnn_available():
        _LOG.info("reffree: CNN toggle armed but unavailable "
                  "(torch missing or no checkpoint) — using classical "
                  "background fit only")
        return corrected, ("Reffree CNN requested but unavailable "
                          "(requires torch + trained model) — used "
                          "classical background fit only.")
    try:
        # Lazy import — recon_dl.inference imports torch at module
        # scope, so this must never execute unless the guard above
        # already confirmed torch is importable.
        from recon_dl.inference import ReffreeReconstructor
        reconstructor = ReffreeReconstructor.from_checkpoint(
            _TRACK_C_CHECKPOINT,
        )
        return reconstructor._cnn_correct(corrected), None
    except Exception:
        _LOG.warning("reffree: CNN correction failed — falling back to "
                     "classical background-fit result", exc_info=True)
        return corrected, ("Reffree CNN correction failed — used "
                          "classical background fit only.")


def _prepare_sample_and_ref_fields(
    hologram_path: Path, params: ReconParams,
) -> Tuple[np.ndarray, Optional[np.ndarray], ReconstructionParams, ReconstructionMethod, Optional[str]]:
    """Like ``_prepare_field`` but returns sample and reference *separately*.

    The 5th element is a non-fatal note (``None`` on success) when reference
    mode was armed but loading/demodulating the reference FAILED and the scan
    falls back to the un-referenced field — surfaced so the depth map isn't
    silently unreferenced (B-110).

    ``_prepare_field`` performs the reference division at the
    pre-propagation stage (sample_demod / ref_demod). That's the right
    operation for single-plane reconstruction but **wrong** for
    propagation-aware pipelines like depth scans: propagation does NOT
    commute with pointwise complex division, so

        propagate(sample/ref, z)  ≠  propagate(sample, z) / propagate(ref, z)

    The right-hand side is the physically correct referenced field at
    plane ``z``. Depth maps want the metric judged on that — for every
    trial z. Returning the two demodulated fields separately lets the
    caller hand both to ``compute_depth_map`` so the kernel propagates
    them independently and divides at each z.

    When the user hasn't loaded a reference, the second tuple element
    is ``None`` and callers fall through to the un-referenced path.

    Phase 3: when ``effective_reference_mode() == "reference_free"``,
    ``ref_field`` is always ``None`` — a numerical background fit
    (Zernike/polynomial on the unwrapped phase) has nothing to do with
    a *reference hologram* division, and depth/autofocus scans have no
    unwrap step to hang the fit off of. This is a conscious scope cut:
    reffree background correction only applies to the single-plane
    reconstruct + QPI paths, not the per-z depth scan metric.
    """
    loaded = load_any(hologram_path)
    raw = np.asarray(loaded.array, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = _preprocess_raw(raw, params)
    offaxis = _offaxis_params(params)
    sample_field, _center = extract_complex_field_offaxis(raw, offaxis)

    ref_field: Optional[np.ndarray] = None
    ref_note: Optional[str] = None
    if (params.effective_reference_mode() == "reference"
            and params.reference_path):
        try:
            ref_loaded = load_any(params.reference_path)
            ref_raw = np.asarray(ref_loaded.array, dtype=np.float32)
            if ref_raw.ndim == 3:
                ref_raw = ref_raw[..., 0]
            ref_raw = _preprocess_raw(ref_raw, params)
            ref_field, _ = extract_complex_field_offaxis(ref_raw, offaxis)
        except Exception as exc:
            _LOG.warning(
                "reference subtract failed (%s) — depth scan will run "
                "on un-referenced field", exc,
            )
            ref_field = None
            ref_note = (
                f"Reference division FAILED ({type(exc).__name__}: {exc}); "
                "the depth map is UNREFERENCED — illumination non-uniformity "
                "was not removed, so absolute heights are unreliable. Check "
                "the reference file path/format.")

    base = ReconstructionParams(
        wavelength_m=float(params.wavelength_nm) * 1e-9,
        pixel_size_m=float(params.effective_pixel_um()) * 1e-6,
        z_m=0.0,
        n=float(params.n_medium),
    )
    method = (ReconstructionMethod.ASM
              if params.method.upper() == "ASM"
              else ReconstructionMethod.FRESNEL)
    return sample_field, ref_field, base, method, ref_note


def _prepare_field(
    hologram_path: Path, params: ReconParams,
    *,
    apply_af_roi: bool = False,
) -> Tuple[np.ndarray, ReconstructionParams, ReconstructionMethod, Optional[str]]:
    loaded = load_any(hologram_path)
    raw = np.asarray(loaded.array, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = _preprocess_raw(raw, params)
    offaxis = _offaxis_params(params)
    # ref_note is non-None when reference mode was armed but the division
    # failed (B-110) — threaded out so autofocus/QPI/depth surface it too,
    # not just reconstruct.
    field, _center, ref_note = _extract_field_with_reference(
        raw, params, offaxis)
    # v2.0.7: optional ROI mask for local autofocus. The full complex
    # field still feeds reconstruction (so the user's display doesn't
    # change), but scan metrics see only pixels inside the rectangle —
    # lets the lab focus on a specific cell when the scene has several
    # at different depths.
    if apply_af_roi and params.af_roi is not None:
        y0, x0, y1, x1 = params.af_roi
        h, w = field.shape
        yy0 = int(max(0, min(1, y0)) * (h - 1))
        xx0 = int(max(0, min(1, x0)) * (w - 1))
        yy1 = int(max(0, min(1, y1)) * (h - 1))
        xx1 = int(max(0, min(1, x1)) * (w - 1))
        yy0, yy1 = sorted((yy0, yy1))
        xx0, xx1 = sorted((xx0, xx1))
        if yy1 > yy0 and xx1 > xx0:
            mask = np.zeros(field.shape, dtype=np.float32)
            mask[yy0:yy1 + 1, xx0:xx1 + 1] = 1.0
            field = field * mask
    # v2.0.3: hand the effective pixel (camera_pixel / M) to the kernel.
    # v1 did this via ReconTab; v2 was missing the magnification widget
    # altogether, so autofocus/QPI/depth were silently running on the
    # raw camera-pixel value — off by M² in z for every microscope setup.
    # v2.0.8: pass ``n_medium`` into the propagation kernel too. v2.0.2
    # hardcoded ``n=1.0`` (vacuum/air), which silently made every
    # aqueous-culture autofocus converge to the wrong z because the
    # effective wavelength in water is λ/n_medium, not λ.
    base = ReconstructionParams(
        wavelength_m=float(params.wavelength_nm) * 1e-9,
        pixel_size_m=float(params.effective_pixel_um()) * 1e-6,
        z_m=0.0,
        n=float(params.n_medium),
    )
    method = (ReconstructionMethod.ASM
              if params.method.upper() == "ASM"
              else ReconstructionMethod.FRESNEL)
    return field, base, method, ref_note


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AutofocusResult:
    best_z_m: float
    score: float
    scanned: int
    runtime_ms: float
    # Non-fatal diagnostic surfaced from the core result (degenerate/flat
    # focus landscape → best_z_m is a best guess, not a real focus).
    warning: Optional[str] = None


@dataclass
class MultiFocusResult:
    candidates: List[Any]  # FocusCandidate from core.autofocus
    runtime_ms: float
    warning: Optional[str] = None   # B-110 reference-division fallback note


@dataclass
class QPIOneShotResult:
    qpi: QPIResult
    runtime_ms: float
    warning: Optional[str] = None   # B-110 reference-division fallback note


@dataclass
class QPIBatchResultWrap:
    entries: List[QPIBatchEntry]
    runtime_ms: float
    warning: Optional[str] = None   # B-110 reference-division fallback note


@dataclass
class DepthMapResultWrap:
    result: DepthMapResult
    clusters: List[ClusterHeight]
    runtime_ms: float
    warning: Optional[str] = None   # B-110 reference-division fallback note


@dataclass
class ReportExportResult:
    path: Path
    runtime_ms: float


@dataclass
class BundleExportResult:
    files: List[Path]
    runtime_ms: float


# ---------------------------------------------------------------------------
# Driver — one thread pool, lots of job types.
# ---------------------------------------------------------------------------

class ScienceDriver:
    """Single-threaded job runner for all non-recon pipeline calls.

    Carries a :class:`threading.Event` so the UI can ``cancel()`` a
    running scan — core scan loops poll ``cancel_check`` between
    propagations and raise :class:`AutofocusCancelled` when the event
    is set. Jobs that finish naturally (not cancelled) emit audit
    records via :func:`_emit_audit`."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="science",
        )
        self._inflight: Optional[Future] = None
        self._lock = threading.Lock()
        # Shared cancellation signal. Each ``_submit`` clears it so a
        # cancel on a completed job doesn't poison the next one.
        self._cancel_event = threading.Event()

    def _cancel_check(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> bool:
        """Request a cancel. Returns True if an in-flight job was
        flagged. Core scan loops observe the flag on their next step
        boundary and raise AutofocusCancelled. Single-call work like
        QPI-at-one-z can't be interrupted mid-flight but its result
        is discarded in ``_dispatch`` below."""
        with self._lock:
            if self._inflight and not self._inflight.done():
                self._cancel_event.set()
                return True
        return False

    def _submit(
        self,
        fn: Callable[[], Any],
        on_result: Callable[[Any], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        with self._lock:
            if self._inflight and not self._inflight.done():
                on_error(ReconError(
                    "Another analysis is running — wait for it to finish."))
                return
            self._cancel_event.clear()
            self._inflight = self._pool.submit(fn)
            self._inflight.add_done_callback(
                lambda f: self._dispatch(f, on_result, on_error),
            )

    def _dispatch(self, fut, on_result, on_error):
        cancelled = self._cancel_event.is_set()
        try:
            result = fut.result()
        except AutofocusCancelled:
            on_error(ReconError("Cancelled."))
            return
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("science worker failed")
            on_error(ReconError(str(exc) or repr(exc)))
            return
        if cancelled:
            # Result arrived after Esc — discard so the UI doesn't
            # paint a "done" state on top of a "cancelled" status.
            on_error(ReconError("Cancelled."))
            return
        on_result(result)

    def shutdown(self) -> None:
        self._cancel_event.set()  # unblock any pending check
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- Autofocus (single best z) ------------------------------------

    def run_autofocus(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_min_mm: float, z_max_mm: float, n_steps: int = 40,
        metric: Optional[FocusMetric] = None,
        sample_id: str = "",
        on_result: Callable[[AutofocusResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        """Dispatches to one of five search algorithms based on
        ``params.af_algorithm`` — v1 had six; v2.0.8 restores the
        five most useful (``zscan``, ``coarse_to_fine``, ``robust``,
        ``adaptive_gradient``, ``adaptive_bracketing``). Each branch
        adapts the core result dataclass into :class:`AutofocusResult`
        so the UI stays algorithm-agnostic. Cancel and audit
        semantics are identical across algorithms."""
        chosen_metric = metric or _metric_from_params(params)
        cancel = self._cancel_check
        algo = str(
            getattr(params, "af_algorithm", "robust") or "robust",
        ).lower()

        def run() -> AutofocusResult:
            import time
            t0 = time.monotonic()
            field, base, method, ref_note = _prepare_field(
                hologram_path, params, apply_af_roi=True,
            )
            z_min_m = float(z_min_mm) * 1e-3
            z_max_m = float(z_max_mm) * 1e-3
            scanned = int(n_steps)
            if algo == "coarse_to_fine":
                core_result = coarse_to_fine_search(
                    field, base, method, chosen_metric,
                    z_min_m=z_min_m, z_max_m=z_max_m,
                    coarse_steps=n_steps,
                    fine_tolerance_m=max(1e-9,
                                         (z_max_m - z_min_m) / 1e4),
                    cancel_check=cancel,
                )
            elif algo == "robust":
                core_result = robust_coarse_to_fine_search(
                    field, base, method, chosen_metric,
                    z_min_m=z_min_m, z_max_m=z_max_m,
                    n_coarse=n_steps, refine_factor=8,
                    smooth_sigma=1.5,
                    cancel_check=cancel,
                )
            elif algo == "adaptive_gradient":
                core_result = adaptive_gradient_search(
                    field, base, method, chosen_metric,
                    z_min_m=z_min_m, z_max_m=z_max_m,
                    max_evaluations=n_steps,
                    cancel_check=cancel,
                )
            elif algo == "adaptive_bracketing":
                core_result = adaptive_bracketing_search(
                    field, base, method, chosen_metric,
                    z_min_m=z_min_m, z_max_m=z_max_m,
                    n_refine_levels=3, refine_divisions=8,
                    smooth_sigma=1.0,
                    max_evaluations=n_steps,
                    cancel_check=cancel,
                )
            elif algo == "adaptive_distance":
                # Auto-range discovery. The core helper starts
                # with a tiny window around z = 0 and expands
                # until the peak lands inside the bracket. User's
                # z_min / z_max widgets are ignored *as bounds*
                # but still used as a ceiling hint — adaptive
                # distance reaches at least as far as whichever
                # endpoint is largest in magnitude, with a 50 mm
                # floor so the algorithm doesn't stall out on a
                # zeroed widget.
                max_range = max(
                    abs(z_max_m), abs(z_min_m), 50e-3,
                )
                core_result = adaptive_distance_search(
                    field, base, method, chosen_metric,
                    initial_range_m=0.5e-3,
                    max_range_m=max_range,
                    expand_factor=2.0,
                    signal_threshold=0.3,
                    max_evaluations=n_steps,
                    cancel_check=cancel,
                )
            else:
                # Linear sweep — v2.0.2 behaviour, still the default.
                zs = list(np.linspace(z_min_m, z_max_m, n_steps))
                core_result = autofocus_zscan(
                    field, base, zs, method, chosen_metric,
                    cancel_check=cancel,
                )
                scanned = len(zs)
            best_z_m = _extract_best_z(core_result)
            best_score = _extract_best_score(core_result)
            evaluations = _extract_evaluations(core_result, scanned)
            runtime_ms = (time.monotonic() - t0) * 1000.0
            _emit_audit(
                "autofocus",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_min_mm=z_min_mm, z_max_mm=z_max_mm,
                    n_steps=n_steps, metric=chosen_metric.name,
                    algorithm=algo,
                ),
                result_summary={
                    "best_z_mm": best_z_m * 1e3,
                    "scanned": evaluations,
                    "runtime_ms": runtime_ms,
                },
            )
            return AutofocusResult(
                best_z_m=best_z_m,
                score=best_score,
                scanned=evaluations,
                runtime_ms=runtime_ms,
                # Flat/non-finite diagnostic for whichever algorithm ran —
                # incl. the default 'robust' (from its coarse landscape), not
                # only the linear sweep (2026-07-08, B-100). Combined with the
                # reference-division fallback note (B-110) so neither masks the
                # other.
                warning=_join_notes(ref_note, _landscape_warning(core_result)),
            )
        self._submit(run, on_result, on_error)

    # ---- Multi-focus candidates ---------------------------------------

    def find_focus_candidates(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_min_mm: float, z_max_mm: float, n_steps: int = 60,
        metric: Optional[FocusMetric] = None,
        min_prominence: float = 0.05,
        sample_id: str = "",
        on_result: Callable[[MultiFocusResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        chosen = metric or _metric_from_params(params)
        cancel = self._cancel_check

        def run() -> MultiFocusResult:
            import time
            t0 = time.monotonic()
            field, base, method, ref_note = _prepare_field(
                hologram_path, params, apply_af_roi=True,
            )
            cands = find_focus_candidates(
                field, base, method,
                z_min_m=z_min_mm * 1e-3, z_max_m=z_max_mm * 1e-3,
                n_steps=n_steps, metric=chosen,
                min_prominence=min_prominence,
                cancel_check=cancel,
            )
            runtime_ms = (time.monotonic() - t0) * 1000.0
            _emit_audit(
                "multi_focus",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_min_mm=z_min_mm, z_max_mm=z_max_mm, n_steps=n_steps,
                    metric=chosen.name, min_prominence=min_prominence,
                ),
                result_summary={
                    "candidates_mm": [c.z_m * 1e3 for c in cands],
                    "runtime_ms": runtime_ms,
                },
            )
            return MultiFocusResult(
                candidates=list(cands),
                runtime_ms=runtime_ms,
                warning=ref_note,
            )
        self._submit(run, on_result, on_error)

    # ---- One-shot QPI --------------------------------------------------

    def run_qpi(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_mm: float,
        n_sample: Optional[float] = None,
        n_medium: Optional[float] = None,
        sample_id: str = "",
        on_result: Callable[[QPIOneShotResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        # Honour explicit args when caller passes them (tests), else
        # pick up whatever ReconParams carries — v2 sidebar writes
        # user-edited values there in v2.0.3.
        ns = float(n_sample) if n_sample is not None else float(params.n_sample)
        nm = float(n_medium) if n_medium is not None else float(params.n_medium)

        def run() -> QPIOneShotResult:
            import time
            t0 = time.monotonic()
            field, base, method, ref_note = _prepare_field(hologram_path, params)
            p = ReconstructionParams(
                wavelength_m=base.wavelength_m,
                pixel_size_m=base.pixel_size_m,
                z_m=z_mm * 1e-3, n=base.n,
            )
            recon = propagate(field, p, method)
            wrapped = np.angle(recon).astype(np.float32, copy=False)
            # v2.0.5: honour the user-selected unwrap method. Falls
            # back to gradient integration if the user typed a value
            # the enum doesn't recognise (e.g. state file from a newer
            # version whose method we don't implement yet).
            method_name = str(params.unwrap_method or "").upper()
            try:
                unwrap_cfg = UnwrapConfig(method=UnwrapMethod[method_name])
            except KeyError:
                unwrap_cfg = UnwrapConfig(
                    method=UnwrapMethod.GRADIENT_INTEGRATION,
                )
            try:
                unwrapped = unwrap_phase_advanced(
                    wrapped,
                    unwrap_cfg,
                    complex_field=recon,
                    wavelength_m=base.wavelength_m,
                    pixel_size_m=base.pixel_size_m,
                )
            except Exception:
                _LOG.warning("qpi: unwrap fallback to wrapped", exc_info=True)
                unwrapped = wrapped
            # Phase 3: reference-free mode must correct the QPI phase
            # too — dry mass / OPD / step height computed on an
            # uncorrected background would silently carry the very
            # aberration the mode exists to remove (2026-07-05 review).
            # Applied on the raw unwrapped phase, before the reflection
            # halving: the fit is linear, so fit-then-halve equals
            # halve-then-fit, and this keeps parity with the
            # reconstruct-path ordering.
            reffree_note = None
            if params.effective_reference_mode() == "reference_free":
                unwrapped, reffree_note = _reffree_correct_unwrapped(
                    unwrapped, np.abs(recon).astype(np.float32), params,
                )
            # v2.0.7: reflection DHM sees the OPL twice (light
            # traverses the surface, reflects, traverses again).
            # Halve the unwrapped phase *before* compute_qpi so every
            # downstream derived quantity (OPD, height, dry mass,
            # step height) reflects the real single-pass value —
            # this is the "3 µm object measured as 6 µm" fix.
            if str(getattr(params, "optical_mode",
                           "transmission")).lower() == "reflection":
                unwrapped = unwrapped * 0.5
            qpi = compute_qpi(
                unwrapped,
                wavelength_m=base.wavelength_m,
                pixel_size_m=base.pixel_size_m,
                mode=QPIMode.BOTH,
                n_sample=ns, n_medium=nm,
                compute_psd=False,
            )
            runtime_ms = (time.monotonic() - t0) * 1000.0
            opd_nm = None
            if qpi.phase_stats is not None:
                opd_nm = qpi.phase_stats.range_nm
            _emit_audit(
                "qpi",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_mm=z_mm, n_sample=ns, n_medium=nm,
                ),
                result_summary={
                    "opd_range_nm": opd_nm,
                    "total_dry_mass_pg": qpi.total_dry_mass_pg,
                    "step_height_m": qpi.step_height_m,
                    "runtime_ms": runtime_ms,
                },
            )
            return QPIOneShotResult(
                qpi=qpi,
                runtime_ms=runtime_ms,
                warning=ref_note,
            )
        self._submit(run, on_result, on_error)

    # ---- QPI batch (one entry per focus candidate) --------------------

    def run_qpi_batch(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_min_mm: float, z_max_mm: float, n_steps: int = 60,
        n_sample: Optional[float] = None, n_medium: Optional[float] = None,
        sample_id: str = "",
        on_result: Callable[[QPIBatchResultWrap], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        ns = float(n_sample) if n_sample is not None else float(params.n_sample)
        nm = float(n_medium) if n_medium is not None else float(params.n_medium)
        af_metric = _metric_from_params(params)
        cancel = self._cancel_check

        def run() -> QPIBatchResultWrap:
            import time
            t0 = time.monotonic()
            field, base, method, ref_note = _prepare_field(hologram_path, params)
            cands = find_focus_candidates(
                field, base, method,
                z_min_m=z_min_mm * 1e-3, z_max_m=z_max_mm * 1e-3,
                n_steps=n_steps, metric=af_metric,
                min_prominence=0.05,
                cancel_check=cancel,
            )
            entries = run_qpi_for_candidates(
                field, base, method, cands,
                n_sample=ns, n_medium=nm, compute_psd=False,
            )
            runtime_ms = (time.monotonic() - t0) * 1000.0
            _emit_audit(
                "qpi_batch",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_min_mm=z_min_mm, z_max_mm=z_max_mm, n_steps=n_steps,
                    n_sample=ns, n_medium=nm,
                ),
                result_summary={
                    "candidates": len(entries),
                    "runtime_ms": runtime_ms,
                },
            )
            return QPIBatchResultWrap(
                entries=list(entries),
                runtime_ms=runtime_ms,
                warning=ref_note,
            )
        self._submit(run, on_result, on_error)

    # ---- Depth map + clusters -----------------------------------------

    def run_depth_map(
        self,
        hologram_path: Path,
        params: ReconParams,
        *,
        z_min_mm: float, z_max_mm: float, n_steps: int = 40,
        window_size: int = 5,
        metric: Optional[FocusMetric] = None,
        sample_id: str = "",
        on_result: Callable[[DepthMapResultWrap], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        chosen = metric or _metric_from_params(params)
        cancel = self._cancel_check

        def run() -> DepthMapResultWrap:
            import time
            t0 = time.monotonic()
            # Depth scans need sample and reference SEPARATELY so the
            # kernel can propagate them independently to each trial z
            # and divide there — propagating a pre-divided field gives
            # mathematically wrong focus metrics. ``_prepare_field``'s
            # division-then-propagate behaviour is fine for single-plane
            # reconstruction but misleads autofocus / depth.
            sample_field, ref_field, base, method, ref_note = (
                _prepare_sample_and_ref_fields(hologram_path, params)
            )
            # ROI mask still applies — but we mask the sample only;
            # reference division below uses the full ref field so the
            # division stays physically meaningful inside the ROI.
            if params.af_roi is not None:
                y0, x0, y1, x1 = params.af_roi
                h, w = sample_field.shape
                yy0 = int(max(0, min(1, y0)) * (h - 1))
                xx0 = int(max(0, min(1, x0)) * (w - 1))
                yy1 = int(max(0, min(1, y1)) * (h - 1))
                xx1 = int(max(0, min(1, x1)) * (w - 1))
                yy0, yy1 = sorted((yy0, yy1))
                xx0, xx1 = sorted((xx0, xx1))
                if yy1 > yy0 and xx1 > xx0:
                    mask = np.zeros(sample_field.shape, dtype=np.float32)
                    mask[yy0:yy1 + 1, xx0:xx1 + 1] = 1.0
                    sample_field = sample_field * mask
            result = compute_depth_map(
                sample_field, base, method,
                z_min_m=z_min_mm * 1e-3, z_max_m=z_max_mm * 1e-3,
                n_steps=n_steps, window_size=window_size, metric=chosen,
                cancel_check=cancel,
                ref_field=ref_field,
            )
            try:
                clusters = segment_depth_clusters(
                    result,
                    confidence_threshold_frac=0.30,
                    min_area_px=40,
                )
            except Exception:
                # Log it — a crashing segmenter must be distinguishable from a
                # genuinely empty scene, both of which record clusters:0
                # (2026-07-08 review).
                _LOG.exception("depth cluster segmentation failed; "
                               "reporting no clusters")
                clusters = []
            runtime_ms = (time.monotonic() - t0) * 1000.0
            _emit_audit(
                "depth_map",
                _params_for_audit(
                    params, hologram=hologram_path, sample_id=sample_id,
                    z_min_mm=z_min_mm, z_max_mm=z_max_mm, n_steps=n_steps,
                    window_size=window_size, metric=chosen.name,
                ),
                result_summary={
                    "z_min_mm_detected": float(result.z_map.min()) * 1e3,
                    "z_max_mm_detected": float(result.z_map.max()) * 1e3,
                    "clusters": len(clusters),
                    "runtime_ms": runtime_ms,
                },
            )
            return DepthMapResultWrap(
                result=result,
                clusters=list(clusters),
                runtime_ms=runtime_ms,
                warning=ref_note,
            )
        self._submit(run, on_result, on_error)

    # ---- Report export ------------------------------------------------

    def export_report(
        self,
        out_path: Path,
        *,
        last_recon_params: dict,
        phase_image: Optional[np.ndarray],
        amplitude_image: Optional[np.ndarray],
        sample_id: str = "",
        on_result: Callable[[ReportExportResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        def run() -> ReportExportResult:
            import time
            t0 = time.monotonic()
            path = generate_html_report(
                output_path=out_path,
                title="DHM Reconstruction Report",
                operator=sample_id or None,
                recon_params=last_recon_params,
                autofocus_summary=None,
                qpi_result=None,
                phase_image=phase_image,
                amplitude_image=amplitude_image,
                height_image=None,
                notes="",
            )
            return ReportExportResult(
                path=Path(path),
                runtime_ms=(time.monotonic() - t0) * 1000.0,
            )
        self._submit(run, on_result, on_error)

    # ---- Tomography bundle export -------------------------------------

    def export_bundle(
        self,
        out_dir: Path,
        *,
        depth_result: DepthMapResult,
        clusters: Sequence,
        qpi_entries: Sequence,
        sample_id: str = "",
        pixel_size_m: Optional[float] = None,
        on_result: Callable[[BundleExportResult], None],
        on_error: Callable[[ReconError], None],
    ) -> None:
        from datetime import datetime

        def run() -> BundleExportResult:
            import time
            t0 = time.monotonic()
            base_name = f"tomography_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            paths = write_tomography_bundle(
                out_dir, depth_result, clusters, qpi_entries,
                base_name=base_name,
                sample_id=sample_id,
                pixel_size_m=pixel_size_m,
            )
            return BundleExportResult(
                files=list(paths),
                runtime_ms=(time.monotonic() - t0) * 1000.0,
            )
        self._submit(run, on_result, on_error)


__all__ = [
    "ScienceDriver",
    "AutofocusResult",
    "MultiFocusResult",
    "QPIOneShotResult",
    "QPIBatchResultWrap",
    "DepthMapResultWrap",
    "ReportExportResult",
    "BundleExportResult",
    "write_qpi_csv",
    "write_qpi_batch_csv",
    "available_focus_metrics",
    "available_autofocus_algorithms",
    "af_algorithm_input_profile",
]
