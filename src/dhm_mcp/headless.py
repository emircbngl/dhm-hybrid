"""Qt-free ``ToolContext`` driver for the ``dhm-mcp`` server.

:class:`HeadlessSession` plays the role the GUI panel plays for the
in-app copilot (``src/gui/panels/ai_panel.py``): it owns the recon
parameter dict, the last-computed arrays, a virtual stage/shutter/LED,
and a :meth:`build_context` factory that wires all of that into a
:class:`core.ai.tools.ToolContext`. The *same* ``core.ai.tool_impls``
handlers and ``ToolRegistry`` run against it — dispatch (schema
validation, numeric clamping, confirm-gate, audit) is identical to the
in-app path, only the ``ToolContext`` plumbing differs.

No Qt. No GUI thread hop — every ``invoke_*`` call runs the real
``core`` pipeline in-line and returns once it's done, which is exactly
what an MCP tool call wants (synchronous request/response).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.ai.security import ALLOWED_HOLOGRAM_EXTENSIONS, validate_path
from core.ai.tools import ToolContext
from core.audit import AuditLog, get_audit_log
from core.errors import ErrorCenter, get_error_center
from core.pipelines.reffree_hybrid import LAB_OPTICS, OpticalConfig
from core.sample_map import SampleMap
from core.stage import MockStage

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings / confirm stand-ins
# ---------------------------------------------------------------------------

@dataclass
class _HeadlessSettings:
    """Minimal stand-in for the app's ``AISettings`` the handlers read.

    ``restrict_to_home`` mirrors the GUI default (True) — an MCP client
    is just as capable of a hallucinated path as the local LLM, so the
    same guardrail applies. ``audit_redact_for_llm`` also matches the
    GUI default.
    """
    restrict_to_home: bool = True
    audit_redact_for_llm: bool = True


def _default_confirm(tool_name: str, args: dict) -> bool:
    """MCP has no modal dialog — default-deny irreversible tools.

    Set ``DHM_MCP_ALLOW_IRREVERSIBLE=1`` to opt into auto-approval
    (e.g. a scripted/CI run that intentionally drives real hardware
    moves). The safe default is rejection; the audit log still
    records the attempt via the registry's ``ai.tool.*.start`` entry.
    """
    allowed = os.environ.get("DHM_MCP_ALLOW_IRREVERSIBLE", "") == "1"
    if not allowed:
        _LOG.info("dhm-mcp: confirm gate rejected %r (set "
                  "DHM_MCP_ALLOW_IRREVERSIBLE=1 to auto-approve)", tool_name)
    return allowed


# ---------------------------------------------------------------------------
# HeadlessSession
# ---------------------------------------------------------------------------

_DEFAULT_RECON_PARAMS: dict = {
    "wavelength_nm": LAB_OPTICS.wavelength_m * 1e9,
    "pixel_um": LAB_OPTICS.effective_pixel_m * 1e6,
    "z_mm": 0.0,
    "n_medium": LAB_OPTICS.n_medium,
    "mask_radius": LAB_OPTICS.mask_radius,
    "subtract_mean": True,
    "hann_window": False,
}


@dataclass
class HeadlessSession:
    """Owns recon parameters + the latest arrays for one MCP session.

    A fresh instance is created once per server process (``server.py``
    holds a module-level singleton) so state persists across tool
    calls within one MCP client connection, exactly like the GUI panel
    persists state across chat turns.
    """

    recon_params: dict = field(default_factory=lambda: dict(_DEFAULT_RECON_PARAMS))
    # bg_order is intentionally NOT pre-seeded: subtract_background has two
    # independent knobs (n_terms for zernike, polynomial_order for polynomial)
    # with different sensible defaults (15 vs 4). A single shared default would
    # feed a degenerate order to whichever method it doesn't suit. invoke_recon
    # resolves the method-appropriate default when bg_order is unset.
    reference_mode: dict = field(default_factory=lambda: {"mode": "off",
                                                          "bg_method": "zernike"})

    # Latest arrays — replaced wholesale, never mutated in place (same
    # contract the GUI panel's ``_get_last_field`` documents).
    raw: Optional[np.ndarray] = None
    reference_raw: Optional[np.ndarray] = None
    demod_complex: Optional[np.ndarray] = None
    recon_complex: Optional[np.ndarray] = None
    phase_unwrapped: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None

    loaded_path: Optional[str] = None

    last_recon_summary: Optional[dict] = None
    last_af_summary: Optional[dict] = None
    last_qpi_summary: Optional[dict] = None
    last_depth_summary: Optional[dict] = None

    stage: MockStage = field(default_factory=MockStage)
    sample_map: SampleMap = field(default_factory=SampleMap)
    audit: AuditLog = field(default_factory=get_audit_log)
    error_center: ErrorCenter = field(default_factory=get_error_center)
    settings: _HeadlessSettings = field(default_factory=_HeadlessSettings)

    # ------------------------------------------------------------------
    # Optical config derived from recon_params (kept in sync by
    # set_recon_param / set_reference_mode).
    # ------------------------------------------------------------------

    def _optical_config(self) -> OpticalConfig:
        p = self.recon_params
        return OpticalConfig(
            wavelength_m=float(p.get("wavelength_nm", 632.8)) * 1e-9,
            effective_pixel_m=float(p.get("pixel_um", 0.088)) * 1e-6,
            n_medium=float(p.get("n_medium", 1.337)),
            mask_radius=int(p.get("mask_radius", 80)),
        )

    # ------------------------------------------------------------------
    # State snapshot — same shape core.ai.context.StateSnapshot exposes.
    # ------------------------------------------------------------------

    def state(self) -> "_Snapshot":
        shape = None
        dtype = None
        if self.raw is not None:
            shape = tuple(int(s) for s in self.raw.shape[:2])
            dtype = str(self.raw.dtype)
        return _Snapshot(
            loaded_path=self.loaded_path,
            loaded_shape=shape,
            loaded_dtype=dtype,
            recon_params=dict(self.recon_params),
            autofocus_params={},
            qpi_params={},
            last_recon_summary=self.last_recon_summary,
            last_af_summary=self.last_af_summary,
            last_qpi_summary=self.last_qpi_summary,
            last_depth_summary=self.last_depth_summary,
            stage_position_mm=self.stage.get_position(),
            audit_tail=tuple(self.audit.entries_today(20)),
        )

    # ------------------------------------------------------------------
    # Pipeline drivers (ctx.invoke_* / ctx.load_hologram / ctx.set_recon_param)
    # ------------------------------------------------------------------

    def _invalidate_derived(self) -> None:
        """Drop every cached derived field.

        Single invalidation contract for ALL mutators that change what
        invoke_recon would produce (hologram, recon params, reference mode,
        autofocus z) — otherwise inspect_* reports the OLD reconstruction as
        if it reflected the new state (2026-07-06 review, twice: the first
        pass fixed load/set_recon_param and missed set_reference_mode +
        invoke_autofocus's direct z write)."""
        self.demod_complex = None
        self.recon_complex = None
        self.phase_unwrapped = None
        self.depth = None

    def load_hologram(self, path: str) -> dict:
        from core.ingestion import load_any
        loaded = load_any(path)
        self.raw = np.asarray(loaded.array)
        self.loaded_path = str(loaded.path)
        self._invalidate_derived()
        shape = list(self.raw.shape[:2])
        return {"path": self.loaded_path, "shape": shape,
                "dtype": str(self.raw.dtype)}

    def set_recon_param(self, params: dict) -> dict:
        if not params:
            return {"updated": {}, "current": dict(self.recon_params)}
        self.recon_params.update(params)
        # Recon params feed the optical config (wavelength/pixel/z/...), which
        # drives demodulation and propagation — see _invalidate_derived.
        self._invalidate_derived()
        return {"updated": dict(params), "current": dict(self.recon_params)}

    def invoke_recon(self, args: dict) -> dict:
        from core.background_phase import subtract_background
        from core.phase_unwrap import unwrap_phase_advanced
        from core.pipelines.reffree_hybrid import (
            demodulate, propagate_field, safe_reference_divide,
        )

        if self.raw is None:
            return {"error": "no hologram loaded; call load_hologram first"}

        cfg = self._optical_config()
        fc = demodulate(self.raw, cfg)
        self.demod_complex = fc

        z_mm = float(self.recon_params.get("z_mm", 0.0))
        z_m = z_mm * 1e-3

        mode = str(self.reference_mode.get("mode", "off"))
        ref_fc = None
        if mode == "reference" and self.reference_raw is not None:
            ref_fc = demodulate(self.reference_raw, cfg)

        sample = propagate_field(fc, z_m, cfg)
        if ref_fc is not None:
            ref_at_z = propagate_field(ref_fc, z_m, cfg)
            field = safe_reference_divide(sample, ref_at_z)
        else:
            field = sample

        amp = np.abs(field).astype(np.float32)
        phase = np.angle(field).astype(np.float32)
        unwrapped = unwrap_phase_advanced(
            phase, complex_field=field,
            wavelength_m=cfg.wavelength_m, pixel_size_m=cfg.effective_pixel_m,
        ).astype(np.float32)

        if mode == "reference_free":
            bg_method = str(self.reference_mode.get("bg_method", "zernike"))
            bg_order = self.reference_mode.get("bg_order")
            # Route the single bg_order knob to the parameter the active method
            # actually consumes, and give the other its own correct default —
            # not the same value (2026-07-06 review).
            if bg_method == "polynomial":
                poly_order = int(bg_order) if bg_order is not None else 4
                unwrapped, _ = subtract_background(
                    unwrapped, amplitude=amp, method="polynomial",
                    n_terms=15, polynomial_order=poly_order,
                )
            else:
                n_terms = int(bg_order) if bg_order is not None else 15
                unwrapped, _ = subtract_background(
                    unwrapped, amplitude=amp, method="zernike",
                    n_terms=n_terms, polynomial_order=4,
                )
            unwrapped = unwrapped.astype(np.float32)

        self.recon_complex = field.astype(np.complex64)
        self.phase_unwrapped = unwrapped

        summary = {
            "shape": list(field.shape),
            "z_mm": z_mm,
            "reference_mode": mode,
            "phase_mean_rad": float(np.mean(unwrapped)),
            "phase_std_rad": float(np.std(unwrapped)),
            # Unsuffixed aliases: the GUI emits phase_mean/phase_std and
            # record_timelapse's extractor only reads phase_std — without
            # these, every MCP timelapse frame silently lost the phase-drift
            # signal that is the tool's purpose (2026-07-08 review).
            "phase_mean": float(np.mean(unwrapped)),
            "phase_std": float(np.std(unwrapped)),
            "amplitude_mean": float(np.mean(amp)),
        }
        self.last_recon_summary = summary
        return {"submitted": True, "summary": summary}

    def invoke_autofocus(self, args: dict) -> dict:
        from core.autofocus import FocusMetric
        from core.autofocus.search_classic import robust_coarse_to_fine_search
        from core.pipelines.reffree_hybrid import demodulate
        from core.reconstruction import ReconstructionMethod, ReconstructionParams

        if self.raw is None:
            return {"error": "no hologram loaded; call load_hologram first"}

        cfg = self._optical_config()
        fc = self.demod_complex if self.demod_complex is not None \
            else demodulate(self.raw, cfg)
        self.demod_complex = fc

        z_min_m = float(args.get("z_min_mm", -cfg.z_range_mm)) * 1e-3
        z_max_m = float(args.get("z_max_mm", cfg.z_range_mm)) * 1e-3
        # LAPLACIAN_VARIANCE = the app-wide default metric (GUI parity).
        metric_name = str(args.get("metric", "LAPLACIAN_VARIANCE"))
        try:
            metric = FocusMetric[metric_name]
        except KeyError:
            metric = FocusMetric.LAPLACIAN_VARIANCE
        n_steps = int(args.get("n_steps", cfg.z_steps))

        base = ReconstructionParams(
            wavelength_m=cfg.wavelength_m, pixel_size_m=cfg.effective_pixel_m,
            z_m=0.0, n=cfg.n_medium,
        )
        # robust: settled 2026-07-06 on the 9-scene real-lab benchmark —
        # the only top performer across metrics (adaptive_gradient, used
        # here before, is unreliable with entropy-like flat landscapes).
        res = robust_coarse_to_fine_search(
            field=fc, base_params=base, method=ReconstructionMethod.ASM,
            metric=metric, z_min_m=z_min_m, z_max_m=z_max_m,
            n_coarse=n_steps, refine_factor=8, smooth_sigma=1.5,
        )
        best_z_mm = float(res.best_z_m) * 1e3
        self.recon_params["z_mm"] = best_z_mm
        # Direct z write bypasses set_recon_param — apply the same
        # invalidation so inspect_* can't serve the old-z reconstruction
        # while the snapshot advertises the new z (2026-07-06 review #2).
        self._invalidate_derived()
        summary = {
            "best_z_mm": best_z_mm,
            "metric": metric.value,
            "best_score": float(res.best_score),
        }
        self.last_af_summary = summary
        return {"submitted": True, "summary": summary}

    def invoke_qpi(self, args: dict) -> dict:
        from core import qpi as qpi_mod

        if self.phase_unwrapped is None:
            return {"error": "no phase_unwrapped available yet — run "
                              "reconstruction first"}

        cfg = self._optical_config()
        n_sample = float(args.get("n_sample", 1.38))
        n_medium = float(args.get("n_medium", cfg.n_medium))

        result = qpi_mod.compute_qpi(
            self.phase_unwrapped, cfg.wavelength_m, cfg.effective_pixel_m,
            n_sample=n_sample, n_medium=n_medium, compute_psd=False,
        )
        summary: dict = {
            "opd_max_nm": float(np.max(result.opd_nm)),
            "opd_min_nm": float(np.min(result.opd_nm)),
        }
        if result.phase_stats is not None:
            summary["phase_std_rad"] = float(result.phase_stats.std_rad)
            summary["opd_range_nm"] = float(result.phase_stats.range_nm)
        if result.total_dry_mass_pg is not None:
            summary["total_dry_mass_pg"] = float(result.total_dry_mass_pg)
        self.last_qpi_summary = summary
        return {"submitted": True, "summary": summary}

    def invoke_depth_map(self, args: dict) -> dict:
        from core.autofocus.metrics import FocusMetric
        from core.depth_map import compute_depth_map
        from core.pipelines.reffree_hybrid import demodulate
        from core.reconstruction import ReconstructionMethod, ReconstructionParams

        if self.raw is None:
            return {"error": "no hologram loaded; call load_hologram first"}

        cfg = self._optical_config()
        fc = self.demod_complex if self.demod_complex is not None \
            else demodulate(self.raw, cfg)
        self.demod_complex = fc

        z_min_m = float(args["z_min_mm"]) * 1e-3
        z_max_m = float(args["z_max_mm"]) * 1e-3
        metric_name = str(args.get("metric", "LAPLACIAN_VARIANCE"))
        try:
            metric = FocusMetric[metric_name]
        except KeyError:
            metric = FocusMetric.LAPLACIAN_VARIANCE
        n_steps = int(args.get("n_steps", 40))
        window_size = int(args.get("window_size", 5))

        base = ReconstructionParams(
            wavelength_m=cfg.wavelength_m, pixel_size_m=cfg.effective_pixel_m,
            z_m=0.0, n=cfg.n_medium,
        )
        result = compute_depth_map(
            fc, base, ReconstructionMethod.ASM, z_min_m, z_max_m,
            n_steps=n_steps, window_size=window_size, metric=metric,
        )
        self.depth = np.asarray(result.z_map, dtype=np.float32)
        summary = {
            "z_map_shape": list(self.depth.shape),
            "z_mean_mm": float(np.mean(result.z_map)) * 1e3,
            "z_std_mm": float(np.std(result.z_map)) * 1e3,
            "metric": metric.value,
        }
        self.last_depth_summary = summary
        return {"submitted": True, "summary": summary}

    def invoke_find_focus_candidates(self, args: dict) -> dict:
        from core.autofocus.analysis import find_focus_candidates
        from core.autofocus.metrics import FocusMetric
        from core.pipelines.reffree_hybrid import demodulate
        from core.reconstruction import ReconstructionMethod, ReconstructionParams

        if self.raw is None:
            return {"error": "no hologram loaded; call load_hologram first"}

        cfg = self._optical_config()
        fc = self.demod_complex if self.demod_complex is not None \
            else demodulate(self.raw, cfg)
        self.demod_complex = fc

        z_min_m = float(args["z_min_mm"]) * 1e-3
        z_max_m = float(args["z_max_mm"]) * 1e-3
        metric_name = str(args.get("metric", "ENTROPY"))
        try:
            metric = FocusMetric[metric_name]
        except KeyError:
            metric = FocusMetric.ENTROPY
        n_steps = int(args.get("n_steps", 60))
        top_k = int(args.get("top_k", 5))

        base = ReconstructionParams(
            wavelength_m=cfg.wavelength_m, pixel_size_m=cfg.effective_pixel_m,
            z_m=0.0, n=cfg.n_medium,
        )
        candidates = find_focus_candidates(
            fc, base, ReconstructionMethod.ASM, z_min_m, z_max_m,
            n_steps=n_steps, metric=metric, max_candidates=top_k,
        )
        return {
            "submitted": True,
            "candidates": [
                {"z_mm": float(c.z_m) * 1e3, "score": float(c.score),
                 "prominence": float(c.prominence), "rank": int(c.rank)}
                for c in candidates
            ],
        }

    # ------------------------------------------------------------------
    # Capture / sample-map / time-lapse (sprint-2 tools)
    # ------------------------------------------------------------------

    def capture_frame(self) -> Any:
        """No live camera headless — fall back to the loaded raw
        hologram, exactly like the GUI panel's v1 default."""
        return self.raw

    @staticmethod
    def measure_sharpness(frame: Any) -> float:
        if frame is None:
            return -float("inf")
        a = np.asarray(frame)
        if a.ndim == 0 or a.size == 0:
            return -float("inf")
        amp = np.abs(a).astype(np.float32, copy=False)
        if amp.shape[0] < 3 or amp.shape[1] < 3:
            return -float("inf")
        lap = (
            amp[1:-1, :-2] + amp[1:-1, 2:]
            + amp[:-2, 1:-1] + amp[2:, 1:-1]
            - 4.0 * amp[1:-1, 1:-1]
        )
        v = float(np.var(lap))
        return v if np.isfinite(v) else -float("inf")

    def persist_sample_map(self) -> Optional[str]:
        out_dir = Path.home() / ".dhm-reconstruction" / "sample_maps"
        out_path = out_dir / f"{self.sample_map.sample_id or 'sample'}.json"
        try:
            self.sample_map.save(out_path)
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            _LOG.warning("dhm-mcp: failed to persist sample map: %s", exc)
            return None
        return str(out_path)

    def capture_and_process(self, args: dict) -> dict:
        """Minimal single-frame capture + optional recon/QPI, for
        record_timelapse / map_sample_grid. Segmentation ('segment':
        True) is intentionally not wired headless — no camera means no
        new cells to find beyond what a full reconstruction already
        reports; callers relying on map_sample_grid's cell list will
        just see an empty per-step cell set in headless mode."""
        out: dict = {}
        if bool(args.get("run_recon", True)) and self.raw is not None:
            r = self.invoke_recon({})
            if isinstance(r, dict) and "summary" in r:
                out.update(r["summary"])
        if bool(args.get("run_qpi", False)) and self.phase_unwrapped is not None:
            q = self.invoke_qpi({})
            if isinstance(q, dict) and "summary" in q:
                out.update(q["summary"])
        out.setdefault("cells", [])
        return out

    # ------------------------------------------------------------------
    # Observation hooks (get_last_field / set_reference_mode)
    # ------------------------------------------------------------------

    def get_last_field(self, target: str) -> Optional[np.ndarray]:
        if target == "recon_complex":
            return self.recon_complex
        if target == "phase_unwrapped":
            return self.phase_unwrapped
        if target == "raw":
            return self.raw
        if target == "depth":
            return self.depth
        return None

    def set_reference_mode(self, args: dict) -> dict:
        mode = str(args.get("mode", "")).strip().lower()
        if mode not in ("off", "reference", "reference_free"):
            return {"error": f"unknown mode {mode!r}"}
        # A reference hologram can be supplied inline (headless has no file
        # dialog). Without this, reference_raw stayed None forever and
        # "reference" mode was unreachable over MCP (2026-07-06 review). The
        # path is already validated by the tool layer (validate_path).
        ref_path = args.get("reference_path")
        if ref_path:
            from core.ingestion import load_any
            loaded = load_any(str(ref_path))
            self.reference_raw = np.asarray(loaded.array)
        if mode == "reference" and self.reference_raw is None:
            return {"error": "load a reference hologram first "
                             "(pass reference_path)"}
        self.reference_mode["mode"] = mode
        if "bg_method" in args:
            self.reference_mode["bg_method"] = str(args["bg_method"])
        if "bg_order" in args:
            self.reference_mode["bg_order"] = int(args["bg_order"])
        # Mode/bg knobs change what invoke_recon produces (reference division
        # vs background fit) just like recon params do — same invalidation
        # contract (2026-07-06 review #2).
        self._invalidate_derived()
        return {"ok": True, "mode": mode,
                "bg_method": self.reference_mode.get("bg_method"),
                "bg_order": self.reference_mode.get("bg_order"),
                "reference_loaded": self.reference_raw is not None}

    # ------------------------------------------------------------------
    # ToolContext factory
    # ------------------------------------------------------------------

    def build_context(self) -> ToolContext:
        """Build a fresh :class:`ToolContext` bound to this session.

        Device hooks (shutter/led/orchestrator) stay ``None`` — headless
        v1 has no camera/illumination hardware to drive; ``list_devices``
        still works (it queries the registry, not these hooks) and the
        shutter/LED/acquire_grid tools return the standard "device not
        configured" error, same as an in-app session without hardware
        wired.
        """
        return ToolContext(
            state=self.state,
            last_recon_summary=lambda: self.last_recon_summary,
            last_af_summary=lambda: self.last_af_summary,
            last_qpi_summary=lambda: self.last_qpi_summary,
            last_depth_summary=lambda: self.last_depth_summary,
            audit_tail=lambda limit: list(self.audit.entries_today(limit)),
            load_hologram=self.load_hologram,
            set_recon_param=self.set_recon_param,
            invoke_recon=self.invoke_recon,
            invoke_autofocus=self.invoke_autofocus,
            invoke_qpi=self.invoke_qpi,
            invoke_depth_map=self.invoke_depth_map,
            invoke_find_focus_candidates=self.invoke_find_focus_candidates,
            stage=self.stage,
            capture_frame=self.capture_frame,
            sample_map=self.sample_map,
            measure_sharpness=self.measure_sharpness,
            persist_sample_map=self.persist_sample_map,
            capture_and_process=self.capture_and_process,
            audit=self.audit,
            error_center=self.error_center,
            confirm=_default_confirm,
            settings=self.settings,
            is_cancelled=None,
            shutter=None,
            led=None,
            orchestrator=None,
            get_last_field=self.get_last_field,
            set_reference_mode=self.set_reference_mode,
        )


@dataclass(frozen=True)
class _Snapshot:
    """Local stand-in for ``core.ai.context.StateSnapshot``.

    Same field names/shape (``as_dict`` included) so ``get_state`` /
    ``inspect_phase``'s ``recon_params`` lookup behave identically to
    the in-app snapshot. Defined here rather than imported to avoid a
    hard dependency on the exact frozen-dataclass identity — handlers
    only touch attributes, never ``isinstance``.
    """
    loaded_path: Optional[str] = None
    loaded_shape: Optional[tuple] = None
    loaded_dtype: Optional[str] = None
    recon_params: dict = field(default_factory=dict)
    autofocus_params: dict = field(default_factory=dict)
    qpi_params: dict = field(default_factory=dict)
    last_recon_summary: Optional[dict] = None
    last_af_summary: Optional[dict] = None
    last_qpi_summary: Optional[dict] = None
    last_depth_summary: Optional[dict] = None
    stage_position_mm: Optional[tuple] = None
    audit_tail: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "loaded_path": self.loaded_path,
            "loaded_shape": list(self.loaded_shape) if self.loaded_shape else None,
            "loaded_dtype": self.loaded_dtype,
            "recon_params": self.recon_params,
            "autofocus_params": self.autofocus_params,
            "qpi_params": self.qpi_params,
            "last_recon_summary": self.last_recon_summary,
            "last_af_summary": self.last_af_summary,
            "last_qpi_summary": self.last_qpi_summary,
            "last_depth_summary": self.last_depth_summary,
            "stage_position_mm": list(self.stage_position_mm)
            if self.stage_position_mm else None,
            "audit_tail": list(self.audit_tail),
        }


__all__ = [
    "HeadlessSession",
]
