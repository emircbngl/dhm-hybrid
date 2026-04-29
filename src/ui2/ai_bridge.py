"""Bridge between v2 ``DhmApp`` and the ``core.ai`` agent — v2.1.z+.

The AI agent's :class:`core.ai.tools.ToolContext` expects each
``invoke_*`` callable to **block** until the underlying pipeline
finishes. v2's drivers are async (callback-based), so the AI
worker thread would race the cache reads without a sync wrapper.

This module provides:

* :func:`make_v2_tool_context` — build a :class:`ToolContext`
  from a live :class:`DhmApp`. Adapter functions wrap each
  driver submit with a :class:`threading.Event` that the
  callback signals on completion.
* :class:`PendingMailbox` — single-slot mailbox the v2 drivers
  already use; we reuse the pattern for AI tool calls.

Why not just import v1's panel
------------------------------
v1's ``ToolContext`` builder bakes Qt main-window callable's
into closures. We can't reuse those without dragging Qt into
v2. Re-implementing here keeps v2 framework-agnostic.

The bridge does NOT spawn a window — only the data layer. The
DPG window lives in ``ui2/ai_panel.py``.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


_LOG = logging.getLogger(__name__)


# Hard timeout for any blocking pipeline call (autofocus on a
# 2048² hologram can take ~5 s; we cap at 5 minutes for safety).
_HARD_TIMEOUT_S = 300.0


def _wait(event: threading.Event,
          result_holder: dict,
          timeout_s: float = _HARD_TIMEOUT_S) -> dict:
    """Block until event fires or timeout. Returns the dict
    populated by the result/error callback, or a synthetic
    ``{"error": "timeout"}`` if the wait expired."""
    if event.wait(timeout=timeout_s):
        return result_holder.get("payload") or {}
    return {"error": f"timeout after {timeout_s:.0f}s"}


def _summarise_recon(result: Any) -> dict:
    """Extract LLM-friendly scalars from a ``ReconResult``-like
    object. Falls back to ``{"ok": True}`` when introspection
    fails — better than raising and breaking the agent loop."""
    out: dict = {"ok": True}
    try:
        if hasattr(result, "phase"):
            phase = np.asarray(result.phase)
            out["phase_std"] = float(np.std(phase))
            out["shape"] = list(phase.shape)
        if hasattr(result, "z_m"):
            out["z_mm"] = float(result.z_m) * 1e3
        if hasattr(result, "params"):
            try:
                out["wavelength_nm"] = float(result.params.wavelength_m) * 1e9
            except Exception:
                pass
    except Exception:  # noqa: BLE001
        _LOG.debug("_summarise_recon: introspection failed",
                   exc_info=True)
    return out


def _summarise_autofocus(result: Any) -> dict:
    out: dict = {}
    for attr, key in (("best_z_m", "z_mm"),
                      ("score", "score"),
                      ("scanned", "evaluations"),
                      ("runtime_ms", "runtime_ms")):
        try:
            v = getattr(result, attr, None)
            if v is None:
                continue
            if attr == "best_z_m":
                out[key] = float(v) * 1e3
            else:
                out[key] = float(v) if isinstance(v, (int, float)) \
                    else v
        except Exception:
            continue
    out.setdefault("ok", True)
    return out


def _summarise_qpi(result: Any) -> dict:
    qpi = getattr(result, "qpi", None) or result
    out: dict = {"ok": True}
    try:
        if hasattr(qpi, "phase_stats"):
            ps = qpi.phase_stats
            if ps is not None and hasattr(ps, "range_nm"):
                out["opd_range_nm"] = float(ps.range_nm)
        if hasattr(qpi, "total_dry_mass_pg"):
            v = qpi.total_dry_mass_pg
            if v is not None:
                out["total_dry_mass_pg"] = float(v)
        if hasattr(qpi, "step_height_m"):
            v = qpi.step_height_m
            if v is not None:
                out["step_height_nm"] = float(v) * 1e9
    except Exception:
        _LOG.debug("_summarise_qpi: introspection failed", exc_info=True)
    return out


def _summarise_depth(result: Any) -> dict:
    out: dict = {"ok": True}
    try:
        zmap = getattr(getattr(result, "result", None), "z_map", None)
        if zmap is not None:
            out["z_min_mm"] = float(np.min(zmap)) * 1e3
            out["z_max_mm"] = float(np.max(zmap)) * 1e3
        clusters = getattr(result, "clusters", None) or []
        out["cluster_count"] = len(clusters)
    except Exception:
        _LOG.debug("_summarise_depth: introspection failed", exc_info=True)
    return out


def _summarise_focus_candidates(result: Any) -> dict:
    out: dict = {"ok": True}
    try:
        cands = getattr(result, "candidates", None) or []
        out["candidate_count"] = len(cands)
        out["candidates"] = [
            {"z_mm": float(c.z_m) * 1e3,
             "score": float(c.score),
             "prominence": float(c.prominence)}
            for c in cands[:20]
        ]
    except Exception:
        _LOG.debug("_summarise_focus: introspection failed",
                   exc_info=True)
    return out


def make_v2_tool_context(app: Any) -> Any:
    """Build a :class:`core.ai.tools.ToolContext` against the
    live :class:`ui2.app.DhmApp` instance.

    Parameters
    ----------
    app
        A constructed :class:`DhmApp`. Must have its drivers
        wired (``app._driver``, ``app._science``) and the v2.1.z
        device hooks installed (``app._v21z_device_hooks`` —
        otherwise mock devices are constructed on-demand).

    Returns
    -------
    ToolContext
        Ready to feed into ``Agent(tool_registry, client, ctx)``.
    """
    from core.ai.tools import ToolContext
    from core.audit import get_audit_log
    from core.errors import get_error_center
    from core.sample_map import SampleMap

    # Ensure device hooks exist — DhmApp's AI panel-side ctor
    # populates these but the bridge can be called outside that
    # path (CLI smoke, tests).
    if not hasattr(app, "_v21z_device_hooks"):
        try:
            from core.devices import make_device
            shutter = make_device("mock_shutter")
            led = make_device("mock_led")
            app._v21z_device_hooks = (shutter, led, None)
        except Exception:
            app._v21z_device_hooks = (None, None, None)

    shutter, led, _orchestrator = app._v21z_device_hooks
    sample_map = getattr(app, "_ai_sample_map", None) or SampleMap()
    app._ai_sample_map = sample_map  # cache for later turns

    # ── State snapshot ───────────────────────────────────────
    def _state() -> Any:
        from core.ai.context import StateSnapshot
        loaded = getattr(app, "_current_hologram", None)
        last = getattr(app, "_last_recon", None)
        shape = None
        try:
            if last is not None and hasattr(last, "phase"):
                shape = tuple(int(x) for x in last.phase.shape)
        except Exception:
            shape = None
        return StateSnapshot(
            loaded_path=str(loaded) if loaded else None,
            loaded_shape=shape,
            loaded_dtype=None,
            params=_snapshot_params(app),
            recon_summary=getattr(app, "_ai_cache_recon", None),
            af_summary=getattr(app, "_ai_cache_af", None),
            qpi_summary=getattr(app, "_ai_cache_qpi", None),
            depth_summary=getattr(app, "_ai_cache_depth", None),
        )

    def _snapshot_params(app) -> dict:
        p = app._params
        # Compact dict the LLM can read.
        return {
            "wavelength_nm": float(p.wavelength_nm),
            "pixel_um": float(p.pixel_um),
            "z_mm": float(p.z_mm),
            "n_medium": float(p.n_medium),
            "mask_radius": int(p.mask_radius),
            "method": str(p.method),
            "magnification": float(p.magnification),
            "autofocus_metric": str(p.autofocus_metric),
            "af_z_min_mm": float(p.af_z_min_mm),
            "af_z_max_mm": float(p.af_z_max_mm),
            "af_n_steps": int(p.af_n_steps),
        }

    # ── Sync pipeline drivers ────────────────────────────────
    def _load_hologram(path_str: str) -> dict:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            return {"error": f"file not found: {path}"}
        try:
            app._load_hologram(path)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"load failed: {exc}"}
        return {
            "loaded": True,
            "path": str(path),
        }

    def _set_recon_param(updates: dict) -> dict:
        """Mutate ReconParams in place + push to DPG widgets if
        possible. Returns the new full param snapshot."""
        try:
            for k, v in updates.items():
                if hasattr(app._params, k):
                    setattr(app._params, k, v)
            # Best-effort widget refresh — non-fatal if DPG not up.
            try:
                if hasattr(app, "_refresh_param_widgets"):
                    app._refresh_param_widgets()
            except Exception:
                pass
            app._mark_dirty()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"set_recon_param failed: {exc}"}
        return {"updated": dict(updates),
                "current": _snapshot_params(app)}

    def _invoke_recon(_args: dict) -> dict:
        """Submit a reconstruction + block until result. The
        v2 driver is single-threaded so a sequential AI turn
        never collides with the GUI's own pipeline."""
        if app._current_hologram is None:
            return {"error": "no hologram loaded"}
        done = threading.Event()
        holder: dict = {}

        def _ok(result):
            holder["payload"] = _summarise_recon(result)
            app._ai_cache_recon = holder["payload"]
            # Also flow through the GUI's normal post-result path
            # so the UI repaints. Defer if mailbox-only is desired.
            try:
                app._post_result(result)
            except Exception:
                pass
            done.set()

        def _err(err):
            holder["payload"] = {"error": str(err)}
            done.set()

        try:
            app._driver.submit(
                app._current_hologram, app._params,
                sample_id=app._sample_id,
                on_result=_ok, on_error=_err,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"submit failed: {exc}"}
        return _wait(done, holder)

    def _invoke_autofocus(args: dict) -> dict:
        if app._current_hologram is None:
            return {"error": "no hologram loaded"}
        z_min = float(args.get("z_min_mm", app._params.af_z_min_mm))
        z_max = float(args.get("z_max_mm", app._params.af_z_max_mm))
        n_steps = int(args.get("n_steps", app._params.af_n_steps))
        done = threading.Event()
        holder: dict = {}

        def _ok(res):
            holder["payload"] = _summarise_autofocus(res)
            app._ai_cache_af = holder["payload"]
            done.set()

        def _err(err):
            holder["payload"] = {"error": str(err)}
            done.set()

        try:
            app._science.run_autofocus(
                app._current_hologram, app._params,
                z_min_mm=z_min, z_max_mm=z_max, n_steps=n_steps,
                sample_id=app._sample_id,
                on_result=_ok, on_error=_err,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"autofocus submit failed: {exc}"}
        return _wait(done, holder)

    def _invoke_qpi(args: dict) -> dict:
        if app._current_hologram is None:
            return {"error": "no hologram loaded"}
        z_mm = float(args.get("z_mm", app._params.z_mm))
        done = threading.Event()
        holder: dict = {}

        def _ok(res):
            holder["payload"] = _summarise_qpi(res)
            app._ai_cache_qpi = holder["payload"]
            done.set()

        def _err(err):
            holder["payload"] = {"error": str(err)}
            done.set()

        try:
            app._science.run_qpi(
                app._current_hologram, app._params,
                z_mm=z_mm, sample_id=app._sample_id,
                on_result=_ok, on_error=_err,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"qpi submit failed: {exc}"}
        return _wait(done, holder)

    def _invoke_depth_map(args: dict) -> dict:
        if app._current_hologram is None:
            return {"error": "no hologram loaded"}
        z_min = float(args.get("z_min_mm", app._params.af_z_min_mm))
        z_max = float(args.get("z_max_mm", app._params.af_z_max_mm))
        n_steps = int(args.get("n_steps", app._params.af_n_steps))
        done = threading.Event()
        holder: dict = {}

        def _ok(res):
            holder["payload"] = _summarise_depth(res)
            app._ai_cache_depth = holder["payload"]
            done.set()

        def _err(err):
            holder["payload"] = {"error": str(err)}
            done.set()

        try:
            app._science.run_depth_map(
                app._current_hologram, app._params,
                z_min_mm=z_min, z_max_mm=z_max, n_steps=n_steps,
                sample_id=app._sample_id,
                on_result=_ok, on_error=_err,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"depth_map submit failed: {exc}"}
        return _wait(done, holder)

    def _invoke_find_focus_candidates(args: dict) -> dict:
        if app._current_hologram is None:
            return {"error": "no hologram loaded"}
        z_min = float(args.get("z_min_mm", app._params.af_z_min_mm))
        z_max = float(args.get("z_max_mm", app._params.af_z_max_mm))
        n_steps = int(args.get("n_steps", 60))
        done = threading.Event()
        holder: dict = {}

        def _ok(res):
            holder["payload"] = _summarise_focus_candidates(res)
            done.set()

        def _err(err):
            holder["payload"] = {"error": str(err)}
            done.set()

        try:
            app._science.find_focus_candidates(
                app._current_hologram, app._params,
                z_min_mm=z_min, z_max_mm=z_max, n_steps=n_steps,
                sample_id=app._sample_id,
                on_result=_ok, on_error=_err,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"find_candidates submit failed: {exc}"}
        return _wait(done, holder)

    # ── Capture / sample-map / measurement (v2 stubs) ────────
    def _capture_frame() -> Any:
        # v2 doesn't have a separate "current camera frame" — we
        # use the loaded hologram array (same as v1's MockStage
        # fallback). When the camera live-feed is running, prefer
        # the latest live frame.
        if getattr(app, "_latest_live_frame", None) is not None:
            return app._latest_live_frame
        path = getattr(app, "_current_hologram", None)
        if path is None:
            return None
        try:
            from core.ingestion import load_any
            arr = np.asarray(load_any(path).array, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr
        except Exception:
            return None

    def _measure_sharpness(arr: Any) -> float:
        try:
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim != 2 or a.size < 9:
                return float("-inf")
            lap = (
                a[2:, 1:-1] + a[:-2, 1:-1]
                + a[1:-1, 2:] + a[1:-1, :-2]
                - 4.0 * a[1:-1, 1:-1]
            )
            v = float(np.var(lap))
            return v if np.isfinite(v) else float("-inf")
        except Exception:
            return float("-inf")

    def _persist_sample_map() -> Optional[str]:
        try:
            from core.user_profile import user_state_dir
            sample_id = sample_map.sample_id or "sample"
            base = user_state_dir() / "sample_maps"
            target = base / f"{sample_id}.json"
            sample_map.save(target)
            return str(target)
        except Exception:
            _LOG.warning("persist_sample_map failed", exc_info=True)
            return None

    def _capture_and_process(args: dict) -> dict:
        out: dict = {}
        run_recon = bool(args.get("run_recon", False))
        run_qpi = bool(args.get("run_qpi", False))
        if run_recon:
            r = _invoke_recon({})
            for k in ("phase_std", "z_mm"):
                if k in r:
                    out[k] = r[k]
        if run_qpi:
            q = _invoke_qpi({})
            for k in ("opd_range_nm", "total_dry_mass_pg",
                      "step_height_nm"):
                if k in q:
                    out[k] = q[k]
        return out

    # ── Audit ────────────────────────────────────────────────
    def _audit_tail(limit: int) -> list:
        try:
            return list(get_audit_log().entries_today(int(limit)))
        except Exception:
            return []

    # ── Confirm modal — DPG version is a placeholder until we
    # build a v2 confirmation dialog; default-True for now so
    # irreversible tools (none in v2 yet) don't deadlock the
    # demo. ───────────────────────────────────────────────────
    def _confirm(_name: str, _args: dict) -> bool:
        _LOG.info("ai: confirm requested for %s — auto-approved "
                  "(v2 modal not yet implemented)", _name)
        return True

    # ── Settings ─────────────────────────────────────────────
    settings = getattr(app._settings, "ai", None)

    return ToolContext(
        state=_state,
        last_recon_summary=lambda: getattr(app, "_ai_cache_recon", None),
        last_af_summary=lambda: getattr(app, "_ai_cache_af", None),
        last_qpi_summary=lambda: getattr(app, "_ai_cache_qpi", None),
        last_depth_summary=lambda: getattr(app, "_ai_cache_depth", None),
        audit_tail=_audit_tail,
        load_hologram=_load_hologram,
        set_recon_param=_set_recon_param,
        invoke_recon=_invoke_recon,
        invoke_autofocus=_invoke_autofocus,
        invoke_qpi=_invoke_qpi,
        invoke_depth_map=_invoke_depth_map,
        invoke_find_focus_candidates=_invoke_find_focus_candidates,
        stage=_LegacyStageStub(),
        capture_frame=_capture_frame,
        sample_map=sample_map,
        measure_sharpness=_measure_sharpness,
        persist_sample_map=_persist_sample_map,
        capture_and_process=_capture_and_process,
        audit=get_audit_log(),
        error_center=get_error_center(),
        confirm=_confirm,
        settings=settings,
        is_cancelled=getattr(app, "_ai_cancel_check", None),
        shutter=shutter,
        led=led,
        orchestrator=None,
    )


class _LegacyStageStub:
    """Hybrid v1+v2 stage stand-in.

    Wraps :class:`core.devices.mock_stage.MockStage` (the APT-style
    one with speed + step + jog) and **adds the four v1 legacy
    methods** (``get_position`` / ``move_relative`` / ``move_absolute``
    / ``home``). That way both tool families compose against the
    same instance:

    * Legacy stage_* tools (v1 sprint 2): mm-based, return tuples.
    * APT stage_* tools (v2.1.z+): µm-based, return dicts via the
      :class:`StageDevice` Protocol surface.

    A real lab swap-in for vendor hardware extends MockStage too
    so the same hybrid contract carries through.
    """

    def __init__(self):
        # Late import — avoid forcing core.devices on every test
        # that constructs a tool context but doesn't touch the
        # stage. ImportError → fall back to a no-op stub.
        try:
            from core.devices.mock_stage import MockStage
            self._mock = MockStage(name="bridge-stage")
            self._mock.connect()
        except Exception:  # pragma: no cover - defensive
            self._mock = None

    # ── v1 legacy: mm-based, return tuples ────────────────────
    def get_position(self):
        if self._mock is None:
            return (0.0, 0.0, 0.0)
        x, y, z = self._mock.position_um
        return (x / 1000.0, y / 1000.0, z / 1000.0)

    def move_relative(self, dx: float, dy: float, dz: float):
        if self._mock is None:
            return (0.0, 0.0, 0.0)
        self._mock.move_by(
            float(dx) * 1000.0,
            float(dy) * 1000.0,
            float(dz) * 1000.0,
        )
        x, y, z = self._mock.position_um
        return (x / 1000.0, y / 1000.0, z / 1000.0)

    def move_absolute(self, x: float, y: float, z: float):
        if self._mock is None:
            return (0.0, 0.0, 0.0)
        self._mock.move_to(
            float(x) * 1000.0,
            float(y) * 1000.0,
            float(z) * 1000.0,
        )
        return (float(x), float(y), float(z))

    def home(self):
        if self._mock is None:
            return (0.0, 0.0, 0.0)
        self._mock.home()
        return (0.0, 0.0, 0.0)

    # ── APT delegation: pass through everything else ──────────
    @property
    def position_um(self):
        if self._mock is None:
            return (0.0, 0.0, 0.0)
        return self._mock.position_um

    @property
    def is_connected(self):
        return self._mock is not None and self._mock.is_connected

    def connect(self):
        if self._mock is not None:
            self._mock.connect()

    def disconnect(self):
        if self._mock is not None:
            self._mock.disconnect()

    def move_to(self, x_um, y_um, z_um=0.0):
        if self._mock is None:
            raise RuntimeError("stage backend not available")
        self._mock.move_to(x_um, y_um, z_um)

    def move_by(self, dx_um, dy_um, dz_um=0.0):
        if self._mock is None:
            raise RuntimeError("stage backend not available")
        self._mock.move_by(dx_um, dy_um, dz_um)

    def set_speed_um_per_s(self, v):
        if self._mock is None:
            raise RuntimeError("stage backend not available")
        self._mock.set_speed_um_per_s(v)

    @property
    def speed_um_per_s(self):
        return self._mock.speed_um_per_s if self._mock else 0.0

    def set_step_size_um(self, step_um):
        if self._mock is None:
            raise RuntimeError("stage backend not available")
        self._mock.set_step_size_um(step_um)

    @property
    def step_size_um(self):
        return self._mock.step_size_um if self._mock else 0.0

    def jog(self, axis, direction):
        if self._mock is None:
            raise RuntimeError("stage backend not available")
        self._mock.jog(axis, direction)

    def stop_motion(self):
        if self._mock is not None:
            self._mock.stop_motion()


__all__ = [
    "make_v2_tool_context",
]
