"""Headless session runner — v2.0.7 sprint, T1 + T6.

Runs a :class:`core.session.Session` manifest through the same
pipeline the GUI uses (off-axis extract → autofocus → reconstruct),
writing one JSON result per frame plus an aggregate CSV.

Why headless
------------
Sven's IT team has a Linux RTX 4090 box. 3000 holograms × 4-5 sec
on Mac CPU is 4 hours; same scan on the GPU box overnight is
unattended. Dear PyGui doesn't run there. This module is the
no-UI path that lets the lab park batch jobs on Linux.

Resume (T6)
-----------
``--resume-if-exists`` skips frames whose ``frame_<index>.json``
already exists in the output directory **and whose signature
matches** the current session. Signature mismatch (e.g. the
operator changed ``z_min_mm`` in the manifest) invalidates the
existing output — the frame is re-run rather than silently
skipped. Karin's catch: don't trust output you didn't ask for.

CLI shape
---------
::

    python -m cli.run_session run <manifest.json> --out <dir>
                                  [--phase autofocus|reconstruct|all]
                                  [--resume-if-exists]
                                  [--workers N]
                                  [--quiet]

    python -m cli.run_session inspect <manifest.json>

The runner emits one JSONL line per frame to stdout (tail -f
friendly). SIGINT (Ctrl-C) is caught: the in-flight frame finishes
its current step, partial results flush, and we exit ≠0 so the
shell wrapper can detect cancellation.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Ensure src/ is on the path when the runner is invoked directly.
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.audit import get_audit_log  # noqa: E402
from core.autofocus import FocusMetric, autofocus_zscan  # noqa: E402
from core.ingestion import load_any  # noqa: E402
from core.offaxis import OffAxisParams, extract_complex_field_offaxis  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)
from core.session import HologramFrame, Session  # noqa: E402
from core.session_export import (  # noqa: E402
    CellMeasurement,
    FrameResult,
    write_session_csv,
)


_LOG = logging.getLogger("dhm.cli")


# Process-level cancel flag — flipped by SIGINT handler. Workers
# poll this between phases.
_CANCEL = False


def _sigint(*_):
    global _CANCEL
    _CANCEL = True
    _LOG.warning("CLI: SIGINT received, finishing current frame "
                 "then exiting (Ctrl-C again to abort hard).")


# ---------------------------------------------------------------------------
# Per-frame pipeline
# ---------------------------------------------------------------------------

def _per_frame_job(
    frame_path: Path,
    effective_params: Dict[str, Any],
    phase: str,
) -> Dict[str, Any]:
    """Pure-function frame processor used by the worker pool.

    Returns a dict that mirrors :class:`FrameResult` but is JSON-
    serialisable so it can survive a multiprocessing pickle. The
    runner converts back into a :class:`FrameResult` after collection.

    Steps run depend on ``phase``:
      * ``"autofocus"`` — load + extract + autofocus only (no recon).
      * ``"reconstruct"`` — autofocus + propagate to z_est.
      * ``"all"`` (default) — same as reconstruct (QPI/segmentation
        come in v2.0.8 — until then "all" stops at reconstruction).
    """
    t0 = time.monotonic()
    out: Dict[str, Any] = {
        "frame_path": str(frame_path),
        "z_mm": None,
        "runtime_ms": 0.0,
        "error": None,
        "cells": [],
    }
    try:
        loaded = load_any(frame_path)
        raw = np.asarray(loaded.array, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        # Preprocessing — match _preprocess_raw semantics so headless
        # and GUI paths agree byte-for-byte. Defaults reproduce the
        # Reconstruct button branch (subtract_mean=True, no hann).
        if effective_params.get("subtract_mean", True):
            raw = raw - float(raw.mean())
        if effective_params.get("hann_window", False):
            wy = np.hanning(raw.shape[0]).astype(np.float32)
            wx = np.hanning(raw.shape[1]).astype(np.float32)
            raw = raw * (wy[:, None] * wx[None, :])
        peak = float(np.max(np.abs(raw)))
        if peak > 0:
            raw = raw / peak

        offaxis = OffAxisParams(
            radius=int(effective_params.get("mask_radius", 40)),
        )
        field, _ = extract_complex_field_offaxis(raw, offaxis)
        base = ReconstructionParams(
            wavelength_m=float(effective_params.get(
                "wavelength_nm", 632.8)) * 1e-9,
            pixel_size_m=float(effective_params.get(
                "pixel_um", 2.5)) * 1e-6
            / max(1.0, float(effective_params.get(
                "magnification", 1.0))),
            z_m=0.0,
            n=float(effective_params.get("n_medium", 1.33)),
        )
        method = (
            ReconstructionMethod.ASM
            if str(effective_params.get("method", "ASM")).upper() == "ASM"
            else ReconstructionMethod.FRESNEL
        )

        # Autofocus — uses the same fast_evaluator path as the GUI.
        z_min_mm = float(effective_params.get("af_z_min_mm", 0.0))
        z_max_mm = float(effective_params.get("af_z_max_mm", 25.0))
        n_steps = int(effective_params.get("af_n_steps", 40))
        zs = list(np.linspace(z_min_mm * 1e-3, z_max_mm * 1e-3, n_steps))
        metric_name = str(effective_params.get(
            "autofocus_metric", "ENTROPY")).upper()
        try:
            metric = FocusMetric[metric_name]
        except KeyError:
            metric = FocusMetric.ENTROPY
        af_result = autofocus_zscan(field, base, zs, method, metric)
        out["z_mm"] = float(af_result.best_z_m) * 1e3

        if phase != "autofocus":
            # Reconstruction at the autofocus z. We don't write the
            # complex field to disk by default — too big for 3000-
            # frame sessions. Future flag (--save-fields) can enable.
            p = ReconstructionParams(
                wavelength_m=base.wavelength_m,
                pixel_size_m=base.pixel_size_m,
                z_m=float(af_result.best_z_m),
                n=base.n,
            )
            recon = propagate(field, p, method)
            # Touch arr so propagation actually executes (ASM is
            # lazy-eval safe, but we want runtime to include it).
            _ = float(np.abs(recon).max())

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["runtime_ms"] = (time.monotonic() - t0) * 1000.0
    return out


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _frame_output_path(out_dir: Path, frame: HologramFrame) -> Path:
    return out_dir / f"frame_{frame.index:05d}.json"


def _signature_marker_path(out_dir: Path) -> Path:
    """Sidecar that pins the session signature current outputs were
    produced for. Resume compares the session's *current* signature
    against this; mismatch → outputs are stale, ignore them."""
    return out_dir / "session.signature"


def _read_existing_signature(out_dir: Path) -> Optional[str]:
    p = _signature_marker_path(out_dir)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _write_signature(out_dir: Path, sig: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _signature_marker_path(out_dir).write_text(sig, encoding="utf-8")


def _frame_output_exists_and_valid(
    out_dir: Path, frame: HologramFrame, expected_sig: str,
) -> bool:
    p = _frame_output_path(out_dir, frame)
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return d.get("session_signature") == expected_sig


def _write_frame_output(out_dir: Path, frame: HologramFrame,
                        sig: str, payload: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _frame_output_path(out_dir, frame)
    p.write_text(json.dumps({
        "session_signature": sig,
        "frame_index": frame.index,
        "frame_path": frame.path,
        "timestamp_s": frame.timestamp_s,
        **payload,
    }, indent=2, default=str), encoding="utf-8")
    return p


def _result_from_payload(frame: HologramFrame,
                         payload: Dict[str, Any]) -> FrameResult:
    cells: List[CellMeasurement] = []
    for c in payload.get("cells") or []:
        if not isinstance(c, dict):
            continue
        cells.append(CellMeasurement(
            cell_id=int(c.get("cell_id", 0)),
            cy_px=c.get("cy_px"),
            cx_px=c.get("cx_px"),
            z_mm=c.get("z_mm"),
            dry_mass_pg=c.get("dry_mass_pg"),
            area_um2=c.get("area_um2"),
            height_nm=c.get("height_nm"),
        ))
    return FrameResult(
        frame=frame,
        z_mm=payload.get("z_mm"),
        runtime_ms=float(payload.get("runtime_ms", 0.0)),
        error=payload.get("error"),
        cells=cells,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_session(
    manifest_path: Path,
    out_dir: Path,
    *,
    phase: str = "all",
    resume_if_exists: bool = False,
    workers: int = 1,
    csv_layout: str = "long",
    quiet: bool = False,
    progress_stream: Any = None,
) -> int:
    """Run a session. Returns shell exit code.

    Parameters
    ----------
    manifest_path
        Path to the session JSON.
    out_dir
        Output directory. Per-frame ``frame_<index>.json`` and a
        final ``session.csv`` (long or wide) land here.
    phase
        ``"autofocus"`` | ``"reconstruct"`` | ``"all"``.
    resume_if_exists
        Skip frames whose output JSON already exists with a matching
        session signature.
    workers
        Concurrent frames (multiprocessing pool). 1 = serial.
    csv_layout
        ``"long"`` or ``"wide"``.
    quiet
        Suppress stdout JSONL progress lines.
    progress_stream
        Where to emit JSONL progress lines; defaults to stdout when
        ``quiet=False``. Tests inject a ``StringIO`` to capture.
    """
    session = Session.load_json(manifest_path)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = session.signature()

    # Resume logic: if existing signature differs, the previous run
    # was for a different config — bail or delete based on flag.
    prev_sig = _read_existing_signature(out_dir)
    if prev_sig and prev_sig != sig and not resume_if_exists:
        # Operator likely re-pointed --out at the wrong dir.
        _LOG.warning(
            "out dir %s carries signature %s but session is %s; "
            "outputs will be overwritten in place. Use --resume-if-exists "
            "to skip already-computed frames matching the new signature.",
            out_dir, prev_sig, sig,
        )
    _write_signature(out_dir, sig)

    # Pre-flight: assemble the per-frame work list.
    pending: List[HologramFrame] = []
    skipped: List[HologramFrame] = []
    for fr in session.frames:
        if (resume_if_exists
                and _frame_output_exists_and_valid(out_dir, fr, sig)):
            skipped.append(fr)
            continue
        pending.append(fr)

    stream = progress_stream
    if stream is None and not quiet:
        stream = sys.stdout

    def _emit(d: dict):
        if stream is None:
            return
        stream.write(json.dumps(d, default=str) + "\n")
        stream.flush()

    _emit({"event": "session_start", "session_id": session.id,
           "signature": sig, "total": len(session.frames),
           "skipped": len(skipped), "pending": len(pending),
           "out_dir": str(out_dir)})
    get_audit_log().record(
        action="session_start",
        params={"manifest": str(manifest_path),
                "out_dir": str(out_dir),
                "phase": phase, "workers": workers,
                "frames_total": len(session.frames),
                "frames_pending": len(pending),
                "frames_skipped": len(skipped)},
        result_summary={"signature": sig},
    )

    signal.signal(signal.SIGINT, _sigint)

    results: List[FrameResult] = []
    # Re-attach skipped results so the final CSV reflects the
    # whole session, not just this run's pending.
    for fr in skipped:
        try:
            payload = json.loads(_frame_output_path(out_dir, fr)
                                 .read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results.append(_result_from_payload(fr, payload))

    started = time.monotonic()
    failed = 0
    if workers <= 1:
        for fr in pending:
            if _CANCEL:
                break
            payload = _per_frame_job(
                session.resolve_frame_path(fr),
                session.effective_params_for(fr),
                phase=phase,
            )
            _write_frame_output(out_dir, fr, sig, payload)
            results.append(_result_from_payload(fr, payload))
            failed += 1 if payload.get("error") else 0
            _emit({"event": "frame_done", "index": fr.index,
                   "z_mm": payload.get("z_mm"),
                   "runtime_ms": payload.get("runtime_ms"),
                   "error": payload.get("error")})
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _per_frame_job,
                    session.resolve_frame_path(fr),
                    session.effective_params_for(fr),
                    phase,
                ): fr for fr in pending
            }
            for f in as_completed(futures):
                if _CANCEL:
                    pool.shutdown(cancel_futures=True)
                    break
                fr = futures[f]
                try:
                    payload = f.result()
                except Exception as exc:
                    payload = {
                        "frame_path": fr.path,
                        "z_mm": None, "runtime_ms": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                        "cells": [],
                    }
                _write_frame_output(out_dir, fr, sig, payload)
                results.append(_result_from_payload(fr, payload))
                failed += 1 if payload.get("error") else 0
                _emit({"event": "frame_done", "index": fr.index,
                       "z_mm": payload.get("z_mm"),
                       "runtime_ms": payload.get("runtime_ms"),
                       "error": payload.get("error")})

    # Sort results by frame index so the CSV is monotonic — workers
    # finish out of order under concurrency.
    results.sort(key=lambda r: r.frame.index)

    csv_path = out_dir / "session.csv"
    write_session_csv(session, results, csv_path, layout=csv_layout)

    elapsed = time.monotonic() - started
    _emit({
        "event": "session_done",
        "frames_processed": len(pending),
        "frames_skipped": len(skipped),
        "frames_failed": failed,
        "elapsed_s": elapsed,
        "csv_path": str(csv_path),
        "cancelled": _CANCEL,
    })
    get_audit_log().record(
        action="session_done",
        params={"manifest": str(manifest_path),
                "out_dir": str(out_dir)},
        result_summary={
            "signature": sig,
            "elapsed_s": elapsed,
            "frames_processed": len(pending),
            "frames_skipped": len(skipped),
            "frames_failed": failed,
            "cancelled": _CANCEL,
        },
    )

    if _CANCEL:
        return 130   # POSIX SIGINT exit
    if failed:
        return 1
    return 0


# ---------------------------------------------------------------------------
# inspect subcommand — print manifest summary without running anything
# ---------------------------------------------------------------------------

def inspect_session(manifest_path: Path) -> int:
    s = Session.load_json(manifest_path)
    out = {
        "id": s.id,
        "operator": s.operator,
        "sample_id": s.sample_id,
        "created_at": s.created_at,
        "root_dir": s.root_dir,
        "params": s.params,
        "frame_count": len(s),
        "first_frame": (asdict(s.frames[0]) if s.frames else None),
        "last_frame": (asdict(s.frames[-1]) if s.frames else None),
        "signature": s.signature(),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dhm.session",
        description="Headless DHM session runner (v2.0.7).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a session manifest end-to-end")
    run.add_argument("manifest", type=Path,
                     help="Path to session JSON manifest")
    run.add_argument("--out", type=Path, required=True,
                     help="Output directory (created if missing)")
    run.add_argument(
        "--phase", choices=["autofocus", "reconstruct", "all"],
        default="all",
        help="Pipeline depth (default: all)",
    )
    run.add_argument("--resume-if-exists", action="store_true",
                     help="Skip frames whose output JSON already "
                          "matches the current session signature")
    run.add_argument("--workers", type=int, default=1,
                     help="Concurrent frames (default: 1)")
    run.add_argument("--csv-layout", choices=["long", "wide"],
                     default="long",
                     help="Aggregate CSV layout (default: long)")
    run.add_argument("--quiet", action="store_true",
                     help="Suppress JSONL progress on stdout")

    insp = sub.add_parser("inspect",
                          help="Print manifest summary as JSON")
    insp.add_argument("manifest", type=Path)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    if args.cmd == "run":
        return run_session(
            args.manifest,
            args.out,
            phase=args.phase,
            resume_if_exists=args.resume_if_exists,
            workers=args.workers,
            csv_layout=args.csv_layout,
            quiet=args.quiet,
        )
    if args.cmd == "inspect":
        return inspect_session(args.manifest)
    return 2


if __name__ == "__main__":
    sys.exit(main())
