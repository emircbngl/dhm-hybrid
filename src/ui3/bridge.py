"""WorkerBridge — Qt signal facade over the shared Qt-free compute drivers.

ui3 reuses the already-correct, already-tested science layer from
``core.drivers.reconstruction`` (``ReconstructionDriver``) and
``core.drivers.workers`` (``ScienceDriver``) — framework-free
(ThreadPoolExecutor, plain callbacks), encoding hard-won correctness
(subtract_mean parity, reference-mode gating, reference-free QPI correction,
reflection halving, per-z reference division). ui3 rebuilds the
*presentation*, not the compute.

The drivers fire their ``on_result``/``on_error`` callbacks on a worker thread.
This bridge is a ``QObject`` living on the GUI thread; the callbacks emit Qt
signals, so Qt marshals the result onto the GUI thread automatically (a
cross-thread ``emit`` is a queued connection). Panels connect to these signals
and never touch a worker thread themselves.

(2026-07-06: the compute drivers were relocated from ``ui2/`` to
``core/drivers/`` — ui3 no longer imports the ui2 package at all.)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from core.drivers.reconstruction import ReconParams, ReconstructionDriver
from core.drivers.workers import ScienceDriver


class WorkerBridge(QObject):
    """One bridge per window. Owns the two compute drivers."""

    # Reconstruction
    recon_done = Signal(object)          # ReconResult
    recon_error = Signal(str)
    # Autofocus
    autofocus_done = Signal(object)      # AutofocusResult
    autofocus_error = Signal(str)
    focus_candidates_done = Signal(object)   # list of candidates
    focus_candidates_error = Signal(str)
    # QPI
    qpi_done = Signal(object)            # QPIOneShotResult
    qpi_error = Signal(str)
    qpi_batch_done = Signal(object)      # QPIBatchResultWrap
    qpi_batch_error = Signal(str)
    # Depth map
    depth_done = Signal(object)          # DepthMapResultWrap
    depth_error = Signal(str)
    # Cross-cutting: (busy, human-readable label)
    busy_changed = Signal(bool, str)
    # Internal: posted from worker threads to re-evaluate idleness ON THE
    # GUI THREAD (see _end/_publish_idle).
    _idle_check = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._recon = ReconstructionDriver()
        self._science = ScienceDriver()
        # busy_changed is reference-counted across the two independent
        # executors (recon + science). Without this, two overlapping ops
        # (e.g. a reconstruct and a depth map) would fight over one boolean:
        # whichever finished first emitted busy=False and cleared the
        # indicator while the other was still running (2026-07-06 review).
        # _begin runs on the GUI thread; _end runs on a worker thread (driver
        # callbacks), so the counter needs a lock — AND the idle decision
        # must be made on the GUI thread: a busy=False emitted directly from
        # the worker is a QUEUED event, so a user starting a new op before it
        # is delivered would see the fresh busy=True clobbered by the stale
        # idle (2026-07-06 review #2). _end only posts _idle_check; the
        # GUI-thread slot re-reads the counter at delivery time.
        self._busy_lock = threading.Lock()
        self._inflight = 0
        self._idle_check.connect(self._publish_idle)

    # -- helpers --------------------------------------------------------
    def _begin(self, label: str) -> None:
        with self._busy_lock:
            self._inflight += 1
        self.busy_changed.emit(True, label)

    def _end(self) -> None:
        with self._busy_lock:
            self._inflight = max(0, self._inflight - 1)
        # Do NOT emit busy_changed here (worker thread) — post a check that
        # runs on the GUI thread against the counter as it is THEN.
        self._idle_check.emit()

    def _publish_idle(self) -> None:
        """GUI-thread slot: publish idle only if nothing is in flight *now*.

        If a new op started between the worker's _end and this delivery,
        _inflight is already >0 and the stale idle is correctly dropped.
        (Two _ends draining to zero can post two checks — the duplicate
        busy=False is idempotent for every listener.)"""
        with self._busy_lock:
            idle = self._inflight == 0
        if idle:
            self.busy_changed.emit(False, "")

    # -- reconstruction -------------------------------------------------
    def reconstruct(self, path: Path, params: ReconParams,
                    sample_id: str = "") -> None:
        self._begin("Reconstructing…")

        def ok(res):
            self._end()
            self.recon_done.emit(res)

        def err(e):
            self._end()
            self.recon_error.emit(getattr(e, "message", str(e)))

        self._recon.submit(Path(path), params, sample_id=sample_id,
                           on_result=ok, on_error=err)

    # -- autofocus ------------------------------------------------------
    def autofocus(self, path: Path, params: ReconParams, *,
                  z_min_mm: float, z_max_mm: float, n_steps: int = 40,
                  sample_id: str = "") -> None:
        self._begin("Autofocusing…")

        def ok(res):
            self._end()
            self.autofocus_done.emit(res)

        def err(e):
            self._end()
            self.autofocus_error.emit(getattr(e, "message", str(e)))

        self._science.run_autofocus(
            Path(path), params, z_min_mm=z_min_mm, z_max_mm=z_max_mm,
            n_steps=n_steps, sample_id=sample_id, on_result=ok, on_error=err)

    def find_focus_candidates(self, path: Path, params: ReconParams, *,
                              z_min_mm: float, z_max_mm: float,
                              n_steps: int = 60, sample_id: str = "") -> None:
        self._begin("Scanning focus landscape…")

        def ok(res):
            self._end()
            self.focus_candidates_done.emit(res)

        def err(e):
            self._end()
            self.focus_candidates_error.emit(getattr(e, "message", str(e)))

        self._science.find_focus_candidates(
            Path(path), params, z_min_mm=z_min_mm, z_max_mm=z_max_mm,
            n_steps=n_steps, sample_id=sample_id, on_result=ok, on_error=err)

    # -- QPI ------------------------------------------------------------
    def qpi(self, path: Path, params: ReconParams, *, z_mm: float,
            n_sample: Optional[float] = None, n_medium: Optional[float] = None,
            sample_id: str = "") -> None:
        self._begin("Computing QPI…")

        def ok(res):
            self._end()
            self.qpi_done.emit(res)

        def err(e):
            self._end()
            self.qpi_error.emit(getattr(e, "message", str(e)))

        self._science.run_qpi(
            Path(path), params, z_mm=z_mm, n_sample=n_sample,
            n_medium=n_medium, sample_id=sample_id, on_result=ok, on_error=err)

    def qpi_batch(self, path: Path, params: ReconParams, *,
                  z_min_mm: float, z_max_mm: float, n_steps: int = 60,
                  n_sample: Optional[float] = None,
                  n_medium: Optional[float] = None, sample_id: str = "") -> None:
        self._begin("QPI batch across focus candidates…")

        def ok(res):
            self._end()
            self.qpi_batch_done.emit(res)

        def err(e):
            self._end()
            self.qpi_batch_error.emit(getattr(e, "message", str(e)))

        self._science.run_qpi_batch(
            Path(path), params, z_min_mm=z_min_mm, z_max_mm=z_max_mm,
            n_steps=n_steps, n_sample=n_sample, n_medium=n_medium,
            sample_id=sample_id, on_result=ok, on_error=err)

    # -- depth map ------------------------------------------------------
    def depth_map(self, path: Path, params: ReconParams, *,
                  z_min_mm: float, z_max_mm: float, n_steps: int = 40,
                  window_size: int = 5, sample_id: str = "") -> None:
        self._begin("Building depth map…")

        def ok(res):
            self._end()
            self.depth_done.emit(res)

        def err(e):
            self._end()
            self.depth_error.emit(getattr(e, "message", str(e)))

        self._science.run_depth_map(
            Path(path), params, z_min_mm=z_min_mm, z_max_mm=z_max_mm,
            n_steps=n_steps, window_size=window_size, sample_id=sample_id,
            on_result=ok, on_error=err)

    # -- lifecycle ------------------------------------------------------
    def cancel(self) -> bool:
        a = self._recon.cancel()
        b = self._science.cancel()
        return bool(a or b)

    def shutdown(self) -> None:
        try:
            self._recon.shutdown()
        finally:
            self._science.shutdown()
