"""TimelapsePanel — time-series / multi-position / tracking entry panel (ui3).

Reads back to the two runners that already carry the physics/scheduling
logic — this panel is presentation + a Qt worker thread only:

* :mod:`core.timelapse` (:class:`TimelapseRunner`, :class:`TimelapseSchedule`)
  — single-position: interval, total-frame / max-duration stop conditions,
  one TIFF + a :class:`core.session.Session` manifest per run.
* :mod:`core.multi_position_timelapse`
  (:class:`MultiPositionTimelapseRunner`, :class:`MultiPositionSchedule`)
  — multi-position: same cadence idea, but every "tick" drives an
  :class:`core.devices.orchestrator.AcquisitionPlan` across a list of
  :class:`~core.devices.orchestrator.StagePosition` waypoints (LED intensity,
  shutter-per-frame, settle time — all per the orchestrator's existing
  contract, unchanged here).
* :mod:`core.tracking` (:func:`link_detections`, :class:`Track`) — optional
  "Link detections" hook. No cell detector is wired into this panel (the
  synthetic camera emits fringe patterns, not labelled cells), so the hook
  runs the real linker against whatever per-frame detections have been
  recorded on this panel (none, out of the box) and reports the resulting
  track count — this keeps the control honestly connected to
  ``core.tracking`` instead of a decorative button that silently no-ops.

Threading
---------
Neither runner is Qt-aware (both are plain classes with a blocking ``run()``
+ thread-safe ``cancel()``), so this panel drives each inside a small
``QThread`` subclass (:class:`_TimelapseWorker`) that owns the runner
instance, forwards its ``on_progress`` / ``on_tick`` callbacks as queued Qt
signals (the callbacks fire on the worker thread; Qt marshals a cross-thread
``emit`` onto the GUI thread automatically — same pattern
``ui3.bridge.WorkerBridge`` uses for the compute drivers), and emits a
``finished_result`` signal with the ``TimelapseResult`` /
``MultiPositionResult`` when ``run()`` returns. Stop --> ``runner.cancel()``
(latency bounded to ~100 ms per both runners' docstrings) then the thread is
joined via ``wait()``.

No real hardware is required or assumed: the camera is always
``core.cameras.synthetic`` (deterministic fringe generator, always
available); in multi-position mode the stage/shutter/LED are the
``core.devices`` mock backends. Nothing here blocks the GUI thread or can
crash for lack of hardware.

Controls
--------
* Mode combo — ``Single-position`` / ``Multi-position``.
* Interval (s) — ``QDoubleSpinBox``, shared by both schedules.
* Total frames (single) / Total ticks (multi) — ``QSpinBox``, 0 = unbounded
  (mapped to ``None`` on the schedule so callers must supply *something* to
  eventually stop, per the runners' own docstrings — the "no bound at all"
  footgun is a caller decision, not one this panel avoids for the user, but
  Stop always works regardless).
* Max duration (s) — optional ``QSpinBox``, 0 = unbounded.
* Output directory — ``QFileDialog.getExistingDirectory``.
* Multi-position position table — a plain x/y/z (µm) ``QTableWidget`` with
  Add-row / Remove-row buttons.
* Start / Stop — runs the schedule on a background ``QThread``; Stop calls
  ``cancel()`` on the live runner.
* Progress — frame/tick counter label + a ``QProgressBar`` (indeterminate
  when the schedule is unbounded, i.e. ``expected_*() is None``).
* "Link detections" — optional section, off to the side; explains the hook
  and, on click, calls ``core.tracking.link_detections`` against the
  recorded per-frame detections (empty list out of the box → "0 tracks",
  which is correct, not broken).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui3.context import PanelContext
from ui3.design import Space

_LOG = logging.getLogger("ui3.panels.timelapse")

_MODE_SINGLE = "Single-position"
_MODE_MULTI = "Multi-position"


# ---------------------------------------------------------------------------
# Background worker — owns the runner, forwards progress as Qt signals.
# ---------------------------------------------------------------------------

class _TimelapseWorker(QThread):
    """Runs one :class:`TimelapseRunner` or
    :class:`MultiPositionTimelapseRunner` to completion on a background
    thread.

    ``runner`` must already be fully constructed (schedule + camera/devices
    + out_dir); this class only supplies the thread + signal plumbing. The
    runner's own ``on_progress``/``on_tick`` callback is wrapped so it always
    re-emits through ``progress`` regardless of which runner shape is in
    play (the two runners use different callback signatures — see
    ``core.timelapse.TimelapseRunner`` (``frame_idx, expected_total``) vs.
    ``core.multi_position_timelapse.MultiPositionTimelapseRunner``
    (``tick, n_positions``) — both are 2-tuples of ``(done, total_or_None)``
    so a single ``progress(int, object)`` signal covers both.
    """

    progress = Signal(int, object)          # (done_count, expected_total_or_None)
    finished_result = Signal(object)        # TimelapseResult | MultiPositionResult
    failed = Signal(str)

    def __init__(self, runner: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._runner = runner

    def run(self) -> None:  # noqa: D102 - QThread override
        try:
            result = self._runner.run()
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.exception("timelapse worker failed")
            self.failed.emit(str(exc))
            return
        self.finished_result.emit(result)

    def request_cancel(self) -> None:
        try:
            self._runner.cancel()
        except Exception:  # pragma: no cover - defensive
            _LOG.debug("runner.cancel() raised", exc_info=True)


class TimelapsePanel(QWidget):
    """Time-series + multi-position + tracking-hook acquisition panel."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._out_dir: Optional[Path] = None
        self._worker: Optional[_TimelapseWorker] = None
        self._detections_by_frame: List[list] = []
        self._build_ui()
        self._on_mode_changed(self.cmb_mode.currentText())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.md)
        root.setSpacing(Space.md)

        heading = QLabel("Time-lapse", self)
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        # -- Schedule card ------------------------------------------------
        card = QFrame(self)
        card.setProperty("role", "card")
        form = QFormLayout(card)

        self.cmb_mode = QComboBox(card)
        self.cmb_mode.addItems([_MODE_SINGLE, _MODE_MULTI])
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Mode", self.cmb_mode)

        self.spin_interval = QDoubleSpinBox(card)
        self.spin_interval.setRange(0.1, 24 * 3600.0)
        self.spin_interval.setDecimals(1)
        self.spin_interval.setSuffix(" s")
        self.spin_interval.setValue(5.0)
        form.addRow("Interval", self.spin_interval)

        self.spin_total = QSpinBox(card)
        self.spin_total.setRange(0, 1_000_000)
        self.spin_total.setSpecialValueText("unbounded")
        self.spin_total.setValue(5)
        self.lbl_total = QLabel("Total frames")
        form.addRow(self.lbl_total, self.spin_total)

        self.spin_max_duration = QSpinBox(card)
        self.spin_max_duration.setRange(0, 24 * 3600)
        self.spin_max_duration.setSpecialValueText("unbounded")
        self.spin_max_duration.setSuffix(" s")
        self.spin_max_duration.setValue(0)
        form.addRow("Max duration", self.spin_max_duration)

        out_row = QHBoxLayout()
        self.lbl_out_dir = QLabel("(none selected)", card)
        self.lbl_out_dir.setProperty("role", "muted")
        self.lbl_out_dir.setWordWrap(True)
        btn_browse = QPushButton("Choose…", card)
        btn_browse.clicked.connect(self._on_choose_out_dir)
        out_row.addWidget(self.lbl_out_dir, stretch=1)
        out_row.addWidget(btn_browse)
        form.addRow("Output dir", out_row)

        root.addWidget(card)

        # -- Multi-position table ------------------------------------------
        self.grp_positions = QGroupBox("Positions (µm)", self)
        pos_lay = QVBoxLayout(self.grp_positions)

        self.tbl_positions = QTableWidget(0, 4, self.grp_positions)
        self.tbl_positions.setHorizontalHeaderLabels(["x", "y", "z", "label"])
        self.tbl_positions.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_positions.setSelectionBehavior(QAbstractItemView.SelectRows)
        pos_lay.addWidget(self.tbl_positions)

        pos_btn_row = QHBoxLayout()
        btn_add_pos = QPushButton("Add", self.grp_positions)
        btn_add_pos.clicked.connect(self._on_add_position)
        btn_del_pos = QPushButton("Remove selected", self.grp_positions)
        btn_del_pos.clicked.connect(self._on_remove_position)
        pos_btn_row.addWidget(btn_add_pos)
        pos_btn_row.addWidget(btn_del_pos)
        pos_btn_row.addStretch(1)
        pos_lay.addLayout(pos_btn_row)

        root.addWidget(self.grp_positions)

        # -- Start/Stop + progress -----------------------------------------
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start", self)
        self.btn_start.setProperty("role", "primary")
        self.btn_start.clicked.connect(self._on_start_clicked)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop", self)
        self.btn_stop.setProperty("role", "danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self.btn_stop)
        root.addLayout(btn_row)

        self.lbl_progress = QLabel("Idle", self)
        self.lbl_progress.setProperty("role", "readout")
        root.addWidget(self.lbl_progress)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        # -- Tracking hook ---------------------------------------------
        grp_track = QGroupBox("Tracking", self)
        track_lay = QVBoxLayout(grp_track)
        info = QLabel(
            "Optional: link per-frame cell detections into stable tracks "
            "across the captured series (core.tracking.link_detections, "
            "nearest-neighbour + Hungarian assignment). No detector is "
            "wired into this panel yet — this runs the real linker "
            "against whatever detections this run recorded (empty by "
            "default outside a detection pipeline).",
            grp_track,
        )
        info.setProperty("role", "muted")
        info.setWordWrap(True)
        track_lay.addWidget(info)

        track_btn_row = QHBoxLayout()
        self.btn_link = QPushButton("Link detections", grp_track)
        self.btn_link.clicked.connect(self._on_link_detections_clicked)
        track_btn_row.addWidget(self.btn_link)
        track_btn_row.addStretch(1)
        track_lay.addLayout(track_btn_row)

        self.lbl_track_result = QLabel("", grp_track)
        self.lbl_track_result.setProperty("role", "muted")
        track_lay.addWidget(self.lbl_track_result)

        root.addWidget(grp_track)
        root.addStretch(1)

    # ------------------------------------------------------------------
    # Mode UX
    # ------------------------------------------------------------------
    def _on_mode_changed(self, mode: str) -> None:
        is_multi = mode == _MODE_MULTI
        self.grp_positions.setVisible(is_multi)
        self.lbl_total.setText("Total ticks" if is_multi else "Total frames")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    def _on_choose_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if d:
            self._out_dir = Path(d)
            self.lbl_out_dir.setText(str(self._out_dir))

    # ------------------------------------------------------------------
    # Position table
    # ------------------------------------------------------------------
    def _on_add_position(self) -> None:
        row = self.tbl_positions.rowCount()
        self.tbl_positions.insertRow(row)
        for col, default in enumerate(("0.0", "0.0", "0.0", f"pos{row}")):
            self.tbl_positions.setItem(row, col, QTableWidgetItem(default))

    def _on_remove_position(self) -> None:
        rows = sorted({idx.row() for idx in self.tbl_positions.selectedIndexes()},
                     reverse=True)
        for r in rows:
            self.tbl_positions.removeRow(r)

    def _read_positions(self) -> list:
        from core.devices.orchestrator import StagePosition

        out = []
        for row in range(self.tbl_positions.rowCount()):
            def _cell(col: int, default: str = "0.0") -> str:
                item = self.tbl_positions.item(row, col)
                return item.text() if item is not None else default

            try:
                x = float(_cell(0))
                y = float(_cell(1))
                z = float(_cell(2))
            except ValueError:
                continue
            label = _cell(3, f"pos{row}")
            out.append(StagePosition(x_um=x, y_um=y, z_um=z, label=label))
        return out

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------
    def _on_start_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._ctx.set_status("Time-lapse already running.", "warn")
            return
        if self._out_dir is None:
            self._ctx.set_status(
                "Choose an output directory before starting.", "warn")
            return

        mode = self.cmb_mode.currentText()
        interval_s = float(self.spin_interval.value())
        total_raw = int(self.spin_total.value())
        total = total_raw if total_raw > 0 else None
        max_dur_raw = int(self.spin_max_duration.value())
        max_duration_s = float(max_dur_raw) if max_dur_raw > 0 else None

        try:
            if mode == _MODE_MULTI:
                runner, expected = self._build_multi_position_runner(
                    interval_s=interval_s, total_ticks=total,
                    max_duration_s=max_duration_s)
            else:
                runner, expected = self._build_single_position_runner(
                    interval_s=interval_s, total_frames=total,
                    max_duration_s=max_duration_s)
        except Exception as exc:
            self._ctx.set_status(f"Could not start time-lapse: {exc}", "error")
            return

        self._detections_by_frame = []
        self._configure_progress_bar(expected)
        self.lbl_progress.setText("Starting…")

        worker = _TimelapseWorker(runner, self)
        worker.progress.connect(self._on_worker_progress)
        worker.finished_result.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._ctx.set_status("Time-lapse started.", "info")
        worker.start()

    def _configure_progress_bar(self, expected: Optional[int]) -> None:
        if expected is None:
            # Unbounded schedule -> indeterminate ("busy") bar.
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, max(1, expected))
            self.progress_bar.setValue(0)

    def _build_single_position_runner(self, *, interval_s: float,
                                      total_frames: Optional[int],
                                      max_duration_s: Optional[float]):
        from core.cameras.synthetic import make as make_synthetic_camera
        from core.timelapse import TimelapseRunner, TimelapseSchedule

        schedule = TimelapseSchedule(
            interval_s=interval_s, total_frames=total_frames,
            max_duration_s=max_duration_s)
        camera = make_synthetic_camera()
        runner = TimelapseRunner(
            camera=camera, schedule=schedule, out_dir=self._out_dir,
            sample_id="ui3-timelapse", operator="ui3",
            on_progress=self._emit_single_progress,
            on_frame=self._record_frame_detection,
        )
        return runner, schedule.expected_frame_count()

    def _build_multi_position_runner(self, *, interval_s: float,
                                     total_ticks: Optional[int],
                                     max_duration_s: Optional[float]):
        from core.cameras.synthetic import make as make_synthetic_camera
        from core.devices.mock_led import make as make_mock_led
        from core.devices.mock_shutter import make as make_mock_shutter
        from core.devices.mock_stage import make as make_mock_stage
        from core.multi_position_timelapse import (
            MultiPositionSchedule, MultiPositionTimelapseRunner,
        )

        positions = self._read_positions()
        if not positions:
            raise ValueError(
                "add at least one position for multi-position mode")

        schedule = MultiPositionSchedule(
            interval_s=interval_s, positions=positions,
            total_ticks=total_ticks, max_duration_s=max_duration_s)

        camera = make_synthetic_camera()
        stage = make_mock_stage()
        stage.connect()
        shutter = make_mock_shutter()
        shutter.connect()
        led = make_mock_led()
        led.connect()

        runner = MultiPositionTimelapseRunner(
            schedule=schedule, out_dir=self._out_dir,
            camera=camera, stage=stage, shutter=shutter, led=led,
            sample_id="ui3-timelapse-multi", operator="ui3",
            on_tick=self._emit_multi_progress,
        )
        return runner, schedule.expected_total_frames()

    # -- worker -> Qt-signal shims (called on the worker thread) ----------
    def _emit_single_progress(self, frame_idx: int,
                              expected: Optional[int]) -> None:
        if self._worker is not None:
            self._worker.progress.emit(frame_idx + 1, expected)

    def _emit_multi_progress(self, tick: int, n_pos: int) -> None:
        if self._worker is not None:
            self._worker.progress.emit(tick, None)

    def _record_frame_detection(self, frame_idx: int, arr) -> None:
        # No detector wired; keep the per-frame slot present (possibly
        # empty) so link_detections() always sees a well-formed sequence
        # sized to the frames actually captured.
        self._detections_by_frame.append([])

    def _on_stop_clicked(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
        self.btn_stop.setEnabled(False)
        self._ctx.set_status("Stopping time-lapse…", "info")

    def shutdown(self) -> None:
        """Cancel + join the timelapse worker — called by the shell on window
        close so an overnight run doesn't outlive the widget graph."""
        w = self._worker
        if w is not None:
            try:
                w.request_cancel()
                w.wait(3000)
            except Exception:
                pass

    def closeEvent(self, event) -> None:  # noqa: N802
        self.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Worker signal handlers (GUI thread)
    # ------------------------------------------------------------------
    def _on_worker_progress(self, done: int, expected: Optional[int]) -> None:
        if expected is None:
            self.lbl_progress.setText(f"{done} captured…")
        else:
            self.lbl_progress.setText(f"{done} / {expected} captured")
            self.progress_bar.setValue(min(done, self.progress_bar.maximum()))

    def _on_worker_finished(self, result: Any) -> None:
        self._reset_run_state()
        cancelled = bool(getattr(result, "cancelled", False))
        frames = getattr(result, "frames_captured", None)
        if frames is None:
            frames = getattr(result, "frames_total", 0)
        status = "stopped" if cancelled else "complete"
        self.lbl_progress.setText(f"{status}: {frames} frame(s) captured")
        if not self.progress_bar.maximum() == 0:
            self.progress_bar.setValue(self.progress_bar.maximum())
        level = "warn" if cancelled else "ok"
        self._ctx.set_status(f"Time-lapse {status} ({frames} frames).", level)
        self._ctx.toast(f"Time-lapse {status}: {frames} frame(s).", level)

    def _on_worker_failed(self, message: str) -> None:
        self._reset_run_state()
        self.lbl_progress.setText("Failed")
        self._ctx.set_status(f"Time-lapse failed: {message}", "error")
        self._ctx.toast(f"Time-lapse failed: {message}", "error")

    def _reset_run_state(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    # Tracking hook
    # ------------------------------------------------------------------
    def _on_link_detections_clicked(self) -> None:
        from core.tracking import link_detections

        tracks = link_detections(self._detections_by_frame)
        self.lbl_track_result.setText(
            f"{len(tracks)} track(s) across "
            f"{len(self._detections_by_frame)} frame(s).")
        self._ctx.set_status(
            f"Linked {len(tracks)} track(s).", "info")


__all__ = ["TimelapsePanel"]
