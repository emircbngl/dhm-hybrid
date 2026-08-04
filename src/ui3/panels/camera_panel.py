"""CameraPanel — live camera feed + TIFF stack recording (ui3).

Mirrors ``ui2``'s camera live-feed section (``src/ui2/app.py``
``_build_camera_source`` / ``_on_camera_start`` / ``_on_camera_stop`` /
``_on_camera_record`` / ``_post_camera_frame`` / ``_handle_camera_frame``) and
reuses the exact same compute-adjacent pieces:

* :mod:`ui2.camera_feed` — ``CameraSource`` Protocol, ``SyntheticCamera``,
  ``AcquisitionThread``, ``TiffStackRecorder``. These are framework-free (the
  acquisition loop is a plain ``threading.Thread``) so ui3 reuses them as-is,
  same as ``WorkerBridge`` reuses ``ui2.workers``/``ui2.reconstruction``.
* :mod:`core.cameras` — the backend registry (``available_backends`` /
  ``make_camera``). The dropdown always offers ``'synthetic'`` (ui2's
  hardcoded default) plus whatever real backends have their SDK installed on
  this machine.

Qt-native adaptation (ui2 -> ui3, unchanged physics/logic):

* ui2's ``AcquisitionThread`` calls back on the *worker* thread, and ui2's own
  DPG render loop marshals via a mailbox (``_post_mailbox``) polled on a
  timer. ui3 has no such mailbox — instead this panel gives the thread two
  plain-Python callables that immediately ``.emit()`` Qt signals owned by this
  panel (``_frame_ready`` / ``_fps_ready``). Since the receiving slots are
  connected with the default (auto) connection type and this ``QObject``
  lives on the GUI thread, Qt queues the delivery onto the GUI thread exactly
  like ``WorkerBridge`` does for the science drivers — no busy-loop, no
  polling, no behavioural difference from ui2's eventual "paint the newest
  frame" contract.
* "Push to input" replaces ui2's implicit always-on
  ``panel_input.set_image(frame)`` on every frame — continuously repainting
  the *shared* input panel from a live feed would fight any hologram a user
  has loaded into the same grid slot. This panel keeps its own preview
  (always live) and offers an explicit button to snapshot the latest frame
  into ``ctx.show_in_panel('input', frame)``, matching ui2's "Snapshot live
  frame" affordance (``_latest_live_frame`` cache) without stealing the grid.
* Record mirrors ui2's ``_on_camera_record`` exactly: pick a destination via a
  save dialog, attach a fresh ``TiffStackRecorder``, stop+restart the feed if
  it was already running so the recorder gets every frame from t=0.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.cameras import available_backends, make_camera
from core.drivers.camera_feed import (
    AcquisitionThread,
    CameraSource,
    SyntheticCamera,
    TiffStackRecorder,
)
from ui3.context import PanelContext
from ui3.design import Space
from ui3.viewport import ImagePanel

_LOG = logging.getLogger(__name__)

# Always offered regardless of what core.cameras.available_backends() finds —
# ui2's default source with no hardware attached (parity with
# ``ui2.app.DhmApp._build_camera_source``).
_SYNTHETIC = "synthetic"


class _AcquisitionSignals(QObject):
    """Tiny QObject the acquisition thread's callbacks emit through.

    Kept separate from ``CameraPanel`` only so the callables handed to
    :class:`AcquisitionThread` are plain bound-signal emits (cheap, GIL-safe)
    — the actual frame/FPS handling happens in slots on ``CameraPanel``
    itself, invoked on the GUI thread via Qt's queued cross-thread delivery.
    """
    frame_ready = Signal(object)   # np.ndarray
    fps_ready = Signal(float)
    error = Signal(str)            # acquisition thread died (start/grab)


def _default_record_path() -> Path:
    return Path.home() / f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"


class CameraPanel(QWidget):
    """Backend picker + start/stop/record controls + a live preview."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._thread: Optional[AcquisitionThread] = None
        self._source: Optional[CameraSource] = None
        self._recorder: Optional[TiffStackRecorder] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._fps: float = 0.0

        self._signals = _AcquisitionSignals()
        self._signals.frame_ready.connect(self._on_frame, Qt.QueuedConnection)
        self._signals.fps_ready.connect(self._on_fps, Qt.QueuedConnection)
        self._signals.error.connect(self._on_acq_error, Qt.QueuedConnection)

        self._build_ui()
        self._populate_backends()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.md)
        root.setSpacing(Space.md)

        heading = QLabel("Camera")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        # -- Backend + controls card -------------------------------------
        card = QFrame(self)
        card.setProperty("role", "card")
        form = QFormLayout(card)

        self.cmb_backend = QComboBox(card)
        form.addRow("Backend", self.cmb_backend)

        root.addWidget(card)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start", self)
        self.btn_start.setProperty("role", "primary")
        self.btn_start.clicked.connect(self._on_start_clicked)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop", self)
        self.btn_stop.setProperty("role", "danger")  # match timelapse Stop
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        self.btn_record = QPushButton("Record…", self)
        self.btn_record.clicked.connect(self._on_record_clicked)
        btn_row.addWidget(self.btn_record)
        root.addLayout(btn_row)

        # -- Status readout ------------------------------------------------
        self.lbl_status = QLabel("Stopped", self)
        self.lbl_status.setProperty("role", "readout")
        root.addWidget(self.lbl_status)

        # -- Live preview --------------------------------------------------
        palette = self._ctx.palette
        self.preview = ImagePanel("Live preview", palette) if palette is not None \
            else ImagePanel("Live preview", _fallback_palette())
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.preview, stretch=1)

        self.btn_push_to_input = QPushButton("Push to input", self)
        self.btn_push_to_input.clicked.connect(self._on_push_to_input_clicked)
        self.btn_push_to_input.setEnabled(False)
        root.addWidget(self.btn_push_to_input)

    def _populate_backends(self) -> None:
        names = [_SYNTHETIC]
        for info in available_backends():
            if info.name not in names:
                names.append(info.name)
        self.cmb_backend.clear()
        self.cmb_backend.addItems(names)
        idx = self.cmb_backend.findText(_SYNTHETIC)
        if idx >= 0:
            self.cmb_backend.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Backend construction
    # ------------------------------------------------------------------
    def _build_source(self, name: str) -> CameraSource:
        if name == _SYNTHETIC:
            return SyntheticCamera(size_px=512, target_fps=30.0)
        return make_camera(name)

    # ------------------------------------------------------------------
    # Start / stop / record
    # ------------------------------------------------------------------
    def _on_start_clicked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._ctx.set_status("Camera already running.", "info")
            return
        name = self.cmb_backend.currentText() or _SYNTHETIC
        try:
            source = self._build_source(name)
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            self._ctx.set_status(f"Camera init failed: {exc}", "error")
            self._ctx.toast(f"Camera init failed: {exc}", "error")
            return
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.exception("camera init raised unexpectedly")
            self._ctx.set_status(f"Camera init failed: {exc}", "error")
            self._ctx.toast(f"Camera init failed: {exc}", "error")
            return

        self._source = source
        self._thread = AcquisitionThread(
            source=source,
            on_frame=self._signals.frame_ready.emit,
            on_fps=self._signals.fps_ready.emit,
            recorder=self._recorder,
            on_error=self._signals.error.emit,
        )
        self._thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._ctx.set_status(
            f"Camera running ({type(source).__name__}).", "ok")
        self._ctx.toast("Camera live feed started.", "ok")

    def _on_stop_clicked(self) -> None:
        thread = self._thread
        if thread is None:
            self._ctx.set_status("No camera is running.", "info")
            return
        thread.stop()
        # Daemon thread, so joining isn't required for a clean process exit —
        # but a deterministic "Stop" affordance should actually leave the
        # feed quiescent when the caller checks right after. Join with a
        # short bound (the acquisition loop's sleep chunks are <=100ms).
        thread.join(timeout=2.0)
        self._thread = None
        self._source = None
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                _LOG.debug("recorder stop raised", exc_info=True)
        self._recorder = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_status_line()
        self._ctx.set_status("Camera stopped.", "info")

    def _on_record_clicked(self) -> None:
        default = _default_record_path()
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Record camera to TIFF stack", str(default),
            "TIFF stack (*.tif *.tiff)")
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() not in (".tif", ".tiff"):
            path = path.with_suffix(".tif")

        was_running = self._thread is not None and self._thread.is_alive()
        if was_running:
            self._on_stop_clicked()

        recorder = TiffStackRecorder(path)
        try:
            recorder.start()
        except Exception as exc:
            self._ctx.set_status(f"Recorder init failed: {exc}", "error")
            self._ctx.toast(f"Recorder init failed: {exc}", "error")
            return
        self._recorder = recorder
        self._ctx.set_status(f"Recording to {path.name}…", "info")
        self._on_start_clicked()

    def _on_push_to_input_clicked(self) -> None:
        if self._latest_frame is None:
            self._ctx.set_status("No live frame to push yet.", "warn")
            return
        self._ctx.show_in_panel("input", self._latest_frame)
        self._ctx.set_status("Pushed live frame to input.", "ok")

    # ------------------------------------------------------------------
    # Frame / fps slots (GUI thread, via Qt queued signal delivery)
    # ------------------------------------------------------------------
    def _on_frame(self, frame: object) -> None:
        arr = np.asarray(frame, dtype=np.float32)
        self._latest_frame = arr
        try:
            self.preview.set_image(arr)
        except Exception:
            _LOG.debug("camera preview paint failed", exc_info=True)
        self.btn_push_to_input.setEnabled(True)
        self._refresh_status_line()

    def _on_fps(self, fps: object) -> None:
        self._fps = float(fps)
        self._refresh_status_line()

    def _on_acq_error(self, message: str) -> None:
        """The acquisition thread died mid-run (B-109). Surface it and reset
        the UI to 'stopped' so the user isn't left with a frozen frame while
        Start/Record still read as armed. The thread already ran its own
        source/recorder cleanup before dying."""
        self._thread = None
        self._source = None
        self._recorder = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_status_line()
        self._ctx.set_status(f"Camera stopped — {message}", "error")
        self._ctx.toast(f"Camera stopped: {message}", "error")

    def _refresh_status_line(self) -> None:
        running = self._thread is not None and self._thread.is_alive()
        state = "LIVE" if running else "Stopped"
        rec = f"  REC ● {self._recorder.frames_written} frames" \
            if self._recorder is not None else ""
        self.lbl_status.setText(f"{state}   {self._fps:.1f} fps{rec}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._thread is not None:
            self._on_stop_clicked()
        super().closeEvent(event)


def _fallback_palette():
    from ui3.design import DARK
    return DARK


__all__ = ["CameraPanel"]
