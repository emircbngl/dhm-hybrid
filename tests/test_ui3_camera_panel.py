"""Tests for ``ui3.panels.camera_panel.CameraPanel``.

Offscreen Qt (QT_QPA_PLATFORM=offscreen), real ``WorkerBridge`` (unused by
this panel directly, but constructed the same way as the other ui3 panel
tests for a consistent fixture set) driven against the real
``ui2.camera_feed`` acquisition pipeline with the synthetic backend — no
compute is mocked, this proves the panel actually starts a real
acquisition thread, receives real frames, and stops cleanly.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DHM_USER", "ci")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


class _FakeContext:
    """Minimal stand-in for ``ui3.context.PanelContext`` — mirrors the
    pattern in ``tests/test_ui3_focus_panel.py``. This panel doesn't
    touch ``get_params``/``last_recon``/``get_field`` so those are
    stubbed trivially."""

    def __init__(self, bridge, hologram_path: Optional[Path] = None):
        self.bridge = bridge
        self._hologram_path = hologram_path
        self.status_log: List[tuple] = []
        self.toast_log: List[tuple] = []
        self.shown: List[tuple] = []
        self.palette = None

    def get_params(self):
        return None

    def params_changed(self):
        pass

    def hologram_path(self):
        return self._hologram_path

    def last_recon(self):
        return None

    def last_depth(self):
        return None

    def get_field(self, target: str):
        return None

    def set_status(self, text: str, level: str = "info") -> None:
        self.status_log.append((text, level))

    def toast(self, text: str, level: str = "info") -> None:
        self.toast_log.append((text, level))

    def show_in_panel(self, target: str, array, **kw) -> None:
        self.shown.append((target, array))


def _pump_until(qapp, predicate, deadline_s: float = 20.0) -> None:
    deadline = time.time() + deadline_s
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)


@pytest.fixture()
def bridge():
    from ui3.bridge import WorkerBridge
    b = WorkerBridge()
    yield b
    b.shutdown()


@pytest.fixture()
def ctx(bridge):
    return _FakeContext(bridge)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_panel_builds_with_synthetic_default(qapp, ctx):
    from ui3.panels.camera_panel import CameraPanel

    panel = CameraPanel(ctx)
    try:
        items = [panel.cmb_backend.itemText(i)
                 for i in range(panel.cmb_backend.count())]
        assert "synthetic" in items
        assert panel.cmb_backend.currentText() == "synthetic"
        assert panel.btn_start.isEnabled()
        assert not panel.btn_stop.isEnabled()
        assert not panel.btn_push_to_input.isEnabled()
        assert "Stopped" in panel.lbl_status.text()
    finally:
        panel.close()


def test_backend_list_matches_core_cameras_registry(qapp, ctx):
    from core.cameras import available_backends
    from ui3.panels.camera_panel import CameraPanel

    panel = CameraPanel(ctx)
    try:
        items = {panel.cmb_backend.itemText(i)
                 for i in range(panel.cmb_backend.count())}
        expected = {"synthetic"} | {b.name for b in available_backends()}
        assert items == expected
    finally:
        panel.close()


# ---------------------------------------------------------------------------
# Start / stop lifecycle (real AcquisitionThread + SyntheticCamera)
# ---------------------------------------------------------------------------

def test_start_produces_frames_and_stop_is_clean(qapp, ctx):
    from ui3.panels.camera_panel import CameraPanel

    panel = CameraPanel(ctx)
    try:
        panel.btn_start.click()
        assert not panel.btn_start.isEnabled()
        assert panel.btn_stop.isEnabled()

        _pump_until(qapp, lambda: panel._latest_frame is not None)
        assert panel._latest_frame is not None
        assert panel.preview.has_image()
        assert panel.btn_push_to_input.isEnabled()
        assert isinstance(panel._latest_frame, np.ndarray)
        assert panel._latest_frame.ndim == 2

        # A few more frames should keep arriving over a short window.
        n0 = 0
        first_frame = panel._latest_frame.copy()
        _pump_until(qapp, lambda: not np.array_equal(
            panel._latest_frame, first_frame), deadline_s=3.0)

        panel.btn_stop.click()
        assert panel.btn_start.isEnabled()
        assert not panel.btn_stop.isEnabled()
        assert panel._thread is None
        assert "Stopped" in panel.lbl_status.text()
    finally:
        panel.close()


def test_start_when_already_running_warns_without_restarting(qapp, ctx):
    from ui3.panels.camera_panel import CameraPanel

    panel = CameraPanel(ctx)
    try:
        panel.btn_start.click()
        _pump_until(qapp, lambda: panel._latest_frame is not None)
        thread_before = panel._thread

        # ``btn_start`` is disabled once running (a real user can't
        # double-click it) — exercise the guard directly, same as
        # driving the handler ui2's own re-entrancy check protects.
        panel._on_start_clicked()
        assert panel._thread is thread_before
        assert any("already running" in t.lower() for t, _ in ctx.status_log)

        panel.btn_stop.click()
    finally:
        panel.close()


def test_stop_without_running_camera_warns(qapp, ctx):
    from ui3.panels.camera_panel import CameraPanel

    panel = CameraPanel(ctx)
    try:
        # ``btn_stop`` starts disabled (no camera running) — call the
        # handler directly to exercise the "nothing to stop" guard.
        panel._on_stop_clicked()
        assert any("no camera is running" in t.lower() for t, _ in ctx.status_log)
    finally:
        panel.close()


def test_push_to_input_forwards_latest_frame(qapp, ctx):
    from ui3.panels.camera_panel import CameraPanel

    panel = CameraPanel(ctx)
    try:
        panel.btn_start.click()
        _pump_until(qapp, lambda: panel._latest_frame is not None)

        panel.btn_push_to_input.click()

        assert len(ctx.shown) == 1
        target, arr = ctx.shown[0]
        assert target == "input"
        assert arr is not None

        panel.btn_stop.click()
    finally:
        panel.close()


def test_push_to_input_without_frame_warns(qapp, ctx):
    from ui3.panels.camera_panel import CameraPanel

    panel = CameraPanel(ctx)
    try:
        # ``btn_push_to_input`` starts disabled (no frame yet) — call
        # the handler directly to exercise the "nothing to push" guard.
        panel._on_push_to_input_clicked()
        assert ctx.shown == []
        assert any(lvl == "warn" for _, lvl in ctx.status_log)
    finally:
        panel.close()


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def test_record_writes_tiff_stack(qapp, ctx, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from ui3.panels.camera_panel import CameraPanel

    out_path = tmp_path / "camera_test.tif"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **kw: (str(out_path), "")))

    panel = CameraPanel(ctx)
    try:
        panel.btn_record.click()
        assert panel._recorder is not None
        assert panel.btn_stop.isEnabled()

        _pump_until(qapp, lambda: panel._recorder is not None
                    and panel._recorder.frames_written >= 3)

        panel.btn_stop.click()
        assert out_path.exists()

        import tifffile
        with tifffile.TiffFile(str(out_path)) as tf:
            assert len(tf.pages) >= 3
    finally:
        panel.close()


def test_record_cancelled_dialog_does_nothing(qapp, ctx, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from ui3.panels.camera_panel import CameraPanel

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **kw: ("", "")))

    panel = CameraPanel(ctx)
    try:
        panel.btn_record.click()
        assert panel._recorder is None
        assert panel._thread is None
    finally:
        panel.close()
