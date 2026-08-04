"""DevicePanel — stage XYZ HUD + jog/shutter/LED controls (ui3).

Qt functional parity with ``ui2``'s combined device panel
(``src/ui2/device_panel.py`` — W2 device control + W5 stage HUD, DPG
track, v2.2.0-devices). Same device resolution order, same mm-at-the-
boundary conversion, same jog/step/home semantics, same "missing
device renders disabled, never crashes" rule — only the presentation
changes: Qt widgets + ``QTimer`` polling instead of DPG + a daemon
polling thread + app mailbox.

Device resolution (mirrors ``DevicePanelController.__init__``):

1. Pre-wired handles on the app (``ctx`` has no such slot in ui3, so
   this panel accepts optional ``stage=`` / ``shutter=`` / ``led=``
   constructor kwargs a host window can pass through if it already
   wired real hardware).
2. Lazy ``core.devices.make_device("mock_stage" / "mock_shutter" /
   "mock_led")`` — keeps the panel useful with no hardware bound.
3. ``None`` if even the mock backend import fails — the corresponding
   card renders disabled with "not configured" and every action is a
   guarded no-op. No exception ever reaches the caller.

Layout — three cards, same order as ui2:

* **Stage** — XYZ readout in mm (mono, ``role="readout"``), a step
  preset combo (0.001/0.010/0.100/1.000/5.000 mm — ``JOG_STEPS_MM``
  from ``ui2.device_panel``, so the two UIs can never drift), a 3x3
  jog grid (diagonals in the corners, Home in the centre, X/Y on the
  cardinal cells), a dedicated Z +/- row, and connection status.
* **Shutter** — Open/Close toggle button + status text.
* **Illumination** — On/Off toggle + intensity ``QSlider`` (0-100),
  disabled while the LED is off (matches ui2: slider is usable but a
  laborant only expects it to matter with the LED on — this panel
  keeps it always-enabled when a device is present, since neither the
  ``LEDDevice`` protocol nor ui2's actual widget disables it while
  off; see ``_render_refresh`` there — slider always ``enabled=True``
  when configured).

Polling: ``QTimer`` at ``REFRESH_HZ`` (5 Hz — same cadence as
``ui2.device_panel.REFRESH_HZ``, duplicated as a constant here rather
than imported because ``ui2.device_panel`` hard-imports ``dearpygui``
at module scope, which is not installed in every environment this
Qt-only frontend runs in) ticks on the GUI thread — the mock backends
are sub-millisecond so a direct synchronous read is safe here (no
worker thread needed, matching ui2's rationale for its DPG
render-thread reads). Dirty-check: a fresh snapshot is compared
(``==``, frozen dataclass) against the last one and the HUD widgets
are only touched when something changed, so an idle panel doesn't
force a repaint every 200 ms.

Jog against an axis-limited stage: ``jog()`` reads
``stage.position_um`` and pads any missing axis with 0.0 before
calling ``move_to`` — same tolerance ui2's ``DevicePanelController.jog``
applies for a real 2-axis stage that only returns ``(x, y)``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ui3.context import PanelContext

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants mirrored from ``ui2.device_panel`` (kept in lockstep by hand,
# NOT imported — that module hard-imports ``dearpygui`` at module scope,
# which isn't installed in every environment this Qt-only frontend must
# run in; duplicating five constants + one frozen dataclass is cheaper
# than making the Qt frontend depend on a DPG backend being importable).
# ---------------------------------------------------------------------------

JOG_STEPS_MM: Tuple[float, ...] = (0.001, 0.010, 0.100, 1.000, 5.000)
DEFAULT_STEP_MM: float = 0.100
REFRESH_HZ: float = 5.0


@dataclass(frozen=True)
class DeviceSnapshot:
    stage_position_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    stage_connected: bool = False
    shutter_open: Optional[bool] = None        # None = no device
    shutter_connected: bool = False
    led_on: Optional[bool] = None              # None = no device
    led_intensity_pct: float = 0.0
    led_connected: bool = False
    error: Optional[str] = None


def _try_make(backend_name: str):
    """Best-effort device factory — ``None`` if the backend isn't
    registered or importable (e.g. a slimmed-down deployment)."""
    try:
        from core.devices import make_device
        return make_device(backend_name)
    except Exception as exc:
        _LOG.debug("make_device(%s) failed: %s", backend_name, exc)
        return None


def _step_label(step_mm: float) -> str:
    """Sub-millimetre steps render as um so '0.001 mm' doesn't read as
    a typo — mirrors ``ui2.device_panel._step_label``."""
    if step_mm < 1.0:
        return f"{step_mm * 1000:g} um"
    return f"{step_mm:g} mm"


def _parse_step_label(label: str) -> float:
    label = label.strip()
    if label.endswith("um"):
        return float(label[:-2].strip()) / 1000.0
    if label.endswith("mm"):
        return float(label[:-2].strip())
    return DEFAULT_STEP_MM


def _fmt_mm(v: float) -> str:
    return f"{v:+8.4f}"


class DevicePanel(QWidget):
    """Stage HUD + jog/shutter/LED controls, built against the frozen
    ``PanelContext``. Devices are resolved independently of the
    reconstruction pipeline — this panel never touches ``ctx.bridge``
    or ``ctx.get_params()``; it only uses ``ctx.palette`` (implicitly,
    via the shared QSS) and ``ctx.set_status`` / ``ctx.toast`` for
    feedback.
    """

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None,
                 *, stage=None, shutter=None, led=None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self.step_mm: float = DEFAULT_STEP_MM
        self.snapshot: DeviceSnapshot = DeviceSnapshot()
        self._last_snapshot: Optional[DeviceSnapshot] = None

        # Resolution: caller-supplied handle wins, else lazy mock, else
        # None (disabled card). Same order as DevicePanelController.
        self.stage = stage or _try_make("mock_stage")
        self.shutter = shutter or _try_make("mock_shutter")
        self.led = led or _try_make("mock_led")

        for d in (self.stage, self.shutter, self.led):
            if d is not None and not getattr(d, "is_connected", True):
                try:
                    d.connect()
                except Exception as exc:  # pragma: no cover - defensive
                    _LOG.warning("device connect failed: %s", exc)

        self._build_ui()
        self._refresh_snapshot()
        self._render()

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000.0 / REFRESH_HZ))
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        heading = QLabel("Devices")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        root.addWidget(self._build_stage_card())
        root.addWidget(self._build_shutter_card())
        root.addWidget(self._build_led_card())
        root.addStretch(1)

        self.lbl_connection = QLabel("", self)
        self.lbl_connection.setProperty("role", "muted")
        self.lbl_connection.setWordWrap(True)
        root.addWidget(self.lbl_connection)

    def _build_stage_card(self) -> QFrame:
        card = QFrame(self)
        card.setProperty("role", "card")
        lay = QVBoxLayout(card)

        title = QLabel("STAGE", card)
        title.setProperty("role", "muted")
        lay.addWidget(title)

        # XYZ HUD — mono readout, one row per axis.
        hud = QGridLayout()
        self.lbl_x = QLabel(_fmt_mm(0.0), card)
        self.lbl_y = QLabel(_fmt_mm(0.0), card)
        self.lbl_z = QLabel(_fmt_mm(0.0), card)
        for lbl in (self.lbl_x, self.lbl_y, self.lbl_z):
            lbl.setProperty("role", "readout")
        for row, (axis, lbl) in enumerate(
                (("X", self.lbl_x), ("Y", self.lbl_y), ("Z", self.lbl_z))):
            axis_lbl = QLabel(axis, card)
            axis_lbl.setProperty("role", "muted")
            unit_lbl = QLabel("mm", card)
            unit_lbl.setProperty("role", "muted")
            hud.addWidget(axis_lbl, row, 0)
            hud.addWidget(lbl, row, 1)
            hud.addWidget(unit_lbl, row, 2)
        lay.addLayout(hud)

        self.lbl_stage_status = QLabel("not configured", card)
        self.lbl_stage_status.setProperty("role", "muted")
        lay.addWidget(self.lbl_stage_status)

        # Step preset.
        step_row = QHBoxLayout()
        step_label = QLabel("Step", card)
        step_label.setProperty("role", "muted")
        step_row.addWidget(step_label)
        self.cmb_step = QComboBox(card)
        self.cmb_step.addItems([_step_label(s) for s in JOG_STEPS_MM])
        self.cmb_step.setCurrentText(_step_label(DEFAULT_STEP_MM))
        self.cmb_step.currentTextChanged.connect(self._on_step_changed)
        step_row.addWidget(self.cmb_step, stretch=1)
        lay.addLayout(step_row)

        # 3x3 jog grid — diagonals in the corners, Home centre.
        grid = QGridLayout()
        grid.setSpacing(4)
        self.btn_diag_nxpy = QPushButton("\\  -X+Y", card)
        self.btn_yplus = QPushButton("Y +", card)
        self.btn_diag_pxpy = QPushButton("/  +X+Y", card)
        self.btn_xminus = QPushButton("X -", card)
        self.btn_home = QPushButton("Home", card)
        self.btn_xplus = QPushButton("X +", card)
        self.btn_diag_nxny = QPushButton("/  -X-Y", card)
        self.btn_yminus = QPushButton("Y -", card)
        self.btn_diag_pxny = QPushButton("\\  +X-Y", card)

        grid.addWidget(self.btn_diag_nxpy, 0, 0)
        grid.addWidget(self.btn_yplus, 0, 1)
        grid.addWidget(self.btn_diag_pxpy, 0, 2)
        grid.addWidget(self.btn_xminus, 1, 0)
        grid.addWidget(self.btn_home, 1, 1)
        grid.addWidget(self.btn_xplus, 1, 2)
        grid.addWidget(self.btn_diag_nxny, 2, 0)
        grid.addWidget(self.btn_yminus, 2, 1)
        grid.addWidget(self.btn_diag_pxny, 2, 2)
        lay.addLayout(grid)

        self.btn_yplus.clicked.connect(lambda: self._jog("y", +1))
        self.btn_yminus.clicked.connect(lambda: self._jog("y", -1))
        self.btn_xplus.clicked.connect(lambda: self._jog("x", +1))
        self.btn_xminus.clicked.connect(lambda: self._jog("x", -1))
        self.btn_home.clicked.connect(self._on_home_clicked)
        self.btn_diag_nxpy.clicked.connect(lambda: self._jog_diag(-1, +1))
        self.btn_diag_pxpy.clicked.connect(lambda: self._jog_diag(+1, +1))
        self.btn_diag_nxny.clicked.connect(lambda: self._jog_diag(-1, -1))
        self.btn_diag_pxny.clicked.connect(lambda: self._jog_diag(+1, -1))

        # Z gets its own row (most-touched axis).
        z_row = QHBoxLayout()
        z_label = QLabel("Z", card)
        z_label.setProperty("role", "muted")
        z_row.addWidget(z_label)
        self.btn_zminus = QPushButton("Z -", card)
        self.btn_zplus = QPushButton("Z +", card)
        self.btn_zminus.clicked.connect(lambda: self._jog("z", -1))
        self.btn_zplus.clicked.connect(lambda: self._jog("z", +1))
        z_row.addWidget(self.btn_zminus)
        z_row.addWidget(self.btn_zplus)
        lay.addLayout(z_row)

        self._stage_buttons = [
            self.btn_diag_nxpy, self.btn_yplus, self.btn_diag_pxpy,
            self.btn_xminus, self.btn_home, self.btn_xplus,
            self.btn_diag_nxny, self.btn_yminus, self.btn_diag_pxny,
            self.btn_zminus, self.btn_zplus,
        ]
        return card

    def _build_shutter_card(self) -> QFrame:
        card = QFrame(self)
        card.setProperty("role", "card")
        lay = QVBoxLayout(card)

        title = QLabel("SHUTTER", card)
        title.setProperty("role", "muted")
        lay.addWidget(title)

        row = QHBoxLayout()
        self.btn_shutter = QPushButton("Open", card)
        self.btn_shutter.clicked.connect(self._on_shutter_clicked)
        row.addWidget(self.btn_shutter)
        self.lbl_shutter_status = QLabel("closed", card)
        self.lbl_shutter_status.setProperty("role", "muted")
        row.addWidget(self.lbl_shutter_status)
        row.addStretch(1)
        lay.addLayout(row)
        return card

    def _build_led_card(self) -> QFrame:
        card = QFrame(self)
        card.setProperty("role", "card")
        lay = QVBoxLayout(card)

        title = QLabel("ILLUMINATION", card)
        title.setProperty("role", "muted")
        lay.addWidget(title)

        row = QHBoxLayout()
        self.btn_led = QPushButton("On", card)
        self.btn_led.clicked.connect(self._on_led_clicked)
        row.addWidget(self.btn_led)
        self.lbl_led_status = QLabel("off", card)
        self.lbl_led_status.setProperty("role", "muted")
        row.addWidget(self.lbl_led_status)
        row.addStretch(1)
        lay.addLayout(row)

        self.sld_intensity = QSlider(Qt.Horizontal, card)
        self.sld_intensity.setRange(0, 100)
        self.sld_intensity.setValue(0)
        self.sld_intensity.valueChanged.connect(self._on_intensity_changed)
        lay.addWidget(self.sld_intensity)
        return card

    # ------------------------------------------------------------------
    # Snapshot / polling
    # ------------------------------------------------------------------
    def _refresh_snapshot(self) -> DeviceSnapshot:
        sx_mm = sy_mm = sz_mm = 0.0
        stage_conn = False
        if self.stage is not None:
            try:
                pos = self.stage.position_um
                sx_mm = pos[0] / 1000.0
                sy_mm = pos[1] / 1000.0
                sz_mm = pos[2] / 1000.0 if len(pos) >= 3 else 0.0
                stage_conn = bool(getattr(self.stage, "is_connected", True))
            except Exception as exc:
                _LOG.debug("stage poll failed: %s", exc)

        sh_open = None
        sh_conn = False
        if self.shutter is not None:
            try:
                sh_open = bool(self.shutter.is_open)
                sh_conn = bool(getattr(self.shutter, "is_connected", True))
            except Exception as exc:
                _LOG.debug("shutter poll failed: %s", exc)

        led_on = None
        led_pct = 0.0
        led_conn = False
        if self.led is not None:
            try:
                led_on = bool(self.led.is_on)
                led_pct = float(self.led.intensity_percent)
                led_conn = bool(getattr(self.led, "is_connected", True))
            except Exception as exc:
                _LOG.debug("led poll failed: %s", exc)

        snap = DeviceSnapshot(
            stage_position_mm=(sx_mm, sy_mm, sz_mm),
            stage_connected=stage_conn,
            shutter_open=sh_open,
            shutter_connected=sh_conn,
            led_on=led_on,
            led_intensity_pct=led_pct,
            led_connected=led_conn,
        )
        self.snapshot = snap
        return snap

    def _on_tick(self) -> None:
        snap = self._refresh_snapshot()
        # Dirty-check — only touch widgets when something changed, so
        # an idle panel doesn't repaint every tick.
        if snap != self._last_snapshot:
            self._last_snapshot = snap
            self._render()

    # ------------------------------------------------------------------
    # Render — pull from snapshot, push to widgets.
    # ------------------------------------------------------------------
    def _render(self) -> None:
        snap = self.snapshot

        self.lbl_x.setText(_fmt_mm(snap.stage_position_mm[0]))
        self.lbl_y.setText(_fmt_mm(snap.stage_position_mm[1]))
        self.lbl_z.setText(_fmt_mm(snap.stage_position_mm[2]))

        stage_present = self.stage is not None
        for btn in self._stage_buttons:
            btn.setEnabled(stage_present)
        self.cmb_step.setEnabled(stage_present)
        if snap.stage_connected:
            self.lbl_stage_status.setText("linked")
        elif not stage_present:
            self.lbl_stage_status.setText("not configured")
        else:
            self.lbl_stage_status.setText("offline")

        # Shutter
        shutter_present = self.shutter is not None
        self.btn_shutter.setEnabled(shutter_present)
        if snap.shutter_open is None:
            self.btn_shutter.setText("Open")
            self.lbl_shutter_status.setText("not configured")
        elif snap.shutter_open:
            self.btn_shutter.setText("Close")
            self.lbl_shutter_status.setText("open")
        else:
            self.btn_shutter.setText("Open")
            self.lbl_shutter_status.setText("closed")

        # LED
        led_present = self.led is not None
        self.btn_led.setEnabled(led_present)
        self.sld_intensity.setEnabled(led_present)
        if snap.led_on is None:
            self.btn_led.setText("On")
            self.lbl_led_status.setText("not configured")
        elif snap.led_on:
            self.btn_led.setText("Off")
            self.lbl_led_status.setText(f"on  {snap.led_intensity_pct:.1f} %")
        else:
            self.btn_led.setText("On")
            self.lbl_led_status.setText("off")
        # Keep slider synced without re-entrant valueChanged storms.
        if abs(self.sld_intensity.value() - snap.led_intensity_pct) > 0.5:
            self.sld_intensity.blockSignals(True)
            self.sld_intensity.setValue(int(round(snap.led_intensity_pct)))
            self.sld_intensity.blockSignals(False)

        parts = [
            f"stage: {_conn_word(self.stage, snap.stage_connected)}",
            f"shutter: {_conn_word(self.shutter, snap.shutter_connected)}",
            f"LED: {_conn_word(self.led, snap.led_connected)}",
        ]
        self.lbl_connection.setText("  ·  ".join(parts))

    # ------------------------------------------------------------------
    # Stage actions
    # ------------------------------------------------------------------
    def _jog(self, axis: str, direction: int) -> None:
        if self.stage is None:
            return
        delta = direction * self.step_mm * 1000.0  # mm -> um
        try:
            pos = tuple(self.stage.position_um)
        except Exception as exc:
            _LOG.warning("jog: could not read stage position: %s", exc)
            self._ctx.set_status(f"Stage read failed: {exc}", "error")
            return
        if len(pos) < 3:
            # Pad missing axes instead of no-op — matches ui2's
            # tolerance for a 2-axis stage.
            pos = pos + (0.0,) * (3 - len(pos))
        x, y, z = pos[:3]
        dx = delta if axis == "x" else 0.0
        dy = delta if axis == "y" else 0.0
        dz = delta if axis == "z" else 0.0
        try:
            self.stage.move_to(x + dx, y + dy, z + dz)
        except Exception as exc:
            _LOG.warning("jog failed: %s", exc)
            self._ctx.set_status(f"Jog failed: {exc}", "error")
        self._refresh_snapshot()
        self._render()

    def _jog_diag(self, sx: int, sy: int) -> None:
        self._jog("x", sx)
        self._jog("y", sy)

    def _on_home_clicked(self) -> None:
        if self.stage is None:
            return
        try:
            self.stage.home()
        except Exception as exc:
            _LOG.warning("home failed: %s", exc)
            self._ctx.set_status(f"Home failed: {exc}", "error")
        self._refresh_snapshot()
        self._render()
        self._ctx.set_status("Stage homed.", "ok")

    def _on_step_changed(self, label: str) -> None:
        step = _parse_step_label(label)
        if step in JOG_STEPS_MM:
            self.step_mm = step
        else:
            self.step_mm = min(JOG_STEPS_MM, key=lambda s: abs(s - step))

    # ------------------------------------------------------------------
    # Shutter / LED actions
    # ------------------------------------------------------------------
    def _on_shutter_clicked(self) -> None:
        if self.shutter is None:
            return
        try:
            if self.shutter.is_open:
                self.shutter.close()
            else:
                self.shutter.open()
        except Exception as exc:
            _LOG.warning("shutter toggle failed: %s", exc)
            self._ctx.set_status(f"Shutter toggle failed: {exc}", "error")
        self._refresh_snapshot()
        self._render()

    def _on_led_clicked(self) -> None:
        if self.led is None:
            return
        try:
            if self.led.is_on:
                self.led.off()
            else:
                self.led.on()
        except Exception as exc:
            _LOG.warning("led toggle failed: %s", exc)
            self._ctx.set_status(f"LED toggle failed: {exc}", "error")
        self._refresh_snapshot()
        self._render()

    def _on_intensity_changed(self, value: int) -> None:
        if self.led is None:
            return
        try:
            self.led.set_intensity(max(0.0, min(100.0, float(value))))
        except Exception as exc:
            _LOG.warning("led intensity failed: %s", exc)
            self._ctx.set_status(f"LED intensity failed: {exc}", "error")
        self._refresh_snapshot()
        self._render()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._timer.stop()
        super().closeEvent(event)


def _conn_word(device, connected: bool) -> str:
    if device is None:
        return "—"
    return "linked" if connected else "offline"


__all__ = ["DevicePanel"]
