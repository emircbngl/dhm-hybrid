"""AI assistant chat panel.

Right-side ``QDockWidget`` holding:
  * a chat history (``QTextBrowser`` rendering rich tool-call blocks),
  * a multi-line input box,
  * Send / Stop buttons,
  * a settings cog opening :class:`AISettingsDialog`,
  * a small "● Connected"/"● Offline" indicator.

The panel owns the :class:`LocalLLMClient`, the :class:`ToolRegistry`,
and the per-turn :class:`ToolContext`. It instantiates one
:class:`AIWorker` per user message and listens to its signals.

GUI-thread tool callables (``invoke_recon`` etc.) are bound methods on
the panel; they post work to ``MainWindow``'s pipeline workers via Qt
signals and block the AI thread until the worker emits its
``*_completed`` signal. The blocking primitive is a per-call
``threading.Event`` plus a one-shot signal connection — simpler than
``QMetaObject.invokeMethod`` with a future, and fits the existing
worker contracts cleanly.
"""
from __future__ import annotations

import html
import json
import logging
import threading
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.ai.client import LLMConfig, LocalLLMClient
from core.ai.context import StateSnapshot
from core.ai.protocol import ChatMessage, ToolCall
from core.ai.tool_impls import build_tool_registry
from core.ai.tools import ToolContext
from core.audit import get_audit_log
from core.errors import ErrorEvent, get_error_center
from core.sample_map import SampleMap
from core.settings_schema import AIDefaults
from core.stage import MockStage, StageInterface
from gui.dialogs.ai_settings_dialog import AISettingsDialog
from gui.workers.ai_worker import AIWorker

_LOG = logging.getLogger(__name__)


# Heuristic timeout for a pipeline call placed via the panel — long
# autofocus runs on a 2048² hologram can take ~8 s; we let it run
# until the underlying worker signals completion. Acts as a safety net
# only.
_PIPELINE_HARD_TIMEOUT_S = 600.0


class AIPanel(QDockWidget):
    """The right-dock chat panel."""

    settings_changed = Signal(object)   # emits new AIDefaults

    def __init__(self, main_window: Any, parent=None) -> None:
        super().__init__("AI Assistant", parent)
        self.setObjectName("ai_panel_dock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
        )

        self._main_window = main_window
        self._registry = build_tool_registry()
        self._stage: StageInterface = MockStage()
        self._sample_map = SampleMap()
        self._history: list[ChatMessage] = []
        self._worker: Optional[AIWorker] = None

        # Resolve initial AI config — main_window may pass a settings
        # blob in; otherwise defaults.
        ai_settings = self._initial_settings()
        self._ai_settings: AIDefaults = ai_settings
        self._client = LocalLLMClient(self._llm_config_from_settings(ai_settings))

        # Cache populated by main_window via ``set_state_*`` setters —
        # the snapshot the worker reads is built from this on each Send.
        self._cache_loaded_path: Optional[str] = None
        self._cache_loaded_shape: Optional[tuple[int, int]] = None
        self._cache_loaded_dtype: Optional[str] = None
        self._cache_recon_summary: Optional[dict] = None
        self._cache_af_summary: Optional[dict] = None
        self._cache_qpi_summary: Optional[dict] = None
        self._cache_depth_summary: Optional[dict] = None

        self._build_ui()
        self._refresh_health()

    # ------------------------------------------------------------------
    # Public surface (used by main_window to feed state cache)
    # ------------------------------------------------------------------

    def set_loaded(self, path: Optional[str], array: Optional[np.ndarray]) -> None:
        self._cache_loaded_path = path
        if array is None:
            self._cache_loaded_shape = None
            self._cache_loaded_dtype = None
        else:
            try:
                shape = tuple(int(x) for x in array.shape[-2:])
                self._cache_loaded_shape = (shape[0], shape[1]) if len(shape) >= 2 else None
            except Exception:  # noqa: BLE001
                self._cache_loaded_shape = None
            self._cache_loaded_dtype = str(array.dtype) if hasattr(array, "dtype") else None

    def set_recon_summary(self, summary: Optional[dict]) -> None:
        self._cache_recon_summary = dict(summary) if summary else None

    def set_af_summary(self, summary: Optional[dict]) -> None:
        self._cache_af_summary = dict(summary) if summary else None

    def set_qpi_summary(self, summary: Optional[dict]) -> None:
        self._cache_qpi_summary = dict(summary) if summary else None

    def set_depth_summary(self, summary: Optional[dict]) -> None:
        self._cache_depth_summary = dict(summary) if summary else None

    def apply_ai_settings(self, settings: AIDefaults) -> None:
        """Wire a new :class:`AIDefaults` into the panel + client."""
        self._ai_settings = settings
        self._client = LocalLLMClient(self._llm_config_from_settings(settings))
        self._refresh_health()
        self.settings_changed.emit(settings)

    def stage(self) -> StageInterface:
        """Expose the panel's stage so main_window can wire status-bar
        position display, etc."""
        return self._stage

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header — model name + connected indicator + cog
        header = QHBoxLayout()
        self._model_label = QLabel(self._ai_settings.model_name)
        self._model_label.setObjectName("ai_model_label")
        self._health_label = QLabel("● checking…")
        self._health_label.setObjectName("ai_health_label")
        cog = QToolButton(root)
        cog.setText("⚙")
        cog.setToolTip("AI settings")
        cog.setAutoRaise(True)
        cog.clicked.connect(self._on_open_settings)
        header.addWidget(self._model_label, 0)
        header.addStretch(1)
        header.addWidget(self._health_label, 0)
        header.addWidget(cog, 0)
        layout.addLayout(header)

        # Chat history
        self._chat = QTextBrowser(root)
        self._chat.setObjectName("ai_chat_browser")
        self._chat.setOpenExternalLinks(False)
        self._chat.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._chat.setMinimumWidth(320)
        layout.addWidget(self._chat, 1)

        # Input box
        self._input = QPlainTextEdit(root)
        self._input.setObjectName("ai_input_box")
        self._input.setPlaceholderText(
            "Ask the assistant. Examples: \"What's loaded?\"  ·  "
            "\"Find focus candidates between -25 and 25 mm.\"\n"
            "Cmd+Return to send."
        )
        self._input.setFixedHeight(80)
        self._input.installEventFilter(self)
        layout.addWidget(self._input, 0)

        # Send / Stop row
        actions = QHBoxLayout()
        self._send_btn = QPushButton("Send", root)
        self._send_btn.setObjectName("ai_send_button")
        self._send_btn.setShortcut(QKeySequence("Ctrl+Return"))
        self._send_btn.clicked.connect(self._on_send_clicked)
        self._stop_btn = QPushButton("Stop", root)
        self._stop_btn.setObjectName("ai_stop_button")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        clear_btn = QPushButton("Clear", root)
        clear_btn.setToolTip("Clear chat history (does not reset model)")
        clear_btn.clicked.connect(self._on_clear_clicked)
        actions.addWidget(self._send_btn)
        actions.addWidget(self._stop_btn)
        actions.addStretch(1)
        actions.addWidget(clear_btn)
        layout.addLayout(actions)

        self.setWidget(root)
        self._append_html(
            "<p style='color:#888;font-size:11px;'>"
            "Tip: the assistant can drive reconstruction, autofocus, QPI, "
            "and depth maps via tool calls. Type a goal in plain language."
            "</p>"
        )

    # ------------------------------------------------------------------
    # Send / Stop / Clear
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        # Cmd+Return / Ctrl+Return inside the input → send.
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            assert isinstance(event, QKeyEvent)
            mod = event.modifiers()
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (
                mod & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
            ):
                self._on_send_clicked()
                return True
        return super().eventFilter(obj, event)

    def _on_send_clicked(self) -> None:
        if self._worker is not None:
            return  # send disabled visually + here
        prompt = self._input.toPlainText().strip()
        if not prompt:
            return
        self._input.clear()
        self._append_user(prompt)

        snapshot = self._build_snapshot()
        ctx = self._build_tool_context()
        worker = AIWorker(
            client=self._client,
            registry=self._registry,
            ctx=ctx,
            prompt=prompt,
            history=list(self._history),
            snapshot=snapshot,
            max_iterations=int(self._ai_settings.max_iterations),
            parent=self,
        )
        worker.assistant_text.connect(self._on_assistant_text)
        worker.tool_call_start.connect(self._on_tool_call_start)
        worker.tool_call_done.connect(self._on_tool_call_done)
        worker.completed.connect(self._on_worker_done)
        worker.iteration_cap.connect(self._on_iteration_cap)
        worker.cancelled_by_user.connect(self._on_cancelled)
        worker.error_event.connect(self._on_error_event)
        worker.finished.connect(self._on_worker_finished)

        # Track the user's message in shared history so the next turn sees it.
        self._history.append(ChatMessage(role="user", content=prompt))

        self._worker = worker
        self._send_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        worker.start()

    def _on_stop_clicked(self) -> None:
        if self._worker is None:
            return
        self._worker.requestInterruption()
        self._append_status("Cancelling…")

    def _on_clear_clicked(self) -> None:
        self._history.clear()
        self._chat.clear()
        self._append_html(
            "<p style='color:#888;font-size:11px;'>Chat cleared.</p>"
        )

    # ------------------------------------------------------------------
    # Worker signal handlers (run on the GUI thread)
    # ------------------------------------------------------------------

    def _on_assistant_text(self, text: str) -> None:
        self._history.append(ChatMessage(role="assistant", content=text))
        self._append_assistant(text)

    def _on_tool_call_start(self, call: ToolCall) -> None:
        if call is None:
            return
        try:
            args = json.loads(call.arguments_json) if call.arguments_json else {}
        except Exception:  # noqa: BLE001
            args = {"_raw": call.arguments_json}
        self._append_tool_start(call.name, args)

    def _on_tool_call_done(self, call: ToolCall, ok: bool, result: dict) -> None:
        if call is None:
            return
        self._append_tool_done(call.name, bool(ok), dict(result))

    def _on_iteration_cap(self, max_iter: int) -> None:
        self._append_status(f"Hit iteration cap ({max_iter}). Refine and try again.")

    def _on_cancelled(self) -> None:
        self._append_status("Cancelled.")

    def _on_error_event(self, err: ErrorEvent) -> None:
        self._append_error(err)
        try:
            get_error_center().emit(err)
        except Exception:  # noqa: BLE001
            pass

    def _on_worker_done(self) -> None:
        # natural finish — no extra text needed
        pass

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def _on_open_settings(self) -> None:
        dlg = AISettingsDialog(self._ai_settings, parent=self)
        if dlg.exec() == AISettingsDialog.DialogCode.Accepted:
            new_settings = dlg.result_settings()
            if new_settings is not None:
                self.apply_ai_settings(new_settings)
                self._model_label.setText(new_settings.model_name)

    # ------------------------------------------------------------------
    # Snapshot + ToolContext factory
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> StateSnapshot:
        """Capture the GUI thread's view of the world for this turn."""
        recon_params = self._gather_recon_params()
        af_params = self._gather_autofocus_params()
        qpi_params = self._gather_qpi_params()

        try:
            audit_tail = tuple(get_audit_log().entries_today(20))
        except Exception:  # noqa: BLE001
            audit_tail = ()

        try:
            stage_pos = self._stage.get_position()
        except Exception:  # noqa: BLE001
            stage_pos = None

        return StateSnapshot(
            loaded_path=self._cache_loaded_path,
            loaded_shape=self._cache_loaded_shape,
            loaded_dtype=self._cache_loaded_dtype,
            recon_params=recon_params,
            autofocus_params=af_params,
            qpi_params=qpi_params,
            last_recon_summary=self._cache_recon_summary,
            last_af_summary=self._cache_af_summary,
            last_qpi_summary=self._cache_qpi_summary,
            last_depth_summary=self._cache_depth_summary,
            stage_position_mm=stage_pos,
            audit_tail=audit_tail,
        )

    def _build_tool_context(self) -> ToolContext:
        # v2.1.z: lazily-built device hooks. Cached on the panel
        # so a second AI Send re-uses the same shutter / led
        # instances (state survives between turns; opening the
        # shutter on turn 1 means turn 2 sees ``is_open=True``).
        if not hasattr(self, "_v21z_device_hooks"):
            self._v21z_device_hooks = self._make_device_hooks()
        shutter, led, orchestrator = self._v21z_device_hooks
        # Cancel hook — wired to the in-flight worker. Tools that
        # poll ``ctx.is_cancelled`` (record_timelapse, future
        # acquire_grid implementations) get clean abort semantics.
        worker = self._worker

        def _is_cancelled() -> bool:
            try:
                return bool(worker is not None
                            and worker.isInterruptionRequested())
            except Exception:
                return False

        return ToolContext(
            state=self._build_snapshot,
            last_recon_summary=lambda: self._cache_recon_summary,
            last_af_summary=lambda: self._cache_af_summary,
            last_qpi_summary=lambda: self._cache_qpi_summary,
            last_depth_summary=lambda: self._cache_depth_summary,
            audit_tail=self._audit_tail,
            load_hologram=self._tool_load_hologram,
            set_recon_param=self._tool_set_recon_param,
            invoke_recon=self._tool_invoke_recon,
            invoke_autofocus=self._tool_invoke_autofocus,
            invoke_qpi=self._tool_invoke_qpi,
            invoke_depth_map=self._tool_invoke_depth_map,
            invoke_find_focus_candidates=self._tool_invoke_find_focus_candidates,
            stage=self._stage,
            capture_frame=self._capture_frame,
            sample_map=self._sample_map,
            measure_sharpness=self._measure_sharpness,
            persist_sample_map=self._persist_sample_map,
            capture_and_process=self._capture_and_process,
            audit=get_audit_log(),
            error_center=get_error_center(),
            confirm=self._confirm,
            settings=self._ai_settings,
            is_cancelled=_is_cancelled,
            shutter=shutter,
            led=led,
            orchestrator=orchestrator,
        )

    def _make_device_hooks(self):
        """Default-construct mock shutter + LED + orchestrator
        callable so the v2.1.z device tools (``shutter_*``,
        ``led_*``, ``acquire_grid``) work end-to-end out of the
        box. Lab can swap in a real backend later by reaching
        ``panel._v21z_device_hooks = (vendor_shutter, vendor_led,
        my_orchestrator)``.

        Returns
        -------
        tuple
            ``(shutter, led, orchestrator)`` — any element can
            be ``None`` if construction failed; the corresponding
            tools then return a friendly "device not configured"
            error rather than crashing.
        """
        shutter = led = None
        try:
            from core.devices import make_device
            shutter = make_device("mock_shutter")
            led = make_device("mock_led")
        except Exception:  # noqa: BLE001
            _LOG.debug("v2.1.z device defaults not available",
                       exc_info=True)

        def _orchestrator_callable(args: dict) -> dict:
            """Adapter: AI tool args → AcquisitionPlan + run_plan.

            Builds a rows × cols grid centred on the stage's
            current position. Camera defaults to the panel's
            existing capture-frame source when no live camera is
            wired."""
            try:
                from core.devices.orchestrator import (
                    AcquisitionPlan, StagePosition, run_plan,
                )
            except Exception as exc:  # noqa: BLE001
                return {"error": f"orchestrator import failed: {exc}"}

            rows = int(args.get("rows", 1))
            cols = int(args.get("cols", 1))
            spacing_x = float(args.get("spacing_x_um", 100.0))
            spacing_y = float(args.get("spacing_y_um", 100.0))
            # Centre the grid on (0, 0). The legacy stage uses mm
            # coordinates; the orchestrator expects µm. We map by
            # treating the AI-side positions as the stage's native
            # unit (µm) — caller pre-converts mm → µm if needed.
            positions = []
            for r in range(rows):
                for c in range(cols):
                    positions.append(StagePosition(
                        y_um=(r - (rows - 1) / 2.0) * spacing_y,
                        x_um=(c - (cols - 1) / 2.0) * spacing_x,
                        z_um=0.0,
                        label=f"r{r}c{c}",
                    ))
            plan = AcquisitionPlan(
                positions=positions,
                led_intensity_percent=args.get("led_intensity_percent"),
                shutter_per_frame=bool(args.get(
                    "shutter_per_frame", True,
                )),
                settle_time_s=float(args.get("settle_time_s", 0.05)),
            )

            class _PanelCamera:
                """Adapter from the panel's ``_capture_frame``
                helper to the CameraSource Protocol the
                orchestrator expects."""

                def __init__(self, panel):
                    self._panel = panel

                @property
                def size(self):
                    arr = self._panel._capture_frame()
                    if arr is None:
                        return (0, 0)
                    return tuple(int(x) for x in arr.shape[:2])

                @property
                def fps(self):
                    return 30.0

                def start(self):
                    pass

                def stop(self):
                    pass

                def grab(self):
                    arr = self._panel._capture_frame()
                    if arr is None:
                        import numpy as _np
                        return _np.zeros((1, 1), dtype=_np.float32)
                    return arr

            cam = _PanelCamera(self)
            try:
                # Use the cached hooks so shutter/LED state across
                # turns matches what the AI just set.
                sh, ld, _ = getattr(
                    self, "_v21z_device_hooks",
                    (shutter, led, None),
                )
                # NB: orchestrator stage is the panel's legacy
                # ``ctx.stage`` (MockStage) — its API doesn't
                # match the StageDevice Protocol exactly, but
                # ``run_plan`` only calls ``connect/move_to/...``
                # via duck typing. Skip stage when its API is
                # incompatible.
                stage_for_plan = None
                if hasattr(self._stage, "move_to"):
                    stage_for_plan = self._stage
                result = run_plan(
                    plan, camera=cam,
                    stage=stage_for_plan,
                    shutter=sh, led=ld,
                    sleep=lambda s: None,
                )
            except Exception as exc:  # noqa: BLE001
                return {"error": f"run_plan failed: {exc}"}

            # Compact summary the LLM can read.
            return {
                "ok": True,
                "frames_captured": sum(
                    1 for r in result.frames if r.frame is not None
                ),
                "frames_total": len(result.frames),
                "errors": [
                    {"index": r.index, "label": r.position.label,
                     "error": r.error}
                    for r in result.frames if r.error
                ],
                "elapsed_s": round(result.elapsed_s, 3),
            }

        return shutter, led, _orchestrator_callable

    def _audit_tail(self, limit: int) -> list[dict]:
        try:
            return list(get_audit_log().entries_today(int(limit)))
        except Exception:  # noqa: BLE001
            return []

    def _confirm(self, tool_name: str, args: dict) -> bool:
        """Show a modal confirmation for an irreversible AI tool call.

        v2.1.z fix: the previous implementation used
        ``QTimer.singleShot(0, _run)`` from the AI worker thread —
        ``QTimer`` posts to the *calling* thread's event loop, but
        the worker thread has no Qt event loop, so the timer never
        fired and the 120 s ``done.wait`` would silently expire on
        every irreversible call. Confirmation gates were dead code.

        This rewrite uses ``QMetaObject.invokeMethod(self,
        '_run_confirm_dialog', Qt.QueuedConnection, ...)`` — slots
        always run on the receiver's thread (the GUI thread for
        ``self``), regardless of who invoked them.
        """
        from PySide6.QtCore import QMetaObject, Qt as QtNs, QThread

        # Stash request payload + result slot on the instance so the
        # slot can read them. A more elegant approach would be
        # ``Q_ARG``-passing the dict, but PySide6's invokeMethod
        # arg-marshalling for arbitrary Python objects is brittle.
        self._pending_confirm = {
            "tool_name": str(tool_name),
            "args": dict(args),
            "result": False,
            "done": threading.Event(),
        }

        if QThread.currentThread() is self.thread():
            # Already on GUI thread — run inline.
            self._run_confirm_dialog()
        else:
            QMetaObject.invokeMethod(
                self, "_run_confirm_dialog",
                QtNs.ConnectionType.QueuedConnection,
            )
            self._pending_confirm["done"].wait(timeout=120.0)
        return bool(self._pending_confirm["result"])

    @Slot()
    def _run_confirm_dialog(self) -> None:
        """Slot — runs on the GUI thread. Spawns the modal +
        records the answer back into ``self._pending_confirm``."""
        try:
            payload = getattr(self, "_pending_confirm", None)
            if not payload:
                return
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("AI confirmation required")
            box.setText(
                "The assistant wants to run an irreversible tool:"
                f"\n\n{payload['tool_name']}",
            )
            box.setInformativeText(
                f"Arguments:\n{json.dumps(payload['args'], indent=2)[:600]}"
                "\n\nAllow?",
            )
            box.setStandardButtons(
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
            )
            payload["result"] = (
                box.exec() == QMessageBox.StandardButton.Yes
            )
        finally:
            try:
                self._pending_confirm["done"].set()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # GUI-thread tool callbacks
    #
    # Each marshals the work to the main window via QTimer.singleShot
    # so the AI thread doesn't touch QWidgets directly. The AI thread
    # blocks on a ``threading.Event`` until the GUI helper sets a result.
    # ------------------------------------------------------------------

    def _run_on_gui(self, fn: Callable[[], dict],
                    timeout_s: float = _PIPELINE_HARD_TIMEOUT_S) -> dict:
        from PySide6.QtCore import QThread, QTimer
        if QThread.currentThread() is self.thread():
            return fn()

        result_holder: dict = {}
        done = threading.Event()

        def _runner() -> None:
            try:
                result_holder.update(fn())
            except Exception as exc:  # noqa: BLE001
                result_holder.update({"error": type(exc).__name__, "message": str(exc)})
            finally:
                done.set()

        QTimer.singleShot(0, _runner)
        if not done.wait(timeout=timeout_s):
            return {"error": "timeout", "timeout_s": timeout_s}
        return dict(result_holder)

    def _tool_load_hologram(self, path: str) -> dict:
        return self._run_on_gui(lambda: self._gui_load_hologram(path))

    def _gui_load_hologram(self, path: str) -> dict:
        mw = self._main_window
        # ``MainWindow._load_file_path`` is the canonical loader (toolbar
        # + drag-drop both call it). It returns False on failure so we
        # check both the cache update *and* the return value.
        loader = (
            getattr(mw, "_load_file_path", None)
            or getattr(mw, "_load_hologram_path", None)
            or getattr(mw, "_load_file", None)
            or getattr(mw, "load_hologram", None)
        )
        if loader is None:
            return {"error": "main window has no hologram loader"}
        try:
            ok = loader(path)
        except Exception as exc:  # noqa: BLE001
            return {"error": "load failed", "message": str(exc)}
        if ok is False:
            return {"error": "load failed", "message": "loader rejected the file"}
        # Refresh the cache from main_window state — _load_file_path
        # writes _loaded_path / _loaded_array, so we reflect those.
        loaded_path = getattr(mw, "_loaded_path", None)
        loaded_array = getattr(mw, "_loaded_array", None)
        self.set_loaded(str(loaded_path) if loaded_path else None, loaded_array)
        shape = self._cache_loaded_shape
        return {
            "path": str(loaded_path) if loaded_path else path,
            "shape": list(shape) if shape else None,
            "dtype": self._cache_loaded_dtype,
        }

    def _tool_set_recon_param(self, params: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_set_recon_param(params))

    def _gui_set_recon_param(self, params: dict) -> dict:
        mw = self._main_window
        rtab = getattr(getattr(mw, "sidebar_tabs", None), "recon_tab", None)
        if rtab is None or not hasattr(rtab, "set_state"):
            return {"error": "recon sidebar tab unavailable"}
        try:
            rtab.set_state(dict(params))
        except Exception as exc:  # noqa: BLE001
            return {"error": "set_state failed", "message": str(exc)}
        return {"updated": dict(params), "current": self._gather_recon_params()}

    def _tool_invoke_recon(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_recon(args),
                                timeout_s=_PIPELINE_HARD_TIMEOUT_S)

    def _gui_invoke_recon(self, args: dict) -> dict:
        mw = self._main_window
        trigger = getattr(mw, "_trigger_reconstruction", None)
        if trigger is None:
            return {"error": "main window has no _trigger_reconstruction"}
        try:
            trigger()
        except Exception as exc:  # noqa: BLE001
            return {"error": "reconstruction trigger failed", "message": str(exc)}
        return {"submitted": True,
                "current_recon_params": self._gather_recon_params(),
                "summary": self._cache_recon_summary}

    def _tool_invoke_autofocus(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_autofocus(args),
                                timeout_s=_PIPELINE_HARD_TIMEOUT_S)

    def _gui_invoke_autofocus(self, args: dict) -> dict:
        mw = self._main_window
        # Push z range + metric into the focus tab if asked, then trigger.
        ftab = getattr(getattr(mw, "sidebar_tabs", None), "focus_tab", None)
        if ftab is not None and hasattr(ftab, "set_state"):
            patch: dict = {}
            if "z_min_mm" in args:
                patch["z_min_mm"] = float(args["z_min_mm"])
            if "z_max_mm" in args:
                patch["z_max_mm"] = float(args["z_max_mm"])
            if "metric" in args:
                patch["metric"] = str(args["metric"])
            if "n_steps" in args:
                patch["n_steps"] = int(args["n_steps"])
            try:
                if patch:
                    ftab.set_state(patch)
            except Exception:  # noqa: BLE001 — best-effort
                pass
        trigger = (
            getattr(mw, "_trigger_autofocus", None)
            or getattr(mw, "_on_autofocus_triggered", None)
        )
        if trigger is None:
            return {"error": "main window has no autofocus trigger"}
        try:
            trigger()
        except Exception as exc:  # noqa: BLE001
            return {"error": "autofocus trigger failed", "message": str(exc)}
        return {"submitted": True, "summary": self._cache_af_summary}

    def _tool_invoke_qpi(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_qpi(args),
                                timeout_s=_PIPELINE_HARD_TIMEOUT_S)

    def _gui_invoke_qpi(self, args: dict) -> dict:
        mw = self._main_window
        qtab = getattr(getattr(mw, "sidebar_tabs", None), "qpi_tab", None)
        if qtab is not None and hasattr(qtab, "set_state"):
            patch: dict = {}
            if "n_sample" in args:
                patch["n_sample"] = float(args["n_sample"])
            if "n_medium" in args:
                patch["n_medium"] = float(args["n_medium"])
            if "phase_offset" in args:
                patch["phase_offset"] = float(args["phase_offset"])
            try:
                if patch:
                    qtab.set_state(patch)
            except Exception:  # noqa: BLE001
                pass
        trigger = (
            getattr(mw, "_trigger_qpi", None)
            or getattr(mw, "_on_qpi_triggered", None)
        )
        if trigger is None:
            return {"error": "main window has no QPI trigger"}
        try:
            trigger()
        except Exception as exc:  # noqa: BLE001
            return {"error": "QPI trigger failed", "message": str(exc)}
        return {"submitted": True, "summary": self._cache_qpi_summary}

    def _tool_invoke_depth_map(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_depth_map(args),
                                timeout_s=_PIPELINE_HARD_TIMEOUT_S)

    def _gui_invoke_depth_map(self, args: dict) -> dict:
        mw = self._main_window
        trigger = getattr(mw, "_on_compute_depth_map_triggered", None)
        if trigger is None:
            return {"error": "main window has no depth-map command"}
        try:
            trigger()
        except Exception as exc:  # noqa: BLE001
            return {"error": "depth map failed", "message": str(exc)}
        return {"submitted": True, "summary": self._cache_depth_summary}

    def _tool_invoke_find_focus_candidates(self, args: dict) -> dict:
        return self._run_on_gui(
            lambda: self._gui_invoke_find_focus_candidates(args),
            timeout_s=_PIPELINE_HARD_TIMEOUT_S,
        )

    def _gui_invoke_find_focus_candidates(self, args: dict) -> dict:
        mw = self._main_window
        trigger = getattr(mw, "_on_find_focus_candidates_triggered", None)
        if trigger is None:
            return {"error": "main window has no find-focus-candidates command"}
        try:
            trigger()
        except Exception as exc:  # noqa: BLE001
            return {"error": "find_focus_candidates failed", "message": str(exc)}
        return {"submitted": True}

    # ------------------------------------------------------------------
    # Sprint 2: capture / sharpness / map / capture-and-process
    # ------------------------------------------------------------------

    def _capture_frame(self):
        """Return the most recent raw camera image (numpy 2-D).

        v1 (no real camera): falls back to the loaded hologram. Real
        hardware drivers replace this with a live frame grab. Stage
        focus search uses the result purely for sharpness comparisons,
        which the ``measure_sharpness`` callback computes.
        """
        mw = self._main_window
        arr = getattr(mw, "_loaded_array", None)
        if arr is None:
            return None
        return arr

    @staticmethod
    def _measure_sharpness(frame) -> float:
        """Laplacian variance on the frame magnitude.

        Pure numpy so it runs on the AI thread without marshaling. We
        cast to float32 to avoid surprises with uint16 cameras and
        clamp NaN/Inf so a degenerate frame returns -inf rather than
        poisoning the search.
        """
        if frame is None:
            return -float("inf")
        a = np.asarray(frame)
        if a.ndim == 0 or a.size == 0:
            return -float("inf")
        amp = np.abs(a).astype(np.float32, copy=False)
        # Discrete Laplacian on the interior pixels.
        if amp.shape[0] < 3 or amp.shape[1] < 3:
            return -float("inf")
        lap = (
            amp[1:-1, :-2] + amp[1:-1, 2:]
            + amp[:-2, 1:-1] + amp[2:, 1:-1]
            - 4.0 * amp[1:-1, 1:-1]
        )
        v = float(np.var(lap))
        if not np.isfinite(v):
            return -float("inf")
        return v

    def _persist_sample_map(self) -> Optional[str]:
        """Write the current sample map under the per-user state dir.

        Path is ``<root>/users/<sanitised>/sample_maps/<sample_id>.json``
        via :func:`core.user_profile.user_state_dir` — the same place
        Ui2State, presets, and the audit log sit. Falling back to
        ``~/.dhm-reconstruction`` flat directly here would break the
        multi-user model from v2.0.7.

        Returns the resulting path, or None if persistence failed (we
        log but never raise; mapping should never break because the
        disk is full or read-only).
        """
        try:
            from core.user_profile import user_state_dir
            sample_id = self._sample_map.sample_id or "sample"
            base = user_state_dir() / "sample_maps"
            target = base / f"{sample_id}.json"
            self._sample_map.save(target)
            return str(target)
        except Exception:  # noqa: BLE001
            _LOG.warning("ai: sample map persist failed", exc_info=True)
            return None

    def _capture_and_process(self, args: dict) -> dict:
        """Capture one frame and (optionally) run the pipeline.

        This is the panel-side implementation that the AI thread calls
        through ToolContext. It marshals to the GUI thread so it can
        touch the existing pipeline workers safely. The returned dict
        carries scalars the LLM compares across frames in time-lapse
        and across grid points in mapping.
        """
        return self._run_on_gui(
            lambda: self._gui_capture_and_process(args),
            timeout_s=_PIPELINE_HARD_TIMEOUT_S,
        )

    @staticmethod
    def _wait_for_signal(signal, *, timeout_ms: int = 60_000) -> bool:
        """Spin a nested ``QEventLoop`` until ``signal`` fires (or timeout).

        Two reasons we need this:

        1. ``_trigger_reconstruction`` etc. submit jobs to a background
           ``QThread`` — they return immediately. Reading the panel's
           summary cache right after returns the *previous* frame's
           result. Without an explicit wait, ``record_timelapse`` and
           ``map_sample_grid`` would silently report stale data on every
           iteration.
        2. We can't ``threading.Event.wait`` here — this method runs on
           the GUI thread, and the worker's ``recon_completed`` slot is
           queued onto the *same* GUI thread. A hard wait would deadlock.

        ``QEventLoop`` keeps the event queue draining so the queued
        slots (including ``MainWindow._on_recon_completed`` which feeds
        ``set_recon_summary``) actually run before we return. Connection
        order matters: the main-window handler is connected during
        ``__init__``, so it runs *before* our temporary handler — by
        the time the loop exits, the cache is fresh.
        """
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication

        loop = QEventLoop()
        # Mutable flag captured by the signal handler so the caller can
        # tell signal-fired apart from timeout — we used to read
        # ``timer.remainingTime()`` after stop(), which returns -1
        # regardless and broke the discrimination.
        state = {"fired": False}

        def _on_fired(*_a, **_kw):
            state["fired"] = True
            loop.quit()

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)

        try:
            signal.connect(_on_fired)
        except Exception:  # noqa: BLE001
            return False

        timer.start(int(timeout_ms))
        try:
            loop.exec()
        finally:
            try:
                signal.disconnect(_on_fired)
            except (RuntimeError, TypeError):
                pass
            timer.stop()
        # Drain anything still queued (defensive — Qt usually clears
        # before exec returns, but a same-tick emission can leave one
        # slot pending).
        QApplication.processEvents()
        return bool(state["fired"])

    def _gui_capture_and_process(self, args: dict) -> dict:
        mw = self._main_window
        run_recon = bool(args.get("run_recon", False))
        run_qpi = bool(args.get("run_qpi", False))
        segment = bool(args.get("segment", False))

        frame = self._capture_frame()
        if frame is None:
            return {"error": "no frame to capture (load a hologram first)"}

        out: dict = {"shape": list(frame.shape) if hasattr(frame, "shape") else None}

        if run_recon:
            trigger = getattr(mw, "_trigger_reconstruction", None)
            if callable(trigger):
                try:
                    trigger()
                except Exception as exc:  # noqa: BLE001
                    out["recon_error"] = str(exc)
                else:
                    # Wait for the worker to finish before reading the
                    # cache — without this we'd return stale state from
                    # the previous frame.
                    worker = getattr(mw, "_recon_worker", None)
                    sig = getattr(worker, "recon_completed", None)
                    if sig is not None:
                        if not self._wait_for_signal(sig, timeout_ms=60_000):
                            out["recon_timeout"] = True
            recon = self._cache_recon_summary or {}
            for k in ("phase_std", "z_mm"):
                if k in recon:
                    out[k] = recon[k]

        if run_qpi:
            trigger = (
                getattr(mw, "_trigger_qpi", None)
                or getattr(mw, "_on_qpi_triggered", None)
            )
            if callable(trigger):
                try:
                    trigger()
                except Exception as exc:  # noqa: BLE001
                    out["qpi_error"] = str(exc)
                else:
                    # QPI worker is created on demand by the trigger so
                    # we look it up *after* the call. Same race fix as
                    # recon.
                    worker = getattr(mw, "_qpi_worker", None)
                    sig = getattr(worker, "qpi_completed", None)
                    if sig is not None:
                        if not self._wait_for_signal(sig, timeout_ms=60_000):
                            out["qpi_timeout"] = True
            qpi = self._cache_qpi_summary or {}
            for k in ("total_dry_mass_pg", "opd_range_nm"):
                if k in qpi:
                    out[k] = qpi[k]

        if segment:
            cells = self._segment_cells_from_state()
            out["cells"] = cells
            out["cell_count"] = len(cells)

        return out

    def _segment_cells_from_state(self) -> list[dict]:
        """Best-effort cell list from the latest reconstruction's phase.

        Wraps :func:`core.qpi.segment_cell_phase` against the current
        ``_phase_unwrapped`` array. Returns an empty list if no phase
        is available; never raises (segmentation can fail on non-cell
        scenes — that's fine for mapping, the grid point just records
        zero cells).
        """
        mw = self._main_window
        phase = getattr(mw, "_phase_unwrapped", None)
        if phase is None:
            return []
        try:
            from core.qpi import (
                compute_cell_morphology,
                phase_to_opd,
                segment_cell_phase,
            )
        except Exception:  # noqa: BLE001
            return []

        try:
            wavelength_m = float(mw.sidebar_tabs.recon_tab.wavelength_nm.value()) * 1e-9
            pixel_um = float(mw.sidebar_tabs.recon_tab.pixel_um.value())
            mask = segment_cell_phase(phase_to_opd(phase, wavelength_m))
        except Exception:  # noqa: BLE001
            return []

        # ``mask`` is a labelled int array; iterate unique labels (>0)
        # and emit a per-cell row. We avoid pulling all of skimage just
        # for centroids — np.argwhere + means is enough for the AI's
        # purpose (it only needs approximate XY).
        cells: list[dict] = []
        if mask is None or getattr(mask, "size", 0) == 0:
            return cells
        try:
            labels = np.unique(mask)
            labels = labels[labels > 0]
            pixel_um2 = pixel_um * pixel_um
            for lab in labels[:64]:  # cap per-grid-point — mapping isn't
                                     # the place to enumerate hundreds
                ys, xs = np.where(mask == lab)
                if ys.size == 0:
                    continue
                cells.append({
                    "centroid_y_px": int(ys.mean()),
                    "centroid_x_px": int(xs.mean()),
                    "area_um2": float(ys.size * pixel_um2),
                })
        except Exception:  # noqa: BLE001
            return []
        return cells

    # ------------------------------------------------------------------
    # State gatherers (read sidebar widgets on the GUI thread)
    # ------------------------------------------------------------------

    def _gather_recon_params(self) -> dict:
        mw = self._main_window
        rtab = getattr(getattr(mw, "sidebar_tabs", None), "recon_tab", None)
        if rtab is None:
            return {}
        getter = getattr(rtab, "get_state", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter())
        except Exception:  # noqa: BLE001
            return {}

    def _gather_autofocus_params(self) -> dict:
        mw = self._main_window
        ftab = getattr(getattr(mw, "sidebar_tabs", None), "focus_tab", None)
        if ftab is None:
            return {}
        getter = getattr(ftab, "get_state", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter())
        except Exception:  # noqa: BLE001
            return {}

    def _gather_qpi_params(self) -> dict:
        mw = self._main_window
        qtab = getattr(getattr(mw, "sidebar_tabs", None), "qpi_tab", None)
        if qtab is None:
            return {}
        getter = getattr(qtab, "get_state", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter())
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _refresh_health(self) -> None:
        # Quick best-effort probe. If it fails we say "Offline" but
        # the user can still hit Send — the worker will surface the
        # error if it really is dead.
        try:
            ok = self._client.health_check()
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            self._health_label.setText("● Connected")
            self._health_label.setStyleSheet("color:#3c3;")
        else:
            self._health_label.setText("● Offline")
            self._health_label.setStyleSheet("color:#c33;")

    # ------------------------------------------------------------------
    # Chat rendering
    # ------------------------------------------------------------------

    def _append_html(self, html_block: str) -> None:
        self._chat.append(html_block)
        self._chat.verticalScrollBar().setValue(self._chat.verticalScrollBar().maximum())

    def _append_user(self, text: str) -> None:
        self._append_html(
            f"<div style='margin-top:6px;'>"
            f"<b style='color:#06b;'>You:</b> {html.escape(text)}"
            f"</div>"
        )

    def _append_assistant(self, text: str) -> None:
        self._append_html(
            f"<div style='margin-top:6px;'>"
            f"<b style='color:#693;'>Assistant:</b> "
            f"<span>{html.escape(text)}</span>"
            f"</div>"
        )

    def _append_tool_start(self, name: str, args: dict) -> None:
        body = html.escape(json.dumps(args, default=str))
        self._append_html(
            f"<div style='margin:4px 0 0 18px;font-family:monospace;color:#666;'>"
            f"→ <b>{html.escape(name)}</b>({body})"
            f"</div>"
        )

    def _append_tool_done(self, name: str, ok: bool, result: dict) -> None:
        colour = "#393" if ok else "#c33"
        marker = "✓" if ok else "✗"
        body = html.escape(json.dumps(_truncate_for_display(result), default=str))
        self._append_html(
            f"<div style='margin:0 0 4px 18px;font-family:monospace;color:{colour};'>"
            f"{marker} {html.escape(name)} → {body}"
            f"</div>"
        )

    def _append_status(self, text: str) -> None:
        self._append_html(
            f"<div style='margin:6px 0;color:#888;font-style:italic;'>"
            f"{html.escape(text)}"
            f"</div>"
        )

    def _append_error(self, err: ErrorEvent) -> None:
        self._append_html(
            f"<div style='margin:6px 0;color:#c33;'>"
            f"<b>Error:</b> {html.escape(err.title)} — {html.escape(err.cause)}"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _initial_settings(self) -> AIDefaults:
        mw = self._main_window
        settings = getattr(mw, "_settings", None)
        if settings is not None and hasattr(settings, "ai"):
            return settings.ai
        return AIDefaults()

    @staticmethod
    def _llm_config_from_settings(s: AIDefaults) -> LLMConfig:
        return LLMConfig(
            endpoint=s.endpoint_url,
            model=s.model_name,
            temperature=float(s.temperature),
            max_tokens=int(s.max_tokens),
            request_timeout_s=float(s.request_timeout_s),
        )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _truncate_for_display(obj: Any, max_chars: int = 400) -> Any:
    text = json.dumps(obj, default=str)
    if len(text) <= max_chars:
        return obj
    return {"_truncated": True, "preview": text[:max_chars] + "…"}


__all__ = ["AIPanel"]
