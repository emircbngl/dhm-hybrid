"""AIPanel — local-LLM copilot chat panel (ui3, Qt-native).

Ports the v1 ``gui.panels.ai_panel.AIPanel`` + ``gui.workers.ai_worker.AIWorker``
pattern (a working ``QThread`` running ``core.ai.Agent``) onto the frozen ui3
``PanelContext`` contract, instead of the broken ui2 daemon-thread port.

Architecture
------------
* :class:`AIWorker` (``QThread``) owns one :class:`core.ai.agent.Agent` turn.
  It never touches Qt widgets — it only relays :class:`core.ai.agent.AgentEvent`
  as Qt signals, exactly like the v1 worker. The GUI thread (this panel)
  renders each event into the transcript.
* The panel builds a fresh :class:`core.ai.tools.ToolContext` per Send. Every
  ``invoke_*`` callable in that context is a *blocking bridge*: it posts the
  actual work onto the GUI thread (where ``ctx.bridge`` — a real
  ``ui3.bridge.WorkerBridge`` — lives) via
  ``QMetaObject.invokeMethod(self, ..., Qt.QueuedConnection)``, then blocks
  the AI thread on a ``threading.Event`` until the bridge's completion signal
  fires (or a timeout elapses). Queued ``invokeMethod`` calls always run on
  the *receiver's* thread regardless of the caller's — unlike
  ``QTimer.singleShot(0, callable)``, which posts to the *calling* thread's
  event loop and never fires from a ``QThread`` that doesn't pump its own
  loop (v1 hit and fixed this exact bug in ``AIPanel._confirm``, but never
  generalised the fix to its ``_run_on_gui``; ui2's port skipped the wait
  entirely and returned stale summaries).
* State/summary callables (``last_recon_summary`` etc.) are built fresh each
  turn from ``ctx.last_recon()`` / ``ctx.last_depth()`` — no separate cache is
  kept here since ``PanelContext`` already exposes the live results.
* ``render_view`` tool results ``{"ok": True, "path": ...}`` are rendered
  inline as a ``QPixmap`` in the transcript — something the old DPG frontend
  could never do.

No Qt import happens until the panel actually needs one bridged call, so
constructing the panel with no local LLM reachable never crashes: the
``LocalLLMClient`` is lazy (see ``core.ai.client``), and ``health_check()`` is
wrapped in try/except to report an "unavailable" pill instead of raising.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.ai.agent import Agent, AgentEvent
from core.ai.client import LLMConfig, LocalLLMClient
from core.ai.context import StateSnapshot, build_system_prompt
from core.ai.protocol import ChatMessage, ToolCall
from core.ai.tool_impls import build_tool_registry
from core.ai.tools import ToolContext, ToolRegistry
from ui3.context import PanelContext
from ui3.design import Space

_LOG = logging.getLogger("ui3.panels.ai")

# Heuristic ceiling on a single bridged pipeline call (autofocus over a
# wide z range on a large hologram can run for several seconds). Acts as
# a safety net only — normal completion is via the bridge's *_done signal.
_PIPELINE_TIMEOUT_S = 600.0
_DEFAULT_ENDPOINT = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b-instruct"


def _disconnect_one(signal, connection) -> None:
    """Disconnect ONE captured connection — never the whole signal.

    The bridged tool calls below connect a short-lived capturing slot to a
    bridge signal (e.g. ``recon_done``) to collect a single op's result, then
    must detach it. A bare ``signal.disconnect()`` (no argument) severs *every*
    slot on that signal — including MainWindow's viewport paint and the rich
    panels — so after the copilot ran one op the whole app stopped reacting to
    reconstruct/autofocus/qpi/depth (2026-07-06 review). Passing the
    ``QMetaObject.Connection`` handle returned by ``connect`` detaches only
    ours. ``connect`` can return a falsy handle on some bindings; skip those.
    """
    if not connection:
        return
    try:
        signal.disconnect(connection)
    except (RuntimeError, TypeError):
        pass


# ===========================================================================
# Health probe — one-shot reachability check off the GUI thread
# ===========================================================================

class _HealthCheckThread(QThread):
    """One-shot LLM reachability probe off the GUI thread.

    ``LocalLLMClient.health_check()`` does a blocking ``requests.get`` with a
    2s timeout; running it here keeps panel construction instant and never
    lets synchronous socket I/O race the GUI event loop (2026-07-05 review)."""

    result = Signal(bool)

    def __init__(self, client, parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:  # noqa: D401 - QThread API
        ok = False
        try:
            ok = bool(self._client and self._client.health_check())
        except Exception:  # noqa: BLE001
            ok = False
        self.result.emit(ok)


# ===========================================================================
# AIWorker — QThread driving one Agent turn
# ===========================================================================

class AIWorker(QThread):
    """Runs one user prompt through :class:`core.ai.agent.Agent` off the GUI
    thread. Ported 1:1 from ``gui.workers.ai_worker.AIWorker`` — the only
    change is the module path of its dependencies (unchanged otherwise)."""

    assistant_text = Signal(str)
    tool_call_start = Signal(object)              # ToolCall
    tool_call_done = Signal(object, bool, dict)   # (ToolCall, ok, result)
    completed = Signal()
    iteration_cap = Signal(int)
    cancelled_by_user = Signal()
    error_event = Signal(str)

    def __init__(
        self,
        *,
        client: LocalLLMClient,
        registry: ToolRegistry,
        ctx: ToolContext,
        prompt: str,
        history: list[ChatMessage],
        snapshot: StateSnapshot,
        max_iterations: int = 8,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._registry = registry
        self._ctx = ctx
        self._prompt = str(prompt)
        self._history = list(history)
        self._snapshot = snapshot
        self._max = int(max_iterations)
        self.setObjectName("AIWorker")

    def run(self) -> None:  # noqa: D401 - QThread API
        try:
            agent = Agent(
                client=self._client,
                registry=self._registry,
                ctx=self._ctx,
                max_iterations=self._max,
            )
            system_prompt = build_system_prompt(self._snapshot, len(self._registry))
            for event in agent.run(
                user_prompt=self._prompt,
                system_prompt=system_prompt,
                history=self._history,
                cancel=self.isInterruptionRequested,
            ):
                self._dispatch(event)
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            _LOG.exception("AIWorker crashed: %s", exc)
            self.error_event.emit(str(exc))

    def _dispatch(self, event: AgentEvent) -> None:
        kind = event.kind
        if kind == "assistant_text":
            self.assistant_text.emit(event.payload.get("text", ""))
        elif kind == "tool_call_start":
            self.tool_call_start.emit(event.payload.get("call"))
        elif kind == "tool_call_done":
            call = event.payload.get("call")
            ok = bool(event.payload.get("ok", False))
            result = event.payload.get("result", {}) or {}
            self.tool_call_done.emit(call, ok, dict(result))
        elif kind == "completed":
            self.completed.emit()
        elif kind == "iteration_cap":
            self.iteration_cap.emit(self._max)
        elif kind == "cancelled":
            self.cancelled_by_user.emit()
        elif kind == "error":
            err = event.payload.get("event")
            message = getattr(err, "cause", None) or str(err)
            self.error_event.emit(str(message))


# ===========================================================================
# AIPanel
# ===========================================================================

class AIPanel(QWidget):
    """Chat panel wired to a real ``core.ai.Agent`` via the frozen
    ``PanelContext`` contract (``ctx.bridge`` drives recon/autofocus/qpi/
    depth; ``ctx.get_field`` feeds the observation tools)."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._registry: ToolRegistry = build_tool_registry(include_devices=False)
        self._history: list[ChatMessage] = []
        self._worker: Optional[AIWorker] = None
        self._client: Optional[LocalLLMClient] = None
        self._health_thread: Optional["_HealthCheckThread"] = None
        # Every health probe ever started that hasn't finished yet. A
        # superseded probe is NOT orphaned (a QThread destroyed while running
        # crashes); it stays here until its finished signal removes it, and
        # shutdown() waits on all of them (2026-07-06 review).
        self._health_threads: set["_HealthCheckThread"] = set()
        self._reference_mode = "off"

        self._build_ui()
        self._rebuild_client()
        # NO automatic health check on construction: it does synchronous
        # network I/O (requests.get, 2s timeout) which must never touch the
        # GUI thread inside __init__, and repeated construction racing
        # urllib3 socket teardown segfaults the offscreen test process
        # (2026-07-05 review). The pill stays neutral until the user applies
        # settings / clicks Reconnect / sends a message — all of which run
        # the probe on a one-shot background thread (_refresh_health).
        self._set_health("● not checked", "muted")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.md)
        root.setSpacing(Space.sm)

        heading = QLabel("AI copilot", self)
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        # -- endpoint / model settings row ---------------------------------
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Endpoint", self))
        self.endpoint_edit = QLineEdit(_DEFAULT_ENDPOINT, self)
        self.endpoint_edit.setPlaceholderText(_DEFAULT_ENDPOINT)
        self.endpoint_edit.editingFinished.connect(self._on_settings_changed)
        settings_row.addWidget(self.endpoint_edit, 1)

        settings_row.addWidget(QLabel("Model", self))
        self.model_combo = QComboBox(self)
        self.model_combo.setEditable(True)
        self.model_combo.addItems([
            _DEFAULT_MODEL, "llama3.2", "qwen2.5:14b-instruct", "mistral",
        ])
        self.model_combo.setCurrentText(_DEFAULT_MODEL)
        self.model_combo.currentTextChanged.connect(self._on_settings_changed)
        settings_row.addWidget(self.model_combo, 1)

        self.health_label = QLabel("checking…", self)
        self.health_label.setProperty("role", "muted")
        settings_row.addWidget(self.health_label)
        root.addLayout(settings_row)

        # -- transcript -----------------------------------------------------
        self.transcript = QPlainTextEdit(self)
        self.transcript.setObjectName("ai_transcript")
        self.transcript.setReadOnly(True)
        self.transcript.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.transcript.setMinimumHeight(220)
        root.addWidget(self.transcript, 1)

        # Container for inline rendered images (render_view results). A
        # simple vertical stack of QLabel(pixmap) rows appended alongside
        # the transcript — QPlainTextEdit can't embed pixmaps directly.
        self.images_frame = QFrame(self)
        self.images_frame.setProperty("role", "card")
        self._images_layout = QVBoxLayout(self.images_frame)
        self._images_layout.setContentsMargins(Space.sm, Space.sm, Space.sm, Space.sm)
        self.images_frame.setVisible(False)
        root.addWidget(self.images_frame)

        # -- input row --------------------------------------------------
        input_row = QHBoxLayout()
        self.input_edit = QLineEdit(self)
        self.input_edit.setPlaceholderText(
            "Ask the assistant — e.g. \"run autofocus -10 to 10 mm\". "
            "Enter to send.")
        self.input_edit.installEventFilter(self)
        input_row.addWidget(self.input_edit, 1)

        self.send_btn = QPushButton("Send", self)
        self.send_btn.setProperty("role", "primary")
        self.send_btn.clicked.connect(self._on_send_clicked)
        input_row.addWidget(self.send_btn)

        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.setProperty("role", "danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        input_row.addWidget(self.stop_btn)

        clear_btn = QPushButton("Clear", self)
        clear_btn.clicked.connect(self._on_clear_clicked)
        input_row.addWidget(clear_btn)

        root.addLayout(input_row)

        self._append_status(
            "Tip: the assistant can drive reconstruction, autofocus, QPI, "
            "and depth maps via tool calls, plus inspect/render the current "
            "field. Type a goal in plain language.")

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override name
        if obj is self.input_edit and event.type() == QEvent.Type.KeyPress:
            assert isinstance(event, QKeyEvent)
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._on_send_clicked()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _on_settings_changed(self) -> None:
        self._rebuild_client()
        self._refresh_health()

    def _rebuild_client(self) -> None:
        endpoint = self.endpoint_edit.text().strip() or _DEFAULT_ENDPOINT
        model = self.model_combo.currentText().strip() or _DEFAULT_MODEL
        self._client = LocalLLMClient(LLMConfig(endpoint=endpoint, model=model))

    def _set_health(self, text: str, role: str) -> None:
        """Set the health pill's text AND status colour together (role 'ok' /
        'danger' / 'muted'), repolishing so the ● dot recolours immediately —
        previously the pill stayed muted grey for every state."""
        self.health_label.setText(text)
        self.health_label.setProperty("role", role)
        self.health_label.style().unpolish(self.health_label)
        self.health_label.style().polish(self.health_label)

    def _refresh_health(self) -> None:
        """Kick a background health check — never blocks the GUI thread."""
        client = self._client
        if client is None:
            self._set_health("● unavailable", "danger")
            return
        # Supersede any in-flight probe: stop it from updating the label, but
        # let it run to completion and clean itself up. Do NOT wait on it here
        # (that blocked the GUI thread up to 50ms) and NEVER drop the only
        # reference to a still-running QThread — it stays tracked in
        # _health_threads and is destroyed only after finished (2026-07-06).
        prev = self._health_thread
        if prev is not None and prev.isRunning():
            try:
                prev.result.disconnect(self._on_health_result)
            except (RuntimeError, TypeError):
                pass
        self._set_health("● checking…", "muted")
        t = _HealthCheckThread(client, self)
        self._health_threads.add(t)
        t.result.connect(self._on_health_result)
        # finished is emitted with the QThread object's affinity (GUI thread),
        # so this discard runs on the GUI thread — safe to mutate the set.
        t.finished.connect(lambda: self._health_threads.discard(t))
        self._health_thread = t
        t.start()

    @Slot(bool)
    def _on_health_result(self, ok: bool) -> None:
        if ok:
            self._set_health("● connected", "ok")
        else:
            self._set_health("● unavailable", "danger")

    # ------------------------------------------------------------------
    # Send / Stop / Clear
    # ------------------------------------------------------------------
    def _on_send_clicked(self) -> None:
        if self._worker is not None:
            return
        prompt = self.input_edit.text().strip()
        if not prompt:
            return
        if self._client is None:
            self._append_status("AI client unavailable — check endpoint/model.")
            return
        self.input_edit.clear()
        self._append_user(prompt)

        snapshot = self._build_snapshot()
        tool_ctx = self._build_tool_context()
        worker = AIWorker(
            client=self._client,
            registry=self._registry,
            ctx=tool_ctx,
            prompt=prompt,
            history=list(self._history),
            snapshot=snapshot,
            max_iterations=8,
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

        self._history.append(ChatMessage(role="user", content=prompt))

        self._worker = worker
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._ctx.set_status("AI: thinking…", "info")
        worker.start()

    def _on_stop_clicked(self) -> None:
        if self._worker is None:
            return
        self._worker.requestInterruption()
        self._append_status("Cancelling…")

    def shutdown(self) -> None:
        """Stop the AI worker thread and wait for it — called by the shell on
        window close so a running turn doesn't outlive the widget graph."""
        # Wait on EVERY in-flight health probe (a superseded one may still be
        # blocked in requests.get), not just the latest, so no running QThread
        # is destroyed with the widget graph (2026-07-06 review).
        threads = [self._worker, *self._health_threads]
        for th in threads:
            if th is not None:
                try:
                    if hasattr(th, "requestInterruption"):
                        th.requestInterruption()
                    th.wait(3000)
                except Exception:
                    pass

    def closeEvent(self, event) -> None:  # noqa: N802
        self.shutdown()
        super().closeEvent(event)

    def _on_clear_clicked(self) -> None:
        self._history.clear()
        self.transcript.clear()
        self._clear_images()
        self._append_status("Chat cleared.")

    # ------------------------------------------------------------------
    # Worker signal handlers (GUI thread)
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
        if ok and call.name == "render_view":
            self._show_render_result(result)

    def _on_iteration_cap(self, max_iter: int) -> None:
        self._append_status(f"Hit iteration cap ({max_iter}). Refine and try again.")

    def _on_cancelled(self) -> None:
        self._append_status("Cancelled.")

    def _on_error_event(self, message: str) -> None:
        self._append_error(message)
        self._ctx.set_status(f"AI error: {message}", "error")

    def _on_worker_done(self) -> None:
        self._ctx.set_status("AI: done.", "ok")

    def _on_worker_finished(self) -> None:
        self._worker = None
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Snapshot + ToolContext factory
    # ------------------------------------------------------------------
    def _build_snapshot(self) -> StateSnapshot:
        params = self._ctx.get_params()
        recon_params = _recon_params_dict(params)
        af_params = {
            "z_min_mm": getattr(params, "af_z_min_mm", None),
            "z_max_mm": getattr(params, "af_z_max_mm", None),
            "metric": getattr(params, "autofocus_metric", None),
            "n_steps": getattr(params, "af_n_steps", None),
        }
        qpi_params = {
            "n_sample": getattr(params, "n_sample", None),
            "n_medium": getattr(params, "n_medium", None),
        }

        path = self._ctx.hologram_path()
        loaded_shape = None
        loaded_dtype = None
        raw = self._ctx.get_field("raw")
        if raw is not None:
            try:
                shape = tuple(int(x) for x in np.asarray(raw).shape[-2:])
                loaded_shape = (shape[0], shape[1]) if len(shape) >= 2 else None
                loaded_dtype = str(np.asarray(raw).dtype)
            except Exception:  # noqa: BLE001
                pass

        return StateSnapshot(
            loaded_path=str(path) if path else None,
            loaded_shape=loaded_shape,
            loaded_dtype=loaded_dtype,
            recon_params=recon_params,
            autofocus_params=af_params,
            qpi_params=qpi_params,
            last_recon_summary=self._recon_summary(),
            last_af_summary=None,
            last_qpi_summary=None,
            last_depth_summary=self._depth_summary(),
            stage_position_mm=None,
            audit_tail=(),
        )

    def _build_tool_context(self) -> ToolContext:
        # Read self._worker DYNAMICALLY: the context is built BEFORE the
        # AIWorker for this turn is constructed/assigned, so binding the
        # attribute here captured the PREVIOUS worker (None on first send)
        # and Stop could never cancel a polling tool (2026-07-06 review #2).
        def _is_cancelled() -> bool:
            try:
                w = self._worker
                return bool(w is not None and w.isInterruptionRequested())
            except Exception:  # noqa: BLE001
                return False

        from core.audit import get_audit_log
        from core.errors import get_error_center
        from core.sample_map import SampleMap
        from core.settings_schema import AIDefaults
        from core.stage import MockStage

        return ToolContext(
            state=self._build_snapshot,
            last_recon_summary=self._recon_summary,
            last_af_summary=lambda: None,
            last_qpi_summary=lambda: None,
            last_depth_summary=self._depth_summary,
            audit_tail=lambda limit: [],
            load_hologram=self._tool_load_hologram,
            set_recon_param=self._tool_set_recon_param,
            invoke_recon=self._tool_invoke_recon,
            invoke_autofocus=self._tool_invoke_autofocus,
            invoke_qpi=self._tool_invoke_qpi,
            invoke_depth_map=self._tool_invoke_depth_map,
            invoke_find_focus_candidates=self._tool_invoke_find_focus_candidates,
            stage=MockStage(),
            capture_frame=lambda: self._ctx.get_field("raw"),
            sample_map=SampleMap(),
            measure_sharpness=_measure_sharpness,
            persist_sample_map=lambda: None,
            capture_and_process=self._tool_capture_and_process,
            audit=get_audit_log(),
            error_center=get_error_center(),
            confirm=lambda name, args: True,
            settings=AIDefaults(),
            is_cancelled=_is_cancelled,
            shutter=None,
            led=None,
            orchestrator=None,
            get_last_field=self._ctx.get_field,
            set_reference_mode=self._set_reference_mode,
        )

    # ------------------------------------------------------------------
    # Summary helpers (read live results off the PanelContext)
    # ------------------------------------------------------------------
    def _recon_summary(self) -> Optional[dict]:
        r = self._ctx.last_recon()
        if r is None:
            return None
        phase = getattr(r, "phase", None)
        out: dict = {"runtime_ms": getattr(r, "runtime_ms", None)}
        if phase is not None:
            try:
                out["phase_std"] = float(np.std(np.asarray(phase)))
            except Exception:  # noqa: BLE001
                pass
        params = self._ctx.get_params()
        out["z_mm"] = float(getattr(params, "z_mm", 0.0))
        return out

    def _depth_summary(self) -> Optional[dict]:
        d = self._ctx.last_depth()
        if d is None:
            return None
        result = getattr(d, "result", d)
        z_map = getattr(result, "z_map", None)
        out: dict = {"runtime_ms": getattr(d, "runtime_ms", None)}
        if z_map is not None:
            try:
                arr = np.asarray(z_map)
                out["z_min_m"] = float(np.nanmin(arr))
                out["z_max_m"] = float(np.nanmax(arr))
            except Exception:  # noqa: BLE001
                pass
        return out

    # ------------------------------------------------------------------
    # GUI-thread bridge helpers — block the AI thread until the bridge's
    # completion signal fires (or a hard timeout as a safety net).
    # ------------------------------------------------------------------
    def _run_on_gui(self, fn: Callable[[], dict],
                    timeout_s: float = _PIPELINE_TIMEOUT_S) -> dict:
        if QThread.currentThread() is self.thread():
            return fn()

        # ``QTimer.singleShot(0, callable)`` posts to the *calling*
        # thread's event loop, not the receiver's — a bare callable has
        # no thread affinity, so on a QThread that never starts its own
        # Qt event loop (this AIWorker just runs ``agent.run()``
        # synchronously) the timer never fires and this would block
        # until ``timeout_s`` on every single pipeline call. v1 hit and
        # fixed this exact bug in ``_confirm`` (see its docstring) but
        # the fix was never generalised to this helper. Route through
        # ``QMetaObject.invokeMethod`` on ``self`` instead — queued
        # slot invocations always run on the *receiver's* thread (the
        # GUI thread for ``self``), regardless of the caller's thread.
        from PySide6.QtCore import QMetaObject, Qt as QtNs

        result_holder: dict = {}
        done = threading.Event()
        self._pending_gui_call = {"fn": fn, "result": result_holder, "done": done}

        QMetaObject.invokeMethod(
            self, "_run_pending_gui_call", QtNs.ConnectionType.QueuedConnection,
        )
        if not done.wait(timeout=timeout_s):
            return {"error": "timeout", "timeout_s": timeout_s}
        return dict(result_holder)

    @Slot()
    def _run_pending_gui_call(self) -> None:
        """Slot — always runs on the GUI thread. Executes the callable
        stashed by ``_run_on_gui`` and signals completion."""
        payload = getattr(self, "_pending_gui_call", None)
        if not payload:
            return
        try:
            payload["result"].update(payload["fn"]())
        except Exception as exc:  # noqa: BLE001
            payload["result"].update({"error": type(exc).__name__, "message": str(exc)})
        finally:
            payload["done"].set()

    @staticmethod
    def _wait_for_signal(signals, *, timeout_ms: int = 60_000) -> bool:
        """Spin a nested QEventLoop until any of ``signals`` fires (or
        timeout).

        Runs on the GUI thread (called from inside ``_run_on_gui``'s
        ``fn``), so a hard ``threading.Event.wait`` here would starve the
        very event loop that delivers the bridge's queued completion
        signal — deadlock. A ``QEventLoop`` keeps Qt's queue draining
        instead. Accepts either a single signal or an iterable of
        signals (e.g. a driver's ``*_done`` and ``*_error`` pair) — the
        loop quits as soon as *any* of them fires, so an error reply
        doesn't block until the timeout waiting for the success signal.
        """
        from PySide6.QtCore import QEventLoop
        from PySide6.QtWidgets import QApplication

        sig_list = list(signals) if hasattr(signals, "__iter__") else [signals]

        loop = QEventLoop()
        state = {"fired": False}

        def _on_fired(*_a, **_kw):
            state["fired"] = True
            loop.quit()

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)

        connected = []
        for sig in sig_list:
            try:
                sig.connect(_on_fired)
                connected.append(sig)
            except Exception:  # noqa: BLE001
                continue

        timer.start(int(timeout_ms))
        try:
            loop.exec()
        finally:
            for sig in connected:
                try:
                    sig.disconnect(_on_fired)
                except (RuntimeError, TypeError):
                    pass
            timer.stop()
        QApplication.processEvents()
        return bool(state["fired"])

    def _tool_load_hologram(self, path: str) -> dict:
        return self._run_on_gui(lambda: self._gui_load_hologram(path))

    def _gui_load_hologram(self, path: str) -> dict:
        # ui3's PanelContext has no generic "load a file" hook — loading
        # is owned by MainWindow's toolbar/drag-drop path, not exposed on
        # the frozen contract. Report a friendly, honest error instead of
        # guessing at a private MainWindow method.
        return {"error": "load_hologram is not available from this panel; "
                          "use File → Load hologram, then ask me to "
                          "continue from the loaded file."}

    def _tool_set_recon_param(self, params: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_set_recon_param(params))

    def _gui_set_recon_param(self, patch: dict) -> dict:
        p = self._ctx.get_params()
        applied = {}
        for key, val in patch.items():
            if hasattr(p, key):
                setattr(p, key, val)
                applied[key] = val
        if applied:
            self._ctx.params_changed()
        return {"updated": applied, "current": _recon_params_dict(p)}

    def _tool_invoke_recon(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_recon(args))

    def _gui_invoke_recon(self, args: dict) -> dict:
        path = self._ctx.hologram_path()
        if path is None:
            return {"error": "no hologram loaded"}
        params = self._ctx.get_params()
        sig = self._ctx.bridge.recon_done
        err_sig = self._ctx.bridge.recon_error
        got: dict = {}
        c_done = sig.connect(lambda r: got.update(recon=r))
        c_err = err_sig.connect(lambda m: got.update(err=m))
        try:
            self._ctx.bridge.reconstruct(path, params)
            # The drivers reject an overlapping op SYNCHRONOUSLY (on_error
            # runs inline during the bridge call), so the error may already
            # be captured — entering the nested event loop then would wait
            # the full timeout for an emission that already happened
            # (2026-07-06 review #2). Same guard on every bridged tool.
            if not got:
                self._wait_for_signal((sig, err_sig), timeout_ms=60_000)
        finally:
            _disconnect_one(sig, c_done)
            _disconnect_one(err_sig, c_err)
        if "err" in got:
            return {"error": "reconstruction failed", "message": got["err"]}
        if "recon" not in got:
            return {"error": "timeout waiting for reconstruction"}
        return {"submitted": True, "summary": self._recon_summary()}

    def _tool_invoke_autofocus(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_autofocus(args))

    def _gui_invoke_autofocus(self, args: dict) -> dict:
        path = self._ctx.hologram_path()
        if path is None:
            return {"error": "no hologram loaded"}
        params = self._ctx.get_params()
        z_min = float(args.get("z_min_mm", params.af_z_min_mm))
        z_max = float(args.get("z_max_mm", params.af_z_max_mm))
        n_steps = int(args.get("n_steps", params.af_n_steps))

        got: dict = {}
        sig = self._ctx.bridge.autofocus_done
        err_sig = self._ctx.bridge.autofocus_error
        c_done = sig.connect(lambda r: got.update(result=r))
        c_err = err_sig.connect(lambda m: got.update(err=m))
        try:
            self._ctx.bridge.autofocus(
                path, params, z_min_mm=z_min, z_max_mm=z_max, n_steps=n_steps)
            if not got:  # sync busy-rejection — see _gui_invoke_recon
                self._wait_for_signal((sig, err_sig), timeout_ms=60_000)
        finally:
            _disconnect_one(sig, c_done)
            _disconnect_one(err_sig, c_err)
        if "err" in got:
            return {"error": "autofocus failed", "message": got["err"]}
        result = got.get("result")
        if result is None:
            return {"error": "timeout waiting for autofocus"}
        best_z_mm = float(result.best_z_m) * 1e3
        params.z_mm = best_z_mm
        self._ctx.params_changed()
        return {
            "submitted": True,
            "best_z_mm": best_z_mm,
            "score": float(result.score),
            "scanned": int(result.scanned),
        }

    def _tool_invoke_qpi(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_qpi(args))

    def _gui_invoke_qpi(self, args: dict) -> dict:
        path = self._ctx.hologram_path()
        if path is None:
            return {"error": "no hologram loaded"}
        params = self._ctx.get_params()
        if "n_sample" in args:
            params.n_sample = float(args["n_sample"])
        if "n_medium" in args:
            params.n_medium = float(args["n_medium"])

        got: dict = {}
        sig = self._ctx.bridge.qpi_done
        err_sig = self._ctx.bridge.qpi_error
        c_done = sig.connect(lambda r: got.update(result=r))
        c_err = err_sig.connect(lambda m: got.update(err=m))
        try:
            self._ctx.bridge.qpi(
                path, params, z_mm=float(params.z_mm),
                n_sample=params.n_sample, n_medium=params.n_medium)
            if not got:  # sync busy-rejection — see _gui_invoke_recon
                self._wait_for_signal((sig, err_sig), timeout_ms=60_000)
        finally:
            _disconnect_one(sig, c_done)
            _disconnect_one(err_sig, c_err)
        if "err" in got:
            return {"error": "QPI failed", "message": got["err"]}
        result = got.get("result")
        if result is None:
            return {"error": "timeout waiting for QPI"}
        qpi = result.qpi
        out: dict = {"submitted": True, "runtime_ms": float(result.runtime_ms)}
        if qpi.total_dry_mass_pg is not None:
            out["total_dry_mass_pg"] = float(qpi.total_dry_mass_pg)
        try:
            out["opd_range_nm"] = [float(np.min(qpi.opd_nm)), float(np.max(qpi.opd_nm))]
        except Exception:  # noqa: BLE001
            pass
        return out

    def _tool_invoke_depth_map(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_depth_map(args))

    def _gui_invoke_depth_map(self, args: dict) -> dict:
        path = self._ctx.hologram_path()
        if path is None:
            return {"error": "no hologram loaded"}
        params = self._ctx.get_params()
        z_min = float(args.get("z_min_mm", params.af_z_min_mm))
        z_max = float(args.get("z_max_mm", params.af_z_max_mm))
        n_steps = int(args.get("n_steps", 40))
        window_size = int(args.get("window_size", 5))

        got: dict = {}
        sig = self._ctx.bridge.depth_done
        err_sig = self._ctx.bridge.depth_error
        c_done = sig.connect(lambda r: got.update(result=r))
        c_err = err_sig.connect(lambda m: got.update(err=m))
        try:
            self._ctx.bridge.depth_map(
                path, params, z_min_mm=z_min, z_max_mm=z_max,
                n_steps=n_steps, window_size=window_size)
            if not got:  # sync busy-rejection — see _gui_invoke_recon
                self._wait_for_signal((sig, err_sig), timeout_ms=120_000)
        finally:
            _disconnect_one(sig, c_done)
            _disconnect_one(err_sig, c_err)
        if "err" in got:
            return {"error": "depth map failed", "message": got["err"]}
        if got.get("result") is None:
            return {"error": "timeout waiting for depth map"}
        return {"submitted": True, "summary": self._depth_summary()}

    def _tool_invoke_find_focus_candidates(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_invoke_find_focus_candidates(args))

    def _gui_invoke_find_focus_candidates(self, args: dict) -> dict:
        path = self._ctx.hologram_path()
        if path is None:
            return {"error": "no hologram loaded"}
        params = self._ctx.get_params()
        z_min = float(args.get("z_min_mm", params.af_z_min_mm))
        z_max = float(args.get("z_max_mm", params.af_z_max_mm))
        n_steps = max(int(args.get("n_steps", params.af_n_steps)), 2)

        got: dict = {}
        sig = self._ctx.bridge.focus_candidates_done
        err_sig = self._ctx.bridge.focus_candidates_error
        c_done = sig.connect(lambda r: got.update(result=r))
        c_err = err_sig.connect(lambda m: got.update(err=m))
        try:
            self._ctx.bridge.find_focus_candidates(
                path, params, z_min_mm=z_min, z_max_mm=z_max, n_steps=n_steps)
            if not got:  # sync busy-rejection — see _gui_invoke_recon
                self._wait_for_signal((sig, err_sig), timeout_ms=60_000)
        finally:
            _disconnect_one(sig, c_done)
            _disconnect_one(err_sig, c_err)
        if "err" in got:
            return {"error": "find_focus_candidates failed", "message": got["err"]}
        result = got.get("result")
        if result is None:
            return {"error": "timeout waiting for find_focus_candidates"}
        candidates = list(getattr(result, "candidates", []) or [])
        top_k = int(args.get("top_k", 5))
        return {
            "submitted": True,
            "candidates": [
                {"z_mm": float(c.z_m) * 1e3, "score": float(c.score), "rank": int(c.rank)}
                for c in candidates[:top_k]
            ],
        }

    def _tool_capture_and_process(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_capture_and_process(args))

    def _gui_capture_and_process(self, args: dict) -> dict:
        raw = self._ctx.get_field("raw")
        if raw is None:
            return {"error": "no frame to capture (load a hologram first)"}
        out: dict = {"shape": list(np.asarray(raw).shape)}
        if bool(args.get("run_recon", False)):
            r = self._gui_invoke_recon({})
            if "error" not in r:
                summary = r.get("summary") or {}
                for k in ("phase_std", "z_mm"):
                    if k in summary:
                        out[k] = summary[k]
            else:
                out["recon_error"] = r.get("message", r.get("error"))
        if bool(args.get("run_qpi", False)):
            r = self._gui_invoke_qpi({})
            if "error" not in r:
                for k in ("total_dry_mass_pg", "opd_range_nm"):
                    if k in r:
                        out[k] = r[k]
            else:
                out["qpi_error"] = r.get("message", r.get("error"))
        return out

    def _set_reference_mode(self, args: dict) -> dict:
        return self._run_on_gui(lambda: self._gui_set_reference_mode(args))

    def _gui_set_reference_mode(self, args: dict) -> dict:
        mode = str(args.get("mode", "")).strip().lower()
        if mode not in ("off", "reference", "reference_free"):
            return {"error": f"unknown mode {mode!r}"}
        params = self._ctx.get_params()
        params.reference_mode = mode
        # Keep the legacy checkbox flag in lockstep. effective_reference_mode()
        # treats mode=="off" + subtract_reference=True as "reference"; a stale
        # True left over from an earlier reference load would silently override
        # an explicit "off"/"reference_free" here (2026-07-06 review).
        params.subtract_reference = (mode == "reference")
        # Optional inline reference path (validated by the tool layer) — lets
        # the copilot arm reference mode without the file dialog, parity with
        # the headless/MCP path.
        ref_path = args.get("reference_path")
        if ref_path:
            from pathlib import Path as _Path
            params.reference_path = _Path(str(ref_path))
        if "bg_method" in args:
            params.reffree_bg_method = str(args["bg_method"])
        if "bg_order" in args:
            # Route to the knob the ACTIVE method consumes: the pipeline
            # reads reffree_n_terms for zernike and reffree_bg_order for
            # polynomial. Writing bg_order only to reffree_bg_order made the
            # AI's knob a silent no-op in zernike mode, diverging from the
            # headless/MCP behavior (2026-07-06 review #2).
            method = str(params.reffree_bg_method or "zernike")
            if method == "zernike":
                params.reffree_n_terms = int(args["bg_order"])
            else:
                params.reffree_bg_order = int(args["bg_order"])
        self._ctx.params_changed()
        return {"ok": True, "mode": mode}

    # ------------------------------------------------------------------
    # render_view → inline QPixmap
    # ------------------------------------------------------------------
    def _show_render_result(self, result: dict) -> None:
        path = result.get("path")
        if not path:
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        pix = pix.scaledToWidth(320, Qt.TransformationMode.SmoothTransformation)
        row = QLabel(self.images_frame)
        row.setPixmap(pix)
        caption = QLabel(f"{result.get('kind', 'view')} — {path}", self.images_frame)
        caption.setProperty("role", "muted")
        caption.setWordWrap(True)
        self._images_layout.addWidget(row)
        self._images_layout.addWidget(caption)
        self.images_frame.setVisible(True)

    def _clear_images(self) -> None:
        while self._images_layout.count():
            item = self._images_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.images_frame.setVisible(False)

    # ------------------------------------------------------------------
    # Transcript rendering
    # ------------------------------------------------------------------
    def _append(self, text: str) -> None:
        self.transcript.appendPlainText(text)

    def _append_user(self, text: str) -> None:
        self._append(f"You: {text}")

    def _append_assistant(self, text: str) -> None:
        self._append(f"Assistant: {text}")

    def _append_tool_start(self, name: str, args: dict) -> None:
        body = json.dumps(args, default=str)
        self._append(f"  → {name}({body})")

    def _append_tool_done(self, name: str, ok: bool, result: dict) -> None:
        marker = "OK" if ok else "FAIL"
        body = json.dumps(_truncate_for_display(result), default=str)
        self._append(f"  [{marker}] {name} → {body}")

    def _append_status(self, text: str) -> None:
        self._append(f"· {text}")

    def _append_error(self, message: str) -> None:
        self._append(f"! Error: {message}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _recon_params_dict(params: Any) -> dict:
    keys = ("wavelength_nm", "pixel_um", "z_mm", "mask_radius", "n_medium",
            "n_sample", "reference_mode")
    out = {}
    for k in keys:
        if hasattr(params, k):
            out[k] = getattr(params, k)
    # The snapshot's pixel_um feeds physics in the observation tools
    # (inspect_phase dry mass ∝ pixel², render_view scalebar). The pipeline
    # itself always works at the EFFECTIVE pixel (camera pixel / objective
    # magnification); passing the raw camera pixel inflated dry mass by M²
    # and mislabeled the scalebar by M with any magnifying preset
    # (2026-07-06 review #2).
    if hasattr(params, "effective_pixel_um"):
        try:
            eff = float(params.effective_pixel_um())
            raw = out.get("pixel_um")
            out["pixel_um"] = eff
            # Surface the camera pixel separately when it differs, so the
            # copilot doesn't mistake the effective value for a wrong echo
            # of its own set_recon_param(pixel_um=...) call.
            if raw is not None and abs(float(raw) - eff) > 1e-12:
                out["pixel_um_camera"] = float(raw)
        except Exception:  # noqa: BLE001 — fall back to the raw value
            pass
    return out


def _measure_sharpness(frame) -> float:
    if frame is None:
        return -float("inf")
    a = np.asarray(frame)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        return -float("inf")
    amp = np.abs(a).astype(np.float32, copy=False)
    lap = (
        amp[1:-1, :-2] + amp[1:-1, 2:]
        + amp[:-2, 1:-1] + amp[2:, 1:-1]
        - 4.0 * amp[1:-1, 1:-1]
    )
    v = float(np.var(lap))
    return v if np.isfinite(v) else -float("inf")


def _truncate_for_display(obj: Any, max_chars: int = 400) -> Any:
    text = json.dumps(obj, default=str)
    if len(text) <= max_chars:
        return obj
    return {"_truncated": True, "preview": text[:max_chars] + "…"}


__all__ = ["AIPanel", "AIWorker"]
