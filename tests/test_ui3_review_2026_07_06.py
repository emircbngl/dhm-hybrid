"""Regression tests for the 2026-07-06 ui3/observe/dhm_mcp code review.

Each test pins a specific CONFIRMED finding fixed that day (bug registry
B-072..B-081). Qt runs headless via the offscreen platform (set before any Qt
import) — a real QApplication drives real widgets, no stubbing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def _synthetic_hologram(path: Path, n: int = 128) -> Path:
    import tifffile
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.5 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * 0.28 * X + bump))
    holo = np.abs(1 + obj) ** 2
    tifffile.imwrite(path, (holo * 1000).astype(np.uint16))
    return path


# ---------------------------------------------------------------------------
# B-072 — _on_load_reference crashed on a deleted _cb_ref_mode combo
# ---------------------------------------------------------------------------

def test_load_reference_does_not_crash_and_arms_reference(qapp, tmp_path, monkeypatch):
    """CONFIRMED: _on_load_reference called self._cb_ref_mode.setCurrentText,
    but that combo was removed when the inline dock became ReconPanel →
    AttributeError on every reference load. It must mutate params (mode +
    legacy flag) and re-hydrate the ReconPanel instead."""
    from ui3.main_window import MainWindow
    from PySide6.QtWidgets import QFileDialog

    holo = _synthetic_hologram(tmp_path / "ref.tif")
    win = MainWindow()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(holo), "")))

    win._on_load_reference()   # must not raise

    assert win._params.reference_mode == "reference"
    assert win._params.subtract_reference is True
    assert win._params.effective_reference_mode() == "reference"
    win.close()


# ---------------------------------------------------------------------------
# B-073 — bare signal.disconnect() in the AI panel severed every listener
# ---------------------------------------------------------------------------

def test_disconnect_one_only_removes_its_own_connection(qapp):
    """CONFIRMED: the bridged AI tools called sig.disconnect() (no arg),
    dropping *all* slots on recon_done/etc. — after one copilot op the whole
    app stopped reacting. _disconnect_one must detach only its own handle."""
    from ui3.bridge import WorkerBridge
    from ui3.panels.ai_panel import _disconnect_one

    bridge = WorkerBridge()
    other_hits, mine_hits = [], []
    bridge.recon_done.connect(lambda r: other_hits.append(r))
    mine = bridge.recon_done.connect(lambda r: mine_hits.append(r))

    _disconnect_one(bridge.recon_done, mine)

    bridge.recon_done.emit("payload")
    assert other_hits == ["payload"], "unrelated listener was wrongly severed"
    assert mine_hits == [], "own listener should have been detached"
    bridge.shutdown()


# ---------------------------------------------------------------------------
# B-074 — observe.render_view scalebar ignored the downsample stride
# ---------------------------------------------------------------------------

def test_downsample_reports_stride():
    from core.observe import _downsample
    big = np.zeros((1000, 1000), dtype=np.float32)
    small, step = _downsample(big, max_size=256)
    assert step >= 4
    assert max(small.shape) <= 256
    tiny = np.zeros((50, 50), dtype=np.float32)
    same, step1 = _downsample(tiny, max_size=256)
    assert step1 == 1 and same.shape == (50, 50)


def test_render_view_scalebar_uses_effective_pixel_size(monkeypatch):
    """CONFIRMED: the scalebar was drawn with the pre-downsample pixel size but
    the post-downsample frame width → label off by the downsample factor. The
    effective pixel pitch must scale by the stride."""
    import core.observe as observe

    seen = {}

    def _spy(ax, frame_width_px, pixel_size_um):
        seen["frame_width_px"] = frame_width_px
        seen["pixel_size_um"] = pixel_size_um

    monkeypatch.setattr(observe, "_draw_scalebar", _spy)
    field = np.random.default_rng(0).random((1000, 1000)).astype(np.float32)
    observe.render_view(field, kind="amplitude", pixel_size_um=0.2, max_size=256)

    # stride = ceil(1000/256) = 4 → effective pitch = 0.2 * 4 = 0.8
    assert seen["frame_width_px"] <= 256
    assert seen["pixel_size_um"] == pytest.approx(0.8, rel=1e-6)


# ---------------------------------------------------------------------------
# B-075 — state.py stringified af_roi, corrupting the round-trip
# ---------------------------------------------------------------------------

def test_af_roi_round_trips_as_tuple(tmp_path):
    from ui3.state import Ui3State, load_state, save_state
    from ui2.reconstruction import ReconParams

    p = ReconParams()
    p.af_roi = (0.1, 0.2, 0.7, 0.8)
    st = Ui3State()
    st.set_recon_params(p)
    # serialised as a JSON list, never str(tuple)
    assert st.params["af_roi"] == [0.1, 0.2, 0.7, 0.8]

    fp = tmp_path / "s.json"
    save_state(st, fp)
    back = load_state(fp).recon_params()
    assert back.af_roi == (0.1, 0.2, 0.7, 0.8)
    assert isinstance(back.af_roi, tuple)


# ---------------------------------------------------------------------------
# B-076 — busy_changed was a shared boolean across two executors
# ---------------------------------------------------------------------------

def test_busy_changed_is_reference_counted(qapp):
    from ui3.bridge import WorkerBridge
    bridge = WorkerBridge()
    events = []
    bridge.busy_changed.connect(lambda busy, label: events.append(busy))

    bridge._begin("op-A")
    bridge._begin("op-B")
    bridge._end()                      # one still in flight → no idle yet
    assert events[-1] is True, "idle signalled while an op was still running"
    bridge._end()                      # last op done → idle
    assert events[-1] is False
    assert events.count(False) == 1
    bridge.shutdown()


# ---------------------------------------------------------------------------
# B-077 — AI set_reference_mode left the legacy subtract_reference flag stale
# ---------------------------------------------------------------------------

def test_ai_set_reference_mode_keeps_legacy_flag_in_lockstep(qapp):
    from ui3.main_window import MainWindow
    win = MainWindow()
    panel = win._panels.get("ai")
    assert panel is not None

    # Arm reference (legacy flag on), then explicitly turn it off.
    panel._gui_set_reference_mode({"mode": "reference"})
    assert win._params.subtract_reference is True
    panel._gui_set_reference_mode({"mode": "off"})
    assert win._params.subtract_reference is False
    assert win._params.effective_reference_mode() == "off"
    win.close()


# ---------------------------------------------------------------------------
# B-082 — MainWindow._on_depth_done double-painted the shared viewport
# ---------------------------------------------------------------------------

def test_mainwindow_depth_done_caches_without_painting(qapp):
    """DepthPanel owns the viewport paint; the shell handler must only cache
    the result for get_field('depth'), not repaint the spectrum panel."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    zmap = np.linspace(0, 1e-3, 64 * 64).reshape(64, 64).astype(np.float32)
    result = SimpleNamespace(result=SimpleNamespace(z_map=zmap))
    win._on_depth_done(result)
    assert win._last_depth is result          # cached for AI vision / surface
    assert not win.panel_spectrum.has_image()  # not repainted by the shell
    win.close()


# ---------------------------------------------------------------------------
# B-083 — qpi/candidates double-wiring: duplicate status + double populate
# ---------------------------------------------------------------------------

def test_single_status_writer_per_result_signal(qapp, monkeypatch):
    """CONFIRMED (finding 15/17): MainWindow, the panels AND the dialogs all
    wrote status for the same result signal (double/triple status per op).
    Ownership now: QPIPanel owns one-shot qpi status; the dialogs own the
    batch/candidates status; the shell writes none of them."""
    from ui3 import main_window as mw

    calls = []
    orig = mw.MainWindow.set_status

    def spy(self, text, level="info"):
        calls.append((text, level))
        return orig(self, text, level)

    monkeypatch.setattr(mw.MainWindow, "set_status", spy)
    win = mw.MainWindow()

    # One-shot QPI → exactly one status write (QPIPanel's, with dry mass).
    calls.clear()
    fake_qpi = SimpleNamespace(
        qpi=SimpleNamespace(total_dry_mass_pg=12.3, phase_stats=None,
                            step_height_m=None, cell_morph=None),
        runtime_ms=5.0)
    win.bridge.qpi_done.emit(fake_qpi)
    qapp.processEvents()
    qpi_msgs = [t for t, _ in calls if "QPI" in t]
    assert len(qpi_msgs) == 1, calls
    assert "dry mass" in qpi_msgs[0]

    # Candidates → exactly one status write (the dialog's "Found N").
    calls.clear()
    win.bridge.focus_candidates_done.emit(SimpleNamespace(candidates=[]))
    qapp.processEvents()
    found_msgs = [t for t, _ in calls if "ocus candidate" in t]
    assert len(found_msgs) == 1, calls

    win.close()


def test_qpi_batch_dialog_populates_once_and_is_presented(qapp, monkeypatch):
    """CONFIRMED: the dialog populated its table via its own qpi_batch_done
    connection AND the shell called open_with(result) → double populate.
    The shell must only present() (show/raise, no repopulate)."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    dlg = win._dialogs.get("qpi_batch")
    assert dlg is not None

    counts = {"n": 0}
    orig_set_result = dlg.set_result

    def counting_set_result(result):
        counts["n"] += 1
        return orig_set_result(result)

    monkeypatch.setattr(dlg, "set_result", counting_set_result)
    win.bridge.qpi_batch_done.emit(SimpleNamespace(entries=[]))
    qapp.processEvents()

    assert counts["n"] == 1, "table populated more than once per result"
    assert dlg.isVisible(), "shell should present the dialog on batch done"
    win.close()


# ===========================================================================
# Review round #2 (adversarial workflow over the round-1 fix diff) —
# B-086..B-094
# ===========================================================================

def test_headless_set_reference_mode_invalidates_derived_cache():
    """B-086: mode/bg knobs change what invoke_recon produces just like
    recon params — the round-1 fix covered set_recon_param but left
    set_reference_mode (and invoke_autofocus's direct z write) serving the
    stale reconstruction to inspect_*."""
    from dhm_mcp.headless import HeadlessSession
    s = HeadlessSession()
    s.recon_complex = np.ones((4, 4), dtype=np.complex64)
    s.phase_unwrapped = np.ones((4, 4), dtype=np.float32)
    s.depth = np.ones((4, 4), dtype=np.float32)
    out = s.set_reference_mode({"mode": "reference_free"})
    assert out.get("ok") is True
    assert s.recon_complex is None
    assert s.phase_unwrapped is None
    assert s.depth is None


def test_ui3_snapshot_pixel_um_is_effective():
    """B-087: the snapshot's pixel_um feeds dry mass (∝ pixel²) and the
    scalebar; it must be the EFFECTIVE pixel (camera/magnification), not the
    raw camera pixel — 40× objective inflated dry mass 1600×."""
    from ui2.reconstruction import ReconParams
    from ui3.panels.ai_panel import _recon_params_dict
    p = ReconParams(pixel_um=3.45, magnification=40.0,
                    pixel_is_effective=False)
    d = _recon_params_dict(p)
    assert d["pixel_um"] == pytest.approx(3.45 / 40.0)
    # Legacy/direct-imaging setups (pixel already effective) pass through.
    p2 = ReconParams(pixel_um=0.5, magnification=1.0, pixel_is_effective=True)
    assert _recon_params_dict(p2)["pixel_um"] == pytest.approx(0.5)


def test_ai_tool_does_not_wait_after_synchronous_rejection(qapp, monkeypatch, tmp_path):
    """B-088: the drivers reject overlapping ops synchronously (on_error runs
    inline during the bridge call); the captured error must short-circuit the
    nested-event-loop wait instead of stalling the AI tool for the full
    60-120s timeout."""
    from ui3.main_window import MainWindow
    from ui3.panels import ai_panel as ap
    win = MainWindow()
    panel = win._panels.get("ai")
    assert panel is not None
    win._hologram_path = tmp_path / "h.tif"

    def sync_reject(path, params, **kw):
        win.bridge.qpi_error.emit("Another analysis is running")

    monkeypatch.setattr(win.bridge, "qpi", sync_reject)

    def must_not_wait(*a, **k):
        raise AssertionError("entered _wait_for_signal after a synchronous "
                             "rejection was already captured")

    monkeypatch.setattr(ap.AIPanel, "_wait_for_signal",
                        staticmethod(must_not_wait))
    out = panel._gui_invoke_qpi({})
    assert out.get("error") == "QPI failed"
    assert "Another analysis" in out.get("message", "")
    win.close()


def test_stale_idle_from_worker_is_dropped_when_new_op_started(qapp):
    """B-089: the idle decision must be made at DELIVERY time on the GUI
    thread — a busy=False queued by a finishing worker must not clear the
    indicator when a new op started in between."""
    from ui3.bridge import WorkerBridge
    bridge = WorkerBridge()
    events = []
    bridge.busy_changed.connect(lambda busy, label: events.append(busy))

    bridge._begin("op-A")
    # Simulate the worker-thread half of _end: decrement WITHOUT the
    # (queued) idle check having been delivered yet...
    with bridge._busy_lock:
        bridge._inflight -= 1
    # ...user starts a new op before the queued check lands...
    bridge._begin("op-B")
    # ...now the stale check is delivered:
    bridge._publish_idle()
    assert events[-1] is True, "stale idle cleared the busy indicator"
    # And when the real last op ends, idle is published exactly once.
    bridge._end()
    assert events[-1] is False
    bridge.shutdown()


def test_set_status_does_not_auto_toast(qapp, monkeypatch):
    """B-090: set_status auto-toasted ok/warn/error on top of the panels'
    own explicit ctx.toast → two stacked toasts per event. Status and toast
    are now separate channels."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    shown = []
    monkeypatch.setattr(win._toasts, "show_toast",
                        lambda text, level="info": shown.append(text))
    win.set_status("just a status", "ok")
    assert shown == []
    win.toast("a real toast", "ok")
    assert shown == ["a real toast"]
    win.close()


def test_ai_bg_order_routes_to_method_specific_knob(qapp):
    """B-091: bg_order must land on the knob the active method consumes
    (zernike → reffree_n_terms, polynomial → reffree_bg_order); writing only
    reffree_bg_order made the AI knob a silent no-op in zernike mode."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    panel = win._panels.get("ai")
    p = win._params
    default_bg_order = p.reffree_bg_order
    panel._gui_set_reference_mode(
        {"mode": "reference_free", "bg_method": "zernike", "bg_order": 9})
    assert p.reffree_n_terms == 9
    assert p.reffree_bg_order == default_bg_order
    panel._gui_set_reference_mode(
        {"mode": "reference_free", "bg_method": "polynomial", "bg_order": 5})
    assert p.reffree_bg_order == 5
    win.close()


def test_is_cancelled_sees_worker_assigned_after_context_build(qapp):
    """B-092: the tool context is built BEFORE the turn's AIWorker exists;
    binding self._worker at build time froze the PREVIOUS worker (None on
    first send), so Stop could never cancel a polling tool."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    panel = win._panels.get("ai")
    panel._worker = None
    ctx = panel._build_tool_context()
    assert ctx.is_cancelled() is False
    panel._worker = SimpleNamespace(isInterruptionRequested=lambda: True)
    assert ctx.is_cancelled() is True
    win.close()


def test_render_view_spectrum_ffts_complex_fields(monkeypatch):
    """B-093: a complex input to kind='spectrum' is a SPATIAL field and must
    be fft2'd; the old branch fftshift'ed the field itself (quadrant-swapped
    |field| mislabeled log|F|)."""
    import core.observe as observe
    called = {}
    orig_fft2 = np.fft.fft2

    def spy(a, *args, **kw):
        called["fft2"] = True
        return orig_fft2(a, *args, **kw)

    monkeypatch.setattr(np.fft, "fft2", spy)
    field = np.exp(1j * np.linspace(0, 6 * np.pi, 64 * 64)).reshape(64, 64)
    png = observe.render_view(field.astype(np.complex64), kind="spectrum")
    assert called.get("fft2"), "complex spectrum input skipped the FFT"
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_field_unwraps_wrapped_phase_on_demand(qapp):
    """B-094: outside the reffree path unwrapped_phase is None and get_field
    silently returned WRAPPED phase — corrupting OPD/dry mass computed by the
    observation tools. It must unwrap on demand and cache the result."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    # A smooth 8-rad Gaussian bump (wraps beyond 2π; compact + smooth, so
    # every unwrap method recovers it — a pure tilt ramp is degenerate for
    # the FFT-Poisson path).
    x = np.arange(64, dtype=np.float32)
    X, Y = np.meshgrid(x, x)
    true_phase = 8.0 * np.exp(-(((X - 32) ** 2 + (Y - 32) ** 2)
                                / (2 * 10.0 ** 2))).astype(np.float32)
    wrapped = np.angle(np.exp(1j * true_phase)).astype(np.float32)
    assert float(np.ptp(wrapped)) <= 2 * np.pi + 1e-3
    win._last_recon = SimpleNamespace(
        unwrapped_phase=None, phase=wrapped,
        amplitude=np.ones_like(wrapped))
    out = win.get_field("phase_unwrapped")
    assert out is not None
    # Unwrapped: range exceeds what any wrapped map can span (2π), and the
    # surface correlates with the true bump (wrapped maps decorrelate at the
    # 2π discontinuities).
    assert float(np.ptp(out)) > 2 * np.pi + 0.2, \
        "still wrapped — on-demand unwrap did not run"
    # Structurally closer to the truth than the wrapped input (absolute
    # fidelity is the unwrapper's own tested domain, not this contract).
    corr_out = float(np.corrcoef(out.ravel(), true_phase.ravel())[0, 1])
    corr_wrapped = float(np.corrcoef(wrapped.ravel(), true_phase.ravel())[0, 1])
    assert corr_out > max(corr_wrapped, 0.9), \
        f"unwrap did not improve on wrapped input (r={corr_out:.3f} vs {corr_wrapped:.3f})"
    assert win._last_recon.unwrapped_phase is not None, "result not cached"
    win.close()


# ---------------------------------------------------------------------------
# dhm_mcp headless — B-078/079/080
# ---------------------------------------------------------------------------

def test_headless_set_recon_param_invalidates_derived_cache():
    """B-078: changing recon params must drop the cached derived fields so a
    later inspect_* doesn't report the stale reconstruction."""
    from dhm_mcp.headless import HeadlessSession
    s = HeadlessSession()
    s.demod_complex = np.ones((4, 4), dtype=np.complex64)
    s.recon_complex = np.ones((4, 4), dtype=np.complex64)
    s.phase_unwrapped = np.ones((4, 4), dtype=np.float32)
    s.depth = np.ones((4, 4), dtype=np.float32)
    s.set_recon_param({"z_mm": 1.5})
    assert s.demod_complex is None
    assert s.recon_complex is None
    assert s.phase_unwrapped is None
    assert s.depth is None


def test_headless_reference_path_populates_reference_raw(tmp_path):
    """B-079: reference mode was unreachable over MCP because reference_raw was
    never populated. set_reference_mode(reference_path=...) must load it."""
    from dhm_mcp.headless import HeadlessSession
    holo = _synthetic_hologram(tmp_path / "ref.tif")
    s = HeadlessSession()
    # Without a reference, arming "reference" is refused.
    assert "error" in s.set_reference_mode({"mode": "reference"})
    # With a path, it loads and arms.
    out = s.set_reference_mode({"mode": "reference", "reference_path": str(holo)})
    assert out.get("ok") is True
    assert s.reference_raw is not None
    assert s.reference_mode["mode"] == "reference"


def test_headless_bg_order_defaults_are_method_specific(tmp_path):
    """B-080: the single bg_order knob fed BOTH n_terms and polynomial_order
    with a shared default → a degenerate order for the unused method. Each
    method must get its own default when bg_order is unset."""
    import dhm_mcp.headless as headless
    from dhm_mcp.headless import HeadlessSession

    calls = []

    def _spy(phase, amplitude=None, method="zernike", n_terms=15,
             polynomial_order=4, **kw):
        calls.append({"method": method, "n_terms": n_terms,
                      "polynomial_order": polynomial_order})
        return phase.astype(np.float32), SimpleNamespace(background=np.zeros_like(phase))

    monkey = pytest.MonkeyPatch()
    monkey.setattr(headless, "subtract_background", _spy, raising=False)
    # subtract_background is imported *inside* invoke_recon; patch the source.
    import core.background_phase as bp
    monkey.setattr(bp, "subtract_background", _spy)

    holo = _synthetic_hologram(tmp_path / "h.tif")
    s = HeadlessSession()
    s.load_hologram(str(holo))

    # polynomial with no explicit bg_order → polynomial_order defaults to 4,
    # NOT 15 (the zernike default).
    s.set_reference_mode({"mode": "reference_free", "bg_method": "polynomial"})
    s.invoke_recon({})
    assert calls[-1]["method"] == "polynomial"
    assert calls[-1]["polynomial_order"] == 4

    # zernike with no explicit bg_order → n_terms defaults to 15.
    s.set_reference_mode({"mode": "reference_free", "bg_method": "zernike"})
    s.invoke_recon({})
    assert calls[-1]["method"] == "zernike"
    assert calls[-1]["n_terms"] == 15
    monkey.undo()
