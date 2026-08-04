"""Regression tests for the 2026-07-08 CLAUDE.md-lens review fixes.

A find→verify→synthesize multi-agent review surfaced a family of silent-degrade
defects (a normal-looking but quantitatively WRONG result, no user signal) plus
one LLM-driven path-traversal write. These pin the fix_now batch.
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


# --- reference_path survives the state round-trip -------------------------

def test_reference_path_round_trips_and_does_not_arm_a_pathless_reference(tmp_path):
    """Persisting reference_mode='reference' while dropping reference_path
    (a Path) re-armed reference mode with no path on relaunch → a silently
    unreferenced reconstruction. The path must survive the round-trip."""
    from core.drivers.reconstruction import ReconParams
    from ui3.state import Ui3State, load_state, save_state

    p = ReconParams()
    p.reference_path = Path("/data/ref.tif")
    p.reference_mode = "reference"
    p.subtract_reference = True
    st = Ui3State()
    st.set_recon_params(p)
    assert st.params["reference_path"] == "/data/ref.tif"   # serialised as str

    fp = tmp_path / "s.json"
    save_state(st, fp)
    back = load_state(fp).recon_params()
    assert back.reference_path == Path("/data/ref.tif")
    assert isinstance(back.reference_path, Path)
    assert back.effective_reference_mode() == "reference"    # mode + path agree


def test_unset_reference_path_reloads_as_none_not_dot(tmp_path):
    from core.drivers.reconstruction import ReconParams
    from ui3.state import Ui3State, load_state, save_state
    st = Ui3State()
    st.set_recon_params(ReconParams())          # reference_path is None
    fp = tmp_path / "s.json"
    save_state(st, fp)
    assert load_state(fp).recon_params().reference_path is None   # not Path('.')


# --- LLM sample_id cannot escape the state dir (security) -----------------

def _grid_args(sample_id: str) -> dict:
    return {"x_min_mm": 0.0, "x_max_mm": 1.0, "y_min_mm": 0.0, "y_max_mm": 1.0,
            "step_mm": 0.5, "sample_id": sample_id}


@pytest.mark.parametrize("evil", [
    "../../../../tmp/pwn/x", "..", ".", "/etc/passwd", "a/b", "a\\b",
    "x" * 65, "name with spaces", "sub;rm -rf",
])  # note: "" is intentionally NOT here — it defaults to the safe "sample"
def test_map_sample_grid_rejects_unsafe_sample_id(evil):
    """CONFIRMED path-traversal: sample_id became state_dir/{id}.json with no
    sanitisation. Anything outside a safe stem must be rejected BEFORE any
    filesystem write (guard runs before ctx is touched, so ctx=None here)."""
    from core.ai.tool_impls import _tool_map_sample_grid
    out = _tool_map_sample_grid(None, _grid_args(evil))
    assert out.get("error") == "invalid sample_id", (evil, out)


def test_map_sample_grid_accepts_safe_sample_id_past_the_guard():
    """A normal id passes the charset guard (it then fails later on ctx=None,
    proving it got PAST the guard rather than being rejected)."""
    from core.ai.tool_impls import _tool_map_sample_grid
    with pytest.raises(AttributeError):     # ctx=None, but guard already passed
        _tool_map_sample_grid(None, _grid_args("cellA_2026"))


# --- coarse_to_fine forwards the ROI into the Golden fine phase -----------

def test_coarse_to_fine_forwards_roi_bounds_to_golden(monkeypatch):
    """The coarse sweep used the ROI evaluator but the Golden fine phase
    rebuilt a full-frame one — ROI autofocus silently optimised the whole
    frame in refinement. roi_bounds must reach golden_section_search."""
    import core.autofocus.search_classic as sc
    from core.autofocus import FocusMetric
    from core.phase_tracker import ROIBounds
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    seen = {}
    real_golden = sc.golden_section_search

    def spy(*args, **kwargs):
        seen["roi_bounds"] = kwargs.get("roi_bounds", "MISSING")
        return real_golden(*args, **kwargs)

    monkeypatch.setattr(sc, "golden_section_search", spy)
    n = 48
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.6 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 10) ** 2)))
    field = ((0.3 + bump) * np.exp(1j * (0.2 * X + bump))).astype(np.complex64)
    base = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=1e-6,
                                z_m=0.0, n=1.0)
    roi = ROIBounds(x0=8, y0=8, x1=40, y1=40)
    sc.coarse_to_fine_search(field, base, ReconstructionMethod.ASM,
                             FocusMetric.LAPLACIAN_VARIANCE,
                             z_min_m=-1e-3, z_max_m=1e-3, coarse_steps=5,
                             roi_bounds=roi)
    assert seen["roi_bounds"] is roi


# --- default 'robust' reports its true evaluation count ------------------

def test_extract_evaluations_honours_total_evaluations():
    """robust's RobustSearchResult exposes total_evaluations, not
    evaluations — the default path silently understated its cost ~2x."""
    from core.drivers.workers import _extract_evaluations
    assert _extract_evaluations(SimpleNamespace(total_evaluations=72), 40) == 72
    assert _extract_evaluations(SimpleNamespace(evaluations=30), 40) == 30
    assert _extract_evaluations(SimpleNamespace(), 40) == 40   # fallback


# --- headless recon summary carries the unsuffixed phase aliases ---------

def _synthetic_offaxis_tif(path: Path, n: int = 128) -> Path:
    import tifffile
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.5 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * 0.28 * X + bump))
    holo = np.abs(1 + obj) ** 2
    tifffile.imwrite(path, (holo * 1000).astype(np.uint16))
    return path


def test_headless_recon_summary_has_unsuffixed_phase_aliases(tmp_path):
    """record_timelapse's extractor only reads 'phase_std'; headless emitted
    only 'phase_std_rad', so every MCP timelapse frame lost the phase-drift
    signal. Both keys must be present now."""
    from dhm_mcp.headless import HeadlessSession
    s = HeadlessSession()
    s.load_hologram(str(_synthetic_offaxis_tif(tmp_path / "h.tif")))
    out = s.invoke_recon({})
    summary = out["summary"]
    assert "phase_std" in summary and "phase_std_rad" in summary
    assert summary["phase_std"] == summary["phase_std_rad"]
    assert "phase_mean" in summary


# --- recon panel keeps the 'no reference loaded' warning -----------------

@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def test_recon_panel_reference_warning_not_overwritten(qapp):
    """Selecting 'Reference' with no reference loaded must LEAVE the warning
    on the status bar — it used to be immediately overwritten by an info
    line, hiding that the reconstruction would come out unreferenced."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    panel = win._recon_panel      # the rich control-dock panel (source of truth)
    assert panel is not None
    win._params.reference_path = None
    panel._on_reference_mode_changed("Reference")
    text = win._status_label.text()
    assert "no reference is loaded" in text
    assert "Reference mode: Reference" not in text
    win.close()


# --- B-108: cell morphology refuses to fabricate a Δn -----------------------

def test_cell_morphology_raises_on_zero_contrast():
    """n_sample==n_medium (contrast unknown) used to silently fabricate
    height/volume from a hardcoded Δn=0.043. It must now RAISE, matching the
    sibling opd_to_height, instead of inventing physical numbers."""
    from core.qpi import compute_cell_morphology, phase_to_opd
    opd = np.full((16, 16), phase_to_opd(1.2, _WL := 632.8e-9), dtype=np.float64)
    mask = np.ones((16, 16), dtype=bool)
    with pytest.raises(ValueError, match="contrast"):
        compute_cell_morphology(opd, pixel_size_m=0.1e-6, mask=mask,
                                n_sample=1.35, n_medium=1.35,
                                wavelength_m=_WL)


def test_compute_qpi_logs_and_falls_back_when_contrast_unknown(caplog):
    """compute_qpi's morphology branch degrades to the whole-field dry-mass
    integral when contrast is unknown — that's fine (dry mass needs no Δn) —
    but it must LOG the reason (was a silent bare except), and it must NOT
    emit a fabricated cell_morph."""
    import logging
    from core.qpi import QPIMode, compute_qpi
    n = 32
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    phase = (3.0 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2)
                            / (2 * (n / 6) ** 2)))).astype(np.float64)  # rad bump
    with caplog.at_level(logging.WARNING, logger="core.qpi"):
        res = compute_qpi(phase, wavelength_m=632.8e-9, pixel_size_m=0.1e-6,
                          mode=QPIMode.BIO, n_sample=1.35, n_medium=1.35)
    assert res.cell_morph is None                    # no fabricated morphology
    assert res.total_dry_mass_pg is not None         # dry mass still computed
    assert any("morphology unavailable" in r.message for r in caplog.records)


# --- B-109: acquisition thread surfaces its death via on_error --------------

class _FlakySource:
    """Minimal CameraSource whose grab() fails after `ok_frames` frames."""
    def __init__(self, fail_on_start=False, ok_frames=1):
        self.fps = 30.0
        self._fail_on_start = fail_on_start
        self._ok = ok_frames
        self.stopped = False

    def start(self):
        if self._fail_on_start:
            raise RuntimeError("device busy")

    def grab(self):
        if self._ok <= 0:
            raise RuntimeError("cable unplugged")
        self._ok -= 1
        return np.zeros((8, 8), dtype=np.float32)

    def stop(self):
        self.stopped = True


def _run_until_error(source) -> str:
    import threading
    from core.drivers.camera_feed import AcquisitionThread
    box = {}
    done = threading.Event()

    def on_error(msg):
        box["msg"] = msg
        done.set()

    th = AcquisitionThread(source, on_frame=lambda f: None,
                           on_error=on_error)
    th.start()
    assert done.wait(timeout=5.0), "on_error never fired"
    th.stop()
    th.join(timeout=2.0)
    return box["msg"]


def test_acquisition_thread_emits_on_error_when_grab_fails():
    """A grab() blow-up mid-run used to kill the pump silently (logged only) →
    the UI kept the last frame with 'recording' armed. on_error must fire."""
    msg = _run_until_error(_FlakySource(ok_frames=1))
    assert "grab failed" in msg and "cable unplugged" in msg


def test_acquisition_thread_emits_on_error_when_start_fails():
    msg = _run_until_error(_FlakySource(fail_on_start=True))
    assert "start failed" in msg and "device busy" in msg


# --- B-110: a runtime reference-division failure is SURFACED, not swallowed --

def _offaxis_holo(n: int = 96) -> np.ndarray:
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.5 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * 0.28 * X + bump))
    return (np.abs(1 + obj) ** 2).astype(np.float32)


def test_extract_field_notes_a_failed_reference_division():
    """reference mode armed + an unloadable reference must fall back to the
    unreferenced field AND return an actionable note (was: silent fallback)."""
    from core.drivers.workers import _extract_field_with_reference
    from core.offaxis import OffAxisParams
    from core.drivers.reconstruction import ReconParams

    holo = _offaxis_holo()
    off = OffAxisParams(radius=30)
    armed = ReconParams(mask_radius=30, reference_mode="reference",
                        subtract_reference=True,
                        reference_path=Path("/nonexistent/ref.tif"))
    field, _c, note = _extract_field_with_reference(holo, armed, off)
    assert note is not None and "UNREFERENCED" in note
    # fell back to the plain unreferenced field
    plain, _c2, note2 = _extract_field_with_reference(
        holo, ReconParams(mask_radius=30, reference_mode="off"), off)
    assert note2 is None
    assert np.allclose(field, plain)


def test_prepare_field_threads_the_reference_note(tmp_path, monkeypatch):
    """_prepare_field (shared by autofocus/qpi/depth) must return the note as
    its 4th element so those pipelines can surface it too."""
    import core.drivers.workers as w
    from core.drivers.reconstruction import ReconParams
    holo_path = tmp_path / "h.tif"
    import tifffile
    tifffile.imwrite(holo_path, (_offaxis_holo() * 1000).astype(np.uint16))
    armed = ReconParams(mask_radius=30, reference_mode="reference",
                        subtract_reference=True,
                        reference_path=Path("/nope/ref.tif"))
    field, base, method, note = w._prepare_field(holo_path, armed)
    assert note is not None and "UNREFERENCED" in note


def test_recon_done_surfaces_reference_note(qapp, monkeypatch):
    """The shell must warn (status + toast) when the reconstruction came back
    unreferenced — not paint a clean 'Reconstructed in N ms'."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    toasts = []
    monkeypatch.setattr(win._toasts, "show_toast",
                        lambda text, level="info": toasts.append((text, level)))
    res = SimpleNamespace(
        amplitude=np.zeros((8, 8), np.float32), phase=np.zeros((8, 8), np.float32),
        unwrapped_phase=None, reffree_note=None, runtime_ms=5.0,
        reference_note="Reference division FAILED (...); this result is "
                       "UNREFERENCED")
    win._on_recon_done(res)
    assert "UNREFERENCED" in win._status_label.text()
    assert any(lvl == "warn" for _t, lvl in toasts)
    win.close()


def test_qpi_and_depth_panels_surface_reference_warning(qapp, monkeypatch):
    from ui3.main_window import MainWindow
    win = MainWindow()
    toasts = []
    monkeypatch.setattr(win._toasts, "show_toast",
                        lambda text, level="info": toasts.append((text, level)))

    qpanel = win._panels.get("qpi")
    # isolate the status surfacing from the readout render (not under test)
    monkeypatch.setattr(qpanel, "_render_last_result", lambda: None)
    qpanel._on_qpi_done(SimpleNamespace(
        qpi=SimpleNamespace(total_dry_mass_pg=12.3), runtime_ms=5.0,
        warning="Reference division FAILED — UNREFERENCED"))
    assert "UNREFERENCED" in win._status_label.text()
    assert any(lvl == "warn" for _t, lvl in toasts)

    dpanel = win._panels.get("depth")
    z = np.linspace(0, 1e-3, 64).reshape(8, 8).astype(np.float32)
    dpanel._on_depth_done(SimpleNamespace(
        result=SimpleNamespace(z_map=z), runtime_ms=5.0,
        warning="Reference division FAILED — UNREFERENCED"))
    assert "UNREFERENCED" in win._status_label.text()
    win.close()


# --- B-111: auto_select_metric rejects peakless (edge-focus) metrics --------

def test_auto_select_metric_prefers_interior_peak_over_monotonic(monkeypatch):
    """A metric whose focus is OUTSIDE the scan (monotonic, no interior peak)
    used to score ~1.0 (curve height) and could beat a metric with a genuine
    interior peak (prominence < 1). The peakless metric is the LEAST reliable
    and must lose — it now scores 0."""
    import core.autofocus.analysis as an
    from core.autofocus import FocusMetric
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    candidates = [fm for fm in FocusMetric if fm != FocusMetric.ENTROPY]
    peaked, monotonic = candidates[0], candidates[1]
    n = 21
    zc = n // 2
    idx = np.arange(n)
    peak_curve = np.exp(-((idx - zc) ** 2) / (2 * 3.0 ** 2))      # interior bump
    ramp_curve = idx / (n - 1)                                    # monotonic edge

    state = {"fm": None, "i": 0}

    def fake_metric(_complex_field, fm):
        if fm is not state["fm"]:
            state["fm"] = fm
            state["i"] = 0
        i = state["i"]
        state["i"] += 1
        if fm is peaked:
            return float(peak_curve[i])
        if fm is monotonic:
            return float(ramp_curve[i])
        return 0.5  # flat → vmax-vmin≈0 → excluded (score -999)

    monkeypatch.setattr(an, "_calc_metric", fake_metric)
    # _is_minimize must be False for our two curves so higher == better.
    monkeypatch.setattr(an, "_is_minimize", lambda fm: False)

    field = (np.random.default_rng(0).random((32, 32))
             + 1j * np.random.default_rng(1).random((32, 32))).astype(np.complex64)
    base = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=1e-6,
                                z_m=0.0, n=1.0)
    chosen = an.auto_select_metric(field, base, ReconstructionMethod.ASM,
                                   z_min_m=-1e-3, z_max_m=1e-3, n_steps=n)
    assert chosen is peaked, f"picked the peakless edge metric {chosen}"


# --- ui3 UX/layout fixes (2026-07-08 user feedback: scroll, overlap) ---------

def test_every_feature_panel_is_scroll_wrapped(qapp):
    """Panels 500–740px tall used to clip inside a shorter dock with no way
    to reach the lower controls. Each dock's widget must be a resizable
    QScrollArea now."""
    from PySide6.QtWidgets import QScrollArea
    from ui3.main_window import MainWindow
    win = MainWindow()
    assert win._docks, "no feature docks mounted"
    for key, dock in win._docks.items():
        if key == "reconstruct":
            continue  # the recon control dock is its own widget
        w = dock.widget()
        assert isinstance(w, QScrollArea), f"{key} dock not scroll-wrapped"
        assert w.widgetResizable(), f"{key} scroll area not resizable"
    win.close()


def test_workflow_modes_claim_every_feature_dock(qapp):
    """An unclaimed dock leaks into every mode (the 'unmanaged → always
    visible' branch). timelapse used to; pin that no dock is orphaned."""
    from ui3.main_window import MainWindow, WORKFLOW_MODES
    win = MainWindow()
    # Reproduce the mode_docks map used by _apply_workflow_mode.
    managed = {"camera", "device", "timelapse", "reconstruct",
               "qpi", "depth", "focus", "ai", "report", "audit"}
    orphans = set(win._docks) - managed
    assert not orphans, f"docks claimed by no workflow mode: {orphans}"
    for mode in WORKFLOW_MODES:      # each mode applies without error
        win._apply_workflow_mode(mode)
    win.close()


def test_present_centered_moves_dialog_onto_parent(qapp):
    """present_centered must reposition a dialog near the parent centre
    instead of leaving it at Qt's default (overlapping) spot."""
    from PySide6.QtWidgets import QDialog
    from ui3.dialogs._present import present_centered
    from ui3.main_window import MainWindow
    win = MainWindow()
    win.setGeometry(200, 150, 1000, 700)
    win.show()
    dlg = QDialog(win)
    dlg.resize(300, 200)
    present_centered(dlg)
    pc = win.frameGeometry().center()
    dc = dlg.frameGeometry().center()
    # centred within a tolerance that also allows the small cascade offset
    assert abs(dc.x() - pc.x()) < 120 and abs(dc.y() - pc.y()) < 120, (
        f"dialog centre {dc} not near parent centre {pc}")
    dlg.close()
    win.close()


def test_loading_a_hologram_syncs_the_recon_panel_label(qapp, tmp_path):
    """The ReconPanel's 'Hologram' label read ctx.hologram_path() only at
    construction, so after a load it still said '(no hologram loaded)' next to
    a clearly-loaded file (2026-07-10 UI audit)."""
    import numpy as np
    import tifffile
    from PySide6.QtWidgets import QLabel
    from ui3.main_window import MainWindow
    holo = tmp_path / "cell.tif"
    tifffile.imwrite(holo, (np.abs(1 + 0.3 * np.exp(
        1j * 2 * np.pi * 0.28 * np.arange(64)))[None].repeat(64, 0)
        * 1000).astype(np.uint16))
    win = MainWindow()
    win._load_path(str(holo))
    rp = win._recon_panel
    texts = [l.text() for l in rp.findChildren(QLabel)]
    assert "cell.tif" in texts, "ReconPanel hologram label not synced on load"
    assert "(no hologram loaded)" not in texts
    win.close()


def test_stale_dock_layout_is_not_restored(qapp, monkeypatch):
    """A window_state saved under a different dock schema must NOT be
    restored — restoring it left mismatched docks floating over the central
    grid (2026-07-10 real-screen bug). Only a matching layout_version restores."""
    from ui3.main_window import MainWindow, _LAYOUT_VERSION
    from ui3.state import Ui3State

    st = Ui3State(onboarding_seen=True)
    st.window_geometry_b64 = "not-empty"
    st.window_state_b64 = "c29tZS1zdGFsZS1zdGF0ZQ=="   # base64, wrong schema
    st.layout_version = _LAYOUT_VERSION - 1              # stale
    win = MainWindow(st)
    called = {"restoreState": False}
    monkeypatch.setattr(win, "restoreState",
                        lambda *a, **k: called.__setitem__("restoreState", True))
    win._restore_geometry()
    assert called["restoreState"] is False, "stale dock layout was restored"

    # A matching version DOES restore.
    st.layout_version = _LAYOUT_VERSION
    win._restore_geometry()
    assert called["restoreState"] is True
    win.close()


def test_export_state_stamps_current_layout_version(qapp):
    from ui3.main_window import MainWindow, _LAYOUT_VERSION
    win = MainWindow()
    out = win.export_state()
    assert out.layout_version == _LAYOUT_VERSION
    win.close()


def test_reset_panel_layout_redocks_a_floated_dock(qapp):
    """A dock the user floated (to inspect it) and couldn't put back must
    return home on Reset panel layout — the missing affordance behind
    'I took a panel out and it stayed like that' (2026-07-10)."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    # float a couple of docks, as the user did
    win._docks["reconstruct"].setFloating(True)
    any_feature = next(k for k in win._docks if k != "reconstruct")
    win._docks[any_feature].setFloating(True)
    assert win._docks["reconstruct"].isFloating()

    win._reset_panel_layout()

    assert not win._docks["reconstruct"].isFloating(), "reconstruct still floating"
    assert not win._docks[any_feature].isFloating(), f"{any_feature} still floating"
    # command is discoverable in the palette too
    assert any(cmd.id == "reset_layout" for cmd in win.commands.all())
    win.close()


def test_workflow_mode_forms_one_tab_group_from_visible_docks(qapp):
    """Merely toggling setVisible() on a pre-tabified 8-dock group left Qt's
    tab bar broken: in Analyse only 'focus' rendered and qpi/depth/ai were
    unreachable ('close one, another appears behind'). Each mode must now
    present ALL its feature docks as one complete tab group (2026-07-10)."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    win.show()

    win._apply_workflow_mode("Analyse")
    want = {"focus", "qpi", "depth", "ai"}
    visible = {k for k, d in win._docks.items()
               if d.isVisibleTo(win) and k != "reconstruct"}
    assert visible == want, f"Analyse should show {want}, showed {visible}"
    # and they are all one tab group, not stacked-without-tabbar
    anchor = win._docks["focus"]
    grouped = {t.objectName().replace("dock_", "")
               for t in win.tabifiedDockWidgets(anchor)} | {"focus"}
    assert want <= grouped, f"docks not one tab group: {grouped}"

    win._apply_workflow_mode("Acquire")
    visible = {k for k, d in win._docks.items()
               if d.isVisibleTo(win) and k != "reconstruct"}
    assert visible == {"camera", "device", "timelapse"}
    win.close()


# --- B-118: spin/combo arrows render as real PNG icons, not blank squares ----

def test_arrow_icons_are_generated_and_referenced(qapp, tmp_path):
    """Qt's stylesheet loader draws NO arrow for the CSS border-triangle trick
    and ignores data: URIs (both verified empirically), so unstyled QSpinBox /
    QComboBox arrows rendered as blank light-gray squares. With a live
    QApplication, build_qss must instead reference generated PNG icon files via
    `image: url(...)`, and those files must exist and be non-empty. Headless
    (no QApplication) must degrade to width:0 arrows, never the blank-square
    fallback."""
    import pathlib
    from ui3 import design
    from ui3.design import build_qss, get_palette, _ensure_arrow_icons

    p = get_palette("dark")
    icons = _ensure_arrow_icons(p)
    assert icons is not None, "with a QApplication, icons must generate"
    for key in ("up", "down", "chev"):
        fp = pathlib.Path(icons[key])
        assert fp.exists() and fp.stat().st_size > 0, f"{key} icon missing/empty"

    qss = build_qss(p)
    # every arrow sub-control resolves to a real image, not a border-triangle
    assert qss.count("image: url(") >= 3
    assert "QSpinBox::up-arrow" in qss and "QComboBox::down-arrow" in qss
    # the blank-square border-triangle trick must be gone
    assert "border-bottom: 4px solid" not in qss

    # Retina crispness: icons are rendered at 2x the logical QSS size.
    from PySide6.QtGui import QImage
    img = QImage(icons["up"])
    assert img.width() == 18 and img.height() == 12  # 2x of logical 9x6


# --- B-119..B-124: polish sprint (2026-07-10 UI audit follow-up) -----------

def test_qss_adds_status_roles_indicator_and_splitter():
    """The polish pass added: QGroupBox::indicator (so checkable group titles
    stay visible on the black high-contrast theme), a collapsed-group rule (no
    orphaned empty frame), QSplitter::handle (grid dividers), and status-tinted
    QLabel roles (the AI health pill)."""
    from ui3.design import build_qss, get_palette
    for name in ("dark", "high_contrast", "light", "midnight"):
        qss = build_qss(get_palette(name))
        assert "QGroupBox::indicator" in qss
        assert 'QGroupBox[collapsed="true"]' in qss
        assert "QSplitter::handle" in qss
        for role in ("ok", "warn", "danger"):
            assert f'QLabel[role="{role}"]' in qss


def test_advanced_group_starts_collapsed_and_toggles(qapp):
    """The Advanced group is a collapsed disclosure row by default (the
    [collapsed] property drops its card frame) and re-frames when expanded."""
    from PySide6.QtWidgets import QGroupBox
    from ui3.main_window import MainWindow
    win = MainWindow()
    adv = next(g for g in win._recon_panel.findChildren(QGroupBox)
               if g.title() == "Advanced")
    assert adv.property("collapsed") is True
    adv.setChecked(True)
    assert adv.property("collapsed") is False
    adv.setChecked(False)
    assert adv.property("collapsed") is True
    win.close()


def test_qpi_batch_tables_fit_headers_not_stretch(qapp):
    """Equal Stretch clipped headers like 'DRY MASS (PG)' mid-word; both QPI
    tables now size columns to content with a stretchy last section."""
    from PySide6.QtWidgets import QHeaderView
    from ui3.main_window import MainWindow
    from ui3.dialogs.qpi_batch import QPIBatchDialog
    win = MainWindow()
    qpi = win._panels.get("qpi")
    if qpi is not None:
        hdr = qpi.batch_table.horizontalHeader()
        assert hdr.sectionResizeMode(0) == QHeaderView.ResizeToContents
        assert hdr.stretchLastSection()
    dlg = QPIBatchDialog(win.panel_context(), win)
    dhdr = dlg.table.horizontalHeader()
    assert dhdr.sectionResizeMode(0) == QHeaderView.ResizeToContents
    assert dhdr.stretchLastSection()
    win.close()


def test_error_toasts_use_recognized_level(qapp):
    """toast('danger') fell back to the blue 'accent' role — a red error toast
    needs level 'error' (which maps to the 'danger' role)."""
    from ui3.widgets.toast import _LEVEL_ROLE
    assert _LEVEL_ROLE.get("error") == "danger"
    assert "danger" not in _LEVEL_ROLE  # 'danger' is a ROLE, not a toast LEVEL
    import pathlib
    base = pathlib.Path(ROOT) / "src" / "ui3" / "panels"
    for fname in ("recon_panel.py", "qpi_panel.py", "report_panel.py"):
        text = (base / fname).read_text()
        assert ', "danger")' not in text, f"{fname} still passes 'danger' level"


def test_ai_health_pill_recolors_by_state(qapp):
    """The health pill role tracks state instead of staying muted grey."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    ai = win._panels.get("ai")
    if ai is not None:
        ai._set_health("● connected", "ok")
        assert ai.health_label.property("role") == "ok"
        ai._set_health("● unavailable", "danger")
        assert ai.health_label.property("role") == "danger"
    win.close()


def test_no_duplicate_file_label_and_timelapse_named_consistently(qapp):
    """The header filename caption was removed (it duplicated the panel's
    'Hologram' group) and the time-lapse dock title now matches its heading."""
    from ui3.main_window import MainWindow
    win = MainWindow()
    assert not hasattr(win, "_file_label")
    d = win._docks.get("timelapse")
    if d is not None:
        assert d.windowTitle() == "Time-lapse"
    win.close()


# --- B-124: window autofits the screen so the control buttons stay reachable -

def test_window_autofits_screen_and_control_dock_scrolls(qapp):
    """A window sized taller than the display pushed the Reconstruct/Autofocus
    buttons off the bottom edge (user report 2026-07-10). The window now autofits
    the usable screen every launch, and the scroll-wrapped control dock makes the
    lower buttons reachable by scrolling on a short window."""
    from PySide6.QtWidgets import QApplication, QScrollArea, QPushButton
    from ui3.main_window import MainWindow
    win = MainWindow()
    win.show()
    QApplication.processEvents()
    avail = (win.screen() or QApplication.primaryScreen()).availableGeometry()
    # autofit: the window fits within the usable screen, never larger.
    assert win.width() <= avail.width()
    assert win.height() <= avail.height()
    # On a short window the control dock must scroll, not clip its buttons.
    # (autofit already fired once on show, so this manual resize sticks.)
    win.resize(win.width(), min(700, avail.height()))
    QApplication.processEvents()
    sa = win._docks["reconstruct"].widget()
    assert isinstance(sa, QScrollArea)
    assert sa.verticalScrollBar().maximum() > 0, "control dock does not scroll"
    assert any(b.text() == "Reconstruct"
               for b in win._recon_panel.findChildren(QPushButton))
    win.close()


# --- B-125: docked panels show their name once (title bar), not 2-3x ---------

def test_docked_panels_hide_redundant_heading_dialogs_keep_it(qapp):
    """Each feature panel repeated its name up to 3x (tab + dock title bar + a
    big in-panel role='heading' label). In a dock the heading is hidden — the
    title bar / tab is the single label — while standalone dialog viewers, which
    have no dock title bar, keep their heading as their only title."""
    from PySide6.QtWidgets import QLabel
    from ui3.main_window import MainWindow
    win = MainWindow()

    def visible_headings(w):
        return [l for l in w.findChildren(QLabel)
                if l.property("role") == "heading" and not l.isHidden()]

    # Control dock + every feature dock: in-panel heading hidden.
    assert visible_headings(win._recon_panel) == []
    for key, panel in win._panels.items():
        assert visible_headings(panel) == [], f"{key} still shows its heading"

    # A standalone dialog keeps its heading (its only title).
    from ui3.dialogs.qpi_batch import QPIBatchDialog
    dlg = QPIBatchDialog(win.panel_context(), win)
    assert len(visible_headings(dlg)) == 1
    win.close()


# --- B-126: design-token tokenization of stray magic control sizes ----------

def test_control_sizes_derive_from_design_tokens(qapp):
    """Tokenization pass (audit #24/#26/#31/#35): input+button min-height derive
    from Space.xl (was a 22/20 mix off the 4-scale); the viewport header uses a
    px Type token (was a 10pt magic number, preserved at 13px); camera/focus
    root margins use Space.md (was a hardcoded 10)."""
    from PySide6.QtGui import QFontInfo
    from ui3.design import build_qss, get_palette, DARK, Space, Type
    qss = build_qss(get_palette("dark"))
    assert f"min-height: {Space.xl}px" in qss
    assert "min-height: 22px" not in qss and "min-height: 20px" not in qss

    from ui3.viewport import ImagePanel
    panel = ImagePanel("X", DARK)
    assert QFontInfo(panel._header.font()).pixelSize() == Type.label

    from ui3.main_window import MainWindow
    win = MainWindow()
    for key in ("camera", "focus"):
        p = win._panels.get(key)
        if p is not None:
            m = p.layout().contentsMargins()
            assert m.left() == Space.md and m.top() == Space.md, key
    win.close()
