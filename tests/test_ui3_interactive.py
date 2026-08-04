"""ui3 interactive dialogs/widgets tests — offscreen Qt.

Covers the four deliverables:
* ``ui3.dialogs.preset_dialogs.PresetStore`` — Qt-free save/list/delete
  round-trip against a tmp-path-backed JSON file.
* ``ui3.dialogs.onboarding.OnboardingDialog`` — builds, pages through, and
  reports the "don't show again" choice.
* ``ui3.widgets.preset_chips.PresetChipRow`` — exclusive selection signal +
  rebuild.
* ``ui3.widgets.line_profile.LineProfileTool`` — sampling on a synthetic
  array via ``core.line_profile``, exercised through an ``ImagePanel``.

Mirrors the ``_synthetic_hologram`` / ``qapp`` fixture pattern from
``tests/test_ui3_spine.py`` and ``tests/test_ui3_recon_panel.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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


def _make_ctx(events=None):
    """A minimal PanelContext for dialogs that only need status/toast +
    palette (no bridge interaction in this test file)."""
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.context import PanelContext
    from ui3.design import DARK

    events = events if events is not None else {"status": [], "toast": []}
    bridge = WorkerBridge()
    params = ReconParams(wavelength_nm=632.8, pixel_um=1.0, z_mm=0.0, mask_radius=40)
    ctx = PanelContext(
        bridge=bridge,
        get_params=lambda: params,
        params_changed=lambda: None,
        hologram_path=lambda: None,
        last_recon=lambda: None,
        last_depth=lambda: None,
        get_field=lambda target: None,
        set_status=lambda text, level="info": events["status"].append((text, level)),
        toast=lambda text, level="info": events["toast"].append((text, level)),
        show_in_panel=lambda target, array, **kw: None,
        palette=DARK,
    )
    return ctx, bridge, params, events


# ---------------------------------------------------------------------------
# PresetStore (Qt-free)
# ---------------------------------------------------------------------------

def test_preset_store_save_list_delete_roundtrip(tmp_path):
    from ui3.dialogs.preset_dialogs import PresetStore

    path = tmp_path / "ui3_presets.json"
    store = PresetStore(path)

    assert store.list() == []
    assert store.get("Lab1") is None

    store.save("Lab1", dict(
        wavelength_nm=700.0, pixel_um=1.0, z_mm=0.0, mask_radius=40,
        method="ASM", magnification=1.0, pixel_is_effective=True,
        n_sample=1.38, n_medium=1.337, autofocus_metric="LAPLACIAN_VARIANCE",
        junk_field_not_in_preset_fields="dropped",
    ))
    assert store.list() == ["Lab1"]
    got = store.get("Lab1")
    assert got["wavelength_nm"] == pytest.approx(700.0)
    assert "junk_field_not_in_preset_fields" not in got

    # Persisted to disk — a fresh store instance sees it.
    store2 = PresetStore(path)
    assert store2.list() == ["Lab1"]

    assert store.delete("Lab1") is True
    assert store.list() == []
    assert store.delete("Lab1") is False  # already gone

    # Deletion persisted too.
    store3 = PresetStore(path)
    assert store3.list() == []


def test_preset_store_missing_file_returns_defaults(tmp_path):
    from ui3.dialogs.preset_dialogs import PresetStore

    store = PresetStore(tmp_path / "does_not_exist.json")
    assert store.list() == []


def test_preset_store_corrupt_file_falls_back(tmp_path):
    from ui3.dialogs.preset_dialogs import PresetStore

    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = PresetStore(path)
    assert store.list() == []


def test_save_preset_dialog_writes_through(qapp, tmp_path):
    from ui3.dialogs.preset_dialogs import PresetStore, SavePresetDialog

    store = PresetStore(tmp_path / "ui3_presets.json")
    params_dict = dict(
        wavelength_nm=532.0, pixel_um=3.45, z_mm=10.0, mask_radius=80,
        method="Fresnel", magnification=40.0, pixel_is_effective=False,
        n_sample=1.4, n_medium=1.337, autofocus_metric="TENENGRAD",
    )
    dlg = SavePresetDialog(store, params_dict, reserved_names=("Cell", "Custom"))
    dlg.edit_name.setText("MyPreset")
    dlg._on_accept()

    assert dlg.saved_name == "MyPreset"
    assert "MyPreset" in store.list()
    assert store.get("MyPreset")["method"] == "Fresnel"


def test_delete_preset_dialog_lists_and_removes(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from ui3.dialogs.preset_dialogs import PresetStore, DeletePresetDialog

    store = PresetStore(tmp_path / "ui3_presets.json")
    store.save("A", dict(wavelength_nm=1.0))
    store.save("B", dict(wavelength_nm=2.0))

    # Auto-confirm the "Delete preset 'X'?" question box.
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    dlg = DeletePresetDialog(store)
    names = [dlg.list_widget.item(i).text() for i in range(dlg.list_widget.count())]
    assert sorted(names) == ["A", "B"]

    dlg.list_widget.setCurrentRow(names.index("A"))
    dlg._on_delete()

    assert dlg.deleted_name == "A"
    assert store.list() == ["B"]
    remaining = [dlg.list_widget.item(i).text() for i in range(dlg.list_widget.count())]
    assert remaining == ["B"]


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------

def test_onboarding_dialog_builds_and_pages(qapp):
    from ui3.dialogs.onboarding import OnboardingDialog, PAGES

    dlg = OnboardingDialog()
    assert dlg.current_page_index() == 0
    assert not dlg.btn_back.isEnabled()

    dlg._go_next()
    assert dlg.current_page_index() == 1
    assert dlg.btn_back.isEnabled()

    dlg._go_back()
    assert dlg.current_page_index() == 0

    # Walk to the last page — Next becomes "Finish".
    for _ in range(len(PAGES) - 1):
        dlg._go_next()
    assert dlg.current_page_index() == len(PAGES) - 1
    assert dlg.btn_next.text() == "Finish"


def test_onboarding_reports_dont_show_again_true(qapp):
    from ui3.dialogs.onboarding import OnboardingDialog

    dlg = OnboardingDialog()
    dlg.cb_dont_show.setChecked(True)

    got = {}
    dlg.finished_with_choice.connect(lambda seen: got.update(seen=seen))
    dlg._finish()

    assert got["seen"] is True


def test_onboarding_reports_dont_show_again_false_by_default(qapp):
    from ui3.dialogs.onboarding import OnboardingDialog

    dlg = OnboardingDialog()
    got = {}
    dlg.finished_with_choice.connect(lambda seen: got.update(seen=seen))
    dlg._on_skip()

    assert got["seen"] is False


def test_onboarding_integration_writes_ui3_state(qapp, tmp_path):
    """Simulates the MainWindow integration: the dialog's reported choice
    is written into Ui3State.onboarding_seen and persisted."""
    from ui3.dialogs.onboarding import OnboardingDialog
    from ui3.state import Ui3State, save_state, load_state

    state = Ui3State()
    assert state.onboarding_seen is False

    dlg = OnboardingDialog()
    dlg.cb_dont_show.setChecked(True)

    def _on_finished(seen: bool) -> None:
        state.onboarding_seen = bool(seen)

    dlg.finished_with_choice.connect(_on_finished)
    dlg._finish()

    assert state.onboarding_seen is True

    path = tmp_path / "ui3_state.json"
    save_state(state, path)
    reloaded = load_state(path)
    assert reloaded.onboarding_seen is True


# ---------------------------------------------------------------------------
# PresetChipRow
# ---------------------------------------------------------------------------

def test_preset_chip_row_selection_signal(qapp):
    from ui3.widgets.preset_chips import PresetChipRow

    row = PresetChipRow(["Cell", "Film", "USAF"])
    assert row.active() == "Cell"  # first chip auto-selected

    got = []
    row.selected.connect(lambda name: got.append(name))

    row._buttons[1].click()
    assert got == ["Film"]
    assert row.active() == "Film"


def test_preset_chip_row_rebuild_preserves_active(qapp):
    from ui3.widgets.preset_chips import PresetChipRow

    row = PresetChipRow(["Cell", "Film"])
    row.set_active("Film")
    assert row.active() == "Film"

    row.rebuild(["Cell", "Film", "USAF", "Lab1"])
    assert row.active() == "Film"
    assert row.labels() == ["Cell", "Film", "USAF", "Lab1"]
    assert row._buttons[row.labels().index("Film")].isChecked()


def test_preset_chip_row_rebuild_drops_stale_active(qapp):
    from ui3.widgets.preset_chips import PresetChipRow

    row = PresetChipRow(["Cell", "Film"])
    row.set_active("Film")

    row.rebuild(["Cell", "USAF"])  # "Film" no longer present
    assert row.active() == "Cell"  # falls back to first


def test_preset_chip_row_rebuild_empty_clears_active(qapp):
    from ui3.widgets.preset_chips import PresetChipRow

    row = PresetChipRow(["Cell"])
    row.rebuild([])
    assert row.active() is None
    assert row.labels() == []


# ---------------------------------------------------------------------------
# LineProfileTool / LineProfileDialog
# ---------------------------------------------------------------------------

def _synthetic_ramp(n=64):
    """A simple horizontal ramp — sampling along a row gives a known,
    monotonically increasing sequence we can assert against."""
    x = np.arange(n, dtype=np.float64)
    return np.tile(x, (n, 1))  # every row is [0, 1, ..., n-1]


def test_line_profile_tool_samples_horizontal_line(qapp):
    from ui3.viewport import ImagePanel
    from ui3.widgets.line_profile import LineProfileTool
    from ui3.design import DARK

    panel = ImagePanel("test", DARK)
    data = _synthetic_ramp(64)
    panel.set_image(data)

    tool = LineProfileTool(panel)
    # x1=62, not 63 — the bilinear sampler needs x+1 in-bounds (core.line_
    # profile._bilinear requires x1 < nx strictly), so the last column of a
    # width-64 array is intentionally out-of-bounds (-> NaN) by design.
    tool.add_line(y0=10, x0=0, y1=10, x1=62, label="row10")
    assert tool.count() == 1

    sampled = tool.sample(0)
    assert sampled is not None
    assert sampled.profile.label == "row10"
    # Sampling row 10 along x=0..62 should reproduce the ramp closely.
    assert sampled.values[0] == pytest.approx(0.0, abs=1e-6)
    assert sampled.values[-1] == pytest.approx(62.0, abs=1e-6)
    assert np.all(np.diff(sampled.values) >= -1e-9)  # monotone non-decreasing


def test_line_profile_tool_multiple_lines_distinct_colours(qapp):
    from ui3.viewport import ImagePanel
    from ui3.widgets.line_profile import LineProfileTool, _DEFAULT_COLOURS
    from ui3.design import DARK

    panel = ImagePanel("test", DARK)
    panel.set_image(_synthetic_ramp(32))

    tool = LineProfileTool(panel)
    tool.add_line(y0=0, x0=0, y1=0, x1=31, label="a")
    tool.add_line(y0=5, x0=0, y1=5, x1=31, label="b")
    assert tool.count() == 2

    all_sampled = tool.sample_all()
    assert len(all_sampled) == 2
    assert all_sampled[0].profile.colour_rgb == _DEFAULT_COLOURS[0]
    assert all_sampled[1].profile.colour_rgb == _DEFAULT_COLOURS[1]


def test_line_profile_tool_clear_removes_all(qapp):
    from ui3.viewport import ImagePanel
    from ui3.widgets.line_profile import LineProfileTool
    from ui3.design import DARK

    panel = ImagePanel("test", DARK)
    panel.set_image(_synthetic_ramp(32))

    tool = LineProfileTool(panel)
    tool.add_line(y0=0, x0=0, y1=0, x1=31)
    tool.add_line(y0=1, x0=0, y1=1, x1=31)
    assert tool.count() == 2

    tool.clear()
    assert tool.count() == 0
    assert tool.sample_all() == []


def test_line_profile_tool_no_image_returns_none(qapp):
    from ui3.viewport import ImagePanel
    from ui3.widgets.line_profile import LineProfileTool
    from ui3.design import DARK

    panel = ImagePanel("test", DARK)  # never set_image'd
    tool = LineProfileTool(panel)
    tool.add_line(y0=0, x0=0, y1=0, x1=10)

    assert tool.sample(0) is None
    assert tool.sample_all() == []


def test_line_profile_dialog_refreshes_stats(qapp):
    from ui3.viewport import ImagePanel
    from ui3.widgets.line_profile import LineProfileTool, LineProfileDialog
    from ui3.design import DARK

    panel = ImagePanel("test", DARK)
    panel.set_image(_synthetic_ramp(32))

    tool = LineProfileTool(panel)
    tool.add_line(y0=0, x0=0, y1=0, x1=31, label="row0")

    ctx, bridge, params, events = _make_ctx()
    dlg = LineProfileDialog(tool, ctx)

    assert dlg.stats_list.count() == 1
    text = dlg.stats_list.item(0).text()
    assert "row0" in text
    assert "min=" in text and "max=" in text and "mean=" in text
    assert any("Line profile" in t for t, _ in events["status"])

    bridge.shutdown()


def test_line_profile_dialog_clear_lines_button(qapp):
    from ui3.viewport import ImagePanel
    from ui3.widgets.line_profile import LineProfileTool, LineProfileDialog
    from ui3.design import DARK

    panel = ImagePanel("test", DARK)
    panel.set_image(_synthetic_ramp(32))

    tool = LineProfileTool(panel)
    tool.add_line(y0=0, x0=0, y1=0, x1=31)

    ctx, bridge, params, events = _make_ctx()
    dlg = LineProfileDialog(tool, ctx)
    assert dlg.stats_list.count() == 1

    dlg._on_clear()
    assert tool.count() == 0
    assert dlg.stats_list.count() == 0

    bridge.shutdown()
