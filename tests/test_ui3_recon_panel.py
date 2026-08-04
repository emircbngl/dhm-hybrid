"""ui3 ReconPanel tests — offscreen Qt, real WorkerBridge (ui2 drivers),
synthetic hologram. Mirrors the ``_synthetic_hologram`` pattern from
``tests/test_ui3_spine.py`` / ``tests/test_ui3_qpi_panel.py``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DHM_USER", "ci")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _synthetic_hologram(tmp_path, n=192):
    import tifffile
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.5 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * 0.28 * X + bump))
    holo = np.abs(1 + obj) ** 2
    p = tmp_path / "holo.tif"
    tifffile.imwrite(p, (holo * 1000).astype(np.uint16))
    return p


@pytest.fixture(scope="module")
def qapp():
    from ui3.app import build_app, apply_theme
    app = build_app([])
    apply_theme(app, "dark")
    return app


def _make_ctx(tmp_path, holo_path):
    """A minimal-but-real PanelContext: a live WorkerBridge, a mutable
    ReconParams, and no-op display/feedback hooks (recorded for assertions)."""
    from ui2.reconstruction import ReconParams
    from ui3.bridge import WorkerBridge
    from ui3.context import PanelContext
    from ui3.design import DARK

    bridge = WorkerBridge()
    params = ReconParams(
        wavelength_nm=632.8, pixel_um=1.0, z_mm=0.0, mask_radius=40,
        n_sample=1.38, n_medium=1.337,
        af_z_min_mm=-0.5, af_z_max_mm=0.5, af_n_steps=12,
    )
    events: dict = {"status": [], "toast": []}

    ctx = PanelContext(
        bridge=bridge,
        get_params=lambda: params,
        params_changed=lambda: None,
        hologram_path=lambda: holo_path,
        last_recon=lambda: None,
        last_depth=lambda: None,
        get_field=lambda target: None,
        set_status=lambda text, level="info": events["status"].append((text, level)),
        toast=lambda text, level="info": events["toast"].append((text, level)),
        show_in_panel=lambda target, array, **kw: None,
        palette=DARK,
    )
    return ctx, bridge, params, events


def _pump_until(qapp, predicate, deadline_s=20):
    deadline = time.time() + deadline_s
    while not predicate() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)


def test_recon_panel_builds_and_hydrates(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    holo = _synthetic_hologram(tmp_path)
    ctx, bridge, params, events = _make_ctx(tmp_path, holo)
    panel = ReconPanel(ctx)

    assert panel.sp_wavelength.value() == pytest.approx(632.8)
    assert panel.sp_pixel.value() == pytest.approx(1.0)
    assert panel.sp_z.value() == pytest.approx(0.0)
    assert panel.sp_mask.value() == 40
    assert panel.cmb_method.currentText() == "ASM"
    assert panel.lbl_file.text() == "holo.tif"
    # Reference-free sub-section hidden by default (mode == off).
    # isVisibleTo(panel), not isVisible() — the panel is never shown/
    # top-level in this offscreen test, so isVisible() would always
    # read False regardless of setVisible().
    assert not panel.grp_reffree.isVisibleTo(panel)

    bridge.shutdown()


def test_basic_controls_write_into_live_params(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    panel.sp_wavelength.setValue(532.0)
    assert params.wavelength_nm == pytest.approx(532.0)

    panel.sp_pixel.setValue(3.45)
    assert params.pixel_um == pytest.approx(3.45)

    panel.sp_magnification.setValue(40.0)
    assert params.magnification == pytest.approx(40.0)

    panel.cb_pixel_is_effective.setChecked(False)
    assert params.pixel_is_effective is False

    panel.sp_z.setValue(12.5)
    assert params.z_mm == pytest.approx(12.5)

    panel.sp_mask.setValue(80)
    assert params.mask_radius == 80

    panel.cmb_method.setCurrentText("Fresnel")
    assert params.method == "Fresnel"

    bridge.shutdown()


def test_advanced_controls_write_into_live_params(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    panel.cmb_optical_mode.setCurrentText("reflection")
    assert params.optical_mode == "reflection"

    panel.sp_n_sample.setValue(1.40)
    assert params.n_sample == pytest.approx(1.40)

    panel.sp_n_medium.setValue(1.000)
    assert params.n_medium == pytest.approx(1.000)

    panel.cmb_af_metric.setCurrentIndex(0)
    assert params.autofocus_metric == panel.cmb_af_metric.currentText()

    panel.sp_af_z_min.setValue(-5.0)
    assert params.af_z_min_mm == pytest.approx(-5.0)

    panel.sp_af_z_max.setValue(5.0)
    assert params.af_z_max_mm == pytest.approx(5.0)

    panel.sp_af_n_steps.setValue(100)
    assert params.af_n_steps == 100

    panel.cb_subtract_mean.setChecked(False)
    assert params.subtract_mean is False

    panel.cb_hann_window.setChecked(True)
    assert params.hann_window is True

    panel.cmb_fft_backend.setCurrentText("numpy")
    assert params.fft_backend == "numpy"

    panel.cmb_unwrap_method.setCurrentText("TIE")
    assert params.unwrap_method == "TIE"

    panel.cb_auto_contrast_amp.setChecked(False)
    assert params.auto_contrast_amplitude is False

    bridge.shutdown()


def test_reference_mode_off_disarms_legacy_flag(qapp, tmp_path):
    """2026-07-05 review fix parity: explicit Off must disarm
    subtract_reference so effective_reference_mode() can't silently
    re-resolve back to 'reference'."""
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    # Move off "Off" first so the subsequent "Off" pick is an actual
    # combo-box transition (setCurrentText on the already-current value
    # doesn't emit currentTextChanged).
    panel.cmb_reference_mode.setCurrentText("Reference")
    params.subtract_reference = True
    panel.cmb_reference_mode.setCurrentText("Off")
    assert params.reference_mode == "off"
    assert params.subtract_reference is False
    assert params.effective_reference_mode() == "off"

    bridge.shutdown()


def test_reference_mode_reference_free_shows_subsection(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    panel.cmb_reference_mode.setCurrentText("Reference-free")
    assert params.reference_mode == "reference_free"
    assert panel.grp_reffree.isVisibleTo(panel)

    panel.cmb_reffree_bg_method.setCurrentText("polynomial")
    assert params.reffree_bg_method == "polynomial"

    panel.sp_reffree_bg_order.setValue(3)
    assert params.reffree_bg_order == 3

    panel.sp_reffree_n_terms.setValue(20)
    assert params.reffree_n_terms == 20

    bridge.shutdown()


def test_reference_mode_selected_without_file_warns(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    panel.cmb_reference_mode.setCurrentText("Reference")
    assert params.reference_mode == "reference"
    assert any("no reference is loaded" in text for text, _ in events["status"])

    bridge.shutdown()


def test_clear_reference_falls_back_to_off(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    params.reference_path = Path("/tmp/ref.tif")
    params.reference_mode = "reference"
    params.subtract_reference = True

    panel._on_clear_reference_clicked()

    assert params.reference_path is None
    assert params.subtract_reference is False
    assert params.reference_mode == "off"
    assert panel.lbl_reference_path.text() == "(none)"

    bridge.shutdown()


def test_clear_reference_preserves_reference_free(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    params.reference_path = Path("/tmp/ref.tif")
    params.reference_mode = "reference_free"
    params.subtract_reference = False

    panel._on_clear_reference_clicked()

    assert params.reference_path is None
    assert params.reference_mode == "reference_free"

    bridge.shutdown()


def test_presets_apply_builtin_values(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    # "Cell" is the combo's first/default entry, so move to "USAF" first —
    # otherwise setCurrentText("Cell") is a no-op (already current) and
    # currentTextChanged never fires.
    panel.cmb_preset.setCurrentText("USAF")
    panel.cmb_preset.setCurrentText("Cell")
    assert params.wavelength_nm == pytest.approx(632.8)
    assert params.pixel_um == pytest.approx(3.45)
    assert params.magnification == pytest.approx(40.0)
    assert params.pixel_is_effective is False
    assert params.method == "ASM"

    panel.cmb_preset.setCurrentText("Film")
    assert params.method == "Fresnel"
    assert params.magnification == pytest.approx(1.0)

    bridge.shutdown()


def test_save_and_delete_user_preset_roundtrip(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    params.wavelength_nm = 700.0
    panel._user_presets["Lab1"] = dict(
        wavelength_nm=700.0, pixel_um=1.0, z_mm=0.0, mask_radius=40,
        method="ASM", magnification=1.0, pixel_is_effective=True,
        n_sample=1.38, n_medium=1.337, autofocus_metric="LAPLACIAN_VARIANCE",
    )
    panel._reload_preset_combo()
    assert "Lab1" in [panel.cmb_preset.itemText(i) for i in range(panel.cmb_preset.count())]

    panel._user_presets.pop("Lab1")
    panel._reload_preset_combo()
    assert "Lab1" not in [panel.cmb_preset.itemText(i) for i in range(panel.cmb_preset.count())]

    bridge.shutdown()


def test_reconstruct_without_hologram_warns(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    ctx, bridge, params, events = _make_ctx(tmp_path, None)
    panel = ReconPanel(ctx)

    panel._on_reconstruct_clicked()
    panel._on_autofocus_clicked()

    assert any("Load a hologram" in text for text, _ in events["status"])
    bridge.shutdown()


def test_reconstruct_end_to_end(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    holo = _synthetic_hologram(tmp_path)
    ctx, bridge, params, events = _make_ctx(tmp_path, holo)
    panel = ReconPanel(ctx)

    got = {}
    bridge.recon_done.connect(lambda r: got.update(done=r))
    bridge.recon_error.connect(lambda m: got.update(err=m))

    panel._on_reconstruct_clicked()
    _pump_until(qapp, lambda: "done" in got or "err" in got)

    assert "done" in got, f"reconstruction failed: {got.get('err')}"
    assert any(level == "ok" for _, level in events["toast"])

    bridge.shutdown()


def test_autofocus_end_to_end_updates_z(qapp, tmp_path):
    from ui3.panels.recon_panel import ReconPanel

    holo = _synthetic_hologram(tmp_path)
    ctx, bridge, params, events = _make_ctx(tmp_path, holo)
    panel = ReconPanel(ctx)

    got = {}
    bridge.autofocus_done.connect(lambda r: got.update(done=r))
    bridge.autofocus_error.connect(lambda m: got.update(err=m))

    panel._on_autofocus_clicked()
    _pump_until(qapp, lambda: "done" in got or "err" in got, deadline_s=30)

    assert "done" in got, f"autofocus failed: {got.get('err')}"
    expected_z_mm = float(got["done"].best_z_m) * 1e3
    assert params.z_mm == pytest.approx(expected_z_mm)
    # sp_z has 4 decimal places — compare at that display precision
    # rather than raw float, since the spinbox necessarily rounds.
    assert panel.sp_z.value() == pytest.approx(expected_z_mm, abs=1e-4)

    bridge.shutdown()
