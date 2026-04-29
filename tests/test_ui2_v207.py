"""Tests for v2.0.7 work:

* Optical mode reflection halves the OPD (2× height bug fix).
* Amplitude percentile auto-contrast clips outliers.
* ROI-gated autofocus only sees masked pixels.
* Migration v7 → v8 backfills the new fields.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Dear PyGui stub — shared pattern with other ui2 tests
# ---------------------------------------------------------------------------

def _install_dpg_stub():
    if "dearpygui" in sys.modules:
        return
    dpg = types.ModuleType("dearpygui.dearpygui")
    for k in ["mvKey_O","mvKey_R","mvKey_K","mvKey_Slash","mvKey_Escape",
              "mvKey_LShift","mvKey_RShift","mvKey_LControl","mvKey_RControl",
              "mvKey_LSuper","mvKey_RSuper","mvKey_0","mvKey_1","mvKey_2",
              "mvKey_3","mvKey_4","mvXAxis","mvYAxis","mvAll",
              "mvThemeCol_WindowBg","mvMouseButton_Left"]:
        setattr(dpg, k, 0)
    for name in [
        "create_context","create_viewport","setup_dearpygui","show_viewport",
        "set_primary_window","is_dearpygui_running","render_dearpygui_frame",
        "destroy_context","stop_dearpygui","add_dynamic_texture","set_value",
        "get_value","configure_item","delete_item","add_menu_item","add_text",
        "add_button","add_combo","add_input_text","add_input_float",
        "add_input_int","add_checkbox","add_spacer","add_separator",
        "add_plot_axis","add_image_series","add_file_extension","bind_theme",
        "add_theme_color","add_theme_style","focus_item","show_item",
        "fit_axis_data","get_item_parent","get_item_label",
        "get_viewport_client_width","get_viewport_client_height",
        "set_viewport_drop_callback","set_viewport_resize_callback",
        "add_key_press_handler","add_item_clicked_handler",
        "bind_item_handler_registry","is_key_down","add_line_series",
        "add_mouse_click_handler","add_slider_float",
    ]:
        setattr(dpg, name, MagicMock(return_value=None))
    dpg.does_item_exist = MagicMock(return_value=False)

    class _CM:
        def __enter__(self): return 0
        def __exit__(self, *a): return False
    dummy_cm = MagicMock(return_value=_CM())
    for name in ["theme","theme_component","texture_registry","window",
                 "child_window","group","menu","file_dialog",
                 "handler_registry","item_handler_registry",
                 "viewport_menu_bar","plot","tooltip","plot_axis",
                 "collapsing_header","menu_bar"]:
        setattr(dpg, name, dummy_cm)
    dpg.last_item = MagicMock(return_value="last")

    parent = types.ModuleType("dearpygui")
    parent.dearpygui = dpg
    sys.modules["dearpygui"] = parent
    sys.modules["dearpygui.dearpygui"] = dpg


_install_dpg_stub()

from core.settings_schema import SCHEMA_VERSION, AppSettings  # noqa: E402
from ui2 import state_store  # noqa: E402
from ui2.app import DhmApp, _to_rgba  # noqa: E402
from ui2.reconstruction import ReconParams  # noqa: E402


# ---------------------------------------------------------------------------
# ReconParams defaults for v2.0.7 fields
# ---------------------------------------------------------------------------

def test_optical_mode_defaults_to_transmission():
    p = ReconParams()
    assert p.optical_mode == "transmission"
    assert p.auto_contrast_amplitude is True
    assert p.af_roi is None


# ---------------------------------------------------------------------------
# Migration v7 → v8
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "ui2_state.json"


def test_migration_v7_payload_backfills_optical_mode(tmp_state):
    tmp_state.write_text(json.dumps({
        "schema_version": 7,
        "ui2": {"theme": "dark", "sample_id": "legacy-v7"},
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.optical_mode == "transmission"
    assert loaded.ui2.auto_contrast_amplitude is True


def test_roundtrip_preserves_optical_mode(tmp_state):
    s = AppSettings.defaults().with_ui2(
        optical_mode="reflection", auto_contrast_amplitude=False,
    )
    state_store.save(s, tmp_state)
    loaded = state_store.load(tmp_state)
    assert loaded.ui2.optical_mode == "reflection"
    assert loaded.ui2.auto_contrast_amplitude is False


# ---------------------------------------------------------------------------
# Reflection mode halves OPD — the 2× height bug regression guard
# ---------------------------------------------------------------------------

def test_reflection_mode_halves_unwrapped_phase_before_qpi():
    """Run the QPI worker with optical_mode='reflection' and assert
    the unwrapped phase fed into compute_qpi is exactly the
    transmission-mode value divided by 2 — that's what makes
    a 3-µm-tall feature report 3 µm instead of 6 µm."""
    from ui2.workers import ScienceDriver
    import time

    fake_loaded = MagicMock()
    fake_loaded.array = np.zeros((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    captured_unwrapped: dict = {}

    fake_qpi = MagicMock()
    fake_qpi.phase_stats = MagicMock(range_nm=100.0)
    fake_qpi.total_dry_mass_pg = None
    fake_qpi.step_height_m = None

    synthetic_unwrapped = np.full((16, 16), 6.0, dtype=np.float32)  # 6 rad

    def fake_compute_qpi(unwrapped, **kwargs):
        captured_unwrapped["arr"] = np.asarray(unwrapped, dtype=np.float32)
        return fake_qpi

    params = ReconParams(optical_mode="reflection")
    driver = ScienceDriver()
    done = [False]

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.zeros((16, 16), dtype=np.complex64),
                             (8, 8))), \
         patch("ui2.workers.propagate",
               return_value=np.zeros((16, 16), dtype=np.complex64)), \
         patch("ui2.workers.unwrap_phase_advanced",
               return_value=synthetic_unwrapped), \
         patch("ui2.workers.compute_qpi", side_effect=fake_compute_qpi):
        driver.run_qpi(
            Path("/dev/null"), params, z_mm=1.0,
            on_result=lambda r: done.__setitem__(0, True),
            on_error=lambda e: done.__setitem__(0, True),
        )
        deadline = time.monotonic() + 3.0
        while not done[0] and time.monotonic() < deadline:
            time.sleep(0.01)
    driver.shutdown()

    assert "arr" in captured_unwrapped
    # Reflection mode must halve the phase before compute_qpi.
    np.testing.assert_allclose(captured_unwrapped["arr"], 3.0, rtol=1e-6)


def test_transmission_mode_passes_unwrapped_phase_untouched():
    """Transmission mode (default) must not rescale — regression
    guard so we don't accidentally halve in both modes."""
    from ui2.workers import ScienceDriver
    import time

    fake_loaded = MagicMock()
    fake_loaded.array = np.zeros((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    captured = {}

    def fake_compute_qpi(unwrapped, **kwargs):
        captured["arr"] = np.asarray(unwrapped, dtype=np.float32).copy()
        qpi = MagicMock()
        qpi.phase_stats = MagicMock(range_nm=None)
        qpi.total_dry_mass_pg = None
        qpi.step_height_m = None
        return qpi

    unwrapped = np.full((8, 8), 5.0, dtype=np.float32)
    params = ReconParams(optical_mode="transmission")
    driver = ScienceDriver()
    done = [False]

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.zeros((8, 8), dtype=np.complex64), (4, 4))), \
         patch("ui2.workers.propagate",
               return_value=np.zeros((8, 8), dtype=np.complex64)), \
         patch("ui2.workers.unwrap_phase_advanced",
               return_value=unwrapped), \
         patch("ui2.workers.compute_qpi", side_effect=fake_compute_qpi):
        driver.run_qpi(
            Path("/dev/null"), params, z_mm=1.0,
            on_result=lambda r: done.__setitem__(0, True),
            on_error=lambda e: done.__setitem__(0, True),
        )
        deadline = time.monotonic() + 3.0
        while not done[0] and time.monotonic() < deadline:
            time.sleep(0.01)
    driver.shutdown()

    np.testing.assert_allclose(captured["arr"], 5.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# Amplitude percentile contrast
# ---------------------------------------------------------------------------

def test_to_rgba_percentile_clips_outliers():
    """99th-percentile clip must squash outliers that otherwise
    flatten the rest of the histogram. Use a real distribution
    (uniform bulk + one huge spike) instead of a constant+spike
    so the 1-99% range is non-degenerate."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
    arr[0, 0] = 1000.0   # single outlier
    rgba_minmax = _to_rgba(arr, colormap="gray", contrast="minmax")
    rgba_pct = _to_rgba(arr, colormap="gray", contrast="percentile")
    # min/max mode normalises by the outlier → bulk ends up near 0.
    # Percentile clip ignores the outlier → bulk spans 0..1 with
    # mean near the distribution's mean (~0.5).
    assert rgba_minmax[..., 0].mean() < 0.05
    assert 0.3 < rgba_pct[..., 0].mean() < 0.7


def test_to_rgba_percentile_constant_array_stays_zero():
    """Flat input yields zero regardless of contrast mode."""
    arr = np.full((16, 16), 0.3, dtype=np.float32)
    rgba = _to_rgba(arr, colormap="gray", contrast="percentile")
    assert rgba[..., 0].min() == 0.0
    assert rgba[..., 0].max() == 0.0


def test_to_rgba_percentile_clipping_keeps_range_in_0_1():
    arr = np.linspace(-10, 1000, 64 * 64,
                      dtype=np.float32).reshape(64, 64)
    rgba = _to_rgba(arr, colormap="gray", contrast="percentile")
    assert rgba.min() >= 0.0
    assert rgba.max() <= 1.0


# ---------------------------------------------------------------------------
# ROI masking feeds the scan only the selected pixels
# ---------------------------------------------------------------------------

def test_prepare_field_applies_roi_mask():
    from ui2 import workers

    fake_loaded = MagicMock()
    fake_loaded.array = np.ones((32, 32), dtype=np.float32)
    fake_loaded.metadata = {}

    full_field = np.full((32, 32), 1.0 + 1.0j, dtype=np.complex64)

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(full_field, (16, 16))):
        # Without ROI — every pixel is kept.
        params = ReconParams()
        field_full, _, _ = workers._prepare_field(
            Path("/dev/null"), params, apply_af_roi=True,
        )
        assert np.count_nonzero(field_full) == 32 * 32

        # With ROI covering upper-left quarter — 16x16 = 256 non-zeros.
        params_roi = ReconParams(af_roi=(0.0, 0.0, 0.5, 0.5))
        field_roi, _, _ = workers._prepare_field(
            Path("/dev/null"), params_roi, apply_af_roi=True,
        )
        # ROI of 0..0.5 normalised on a 32-wide field hits indices 0..15
        # via ``int(0.5 * 31) = 15`` and the inclusive slice
        # ``[0:16, 0:16]`` → exactly 256 live pixels.
        assert np.count_nonzero(field_roi) == 16 * 16
        # And the live region is inside the top-left quadrant.
        assert field_roi[0, 0] != 0
        assert field_roi[20, 20] == 0


def test_prepare_field_skips_roi_when_flag_false():
    """Reconstruction (``apply_af_roi=False``) must never mask the
    field — the user's on-screen preview should always show the
    full frame even with a scan-only ROI set."""
    from ui2 import workers

    fake_loaded = MagicMock()
    fake_loaded.array = np.ones((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.full((16, 16), 1.0 + 1.0j,
                                     dtype=np.complex64), (8, 8))):
        params = ReconParams(af_roi=(0.0, 0.0, 0.1, 0.1))
        field, _, _ = workers._prepare_field(
            Path("/dev/null"), params, apply_af_roi=False,
        )
    assert np.count_nonzero(field) == 16 * 16


def test_prepare_field_ignores_degenerate_roi():
    """Zero-area ROI must not crash and must not zero the field."""
    from ui2 import workers

    fake_loaded = MagicMock()
    fake_loaded.array = np.ones((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.full((16, 16), 1.0 + 0.5j,
                                     dtype=np.complex64), (8, 8))):
        params = ReconParams(af_roi=(0.5, 0.5, 0.5, 0.5))  # zero-area
        field, _, _ = workers._prepare_field(
            Path("/dev/null"), params, apply_af_roi=True,
        )
    assert np.count_nonzero(field) == 16 * 16


# ---------------------------------------------------------------------------
# Info-text composer surfaces the new fields
# ---------------------------------------------------------------------------

def _fake_app():
    app = DhmApp.__new__(DhmApp)
    app._last_recon = None
    app._last_qpi = None
    app._last_qpi_batch = None
    app._last_depth = None
    app._current_hologram = Path("/tmp/test.tif")
    app._sample_id = ""
    app._params = ReconParams()
    return app


def test_info_text_shows_optical_mode():
    app = _fake_app()
    app._params = ReconParams(optical_mode="reflection")
    txt = app._compose_info_text()
    assert "Optical:" in txt
    assert "reflection" in txt


def test_info_text_shows_roi_when_set():
    app = _fake_app()
    app._params = ReconParams(af_roi=(0.1, 0.2, 0.8, 0.9))
    txt = app._compose_info_text()
    assert "AF ROI:" in txt
    assert "0.10" in txt and "0.90" in txt


def test_info_text_omits_roi_when_unset():
    app = _fake_app()
    assert app._params.af_roi is None
    assert "AF ROI:" not in app._compose_info_text()


# ---------------------------------------------------------------------------
# v2.0.9 evening regressions — 5-bug sprint fallout
# ---------------------------------------------------------------------------

def test_load_hologram_defines_flip_before_push():
    """Regression guard for the 2026-04-24 evening Edit that
    assigned ``raw_disp`` AFTER the ``_push_texture(..., raw_disp,
    ...)`` call — raised ``NameError`` and the hologram never
    appeared (blanket ``except`` turned it into a vague "Preview
    failed" toast). Inspects source to pin the order."""
    import inspect
    src = inspect.getsource(DhmApp._load_hologram)
    push_idx = src.find("self._push_texture(")
    flip_idx = src.find("_flip_display_v")
    assert push_idx > 0, "push_texture call disappeared from _load_hologram"
    assert flip_idx > 0, "flip branch disappeared from _load_hologram"
    assert flip_idx < push_idx, (
        "_flip_display_v must be consumed (raw_disp computed) BEFORE "
        "the _push_texture call — order regression"
    )


def test_autofocus_calls_fft2_once_per_scan():
    """Pin the optimisation: ``_make_fast_evaluator`` must compute
    ``fft2(field)`` **once** and reuse the spectrum for every
    ``evaluate(z)`` call. If someone moves the FFT inside the
    closure we'd silently pay N× the cost on a scan — catch it
    synthetically by counting ``fft2`` invocations."""
    from core.autofocus.evaluator import _make_fast_evaluator
    from core.autofocus import FocusMetric
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    class _CountingFFT:
        """Wraps the real backend and counts fft2 calls."""
        def __init__(self, real):
            self._real = real
            self.name = real.name
            self.fft2_calls = 0
            self.ifft2_calls = 0

        def fft2(self, *args, **kwargs):
            self.fft2_calls += 1
            return self._real.fft2(*args, **kwargs)

        def ifft2(self, *args, **kwargs):
            self.ifft2_calls += 1
            return self._real.ifft2(*args, **kwargs)

    from core.fft_backend import get_best_fft_backend
    real = get_best_fft_backend()
    counting = _CountingFFT(real)

    size = 64
    field = np.random.default_rng(0).standard_normal(
        (size, size),
    ).astype(np.complex64)
    base = ReconstructionParams(
        wavelength_m=632.8e-9, pixel_size_m=2.5e-6, z_m=0.0, n=1.0,
    )

    with patch("core.autofocus.evaluator.get_best_fft_backend",
               return_value=counting):
        evaluator = _make_fast_evaluator(
            field, base, ReconstructionMethod.ASM,
            FocusMetric.ENTROPY,
        )
        # 40 calls — should still be exactly 1 fft2.
        for z in np.linspace(5e-3, 15e-3, 40):
            evaluator(float(z))

    assert counting.fft2_calls == 1, (
        f"fft2 was called {counting.fft2_calls} times for 40 z "
        f"evaluations — the pre-computed spectrum optimisation "
        f"is gone (expected 1 call, reused N times)"
    )


def test_load_hologram_runs_without_nameerror(tmp_path):
    """End-to-end-ish drive of ``_load_hologram``. If ``raw_disp``
    is referenced before definition we'd hit NameError here,
    surface as a ``Preview failed`` status — assert neither
    happened."""
    import tifffile
    arr = np.random.default_rng(0).standard_normal(
        (64, 64),
    ).astype(np.float32)
    p = tmp_path / "hol.tif"
    tifffile.imwrite(p, arr)

    app = DhmApp.__new__(DhmApp)
    app._flip_display_v = False
    app._flip_display_h = False
    app._current_hologram = None
    app._last_dir = ""
    app._recent = []
    app._last_recon = None
    app._last_qpi = None
    app._last_qpi_batch = None
    app._last_depth = None
    app._sample_id = ""
    app._params = ReconParams()
    app._toasts = MagicMock()
    panel = MagicMock()
    panel.tex_tag = "tex_stub"
    app.panel_input = panel
    app.panel_amp = panel
    app.panel_phase = panel
    app._push_texture = MagicMock()
    app._set_status = MagicMock()
    app._refresh_reconstruct_tooltip = MagicMock()
    app._remember_recent = MagicMock()
    app._apply_detected_metadata = MagicMock()
    app._refresh_info_text = MagicMock()
    app._mark_dirty = MagicMock()

    app._load_hologram(p)
    for call in app._set_status.call_args_list:
        msg = call.args[0] if call.args else ""
        assert "Preview failed" not in msg, (
            f"_load_hologram produced a Preview failed toast: {msg}"
        )
    assert app._push_texture.called, (
        "_push_texture was never called — hologram wouldn't appear"
    )
