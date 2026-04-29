"""Tests for the scientific parameters the v2 port dropped.

CEO caught magnification missing in pilot test — these tests pin the
fix so we don't regress:

* :class:`ReconParams` carries M, pixel_is_effective, n_sample,
  n_medium, autofocus_metric.
* ``effective_pixel_um()`` divides only when appropriate.
* Presets ship sane magnifications (Cell @ 40×, USAF @ 10×, Film @ 1×).
* TIFF metadata auto-detect populates the sidebar.
* State roundtrip + v3→v4 migration preserves the new fields.
* Info-text composer shows the pixel arithmetic.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Stub Dear PyGui first — shared with test_ui2_logic but kept inline
# so each file can run in isolation.
# ---------------------------------------------------------------------------

def _install_dpg_stub():
    if "dearpygui" in sys.modules:
        return
    dpg = types.ModuleType("dearpygui.dearpygui")
    for k in ["mvKey_O","mvKey_R","mvKey_K","mvKey_Slash","mvKey_Escape",
              "mvKey_LShift","mvKey_RShift","mvKey_LControl","mvKey_RControl",
              "mvKey_LSuper","mvKey_RSuper","mvXAxis","mvYAxis","mvAll",
              "mvThemeCol_WindowBg"]:
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
        "bind_item_handler_registry","is_key_down",
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
                 "viewport_menu_bar","plot","tooltip"]:
        setattr(dpg, name, dummy_cm)
    dpg.last_item = MagicMock(return_value="last")

    parent = types.ModuleType("dearpygui")
    parent.dearpygui = dpg
    sys.modules["dearpygui"] = parent
    sys.modules["dearpygui.dearpygui"] = dpg


_install_dpg_stub()

from core.settings_schema import SCHEMA_VERSION, AppSettings  # noqa: E402
from ui2 import state_store  # noqa: E402
from ui2.app import DhmApp  # noqa: E402
from ui2.reconstruction import ReconParams  # noqa: E402


# ---------------------------------------------------------------------------
# ReconParams field surface
# ---------------------------------------------------------------------------

def test_recon_params_defaults_match_v1():
    """Defaults should mirror v1 so un-touched state behaves identically."""
    p = ReconParams()
    assert p.magnification == 1.0
    assert p.pixel_is_effective is True
    assert p.n_sample == pytest.approx(1.38)
    assert p.n_medium == pytest.approx(1.337)
    assert p.autofocus_metric == "LAPLACIAN_VARIANCE"


@pytest.mark.parametrize("px,mag,effective,expected", [
    (3.45, 40.0, False, 3.45 / 40.0),     # camera pixel on 40× objective
    (3.45, 40.0, True,  3.45),            # user pre-divided
    (5.5,  1.0,  True,  5.5),             # macroscopic imaging, no objective
    (5.5,  1.0,  False, 5.5),             # M=1 and flag False → no-op divide
    (2.2,  10.0, False, 0.22),            # USAF at 10×
])
def test_effective_pixel_um(px, mag, effective, expected):
    p = ReconParams(pixel_um=px, magnification=mag,
                    pixel_is_effective=effective)
    assert p.effective_pixel_um() == pytest.approx(expected, rel=1e-6)


def test_effective_pixel_falls_back_when_magnification_is_zero():
    """Guard: don't ever divide by zero, even if the user types nonsense."""
    p = ReconParams(pixel_um=3.45, magnification=0.0,
                    pixel_is_effective=False)
    assert p.effective_pixel_um() == pytest.approx(3.45)


# ---------------------------------------------------------------------------
# Presets ship meaningful magnifications
# ---------------------------------------------------------------------------

def test_cell_preset_has_microscope_magnification():
    cfg = DhmApp._builtin_presets()["Cell"]
    assert cfg["magnification"] == 40.0
    assert cfg["pixel_is_effective"] is False
    assert cfg["n_sample"] == pytest.approx(1.38)
    assert cfg["n_medium"] == pytest.approx(1.337)


def test_film_preset_has_no_objective():
    cfg = DhmApp._builtin_presets()["Film"]
    assert cfg["magnification"] == 1.0
    assert cfg["pixel_is_effective"] is True


def test_usaf_preset_has_10x():
    cfg = DhmApp._builtin_presets()["USAF"]
    assert cfg["magnification"] == 10.0
    assert cfg["pixel_is_effective"] is False
    assert cfg["autofocus_metric"] == "TENENGRAD"


def test_all_presets_carry_full_physics_set():
    """Each preset must populate every scientific field so applying
    one never leaves a stale value from a previous preset."""
    expected = {"wavelength_nm", "pixel_um", "z_mm", "mask_radius",
                "method", "magnification", "pixel_is_effective",
                "n_sample", "n_medium", "autofocus_metric"}
    for name, cfg in DhmApp._builtin_presets().items():
        missing = expected - set(cfg.keys())
        assert not missing, f"preset {name} missing: {missing}"


# ---------------------------------------------------------------------------
# State roundtrip + migration v3 → v4
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "ui2_state.json"


def test_state_roundtrip_preserves_magnification(tmp_state):
    s = AppSettings.defaults().with_ui2(
        magnification=60.0, pixel_is_effective=False,
        n_sample=1.42, n_medium=1.515,
        autofocus_metric="TENENGRAD",
    )
    state_store.save(s, tmp_state)
    loaded = state_store.load(tmp_state)
    assert loaded.ui2.magnification == 60.0
    assert loaded.ui2.pixel_is_effective is False
    assert loaded.ui2.n_sample == pytest.approx(1.42)
    assert loaded.ui2.n_medium == pytest.approx(1.515)
    assert loaded.ui2.autofocus_metric == "TENENGRAD"


def test_migration_v3_payload_hydrates_with_v1_defaults(tmp_state):
    """A state file that predates v4 should load cleanly with
    M=1, pixel_is_effective=True (v2.0.2 behaviour)."""
    tmp_state.write_text(json.dumps({
        "schema_version": 3,
        "ui2": {
            "theme": "dark",
            "sample_id": "legacy",
            "pixel_um": 3.45,
            # No magnification, no n_sample, no metric — simulate
            # a state file written before v4.
        },
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.ui2.sample_id == "legacy"
    assert loaded.ui2.pixel_um == pytest.approx(3.45)
    # Backfilled defaults:
    assert loaded.ui2.magnification == 1.0
    assert loaded.ui2.pixel_is_effective is True
    assert loaded.ui2.n_sample == pytest.approx(1.38)
    assert loaded.ui2.autofocus_metric == "LAPLACIAN_VARIANCE"


def test_migration_v1_payload_reaches_v4_with_ui2_defaults(tmp_state):
    """Even a v1 Qt-settings-only dump should reach v4 cleanly."""
    tmp_state.write_text(json.dumps({
        "schema_version": 1,
        "recon": {"wavelength_nm": 532.0},
    }), encoding="utf-8")
    loaded = state_store.load(tmp_state)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.recon.wavelength_nm == pytest.approx(532.0)
    assert loaded.ui2.magnification == 1.0
    assert loaded.ui2.autofocus_metric == "LAPLACIAN_VARIANCE"


# ---------------------------------------------------------------------------
# Info-text composer shows pixel arithmetic
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


def test_info_text_shows_effective_pixel_with_m():
    app = _fake_app()
    app._params = ReconParams(pixel_um=3.45, magnification=40.0,
                              pixel_is_effective=False)
    txt = app._compose_info_text()
    # The arithmetic should be visible — user never has to trust us.
    assert "Effective px:" in txt
    assert "0.086" in txt  # 3.45/40 = 0.08625 µm
    assert "×40" in txt


def test_info_text_shows_no_division_when_pixel_is_effective():
    app = _fake_app()
    app._params = ReconParams(pixel_um=0.086, magnification=1.0,
                              pixel_is_effective=True)
    txt = app._compose_info_text()
    assert "Effective px:" in txt
    # No division symbol because we're not dividing.
    assert " / " not in txt.split("Effective px:")[-1].splitlines()[0]


# ---------------------------------------------------------------------------
# Metric listing stays in lockstep with the enum
# ---------------------------------------------------------------------------

def test_available_focus_metrics_nonempty():
    from ui2.workers import available_focus_metrics
    names = available_focus_metrics()
    assert len(names) >= 5
    assert "LAPLACIAN_VARIANCE" in names
    # All names are uppercase so the sidebar combo's default lookup
    # with the string form never fails on a case mismatch.
    for n in names:
        assert n == n.upper()


def test_metric_name_round_trips_to_enum():
    from core.autofocus import FocusMetric
    from ui2.workers import _metric_from_params
    p = ReconParams(autofocus_metric="TENENGRAD")
    assert _metric_from_params(p) == FocusMetric.TENENGRAD


def test_unknown_metric_falls_back_to_default():
    from core.autofocus import FocusMetric
    from ui2.workers import _metric_from_params
    p = ReconParams(autofocus_metric="NOT_A_REAL_METRIC")
    assert _metric_from_params(p) == FocusMetric.LAPLACIAN_VARIANCE


# ---------------------------------------------------------------------------
# Metadata auto-detect
# ---------------------------------------------------------------------------

def test_apply_detected_metadata_sets_magnification():
    app = _fake_app()
    app._params = ReconParams(magnification=1.0, pixel_is_effective=True)
    app._toasts = MagicMock()
    app._apply_detected_metadata({"magnification": 40.0})
    assert app._params.magnification == 40.0
    # When we learn the file's M, the pixel value in the file must
    # be the *camera* pixel — so the effective flag flips off.
    assert app._params.pixel_is_effective is False
    app._toasts.show.assert_called_once()


def test_apply_detected_metadata_sets_pixel_size():
    app = _fake_app()
    app._params = ReconParams(pixel_um=5.0)
    app._toasts = MagicMock()
    app._apply_detected_metadata({"pixel_size_m": 3.45e-6})
    assert app._params.pixel_um == pytest.approx(3.45)


def test_apply_detected_metadata_sets_wavelength():
    app = _fake_app()
    app._params = ReconParams(wavelength_nm=632.8)
    app._toasts = MagicMock()
    app._apply_detected_metadata({"wavelength_m": 532e-9})
    assert app._params.wavelength_nm == pytest.approx(532.0)


def test_apply_detected_metadata_no_keys_no_toast():
    """A plain TIFF without any sidecar metadata shouldn't spam toasts."""
    app = _fake_app()
    app._toasts = MagicMock()
    app._apply_detected_metadata({})
    app._toasts.show.assert_not_called()


def test_apply_detected_metadata_ignores_same_values():
    """If the file agrees with what's already in the sidebar, we don't
    count that as a 'detection' worth toasting."""
    app = _fake_app()
    app._params = ReconParams(magnification=40.0, pixel_is_effective=False)
    app._toasts = MagicMock()
    app._apply_detected_metadata({"magnification": 40.0})
    # Value didn't change, so no toast either.
    app._toasts.show.assert_not_called()
