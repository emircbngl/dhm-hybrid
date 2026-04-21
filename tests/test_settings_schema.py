"""Settings schema + QSettings-backed store: validation, persistence, migration.

Store tests use QSettings with a scoped organization so they don't clobber
the user's real settings on the dev machine.
"""
from __future__ import annotations

import pytest

from core.settings_schema import (
    SCHEMA_VERSION,
    AppSettings,
    AutofocusDefaults,
    QPIDefaults,
    ReconDefaults,
    validate,
)


# ---- Schema validation ------------------------------------------------------


def test_defaults_are_valid():
    assert validate(AppSettings.defaults()) == []


def test_negative_wavelength_is_rejected():
    s = AppSettings.defaults().with_recon(wavelength_nm=-1.0)
    problems = validate(s)
    assert any("wavelength_nm" in p for p in problems)


def test_zero_pixel_size_is_rejected():
    s = AppSettings.defaults().with_recon(pixel_um=0.0)
    assert any("pixel_um" in p for p in validate(s))


def test_z_min_must_be_less_than_z_max():
    s = AppSettings.defaults().with_autofocus(z_min_mm=50.0, z_max_mm=10.0)
    assert any("z_min_mm" in p and "z_max_mm" in p for p in validate(s))


def test_n_medium_below_one_rejected():
    s = AppSettings.defaults().with_recon(n_medium=0.5)
    assert any("n_medium" in p for p in validate(s))


def test_qpi_cell_must_exceed_medium():
    s = AppSettings.defaults().with_qpi(cell_refractive_index=1.3,
                                        medium_refractive_index=1.33)
    assert any("cell_refractive_index" in p for p in validate(s))


def test_with_recon_preserves_other_sections():
    s0 = AppSettings.defaults()
    s1 = s0.with_recon(wavelength_nm=405.0)
    assert s1.recon.wavelength_nm == 405.0
    assert s1.autofocus == s0.autofocus
    assert s1.qpi == s0.qpi
    assert s1.io == s0.io


def test_as_dict_round_trips_primitives():
    d = AppSettings.defaults().as_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert "recon" in d and "wavelength_nm" in d["recon"]


# ---- QSettings-backed store -------------------------------------------------


@pytest.fixture
def _isolated_qsettings(tmp_path):
    """Redirect QSettings IniFormat/UserScope to ``tmp_path`` and wipe
    any state before and after the test. Required because store._qs()
    uses a fixed org/app pair and would otherwise share state across
    tests (and with the real user's preferences)."""
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    # Clear any leftover state from a prior test run.
    QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope,
        "DHM", "Reconstruction",
    ).clear()
    yield tmp_path
    QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope,
        "DHM", "Reconstruction",
    ).clear()


def test_save_then_load_round_trips(_isolated_qsettings):
    from gui import settings_store

    original = AppSettings.defaults().with_recon(
        wavelength_nm=405.0, pixel_um=2.4, z_mm=12.3,
    ).with_autofocus(
        z_min_mm=5.0, z_max_mm=80.0,
    ).with_io(
        last_folder="/tmp/holograms",
    )

    settings_store.save(original)
    loaded = settings_store.load()

    assert loaded.recon.wavelength_nm == 405.0
    assert loaded.recon.pixel_um == 2.4
    assert loaded.recon.z_mm == 12.3
    assert loaded.autofocus.z_min_mm == 5.0
    assert loaded.autofocus.z_max_mm == 80.0
    assert loaded.io.last_folder == "/tmp/holograms"


def test_load_on_empty_returns_defaults(_isolated_qsettings):
    from gui import settings_store

    loaded = settings_store.load()
    assert loaded == AppSettings.defaults()


def _raw_qs():
    """Same format/scope/org/app as settings_store._qs()."""
    from PySide6.QtCore import QSettings
    return QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope,
        "DHM", "Reconstruction",
    )


def test_invalid_on_disk_falls_back_to_defaults(_isolated_qsettings):
    """If someone hand-edits the ini to a bad value, load() must not explode."""
    from gui import settings_store

    qs = _raw_qs()
    qs.setValue("schema/version", 2)
    qs.setValue("v2/recon/wavelength_nm", -100.0)   # invalid
    qs.sync()

    loaded = settings_store.load()
    # Should fall back to defaults, not propagate the bad value
    assert loaded == AppSettings.defaults()


def test_v1_to_v2_migration_stamps_version(_isolated_qsettings):
    """v1.0.0 settings (no schema/version key) should load as defaults and
    leave the schema stamped at v2."""
    from gui import settings_store

    qs = _raw_qs()
    # Simulate v1.0.0 leftovers (ini format can't store raw bytes, so use str)
    qs.setValue("window/geometry", "fake-geometry-blob")
    qs.sync()

    loaded = settings_store.load()
    assert loaded == AppSettings.defaults()

    qs2 = _raw_qs()
    assert int(qs2.value("schema/version")) == SCHEMA_VERSION
    # v1 window state should be preserved, not wiped
    assert qs2.value("window/geometry") == "fake-geometry-blob"


def test_save_rejects_invalid_settings(_isolated_qsettings):
    """save() should refuse garbage rather than write it."""
    from gui import settings_store

    bad = AppSettings.defaults().with_recon(wavelength_nm=-1.0)
    settings_store.save(bad)

    qs = _raw_qs()
    # Nothing should have been written under v2 prefix
    assert qs.value("v2/recon/wavelength_nm") is None
