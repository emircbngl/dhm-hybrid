"""AIDefaults dataclass + v11 → v12 settings_store migration.

The migration only stamps the version (defaults backfill happens at
load time via the dataclass factory). We assert the bump itself, the
new field's defaults, and that ``with_ai`` produces a fresh AppSettings.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings

from core.settings_schema import (
    SCHEMA_VERSION,
    AIDefaults,
    AppSettings,
    validate,
)
from gui import settings_store


@pytest.fixture(autouse=True)
def _scoped_qsettings(tmp_path, monkeypatch):
    """Redirect QSettings ini to tmp dir so tests don't clobber real prefs."""
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    yield


def test_schema_version_bumped_to_at_least_12():
    assert SCHEMA_VERSION >= 12


def test_ai_defaults_fields_match_plan():
    ai = AIDefaults()
    assert ai.enabled is True
    assert ai.endpoint_url.startswith("http://")
    assert ai.model_name
    assert 0.0 <= ai.temperature <= 2.0
    assert ai.max_iterations >= 1
    assert ai.restrict_to_home is True
    assert ai.confirm_irreversible is True
    assert ai.audit_redact_for_llm is True


def test_app_settings_has_ai_field():
    s = AppSettings.defaults()
    assert isinstance(s.ai, AIDefaults)


def test_with_ai_returns_new_instance_with_overrides():
    s = AppSettings.defaults()
    s2 = s.with_ai(model_name="qwen2.5:7b-instruct", temperature=0.5)
    assert s2.ai.model_name == "qwen2.5:7b-instruct"
    assert s2.ai.temperature == 0.5
    # Original is untouched
    assert s.ai.model_name == AIDefaults().model_name


def test_validate_passes_with_default_ai():
    s = AppSettings.defaults()
    assert validate(s) == []


def test_save_load_round_trip_preserves_ai_settings():
    s = AppSettings.defaults().with_ai(
        model_name="llama3.1:8b-instruct",
        temperature=0.4,
        max_iterations=12,
        restrict_to_home=False,
    )
    settings_store.save(s)

    reloaded = settings_store.load()
    assert reloaded.ai.model_name == "llama3.1:8b-instruct"
    assert reloaded.ai.temperature == pytest.approx(0.4)
    assert reloaded.ai.max_iterations == 12
    assert reloaded.ai.restrict_to_home is False


def test_load_with_no_stored_ai_returns_defaults():
    # Fresh QSettings (autouse fixture wipes path); load → defaults.
    reloaded = settings_store.load()
    assert isinstance(reloaded.ai, AIDefaults)
    assert reloaded.ai.model_name == AIDefaults().model_name


def test_migration_v11_to_v12_stamps_version():
    # Stage a v11 ini state by writing the version key directly.
    qs = QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        "DHM",
        "Reconstruction",
    )
    qs.setValue("schema/version", 11)
    qs.sync()
    # Loading should bump the stamp without crashing.
    reloaded = settings_store.load()
    qs2 = QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        "DHM",
        "Reconstruction",
    )
    assert int(qs2.value("schema/version")) == SCHEMA_VERSION
    assert isinstance(reloaded.ai, AIDefaults)
