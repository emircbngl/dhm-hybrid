"""Shared pytest fixtures for the DHM validation test suite.

All synthetic data fixtures are seeded so tests are deterministic across runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic data — seeded so every test sees the same arrays
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_hologram():
    """Return a function that produces a tilted-plane-wave fringe pattern."""
    def _make(shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
        ny, nx = shape
        y, x = np.mgrid[:ny, :nx].astype(np.float32)
        # Tilted plane wave + gaussian envelope → DC + off-axis fringes.
        fringes = 0.5 + 0.4 * np.cos(2.0 * np.pi * (0.12 * x + 0.08 * y))
        cy, cx = ny / 2.0, nx / 2.0
        env = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (0.35 * nx) ** 2)
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.01, size=shape).astype(np.float32)
        return (fringes * env + noise).astype(np.float32)
    return _make


@pytest.fixture
def synthetic_complex_field():
    """Return a function that produces a complex64 field with known phase structure."""
    def _make(shape: Tuple[int, int] = (128, 128)) -> np.ndarray:
        ny, nx = shape
        y, x = np.mgrid[:ny, :nx].astype(np.float32)
        cy, cx = ny / 2.0, nx / 2.0
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        # Gaussian amplitude, quadratic phase — imitates a defocused point source.
        amp = np.exp(-r2 / (0.25 * nx) ** 2).astype(np.float32)
        phase = (0.002 * r2).astype(np.float32)
        return (amp * np.exp(1j * phase)).astype(np.complex64)
    return _make


@pytest.fixture
def flat_complex_field():
    """Return a function that yields a near-zero amplitude deterministic field."""
    def _make(shape: Tuple[int, int] = (128, 128)) -> np.ndarray:
        # Tiny but nonzero so downstream log/divide paths stay finite.
        return (np.full(shape, 1e-6, dtype=np.float32)
                .astype(np.complex64))
    return _make


# ---------------------------------------------------------------------------
# Audit isolation — redirect the on-disk log into tmp_path
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_audit_dir(tmp_path, monkeypatch):
    """Redirect the module-level audit directory into tmp_path/audit.

    Also clears the module singleton so subsequent ``get_audit_log()`` uses
    the tmp directory.
    """
    from core import audit as audit_mod

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(audit_mod, "_AUDIT_DIR", audit_dir)
    # Drop any pre-existing singleton bound to the real home directory.
    monkeypatch.setattr(audit_mod, "_SINGLETON", None)
    return audit_dir


# ---------------------------------------------------------------------------
# Qt — one QApplication shared across the session; skip if offscreen is broken
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qt_app():
    """Session-scoped QApplication.

    Uses the offscreen Qt platform so headless CI doesn't need a display.
    Skipped if Qt cannot initialise at all.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - ships only if PySide6 missing
        pytest.skip(f"PySide6 unavailable: {exc}")

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - offscreen plugin missing
            pytest.skip(f"QApplication could not start (likely no offscreen plugin): {exc}")
    return app
