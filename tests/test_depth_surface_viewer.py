"""Smoke tests for ``gui.widgets.depth_surface_viewer.DepthSurfaceViewer``.

The viewer is a Qt + pyqtgraph.opengl dialog. The render code path
(``GLViewWidget`` instantiation) needs a real OpenGL context; without
one it segfaults rather than raising — which a headless test
environment can't satisfy. So we lock down structural correctness
here and leave the live render test to the manual smoke run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("OpenGL")


def test_viewer_module_imports():
    """Catches missing pyqtgraph.opengl / PyOpenGL up front."""
    from gui.widgets.depth_surface_viewer import DepthSurfaceViewer
    assert DepthSurfaceViewer is not None


def test_viewer_uses_no_negative_phase_clip():
    """Pin: the viewer must NOT zero negative phase values.

    The lab made it explicit on 2026-04-30 that both signs of phase
    carry physical meaning. If anyone adds a ``np.maximum(0, …)`` /
    ``arr[arr < 0] = 0`` pattern to the surface viewer, this fails.
    """
    src = (ROOT / "src" / "gui" / "widgets" /
           "depth_surface_viewer.py").read_text(encoding="utf-8")
    forbidden = (
        "np.maximum(0",
        ".clip(0, ",
        "[arr < 0] = 0",
        "[phase < 0] = 0",
    )
    for needle in forbidden:
        assert needle not in src, (
            f"depth_surface_viewer.py contains '{needle}' — that clips "
            f"negative phase. See feedback memory feedback_phase_signs.md."
        )


def test_viewer_show_phase_validates_array_signature():
    """Function signature accepts a 2-D float32 array."""
    import inspect
    from gui.widgets.depth_surface_viewer import DepthSurfaceViewer
    sig = inspect.signature(DepthSurfaceViewer.show_phase)
    assert "phase_rad" in sig.parameters
    assert "pixel_size_um" in sig.parameters
    sig2 = inspect.signature(DepthSurfaceViewer.show_depth)
    assert "depth_result" in sig2.parameters
