"""Scalebar nice-number ladder + auto-pick tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.scalebar import compute_scalebar, nice_length_um


@pytest.mark.parametrize("target,expected", [
    (0.7, 0.5),
    (1.0, 1.0),
    (1.4, 1.0),
    (1.6, 2.0),
    (3.0, 2.0),  # closer to 2 than 5 in log space
    (3.5, 5.0),  # 3.5 is closer to 5 than 2 in log space (10^0.544 vs 10^0.301)
    (7.0, 5.0),
    (12.0, 10.0),
    (45.0, 50.0),
    (78.0, 100.0),
    (250.0, 200.0),
])
def test_nice_length_um_ladder(target, expected):
    assert nice_length_um(target) == pytest.approx(expected, rel=1e-9)


def test_nice_length_um_zero_safe():
    assert nice_length_um(0.0) == 1.0
    assert nice_length_um(-5.0) == 1.0


def test_compute_scalebar_picks_reasonable_default():
    """1024-px frame at 0.5 µm/px → target ≈ 77 µm; closer to 100 than
    50 in log space, so the bar comes out at 100 µm."""
    spec = compute_scalebar(1024, 0.5)
    assert spec is not None
    assert spec.length_um == pytest.approx(100.0)
    assert spec.length_px == pytest.approx(200.0)
    assert spec.label == "100 µm"


def test_compute_scalebar_label_mm_for_large_lengths():
    """Wide-FOV camera with 10 µm pixels → bar in mm."""
    spec = compute_scalebar(2048, 10.0)
    assert spec is not None
    assert spec.length_um >= 1000.0
    assert "mm" in spec.label


def test_compute_scalebar_fixed_length_override():
    spec = compute_scalebar(1024, 0.345, fixed_length_um=25.0)
    assert spec is not None
    assert spec.length_um == 25.0
    assert spec.label == "25 µm"


def test_compute_scalebar_returns_none_on_bad_input():
    assert compute_scalebar(0, 1.0) is None
    assert compute_scalebar(512, 0.0) is None
    assert compute_scalebar(512, -1.0) is None


def test_compute_scalebar_integration_with_offaxis_pixel():
    """Effective pixel after 10× magnification: 5/10 = 0.5 µm at 512 px →
    expected ~38 µm target → 50 µm bar."""
    spec = compute_scalebar(512, 0.5)
    assert spec is not None
    assert spec.length_um == 50.0


# ---------------------------------------------------------------------------
# v1 (Qt) main_window consolidation — source-level pin
# ---------------------------------------------------------------------------

def test_main_window_uses_core_scalebar_not_local_ladder():
    """v1's ``_on_scalebar_toggled`` used to carry its own 1/2/5-ladder
    (``_nice_scalebar_length``) that could disagree with the canonical
    ``core.scalebar`` ladder used by the image panels (ceil-to-ladder vs
    log-space snap), so the main-window bar and the per-panel bar could
    print different lengths for the same frame. That duplicate function
    must be gone and the module must route through ``core.scalebar``."""
    main_window_path = (
        Path(__file__).resolve().parents[1] / "src" / "gui" / "main_window.py"
    )
    source = main_window_path.read_text(encoding="utf-8")
    assert "_nice_scalebar_length" not in source, (
        "main_window.py must not re-implement its own scalebar ladder; "
        "use core.scalebar.compute_scalebar/nice_length_um instead"
    )
    assert "from core.scalebar import" in source, (
        "main_window.py must import the canonical scalebar helpers from "
        "core.scalebar"
    )


# NOTE (2026-07-06): the ui2 ZoomableImagePanel display-scaling test was
# removed with the DPG frontend retirement (it had been permanently
# skipping anyway — dearpygui is not installed). The display-pixel
# rescale behaviour lives on in ui3's viewport, covered by
# tests/test_ui3_spine.py::test_image_panel_displays_array and the
# core ladder tests above.
