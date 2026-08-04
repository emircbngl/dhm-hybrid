"""Depth-map ref-aware tests + boundary saturation diagnostic.

History (2026-04-30):
* ``compute_depth_map`` had no ``ref_field`` argument. v1's depth handler
  loaded a reference but didn't forward it, so the per-z metric ran on
  un-referenced fields and saturated to the scan boundary on real lab
  data — operator saw a depth file pinned to z_min/z_max instead of
  cell-shaped depth. Fixed by adding ``ref_field`` and forwarding from
  every v1 caller that had ``_reference_fc`` available.
* ``_prepare_af_field`` hardcoded ``n=1.0`` even with the QPI tab's
  ``n_medium`` set to 1.337 (water). Propagation distance scales as
  ``n``, so depth z came out off by ~25% for any aqueous sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _build_field(seed=0):
    """Synthetic +1-order field that looks like a real demodulated hologram."""
    from core.offaxis import OffAxisParams, extract_complex_field_offaxis

    rng = np.random.default_rng(seed)
    h = w = 192
    yy, xx = np.indices((h, w))
    fx = fy = 1.0 / 8.0
    sample_phase = 0.6 * np.sin(2 * np.pi * 3 * yy / h) + 0.3 * (xx / w)
    holo = (1 + 0.5 * np.cos(2 * np.pi * (fx * xx + fy * yy) + sample_phase)
            + 0.3 * np.cos(2 * np.pi * (fx * xx + fy * yy))).astype(np.float32)
    holo += 0.02 * rng.standard_normal(holo.shape).astype(np.float32)
    holo -= holo.mean()
    holo /= np.max(np.abs(holo))
    fc, _ = extract_complex_field_offaxis(holo, OffAxisParams(radius=40))
    return fc


def test_compute_depth_map_accepts_ref_field():
    """ref_field kwarg works and changes the output."""
    from core.depth_map import compute_depth_map
    from core.reconstruction import ReconstructionMethod, ReconstructionParams
    from core.autofocus import FocusMetric

    sample = _build_field(seed=0)
    ref = _build_field(seed=1)  # different noise realisation
    params = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=3.45e-6,
                                  z_m=0.0, n=1.337)

    plain = compute_depth_map(
        sample, params, ReconstructionMethod.ASM,
        z_min_m=-1e-3, z_max_m=1e-3, n_steps=15,
        metric=FocusMetric.LAPLACIAN_VARIANCE,
    )
    referenced = compute_depth_map(
        sample, params, ReconstructionMethod.ASM,
        z_min_m=-1e-3, z_max_m=1e-3, n_steps=15,
        metric=FocusMetric.LAPLACIAN_VARIANCE,
        ref_field=ref,
    )

    assert plain.z_map.shape == referenced.z_map.shape
    # Reference division must change the metric landscape — the maps
    # cannot be bit-identical.
    assert not np.allclose(plain.z_map, referenced.z_map), (
        "ref_field had no effect on the depth map — division wasn't applied"
    )
    # No NaN/Inf in either output.
    for r in (plain, referenced):
        assert not np.isnan(r.z_map).any()
        assert not np.isinf(r.z_map).any()


def test_compute_depth_map_rejects_mismatched_ref_shape():
    """Wrong-sized ref must raise instead of silently broadcasting."""
    from core.depth_map import compute_depth_map
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    sample = _build_field()
    bad_ref = np.zeros((64, 64), dtype=np.complex64)
    params = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=3.45e-6,
                                  z_m=0.0, n=1.0)

    with pytest.raises(ValueError, match="ref_field shape"):
        compute_depth_map(
            sample, params, ReconstructionMethod.ASM,
            z_min_m=-1e-3, z_max_m=1e-3, n_steps=5,
            ref_field=bad_ref,
        )


def test_boundary_saturation_warning_logged(caplog):
    """Pathological flat field saturates argmax → WARNING."""
    import logging

    from core.depth_map import compute_depth_map
    from core.reconstruction import ReconstructionMethod, ReconstructionParams

    # A near-zero field has no usable focus signal. The local Laplacian
    # variance is uniform noise; argmax distributes randomly with a
    # ~2/n_steps bias toward each boundary tie. We size n_steps small
    # enough that the bias forces > 50% to the extremes.
    flat = np.full((96, 96), 1e-6 + 0j, dtype=np.complex64)
    params = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=3.45e-6,
                                  z_m=0.0, n=1.0)
    with caplog.at_level(logging.WARNING, logger="core.depth_map"):
        compute_depth_map(
            flat, params, ReconstructionMethod.ASM,
            z_min_m=-1e-3, z_max_m=1e-3, n_steps=4,
        )
    assert any("saturated to scan boundary" in r.message for r in caplog.records), (
        "No boundary-saturation warning logged for a flat field — the "
        "diagnostic isn't firing on pathological input."
    )


def test_v1_depth_callsites_forward_reference():
    """Source-level pin: every compute_depth_map call in v1 main_window
    must forward the reference through ``_active_reference_fc()`` — the
    checkbox-gated accessor. Catches both regressions: a handler that
    skips the kwarg silently, and one that bypasses the gate by reading
    ``_reference_fc`` directly (2026-07-05 review: depth maps kept
    dividing by the reference after the user unchecked
    'Enable reference subtraction')."""
    src = (ROOT / "src" / "gui" / "main_window.py").read_text(encoding="utf-8")
    # All three handlers (tomography bundle, depth overlay, depth save)
    # use the same pattern. Each must include this exact forwarder line.
    forwarder = "ref_field=self._active_reference_fc()"
    assert src.count(forwarder) >= 3, (
        f"Expected ≥3 gated forwards of the reference into "
        f"compute_depth_map, found {src.count(forwarder)}. Did a new "
        f"handler get added without wiring the gated reference?"
    )
    # No depth/autofocus path may bypass the checkbox gate.
    assert 'ref_field=getattr(self, "_reference_fc", None)' not in src, (
        "Found an ungated ref_field forwarder — use "
        "self._active_reference_fc() so the 'Enable reference "
        "subtraction' checkbox is honored."
    )


def test_prepare_af_field_uses_n_medium():
    """Source-level pin: _prepare_af_field reads n_medium from qpi tab,
    not the stale ``n=1.0`` hardcode."""
    src = (ROOT / "src" / "gui" / "main_window.py").read_text(encoding="utf-8")
    # The hardcoded literal must be gone.
    bad = "ReconstructionParams(wavelength_m=wl, pixel_size_m=px, z_m=0.0, n=1.0)"
    assert bad not in src, (
        "n=1.0 hardcoded in _prepare_af_field — depth/focus z is off by "
        "a factor of n_medium for any aqueous sample. Read from "
        "self.sidebar_tabs.qpi_tab.n_medium instead."
    )
    # And the new path reads from qpi_tab.
    assert "qtab.n_medium.value()" in src
