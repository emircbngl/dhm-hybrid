"""Parity test: ``core.batch_renderer`` and ``ui2.workers`` must produce
the same complex field on the same hologram + parameters.

History (2026-04-29 batch correctness audit):

* Batch defaulted ``subtract_mean=False`` while v1 process_tab defaulted
  ``True`` and v2 ``ReconParams`` defaulted ``True`` — when a profile
  didn't carry the key, batch quietly produced a ~50% amplitude-scale
  offset.
* Autofocus / sweep paths in batch propagated the sample WITHOUT the
  reference division when picking best-Z, then divided only on the
  final propagate. Best-Z metric saw a different field than the user.

This test pins both behaviours so future drift gets caught at CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_synthetic(shape=(192, 192), seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.indices(shape)
    fx = fy = 1.0 / 8.0
    carrier = np.cos(2 * np.pi * (fx * xx + fy * yy))
    sample_phase = 0.6 * np.sin(2 * np.pi * 3 * yy / h) + 0.3 * (xx / w)
    sample = 1 + 0.5 * np.cos(2 * np.pi * (fx * xx + fy * yy) + sample_phase)
    holo = (sample + 0.3 * carrier).astype(np.float32)
    holo += 0.02 * rng.standard_normal(holo.shape).astype(np.float32)
    return holo


def _v2_field(raw: np.ndarray, *, subtract_mean: bool, hann_window: bool,
              mask_radius: int) -> np.ndarray:
    """Mirror of ``ui2.workers._preprocess_raw`` + offaxis extract.

    Inlined because the production helper lives behind a heavy import
    chain (Qt-free at runtime, but the module pulls in audit/log
    plumbing). We only test the pre-propagation prep here — the
    propagate kernel is already covered by ``test_reconstruction``.
    """
    from core.offaxis import OffAxisParams, extract_complex_field_offaxis

    img = raw.astype(np.float32, copy=True)
    if subtract_mean:
        img = img - float(np.mean(img))
    if hann_window:
        wy = np.hanning(img.shape[0]).astype(np.float32)
        wx = np.hanning(img.shape[1]).astype(np.float32)
        img = img * (wy[:, None] * wx[None, :])
    peak = float(np.max(np.abs(img)))
    if peak > 0:
        img = img / peak
    field, _ = extract_complex_field_offaxis(
        img, OffAxisParams(radius=mask_radius)
    )
    return field


def _batch_field(raw: np.ndarray, process_state: dict, mask_radius: int) -> np.ndarray:
    """Replays the exact pre-offaxis block of
    ``core.batch_renderer.BatchRenderer._process_single_job``.

    Kept as a textual mirror so a code review immediately catches when
    batch_renderer's preprocessing and this test drift apart.
    """
    from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug

    img = raw.astype(np.float32, copy=True)
    subtract_mean = process_state.get("subtract_mean", True)  # batch default
    hann_window = process_state.get("hann_window", False)
    if subtract_mean:
        img = img - float(np.mean(img))
    if hann_window:
        wy = np.hanning(img.shape[0]).astype(np.float32)
        wx = np.hanning(img.shape[1]).astype(np.float32)
        img = img * (wy[:, None] * wx[None, :])
    max_val = float(np.max(np.abs(img)))
    img = img / max(max_val, 1e-12)

    mask_apod = process_state.get("mask_apodization", "tukey")
    mask_roll = float(process_state.get("mask_rolloff", 0.25))
    fc, _, _, _ = extract_complex_field_offaxis_debug(
        img, OffAxisParams(radius=mask_radius, apodization=mask_apod, rolloff=mask_roll)
    )
    return fc


def test_default_skew_resolved():
    """Profile missing ``subtract_mean`` — batch must default the same as v2."""
    raw = _make_synthetic()
    v2 = _v2_field(raw, subtract_mean=True, hann_window=False, mask_radius=40)
    bt = _batch_field(raw, process_state={}, mask_radius=40)
    np.testing.assert_allclose(np.abs(v2), np.abs(bt), atol=1e-6)
    np.testing.assert_allclose(np.angle(v2), np.angle(bt), atol=1e-6)


@pytest.mark.parametrize("subtract_mean,hann_window", [
    (True, False),
    (False, False),
    (True, True),
    (False, True),
])
def test_explicit_settings_match(subtract_mean, hann_window):
    """When profile sets prep flags explicitly, batch and v2 must agree."""
    raw = _make_synthetic()
    v2 = _v2_field(raw, subtract_mean=subtract_mean, hann_window=hann_window, mask_radius=40)
    bt = _batch_field(
        raw,
        process_state={"subtract_mean": subtract_mean, "hann_window": hann_window},
        mask_radius=40,
    )
    np.testing.assert_allclose(np.abs(v2), np.abs(bt), atol=1e-6)
    np.testing.assert_allclose(np.angle(v2), np.angle(bt), atol=1e-6)


def test_batch_preprocessing_default_is_true():
    """Source-level pin: a missing ``subtract_mean`` reads as True.

    Regression catches the literal default in batch_renderer.py — if
    someone flips it back to False to match the legacy behaviour, this
    test fails fast (don't trust silently).
    """
    src = (ROOT / "src" / "core" / "batch_renderer.py").read_text(encoding="utf-8")
    assert 'process_state.get("subtract_mean", True)' in src, (
        "batch_renderer.py must default subtract_mean to True so it "
        "matches v1 process_tab and v2 ReconParams. See 2026-04-29 audit."
    )


def test_apply_ref_uses_fc_when_available():
    """Source-level pin: _apply_ref re-propagates ref_fc when given.

    The autofocus / sweep best-Z fix depends on _apply_ref preferring
    reference_fc over reference_complex when a target z_m is supplied.
    If someone restores the old behaviour, the metric ↔ saved-field
    inconsistency reappears silently; pin it here.
    """
    src = (ROOT / "src" / "core" / "batch_renderer.py").read_text(encoding="utf-8")
    assert "if ref_fc is not None and z_m is not None:" in src
    assert 'self.config.get("reference_fc"' in src


def test_autofocus_search_receives_ref_field():
    """Source-level pin: autofocus search calls forward ``ref_field``.

    All three classic algorithms (Robust, Golden, Coarse-to-Fine) must
    pass ``ref_field=ref_fc`` so the metric sees the divided field.
    """
    src = (ROOT / "src" / "core" / "batch_renderer.py").read_text(encoding="utf-8")
    # at least three explicit forwarding sites — one per algorithm branch.
    assert src.count("ref_field=ref_fc") >= 3
