"""Diagnostic: run a synthetic hologram through both the v2 single-frame
path (``ui2/workers._preprocess_raw`` + ``_extract_field_with_reference``)
and the batch path (``core.batch_renderer._process_single_job``), then
quantify divergence. Used during the 2026-04-29 batch correctness audit.

Not a unit test — gives the audit a concrete reproducer instead of a
verbal claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# allow direct script invocation from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.offaxis import OffAxisParams, extract_complex_field_offaxis  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)


def _make_synthetic(shape=(256, 256)) -> np.ndarray:
    """Off-axis hologram with a known +1 order and a phase ramp."""
    rng = np.random.default_rng(0)
    h, w = shape
    yy, xx = np.indices(shape)
    # carrier ~ 1/8 of frequency span at 45 deg, classic off-axis layout.
    fx = 1.0 / 8.0
    fy = 1.0 / 8.0
    carrier = np.cos(2 * np.pi * (fx * xx + fy * yy))
    sample_phase = 0.6 * np.sin(2 * np.pi * 3 * yy / h) + 0.3 * (xx / w)
    sample = (1 + 0.5 * np.cos(2 * np.pi * (fx * xx + fy * yy) + sample_phase))
    holo = (sample + 0.3 * carrier).astype(np.float32)
    holo += 0.02 * rng.standard_normal(holo.shape).astype(np.float32)
    return holo


def _v2_path(raw, *, subtract_mean=True, hann_window=False, mask_radius=40):
    """Mirrors ``ui2.workers._preprocess_raw`` + offaxis extract."""
    raw = raw.astype(np.float32)
    if subtract_mean:
        raw = raw - float(np.mean(raw))
    if hann_window:
        wy = np.hanning(raw.shape[0]).astype(np.float32)
        wx = np.hanning(raw.shape[1]).astype(np.float32)
        raw = raw * (wy[:, None] * wx[None, :])
    peak = float(np.max(np.abs(raw)))
    if peak > 0:
        raw = raw / peak
    field, _ = extract_complex_field_offaxis(raw, OffAxisParams(radius=mask_radius))
    return field


def _batch_path(raw, *, subtract_mean=True, hann_window=False, mask_radius=40,
                mask_apod="tukey", mask_roll=0.25):
    """Mirrors ``core.batch_renderer`` pre-offaxis block.

    The ``subtract_mean`` default tracks ``batch_renderer.py``'s
    ``process_state.get(...)`` default — flipped from False to True
    in the 2026-04-29 audit so batch and v2 stop drifting on
    profiles that don't carry the key.
    """
    img = raw.astype(np.float32)
    if subtract_mean:
        img = img - float(np.mean(img))
    if hann_window:
        wy = np.hanning(img.shape[0]).astype(np.float32)
        wx = np.hanning(img.shape[1]).astype(np.float32)
        img = img * (wy[:, None] * wx[None, :])
    max_val = float(np.max(np.abs(img)))
    img = img / max(max_val, 1e-12)
    field, _ = extract_complex_field_offaxis(
        img,
        OffAxisParams(radius=mask_radius, apodization=mask_apod, rolloff=mask_roll),
    )
    return field


def _reconstruct(field, *, wl_nm=632.8, px_um=5.0, z_mm=0.0, n=1.0):
    p = ReconstructionParams(
        wavelength_m=wl_nm * 1e-9,
        pixel_size_m=px_um * 1e-6,
        z_m=z_mm * 1e-3,
        n=n,
    )
    return propagate(field, p, ReconstructionMethod.ASM, force_python=True)


def _summary(label, field):
    amp = np.abs(field)
    pha = np.angle(field)
    print(f"  {label}: amp mean={amp.mean():.4f} std={amp.std():.4f} | "
          f"pha std={pha.std():.4f}")


def _compare(a, b, label):
    diff = a - b
    wrap = np.angle(np.exp(1j * np.angle(a)) / np.exp(1j * np.angle(b)))
    print(f"\n=== {label} ===")
    print(f"  v2 - batch: max|Δamp|={np.max(np.abs(np.abs(a) - np.abs(b))):.4f}, "
          f"std|Δamp|={np.std(np.abs(a) - np.abs(b)):.4f}")
    print(f"  v2 - batch: max|Δphase wrap|={np.max(np.abs(wrap)):.4f}, "
          f"std|Δphase wrap|={np.std(wrap):.4f}")


def main():
    raw = _make_synthetic()
    print("Synthetic hologram:", raw.shape, raw.dtype, "min", raw.min(), "max", raw.max())

    print("\n--- Scenario 1: profile 'subtract_mean' MISSING (lab default) ---")
    print("  (after 2026-04-29 audit, batch default == v2 default == True)")
    v2_field = _v2_path(raw)            # subtract_mean default = True
    bt_field = _batch_path(raw)         # subtract_mean default = True (post-fix)
    v2_recon = _reconstruct(v2_field)
    bt_recon = _reconstruct(bt_field)
    _summary("v2", v2_recon)
    _summary("batch", bt_recon)
    _compare(v2_recon, bt_recon, "Scenario 1 (default skew)")

    print("\n--- Scenario 2: profile EXPLICITLY sets subtract_mean=True ---")
    v2_field = _v2_path(raw, subtract_mean=True)
    bt_field = _batch_path(raw, subtract_mean=True)
    v2_recon = _reconstruct(v2_field)
    bt_recon = _reconstruct(bt_field)
    _summary("v2", v2_recon)
    _summary("batch", bt_recon)
    _compare(v2_recon, bt_recon, "Scenario 2 (aligned)")

    print("\n--- Scenario 3: subtract_mean=False BOTH (worst-case) ---")
    v2_field = _v2_path(raw, subtract_mean=False)
    bt_field = _batch_path(raw, subtract_mean=False)
    v2_recon = _reconstruct(v2_field)
    bt_recon = _reconstruct(bt_field)
    _summary("v2", v2_recon)
    _summary("batch", bt_recon)
    _compare(v2_recon, bt_recon, "Scenario 3 (both False)")


if __name__ == "__main__":
    main()
