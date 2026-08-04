"""Real-hologram reconstruction validation on the USAF-1951 corpus (2026-07-08).

Hybrid's reconstruction accuracy was validated ONLY on synthetic round-trips
(test_reconstruction_accuracy.py, seeded rng). This wires the real captured
dataset at ``data/220825`` — a USAF-1951 resolution target imaged off-axis,
WITH the original (Phyton/Julia) app's own ASM reconstruction saved as a
reference render — into a real cross-version regression, so Hybrid's ASM is
pinned against an independent implementation on a real known physical target.

Reconstruction config (derived, not guessed):
* The original app fed ``propagate`` its EFFECTIVE pixel size
  (Phyton main.py:1905 ``pixel_size_m=self._effective_pixel_size_m()``).
  The reference is SHARP at the filename's z = 0.043 m, which is only
  physically possible with the CAMERA pixel (3.45 µm); the effective
  0.138 µm pixel at 43 mm would blur past all structure. So for THIS
  capture the effective pixel equalled the camera pixel (magnification 1
  / pixel-is-effective), i.e. propagation pixel = 3.45 µm, z = 0.043 m,
  ASM, n = 1.0. This reproduces the reference at correlation ~0.71.

Only AMPLITUDE is compared: USAF is an amplitude target (chrome bars on
glass), so its unwrapped phase is unstructured and does NOT correlate across
different unwrap implementations (measured ~0.04) — asserting it would be
wrong. The reference BMPs are 8-bit contrast-stretched display renders, so
comparison is scale/offset-invariant Pearson correlation over the four axis
orientations (the apps differ in axis convention).

Marked ``slow`` (loads a 1600×1200 frame + several FFTs) and skips cleanly
when the corpus is absent, so ``pytest -m "not slow"`` / a data-less checkout
stay green. data/220825 is now load-bearing — do not delete it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_USAF_DIR = ROOT / "data" / "220825" / "IMGPLNBACK_USAF"
_HOLO = _USAF_DIR / "IMGPLNBACK_USAF_USAF_TIF.tif"

# Lab-confirmed optics (original report.txt convention); z from the reference
# filename (…_at_z_43000000_m_… = 0.043 m). Propagation pixel = camera pixel
# (see module docstring).
_WL_M = 632.8e-9
_PIXEL_M = 3.45e-6
_Z_M = 0.043

pytestmark = pytest.mark.skipif(
    not _HOLO.exists(),
    reason="real USAF corpus (data/220825) not present in this checkout",
)


def _ref(pattern: str) -> np.ndarray:
    """Grayscale channel of a reference BMP matched by glob (its filename
    carries a capture timestamp)."""
    from PIL import Image
    matches = sorted(_USAF_DIR.glob(pattern))
    if not matches:
        pytest.skip(f"reference {pattern} missing")
    return np.asarray(Image.open(matches[0]))[..., 0].astype(np.float64)


def _load_holo() -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(_HOLO)).astype(np.float64)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def _best_orientation_corr(mine: np.ndarray, ref: np.ndarray) -> float:
    """Max Pearson correlation over the 4 axis orientations — the two apps
    differ in row/col flip convention; correlation is already scale/offset
    invariant so it absorbs the reference render's contrast stretch."""
    cands = (mine, np.flipud(mine), np.fliplr(mine), np.rot90(mine, 2))
    return max(_corr(c, ref) for c in cands)


def _reconstruct(z_m: float, pixel_m: float = _PIXEL_M) -> np.ndarray:
    """Return the complex field reconstructed from the real hologram."""
    from core.offaxis import OffAxisParams, extract_complex_field_offaxis
    from core.reconstruction import (
        ReconstructionMethod, ReconstructionParams, propagate,
    )
    holo = _load_holo()
    field, _center = extract_complex_field_offaxis(
        holo, OffAxisParams(radius=80))
    params = ReconstructionParams(
        wavelength_m=_WL_M, pixel_size_m=pixel_m, z_m=z_m, n=1.0)
    return propagate(field, params, ReconstructionMethod.ASM)


@pytest.mark.slow
def test_real_usaf_amplitude_matches_reference():
    """Hybrid's ASM reconstruction of the real USAF hologram at z=0.043 m
    reproduces the original app's reference amplitude render — a real,
    cross-version, physics-grounded correctness anchor (not a synthetic
    round-trip)."""
    ref_amp = _ref("ASM_Propagated_Field_at_z_43000000_m_*.bmp")
    amp = np.abs(_reconstruct(_Z_M))
    assert amp.shape == ref_amp.shape

    focus_corr = _best_orientation_corr(amp, ref_amp)
    # Empirically 0.71 at the documented config; 0.60 leaves comfortable
    # margin for the lossy 8-bit reference render.
    assert focus_corr >= 0.60, (
        f"reconstruction does not match the reference USAF (corr={focus_corr:.3f})")

    # Focus-sensitivity: a wildly out-of-focus reconstruction correlates
    # measurably worse (0.71 @ focus vs ~0.56 @ 300 mm), proving the match
    # is the reconstruction, not just the target's gross outline surviving
    # any propagation.
    off_corr = _best_orientation_corr(np.abs(_reconstruct(0.30)), ref_amp)
    assert focus_corr > off_corr, (
        f"in-focus corr {focus_corr:.3f} not better than off-focus {off_corr:.3f}")


@pytest.mark.slow
def test_real_data_asm_round_trip_recovers_field():
    """ASM invertibility on REAL captured data: propagating the extracted
    field +z then −z returns it. Complements the synthetic round-trip test
    with a real off-axis field (which carries real sensor noise + a real
    carrier, unlike the seeded synthetic one)."""
    from core.offaxis import OffAxisParams, extract_complex_field_offaxis
    from core.reconstruction import (
        ReconstructionMethod, ReconstructionParams, propagate,
    )
    holo = _load_holo()
    field, _ = extract_complex_field_offaxis(holo, OffAxisParams(radius=80))

    def prop(f, z):
        return propagate(
            f, ReconstructionParams(wavelength_m=_WL_M, pixel_size_m=_PIXEL_M,
                                    z_m=z, n=1.0),
            ReconstructionMethod.ASM)

    back = prop(prop(field, _Z_M), -_Z_M)
    rel_rmse = float(np.linalg.norm(back - field) / np.linalg.norm(field))
    assert rel_rmse < 1e-4, f"ASM round-trip drifted on real data (rel-RMSE={rel_rmse:.2e})"


@pytest.mark.slow
def test_real_offaxis_plus_one_order_is_genuinely_off_center():
    """Auto +1-order detection on the real hologram finds a carrier that is
    clearly OFF the DC term — i.e. this is a real off-axis geometry and the
    detector isn't just latching onto DC."""
    from core.masking import detect_plus_one_order
    holo = _load_holo()
    cy, cx = detect_plus_one_order(holo, exclusion_radius=20)
    h, w = holo.shape[:2]
    dist_from_dc = np.hypot(cy - h // 2, cx - w // 2)
    assert dist_from_dc > 50, (
        f"+1-order center {(cy, cx)} sits on the DC term — off-axis "
        "detection failed on real data")
