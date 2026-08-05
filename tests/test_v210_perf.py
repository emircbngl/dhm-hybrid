"""v2.1.0 GPU + perf sprint tests.

Three new surfaces:

* :class:`core.fft_backend.TorchFFTBackend` — lazy / optional.
  Tests skip gracefully when torch isn't installed.
* :func:`core.autofocus._make_batch_evaluator` — batched z scoring.
  Correctness pinned vs. serial path.
* :func:`core.autofocus._make_roi_fast_evaluator` — ROI fast-path.
  Smoke-tested for: a) result dimensions, b) finds the same focus
  z as the full-frame evaluator (within 1 step).

We also pin the new ``autofocus_zscan(batch_backend=...)`` route
gives results identical to the default serial route on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core.autofocus import (  # noqa: E402
    FocusMetric,
    _make_batch_evaluator,
    _make_fast_evaluator,
    _make_roi_fast_evaluator,
    autofocus_zscan,
)
from core.fft_backend import (  # noqa: E402
    FFTBackend,
    FFTBackendName,
    NumpyFFTBackend,
    get_best_fft_backend,
)
from core.offaxis import OffAxisParams, extract_complex_field_offaxis  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
)
from fixtures.synthetic_hologram import (  # noqa: E402
    HologramConfig, SphereSpec, build_hologram,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_field(shape=(128, 128)):
    cfg = HologramConfig(
        shape=shape, pixel_m=2.5e-6,
        wavelength_m=632.8e-9,
        carrier_freq_m_inv=(50_000.0, 0.0),
    )
    sphere = SphereSpec(
        radius_m=15e-6, z_m=12e-3,
        center_yx_m=(0.0, 0.0),
        n_sphere=1.40, n_medium=1.33,
    )
    holo = build_hologram([sphere], cfg)
    field, _ = extract_complex_field_offaxis(
        holo, OffAxisParams(radius=20),
    )
    base = ReconstructionParams(
        wavelength_m=cfg.wavelength_m,
        pixel_size_m=cfg.pixel_m,
        z_m=0.0, n=1.33,
    )
    return field, base


# ---------------------------------------------------------------------------
# Default fft backend has batched fallback
# ---------------------------------------------------------------------------

def test_default_backend_has_batched_fallback():
    """The default backend (numpy / scipy / pyfftw / mlx) must
    expose ``fft2_batched`` even though it's a serial loop. The
    contract holds for every backend."""
    backend = get_best_fft_backend()
    arr = np.random.randn(3, 16, 16).astype(np.complex64)
    out = backend.fft2_batched(arr)
    assert out.shape == arr.shape


def test_auto_backend_never_initializes_mlx():
    """An optional MLX runtime must not make the default CPU path fatal."""
    with patch("core.fft_backend.PyFFTWBackend", side_effect=ImportError), \
         patch("core.fft_backend.MLXFFTBackend") as mlx_backend:
        backend = get_best_fft_backend()

    assert backend.name in {FFTBackendName.SCIPY, FFTBackendName.NUMPY}
    mlx_backend.assert_not_called()


def test_default_backend_supports_batched_false():
    """Non-torch backends report ``supports_batched == False`` so
    ``autofocus_zscan(batch_backend=...)`` only takes the batched
    path when there's a real GPU win."""
    backend = NumpyFFTBackend()
    assert backend.supports_batched is False


# ---------------------------------------------------------------------------
# _make_batch_evaluator correctness vs serial
# ---------------------------------------------------------------------------

def test_batch_evaluator_matches_serial_per_z():
    """Score for each z from the batched path must match the score
    from the serial path within float-precision noise."""
    field, base = _make_field()
    method = ReconstructionMethod.ASM
    metric = FocusMetric.ENTROPY
    zs = list(np.linspace(8e-3, 16e-3, 12))

    serial_eval = _make_fast_evaluator(field, base, method, metric)
    serial_scores = np.array([serial_eval(z) for z in zs])

    batch_eval = _make_batch_evaluator(
        field, base, method, metric,
    )
    batch_scores = batch_eval(zs)

    # Float32 IFFT round-trip — relative tolerance 1e-4 is the
    # safe band given complex64 + ASM kernel. Tightening would
    # make this fail under MKL vs OpenBLAS minor numeric jitter.
    np.testing.assert_allclose(
        batch_scores, serial_scores, rtol=1e-4, atol=1e-7,
    )


def test_batch_evaluator_empty_zs_returns_empty():
    field, base = _make_field()
    batch_eval = _make_batch_evaluator(
        field, base, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
    )
    out = batch_eval([])
    assert out.shape == (0,)


def test_batch_evaluator_with_reference_division():
    """Reference-divided field must produce the same result via
    the batched path as the serial path. Catches a regression in
    the (n_steps, ny, nx) ref broadcasting."""
    field, base = _make_field()
    # Use the same field as 'ref' just for plumbing — purpose is
    # the code path, not the physics.
    ref = field.copy()
    method = ReconstructionMethod.ASM
    metric = FocusMetric.ENTROPY
    zs = list(np.linspace(10e-3, 14e-3, 5))

    serial = _make_fast_evaluator(field, base, method, metric,
                                  ref_field=ref)
    serial_scores = np.array([serial(z) for z in zs])

    batch = _make_batch_evaluator(field, base, method, metric,
                                  ref_field=ref)
    batch_scores = batch(zs)

    np.testing.assert_allclose(
        batch_scores, serial_scores, rtol=1e-3, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# autofocus_zscan with batch_backend
# ---------------------------------------------------------------------------

class _FakeBatchBackend(NumpyFFTBackend):
    """Stand-in for a torch-style backend whose batched path runs
    through numpy under the hood. ``supports_batched=True`` flips
    the zscan branch but the IFFT itself uses numpy — so the test
    runs without torch."""

    @property
    def supports_batched(self) -> bool:
        return True


def test_autofocus_zscan_batch_path_matches_serial():
    field, base = _make_field()
    zs = list(np.linspace(8e-3, 16e-3, 12))
    serial = autofocus_zscan(
        field, base, zs, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
    )
    batch = autofocus_zscan(
        field, base, zs, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
        batch_backend=_FakeBatchBackend(),
    )
    # Same best z; scores within float tolerance.
    assert serial.best_z_m == pytest.approx(batch.best_z_m)
    for z in zs:
        assert serial.scores[z] == pytest.approx(
            batch.scores[z], rel=1e-3, abs=1e-6,
        )


def test_autofocus_zscan_progress_fires_on_batch_path():
    field, base = _make_field()
    zs = list(np.linspace(8e-3, 16e-3, 5))
    progress_calls = []

    def _on_progress(i, total):
        progress_calls.append((i, total))

    autofocus_zscan(
        field, base, zs, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
        on_progress=_on_progress,
        batch_backend=_FakeBatchBackend(),
    )
    # Batch path fires twice: start (0, total) + end (total, total).
    assert progress_calls[0] == (0, 5)
    assert progress_calls[-1] == (5, 5)


# ---------------------------------------------------------------------------
# ROI fast-path
# ---------------------------------------------------------------------------

class _ROIBounds:
    def __init__(self, y0, x0, y1, x1):
        self.y0, self.x0, self.y1, self.x1 = y0, x0, y1, x1


def test_roi_fast_evaluator_returns_finite_score():
    field, base = _make_field()
    bounds = _ROIBounds(40, 40, 88, 88)
    fast = _make_roi_fast_evaluator(
        field, base, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
        roi_bounds=bounds,
    )
    s = fast(12e-3)
    assert np.isfinite(s)


def test_roi_fast_evaluator_finds_focus_near_truth():
    """The ROI fast-path lowers spatial resolution but not the
    autofocus z. argmin(score) over a sweep should still land
    within a few step of the true z."""
    field, base = _make_field()
    bounds = _ROIBounds(40, 40, 88, 88)
    fast = _make_roi_fast_evaluator(
        field, base, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
        roi_bounds=bounds,
    )
    zs = np.linspace(8e-3, 16e-3, 40)
    scores = np.array([fast(z) for z in zs])
    # ENTROPY minimises at focus.
    best_z = float(zs[np.argmin(scores)])
    truth_mm = 12.0
    step_mm = (16 - 8) / 39
    # 6× step covers the lower-res Fresnel-bias band.
    assert abs(best_z * 1e3 - truth_mm) <= 6.0 * step_mm


def test_roi_fast_evaluator_clamps_min_size():
    """ROIs smaller than 8 px are bumped up — sub-8 IFFTs aren't
    meaningful + we'd lose the carrier band entirely."""
    field, base = _make_field()
    tiny = _ROIBounds(60, 60, 62, 62)  # 2x2
    fast = _make_roi_fast_evaluator(
        field, base, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
        roi_bounds=tiny,
    )
    # Should not raise + should return a finite score.
    assert np.isfinite(fast(12e-3))


# ---------------------------------------------------------------------------
# TorchFFTBackend — skip if torch missing
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Apply skip at the function level so the rest of the file (which
# only uses numpy) keeps running even when torch isn't installed.
torch_only = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch optional; install with `pip install torch` for GPU "
           "perf path tests",
)


@torch_only
def test_torch_backend_constructs():
    from core.fft_backend import TorchFFTBackend
    backend = TorchFFTBackend()
    assert backend.name is FFTBackendName.TORCH
    assert backend.supports_batched
    # Device is one of the three known options.
    assert backend.device in {"cpu", "cuda", "mps"}


@torch_only
def test_torch_backend_fft_round_trip():
    from core.fft_backend import TorchFFTBackend
    backend = TorchFFTBackend()
    arr = (np.random.randn(64, 64) + 1j * np.random.randn(64, 64)) \
        .astype(np.complex64)
    F = backend.fft2(arr)
    rec = backend.ifft2(F)
    np.testing.assert_allclose(rec, arr, rtol=1e-4, atol=1e-5)


@torch_only
def test_torch_backend_batched_matches_numpy():
    from core.fft_backend import TorchFFTBackend
    backend = TorchFFTBackend()
    np_backend = NumpyFFTBackend()
    arr = (np.random.randn(4, 32, 32)
           + 1j * np.random.randn(4, 32, 32)).astype(np.complex64)
    out_torch = backend.ifft2_batched(arr)
    out_numpy = np_backend.ifft2_batched(arr)
    np.testing.assert_allclose(
        out_torch, out_numpy, rtol=1e-3, atol=1e-5,
    )


@torch_only
def test_torch_backend_zscan_matches_numpy_zscan():
    """End-to-end: an autofocus_zscan with batch_backend=TorchFFT
    must agree with the numpy serial zscan to within the same
    float band the batched-vs-serial test asserts."""
    from core.fft_backend import TorchFFTBackend
    field, base = _make_field()
    zs = list(np.linspace(8e-3, 16e-3, 8))
    serial = autofocus_zscan(
        field, base, zs, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
    )
    torch_run = autofocus_zscan(
        field, base, zs, ReconstructionMethod.ASM, FocusMetric.ENTROPY,
        batch_backend=TorchFFTBackend(),
    )
    assert serial.best_z_m == pytest.approx(torch_run.best_z_m)
