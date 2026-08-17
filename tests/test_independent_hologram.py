"""Contract tests for the independent (inverse-crime-free) forward model.

The whole value of `tests/fixtures/independent_hologram.py` is that it does NOT
share a propagator with the reconstruction. That property is not visible in any
output number — a generator that quietly imported `core.reconstruction` would
still produce plausible holograms while silently restoring the inverse crime.
So it is asserted structurally, on the source.

Measured 2026-08-17, same 9 (radius, z) scenes, oracle-masked focus curve,
256x256, 401 steps over -5..45 mm:

| metric      | ASM fixture (noise=0) | independent (full detector chain) |
|-------------|----------------------|-----------------------------------|
| laplacian   | 0.625 mm / 33%       | 0.375 mm / 56%                    |
| phase_var   | 0.250 mm / 56%       | 0.375 mm / 56%                    |
| tenengrad   | 0.375 mm / 56%       | 0.375 mm / 67%                    |

The oracle ceiling survives removing the inverse crime, which is what makes it
a real ceiling rather than an artefact of ASM inverting ASM.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.autofocus.metrics import FocusMetric, _calc_metric  # noqa: E402
from core.offaxis import OffAxisParams, extract_complex_field_offaxis  # noqa: E402
from core.reconstruction import (  # noqa: E402
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)
from fixtures.independent_hologram import (  # noqa: E402
    IndependentConfig,
    IndependentSphere,
    build_independent_hologram,
    rayleigh_sommerfeld_sum,
)

_FIXTURE = ROOT / "tests" / "fixtures" / "independent_hologram.py"


def test_generator_imports_nothing_from_core():
    """The independence guarantee, asserted on the source rather than output.

    If this ever fails, the fixture has been quietly re-coupled to the code it
    is supposed to be independent of, and every number measured with it is
    back to describing ASM inverting ASM.
    """
    tree = ast.parse(_FIXTURE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [
        name for name in imported
        if name == "core" or name.startswith("core.")
        or name.endswith("synthetic_hologram")
    ]
    assert not offenders, (
        f"independent generator imported {offenders}; it must share no "
        "propagation code with the reconstruction path"
    )
    # And no FFT: the reconstruction is an FFT transfer-function method, so
    # reaching for one here is the first step back toward a shared operator.
    source = _FIXTURE.read_text(encoding="utf-8")
    assert "np.fft" not in source, "direct summation must not use an FFT"


def _sphere_field(cfg: IndependentConfig, *, radius_m: float, z_m: float):
    from fixtures.independent_hologram import _sphere_sources

    rng = np.random.default_rng(0)
    src_x, src_y, src_amp = _sphere_sources(
        IndependentSphere(radius_m=radius_m, z_m=z_m), cfg, rng)
    return rayleigh_sommerfeld_sum(src_x, src_y, src_amp, z_m, cfg)


def test_object_modulation_is_physically_scaled():
    """Guards a real bug: double-scaling the already-physical RS amplitude.

    The Rayleigh-Sommerfeld sum returns the PHYSICAL scattered amplitude. The
    ASM fixture instead propagates a transmittance of order 1 and multiplies by
    1/50 to represent a weak object. Copying that 1/50 across (as the first
    draft of this fixture did, 2026-08-17) put the object at ~2 counts of 4095
    against ~30 counts of shot noise, and every focus metric failed by 20-30 mm.

    A 10 um sphere at 12 mm scatters ~1% of the incident amplitude, so the
    modulation must land in a band that is neither buried nor dominant.
    """
    cfg = IndependentConfig(shot_noise=False, read_noise_e=0.0, bit_depth=None)
    field = _sphere_field(cfg, radius_m=10e-6, z_m=12e-3)
    modulation = float(np.abs(field).max()) * cfg.object_amplitude
    assert 1e-3 < modulation < 0.5, (
        f"scattered amplitude {modulation:.5f} of the incident wave is outside "
        "the physically plausible band — check for double-scaling"
    )


def _reconstructed_focus_error_mm(radius_um: float, z_mm: float, seed: int):
    """RS forward -> off-axis extract -> ASM back, oracle-masked focus curve."""
    n = 256
    cfg = IndependentConfig(shape=(n, n))
    result = build_independent_hologram(
        (IndependentSphere(radius_m=radius_um * 1e-6, z_m=z_mm * 1e-3),),
        cfg, rng=np.random.default_rng(seed),
    )
    radius_px = max(8, int(round(n / 5)))
    freq = 1.0 / (cfg.carrier_period_px * cfg.pixel_m * np.sqrt(2.0))
    offset = int(round(freq * n * cfg.pixel_m))
    field, _ = extract_complex_field_offaxis(
        result.hologram,
        OffAxisParams(center_yx=(n // 2 + offset, n // 2 + offset),
                      radius=radius_px),
    )
    half = int(round((radius_um + 4) * 1e-6 / cfg.pixel_m)) + 4
    centre = n // 2
    roi = (slice(centre - half, centre + half + 1),
           slice(centre - half, centre + half + 1))

    z_values = np.linspace(-5e-3, 45e-3, 201)
    curve = []
    for z in z_values:
        recon = propagate(
            field,
            ReconstructionParams(cfg.wavelength_m, cfg.pixel_m, float(z),
                                 cfg.n_medium),
            ReconstructionMethod.ASM,
        )
        curve.append(_calc_metric(recon[roi], FocusMetric.PHASE_VARIANCE))
    peak = z_values[int(np.argmax(curve))] * 1e3
    return peak - z_mm, result


def test_independent_hologram_reconstructs_to_the_true_plane():
    """Physical consistency: a different forward model must still invert.

    RS forward and ASM backward are different approximations of the same
    diffraction physics. If the generator were wrong, the reconstruction would
    not focus anywhere near the true plane.
    """
    error_mm, result = _reconstructed_focus_error_mm(10.0, 12.0, seed=7004)
    assert abs(error_mm) < 1.625, (
        f"independent scene focused {error_mm:+.3f} mm from truth, beyond the "
        "measured axial response of this rig"
    )
    assert result.source_count > 100
    assert result.saturated_fraction < 0.05


def test_detector_chain_is_actually_applied():
    """Shot noise, saturation and quantisation must all be live.

    The ASM fixture models only Gaussian read noise; if these silently no-op,
    the 'independent' generator is only independent in its propagator.
    """
    spheres = (IndependentSphere(radius_m=10e-6, z_m=12e-3),)
    quiet = build_independent_hologram(
        spheres,
        IndependentConfig(shot_noise=False, read_noise_e=0.0, bit_depth=None),
        rng=np.random.default_rng(3),
    ).hologram
    noisy = build_independent_hologram(
        spheres,
        IndependentConfig(shot_noise=True, read_noise_e=6.0, bit_depth=None),
        rng=np.random.default_rng(3),
    ).hologram
    assert noisy.std() > quiet.std(), "shot noise did not change the frame"

    quantised = build_independent_hologram(
        spheres, IndependentConfig(bit_depth=12), rng=np.random.default_rng(3),
    ).hologram
    assert quantised.max() <= 4095.0
    assert np.allclose(quantised, np.round(quantised)), "not quantised to ADU"


def test_empty_scene_is_refused():
    with pytest.raises(ValueError):
        build_independent_hologram((), IndependentConfig())
