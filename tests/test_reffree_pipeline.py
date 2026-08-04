"""Core reference-free pipeline + layering regression (2026-07-05 refactor).

The demod / autofocus / propagate / unwrap chain used to live in
``scripts/benchmark_reffree.py`` and was imported by ``src/recon_dl/
inference.py`` — shipped code depending on a throwaway lab script (a
layering inversion), with 3-4 copies of the reference-division idiom.
It now lives in ``core.pipelines.reffree_hybrid`` and the scripts are thin
wrappers. These tests pin (a) the core pipeline runs, (b) the script
wrappers stay byte-identical to core, and (c) src/ never imports scripts/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.pipelines.reffree_hybrid import (
    LAB_OPTICS,
    OpticalConfig,
    demodulate,
    piston_align,
    preprocess_raw,
    propagate_field,
    reconstruct_in_memory,
    safe_reference_divide,
)


def _carrier_hologram(n: int = 256, fc: float = 0.3, seed: int = 5) -> np.ndarray:
    """A simple off-axis hologram: tilted reference + weak object bump."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    X, Y = np.meshgrid(x, x)
    bump = 0.4 * np.exp(-(((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    obj = (0.3 + bump) * np.exp(1j * (2 * np.pi * fc * X + bump))
    holo = np.abs(1.0 + obj) ** 2
    holo = holo + 0.01 * rng.standard_normal((n, n))
    return (holo * 1000).astype(np.uint16)


_CFG = OpticalConfig(mask_radius=30)   # small mask for the 256-px test image


def test_optical_config_from_camera():
    cfg = OpticalConfig.from_camera(camera_pixel_um=4.4, magnification=50.0)
    assert cfg.effective_pixel_m == pytest.approx(0.088e-6, rel=1e-9)
    assert LAB_OPTICS.effective_pixel_m == pytest.approx(0.088e-6, rel=1e-6)


def test_reconstruct_in_memory_reffree_runs():
    raw = _carrier_hologram()
    fc = demodulate(raw, _CFG)
    unwrapped, z_used, amp = reconstruct_in_memory(
        fc, None, _CFG, bg_method="polynomial", bg_polynomial_order=4,
        fixed_z_m=0.0,
    )
    assert unwrapped.shape == raw.shape
    assert np.all(np.isfinite(unwrapped))
    assert amp.shape == raw.shape
    assert z_used == 0.0


def test_reconstruct_in_memory_referenced_divides():
    raw = _carrier_hologram(seed=1)
    ref = _carrier_hologram(seed=2)
    fc = demodulate(raw, _CFG)
    ref_fc = demodulate(ref, _CFG)
    u_reffree, _, _ = reconstruct_in_memory(fc, None, _CFG, fixed_z_m=0.0)
    u_ref, _, _ = reconstruct_in_memory(fc, ref_fc, _CFG, fixed_z_m=0.0)
    # A reference changes the result — the division actually happens.
    assert not np.allclose(u_reffree, u_ref)


def test_safe_reference_divide_guards_zeros():
    sample = np.ones((4, 4), dtype=np.complex64) * (2 + 0j)
    ref = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    ref = np.pad(ref, ((0, 2), (0, 2)), constant_values=1)
    out = safe_reference_divide(sample, ref)
    assert np.all(np.isfinite(out))            # zeros → divide-by-1, not inf


def test_piston_align_matches_medians():
    a = np.arange(16.0).reshape(4, 4)
    b = a + 3.7
    aligned, delta = piston_align(a, b)
    assert delta == pytest.approx(3.7)
    assert np.median(aligned) == pytest.approx(np.median(b))


# ---------------------------------------------------------------------------
# Wrapper parity — the CLI scripts must delegate to the SAME core code.
# ---------------------------------------------------------------------------

def test_script_wrappers_delegate_to_core():
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark_reffree as bench
    import run_rapor_data_batch as rapor

    raw = _carrier_hologram(seed=7)

    # benchmark._demodulate(raw) == core demodulate(raw, lab optics)
    fc_bench = bench._demodulate(raw)
    fc_core = demodulate(raw, bench._OPTICS)
    assert np.array_equal(fc_bench, fc_core)

    # run_rapor._demodulate returns (fc, spectrum); fc must match core.
    fc_rapor, _spec = rapor._demodulate(raw)
    assert np.array_equal(fc_rapor, demodulate(raw, rapor._OPTICS))

    # run_rapor._propagate_reffree == core propagate_field at the same z.
    z = 0.0
    assert np.array_equal(
        rapor._propagate_reffree(fc_rapor, z),
        propagate_field(fc_rapor, z, rapor._OPTICS),
    )


def test_src_does_not_import_scripts():
    """Layering guard: nothing under src/ may import the lab scripts."""
    import re

    offenders = []
    pat = re.compile(
        r"^\s*(from|import)\s+"
        r"(benchmark_reffree|run_rapor_data_batch|build_track_c_dataset)\b",
        re.MULTILINE,
    )
    for py in (ROOT / "src").rglob("*.py"):
        if pat.search(py.read_text(encoding="utf-8")):
            offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"src/ imports lab scripts (layering inversion): {offenders}"
