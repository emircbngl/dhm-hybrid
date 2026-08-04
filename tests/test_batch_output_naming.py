"""Batch output naming + auto-pair setup-filter regressions (2026-07-05 review).

Bug 1: ``_do_save`` used ``Path.with_suffix(".amp.tiff")`` on base names that
carry a decimal z ("holo_Z_0.5000mm"), so ``.5000mm`` was treated as the
extension and stripped — every same-integer-z slice of a sweep collapsed onto
one filename and silently overwrote. Fixed by appending the channel suffix.

Bug 2: with ``auto_pair_reference`` on (default), setup() dropped every
ref-named input from the job list even when the caller supplied an explicit
reference (making the filter pure data loss), and it never told the user.
Fixed: explicit reference disables the setup filter; dropping emits status.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("PySide6")


def _ensure_qt_app():
    from PySide6.QtCore import QCoreApplication
    if QCoreApplication.instance() is None:
        QCoreApplication([])


def _write_carrier_holo(path: Path, seed: int = 3) -> None:
    import numpy as np
    import tifffile

    rng = np.random.default_rng(seed)
    h = w = 128
    yy, xx = np.indices((h, w))
    carrier = np.cos(2 * np.pi * ((xx + yy) / 8.0))
    holo = (1 + 0.4 * carrier
            + 0.02 * rng.standard_normal((h, w))).astype(np.float32)
    tifffile.imwrite(path, (holo * 1000).astype(np.uint16))


def _base_cfg(in_dir: Path, out_dir: Path, z_mm: float) -> dict:
    return {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "mode": "Iterate files with active Profile",
        "active_state": {
            "recon": {
                "wavelength": 632.8, "pixel_size": 3.45,
                "pixel_is_effective": True, "n": 1.0,
                "mask_radius": 30, "method": "ASM", "z_mm": z_mm,
            },
            "process": {"subtract_mean": True},
        },
        "save_amp": True, "save_pha": False,
        "export_csv": False, "export_report": False,
    }


def test_decimal_z_survives_in_output_filename(tmp_path):
    """z=0.5 mm must yield '..._Z_0.5000mm.amp.tiff' (full decimal kept)."""
    pytest.importorskip("tifffile")
    _ensure_qt_app()
    from core.batch_renderer import BatchRenderer

    in_dir = tmp_path / "in"; in_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir()
    _write_carrier_holo(in_dir / "holo.tif")

    renderer = BatchRenderer()
    renderer.setup(_base_cfg(in_dir, out_dir, z_mm=0.5), profile_manager=None)
    for job in renderer.jobs:
        renderer._process_single_job(job, writer=None)

    names = sorted(p.name for p in out_dir.glob("*.amp.tiff"))
    assert names == ["holo_Z_0.5000mm.amp.tiff"], names


def test_same_integer_z_values_do_not_collide(tmp_path):
    """Two z's sharing the integer part (0.5, 0.7 mm) → two distinct files.

    Pre-fix, with_suffix truncated both to 'holo_Z_0.amp.tiff' and the
    second overwrote the first.
    """
    pytest.importorskip("tifffile")
    _ensure_qt_app()
    from core.batch_renderer import BatchRenderer

    in_dir = tmp_path / "in"; in_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir()
    _write_carrier_holo(in_dir / "holo.tif")

    for z_mm in (0.5, 0.7):
        renderer = BatchRenderer()
        renderer.setup(_base_cfg(in_dir, out_dir, z_mm=z_mm),
                       profile_manager=None)
        for job in renderer.jobs:
            renderer._process_single_job(job, writer=None)

    names = sorted(p.name for p in out_dir.glob("*.amp.tiff"))
    assert names == [
        "holo_Z_0.5000mm.amp.tiff",
        "holo_Z_0.7000mm.amp.tiff",
    ], names


def test_explicit_reference_disables_setup_ref_filter(tmp_path):
    """With an explicit reference in cfg, ref-named inputs stay in the
    job list (per-job auto-pair is skipped anyway, so dropping them was
    pure silent data loss for samples like 'blood_ref.tif')."""
    import numpy as np
    _ensure_qt_app()
    from core.batch_renderer import BatchRenderer

    in_dir = tmp_path / "in"; in_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir()
    for name in ("sample.tif", "blood_ref.tif", "ref_series01.tif"):
        (in_dir / name).write_bytes(b"")

    cfg = _base_cfg(in_dir, out_dir, z_mm=0.0)
    cfg["reference_fc"] = np.ones((8, 8), dtype=np.complex64)

    renderer = BatchRenderer()
    renderer.setup(cfg, profile_manager=None)
    job_names = sorted(j.in_file.name for j in renderer.jobs)
    assert job_names == ["blood_ref.tif", "ref_series01.tif", "sample.tif"]


def test_auto_pair_drop_is_announced(tmp_path):
    """Without an explicit reference the ref-named files are still
    filtered (lab convention) but the drop is surfaced via status."""
    _ensure_qt_app()
    from core.batch_renderer import BatchRenderer

    in_dir = tmp_path / "in"; in_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir()
    for name in ("sample.tif", "sample_ref.tif"):
        (in_dir / name).write_bytes(b"")

    renderer = BatchRenderer()
    messages: list[str] = []
    renderer.status.connect(messages.append)
    renderer.setup(_base_cfg(in_dir, out_dir, z_mm=0.0),
                   profile_manager=None)

    job_names = [j.in_file.name for j in renderer.jobs]
    assert job_names == ["sample.tif"]
    assert any("sample_ref.tif" in m for m in messages), messages
