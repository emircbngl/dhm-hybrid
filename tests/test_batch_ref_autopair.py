"""Reference auto-pair tests for ``core.batch_renderer``.

Lab convention (2026-04-30): each sample recording ``<name>.<ext>`` ships
with a paired reference acquired without the sample, named one of:

    <name>_ref.<ext>   <name>-ref.<ext>
    ref_<name>.<ext>   ref-<name>.<ext>

(case-insensitive). The renderer:
  1. Skips reference files when iterating an input directory.
  2. Discovers the paired reference for each sample at job time.

Both behaviours are covered here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pull helpers without booting Qt — they're module-level functions.
pytest.importorskip("PySide6")
from core.batch_renderer import _find_reference_for, _is_reference_filename


@pytest.mark.parametrize("name", [
    "holo_001_ref.tif",
    "holo_001_REF.tif",
    "holo_001-Ref.tiff",
    "ref_holo_001.tif",
    "REF_holo_001.tif",
    "Ref-holo_001.png",
])
def test_is_reference_filename_true(tmp_path, name):
    assert _is_reference_filename(tmp_path / name)


@pytest.mark.parametrize("name", [
    "holo_001.tif",
    "sample.tiff",
    "reference_lookup.tif",      # 'reference', not 'ref'
    "deref_001.tif",             # 'deref' starts with 'der', not 'ref_'
])
def test_is_reference_filename_false(tmp_path, name):
    assert not _is_reference_filename(tmp_path / name)


def test_find_reference_suffix(tmp_path):
    sample = tmp_path / "holo_001.tif"
    ref = tmp_path / "holo_001_ref.tif"
    sample.write_bytes(b"")
    ref.write_bytes(b"")
    assert _find_reference_for(sample) == ref


def test_find_reference_prefix(tmp_path):
    sample = tmp_path / "holo_001.tif"
    ref = tmp_path / "ref_holo_001.tif"
    sample.write_bytes(b"")
    ref.write_bytes(b"")
    assert _find_reference_for(sample) == ref


def test_find_reference_dash_separator(tmp_path):
    sample = tmp_path / "scan_a.png"
    ref = tmp_path / "scan_a-ref.png"
    sample.write_bytes(b"")
    ref.write_bytes(b"")
    assert _find_reference_for(sample) == ref


def test_find_reference_case_variants(tmp_path):
    """Uppercase ref file is discoverable.

    Compares basenames case-insensitively because macOS APFS / HFS+ are
    case-insensitive by default — ``holo_ref.tif`` and ``holo_REF.tif``
    collide as the same dirent. The matcher's job is to *find* the
    file; the on-disk case is whatever the filesystem chose to keep.
    """
    sample = tmp_path / "holo.tif"
    ref = tmp_path / "holo_REF.tif"
    sample.write_bytes(b"")
    ref.write_bytes(b"")
    found = _find_reference_for(sample)
    assert found is not None
    assert found.name.lower() == "holo_ref.tif"


def test_find_reference_none_when_missing(tmp_path):
    sample = tmp_path / "holo_orphan.tif"
    sample.write_bytes(b"")
    assert _find_reference_for(sample) is None


def test_ref_file_does_not_self_pair(tmp_path):
    """A reference file shouldn't try to find its own reference."""
    ref = tmp_path / "holo_ref.tif"
    ref.write_bytes(b"")
    assert _find_reference_for(ref) is None


def test_extension_must_match(tmp_path):
    """Different extensions are different files — don't cross-match."""
    sample = tmp_path / "holo.tif"
    fake_ref = tmp_path / "holo_ref.png"  # extension differs
    sample.write_bytes(b"")
    fake_ref.write_bytes(b"")
    assert _find_reference_for(sample) is None


def test_auto_pair_changes_reconstruction(tmp_path):
    """End-to-end: a paired ref alongside the sample changes the saved
    amplitude vs. the same sample run alone.

    The numerical content of the change isn't what we test — that's
    what the parity tests cover. Here we just prove the auto-pair
    machinery actually wires the discovered reference into ``_apply_ref``,
    so the saved file changes when a sibling ``_ref`` exists.
    """
    pytest.importorskip("PySide6")
    pytest.importorskip("tifffile")

    import numpy as np
    import tifffile

    from PySide6.QtCore import QCoreApplication
    if QCoreApplication.instance() is None:
        QCoreApplication([])

    from core.batch_renderer import BatchRenderer

    rng = np.random.default_rng(11)
    h = w = 192
    yy, xx = np.indices((h, w))
    fx = fy = 1.0 / 8.0
    carrier = np.cos(2 * np.pi * (fx * xx + fy * yy))
    sample = (1 + 0.5 * np.cos(2 * np.pi * (fx * xx + fy * yy)
                                + 0.3 * np.sin(2 * np.pi * 3 * yy / h))
              + 0.3 * carrier).astype(np.float32)
    ref = (1 + 0.3 * carrier
           + 0.05 * rng.standard_normal((h, w)).astype(np.float32))

    # Two separate input dirs so we can compare paired vs. orphan runs.
    paired_in = tmp_path / "paired_in"
    paired_in.mkdir()
    tifffile.imwrite(paired_in / "holo.tif", sample.astype(np.uint16))
    tifffile.imwrite(paired_in / "holo_ref.tif", ref.astype(np.uint16))

    orphan_in = tmp_path / "orphan_in"
    orphan_in.mkdir()
    tifffile.imwrite(orphan_in / "holo.tif", sample.astype(np.uint16))

    def _run(in_dir: Path, out_dir: Path) -> np.ndarray:
        out_dir.mkdir()
        renderer = BatchRenderer()
        renderer.setup({
            "input_dir": str(in_dir),
            "output_dir": str(out_dir),
            "mode": "Iterate files with active Profile",
            "active_state": {
                "recon": {
                    "wavelength": 632.8, "pixel_size": 3.45,
                    "pixel_is_effective": True, "n": 1.0,
                    "mask_radius": 40, "method": "ASM", "z_mm": 0.0,
                },
                "process": {"subtract_mean": True},
            },
            "save_amp": True, "save_pha": False,
            "export_csv": False, "export_report": False,
        }, profile_manager=None)
        # Sync execution — don't bother with the QThread loop.
        for job in renderer.jobs:
            renderer._process_single_job(job, writer=None)
        amp_path = next(out_dir.glob("*.amp.tiff"))
        return tifffile.imread(amp_path).astype(np.float32)

    paired_amp = _run(paired_in, tmp_path / "paired_out")
    orphan_amp = _run(orphan_in, tmp_path / "orphan_out")

    diff = np.abs(paired_amp - orphan_amp).mean()
    assert diff > 0, (
        "Auto-paired run produced an identical amplitude to the orphan run "
        "— reference division didn't activate."
    )


def _ensure_qt_app():
    from PySide6.QtCore import QCoreApplication
    if QCoreApplication.instance() is None:
        QCoreApplication([])


def _make_in_dir_with_refs(tmp_path: Path) -> Path:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for stem in ("a_001", "a_002", "a_003"):
        (in_dir / f"{stem}.tif").write_bytes(b"")
        (in_dir / f"{stem}_ref.tif").write_bytes(b"")
    (in_dir / "ref_only.tif").write_bytes(b"")  # ref without sample
    return in_dir


def test_setup_filters_ref_files(tmp_path):
    """``setup()`` must drop reference files when auto-pair is on."""
    pytest.importorskip("PySide6")
    from core.batch_renderer import BatchRenderer
    _ensure_qt_app()

    in_dir = _make_in_dir_with_refs(tmp_path)
    out_dir = tmp_path / "out"

    renderer = BatchRenderer()
    renderer.setup({
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "mode": "Iterate files with active Profile",
        "active_state": {"recon": {}, "process": {}},
        # auto_pair_reference defaults to True, but be explicit.
        "auto_pair_reference": True,
    }, profile_manager=None)

    job_names = sorted(j.in_file.name for j in renderer.jobs)
    # Three samples kept, four ref files dropped (3 paired + ref_only).
    assert job_names == ["a_001.tif", "a_002.tif", "a_003.tif"]


def test_setup_keeps_ref_files_when_toggle_off(tmp_path):
    """Toggle off → every image becomes a job (legacy behaviour)."""
    pytest.importorskip("PySide6")
    from core.batch_renderer import BatchRenderer
    _ensure_qt_app()

    in_dir = _make_in_dir_with_refs(tmp_path)
    out_dir = tmp_path / "out"

    renderer = BatchRenderer()
    renderer.setup({
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "mode": "Iterate files with active Profile",
        "active_state": {"recon": {}, "process": {}},
        "auto_pair_reference": False,
    }, profile_manager=None)

    job_names = sorted(j.in_file.name for j in renderer.jobs)
    # Without auto-pair, refs are reconstructed too — 7 files total.
    expected = sorted([
        "a_001.tif", "a_001_ref.tif",
        "a_002.tif", "a_002_ref.tif",
        "a_003.tif", "a_003_ref.tif",
        "ref_only.tif",
    ])
    assert job_names == expected


def test_toggle_off_skips_auto_demodulation(tmp_path):
    """Even with paired files on disk, toggle off must NOT divide.

    Compares amplitude with toggle on vs. off — when off, the renderer
    should produce the same output as a run with no sibling at all.
    """
    pytest.importorskip("PySide6")
    pytest.importorskip("tifffile")

    import numpy as np
    import tifffile

    _ensure_qt_app()
    from core.batch_renderer import BatchRenderer

    rng = np.random.default_rng(13)
    h = w = 192
    yy, xx = np.indices((h, w))
    fx = fy = 1.0 / 8.0
    carrier = np.cos(2 * np.pi * (fx * xx + fy * yy))
    sample = (1 + 0.5 * np.cos(2 * np.pi * (fx * xx + fy * yy)
                                + 0.3 * np.sin(2 * np.pi * 3 * yy / h))
              + 0.3 * carrier).astype(np.float32)
    ref = (1 + 0.3 * carrier
           + 0.05 * rng.standard_normal((h, w)).astype(np.float32))

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    tifffile.imwrite(in_dir / "holo.tif", sample.astype(np.uint16))
    tifffile.imwrite(in_dir / "holo_ref.tif", ref.astype(np.uint16))

    def _run(out_dir: Path, *, auto_pair: bool) -> np.ndarray:
        out_dir.mkdir()
        r = BatchRenderer()
        r.setup({
            "input_dir": str(in_dir),
            "output_dir": str(out_dir),
            "mode": "Iterate files with active Profile",
            "active_state": {
                "recon": {
                    "wavelength": 632.8, "pixel_size": 3.45,
                    "pixel_is_effective": True, "n": 1.0,
                    "mask_radius": 40, "method": "ASM", "z_mm": 0.0,
                },
                "process": {"subtract_mean": True},
            },
            "auto_pair_reference": auto_pair,
            "save_amp": True, "save_pha": False,
            "export_csv": False, "export_report": False,
        }, profile_manager=None)
        for job in r.jobs:
            if job.in_file.name == "holo.tif":
                r._process_single_job(job, writer=None)
        amp_path = next(out_dir.glob("holo*.amp.tiff"))
        return tifffile.imread(amp_path).astype(np.float32)

    paired = _run(tmp_path / "on", auto_pair=True)
    orphan = _run(tmp_path / "off", auto_pair=False)
    diff = np.abs(paired - orphan).mean()
    assert diff > 0, (
        "Toggle off path produced an identical amplitude to the auto-pair "
        "path — the toggle isn't actually gating reference division."
    )
