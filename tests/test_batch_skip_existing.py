"""Skip-existing batch flag tests.

When the user resumes a cancelled batch (or runs over a partly-processed
folder), the renderer must skip files whose expected outputs already
sit in the output directory. Each ``save_*`` flag widens the set of
files we need to find before declaring "already done"; missing one
re-queues the job.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("PySide6")
pytest.importorskip("tifffile")


def _ensure_qt_app():
    from PySide6.QtCore import QCoreApplication
    if QCoreApplication.instance() is None:
        QCoreApplication([])


def _make_dummy_holos(in_dir: Path, n: int = 3) -> list[str]:
    import numpy as np
    import tifffile
    in_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n):
        name = f"holo_{i:03d}.tif"
        tifffile.imwrite(
            in_dir / name,
            (np.random.default_rng(i).random((96, 96)) * 65000).astype("uint16"),
        )
        names.append(name)
    return names


def _make_existing_outputs(out_dir: Path, names: list[str], suffix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        stem = Path(name).stem
        # Single-mode output filename pattern: <stem>_Z_<z>mm<suffix>
        (out_dir / f"{stem}_Z_0.0000mm{suffix}").write_bytes(b"FAKE")


def test_skip_existing_skips_finished_jobs(tmp_path):
    _ensure_qt_app()
    from core.batch_renderer import BatchRenderer

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    names = _make_dummy_holos(in_dir, n=3)
    _make_existing_outputs(out_dir, names, ".amp.tiff")
    _make_existing_outputs(out_dir, names, ".pha.tiff")

    renderer = BatchRenderer()
    renderer.setup({
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "mode": "Iterate files with active Profile",
        "active_state": {"recon": {}, "process": {}},
        "skip_existing": True,
    }, profile_manager=None)

    # All three jobs are queued; we test the skip predicate directly.
    assert all(renderer._job_already_finished(j) for j in renderer.jobs)


def test_skip_existing_keeps_partial_jobs(tmp_path):
    """If only amp exists and pha is requested, the job must rerun."""
    _ensure_qt_app()
    from core.batch_renderer import BatchRenderer

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    names = _make_dummy_holos(in_dir, n=2)
    _make_existing_outputs(out_dir, names, ".amp.tiff")  # pha missing

    renderer = BatchRenderer()
    renderer.setup({
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "mode": "Iterate files with active Profile",
        "active_state": {"recon": {}, "process": {}},
        "skip_existing": True,
        "save_amp": True,
        "save_pha": True,
    }, profile_manager=None)

    assert not any(renderer._job_already_finished(j) for j in renderer.jobs)


def test_skip_existing_off_runs_everything(tmp_path):
    """Toggle off → predicate isn't even consulted in run(); but the
    helper itself must still distinguish for callers that probe it.
    Sanity check: when toggle is off, default config has no override
    and everything queues."""
    _ensure_qt_app()
    from core.batch_renderer import BatchRenderer

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _make_dummy_holos(in_dir, n=2)

    renderer = BatchRenderer()
    renderer.setup({
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "mode": "Iterate files with active Profile",
        "active_state": {"recon": {}, "process": {}},
        "skip_existing": False,
    }, profile_manager=None)

    # No prior outputs — every job must run.
    assert not any(renderer._job_already_finished(j) for j in renderer.jobs)
