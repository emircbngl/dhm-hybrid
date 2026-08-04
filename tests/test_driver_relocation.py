"""Driver relocation pins (2026-07-06): ui2 → core.drivers.

The Qt-free compute drivers moved to ``core/drivers/``; the old
``ui2.reconstruction`` / ``ui2.workers`` / ``ui2.camera_feed`` paths are
sys.modules-aliasing shims. These tests pin the contract that makes the
move invisible: old and new paths are the SAME module object, so
``patch("ui2.workers.X")`` still patches the globals the driver code
actually reads, and class identity holds across both paths.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_old_and_new_paths_are_the_same_module():
    import core.drivers.camera_feed as new_cam
    import core.drivers.reconstruction as new_rec
    import core.drivers.workers as new_wrk
    import ui2.camera_feed as old_cam
    import ui2.reconstruction as old_rec
    import ui2.workers as old_wrk

    assert old_rec is new_rec
    assert old_wrk is new_wrk
    assert old_cam is new_cam
    # Parent-package attribute rebound to the alias too.
    import ui2
    assert ui2.workers is new_wrk


def test_class_identity_across_paths():
    from core.drivers.reconstruction import ReconParams as NewParams
    from ui2.reconstruction import ReconParams as OldParams
    assert OldParams is NewParams


def test_patch_on_old_path_reaches_driver_globals():
    """The whole point of the aliasing shim: dozens of existing tests
    patch("ui2.workers.<name>"); those patches must hit the module globals
    the relocated driver code executes against."""
    import core.drivers.workers as drivers_workers

    with patch("ui2.workers.load_any", return_value="SENTINEL"):
        assert drivers_workers.load_any("whatever") == "SENTINEL"


def test_repo_root_survived_the_deeper_move():
    """B-096: the module moved one directory deeper but _REPO_ROOT kept
    parents[2] → it resolved to <repo>/src, the Track C checkpoint path
    went stale, and reffree_cnn_available() silently returned False even
    with a valid checkpoint installed."""
    import core.drivers.workers as w
    assert w._REPO_ROOT == ROOT, (
        f"_REPO_ROOT resolves to {w._REPO_ROOT}, expected repo root {ROOT}")
    # The checkpoint path must point under <repo>/models, never <repo>/src.
    assert w._TRACK_C_CHECKPOINT == ROOT / "models" / "track_c" / "v0.1" / "model.pt"


def test_camera_feed_import_stays_light():
    """B-097: the eager core/drivers/__init__ dragged matplotlib + the whole
    science stack into `import core.drivers.camera_feed` (numpy-only on its
    own). The package init must stay lazy (PEP 562), like ui2/__init__."""
    import subprocess
    import sys as _sys
    code = (
        "import sys; sys.path.insert(0, r'" + str(ROOT / "src") + "');"
        "import core.drivers.camera_feed;"
        "heavy = [m for m in ('matplotlib', 'skimage', 'core.drivers.workers',"
        " 'core.report', 'core.qpi') if m in sys.modules];"
        "assert not heavy, f'camera_feed import dragged in {heavy}';"
        "print('light')"
    )
    out = subprocess.run([_sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "light" in out.stdout


def test_lazy_reexports_still_resolve():
    """The convenience API survives the lazy init."""
    import core.drivers as d
    from core.drivers.workers import ScienceDriver as Direct
    assert d.ScienceDriver is Direct
    assert d.ReconParams().af_algorithm == "robust"


def test_ui3_and_core_no_longer_import_ui2():
    """Layering pin: nothing under src/ui3 or src/core imports the ui2
    package anymore (that was the retirement blocker + an inverted
    dependency in core.cameras.synthetic)."""
    import re
    pattern = re.compile(r"^\s*(from ui2[.\s]|import ui2)", re.MULTILINE)
    offenders = []
    for base in (ROOT / "src" / "ui3", ROOT / "src" / "core"):
        for py in base.rglob("*.py"):
            if pattern.search(py.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(py.relative_to(ROOT)))
    assert offenders == [], f"ui2 imports crept back in: {offenders}"
