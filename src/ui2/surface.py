"""3D surface preview via matplotlib subprocess.

Dear PyGui's built-in plot engine is 2D-only, and we don't want to
pull in pyvista or OpenGL boilerplate just to show a height-map. The
compromise: write the height array to a tmp ``.npz``, spawn
``python -m ui2.surface`` which opens a standalone matplotlib 3D
window. No shared process, no blocking the UI loop.

Usage (from the app):

    from .surface import open_surface
    open_surface(height_array, title="Phase (rad)")
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_LOG = logging.getLogger(__name__)


def open_surface(height: np.ndarray, *, title: str = "Surface") -> None:
    """Spawn a detached matplotlib window showing ``height`` as a 3D
    surface. Returns immediately; the subprocess lives independently."""
    arr = np.asarray(height, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"surface input must be 2D (got {arr.shape})")

    tmp = Path(tempfile.mkdtemp(prefix="dhm_surface_"))
    payload = tmp / "data.npz"
    np.savez_compressed(payload, height=arr, title=title)

    here = Path(__file__).resolve()
    # Child script = this module's __main__ path, invoked as a file
    # so the child doesn't need ``ui2`` on sys.path.
    cmd = [sys.executable, str(here), "--render", str(payload)]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except Exception:
        _LOG.exception("surface subprocess failed")


def _render_from_payload(payload: Path) -> int:
    """Run inside the subprocess — load the .npz and show a 3D plot."""
    try:
        import matplotlib
        matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available — install it to view surfaces.",
              file=sys.stderr)
        return 1

    data = np.load(payload, allow_pickle=True)
    height = np.asarray(data["height"], dtype=np.float32)
    title = str(data["title"]) if "title" in data.files else "Surface"

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ny, nx = height.shape
    ys, xs = np.mgrid[:ny, :nx]
    # Downsample for responsiveness: matplotlib's 3d gets slow above
    # ~256×256. Preserve detail where the user has zoomed.
    stride = max(1, max(ny, nx) // 200)
    ax.plot_surface(
        xs[::stride, ::stride],
        ys[::stride, ::stride],
        height[::stride, ::stride],
        cmap="viridis",
        linewidth=0, antialiased=True,
    )
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel("value")
    plt.tight_layout()
    plt.show()

    # Clean up the temp payload dir once the window closes. Log loudly
    # on failure — v2.0.1 silently swallowed cleanup errors, letting
    # /tmp accumulate dhm_surface_* directories over long sessions.
    try:
        os.remove(payload)
    except Exception as exc:
        print(f"warning: couldn't remove {payload}: {exc}", file=sys.stderr)
    try:
        os.rmdir(payload.parent)
    except Exception as exc:
        print(f"warning: couldn't remove tmp dir {payload.parent}: {exc}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--render":
        raise SystemExit(_render_from_payload(Path(sys.argv[2])))
    print("usage: python surface.py --render <payload.npz>",
          file=sys.stderr)
    sys.exit(2)


__all__ = ["open_surface"]
