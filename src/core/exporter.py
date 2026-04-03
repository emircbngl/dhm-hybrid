from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ExportPaths:
    root: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


def save_npy(path: Path, arr: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(arr))
    return path


def save_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def save_tiff(path: Path, img: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tifffile is required for TIFF export. Install it with: pip install tifffile"
        ) from e

    tifffile.imwrite(str(path), np.array(img))
    return path


def save_png(path: Path, img: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(img)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        x = arr.astype(np.float32, copy=False)
        x = x - float(np.min(x))
        mx = float(np.max(x))
        if mx > 0:
            x = x / mx
        out = (x * 255.0).clip(0, 255).astype(np.uint8)
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        out = arr
        mode = None
    else:
        raise ValueError(f"Unsupported image shape for PNG export: {arr.shape}")

    try:
        from PIL import Image  # type: ignore

        im = Image.fromarray(out, mode=mode) if mode is not None else Image.fromarray(out)
        im.save(str(path))
        return path
    except ModuleNotFoundError:
        try:
            import imageio.v2 as imageio  # type: ignore

            imageio.imwrite(str(path), out)
            return path
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "PNG export requires Pillow or imageio. Install with: pip install pillow"
            ) from e


def save_csv(path: Path, table: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), np.array(table), delimiter=",")
    return path


def save_mat(path: Path, mapping: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    from scipy.io import savemat

    savemat(str(path), {k: np.array(v) for k, v in mapping.items()})
    return path


def default_export_root(base: Optional[Path] = None) -> Path:
    if base is None:
        base = Path.cwd()
    return base / "output"
