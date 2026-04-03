from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .metadata_reader import read_embedded_metadata


class UnsupportedFormatError(RuntimeError):
    pass


class OptionalDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedData:
    path: Path
    array: np.ndarray
    metadata: Dict[str, Any]


def _normalize_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_npy(path: Union[str, Path]) -> LoadedData:
    p = _normalize_path(path)
    arr = np.load(p)
    return LoadedData(path=p, array=np.asarray(arr), metadata={})


def load_png(path: Union[str, Path]) -> LoadedData:
    p = _normalize_path(path)
    try:
        import imageio.v3 as iio  # type: ignore
    except ModuleNotFoundError as e:
        raise OptionalDependencyError(
            "imageio is required for PNG loading. Install it with: pip install imageio"
        ) from e

    arr = iio.imread(p)
    return LoadedData(path=p, array=np.asarray(arr), metadata={})


def load_tiff(path: Union[str, Path]) -> LoadedData:
    p = _normalize_path(path)
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as e:
        raise OptionalDependencyError(
            "tifffile is required for TIFF loading. Install it with: pip install tifffile"
        ) from e

    try:
        try:
            arr = tifffile.imread(p)
        except ValueError as e:
            msg = str(e)
            if "requires the 'imagecodecs' package" in msg:
                try:
                    import imagecodecs  # type: ignore  # noqa: F401
                except ModuleNotFoundError as dep_e:
                    raise OptionalDependencyError(
                        "This TIFF uses LZW compression and requires imagecodecs. Install it with: pip install imagecodecs"
                    ) from dep_e

                arr = tifffile.imread(p)
            else:
                raise
    except Exception:
        try:
            from PIL import Image  # type: ignore
        except ModuleNotFoundError as e:
            raise OptionalDependencyError(
                "Failed to decode TIFF via tifffile. Install Pillow for a fallback decoder: pip install pillow"
            ) from e

        with Image.open(p) as im:
            frames = []
            try:
                i = 0
                while True:
                    im.seek(i)
                    frames.append(np.array(im))
                    i += 1
            except EOFError:
                pass

            if not frames:
                raise RuntimeError(f"Failed to decode TIFF: {p}")

            arr = frames[0] if len(frames) == 1 else np.stack(frames, axis=0)

    meta: Dict[str, Any] = {}
    try:
        with tifffile.TiffFile(p) as tf:
            meta.update({"pages": len(tf.pages)})
            if tf.pages:
                page = tf.pages[0]
                tags = page.tags
                for key in ["XResolution", "YResolution", "ResolutionUnit", "ImageDescription"]:
                    if key in tags:
                        try:
                            meta[key] = tags[key].value
                        except Exception:
                            pass
    except Exception:
        pass

    acq = read_embedded_metadata(p, extra=meta)
    if acq.wavelength_m is not None:
        meta["wavelength_m"] = acq.wavelength_m
    if acq.pixel_size_m is not None:
        meta["pixel_size_m"] = acq.pixel_size_m
    if acq.magnification is not None:
        meta["magnification"] = acq.magnification
    if acq.source is not None:
        meta["metadata_source"] = acq.source

    return LoadedData(path=p, array=np.asarray(arr), metadata=meta)


def load_video(path: Union[str, Path], frame_index: int = 0) -> LoadedData:
    p = _normalize_path(path)
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as e:
        raise OptionalDependencyError(
            "opencv-python is required for video loading. Install it with: pip install opencv-python"
        ) from e

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {p}")

    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_index} from {p}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        meta = {
            "frame_index": frame_index,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        return LoadedData(path=p, array=np.asarray(gray), metadata=meta)
    finally:
        cap.release()


def load_any(path: Union[str, Path]) -> LoadedData:
    p = _normalize_path(path)
    suffix = p.suffix.lower()

    if suffix == ".npy":
        return load_npy(p)
    if suffix in {".tif", ".tiff"}:
        return load_tiff(p)
    if suffix == ".png":
        return load_png(p)
    if suffix in {".mp4", ".avi", ".mov"}:
        return load_video(p)

    raise UnsupportedFormatError(f"Unsupported file format: {suffix}")
