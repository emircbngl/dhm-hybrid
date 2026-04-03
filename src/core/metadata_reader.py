from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class AcquisitionMetadata:
    wavelength_m: Optional[float] = None
    pixel_size_m: Optional[float] = None
    magnification: Optional[float] = None
    source: Optional[str] = None


def _as_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            num, den = value
            return float(num) / float(den)
        return float(value)
    except Exception:
        return None


def _pixel_size_from_resolution(
    x_res: Any, y_res: Any, unit: Any
) -> Optional[float]:
    x = _as_float(x_res)
    y = _as_float(y_res)
    if x is None or y is None:
        return None

    unit_code = None
    try:
        unit_code = int(unit)
    except Exception:
        pass

    if unit_code == 2:
        meters_per_unit = 0.0254
    elif unit_code == 3:
        meters_per_unit = 0.01
    else:
        return None

    px_m = meters_per_unit / x
    py_m = meters_per_unit / y
    return float((px_m + py_m) / 2.0)


def _parse_wavelength_m(text: str) -> Optional[float]:
    patterns = [
        r"wavelength\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(nm|um|µm|mm|m)",
        r"lambda\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(nm|um|µm|mm|m)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "nm":
            return val * 1e-9
        if unit in {"um", "µm"}:
            return val * 1e-6
        if unit == "mm":
            return val * 1e-3
        if unit == "m":
            return val
    return None


def _parse_pixel_size_m(text: str) -> Optional[float]:
    patterns = [
        r"pixel\s*size\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(nm|um|µm|mm|m)",
        r"pixelsize\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(nm|um|µm|mm|m)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "nm":
            return val * 1e-9
        if unit in {"um", "µm"}:
            return val * 1e-6
        if unit == "mm":
            return val * 1e-3
        if unit == "m":
            return val
    return None


def _parse_magnification(text: str) -> Optional[float]:
    m = re.search(r"(magnification|mag)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*x", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(2))
    m = re.search(r"\b([0-9]*\.?[0-9]+)\s*x\b", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def read_embedded_metadata(path: Union[str, Path], extra: Optional[Dict[str, Any]] = None) -> AcquisitionMetadata:
    p = path if isinstance(path, Path) else Path(path)
    extra = extra or {}

    wavelength_m: Optional[float] = None
    pixel_size_m: Optional[float] = None
    magnification: Optional[float] = None
    source: Optional[str] = "file_header"

    if p.suffix.lower() in {".tif", ".tiff"}:
        pixel_size_m = _pixel_size_from_resolution(
            extra.get("XResolution"), extra.get("YResolution"), extra.get("ResolutionUnit")
        )

        desc = extra.get("ImageDescription")
        if isinstance(desc, (bytes, bytearray)):
            try:
                desc = desc.decode("utf-8", errors="ignore")
            except Exception:
                desc = None

        if isinstance(desc, str):
            wavelength_m = wavelength_m or _parse_wavelength_m(desc)
            pixel_size_m = pixel_size_m or _parse_pixel_size_m(desc)
            magnification = magnification or _parse_magnification(desc)

    return AcquisitionMetadata(
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_size_m,
        magnification=magnification,
        source=source,
    )
