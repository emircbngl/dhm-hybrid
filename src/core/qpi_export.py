"""Tabular export of :class:`~core.qpi.QPIResult` for downstream analysis.

The HTML report is for humans; this module is for R/Python pipelines. One
``QPIResult`` becomes one CSV row with every scalar field flattened into
its own column — trivial to ``pd.read_csv`` and concatenate across runs.

The design is intentionally boring: no third-party dependencies, no
file-format options, no pandas. Just ``csv.DictWriter`` so that the
lab's LIMS/ELN import jobs keep working on minimal Python installs.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

_LOG = logging.getLogger(__name__)

# Ordered column set. Adding a new scalar to QPIResult? Append here — the
# lab's import jobs will then pick it up on the next run. Do not reorder
# existing columns or you will silently shift downstream spreadsheets.
_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "timestamp_utc",
    "app_version",
    # Cell-summary scalars
    "total_dry_mass_pg",
    "step_height_nm",
    # Phase statistics
    "opd_range_nm",
    "opd_mean_nm",
    "opd_std_nm",
    # Cell morphology (single largest component in current release)
    "area_um2",
    "volume_fL",
    "dry_mass_density_pg_um2",
    "max_height_nm",
    "mean_height_nm",
    "perimeter_um",
    "circularity",
    "aspect_ratio",
    "eccentricity",
    "mean_phase_rad",
    "phase_std_rad",
    # Roughness
    "Ra_m",
    "Rq_m",
    "Rz_m",
)


def _as_dict(obj: Any) -> Mapping[str, Any]:
    """Best-effort conversion to a flat mapping. Returns ``{}`` on None."""
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return obj
    # Fall back to attribute introspection for simple namespaces.
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")
            and not callable(getattr(obj, k, None))}


def qpi_result_to_row(
    result: Any,
    *,
    sample_id: str = "",
    app_version: str = "",
    timestamp: Optional[datetime] = None,
) -> dict[str, Any]:
    """Flatten a :class:`QPIResult` into a single ``dict`` keyed by column.

    Missing fields are left out — :func:`csv.DictWriter` fills them with
    empty strings so the resulting file is still rectangular. This makes
    the function robust to partial QPI runs (e.g. microstructure-only
    mode where cell morphology is ``None``).
    """
    row: dict[str, Any] = {
        "sample_id": sample_id or "",
        "timestamp_utc": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "app_version": app_version or "",
    }

    # Flat scalars directly on the result object.
    for key in ("total_dry_mass_pg", "step_height_m"):
        val = getattr(result, key, None)
        if val is None:
            continue
        if key == "step_height_m":
            # Report in nm to stay consistent with the HTML report.
            row["step_height_nm"] = float(val) * 1e9
        else:
            row[key] = float(val)

    # phase_stats has ``range_nm``/``mean_nm``/``std_nm`` (see core.qpi).
    ph = _as_dict(getattr(result, "phase_stats", None))
    if ph:
        for src, dst in (("range_nm", "opd_range_nm"),
                         ("mean_nm", "opd_mean_nm"),
                         ("std_nm", "opd_std_nm")):
            if src in ph and ph[src] is not None:
                row[dst] = float(ph[src])

    # cell_morph is a CellMorphology dataclass — every field is already
    # a scalar in the target unit.
    cm = _as_dict(getattr(result, "cell_morph", None))
    for key in ("area_um2", "volume_fL", "dry_mass_density_pg_um2",
                "max_height_nm", "mean_height_nm", "perimeter_um",
                "circularity", "aspect_ratio", "eccentricity",
                "mean_phase_rad", "phase_std_rad"):
        if key in cm and cm[key] is not None:
            row[key] = float(cm[key])

    # roughness is a RoughnessResult — Ra/Rq/Rz in metres.
    rg = _as_dict(getattr(result, "roughness", None))
    for key in ("Ra", "Rq", "Rz"):
        if key in rg and rg[key] is not None:
            row[f"{key}_m"] = float(rg[key])

    return row


def write_qpi_csv(
    path: Union[str, Path],
    result: Any,
    *,
    sample_id: str = "",
    app_version: str = "",
    append: bool = False,
) -> Path:
    """Write a single :class:`QPIResult` to a CSV file.

    When ``append=True`` and the file already exists, the header is
    skipped so the lab can accumulate many runs into one sheet. When
    appending against an empty/missing file the header is still written,
    so the result is always a self-describing CSV.
    """
    out = Path(path)
    row = qpi_result_to_row(result, sample_id=sample_id,
                            app_version=app_version)

    write_header = not (append and out.exists() and out.stat().st_size > 0)
    mode = "a" if append else "w"

    with out.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_COLUMNS),
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    _LOG.info("qpi-export: wrote %s (%s)", out,
              "appended" if append and not write_header else "new")
    return out


__all__ = ["qpi_result_to_row", "write_qpi_csv"]
