"""Multi-candidate QPI batch.

When :func:`core.autofocus.find_focus_candidates` turns up several
plausible focus planes in a multi-object scene, running QPI *once*
against the landscape's best-z discards information about the other
objects. This module runs QPI against **each** candidate separately
and hands back a list of ``(candidate, QPIResult)`` pairs — the
biologist can then compare dry mass / morphology / roughness across
focus planes without repeating the whole reconstruction pipeline by
hand.

The loop is intentionally synchronous. Each candidate costs one
reconstruction + one unwrap + one ``compute_qpi`` call — at 256×256
that's on the order of 50–80 ms, so five candidates finish in well
under a second. A future UI wrapper can move this to a worker; core
stays blocking and testable.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .autofocus import FocusCandidate
from .fft_backend import get_best_fft_backend
from .phase_unwrap import UnwrapConfig, UnwrapMethod, unwrap_phase_advanced
from .qpi import QPIMode, QPIResult, compute_qpi
from .qpi_export import qpi_result_to_row
from .reconstruction import (
    ReconstructionMethod,
    ReconstructionParams,
    propagate,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class QPIBatchEntry:
    """One candidate's QPI run.

    ``candidate`` carries the z / prominence that got us here;
    ``qpi_result`` is the QPIResult that ``compute_qpi`` produced at
    that plane.
    """
    candidate: FocusCandidate
    qpi_result: QPIResult


def run_qpi_for_candidates(
    field: np.ndarray,
    base_params: ReconstructionParams,
    method: ReconstructionMethod,
    candidates: Sequence[FocusCandidate],
    *,
    n_sample: float = 1.38,
    n_medium: float = 1.337,
    mode: QPIMode = QPIMode.BOTH,
    cell_threshold: float = 0.15,
    compute_psd: bool = False,
    unwrap_method: UnwrapMethod = UnwrapMethod.GRADIENT_INTEGRATION,
) -> List[QPIBatchEntry]:
    """Run QPI once per candidate, return a list in the same order.

    Each candidate's ``z_m`` overrides ``base_params.z_m`` for its own
    reconstruction. Phase unwrap is done with
    :mod:`core.phase_unwrap`'s gradient-integration method — the same
    code path the app uses for its production QPI runs, so the result
    is comparable with a single-focus run at the same z.
    """
    if not candidates:
        return []

    fft = get_best_fft_backend()
    unwrap_cfg = UnwrapConfig(method=unwrap_method)
    out: List[QPIBatchEntry] = []

    for cand in candidates:
        params = ReconstructionParams(
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            z_m=float(cand.z_m),
            n=base_params.n,
        )
        recon = propagate(field, params, method, fft=fft, force_python=True)
        wrapped = np.angle(recon).astype(np.float32, copy=False)
        try:
            unwrapped = unwrap_phase_advanced(
                wrapped, unwrap_cfg,
                complex_field=recon,
                wavelength_m=base_params.wavelength_m,
                pixel_size_m=base_params.pixel_size_m,
            )
        except Exception:
            _LOG.warning(
                "qpi-batch: unwrap failed for candidate z=%.3f mm, "
                "falling back to wrapped phase",
                cand.z_m * 1e3,
                exc_info=True,
            )
            unwrapped = wrapped

        qpi_result = compute_qpi(
            unwrapped,
            wavelength_m=base_params.wavelength_m,
            pixel_size_m=base_params.pixel_size_m,
            mode=mode,
            n_sample=n_sample,
            n_medium=n_medium,
            cell_threshold=cell_threshold,
            compute_psd=compute_psd,
        )
        out.append(QPIBatchEntry(candidate=cand, qpi_result=qpi_result))

    return out


# ---------------------------------------------------------------------------
# CSV export — one row per entry, reuses core.qpi_export column set + extras
# ---------------------------------------------------------------------------

_EXTRA_COLUMNS = (
    "candidate_rank",
    "candidate_z_mm",
    "candidate_prominence",
)


def qpi_batch_to_rows(
    entries: Sequence[QPIBatchEntry],
    *,
    sample_id: str = "",
    app_version: str = "",
) -> List[dict]:
    """Flatten each entry into a row dict compatible with
    :func:`core.qpi_export.qpi_result_to_row` plus three batch-specific
    columns (``candidate_rank``, ``candidate_z_mm``,
    ``candidate_prominence``)."""
    rows: List[dict] = []
    for entry in entries:
        base = qpi_result_to_row(
            entry.qpi_result, sample_id=sample_id, app_version=app_version,
        )
        base["candidate_rank"] = entry.candidate.rank
        base["candidate_z_mm"] = f"{entry.candidate.z_m * 1e3:.4f}"
        base["candidate_prominence"] = f"{entry.candidate.prominence:.4g}"
        rows.append(base)
    return rows


def write_qpi_batch_csv(
    path,
    entries: Sequence[QPIBatchEntry],
    *,
    sample_id: str = "",
    app_version: str = "",
) -> None:
    """Write every entry to one CSV — one row per candidate.

    Columns: the stable set from :mod:`core.qpi_export` (sample_id,
    timestamp, total_dry_mass_pg, cell-morph fields, roughness) plus
    three trailing batch columns. DictWriter ``extrasaction='ignore'``
    drops anything else, keeping the file rectangular.
    """
    from pathlib import Path

    from .qpi_export import _COLUMNS as _BASE_COLUMNS

    fieldnames = list(_BASE_COLUMNS) + list(_EXTRA_COLUMNS)
    rows = qpi_batch_to_rows(
        entries, sample_id=sample_id, app_version=app_version,
    )

    out = Path(path)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _LOG.info(
        "qpi-batch: wrote %s (%d rows)", out, len(rows),
    )


__all__ = [
    "QPIBatchEntry",
    "run_qpi_for_candidates",
    "qpi_batch_to_rows",
    "write_qpi_batch_csv",
]
