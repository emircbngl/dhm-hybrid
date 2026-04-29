"""NIST-traceable calibration workflow — v2.0.8 sprint, D3.

Karin's pain (Lindqvist 2026-04-27):

    > "Her hafta NIST traceable 10µm polystyrene bead ile sistem
    > doğruluk kontrolü yapmalıyım. Şu an manuel — bead hologramını
    > yüklüyorum, 'lateral 10µm okuyor mu?' diye bakıyorum.
    > Reviewer 'kalibrasyon nasıl?' dediğinde Excel kayıt
    > göstermek istemiyorum."

This module is the data layer for the calibration workflow. It:

1. Measures the apparent diameter of a focused phase sphere from
   a reconstructed phase image (or a synthetic stand-in).
2. Compares against the bead's known nominal diameter (NIST
   certificate) and computes a percent drift.
3. Persists each check to ``~/.dhm-reconstruction/users/<u>/
   calibration_history.jsonl``, audited via the central audit log.
4. Classifies the result with traffic-light thresholds the lab
   uses for go/no-go: green < 2 %, yellow 2–5 %, red > 5 %.

The DPG dialog (``ui2.dialogs.show_calibration_check``) wraps this
module — measurement + classification + record stay headless and
testable.

Traceability
------------
NIST 4000-series polystyrene bead standards have certified
diameters with ±0.1 µm uncertainty. A reviewer asking "how do you
calibrate?" gets "weekly bead check, history.jsonl, drift < 5 %"
as the answer. The CSV exporter produces a JSONL → CSV adapter so
it goes straight into a Methods supplement.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np


_LOG = logging.getLogger(__name__)


class CalibrationStatus(str, Enum):
    """Traffic-light verdict the lab uses for go/no-go."""
    GREEN = "green"     # < 2 % drift — system OK
    YELLOW = "yellow"   # 2 – 5 % drift — recheck soon, still usable
    RED = "red"         # > 5 % drift — re-calibrate before any data


@dataclass
class CalibrationCheck:
    """One bead-measurement record. JSON-serialisable for history.

    Attributes
    ----------
    timestamp
        ISO-8601 UTC string. Used as audit anchor.
    operator
        Username (from ``user_profile.current_user``).
    nominal_diameter_um
        NIST certificate value (e.g. 10.000 µm with ±0.1 µm
        uncertainty).
    measured_diameter_um
        What the lab's reconstruction produced.
    drift_percent
        ``100 · (measured − nominal) / nominal``. Positive =
        measurement overshoots, negative = undershoots.
    status
        :class:`CalibrationStatus` from the absolute drift.
    pixel_size_um
        Effective camera pixel (after magnification) — recorded
        for traceability.
    notes
        Free-form text. Lab uses for batch / cuvette / objective
        info that's not first-class state.
    """
    timestamp: str
    operator: str
    nominal_diameter_um: float
    measured_diameter_um: float
    drift_percent: float
    status: str
    pixel_size_um: float = 0.0
    notes: str = ""


def classify(drift_percent: float,
             *,
             yellow_threshold: float = 2.0,
             red_threshold: float = 5.0,
             ) -> CalibrationStatus:
    """Map an absolute drift percentage to traffic-light status.

    Defaults reproduce the lab's go/no-go rule. Caller may
    override for a stricter / looser regime (e.g. 1 % yellow for
    publication-grade data; 10 % red for screening).
    """
    a = abs(float(drift_percent))
    if a < yellow_threshold:
        return CalibrationStatus.GREEN
    if a < red_threshold:
        return CalibrationStatus.YELLOW
    return CalibrationStatus.RED


# ---------------------------------------------------------------------------
# Measurement — area threshold of |sin(phase)|
# ---------------------------------------------------------------------------

def measure_bead_diameter(
    phase: np.ndarray,
    pixel_size_um: float,
    *,
    centre_yx: Optional[tuple] = None,
    window_px: int = 60,
    threshold_frac: float = 0.3,
) -> float:
    """Estimate bead diameter (µm) from a focused phase image.

    Strategy
    --------
    1. Take ``sin(phase)`` so wraps don't poison thresholding —
    the bead contour shows up as a contiguous high-intensity
    region regardless of how many full-period wraps the OPL
    contains.
    2. Within a ``window_px`` square around the bead centre,
    threshold at ``threshold_frac × max_dev`` to define the bead
    footprint.
    3. Equivalent-disk diameter from the area: ``2·sqrt(A/π)``.

    This is the same metric ``test_focus_validation`` uses for the
    triple-sphere lateral check; bringing it here keeps both call
    sites on one definition.

    Parameters
    ----------
    phase
        Reconstructed phase image (radians, wrapped or unwrapped).
    pixel_size_um
        Effective camera pixel size (camera_pixel / M) in µm.
    centre_yx
        Optional bead centre. Defaults to image centre — fine for
        single-bead calibration scenes.
    window_px
        Half-size of the analysis ROI around the centre. Generous
        cap so the measurement isn't sensitive to ROI placement
        within reason.
    threshold_frac
        Fraction of peak deviation that defines the bead boundary.
        0.3 = matches the existing lateral-diameter test.

    Returns
    -------
    float
        Measured diameter in µm, or 0.0 if no contrast was found
        (calibration scene was blank / all-noise).
    """
    a = np.asarray(phase, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"phase must be 2-D; got shape {a.shape}")
    ny, nx = a.shape
    if centre_yx is None:
        iy, ix = ny // 2, nx // 2
    else:
        iy = int(centre_yx[0])
        ix = int(centre_yx[1])
    y0, y1 = max(0, iy - window_px), min(ny, iy + window_px + 1)
    x0, x1 = max(0, ix - window_px), min(nx, ix + window_px + 1)

    # sin() collapses wraps without losing the bead's footprint.
    signal = np.sin(a[y0:y1, x0:x1].astype(np.float64))
    baseline = float(np.median(signal))
    dev = np.abs(signal - baseline)
    peak = float(dev.max())
    if peak < 1e-6:
        return 0.0
    mask = dev >= threshold_frac * peak
    area_px = int(np.count_nonzero(mask))
    if area_px <= 0:
        return 0.0
    eq_radius_px = float(np.sqrt(area_px / np.pi))
    return 2.0 * eq_radius_px * float(pixel_size_um)


# ---------------------------------------------------------------------------
# Public API — record + history
# ---------------------------------------------------------------------------

def record_check(
    *,
    nominal_diameter_um: float,
    measured_diameter_um: float,
    pixel_size_um: float = 0.0,
    notes: str = "",
    operator: Optional[str] = None,
    history_path: Optional[Path] = None,
    yellow_threshold: float = 2.0,
    red_threshold: float = 5.0,
) -> CalibrationCheck:
    """Compute drift, classify, append to per-user history JSONL,
    emit an audit event.

    ``operator`` defaults to :func:`core.user_profile.current_user`
    (the v2.0.7 idiom). ``history_path`` defaults to the user's
    ``calibration_history.jsonl`` under their state dir.
    """
    if nominal_diameter_um <= 0:
        raise ValueError(
            f"nominal_diameter_um must be > 0, got {nominal_diameter_um}",
        )
    drift = 100.0 * (measured_diameter_um - nominal_diameter_um) \
        / nominal_diameter_um
    status = classify(
        drift,
        yellow_threshold=yellow_threshold,
        red_threshold=red_threshold,
    )

    if operator is None:
        from core import user_profile
        operator = user_profile.current_user()

    check = CalibrationCheck(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        operator=operator,
        nominal_diameter_um=float(nominal_diameter_um),
        measured_diameter_um=float(measured_diameter_um),
        drift_percent=float(drift),
        status=status.value,
        pixel_size_um=float(pixel_size_um),
        notes=str(notes),
    )

    if history_path is None:
        from core import user_profile
        history_path = user_profile.user_state_dir(operator) \
            / "calibration_history.jsonl"
    history_path = Path(history_path).expanduser()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(check), default=str) + "\n")

    # Best-effort audit event — never raises.
    try:
        from core.audit import get_audit_log
        get_audit_log().record(
            action="calibration_check",
            params={
                "nominal_diameter_um": nominal_diameter_um,
                "pixel_size_um": pixel_size_um,
                "notes": notes,
            },
            result_summary={
                "measured_diameter_um": measured_diameter_um,
                "drift_percent": drift,
                "status": status.value,
            },
        )
    except Exception:  # pragma: no cover
        _LOG.warning("calibration: audit emission failed", exc_info=True)
    return check


def load_history(path: Path) -> List[CalibrationCheck]:
    """Read a calibration JSONL into a list of :class:`CalibrationCheck`.
    Returns an empty list when the file is absent / unreadable so the
    UI can render a "no history yet" state without a try/except in
    every call site.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return []
    out: List[CalibrationCheck] = []
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                out.append(CalibrationCheck(**d))
            except TypeError:
                # Older / partial records — best effort fill.
                out.append(CalibrationCheck(
                    timestamp=str(d.get("timestamp", "")),
                    operator=str(d.get("operator", "")),
                    nominal_diameter_um=float(d.get(
                        "nominal_diameter_um", 0.0)),
                    measured_diameter_um=float(d.get(
                        "measured_diameter_um", 0.0)),
                    drift_percent=float(d.get("drift_percent", 0.0)),
                    status=str(d.get("status", "green")),
                    pixel_size_um=float(d.get("pixel_size_um", 0.0)),
                    notes=str(d.get("notes", "")),
                ))
    except OSError as exc:
        _LOG.warning("calibration: could not read %s: %s", p, exc)
        return []
    return out


__all__ = [
    "CalibrationStatus",
    "CalibrationCheck",
    "classify",
    "measure_bead_diameter",
    "record_check",
    "load_history",
]
