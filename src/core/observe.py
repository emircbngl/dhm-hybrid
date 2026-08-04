"""Observation / "vision" layer — how an AI agent *sees* results.

Pure, Qt-free functions that turn reconstruction/phase/field arrays into
either (a) rich structured readouts a text LLM can reason over, or (b) a
rendered PNG a vision-capable agent (over MCP) can literally look at.

This is the DHM analogue of the sister project's ``inspect_beam`` /
``inspect_element`` tools. Two consumers share it:

* the in-app local-LLM copilot — reads the numeric ``inspect_*`` dicts;
* the ``dhm-mcp`` headless server — same dicts, plus ``render_view`` PNG
  bytes surfaced as MCP image content so Claude sees the actual field.

Design rules
------------
* Input is always plain numpy (complex field / real phase / magnitude).
  No Qt, no app objects — the caller extracts arrays and passes them.
* Every ``inspect_*`` returns a flat, JSON-friendly dict of scalars
  (no numpy arrays) so it drops straight into a chat prompt or a tool
  result.
* Never raise on degenerate input (empty / all-NaN / constant): report
  it in the dict instead, so a tool call always yields a readable answer.
* Reuses the real engine (focus metrics, qpi segmentation, contrast,
  scalebar) rather than re-deriving physics.
"""
from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np

from core.autofocus.metrics import FocusMetric, _calc_metric

__all__ = [
    "inspect_reconstruction",
    "inspect_phase_map",
    "inspect_field",
    "render_view",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_2d(arr: Any) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def _finite_stats(values: np.ndarray) -> dict:
    """min/max/mean/std/p5/p50/p95 over finite entries; safe on empty."""
    v = np.asarray(values).ravel()
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {"count": 0, "finite_fraction": 0.0}
    return {
        "count": int(v.size),
        "finite_fraction": round(float(finite.size / v.size), 6),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p5": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
    }


def _round_dict(d: dict, ndigits: int = 6) -> dict:
    out = {}
    for k, val in d.items():
        if isinstance(val, float) and np.isfinite(val):
            out[k] = round(val, ndigits)
        elif isinstance(val, dict):
            out[k] = _round_dict(val, ndigits)
        else:
            out[k] = val
    return out


# ---------------------------------------------------------------------------
# inspect_reconstruction — "is this a good reconstruction?"
# ---------------------------------------------------------------------------

def inspect_reconstruction(complex_field: Any) -> dict:
    """Structured readout of a reconstructed complex field.

    Answers the questions an operator asks looking at a fresh recon:
    is it in focus? is there real signal or just noise? how flat is the
    background? All wrap-safe (phase metrics via the engine).
    """
    field = _as_2d(complex_field)
    if not np.iscomplexobj(field):
        # Allow a real phase array too, but flag it.
        field = field.astype(np.complex128)
    if field.size == 0:
        return {"ok": False, "reason": "empty field"}

    finite = np.isfinite(field)
    finite_fraction = float(np.count_nonzero(finite) / field.size)

    # Amplitude stats over FINITE pixels only — a single NaN would
    # otherwise poison np.max/np.min and hide the real contrast.
    amp = np.abs(np.where(finite, field, 0))
    amp_finite = np.abs(field[finite]) if finite.any() else np.array([0.0])

    # Focus / structure — the same metrics autofocus optimises, so an
    # in-focus recon scores high here too (all wrap-safe on the phase).
    # Feed a NaN-free copy so the metric never returns NaN.
    clean = np.where(finite, field, 0)
    focus_lap = _calc_metric(clean, FocusMetric.LAPLACIAN_VARIANCE)
    focus_tenengrad = _calc_metric(clean, FocusMetric.TENENGRAD)
    phase_variance = _calc_metric(clean, FocusMetric.PHASE_VARIANCE)

    # Amplitude contrast (Michelson): high when there's real object
    # modulation, ~0 for a flat/empty field.
    a_max, a_min = float(np.max(amp_finite)), float(np.min(amp_finite))
    contrast = (a_max - a_min) / (a_max + a_min) if (a_max + a_min) > 0 else 0.0
    phase = np.angle(clean)

    # Background flatness: RMS of the wrapped phase after removing a plane
    # is a cheap "how tilted/curved is the residual" proxy. We approximate
    # with the std of the phase gradient magnitude (flat bg → small).
    gy = np.diff(np.unwrap(phase, axis=0), axis=0)
    gx = np.diff(np.unwrap(phase, axis=1), axis=1)
    grad_rms = float(np.sqrt(np.mean(gy ** 2) + np.mean(gx ** 2)))

    return _round_dict({
        "ok": True,
        "shape": list(field.shape),
        "finite_fraction": finite_fraction,
        "amplitude": {"min": a_min, "max": a_max, "mean": float(np.mean(amp))},
        "amplitude_contrast": contrast,
        "phase_variance": phase_variance,
        "focus_laplacian_variance": focus_lap,
        "focus_tenengrad": focus_tenengrad,
        "phase_gradient_rms": grad_rms,
        "verdict": _recon_verdict(finite_fraction, contrast, focus_lap),
    })


def _recon_verdict(finite_fraction: float, contrast: float,
                   focus_lap: float) -> str:
    # Any non-finite pixel in a reconstruction is an anomaly worth
    # flagging (the engine normally sanitises to 0) — flag on < 1.0,
    # not a loose fraction that a single NaN in a big frame would pass.
    if finite_fraction < 1.0:
        return "warning: non-finite pixels present (NaN/Inf)"
    if contrast < 0.02:
        return "low amplitude contrast — likely empty field or no object"
    return "reconstruction has signal; compare focus scores across z"


# ---------------------------------------------------------------------------
# inspect_phase_map — "what does the unwrapped phase tell us?"
# ---------------------------------------------------------------------------

def inspect_phase_map(
    unwrapped_phase: Any,
    *,
    wavelength_m: float = 632.8e-9,
    pixel_size_m: float = 0.088e-6,
    n_sample: float = 1.38,
    n_medium: float = 1.337,
    segment: bool = True,
) -> dict:
    """Structured readout of an unwrapped phase map (QPI-oriented).

    Reports phase/OPD statistics and, optionally, a cell-segmentation
    count + total dry mass via the real qpi engine — the numbers an
    operator would read off a QPI panel.
    """
    phase = _as_2d(unwrapped_phase).astype(np.float64)
    if phase.size == 0:
        return {"ok": False, "reason": "empty phase map"}

    from core import qpi  # local import keeps observe import-cheap

    opd = qpi.phase_to_opd(phase, wavelength_m)
    opd_nm = opd * 1e9

    result: dict = {
        "ok": True,
        "shape": list(phase.shape),
        "phase_rad": _finite_stats(phase),
        "opd_nm": _finite_stats(opd_nm),
        "peak_to_valley_rad": float(np.ptp(phase[np.isfinite(phase)]))
        if np.isfinite(phase).any() else 0.0,
    }

    if segment:
        try:
            # A GENUINE multi-cell count. qpi.segment_cell_phase returns a
            # single-largest-component BINARY (0/255) mask by design, so the
            # old `len(unique(mask[mask>0]))` was structurally 0 or 1 — it
            # reported "1 cell" for any multi-cell field of view, misleading
            # the AI (2026-07-08 review). Do an independent threshold + connected
            # -component label here, dropping sub-resolution specks.
            from scipy.ndimage import label as _ndlabel
            finite_opd = np.nan_to_num(opd, nan=0.0)
            lo, hi = float(finite_opd.min()), float(finite_opd.max())
            binary = finite_opd > (lo + 0.15 * (hi - lo))
            labeled, n_raw = _ndlabel(binary)
            min_area = max(4, int(0.0005 * finite_opd.size))  # ≥0.05% of frame
            keep = [i for i in range(1, n_raw + 1)
                    if int(np.count_nonzero(labeled == i)) >= min_area]
            result["cell_count"] = len(keep)
            if keep:
                cells_mask = np.isin(labeled, keep)
                dm = qpi.compute_dry_mass_total(
                    opd, pixel_size_m, mask=cells_mask,
                )
                result["total_dry_mass_pg"] = float(dm * 1e15)
        except Exception as exc:  # segmentation is best-effort
            result["segmentation_error"] = str(exc)

    return _round_dict(result)


# ---------------------------------------------------------------------------
# inspect_field — histogram-level view of a complex field
# ---------------------------------------------------------------------------

def inspect_field(complex_field: Any, *, bins: int = 16) -> dict:
    """Coarse amplitude/phase histograms + dynamic range of a field.

    A compact distribution view for when the scalar summaries aren't
    enough — e.g. spotting a bimodal amplitude (object vs background) or
    a clipped phase.
    """
    field = _as_2d(complex_field)
    if field.size == 0:
        return {"ok": False, "reason": "empty field"}
    amp = np.abs(field).ravel()
    phase = np.angle(field).ravel() if np.iscomplexobj(field) \
        else field.astype(np.float64).ravel()

    amp_f = amp[np.isfinite(amp)]
    ph_f = phase[np.isfinite(phase)]
    if amp_f.size == 0:
        return {"ok": False, "reason": "no finite pixels"}

    amp_hist, amp_edges = np.histogram(amp_f, bins=bins)
    ph_hist, ph_edges = np.histogram(ph_f, bins=bins, range=(-np.pi, np.pi))
    return {
        "ok": True,
        "amplitude_hist": amp_hist.tolist(),
        "amplitude_edges": [round(float(e), 6) for e in amp_edges],
        "phase_hist": ph_hist.tolist(),
        "phase_edges": [round(float(e), 4) for e in ph_edges],
        "amplitude_dynamic_range_db": round(float(
            20 * np.log10((amp_f.max() + 1e-30) / (amp_f.min() + 1e-30))
        ), 3),
    }


# ---------------------------------------------------------------------------
# render_view — the actual picture (PNG bytes)
# ---------------------------------------------------------------------------

_VALID_KINDS = ("amplitude", "phase", "spectrum", "depth", "raw")


def render_view(
    array: Any,
    kind: str = "amplitude",
    *,
    pixel_size_um: Optional[float] = None,
    colormap: Optional[str] = None,
    max_size: int = 1024,
) -> bytes:
    """Render a 2-D array to PNG bytes for an agent (or the app) to view.

    ``kind`` selects the physical interpretation + default colormap:
    amplitude/raw → grey, phase → twilight, spectrum → log-magnitude
    (viridis), depth → viridis. Percentile (1–99) contrast is applied
    for robustness (the lab's chosen display policy, via core.contrast).
    A scalebar is drawn when ``pixel_size_um`` is given.

    Returns raw PNG bytes so callers can write a file, base64-encode for
    MCP image content, or hand to a Qt/DPG texture. Never raises on
    degenerate input — renders a labelled blank instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    arr = _as_2d(array)
    field_is_complex = np.iscomplexobj(arr)

    # Derive the display image + default colormap from kind.
    if kind == "phase":
        img = np.angle(arr) if field_is_complex else arr.astype(np.float64)
        cmap = colormap or "twilight"
        cbar_label = "phase (rad)"
    elif kind == "spectrum":
        # The input is always a SPATIAL-domain array (raw hologram or a
        # reconstructed complex field) — a complex field still needs fft2.
        # The old complex branch merely fftshift'ed the field itself,
        # rendering quadrant-swapped |field| mislabeled as log|F|
        # (2026-07-06 review #2).
        mag = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
        img = np.log1p(mag)
        cmap = colormap or "viridis"
        cbar_label = "log|F|"
    elif kind == "depth":
        img = arr.astype(np.float64)
        cmap = colormap or "viridis"
        cbar_label = "z (mm)"
    else:  # amplitude / raw
        img = np.abs(arr) if field_is_complex else arr.astype(np.float64)
        cmap = colormap or "gray"
        cbar_label = "amplitude" if kind == "amplitude" else "value"

    img, ds_step = _downsample(np.nan_to_num(img, nan=0.0), max_size)

    # Robust 1–99 percentile display range (core policy). Spectrum/phase
    # keep their natural range where percentile would hide structure.
    if kind in ("amplitude", "raw", "depth"):
        vmin, vmax = _percentile_range(img)
    else:
        vmin, vmax = float(np.min(img)), float(np.max(img))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-9

    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(kind, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    if pixel_size_um and pixel_size_um > 0:
        # After downsampling by ``ds_step``, each displayed pixel spans
        # ``ds_step`` original pixels, so the effective pixel pitch scales up
        # by that factor. Passing the raw pixel size with the shrunken frame
        # width mislabels the bar by ``ds_step`` (2026-07-06 review).
        _draw_scalebar(ax, img.shape[1], pixel_size_um * ds_step)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _downsample(img: np.ndarray, max_size: int) -> tuple[np.ndarray, int]:
    """Stride-decimate ``img`` to fit ``max_size`` and report the stride.

    The stride is returned so the caller can scale a physical pixel pitch to
    the downsampled grid (e.g. for a correctly-labelled scalebar).
    """
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_size)))
    return (img[::step, ::step], step) if step > 1 else (img, 1)


def _percentile_range(img: np.ndarray) -> tuple[float, float]:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def _draw_scalebar(ax, frame_width_px: int, pixel_size_um: float) -> None:
    from core.scalebar import compute_scalebar
    spec = compute_scalebar(frame_width_px, pixel_size_um)
    if spec is None or spec.length_px <= 0:
        return
    margin = max(6, int(frame_width_px * 0.04))
    y = margin
    x1 = frame_width_px - margin
    x0 = x1 - spec.length_px
    ax.plot([x0, x1], [y, y], color="white", linewidth=3, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y + margin * 0.6, spec.label,
            color="white", ha="center", va="top", fontsize=9)
