"""Vector PDF report export — v2.0.9 sprint, P1.

Sven's compliance ask (Lindqvist 2026-04-27):

    > "HTML report güzel — internal review için. Ama Nature /
    > EMBO J / J Cell Biol için PDF + vector figure + raw data
    > CSV bundle istiyorlar."

This module produces a vector PDF the lab can drop straight into a
Methods supplement. Backend: matplotlib (already in the venv) →
``backend_pdf.PdfPages``. Every figure is vector except the
sample image (rasterised at 300 dpi inside the PDF; PNG-equivalent
for filesize, infinite resolution at the colorbar / scale bar).

Why not weasyprint / reportlab?
- HTML→PDF (weasyprint) loses vector control over scale bars + axes
- reportlab is excellent but overlapping with matplotlib for plot
  rendering — adding it just for layout is a cost without payoff

Layout (one page, pluggable):

* **Header** — sample id, operator, timestamp.
* **Hologram** — input image, optional crop box.
* **Phase / amplitude** — reconstructed images side-by-side, with
  scale bar and colorbar, units (radians / a.u.).
* **Line profiles** (optional) — overlay plot of every
  :class:`LineProfile` the operator left.
* **Calibration footer** — last calibration check status (green /
  yellow / red) so the reviewer sees instrument state-of-affairs.

Public API
----------
* :func:`render_pdf_report` — write a one-page PDF to a path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


_LOG = logging.getLogger(__name__)


@dataclass
class PdfReportData:
    """All inputs the renderer needs.

    Most fields are optional — the renderer drops a section
    cleanly when its source isn't provided. Lab can produce a
    bare "hologram + phase" report, or a fully-loaded paper
    figure with profiles + calibration.
    """
    sample_id: str = ""
    operator: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
    )
    pixel_size_um: float = 1.0      # for scale bar
    notes: str = ""

    # Images — all optional. Float arrays in their natural units.
    hologram: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None       # radians
    amplitude: Optional[np.ndarray] = None   # arbitrary

    # Reconstruction parameters dict — printed verbatim under the
    # header. Same shape ``ReconParams.__dict__`` produces.
    recon_params: dict = field(default_factory=dict)

    # Optional autofocus / QPI summary scalars to print below.
    autofocus_z_mm: Optional[float] = None
    qpi_opd_range_nm: Optional[float] = None
    qpi_total_dry_mass_pg: Optional[float] = None

    # Optional line profiles overlay.
    line_profiles: Sequence = ()  # Sequence[SampledProfile]

    # Optional calibration verdict (CalibrationCheck-like).
    calibration_status: Optional[str] = None  # "green"|"yellow"|"red"
    calibration_drift_percent: Optional[float] = None
    calibration_timestamp: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_scale_bar(ax, image_shape: tuple, pixel_size_um: float,
                   *, length_um: float = 10.0,
                   colour: str = "white") -> None:
    """Bottom-right scale bar with caption."""
    if pixel_size_um <= 0 or length_um <= 0:
        return
    ny, nx = image_shape[:2]
    bar_px = float(length_um) / float(pixel_size_um)
    if bar_px <= 0 or bar_px > nx * 0.6:
        return
    margin_px = max(8, int(0.04 * nx))
    y_pos = ny - margin_px - max(2, int(0.01 * ny))
    x_end = nx - margin_px
    x_start = x_end - bar_px
    ax.plot([x_start, x_end], [y_pos, y_pos],
            color=colour, lw=3, solid_capstyle="butt")
    ax.text((x_start + x_end) / 2.0, y_pos - 0.02 * ny,
            f"{length_um:.0f} µm",
            color=colour, ha="center", va="bottom",
            fontsize=8)


def _draw_image(ax, arr: np.ndarray, title: str, *,
                cmap: str = "gray", colorbar_label: str = "",
                pixel_size_um: float = 0.0,
                rasterized: bool = True) -> None:
    """One image panel with title, colorbar, scale bar."""
    im = ax.imshow(arr, cmap=cmap, rasterized=rasterized)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar_label:
        from matplotlib.cm import ScalarMappable  # noqa: F401
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046,
                                  pad=0.04)
        cbar.set_label(colorbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    if pixel_size_um > 0:
        _add_scale_bar(ax, arr.shape, pixel_size_um)


def _draw_line_profiles(ax, profiles: Sequence) -> None:
    """Overlay every line's distance vs. value. NaN samples drop
    naturally."""
    if not profiles:
        ax.axis("off")
        ax.text(0.5, 0.5, "(no line profiles)",
                ha="center", va="center", fontsize=8,
                color="0.5")
        return
    for sp in profiles:
        label = sp.profile.label or "(unlabelled)"
        ax.plot(sp.distance_px, sp.values, label=label,
                color=sp.profile.colour_rgb, lw=1.5)
    ax.set_xlabel("Distance along line (px)", fontsize=8)
    ax.set_ylabel("Phase (rad)", fontsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def _format_param_block(params: dict, max_per_col: int = 6) -> str:
    """Two-column key=value block. Keeps PDF text readable."""
    if not params:
        return "(no params)"
    items = list(params.items())
    return "\n".join(
        f"  {k} = {v}" for k, v in items
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_pdf_report(data: PdfReportData,
                      out_path: Path | str) -> Path:
    """Render ``data`` as a single-page vector PDF at ``out_path``.

    Returns the resolved output path.
    """
    # Defer matplotlib imports to keep import-time cost low for
    # callers who don't need the PDF backend.
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 11), dpi=300)
    # Layout grid: header | top images | profiles | footer
    # Use plain GridSpec for explicit row weights.
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        nrows=4, ncols=2,
        height_ratios=[0.18, 1.0, 0.65, 0.20],
        hspace=0.5, wspace=0.3,
        left=0.07, right=0.93, top=0.95, bottom=0.05,
    )

    # ── Header ──────────────────────────────────────────────────
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    title = "DHM Reconstruction Report"
    header_ax.text(0.0, 1.0, title, fontsize=14, weight="bold",
                   ha="left", va="top")
    line2 = (f"Sample: {data.sample_id or '(unset)'}    "
             f"Operator: {data.operator or '(unset)'}    "
             f"Generated: {data.timestamp}")
    header_ax.text(0.0, 0.55, line2, fontsize=9, ha="left",
                   va="top", color="0.3")
    if data.notes:
        header_ax.text(0.0, 0.20, data.notes, fontsize=9,
                       ha="left", va="top", color="0.3",
                       wrap=True)

    # ── Top row: phase + amplitude (or hologram if those missing)
    if data.phase is not None:
        ax = fig.add_subplot(gs[1, 0])
        _draw_image(ax, data.phase, "Phase",
                    cmap="twilight", colorbar_label="rad",
                    pixel_size_um=data.pixel_size_um)
    elif data.hologram is not None:
        ax = fig.add_subplot(gs[1, 0])
        _draw_image(ax, data.hologram, "Hologram",
                    cmap="gray",
                    pixel_size_um=data.pixel_size_um)

    if data.amplitude is not None:
        ax = fig.add_subplot(gs[1, 1])
        _draw_image(ax, data.amplitude, "Amplitude",
                    cmap="gray", colorbar_label="a.u.",
                    pixel_size_um=data.pixel_size_um)
    elif data.phase is not None and data.hologram is not None:
        ax = fig.add_subplot(gs[1, 1])
        _draw_image(ax, data.hologram, "Hologram",
                    cmap="gray",
                    pixel_size_um=data.pixel_size_um)

    # ── Profiles row ────────────────────────────────────────────
    profiles_ax = fig.add_subplot(gs[2, 0])
    profiles_ax.set_title("Line profiles", fontsize=10)
    _draw_line_profiles(profiles_ax, data.line_profiles)

    # ── Params block ───────────────────────────────────────────
    params_ax = fig.add_subplot(gs[2, 1])
    params_ax.axis("off")
    params_ax.set_title("Reconstruction parameters", fontsize=10,
                        loc="left")
    params_text = _format_param_block(data.recon_params)
    params_ax.text(0.0, 1.0, params_text, fontsize=8,
                   family="monospace", ha="left", va="top")

    # ── Footer ──────────────────────────────────────────────────
    footer_ax = fig.add_subplot(gs[3, :])
    footer_ax.axis("off")
    pieces = []
    if data.autofocus_z_mm is not None:
        pieces.append(f"Autofocus z = {data.autofocus_z_mm:.3f} mm")
    if data.qpi_opd_range_nm is not None:
        pieces.append(f"OPD = {data.qpi_opd_range_nm:.1f} nm")
    if data.qpi_total_dry_mass_pg is not None:
        pieces.append(f"Total dry mass = "
                      f"{data.qpi_total_dry_mass_pg:.1f} pg")
    qpi_line = "    ".join(pieces) if pieces else ""

    cal_line = ""
    if data.calibration_status:
        cal_line = (
            f"Calibration: {data.calibration_status.upper()}"
            f"  (drift = "
            f"{data.calibration_drift_percent or 0:.2f} %, "
            f"checked {data.calibration_timestamp or 'unknown'})"
        )
    footer_text = "\n".join(filter(None, [qpi_line, cal_line]))
    if footer_text:
        footer_ax.text(0.0, 1.0, footer_text, fontsize=9,
                       ha="left", va="top", color="0.3",
                       family="monospace")

    with PdfPages(out) as pdf:
        pdf.savefig(fig)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    return out


__all__ = [
    "PdfReportData",
    "render_pdf_report",
]
