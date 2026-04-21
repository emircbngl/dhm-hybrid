"""Standalone HTML report generator for DHM reconstructions.

Everything (figures, CSS) is embedded so the output file is a single
self-contained HTML document that can be emailed or dropped into a lab
notebook.

Images are rendered with matplotlib's Agg backend so this module can be
called from any thread (GUI or worker).
"""
from __future__ import annotations

import base64
import html
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import; Agg is thread-safe
import matplotlib.pyplot as plt  # noqa: E402

try:
    from __version__ import __version__, __git_sha__, __app_name__  # type: ignore
except Exception:  # pragma: no cover - tests / direct runs
    __version__ = "unknown"
    __git_sha__ = "unknown"
    __app_name__ = "DHM Reconstruction"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_unit(value: Any, unit: str, decimals: int = 3) -> str:
    """Format ``value`` with ``unit`` and a sensible SI rescaling.

    Accepts the raw SI value (metres, kilograms, radians, …) and converts to
    a human-scale unit.  Unknown units fall through as-is.
    """
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not np.isfinite(v):
        return "-"

    u = unit.lower()
    if u in ("m", "meter", "metre"):  # length -> mm
        return f"{v * 1e3:.{decimals}f} mm"
    if u == "mm":
        return f"{v:.{decimals}f} mm"
    if u in ("nm",):
        return f"{v:.{decimals}f} nm"
    if u == "wavelength_m":  # pretty-print wavelength in nm
        return f"{v * 1e9:.{decimals}f} nm"
    if u == "pixel_size_m":
        return f"{v * 1e6:.{decimals}f} µm"
    if u == "z_m":
        return f"{v * 1e3:.{decimals}f} mm"
    if u in ("kg",):  # mass -> pg for dry mass
        return f"{v * 1e15:.{decimals}f} pg"
    if u in ("pg",):
        return f"{v:.{decimals}f} pg"
    if u in ("opd_m", "opd"):
        return f"{v * 1e9:.{decimals}f} nm"
    if u == "rad":
        return f"{v:.{decimals}f} rad"
    if u == "":
        return f"{v:.{decimals}g}"
    return f"{v:.{decimals}g} {unit}"


def _fmt_scalar(value: Any) -> str:
    """Best-effort pretty-printer for arbitrary dict values in the report."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return "-"
        # Compact but readable: switch to scientific for very small/large.
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.4g}"
        return f"{v:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_scalar(v) for v in value)
    return html.escape(str(value))


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------
def _render_image(
    img: np.ndarray,
    *,
    title: str,
    cmap: str,
    cbar_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> str:
    """Render a 2D array to a base64-encoded PNG string (no `data:` prefix)."""
    arr = np.asarray(img)
    if arr.ndim != 2:
        # Flatten channel axis if accidentally passed RGB; otherwise bail.
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr.mean(axis=-1)
        else:
            raise ValueError(f"image must be 2D, got shape {arr.shape}")

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=110)
    try:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
    finally:
        plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(b64: str, alt: str) -> str:
    return (
        f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" '
        f'class="figure"/>'
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
_RECON_LABELS = {
    "wavelength_m": ("Wavelength", "wavelength_m"),
    "pixel_size_m": ("Pixel size", "pixel_size_m"),
    "z_m": ("Propagation distance (z)", "z_m"),
    "method": ("Method", ""),
    "n": ("Refractive index n", ""),
}

_AF_LABELS = {
    "z_found_m": ("Focus z", "z_m"),
    "metric": ("Metric", ""),
    "algorithm": ("Algorithm", ""),
    "evaluations": ("Evaluations", ""),
}

_QPI_LABELS = {
    "opd_mean_m": ("OPD mean", "opd_m"),
    "opd_std_m": ("OPD std", "opd_m"),
    "opd_max_m": ("OPD max", "opd_m"),
    "height_mean_m": ("Height mean", "m"),
    "height_std_m": ("Height std", "m"),
    "height_max_m": ("Height max", "m"),
    "dry_mass_kg": ("Dry mass", "kg"),
    "dry_mass_pg": ("Dry mass", "pg"),
    "roughness_rms_m": ("RMS roughness", "m"),
    "roughness_ra_m": ("Ra roughness", "m"),
}


def _format_table_rows(
    data: dict, label_map: dict[str, tuple[str, str]]
) -> list[tuple[str, str]]:
    """Return ``(label, formatted_value)`` rows for a ``data`` dict.

    Known keys come first in label-map order; remaining keys are rendered
    generically.  Values are plain text (no HTML), so this is reusable by
    the HTML and PDF paths.  ``None`` values are skipped.
    """
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, (label, unit) in label_map.items():
        if key in data and data[key] is not None:
            seen.add(key)
            val = _fmt_unit(data[key], unit) if unit else _fmt_scalar(data[key])
            rows.append((label, val))
    for key, value in data.items():
        if key in seen or value is None:
            continue
        rows.append((str(key), _fmt_scalar(value)))
    return rows


def _dict_table(data: dict, label_map: dict[str, tuple[str, str]]) -> str:
    """Render a labelled-known-keys-first HTML table of ``data``."""
    rows = _format_table_rows(data, label_map)
    if not rows:
        return ""
    # _fmt_scalar already HTML-escapes generic string values; unit-formatted
    # values are pure numeric text.  Labels are known-safe but escape anyway.
    html_rows = [
        f"<tr><th>{html.escape(label)}</th><td>{value}</td></tr>"
        for label, value in rows
    ]
    return '<table class="kv">' + "".join(html_rows) + "</table>"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
    margin:2em auto;max-width:920px;color:#222;line-height:1.45;padding:0 1em;}
h1{font-size:1.6em;margin-bottom:0.1em;}
h2{font-size:1.2em;margin-top:1.6em;border-bottom:1px solid #ddd;padding-bottom:0.2em;}
.meta{color:#666;font-size:0.9em;margin-bottom:1.5em;}
.meta span{margin-right:1.2em;}
table.kv{border-collapse:collapse;margin:0.4em 0 1em 0;}
table.kv th{text-align:left;padding:3px 12px 3px 0;font-weight:600;color:#333;
    vertical-align:top;white-space:nowrap;}
table.kv td{padding:3px 0;font-variant-numeric:tabular-nums;}
.figure{max-width:100%;height:auto;margin:0.6em 0;border:1px solid #eee;
    border-radius:4px;background:#fff;}
.figures{display:flex;flex-wrap:wrap;gap:1em;}
.figures .fig-block{flex:1 1 360px;min-width:320px;}
.notes{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;
    padding:0.8em;border-radius:4px;}
footer{color:#888;font-size:0.8em;margin-top:2.5em;border-top:1px solid #eee;
    padding-top:0.6em;}
""".strip()


def generate_html_report(
    output_path: Path,
    *,
    title: str = "DHM Reconstruction Report",
    operator: Optional[str] = None,
    recon_params: Optional[dict] = None,
    autofocus_summary: Optional[dict] = None,
    qpi_result: Optional[dict] = None,
    phase_image: Optional[np.ndarray] = None,
    amplitude_image: Optional[np.ndarray] = None,
    height_image: Optional[np.ndarray] = None,
    notes: str = "",
) -> Path:
    """Write a standalone HTML report to ``output_path`` and return it.

    Every section is optional: the report renders only the pieces for which
    input is supplied.  The file is written atomically via ``os.replace``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='en'><head>")
    parts.append("<meta charset='utf-8'/>")
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")

    # --- Header --------------------------------------------------------
    parts.append(f"<h1>{html.escape(title)}</h1>")
    meta_bits = [
        f"<span><b>{html.escape(__app_name__)}</b> v{html.escape(str(__version__))}</span>",
        f"<span>git {html.escape(str(__git_sha__))}</span>",
        f"<span>{html.escape(now)}</span>",
    ]
    if operator:
        meta_bits.append(f"<span>operator: {html.escape(str(operator))}</span>")
    parts.append(f"<div class='meta'>{''.join(meta_bits)}</div>")

    # --- Reconstruction parameters ------------------------------------
    if recon_params:
        parts.append("<h2>Reconstruction parameters</h2>")
        parts.append(_dict_table(recon_params, _RECON_LABELS))

    # --- Autofocus summary --------------------------------------------
    if autofocus_summary:
        parts.append("<h2>Autofocus</h2>")
        parts.append(_dict_table(autofocus_summary, _AF_LABELS))

    # --- QPI stats -----------------------------------------------------
    if qpi_result:
        parts.append("<h2>Quantitative phase imaging</h2>")
        parts.append(_dict_table(qpi_result, _QPI_LABELS))

    # --- Figures -------------------------------------------------------
    fig_blocks: list[str] = []
    if phase_image is not None:
        b64 = _render_image(
            phase_image,
            title="Phase (radians)",
            cmap="twilight",
            cbar_label="rad",
        )
        fig_blocks.append(
            "<div class='fig-block'>" + _img_tag(b64, "Phase image") + "</div>"
        )
    if amplitude_image is not None:
        b64 = _render_image(
            amplitude_image,
            title="Amplitude",
            cmap="gray",
            cbar_label="a.u.",
        )
        fig_blocks.append(
            "<div class='fig-block'>" + _img_tag(b64, "Amplitude image") + "</div>"
        )
    if height_image is not None:
        # Convert metres -> nm so the colorbar is readable.
        h_nm = np.asarray(height_image) * 1e9
        b64 = _render_image(
            h_nm,
            title="Height (nm)",
            cmap="viridis",
            cbar_label="nm",
        )
        fig_blocks.append(
            "<div class='fig-block'>" + _img_tag(b64, "Height map") + "</div>"
        )
    if fig_blocks:
        parts.append("<h2>Figures</h2>")
        parts.append("<div class='figures'>" + "".join(fig_blocks) + "</div>")

    # --- Notes ---------------------------------------------------------
    if notes and notes.strip():
        parts.append("<h2>Notes</h2>")
        parts.append(f"<div class='notes'>{html.escape(notes)}</div>")

    # --- Footer --------------------------------------------------------
    parts.append(
        f"<footer>Generated by {html.escape(__app_name__)} "
        f"v{html.escape(str(__version__))} ({html.escape(str(__git_sha__))})."
        f"</footer>"
    )
    parts.append("</body></html>")

    document = "".join(parts)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(document)
    os.replace(tmp_path, output_path)
    return output_path


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
# A4 portrait in inches; square page for image figures keeps aspect clean.
_PDF_PAGE_A4 = (8.27, 11.69)
_PDF_PAGE_SQUARE = (8.27, 8.27)


def _pdf_cover_page(
    pdf,
    *,
    title: str,
    operator: Optional[str],
    timestamp: str,
    notes: str,
) -> None:
    """Emit the cover page: title, app identifiers, timestamp, operator, notes."""
    fig = plt.figure(figsize=_PDF_PAGE_A4)
    try:
        fig.patch.set_facecolor("white")
        # Use a full-page axis with no frame so we can place text freely.
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()

        # Title (centred, upper third).
        ax.text(
            0.5, 0.82, title,
            ha="center", va="center",
            fontsize=22, fontweight="bold", family="sans-serif",
        )

        # App identification block.
        app_line = f"{__app_name__}  v{__version__}  (git {__git_sha__})"
        ax.text(
            0.5, 0.72, app_line,
            ha="center", va="center",
            fontsize=11, color="#555", family="sans-serif",
        )

        # Meta (timestamp, operator).
        ax.text(
            0.5, 0.66, f"Generated: {timestamp}",
            ha="center", va="center",
            fontsize=11, color="#333", family="sans-serif",
        )
        if operator:
            ax.text(
                0.5, 0.62, f"Operator: {operator}",
                ha="center", va="center",
                fontsize=11, color="#333", family="sans-serif",
            )

        # Notes block (optional, wrapped, lower half).
        if notes and notes.strip():
            ax.text(
                0.1, 0.45, "Notes",
                ha="left", va="top",
                fontsize=13, fontweight="bold", family="sans-serif",
            )
            ax.text(
                0.1, 0.41, notes.strip(),
                ha="left", va="top",
                fontsize=10, family="sans-serif",
                wrap=True,
            )

        pdf.savefig(fig)
    finally:
        plt.close(fig)


def _pdf_table_page(
    pdf,
    *,
    heading: str,
    rows: list[tuple[str, str]],
) -> None:
    """Emit a single page holding a 2-column Parameter/Value table."""
    if not rows:
        return
    fig, ax = plt.subplots(figsize=_PDF_PAGE_A4)
    try:
        fig.patch.set_facecolor("white")
        ax.set_axis_off()
        ax.set_title(heading, loc="left", fontsize=14, fontweight="bold",
                     family="sans-serif", pad=16)

        cell_text = [[label, value] for label, value in rows]
        table = ax.table(
            cellText=cell_text,
            colLabels=["Parameter", "Value"],
            loc="center",
            cellLoc="left",
            colLoc="left",
            colWidths=[0.45, 0.45],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        # Bold header row, subtle zebra striping.
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if r == 0:
                cell.set_text_props(fontweight="bold", family="sans-serif")
                cell.set_facecolor("#f2f2f2")
            else:
                cell.set_text_props(family="sans-serif")
                cell.set_facecolor("#ffffff" if r % 2 else "#fafafa")

        pdf.savefig(fig, bbox_inches="tight")
    finally:
        plt.close(fig)


def _pdf_image_page(
    pdf,
    img: np.ndarray,
    *,
    title: str,
    cmap: str,
    cbar_label: str = "",
) -> None:
    """Emit a single square page with an imshow + colorbar."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {arr.shape}")

    fig, ax = plt.subplots(figsize=_PDF_PAGE_SQUARE)
    try:
        fig.patch.set_facecolor("white")
        im = ax.imshow(arr, cmap=cmap, aspect="equal", interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold", family="sans-serif")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
    finally:
        plt.close(fig)


def generate_pdf_report(
    output_path: Path,
    *,
    title: str = "DHM Reconstruction Report",
    operator: Optional[str] = None,
    recon_params: Optional[dict] = None,
    autofocus_summary: Optional[dict] = None,
    qpi_result: Optional[dict] = None,
    phase_image: Optional[np.ndarray] = None,
    amplitude_image: Optional[np.ndarray] = None,
    height_image: Optional[np.ndarray] = None,
    notes: str = "",
) -> Path:
    """Emit a multi-page PDF report. Same inputs as ``generate_html_report``.

    Pages:
      1. Cover — title, app name/version/git sha, timestamp, operator, notes
      2. Reconstruction parameters (tabular)
      3. Autofocus summary (tabular) — skip if ``None``
      4. QPI result (tabular) — skip if ``None``
      5. Phase image — cmap ``twilight``, colorbar in radians
      6. Amplitude image — cmap ``gray``
      7. Height image — cmap ``viridis``, colorbar in nm

    Any section whose inputs are ``None`` is skipped; no blank pages.
    The file is written atomically via ``os.replace``.
    """
    from matplotlib.backends.backend_pdf import PdfPages  # lazy: heavy import

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with PdfPages(tmp_path) as pdf:
        # --- Cover ----------------------------------------------------
        _pdf_cover_page(
            pdf,
            title=title,
            operator=operator,
            timestamp=timestamp,
            notes=notes,
        )

        # --- Parameter tables -----------------------------------------
        if recon_params:
            _pdf_table_page(
                pdf,
                heading="Reconstruction parameters",
                rows=_format_table_rows(recon_params, _RECON_LABELS),
            )
        if autofocus_summary:
            _pdf_table_page(
                pdf,
                heading="Autofocus",
                rows=_format_table_rows(autofocus_summary, _AF_LABELS),
            )
        if qpi_result:
            _pdf_table_page(
                pdf,
                heading="Quantitative phase imaging",
                rows=_format_table_rows(qpi_result, _QPI_LABELS),
            )

        # --- Image pages ---------------------------------------------
        if phase_image is not None:
            _pdf_image_page(
                pdf, phase_image,
                title="Phase (radians)",
                cmap="twilight",
                cbar_label="rad",
            )
        if amplitude_image is not None:
            _pdf_image_page(
                pdf, amplitude_image,
                title="Amplitude",
                cmap="gray",
                cbar_label="a.u.",
            )
        if height_image is not None:
            h_nm = np.asarray(height_image) * 1e9  # metres -> nm
            _pdf_image_page(
                pdf, h_nm,
                title="Height (nm)",
                cmap="viridis",
                cbar_label="nm",
            )

        # PDF metadata for archivists.
        meta = pdf.infodict()
        meta["Title"] = title
        meta["Author"] = operator or __app_name__
        meta["Creator"] = f"{__app_name__} v{__version__}"
        meta["Subject"] = "DHM reconstruction report"

    os.replace(tmp_path, output_path)
    return output_path
