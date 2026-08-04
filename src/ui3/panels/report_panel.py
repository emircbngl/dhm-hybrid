"""ReportPanel — export controls (HTML report / PDF report / QPI CSV /
tomography bundle) for ui3.

Mirrors ``ui2``'s export flow (``ui2/app.py``:
``_on_export_report`` / ``_on_export_qpi_csv`` / ``_on_export_bundle`` +
their ``_perform_export`` bodies) but calls the underlying ``core``
exporters directly instead of going through ``ScienceDriver`` /
``_post_mailbox``:

* ``core.report.generate_html_report`` — standalone HTML report (figures
  embedded as base64 PNG).
* ``core.pdf_report.render_pdf_report`` — single-page vector PDF, built
  from a :class:`~core.pdf_report.PdfReportData`.
* ``core.qpi_export.write_qpi_csv`` — one CSV row per
  :class:`~core.qpi.QPIResult` (this panel always computes a *fresh*
  one-shot QPI at the current ``params.z_mm`` — see below).
* ``core.depth_map.write_tomography_bundle`` — the actual "tomography
  bundle" writer ui2 calls "bundle export" (``ScienceDriver.export_bundle``
  wraps this same function, *not* ``core.zenodo_bundle``). Needs a computed
  depth map (``ctx.last_depth()``) plus its clusters.

``WorkerBridge`` is the frozen ui3 contract and does **not** expose export
methods (see ``ui3/bridge.py``) — these are fast, synchronous file-I/O
calls (matplotlib Agg render + csv/zip write), not long-running science
compute, so this panel calls the ``core`` functions directly on the GUI
thread, exactly like every other lightweight action in ui3 (e.g.
``ctx.show_in_panel``). Every export is wrapped in try/except so a bad
path or missing prerequisite becomes a status + toast, never a traceback.

QPI is *not* cached anywhere in ``PanelContext`` (unlike ``ui2`` which
keeps ``self._last_qpi`` on the window). Per the task spec, the QPI CSV
export recomputes QPI on demand from ``ctx.last_recon()`` +
``ctx.get_field("phase_unwrapped")`` + the live ``ReconParams`` — if
either is missing, the button warns "run QPI first" (well, "reconstruct
first") without crashing.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui3.context import PanelContext
from ui3.design import Space

_LOG = logging.getLogger("ui3.panels.report")

_DEFAULT_EXPORT_DIR = Path.home() / ".dhm-reconstruction" / "exports"


def _default_export_dir() -> Path:
    """Return (creating if needed) the default export directory."""
    try:
        _DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        _LOG.debug("could not create default export dir", exc_info=True)
    return _DEFAULT_EXPORT_DIR


def _timestamped_name(stem: str, suffix: str) -> str:
    return f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"


class ReportPanel(QWidget):
    """Export controls. Built against the frozen ``PanelContext``."""

    def __init__(self, ctx: PanelContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx

        root = QVBoxLayout(self)
        root.setContentsMargins(Space.md, Space.md, Space.md, Space.md)
        root.setSpacing(Space.md)

        heading = QLabel("Export")
        heading.setProperty("role", "heading")
        root.addWidget(heading)

        root.addWidget(self._build_report_group())
        root.addWidget(self._build_qpi_group())
        root.addWidget(self._build_bundle_group())

        self._path_label = QLabel("No export yet.")
        self._path_label.setProperty("role", "muted")
        self._path_label.setWordWrap(True)
        self._path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self._path_label)

        root.addStretch(1)

    # -- groups -----------------------------------------------------------

    def _build_report_group(self) -> QGroupBox:
        box = QGroupBox("Report")
        outer = QVBoxLayout(box)

        row1 = QHBoxLayout()
        self.html_btn = QPushButton("Export HTML report")
        self.html_btn.setProperty("role", "primary")
        self.html_btn.clicked.connect(self._on_export_html)
        row1.addWidget(self.html_btn)
        row1.addStretch(1)
        outer.addLayout(row1)

        row2 = QHBoxLayout()
        self.pdf_btn = QPushButton("Export PDF report")
        self.pdf_btn.clicked.connect(self._on_export_pdf)
        row2.addWidget(self.pdf_btn)
        row2.addStretch(1)
        outer.addLayout(row2)

        return box

    def _build_qpi_group(self) -> QGroupBox:
        box = QGroupBox("QPI")
        outer = QVBoxLayout(box)
        row = QHBoxLayout()
        self.qpi_csv_btn = QPushButton("Export QPI CSV")
        self.qpi_csv_btn.clicked.connect(self._on_export_qpi_csv)
        row.addWidget(self.qpi_csv_btn)
        row.addStretch(1)
        outer.addLayout(row)
        return box

    def _build_bundle_group(self) -> QGroupBox:
        box = QGroupBox("Tomography bundle")
        outer = QVBoxLayout(box)
        row = QHBoxLayout()
        self.bundle_btn = QPushButton("Export tomography bundle")
        self.bundle_btn.clicked.connect(self._on_export_bundle)
        row.addWidget(self.bundle_btn)
        row.addStretch(1)
        outer.addLayout(row)
        return box

    # -- helpers ------------------------------------------------------------

    def _report_recon_params_dict(self) -> dict:
        """Mirror ``ui2.app._perform_export``'s ``last_recon_params`` dict —
        same keys ``core.report.generate_html_report``'s ``_RECON_LABELS``
        table expects."""
        p = self.ctx.get_params()
        return {
            "wavelength_m": float(p.wavelength_nm) * 1e-9,
            "pixel_size_m": float(p.effective_pixel_um()) * 1e-6,
            "z_m": float(p.z_mm) * 1e-3,
            "method": p.method,
            "mask_radius": int(p.mask_radius),
        }

    def _prompt_save_path(self, title: str, default_name: str,
                          filter_str: str) -> Optional[Path]:
        default_dir = _default_export_dir()
        start = str(default_dir / default_name)
        path_str, _ = QFileDialog.getSaveFileName(self, title, start, filter_str)
        if not path_str:
            return None
        return Path(path_str)

    def _prompt_save_dir(self, title: str) -> Optional[Path]:
        default_dir = _default_export_dir()
        dir_str = QFileDialog.getExistingDirectory(self, title, str(default_dir))
        if not dir_str:
            return None
        return Path(dir_str)

    def _report_success(self, path: Path) -> None:
        msg = f"Saved: {path}"
        self._path_label.setText(msg)
        self.ctx.set_status(msg, "ok")
        self.ctx.toast(msg, "ok")

    def _report_failure(self, action: str, exc: Exception) -> None:
        msg = f"{action} failed: {exc}"
        self.ctx.set_status(msg, "error")
        self.ctx.toast(msg, "error")
        _LOG.warning("%s", msg, exc_info=True)

    def _compute_fresh_qpi(self) -> Any:
        """Recompute a one-shot :class:`~core.qpi.QPIResult` from the
        current reconstruction + live params. Returns ``None`` (having
        already warned) when the prerequisites aren't there.

        QPI has no cache slot on ``PanelContext`` — the QPI panel keeps
        its own ``_oneshot_result``, but this panel is independent, so it
        recomputes from the same inputs ``ui2.workers.run_qpi`` uses
        (``core.qpi.compute_qpi``). This keeps the export in lock-step
        with whatever reconstruction is currently loaded rather than
        risking a stale cached QPI from a different z/hologram.
        """
        recon = self.ctx.last_recon()
        if recon is None:
            self.ctx.set_status(
                "Run reconstruction first, then run QPI.", "warn")
            self.ctx.toast("Reconstruct first.", "warn")
            return None

        phase_unwrapped = self.ctx.get_field("phase_unwrapped")
        if phase_unwrapped is None:
            self.ctx.set_status("Run QPI first.", "warn")
            self.ctx.toast("Run QPI first.", "warn")
            return None

        from core.qpi import compute_qpi, QPIMode

        p = self.ctx.get_params()
        try:
            return compute_qpi(
                np.asarray(phase_unwrapped),
                wavelength_m=float(p.wavelength_nm) * 1e-9,
                pixel_size_m=float(p.effective_pixel_um()) * 1e-6,
                mode=QPIMode.BOTH,
                n_sample=float(p.n_sample),
                n_medium=float(p.n_medium),
            )
        except Exception as exc:
            self._report_failure("QPI compute", exc)
            return None

    # -- HTML report --------------------------------------------------------

    def _on_export_html(self) -> None:
        recon = self.ctx.last_recon()
        if recon is None:
            self.ctx.set_status("Run reconstruction first.", "warn")
            self.ctx.toast("Reconstruct first.", "warn")
            return

        default_name = _timestamped_name("dhm_report", ".html")
        path = self._prompt_save_path(
            "Export HTML report", default_name, "HTML report (*.html)")
        if path is None:
            return
        if path.suffix.lower() != ".html":
            path = path.with_suffix(".html")

        try:
            from core.report import generate_html_report

            phase = getattr(recon, "unwrapped_phase", None)
            if phase is None or not getattr(phase, "size", 0):
                phase = getattr(recon, "phase", None)
            amplitude = getattr(recon, "amplitude", None)

            out_path = generate_html_report(
                output_path=path,
                title="DHM Reconstruction Report",
                recon_params=self._report_recon_params_dict(),
                phase_image=phase,
                amplitude_image=amplitude,
            )
            self._report_success(out_path)
        except Exception as exc:
            self._report_failure("HTML report export", exc)

    # -- PDF report --------------------------------------------------------

    def _on_export_pdf(self) -> None:
        recon = self.ctx.last_recon()
        if recon is None:
            self.ctx.set_status("Run reconstruction first.", "warn")
            self.ctx.toast("Reconstruct first.", "warn")
            return

        default_name = _timestamped_name("dhm_report", ".pdf")
        path = self._prompt_save_path(
            "Export PDF report", default_name, "PDF report (*.pdf)")
        if path is None:
            return
        if path.suffix.lower() != ".pdf":
            path = path.with_suffix(".pdf")

        try:
            from core.pdf_report import PdfReportData, render_pdf_report

            p = self.ctx.get_params()
            phase = getattr(recon, "unwrapped_phase", None)
            if phase is None or not getattr(phase, "size", 0):
                phase = getattr(recon, "phase", None)
            amplitude = getattr(recon, "amplitude", None)

            data = PdfReportData(
                pixel_size_um=float(p.effective_pixel_um()),
                phase=phase,
                amplitude=amplitude,
                recon_params=self._report_recon_params_dict(),
            )
            out_path = render_pdf_report(data, path)
            self._report_success(out_path)
        except Exception as exc:
            self._report_failure("PDF report export", exc)

    # -- QPI CSV -------------------------------------------------------------

    def _on_export_qpi_csv(self) -> None:
        qpi_result = self._compute_fresh_qpi()
        if qpi_result is None:
            return

        default_name = _timestamped_name("qpi", ".csv")
        path = self._prompt_save_path(
            "Export QPI CSV", default_name, "CSV (*.csv)")
        if path is None:
            return
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        try:
            from core.qpi_export import write_qpi_csv

            out_path = write_qpi_csv(path, qpi_result)
            self._report_success(out_path)
        except Exception as exc:
            self._report_failure("QPI CSV export", exc)

    # -- Tomography bundle ---------------------------------------------------

    def _on_export_bundle(self) -> None:
        depth = self.ctx.last_depth()
        if depth is None:
            self.ctx.set_status(
                "Compute the depth map first (Depth panel).", "warn")
            self.ctx.toast("Compute depth map first.", "warn")
            return

        out_dir = self._prompt_save_dir("Select tomography bundle directory")
        if out_dir is None:
            return

        try:
            from core.depth_map import write_tomography_bundle

            depth_result = getattr(depth, "result", depth)
            clusters = getattr(depth, "clusters", []) or []
            p = self.ctx.get_params()
            base_name = _timestamped_name("tomography", "")

            written = write_tomography_bundle(
                out_dir, depth_result, clusters,
                base_name=base_name,
                pixel_size_m=float(p.effective_pixel_um()) * 1e-6,
            )
            msg = f"Bundle saved: {len(written)} file(s) in {out_dir}"
            self._path_label.setText(msg)
            self.ctx.set_status(msg, "ok")
            self.ctx.toast(msg, "ok")
        except Exception as exc:
            self._report_failure("Tomography bundle export", exc)


__all__ = ["ReportPanel"]
