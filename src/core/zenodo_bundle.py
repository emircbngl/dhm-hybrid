"""Zenodo-ready supplementary bundle — v2.0.9 sprint, P2.

Sven asked for a citation-grade archive: one zip the lab can drop
on Zenodo as a paper supplement. It carries:

* ``figures.pdf``     — vector report (from :mod:`core.pdf_report`).
* ``raw_data.csv``    — long-format session export (T2 of v2.0.7).
* ``params.json``     — the recon params dict that produced the figures.
* ``checksum.txt``    — SHA-256 of every file inside the bundle.
* ``README.md``       — auto-generated, names the lab + DOI prep info.

Why bundling matters
--------------------
Reviewers of cell-biology + biophysics papers increasingly demand
a 'reproducibility supplement': not just figure PDFs but the
underlying numerical tables + the parameter set the authors used.
Having one tool produce that bundle saves the lab a half-day of
manual copy-pasting per paper revision.

Public API
----------
* :func:`build_bundle` — write a ``bundle.zip`` to disk.
* :func:`bundle_manifest` — render the README contents as a string
  (useful for tests + previews).

This module is **storage-agnostic** — callers pass the PDF + CSV
content / paths in. We don't run ``render_pdf_report`` here so
the bundle code can be exercised independently of matplotlib.
"""
from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


_LOG = logging.getLogger(__name__)


@dataclass
class BundleSpec:
    """All inputs the bundler needs.

    Required: a session-level identifier (``sample_id``) and a
    figures PDF path. Everything else has a default; missing
    pieces appear as "(none)" in the README rather than crashing
    the build.
    """
    sample_id: str
    operator: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
    )

    # Required artefact: the vector PDF figure.
    figures_pdf: Optional[Path] = None

    # Optional artefacts.
    raw_data_csv: Optional[Path] = None
    params: Dict = field(default_factory=dict)
    notes: str = ""

    # Free-form attachments (caller can drop extra files in the
    # bundle — e.g. a Python script that reproduces the figures).
    extras: Dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_hex(path: Path) -> str:
    """SHA-256 hex of a file, computed in 64 KB chunks (no need
    to load the whole bundle into RAM for big PDFs)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_manifest(spec: BundleSpec) -> str:
    """Render the README.md content. Plain Markdown — Zenodo will
    render it as HTML on upload."""
    lines = [
        f"# DHM Supplementary Bundle — {spec.sample_id}",
        "",
        f"Generated: {spec.timestamp}",
        f"Operator: {spec.operator or '(unset)'}",
        "",
        "## Contents",
        "",
        f"- `figures.pdf` — vector reconstruction report.",
    ]
    if spec.raw_data_csv:
        lines.append(
            "- `raw_data.csv` — long-format per-frame measurements."
        )
    lines.append("- `params.json` — reconstruction parameters used.")
    lines.append("- `checksum.txt` — SHA-256 hash of every file.")
    if spec.extras:
        lines.append("")
        lines.append("Additional attachments:")
        for name in sorted(spec.extras.keys()):
            lines.append(f"- `{name}` — extra attachment.")
    if spec.notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(spec.notes)
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "Every result in this bundle was produced by the DHM "
        "Reconstruction tool. To reproduce, load the parameters "
        "from `params.json` into the v2.0.x app and rerun on the "
        "raw hologram(s) referenced in `raw_data.csv`'s "
        "``frame_path`` column."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_bundle(spec: BundleSpec, out_path: Path | str) -> Path:
    """Build ``out_path`` as a ZIP containing the bundle.

    Returns the resolved path. Existing files at ``out_path`` are
    overwritten (Zenodo bundles are atomic; partial archives
    aren't useful)."""
    if not spec.figures_pdf:
        raise ValueError("BundleSpec requires figures_pdf "
                         "to point at a real PDF")
    figures_pdf = Path(spec.figures_pdf).expanduser().resolve()
    if not figures_pdf.exists():
        raise FileNotFoundError(
            f"figures_pdf not found: {figures_pdf}",
        )

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build params.json text + README + checksum content first so
    # the zip write is one pass.
    params_text = json.dumps({
        "sample_id": spec.sample_id,
        "operator": spec.operator,
        "timestamp": spec.timestamp,
        **spec.params,
    }, indent=2, default=str, sort_keys=True)
    readme_text = bundle_manifest(spec)

    # Sums for every file we'll include — README + params written
    # later, so we sum what's already on disk.
    file_sums: list[tuple[str, str]] = []
    file_sums.append(("figures.pdf", _sha256_hex(figures_pdf)))
    if spec.raw_data_csv:
        csv_path = Path(spec.raw_data_csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"raw_data_csv not found: {csv_path}",
            )
        file_sums.append(("raw_data.csv", _sha256_hex(csv_path)))
    extras_resolved: Dict[str, Path] = {}
    for name, p in spec.extras.items():
        rp = Path(p).expanduser().resolve()
        if not rp.exists():
            raise FileNotFoundError(
                f"extra attachment {name} not found: {rp}",
            )
        extras_resolved[name] = rp
        file_sums.append((name, _sha256_hex(rp)))
    # In-memory generated files: hash them by hashing their bytes.
    file_sums.append((
        "params.json",
        hashlib.sha256(params_text.encode("utf-8")).hexdigest(),
    ))
    file_sums.append((
        "README.md",
        hashlib.sha256(readme_text.encode("utf-8")).hexdigest(),
    ))

    checksum_text = "\n".join(
        f"{sha}  {name}" for name, sha in sorted(file_sums)
    ) + "\n"

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(figures_pdf, arcname="figures.pdf")
        if spec.raw_data_csv:
            zf.write(Path(spec.raw_data_csv).expanduser().resolve(),
                     arcname="raw_data.csv")
        for name, p in extras_resolved.items():
            zf.write(p, arcname=name)
        zf.writestr("params.json", params_text)
        zf.writestr("README.md", readme_text)
        zf.writestr("checksum.txt", checksum_text)
    return out


__all__ = [
    "BundleSpec",
    "bundle_manifest",
    "build_bundle",
]
