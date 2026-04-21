# Roadmap

## Faz 1 — v1.0.0 (2026-04-20, shipped)

Commercial foundations. See `CHANGELOG.md` for the full list.

- Versioning + changelog + About dialog
- Audit trail (JSONL)
- HTML report export
- Core robustness (NaN/edge-case guards, ENTROPY fallback)
- UX polish (right-click context menu, surfaced errors, Generate Report action)

## Faz 2 — v1.1 (planned, 8–12 weeks)

### AI Cell Segmentation
- Integrate Cellpose 3.x via PyTorch MPS (Apple Silicon GPU).
- Ship three pre-trained models: `cyto2`, `nuclei`, `livecell`.
- Segmentation panel in QPI tab: run segmentation, review masks, apply to height/mass maps.
- Export: per-cell CSV with area, volume, dry mass, perimeter, eccentricity, circularity.

### Licensing Framework
- Node-locked license file (RSA-signed JSON), validated on startup.
- Seat-based enforcement with optional floating license server.
- Trial mode (30 days, watermarked reports).
- License management UI in Help menu.

### Report Polish
- PDF output in addition to HTML (WeasyPrint).
- Configurable report templates (lab logo, custom header, per-experiment metadata).
- Batch report generation across a job queue.

## Faz 3 — v1.2 (planned, 12–16 weeks)

### Optical Correction
- Interactive reference-wave fitting (polynomial tilt/curvature subtraction).
- Aberration compensation (Zernike-basis fit against a flat-field hologram).
- Telecentric correction for off-axis setups with non-parallel reference wave.
- Multi-λ reconstruction with dispersion compensation.

### Multi-user & Data Management
- User accounts with RBAC (operator / analyst / admin).
- Audit log viewer with filters and export.
- Experiment hierarchy (project → session → hologram) in SQLite.
- HDF5 dataset export with full provenance.

### Throughput
- Parallel z-evaluation in autofocus (target 2.5–3.5× on Apple Silicon).
- GPU-resident FFT pipeline via MLX (end-to-end on device, no host roundtrips).
- Remote batch submission to a compute node.

---

## Out of scope (not planned)

- Windows/Linux builds — Apple Silicon only for v1.x. Revisit post-v2.
- Cloud upload / SaaS mode.
- Custom acquisition hardware SDKs beyond the current NI-IMAQdx integration.
