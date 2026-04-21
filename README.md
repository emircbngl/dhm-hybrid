# DHM Reconstruction (Mac)

**Version:** 1.0.0 · **Build:** 2026-04-20 · **Platform:** Apple Silicon only

A high-performance Digital Holographic Microscopy (DHM) analysis laboratory optimized for Apple Silicon.

See [`CHANGELOG.md`](CHANGELOG.md) for release notes and [`docs/ROADMAP.md`](docs/ROADMAP.md) for Faz 2 / Faz 3 commitments.

## Features
- **Mac Optimization**: Leveraging MLX and Metal for GPU acceleration.
- **Data Ingestion**: Support for TIFF, FITS, NPY, and video formats.
- **Advanced Analysis**: Phase reconstruction, unwrapping, and AI-powered cell segmentation.
- **Modern GUI**: Native macOS interface built with PySide6.
- **Traceability (v1.0.0)**: Audit log (JSONL) · HTML report export · About dialog with version/git SHA.

## Setup

1.  **Prerequisites**:
    -   macOS with Apple Silicon (M1/M2/M3/M4).
    -   Python 3.10+.

2.  **Installation**:
    ```bash
    # Create a virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Running the Application**:
    ```bash
    python src/main.py
    ```

## Development
-   Documentation is located in `cline_docs/`.
-   Source code is in `src/`.
-   Tests: `venv/bin/python -m pytest tests/` (25 tests, ~1s).

## Commercial Use

DHM Reconstruction is distributed for commercial lab deployment. See [`docs/COMMERCIAL.md`](docs/COMMERCIAL.md) for licensing terms, [`docs/SLA.md`](docs/SLA.md) for support commitments, and [`docs/ESCROW.md`](docs/ESCROW.md) for source-code escrow options. Security disclosures: [`SECURITY.md`](SECURITY.md).
