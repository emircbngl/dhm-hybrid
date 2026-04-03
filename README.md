# DHM Reconstruction (Mac)

A high-performance Digital Holographic Microscopy (DHM) analysis laboratory optimized for Apple Silicon.

## Features
- **Mac Optimization**: Leveraging MLX and Metal for GPU acceleration.
- **Data Ingestion**: Support for TIFF, FITS, NPY, and video formats.
- **Advanced Analysis**: Phase reconstruction, unwrapping, and AI-powered cell segmentation.
- **Modern GUI**: Native macOS interface built with PySide6.

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
