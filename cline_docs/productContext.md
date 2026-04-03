# Product Context

## Problem Statement
Traditional DHM analysis software is often reliant on CUDA (NVIDIA) or unoptimized CPU code, making it inefficient on Mac devices. Researchers using Macs lack a high-performance, native solution that leverages Apple Silicon's unified memory and neural engine capabilities for rapid reconstruction and analysis of holographic data.

## User Experience
-   **Workflow**: Load Data -> GPU Process -> Analyze -> Report.
-   **Interface**: A modern, "Mac-like" GUI using PySide6.
-   **Interaction**: Drag-and-drop file loading, smooth zooming/panning, virtual refocusing via scroll wheel, real-time 3D manipulation.
-   **Efficiency**: Batch processing allows handling large datasets without manual intervention.

## Functional Requirements
1.  **Data Ingestion**
    -   Universal file support (.tiff, .fits, .npy, video).
    -   Smart metadata reading.
    -   Batch loading from folders.

2.  **Reconstruction & Unwrapping**
    -   Fast FFT/IFFT using MLX (Metal) with fallback options.
    -   Algorithm choice: Angular Spectrum Method (ASM) or Fresnel transform.
    -   Fourier space tools: automatic +1 order detection and optional manual masking.
    -   Automatic and interactive focusing (Z-scan auto-focus using Total Variation / Gradient metrics).
    -   Robust phase unwrapping (Goldstein, Flynn, etc.) running on Metal.

3.  **Analysis**
    -   AI-based cell segmentation (Cellpose/StarDist via CoreML).
    -   3D tracking of cells over time.
    -   Quantitative Phase Imaging (QPI) metrics: Dry Mass, Volume, Shape.

4.  **Visualization & Output**
    -   High-DPI (Retina) ready graphs.
    -   Interactive 3D topography.
    -   Side-by-side comparison views.
    -   Export to MATLAB, CSV, TIFF, ProRes/HEVC.
    -   Save intermediate outputs (spectrum, masks, focus curves) and final outputs (phase, amplitude, metrics).
