# Project Brief: DHM Reconstruction (Mac)

## Project Overview
This project aims to build a high-performance, Mac-native Digital Holographic Microscopy (DHM) data analysis laboratory. It is a post-processing software designed to run on Apple Silicon hardware (M1/M2/M3/M4), leveraging specific accelerators like MLX, MPS, and the Neural Engine to outperform traditional CUDA-based solutions on Mac devices.

## Core Goals
1.  **Mac Optimization**: Maximize performance using Apple Silicon features (Unified Memory, Metal, Neural Engine).
2.  **Complete Analysis Pipeline**: From Data Ingestion -> Reconstruction -> Unwrapping -> Analysis -> Visualization -> Export.
3.  **User Experience**: Modern, native macOS interface (PySide6) with drag-and-drop and fluid interaction.
4.  **No Acquisition**: Purely focused on analyzing existing data (images/videos) efficiently.

## Key Features
-   **Data Ingestion**: Support for multiple formats (.tiff, .fits, .npy, video), batch processing, and metadata extraction.
-   **Reconstruction Engine**: 
    -   Choice between **Angular Spectrum Method (ASM)** and **Fresnel Transform**.
    -   **Fourier Space Automation**: Automatic detection or manual masking of the +1 diffraction order.
    -   **Auto-Focus**: Parallel Z-scan using "Total Variation" or "Gradient" metrics.
    -   **Virtual Refocusing**: Interactive focus adjustment.
-   **Phase Unwrapping**: Metal-optimized algorithms (Goldstein, Least-Squares, Flynn).
-   **Cell Analysis**: AI-powered segmentation (CoreML) and 3D tracking.
-   **Visualization**: Real-time 2D/3D rendering (PyQtGraph, VisPy/PyMetal).
-   **Export**: Save results as images, videos, or raw data structures.

## Technical Constraints
-   **Platform**: macOS (Apple Silicon optimized, M1-M4).
-   **Language**: Python.
-   **GUI Framework**: PySide6.
-   **Compute**: MLX, MPS, CoreML (Prioritizing fastest available FFT implementation).
