# Technology Context

## Development Environment
-   **OS**: macOS (Apple Silicon, M1–M4).
-   **Language**: Python 3.x.
-   **IDE**: VS Code / Cursor (implied).

## Core Libraries & Frameworks
-   **GUI**: `PySide6` (Qt for Python).
-   **Math/Compute**: `mlx` (Apple Machine Learning), `numpy`, `torch` (with MPS support).
-   **Visualization**: `pyqtgraph`, `vispy`, `matplotlib` (for static exports).
-   **AI/ML**: `coremltools`, `Cellpose`/`StarDist` (models).
-   **Image Processing**: `scikit-image`, `opencv-python`.

## FFT Strategy
-   **Primary**: `mlx.core.fft` (Metal-backed) for 2D FFT/IFFT in reconstruction.
-   **Fallback**: `numpy.fft` when MLX is unavailable or input types are incompatible.
-   **Optional**: FFTW (`pyFFTW`) can be supported as an opt-in backend, but may require extra build steps on macOS arm64.

## Hardware Acceleration
-   **GPU**: Used via MLX and Metal Performance Shaders (MPS).
-   **NPU**: Apple Neural Engine used via CoreML for segmentation.
-   **Memory**: Unified Memory Architecture (UMA) optimization.

## File Formats
-   **Input**: .tiff, .tif (stack), .fits, .npy, .mp4, .avi, .mov.
-   **Output**: .mat, .csv, .tiff, ProRes/HEVC video.

## Notes
-   Some TIFF variants (e.g., LZW-compressed) require the optional dependency `imagecodecs` when using `tifffile`.

## Key Constraints
-   Must run natively on Apple Silicon.
-   Dependencies must be compatible with arm64 architecture.
