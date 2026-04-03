# System Patterns

## Architecture
The application follows a modular pipeline architecture:
`Data Loader -> Reconstruction Engine -> Unwrapping Module -> Analysis Core -> Visualization`

### Core Components
1.  **GUI Layer (PySide6)**
    -   Main Window with Dockable Widgets.
    -   Signal/Slot mechanism for async updates.
    -   Separation of UI and Worker threads.

2.  **Compute Layer (MLX / MPS / Metal)**
    -   **Reconstruction**: Handles Fresnel/Angular Spectrum propagation. Optimized with MLX for Apple Silicon. Supports algorithm selection (ASM vs Fresnel).
    -   **Fourier Masking**: Utilities for automatic +1 order detection and optional manual selection/masking.
    -   **Auto-Focus**: Z-scan with multiple focus metrics and safety coarse-to-fine scanning when overly large Z ranges are entered.
    -   **Unwrapping**: Custom Metal kernels or MPS shaders for phase unwrapping algorithms.
    -   **Memory**: Zero-copy data handling using Unified Memory.

3.  **AI Layer (CoreML)**
    -   Pre-trained models (Cellpose/StarDist) converted to CoreML format.
    -   Inference running on the Neural Engine.

4.  **Visualization Layer**
    -   **2D**: PyQtGraph for high-speed image display and plotting.
    -   **3D**: VisPy or PyMetal for hardware-accelerated surface rendering.

## Design Patterns
-   **Model-View-Controller (MVC)**: Separation of data (numpy/mlx arrays), logic (reconstruction algorithms), and UI.
-   **Pipeline Pattern**: Data flows through distinct processing stages.
-   **Worker Pattern**: Long-running tasks (batch processing, large FFTs) run in background threads to keep UI responsive.
-   **Factory Pattern**: For creating different types of file readers and reconstruction algorithms.

## Export Pattern
-   A dedicated exporter component saves intermediate and final outputs (arrays, images, tables) with a consistent naming scheme.
 -   Export is organized per run in a folder containing monitors (Input/Amplitude/Phase/Spectrum/Mask), numeric outputs, and a text report capturing key parameters.

## Post-Processing Pattern
-   Optional frequency-domain filtering (low-pass / high-pass / band-pass) can be applied to the reconstructed complex field prior to amplitude/phase display and export. UI exposes a master enable, filter type selection, cutoff(s), and smooth roll-off.

## Settings / Profiles Pattern
-   UI parameters can be saved and restored via named profiles. Profiles are stored as JSON under the user's home directory and can be switched via a dropdown (with Save/Delete actions) to quickly reproduce reconstructions.

## Auto-Focus Robustness Pattern
-   Z-scan auto-focus must tolerate non-finite recon outputs at some z values (NaN/Inf) and either select the best focus from the remaining finite scores or fail with a clear diagnostic message suggesting narrower z range / method switch / parameter checks.

## Physical Parameters Pattern
-   Propagation depends critically on wavelength and the effective pixel size at the reconstruction plane. Some datasets already provide an effective pixel size; others require scaling by system magnification. Provide a backward-compatible way to interpret `pixel_um` either as already-effective or as sensor pixel pitch with separate magnification.

## Quantitative Phase Pattern
-   Compute OPD from phase: `OPD_m = (wavelength_m / (2π)) * (phase_rad - background_phase_rad)`.
-   Convert OPD to thickness (optional): `thickness_m = OPD_m / (n_sample - n_medium)` when delta-n is known.
-   Compute dry mass surface density (optional): `sigma_kg_per_m2 = OPD_m / (dn/dc)` using user-provided `dn/dc` (typical proteins ~0.18 mL/g).
-   Total dry mass in ROI: `sum(sigma * pixel_area_m2)`; pixel area uses the same effective pixel size used for reconstruction.
-   Always provide a background/offset removal method (manual ROI or robust statistic) to avoid arbitrary phase offsets corrupting OPD.

## Technical Decisions
-   **MLX vs PyTorch**: MLX is chosen for its NumPy-like syntax and direct optimization for Apple Silicon, though PyTorch (MPS) is a fallback.
-   **FFT Backend Selection**: Prefer MLX (`mlx.core.fft`) for speed on Apple Silicon; allow fallback backends for portability.
-   **PySide6**: Chosen for native look and performance over Electron or web-based GUIs.
-   **Zero-Copy**: Critical for performance; avoiding data movement between "CPU RAM" and "GPU VRAM" by using Unified Memory pointers where possible.
