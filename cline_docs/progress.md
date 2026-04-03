# Progress Status

## Todo List
-   [x] Initialize Memory Bank
    -   [x] projectbrief.md
    -   [x] productContext.md
    -   [x] systemPatterns.md
    -   [x] techContext.md
    -   [x] activeContext.md
-   [x] Project Setup
    -   [x] Create virtual environment (Instructions provided in README)
    -   [x] Create `requirements.txt`
    -   [ ] Initialize git repository (User to execute)
    -   [x] Create source directory structure
    -   [x] Create .gitignore
-   [ ] Phase 1: Basic GUI & Data Ingestion
    -   [x] Main Window skeleton (PySide6)
    -   [x] File loading mechanism (Drag & Drop)
    -   [x] Basic ingestion module (TIFF/PNG/NPY/video)
    -   [x] Metadata reader (baseline: TIFF tags + ImageDescription parsing)
-   [x] Phase 2: Reconstruction Engine
    -   [x] Implement FFT/IFFT with MLX (FFT backend selector)
    -   [x] Basic propagation algorithms (ASM/Fresnel)
    -   [x] Fourier +1 order masking (auto detect + manual center)
    -   [x] Auto-Focus (Z-scan with Total Variation / Gradient)
    -   [x] Output saving/export helpers (NPY/CSV, optional TIFF/MAT)
-   [ ] Phase 3: Visualization
    -   [x] Integrate PyQtGraph for 2D display
    -   [x] Add Fourier spectrum monitor (guide) with order center/mask overlay
    -   [x] Add optional post-reconstruction frequency-domain filters (HP/LP/BP)
-   [ ] Phase 4: Unwrapping & Analysis
-   [ ] Phase 5: Optimization & Export

## Status
### Completed
- Export saves monitors and arrays per run folder with report.txt.
- Added multiple autofocus metrics and safety coarse-to-fine scanning for overly wide Z ranges.
- Added post-reconstruction frequency filters with smooth roll-off.
- Added magnification + effective pixel size handling.
- Added settings persistence with named profiles (save/load/delete).
- Reworked UI into a DaVinci/Premiere-like dock workspace (nested/tabbed docks enabled) with a right-side Settings dock and an Info dock under Settings.
- Monitor panels are dockable/floatable and user layout is persisted across app restarts (QMainWindow save/restore geometry + state).
- Quantitative Phase (QP) output monitors (OPD/Thickness/Dry mass) are hidden unless QP is enabled, and their dock placement is preserved when enabling QP after restart.
- Export button moved to the bottom of the Settings panel.
- Interactive line-profile tool (Line tool): click to place a draggable line ROI on any monitor and live-update a Line Graph dock.
- Export-only crop tool: toolbar Crop toggle + ROI placement by click on any monitor; crop is applied only at export time.
- Scale bar overlay (µm): image-attached overlay with auto/manual length control and inclusion in exported PNGs.
- Batch/Render: floating dock with multi-file queue, output folder selection, and sequential run/stop to reconstruct + export.

### Recently Fixed
- ASM propagation could produce NaNs for negative z due to evanescent exponential growth in the transfer function; this caused amplitude/phase monitors and exported recon outputs to appear black.

### In Progress
- Auto-focus reliability: investigate score curve quality (flat/monotonic), NaN handling, metric behavior across z, and improve UI feedback when scan data is not valid.
- Reconstruction quality regression after magnification/effective pixel size scaling: verify physical parameter interpretation, adjust defaults, and add backward-compatible option if needed.

### Next
- Quantitative phase analysis for phase objects: OPD / thickness / dry mass maps + export.
- Manual mask selection UI for +1 order center (spectrum click).
