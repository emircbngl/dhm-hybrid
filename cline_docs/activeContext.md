# Active Context

## Current Focus
Phase 1 implementation: Data ingestion and a Mac-optimized reconstruction core with ASM/Fresnel selection, Fourier +1 order masking tools, auto-focus (Z-scan with multiple focus metrics + safety coarse-to-fine scanning for overly large Z ranges), and a researcher-friendly monitoring/export system (including Fourier spectrum guide and per-run reports).

## Recent Changes
-   Created `cline_docs/` directory.
-   Initialized `projectbrief.md`, `productContext.md`, `systemPatterns.md`, and `techContext.md`.
-   Updated scope to include Apple Silicon M4 compatibility, FFT backend strategy, reconstruction algorithm selection, masking automation, auto-focus metrics, and output saving.
 -   Added sample holograms in `Örnekler/` for local testing. Note: some TIFFs may require `imagecodecs` (e.g., LZW compression).
 -   Extended auto-focus metrics (Tenengrad, Laplacian variance, Brenner, Entropy) and improved robustness for dense/large-range scans.
 -   Added auto-focus safety: overly wide Z ranges trigger a warning and use coarse-to-fine scanning within a safe mm window.
 -   Export now includes an additional `*_autofocus_all_metrics.csv` file with columns for all focus metrics.
 -   Export monitor images are saved as PNG.
 -   Added optional post-reconstruction frequency-domain filters with a cleaner UI (master enable + filter type dropdown) and a configurable smooth roll-off to make noise reduction effects more visible.
 -   Reworked UI into a DaVinci/Premiere-like dock workspace: Settings is a right-side dock with scroll, Info is docked under Settings by default, and monitor views are dockable/floatable panels.
 -   Dock layout is persisted across app restarts via QMainWindow save/restore geometry + state.
 -   Quantitative Phase (QP) output monitors (OPD/Thickness/Dry mass) are hidden unless QP is enabled; their placement is preserved even if QP is enabled after a restart.
 -   Moved Export button to the bottom of the Settings panel.
 -   Added magnification and effective pixel size handling: propagation and auto-focus now use effective pixel size = pixel_um / magnification; export report and info panel include magnification and effective pixel size.
 -   Added settings persistence with named profiles (Profile dropdown + Save/Delete) stored in `~/.dhm_reconstruction_profiles.json`.
 -   Hardened reconstruction display: amplitude/phase monitors now force auto-range and surface errors (non-finite outputs, shape issues) in the status bar and info panel instead of failing silently.
 -   Fixed ASM propagation NaNs for negative z by enforcing evanescent decay with |z|, preventing black amplitude/phase outputs and NaN focus scores.
 -   Investigating reconstruction quality regression reported after adding magnification/effective pixel size scaling; may require backward-compatible mode (treat pixel size as already-effective) and/or safer defaults.
 -   Added scale bar overlay (µm) with auto/manual length control and inclusion in exported PNGs.
 -   Added Batch/Render floating dock with multi-file queue, output folder selection, and sequential run/stop.

## New Focus
-   Add quantitative phase analysis for phase objects (cells, microspheres): compute OPD, thickness, and dry mass from reconstructed phase while keeping the current reconstruction + autofocus workflow stable.
-   Add required physical parameters in UI: refractive index of medium/sample (or delta-n), specific refractive increment (dn/dc) for dry mass, and immersion context (air/water/oil) as user-provided metadata for correct interpretation.

## Planned UI Features
-   DaVinci-like secondary page/tab (e.g. Batch/Render) to run bulk reconstructions and export results.

## Next Steps
1.  Implement file loading (drag & drop) and metadata extraction.
2.  Implement reconstruction core (ASM/Fresnel) using MLX FFT.
3.  Implement Fourier +1 order masking (auto + manual).
4.  Implement Z-scan auto-focus (TV/Gradient metrics).
5.  Implement output saving/export (arrays/images/tables).

## Active Decisions
-   Adopting `cline_docs` structure for documentation/memory.
-   Prioritizing MLX for math operations to leverage Apple Silicon capabilities immediately.
