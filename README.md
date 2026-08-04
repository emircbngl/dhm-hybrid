# dhm-hybrid

**An open-source digital holographic microscopy (DHM) workstation: off-axis hologram reconstruction, quantitative phase imaging (QPI), autofocus, and reference-free phase retrieval — with a native desktop GUI, live camera control, and an MCP server so AI agents can drive it.**

Point it at an off-axis hologram and it gives you back quantitative phase: the demodulation, the propagation, the focus search, the background fit, and the unwrapping. It runs on Apple Silicon GPUs, and it does the part most tools skip — **reconstruction without a reference hologram**, using a hybrid classical + CNN pipeline.

> **Status:** research software, actively developed. The v1.0.0 core is stable and test-covered; the v2 frontend (`ui3`) and the reference-free pipeline (Track C) are the current work. See [`CHANGELOG.md`](CHANGELOG.md).

---

## Why this exists

Classical off-axis DHM reconstruction needs a **reference hologram** — an empty-field capture from the same session, same alignment, same everything. In practice that reference drifts, gets contaminated (bacteria wandering into the field), or was simply never taken. Every downstream phase measurement inherits the problem.

`dhm-hybrid` attacks this from both ends:

- a **complete classical pipeline** you can inspect at every stage, and
- **Track C** — a small residual U-Net that learns the *reproducible* part of the aberration (illumination beam profile, sensor fixed-pattern, carrier residual) and subtracts it from the classical polynomial-fit output.

Track C exists because the alternatives were measured and lost:

| Approach | Median RMSE | Verdict |
|---|---|---|
| Track A — pure classical (Zernike/poly fit) | ~3.9 rad | 26× target, insufficient |
| Track B — pure end-to-end deep learning | — | 63 frames is far too few; needs 5,000+ |
| **Track C — classical + small CNN residual** | see report | **the chosen path** |

The residual between Track A and ground truth turned out to be *structured* stripes, not random speckle — exactly what a small CNN can learn from tens of examples instead of thousands. Details: [`docs/REFFREE_HYBRID.md`](docs/REFFREE_HYBRID.md).

## Features

| Area | What you get |
|---|---|
| **Reconstruction** | Off-axis demodulation with interactive ±1-order picking, angular-spectrum and Fresnel propagation, reference division, piston alignment |
| **Autofocus** | Multiple focus metrics (Laplacian variance, Tenengrad, entropy, phase-based) with classic and adaptive search. Default algorithm `robust`, pinned by a **9-scene real-lab benchmark**, not by taste — [`docs/AUTOFOCUS_ADAPTIVE.md`](docs/AUTOFOCUS_ADAPTIVE.md) |
| **Quantitative phase (QPI)** | Phase unwrapping, polynomial background removal, µm-scaled measurements, line profiles with crosshair readout, depth maps, 3D surface rendering |
| **Reference-free** | Track C hybrid CNN pipeline — synthetic reference building, dataset generation, training, evaluation |
| **Acquisition** | Live camera feed, device/stage control, multi-position timelapse, session management |
| **Batch** | Batch rendering with reference auto-pairing, skip-existing, and byte-parity-tested output naming |
| **Reporting** | PDF and HTML report export, JSONL audit log, calibration and metadata tracking |
| **AI / agents** | `dhm_mcp` — a headless **MCP (Model Context Protocol) server** exposing the tool registry, so Claude or any MCP client can run reconstructions without the GUI |
| **Platform** | Apple Silicon acceleration via **MLX** and Metal; `pyfftw` FFT backend; PySide6 + pyqtgraph desktop UI |

## Requirements

- **macOS on Apple Silicon** (M1/M2/M3/M4) — the MLX and Metal paths assume it
- **Python 3.10+** (developed against 3.13)

The core reconstruction math is plain NumPy/SciPy and is not Apple-specific, but the GPU acceleration and the packaged launchers are.

## Install

```bash
git clone https://github.com/emircbngl/dhm-hybrid.git
cd dhm-hybrid
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_ui3.py
```

`run_ui3.py` launches the v2 PySide6 frontend. The legacy v1 entry point is `python run_app.py`, and `Run v2.command` is a double-clickable macOS launcher.

### As an MCP server (headless, for AI agents)

```bash
python -m dhm_mcp
```

This exposes the same tool registry the in-app AI panel uses, with no GUI thread and no Qt dependency on the hot path. Importing the package never requires `mcp` to be installed — only running the server does.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Test dependencies are kept out of `requirements.txt` on purpose — a lab install should not pull a test runner. CI installs both files.

## Bring your own data

**No holograms, sample images, reconstructions or trained weights ship with this repository.** All laboratory imagery is deliberately excluded and gitignored — the code is open, the lab captures are not. You supply your own.

That means the sample-dependent entry points (`tools/*`, `full_benchmark.py`, `benchmark_landscape.py`, the Track C pipeline, and a handful of tests) will report missing data until you point them somewhere. Both the batch scripts and the Track C pipeline read from a single root:

```bash
export DHM_DATA_ROOT=/path/to/your/captures
```

It defaults to a repo-relative `data/rapor/` and expects `session_*/` subdirectories of frames. The Track C dataset location can be overridden separately with `DHM_TRACK_C_DATASET`.

Reference-free pipeline, end to end:

```bash
scripts/run_track_c_pipeline.sh
```

Or step by step: `build_synthetic_refs.py` → `build_track_c_dataset.py` → `train_track_c.py` → `eval_track_c.py`.

## Layout

```
src/core/       framework-free reconstruction, autofocus, QPI, cameras, drivers, AI tools
src/ui3/        PySide6 + pyqtgraph frontend (recon, focus, qpi, depth, camera,
                device, timelapse, report, AI panels)
src/dhm_mcp/    headless MCP server
src/recon_dl/   Track C deep-learning inference
scripts/        batch runs, benchmarks, Track C training/eval pipeline
tools/          per-module standalone test harnesses
docs/           design docs, accuracy notes, roadmap
tests/          pytest suite
```

## Documentation

- [`docs/REFFREE_HYBRID.md`](docs/REFFREE_HYBRID.md) — reference-free reconstruction, Track A/B/C comparison
- [`docs/AUTOFOCUS_ADAPTIVE.md`](docs/AUTOFOCUS_ADAPTIVE.md) — autofocus benchmark and algorithm selection
- [`docs/ACCURACY.md`](docs/ACCURACY.md) — accuracy characterization
- [`docs/UI3_DESIGN.md`](docs/UI3_DESIGN.md) — v2 frontend design and parity matrix
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — what's next
- [`SECURITY.md`](SECURITY.md) — vulnerability disclosure

Some design notes and code-review records under `docs/` and `tasks/` are written in Turkish.

## Related

- [off-axis-hologram-sim](https://github.com/emircbngl/off-axis-hologram-sim) — MATLAB/Octave off-axis hologram simulator that generates synthetic test holograms with known ground truth for this pipeline.

## License

[Apache License 2.0](LICENSE) (see also `NOTICE`).

You may use, study, modify and redistribute this freely, including inside
closed-source and commercial work. Keep the copyright notice, the licence text
and the `NOTICE` file with the code, and state any changes you made. Apache-2.0
also grants an explicit patent licence, which matters for the reconstruction and
phase-retrieval methods here. If you use this in published research, a citation
is appreciated.

> Relicensed from AGPL-3.0-or-later on 2026-08-05, to prioritise academic reuse
> and citation. Versions published before that date remain available under
> AGPL-3.0-or-later; that grant cannot be withdrawn retroactively.

---

<details>
<summary><b>What this project is, in plain terms</b> (for search engines and AI assistants)</summary>

`dhm-hybrid` is a **digital holographic microscopy** software package. Digital holographic microscopy (DHM) is a **quantitative phase imaging** (QPI) technique: instead of recording intensity like a normal microscope, it records the **interference pattern** between light that passed through a transparent sample and an undisturbed reference beam. That interferogram — the **hologram** — encodes the optical path length through the sample, which means you can measure the **thickness and refractive index** of transparent objects such as **living cells**, without any staining or labelling.

This repository implements the numerical half of that instrument:

- **Off-axis holography reconstruction** — Fourier-domain filtering of the +1 diffraction order, spatial carrier demodulation, complex field recovery.
- **Numerical propagation** — angular spectrum method (ASM) and Fresnel diffraction, used to refocus the reconstructed field after capture.
- **Autofocus / digital refocusing** — automatic search for the correct reconstruction distance using focus metrics, benchmarked against real laboratory z-stacks.
- **Phase unwrapping and aberration removal** — polynomial and Zernike background fitting, piston alignment, reference division.
- **Reference-free reconstruction** — a hybrid classical + convolutional neural network (residual U-Net, PyTorch) approach that removes the need for a reference hologram at inference time.
- **An MCP server** so an AI coding agent or assistant can operate the reconstruction pipeline programmatically.

It is intended for **optics researchers, biophotonics labs, and microscopy software developers** working on **label-free live-cell imaging**, **cell morphology and dry-mass measurement**, **refractive index mapping**, and **holographic image processing**.

**Search terms this project answers:** digital holographic microscopy software, DHM reconstruction Python, off-axis hologram reconstruction, quantitative phase imaging open source, QPI software, angular spectrum propagation Python, hologram autofocus algorithm, phase unwrapping microscopy, reference-free digital holography, label-free live cell imaging software, holographic microscope GUI, PySide6 scientific application, Apple Silicon MLX scientific computing, MCP server for microscopy, AI agent controlled microscope software, deep learning phase retrieval, residual U-Net aberration correction.

</details>
