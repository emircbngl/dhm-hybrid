# DHM Reconstruction App — UI Refactor & New Features PRD

## 1. Context

### Current state
A PySide6 + PyQtGraph desktop application for digital holographic microscopy (DHM) reconstruction. The app currently works but suffers from a cramped single-panel layout where all settings (reconstruction, preprocessing, filters, autofocus, quantitative phase, masking) are stacked in one scrollable right sidebar. When quantitative phase is enabled, the sidebar becomes unusable.

### Tech stack
- **GUI framework:** PySide6 (Qt6) + PyQtGraph (`pg.ImageView`) for image panels
- **Backend:** Python — modular `src/core/` with separate files for reconstruction, autofocus, masking, offaxis, FFT, ingestion, export, metadata, julia_bridge
- **Entry point:** `run_app.py` → `src/main.py`
- **GUI code:** Currently in `src/main.py` (monolithic). `src/gui/` directory exists but is empty.

### Project structure (relevant parts)
```
src/
  core/
    autofocus.py        # Focus metrics (TV, Brenner, Entropy, Laplacian, Tenenbaum)
    reconstruction.py   # ASM/Fresnel propagation
    masking.py          # Order mask creation
    offaxis.py          # Off-axis hologram processing
    fft_backend.py      # FFT operations
    ingestion.py        # Image loading
    exporter.py         # Save results
    metadata_reader.py  # Read image metadata
    julia_bridge.py     # Julia interop (future)
    __init__.py
  gui/                  # EMPTY — to be populated by this refactor
  analysis/
  utils/
  main.py              # Current monolithic GUI + app logic
```

---

## 2. Goals

1. **Declutter the UI** — reorganize settings from one long sidebar into tabbed panels
2. **Flexible image layout** — user-selectable grid arrangements, persisted across sessions, with reset capability
3. **Merged phase panel** — wrapped/unwrapped in same panel with toggle, reducing panel count
4. **Add live camera acquisition** — NI-IMAQdx camera support with continuous/triggered modes, FPS control, and in-app exposure/gain settings
5. **Add adaptive autofocus** — automatic focus tracking during live/video acquisition
6. **Add video/snapshot recording** — capture frames during acquisition
7. **Modularize GUI code** — split monolithic `main.py` into separate GUI modules
8. **Dual-mode operation** — File mode (load image → process) and Live mode (camera → continuous reconstruct)
9. **Keep it simple** — functional, not polished. No unnecessary visual flourish.

---

## 3. New GUI Architecture

### 3.1 File structure

Split `src/main.py` into the following modules inside `src/gui/`:

```
src/gui/
  __init__.py
  main_window.py        # QMainWindow — assembles everything
  toolbar.py            # Top toolbar (Profile, Save, Batch, tools, Record, Snapshot, Mode switch)
  status_bar.py         # Bottom status bar with module badges
  image_grid.py         # Image panel grid manager (layout switching, persistence)
  panels/
    __init__.py
    input_panel.py      # Hologram input display (pg.ImageView)
    amplitude_panel.py  # Amplitude display
    phase_panel.py      # Phase display with wrapped/unwrapped toggle
    spectrum_panel.py   # FFT spectrum display
  sidebar/
    __init__.py
    sidebar_tabs.py     # QTabWidget container for all setting tabs
    camera_tab.py       # Camera connection, acquisition mode, exposure, gain, FPS
    recon_tab.py        # Reconstruction settings (method, wavelength, magnification, pixel, z, masking)
    process_tab.py      # Preprocessing + Filters
    focus_tab.py        # One-shot autofocus + Adaptive autofocus
    quant_tab.py        # Quantitative phase + ROI placement
    record_tab.py       # Video recording + Snapshot settings
  dialogs/
    __init__.py
    layout_picker.py    # Grid layout selection dialog
  workers/
    __init__.py
    acquisition_worker.py      # QThread for camera frame acquisition loop
    reconstruction_worker.py   # QThread for reconstruction (single + continuous)
    autofocus_worker.py        # QThread for one-shot autofocus
    adaptive_focus_worker.py   # QThread for adaptive focus tracking
    recording_worker.py        # QThread for video/snapshot capture

src/core/
  ... (existing files) ...
  camera.py             # NEW — NI-IMAQdx camera abstraction layer
```

### 3.2 Main window layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Toolbar                                                          │
│ (File│Live) [Profile▾] [Save] [Batch] │ [Line][Crop][ROI] │ [⏺Rec][📷] │ ☑Scale bar │
├──────────────────────────────────────────────┬───────────────────┤
│                                              │  Sidebar          │
│          Image Grid Area                     │  ┌─────────────┐  │
│                                              │  │  Tab Bar     │  │
│  ┌──────────┐ ┌──────────┐                   │  │Camera│Recon │  │
│  │  Input   │ │Amplitude │                   │  │Proc │Focus │  │
│  │(hologram)│ │          │                   │  │Quant│Record│  │
│  └──────────┘ └──────────┘                   │  ├─────────────┤  │
│  ┌──────────┐ ┌──────────┐                   │  │             │  │
│  │  Phase   │ │ Spectrum │                   │  │  (tab body) │  │
│  │ [W] [UW] │ │          │                   │  │             │  │
│  └──────────┘ └──────────┘                   │  │             │  │
│                                              │  └─────────────┘  │
│  Layout: [2x2 ▾] [⛶ Maximize]               │                   │
├──────────────────────────────────────────────┴───────────────────┤
│ Status: cam0 30fps │ 1200×1600 │ uint16 │  AF: ON │ REC: ● 142  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 Mode switch — File vs Live

The toolbar contains a segmented toggle at the far left: `(File │ Live)`

**File mode** (current behavior):
- Load image via File → Open or drag-and-drop
- Manual reconstruct button active
- Camera tab shows "No camera needed in File mode"
- Adaptive autofocus works on loaded image sequences (folder of frames)

**Live mode:**
- Camera tab becomes the starting point
- Input panel shows live camera feed (raw hologram)
- Reconstruction runs continuously on each frame (or every Nth frame for performance)
- Amplitude, Phase, Spectrum panels update in real-time
- Manual "Reconstruct" button is replaced by "Live Reconstruct" toggle
- Adaptive autofocus and recording operate on the live stream
- If no camera is connected, shows "Connect a camera to start"

**Mode switch behavior:**
- Switching from File → Live: stops any file-based processing, attempts camera connection
- Switching from Live → File: stops acquisition, disconnects camera gracefully
- Last used mode is persisted via QSettings

### 3.4 Sidebar tabs — content specification

#### Tab 0: Camera (Acquisition) — NEW

```
─── Connection ───
Camera          [cam0 (NI-IMAQdx) ▾]  ← auto-detected list
[  Connect  ]  /  [  Disconnect  ]
Status: ● Connected (1200×1600, mono16)

─── Acquisition mode ───
Mode            [Continuous   ▾]   ← Continuous / Triggered
Trigger source  [Software     ▾]   ← only visible in Triggered mode
FPS target      [30           ]
☑ Limit FPS (unchecked = max speed)

─── Camera settings ───
Exposure (µs)   [════════○═══] 5000
Gain (dB)       [══○═════════] 0.0
Bit depth       [Mono16       ▾]   ← Mono8 / Mono16

─── Live reconstruction ───
☑ Auto-reconstruct each frame
Reconstruct every [1] frame(s)    ← skip frames for performance
┌───────────────────────────────┐
│ Actual FPS: 28.3              │
│ Recon FPS:  28.3              │
│ Dropped:    0                 │
│ Frame:      #1247             │
└───────────────────────────────┘

[  ▶ Start acquisition  ]  /  [  ⏹ Stop  ]
```

**Camera detection:** On app launch (Live mode) or when Camera tab is opened, enumerate NI-IMAQdx interfaces using `niimaqdx.IMAQdxEnumerateCameras()`. Show found cameras in dropdown. If none found, show "No NI cameras detected" with a [Refresh] button.

**FPS control:** When "Limit FPS" is checked, the acquisition loop uses a timer to throttle frame grabbing. When unchecked, grabs at maximum camera rate.

**Live reconstruction toggle:** "Auto-reconstruct each frame" enables continuous reconstruction pipeline. The "every N frames" spinner allows skipping frames for slower machines (e.g., reconstruct every 3rd frame = display updates at 10fps while camera runs at 30fps). Skipped frames are still available for recording if recording is active.

#### Tab 1: Recon (Reconstruction) — unchanged

```
─── Reconstruction ───
Method          [ASM        ▾]
Wavelength (nm) [632,800     ]
Magnification   [1,000000    ]
Pixel (µm)      [1,000000    ]
☑ Pixel is already effective

─── Propagation ───
z (mm)          [55,750000   ]
[    Reconstruct    ]

─── Masking ───
+1 order radius (px)  [80  ]
```

#### Tab 2: Process — unchanged (Preprocessing + Filters)

```
─── Preprocessing ───
☑ Subtract mean
☐ Hann window

─── Filters ───
☐ Enable
Type            [Low-pass    ▾]
Cutoff (0..0.5) [0,1500      ]
Rolloff (0..0.2)[0,0200      ]
```

#### Tab 3: Focus — unchanged (Autofocus)

```
─── One-shot autofocus ───
Metric          [Total Variation ▾]
Z-scan min (mm) [-50,000000  ]
Z-scan max (mm) [50,000000   ]
Z-scan steps    [21          ]
[   Auto-focus (one-shot)   ]

─── Adaptive auto-focus ───
☐ Enable adaptive tracking    ← toggles section below

  Check every N frames  [10     ]
  Drift threshold (%)   [15     ]
  Local window (% range)[10     ]
  ☑ Adaptive N
  Fail limit (K)        [3      ]
  Timeout (frames)      [500    ]
  ┌─────────────────────────────┐
  │ Tracking active             │
  │ Z₀ = 55.75mm  S_ref = 0.832│
  │ Last check: frame 1240 — OK│
  └─────────────────────────────┘
```

**Adaptive autofocus status box** appears only when tracking is enabled. Shows real-time Z₀, S_ref, last check result. Background: subtle green tint when OK, red tint when drift detected.

#### Tab 4: Quant — unchanged (Quantitative Phase)

```
─── Quantitative phase ───
☐ Enable
n (medium)      [1,333000    ]
n (sample)      [1,380000    ]
dn/dc (mL/g)   [0,180000    ]

─── ROI placement ───
BG ROI          [Place]
Analysis ROI    [Place]
```

#### Tab 5: Record (Video + Snapshot)

```
─── Video recording ───
Format          [TIFF stack  ▾]
Save channel    [All         ▾]
Max frames      [1000        ]
Output dir      [Browse      ]

─── Snapshot ───
Format          [PNG         ▾]
☑ Include colorbar
☐ Include scale bar
[   Take snapshot   ]
```

---

## 4. Image Grid System

### 4.1 Available layouts

The user can choose from preset grid arrangements. Each panel (Input, Amplitude, Phase, Spectrum) can be shown or hidden.

| Layout name | Grid | Description |
|-------------|------|-------------|
| 2×2 (default) | 2 columns, 2 rows | All 4 panels equal size |
| 1+2 | 1 large left + 2 stacked right | Input large, Amplitude + Phase right |
| Single | 1 panel fullscreen | User picks which panel |
| 3+1 | 3 panels top + 1 bottom strip | Useful for Line tool with profile below |
| 1×4 horizontal | 4 panels side by side | For wide monitors |

### 4.2 Layout switching

- A small dropdown/combobox in the image grid area corner: `Layout: [2×2 ▾]`
- Double-clicking a panel header maximizes it to fill the entire grid area (toggle back with double-click or Esc)
- Keyboard shortcut: `Ctrl+1` through `Ctrl+4` to maximize individual panels, `Ctrl+0` to restore grid

### 4.3 Persistence

Use `QSettings` to save/restore:

```python
# Save
settings = QSettings("DHM", "Reconstruction")
settings.setValue("grid/layout", "2x2")
settings.setValue("grid/last_maximized", None)
settings.setValue("window/geometry", self.saveGeometry())
settings.setValue("window/state", self.saveState())
settings.setValue("sidebar/last_tab", self.sidebar.currentIndex())

# Restore
layout = settings.value("grid/layout", "2x2")
geometry = settings.value("window/geometry")
state = settings.value("window/state")
```

### 4.4 Reset mechanism

- Menu: `View → Reset Layout` restores factory defaults (2×2, sidebar visible, default sidebar tab)
- Also clears `QSettings` geometry/state entries
- Keyboard shortcut: `Ctrl+Shift+R`

---

## 5. Phase Panel — Wrapped/Unwrapped Toggle

The current two separate panels (Wrapped Phase + Unwrapped Phase) merge into one `PhasePanel`:

```python
class PhasePanel(QWidget):
    """
    Single panel showing either wrapped or unwrapped phase.
    Toggle via two small buttons at bottom-left of the panel.
    """
    def __init__(self):
        super().__init__()
        self.image_view = pg.ImageView()

        # Toggle buttons
        self.btn_wrapped = QPushButton("Wrapped")
        self.btn_unwrapped = QPushButton("Unwrapped")
        # Style: active button gets highlight color, inactive stays muted

        self.wrapped_data = None    # np.ndarray
        self.unwrapped_data = None  # np.ndarray
        self.current_mode = "wrapped"

    def set_mode(self, mode: str):
        """Switch between 'wrapped' and 'unwrapped'."""
        self.current_mode = mode
        if mode == "wrapped" and self.wrapped_data is not None:
            self.image_view.setImage(self.wrapped_data)
        elif mode == "unwrapped" and self.unwrapped_data is not None:
            self.image_view.setImage(self.unwrapped_data)
        self._update_button_styles()
```

---

## 6. Camera Acquisition Engine — NI-IMAQdx

### 6.1 Dependencies

```
# requirements.txt additions
niimaqdx          # NI-IMAQdx Python bindings (via NI DAQmx / Vision Acquisition Software)
```

**Install prerequisite:** NI Vision Acquisition Software must be installed on the machine. This provides the IMAQdx driver and the Python bindings. Without it, the app runs in File mode only (camera features disabled gracefully).

### 6.2 Camera abstraction layer — `src/core/camera.py`

```python
"""
NI-IMAQdx camera abstraction.
Provides a clean interface independent of the specific NI SDK version.
Falls back gracefully if NI drivers are not installed.
"""

try:
    import niimaqdx
    NI_AVAILABLE = True
except ImportError:
    NI_AVAILABLE = False


class CameraError(Exception):
    """Raised on camera connection/acquisition failures."""
    pass


class NICamera:
    """
    Wraps NI-IMAQdx camera operations.
    
    Lifecycle:
        cam = NICamera("cam0")
        cam.open()
        cam.configure(exposure_us=5000, gain_db=0.0, bit_depth="mono16")
        cam.start_acquisition(mode="continuous")
        frame = cam.grab_frame()  # returns np.ndarray
        ...
        cam.stop_acquisition()
        cam.close()
    """

    def __init__(self, camera_name: str):
        if not NI_AVAILABLE:
            raise CameraError("NI-IMAQdx drivers not installed")
        self.camera_name = camera_name
        self.session = None
        self.is_open = False
        self.is_acquiring = False
        self._resolution = (0, 0)   # (height, width)
        self._bit_depth = "mono16"

    @staticmethod
    def enumerate_cameras() -> list[dict]:
        """
        Return list of available NI cameras.
        Each dict: {"name": "cam0", "model": "...", "serial": "...", "bus": "..."}
        Returns empty list if NI drivers not installed.
        """
        if not NI_AVAILABLE:
            return []
        try:
            cameras = niimaqdx.IMAQdxEnumerateCameras(connected_only=True)
            return [
                {
                    "name": cam.InterfaceName,
                    "model": getattr(cam, "ModelName", "Unknown"),
                    "serial": getattr(cam, "SerialNumberHigh", ""),
                    "bus": getattr(cam, "BusType", ""),
                }
                for cam in cameras
            ]
        except Exception:
            return []

    def open(self):
        """Open camera session."""
        ...

    def configure(self, exposure_us: float = 5000, gain_db: float = 0.0,
                  bit_depth: str = "mono16"):
        """Set camera acquisition parameters."""
        ...

    def start_acquisition(self, mode: str = "continuous"):
        """
        Start frame acquisition.
        mode: "continuous" or "triggered"
        """
        ...

    def grab_frame(self) -> "np.ndarray":
        """
        Grab single frame. Blocks until frame available.
        Returns: numpy array (H, W) for mono, dtype uint8 or uint16.
        Raises CameraError on timeout or failure.
        """
        ...

    def stop_acquisition(self):
        """Stop acquisition."""
        ...

    def close(self):
        """Close camera session and release resources."""
        ...

    def get_attribute(self, attr_name: str):
        """Read a camera attribute (e.g., 'AcquisitionAttributes::Timeout')."""
        ...

    def set_attribute(self, attr_name: str, value):
        """Write a camera attribute."""
        ...

    @property
    def resolution(self) -> tuple[int, int]:
        """(height, width) of frames."""
        return self._resolution

    @property
    def dtype(self) -> str:
        """'uint8' or 'uint16' depending on bit depth."""
        return "uint16" if "16" in self._bit_depth else "uint8"
```

### 6.3 Acquisition worker — `src/gui/workers/acquisition_worker.py`

```python
class AcquisitionWorker(QThread):
    """
    Dedicated thread for camera frame grabbing.
    Runs a tight loop: grab frame → emit signal → optional FPS throttle.
    
    CRITICAL: This thread does NOT do reconstruction.
    It only grabs raw frames and passes them to the main thread via signals.
    Reconstruction happens in a separate worker to keep acquisition at full speed.
    """

    frame_ready = Signal(object, int)      # (np.ndarray, frame_number)
    fps_updated = Signal(float)            # actual measured FPS
    error_occurred = Signal(str)           # error message
    acquisition_started = Signal()
    acquisition_stopped = Signal()

    def __init__(self, camera: NICamera):
        super().__init__()
        self.camera = camera
        self.running = False
        self.target_fps = 30        # 0 = unlimited
        self.frame_number = 0

    def run(self):
        self.running = True
        self.frame_number = 0
        self.acquisition_started.emit()

        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        fps_counter = 0
        fps_timer_start = time.perf_counter()

        while self.running:
            loop_start = time.perf_counter()
            try:
                frame = self.camera.grab_frame()
                self.frame_number += 1
                self.frame_ready.emit(frame, self.frame_number)

                # FPS measurement
                fps_counter += 1
                elapsed = time.perf_counter() - fps_timer_start
                if elapsed >= 1.0:
                    self.fps_updated.emit(fps_counter / elapsed)
                    fps_counter = 0
                    fps_timer_start = time.perf_counter()

                # FPS throttle
                if frame_interval > 0:
                    sleep_time = frame_interval - (time.perf_counter() - loop_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except CameraError as e:
                self.error_occurred.emit(str(e))
                break

        self.acquisition_stopped.emit()

    def stop(self):
        self.running = False
```

### 6.4 Live pipeline — frame routing

When in Live mode, frames flow through a pipeline:

```
Camera (AcquisitionWorker thread)
  │
  ├─→ frame_ready signal
  │
  ▼
MainWindow.on_frame_ready(frame, frame_num)     [main thread]
  │
  ├──→ Input panel: display raw hologram          [always, direct]
  │
  ├──→ Reconstruction pipeline (if auto-reconstruct ON):
  │      if frame_num % reconstruct_every == 0:
  │          ReconstructionWorker.process(frame)
  │              │
  │              ├─→ Amplitude panel: update
  │              ├─→ Phase panel: update
  │              └─→ Spectrum panel: update
  │
  ├──→ Adaptive autofocus (if enabled):
  │      AdaptiveFocusWorker.on_new_frame(frame)
  │          └─→ may update z_best → next reconstruction uses new z
  │
  └──→ Recording (if active):
         RecordingWorker.enqueue_frame({
             "input": frame,
             "amplitude": current_amplitude,   # from last recon
             "phase": current_phase,
             "frame_num": frame_num
         })
```

**Performance strategy:**
- Acquisition thread runs at camera speed (never blocks)
- Reconstruction runs in separate thread, drops frames if previous recon isn't finished
- UI updates are debounced — max 30fps display refresh regardless of camera speed
- Recording captures ALL frames (or raw input frames) without dropping

### 6.5 Graceful degradation

If NI drivers are not installed (`NI_AVAILABLE = False`):
- Camera tab shows: "NI Vision Acquisition Software not detected. Install it for camera support. File mode is fully functional."
- Mode switch still shows File/Live but Live is grayed out
- All other features (File mode, autofocus on loaded images, reconstruction) work normally
- No import errors, no crashes

---

## 7. Adaptive Autofocus — Algorithm Specification

### 6.1 Overview

Three-layer system:
1. **Cold start** — full Z-range scan on first run
2. **Periodic monitoring** — check focus quality every N frames
3. **Smart re-scan** — drift detected → local search with exclusion list

### 6.2 State machine

```
States:
  IDLE          — autofocus not active
  CALIBRATING   — initial full scan in progress
  MONITORING    — periodic check mode (normal operation)
  RESCANNING    — drift detected, local search in progress
  FAILED        — all zones exhausted, using best available
```

### 6.3 Data structures

```python
@dataclass
class AdaptiveFocusState:
    z_best: float = 0.0             # Current best Z position (Z₀)
    score_ref: float = 0.0          # Reference metric score (S_ref)
    score_ref_std: float = 0.0      # Std dev from calibration (for auto-threshold)
    frame_count: int = 0            # Frames since last check
    check_interval: int = 10        # N — frames between checks
    base_interval: int = 10         # Original N (for adaptive reset)
    drift_threshold: float = 0.15   # T — relative drift threshold
    local_window_pct: float = 0.10  # δ as percentage of total Z range
    stable_streak: int = 0          # Consecutive OK checks
    fail_streak: int = 0            # Consecutive failed re-scans
    fail_limit: int = 3             # K — max failures before exclusion reset
    timeout_frames: int = 500       # M — frames before exclusion reset
    frames_since_success: int = 0   # Counter for timeout
    excluded_zones: list = field(default_factory=list)  # [(z_min, z_max), ...]
    state: str = "IDLE"             # Current state machine state
```

### 6.4 Core algorithm (pseudocode)

```
function on_new_frame(frame, metric_func, z_range):
    state.frame_count += 1
    state.frames_since_success += 1

    if state.state == IDLE:
        return  # do nothing

    if state.state == CALIBRATING:
        perform_full_scan(z_range, metric_func)
        state.z_best = best_z
        state.score_ref = best_score
        calibrate_threshold(metric_func)  # optional: 50-100 frames → auto T
        state.state = MONITORING
        return

    if state.state == MONITORING:
        if state.frame_count < state.check_interval:
            return  # not time yet

        state.frame_count = 0
        current_score = metric_func(reconstruct_at(state.z_best, frame))

        drift = abs(current_score - state.score_ref) / state.score_ref

        if drift <= state.drift_threshold:
            # OK — adjust adaptive N
            state.stable_streak += 1
            if state.stable_streak >= 3:
                state.check_interval = min(state.base_interval * 2, 40)
            state.fail_streak = 0
            return
        else:
            # Drift detected
            state.stable_streak = 0
            state.check_interval = max(state.base_interval // 2, 3)
            state.state = RESCANNING
            perform_local_scan(state.z_best, state.local_window_pct, z_range)
            return

    if state.state == RESCANNING:
        # (called from within local_scan completion)
        if found_new_focus:
            state.z_best = new_z
            state.score_ref = new_score
            state.frames_since_success = 0
            state.fail_streak = 0
            state.state = MONITORING
        else:
            mark_current_window_excluded()
            expand_window()

            if window_covers_full_range():
                # Check reset conditions (OR logic)
                if should_reset_exclusions():
                    state.excluded_zones.clear()
                    state.state = RESCANNING  # retry with clean list
                else:
                    state.state = FAILED
                    # use best score found so far
            else:
                perform_local_scan(...)  # wider window, skip excluded

function should_reset_exclusions():
    return (
        state.fail_streak >= state.fail_limit          # K consecutive failures
        or state.frames_since_success >= state.timeout_frames  # M frame timeout
        # Manual reset is handled via UI signal
    )
```

### 6.5 Threading

Adaptive autofocus runs in a dedicated `QThread`:

```python
class AdaptiveFocusWorker(QThread):
    focus_updated = Signal(float, float)  # z_best, score_ref
    status_changed = Signal(str)          # state name
    drift_detected = Signal()             # for UI indicator

    def __init__(self, metric_func, z_range, reconstruct_func):
        ...

    def on_new_frame(self, frame: np.ndarray):
        """Called from acquisition thread for each new frame."""
        # Runs the algorithm above
        # Emits signals for UI updates
```

### 6.6 Integration with recording

When video recording is active AND adaptive autofocus is enabled:
- Each captured frame is also passed to `AdaptiveFocusWorker.on_new_frame()`
- The reconstruction uses the tracked `z_best` automatically
- If recording is OFF but adaptive focus is ON, it works with the live preview frames

---

## 8. Video/Snapshot Recording

### 7.1 Video recording

```python
class RecordingWorker(QThread):
    frame_saved = Signal(int)        # frame number
    recording_finished = Signal(str) # output path

    def __init__(self, output_dir, format, channel, max_frames):
        ...
        self.frame_queue = Queue()  # thread-safe frame buffer

    def enqueue_frame(self, frame_data: dict):
        """Called from main thread. frame_data has keys: input, amplitude, phase, spectrum"""
        self.frame_queue.put(frame_data)

    def run(self):
        """Save frames from queue to disk."""
        while self.recording and self.frame_count < self.max_frames:
            frame = self.frame_queue.get()
            self._save_frame(frame)
            self.frame_count += 1
            self.frame_saved.emit(self.frame_count)
```

**Supported formats:**
- TIFF stack — one multi-page TIFF per channel
- AVI (raw) — uncompressed, for maximum compatibility
- MP4 — compressed, for sharing

**Save channels:**
- All — saves every channel in separate files
- Amplitude only
- Phase only
- Input only

### 7.2 Snapshot

Single-frame capture. Saves the current state of all (or selected) panels:
- PNG / TIFF / BMP formats
- Optional colorbar overlay
- Optional scale bar overlay
- File naming: `{channel}_{timestamp}.{ext}`

### 7.3 Toolbar integration

- **Record button** in toolbar: click to start (turns red, shows REC indicator), click again to stop
- **Snapshot button** in toolbar: single-click captures immediately with current Record tab settings
- Starting a recording auto-switches sidebar to Record tab
- Status bar shows `REC: OFF` / `REC: ●` with frame counter

---

## 9. Implementation Plan — Phase by Phase

### Phase 1: GUI modularization (no new features)
**Goal:** Split `main.py` into `src/gui/` modules without breaking anything.

1. Create all files in `src/gui/` with empty class stubs
2. Move toolbar code → `toolbar.py`
3. Move status bar code → `status_bar.py`
4. Move each image panel → `panels/*.py`
5. Move settings sections → `sidebar/*.py` (tab structure, including empty Camera tab stub)
6. Move window assembly → `main_window.py`
7. Update `src/main.py` to just instantiate `MainWindow`
8. **Test:** App should look and work exactly as before, but with tabbed sidebar

### Phase 2: Image grid system
**Goal:** Flexible layout with persistence.

1. Implement `image_grid.py` with `QSplitter`-based layouts
2. Add layout dropdown to image area
3. Implement double-click maximize
4. Add `QSettings` save/restore for layout + window geometry
5. Add `View → Reset Layout` menu action
6. **Test:** Switch layouts, close/reopen app, verify persistence and reset

### Phase 3: Phase panel merge
**Goal:** Combine wrapped/unwrapped into single panel.

1. Implement `phase_panel.py` with toggle buttons
2. Remove old separate wrapped/unwrapped panels
3. Wire toggle to reconstruction output
4. **Test:** Reconstruct, toggle between wrapped/unwrapped, verify data integrity

### Phase 4: Camera acquisition engine
**Goal:** NI-IMAQdx camera support with live feed.

1. Implement `src/core/camera.py` — NICamera class with graceful NI_AVAILABLE fallback
2. Implement `AcquisitionWorker` in `src/gui/workers/acquisition_worker.py`
3. Build Camera tab UI (connection, mode, FPS, exposure, gain, status)
4. Add File/Live mode switch to toolbar
5. Wire live pipeline: frame_ready → Input panel display
6. Add continuous reconstruction toggle (frame_ready → ReconstructionWorker → panels)
7. Implement FPS throttle and frame skip logic
8. Add actual FPS / dropped frames / frame counter to Camera tab status box
9. **Test without NI drivers:** Verify graceful degradation, File mode unaffected
10. **Test with NI camera:** Connect camera, verify live feed, test continuous/triggered modes, test exposure/gain controls

### Phase 5: Adaptive autofocus
**Goal:** Implement the three-layer adaptive focus algorithm.

1. Add `AdaptiveFocusState` dataclass to `src/core/autofocus.py`
2. Add `adaptive_focus_step()` function to `src/core/autofocus.py`
3. Implement `AdaptiveFocusWorker` in `src/gui/workers/`
4. Build Focus tab UI with enable toggle and status display
5. Wire signals: frame input → worker → UI status update + z value
6. Integration: in Live mode, acquisition frames feed adaptive focus automatically
7. **Test with file sequence:** Load folder of frames, enable adaptive focus, verify tracking
8. **Test with live camera:** Enable adaptive focus during live acquisition, verify Z₀ tracking

### Phase 6: Video/snapshot recording
**Goal:** Frame capture during acquisition.

1. Implement `RecordingWorker` in `src/gui/workers/`
2. Build Record tab UI (format, channel, output dir, max frames, FPS selection)
3. Add Record/Snapshot buttons to toolbar
4. Wire: toolbar buttons → worker → file output + status bar
5. Integration: recording captures ALL frames (even if reconstruction skips some)
6. Integration: if adaptive focus is on, recording frames feed both systems
7. **Test:** Record short sequence in each format (TIFF/AVI/MP4), verify file output
8. **Test full pipeline:** Live camera + adaptive focus + recording all running simultaneously

---

## 10. Coding Conventions for Implementation

### Signal/slot naming
```python
# Signals: noun_verb (past tense)
reconstruction_completed = Signal(dict)
focus_updated = Signal(float, float)
frame_captured = Signal(int)

# Slots: on_noun_verb
def on_reconstruction_completed(self, result: dict): ...
def on_focus_updated(self, z: float, score: float): ...
```

### Settings keys
```python
# Hierarchical, lowercase, slash-separated
"app/mode"                 # "file" or "live"
"grid/layout"              # "2x2", "1+2", "single", "3+1", "1x4"
"grid/maximized_panel"     # None or panel name
"window/geometry"          # QByteArray
"window/state"             # QByteArray
"sidebar/active_tab"       # int (0-5)
"camera/last_device"       # str camera name (e.g., "cam0")
"camera/mode"              # "continuous" or "triggered"
"camera/target_fps"        # int
"camera/limit_fps"         # bool
"camera/exposure_us"       # float
"camera/gain_db"           # float
"camera/bit_depth"         # "mono8" or "mono16"
"camera/auto_reconstruct"  # bool
"camera/recon_every_n"     # int (reconstruct every N frames)
"recon/method"             # "ASM", "Fresnel"
"recon/wavelength"         # float
"focus/metric"             # "total_variation", "brenner", etc.
"focus/adaptive_enabled"   # bool
"record/format"            # "tiff", "avi", "mp4"
"record/output_dir"        # str path
"record/channel"           # "all", "amplitude", "phase", "input"
```

### Threading rules
- **Never** access PyQtGraph `ImageView` from worker threads
- Workers emit signals → main thread slots update UI
- Use `Queue` for frame passing to recording worker
- **Acquisition worker:** highest priority, never blocks, never drops frames internally
- **Reconstruction worker:** one at a time, drops incoming frames if previous recon isn't finished (latest-wins)
- **Adaptive focus worker:** persistent thread, paused/resumed via state, receives frames via signal
- **Recording worker:** receives ALL frames via Queue, writes to disk in background
- Main thread only does signal routing and UI updates — no heavy computation

### Threading architecture diagram
```
┌─────────────────┐
│ AcquisitionWorker│ ← grabs frames at camera speed
│   (QThread)      │
└────────┬────────┘
         │ frame_ready(ndarray, int)
         ▼
┌─────────────────┐     ┌──────────────────┐
│   Main Thread    │────→│ReconstructionWorker│ ← may skip frames
│  (signal router) │     │   (QThread)        │
│                  │     └──────────────────┘
│                  │     ┌──────────────────┐
│                  │────→│AdaptiveFocusWorker │ ← every Nth frame
│                  │     │   (QThread)        │
│                  │     └──────────────────┘
│                  │     ┌──────────────────┐
│                  │────→│ RecordingWorker    │ ← ALL frames via Queue
│                  │     │   (QThread)        │
└─────────────────┘     └──────────────────┘
```

### Performance considerations
- **Acquisition thread** runs at camera speed — never throttled by reconstruction
- Adaptive focus check: only computes metric at current Z₀, not full Z-scan (fast)
- Local re-scan: uses `np.linspace` over narrow range, still fast for DHM (numerical propagation)
- Recording: use `Queue(maxsize=100)` to prevent memory overflow; if queue full, drop oldest frame and increment dropped counter
- Image panel updates: debounce to max 30fps display refresh regardless of camera speed
- Reconstruction frame skipping: configurable via "reconstruct every N frames" in Camera tab
- **Memory:** at 1200×1600 uint16, one frame ≈ 3.8MB. A Queue of 100 frames ≈ 380MB. Monitor and warn if approaching limits.

---

## 11. Notes

- The app uses a dark theme (as seen in screenshot). Keep the dark theme consistent.
- PyQtGraph's `ImageView` handles colormap, histogram, and ROI natively — leverage these.
- `julia_bridge.py` exists for future Julia integration. The adaptive autofocus should be designed so its core algorithm can eventually be offloaded to Julia via this bridge.
- The `cline_docs/` directory contains project documentation that may have additional context about architecture decisions.
- All numeric inputs should support both comma and dot as decimal separator (Turkish locale uses comma).
- **NI-IMAQdx dependency is optional.** The app must launch and function fully in File mode even if NI Vision Acquisition Software is not installed. Use try/except import with a module-level `NI_AVAILABLE` flag.
- **Camera resource cleanup is critical.** Always release the camera session on app close, even on crash. Use `atexit` and `QApplication.aboutToQuit` signal as safety nets.
- **Triggered mode:** In triggered acquisition, the camera waits for a trigger signal before capturing each frame. The trigger source (software, external hardware line) should be selectable. Software trigger means the app sends `cam.send_software_trigger()` at the desired interval.
- **Video recording FPS vs camera FPS:** Recording FPS should default to camera FPS but can be set independently. If recording FPS < camera FPS, only every Nth frame is saved. If recording FPS > camera FPS, it clips to camera FPS.
- **Batch mode in File mode:** The existing Batch/Render feature continues to work only in File mode. In Live mode, the batch button is disabled (live processing replaces batch).
