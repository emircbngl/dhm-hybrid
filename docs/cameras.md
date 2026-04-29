# Camera backend integration guide

This document is for whoever wires the actual SDK calls into the
vendor stubs (`src/core/cameras/{pylon,ids,thorlabs}.py`). The
shape is in place; only the SDK-specific blocks are TODO.

## Backend lifecycle

Every backend implements three operations from the
`CameraSource` Protocol:

```python
def start(self) -> None: ...
def stop(self)  -> None: ...
def grab(self)  -> np.ndarray: ...   # 2-D float32 in [0, 1]
```

Plus two read-only properties: `size: tuple[int, int]` and
`fps: float`.

The acquisition thread (`ui2.camera_feed.AcquisitionThread`)
calls `start()` once, polls `grab()` in a tight loop until
cancellation, then `stop()`. Mid-loop exceptions are logged and
the thread continues — a buggy frame doesn't kill the feed.

## Adding / completing a vendor backend

### 1. SDK detection

`is_available()` must be a side-effect-free import probe:

```python
def is_available() -> bool:
    try:
        import pypylon.pylon  # noqa: F401
        return True
    except Exception:
        return False
```

The registry calls this for every backend on every dropdown
populate. Don't open devices here.

### 2. `start()` block

The TODO comments inside each stub show the pattern. Key points:

- **Open device**, configure exposure / pixel format / ROI,
  start grabbing.
- Cache `self._size` from the actual sensor — your installed
  camera might not be 1024×1024.
- Keep the SDK handle on `self._handle` so `stop()` + `grab()`
  can reach it.

### 3. `grab()` block

Convert the SDK's frame buffer to **float32 in [0, 1]**.

```python
arr = sdk_frame.image_buffer
if arr.dtype == np.uint16:
    out = arr.astype(np.float32) / 65535.0
elif arr.dtype == np.uint8:
    out = arr.astype(np.float32) / 255.0
else:
    out = np.clip(arr.astype(np.float32), 0.0, 1.0)
return out
```

Don't return uint16 / uint8 directly — the rest of the pipeline
assumes `[0, 1]` floats and would over-bright every preview.

### 4. `stop()` block

Always release SDK resources, even if `start()` failed. The
acquisition thread doesn't try/except around `stop()` — the
backend is responsible.

## Test surface

Vendor tests (real hardware) belong in a `tests/hw/` folder
that's excluded from the default pytest run via marker
(`@pytest.mark.real_hardware`). CI never runs them; lab box
runs them as a pre-deployment smoke.

Until then:

- **Mock camera** (`core.cameras.mock`) covers Protocol surface
  for headless CI.
- **Synthetic camera** (`core.cameras.synthetic`) covers DPG
  preview path.
- **Vendor stubs** raise `NotImplementedError` when called
  without their SDK + integration block — explicit failure beats
  silent fallback.

## Recording integration

Both `TiffStackRecorder` (lossless, science) and `MP4Recorder`
(presentation) follow the same shape:

```python
rec.start()
while running:
    frame = camera.grab()
    rec.write_frame(frame)
rec.stop()
```

Acquisition thread takes either as the `recorder=` kwarg. New
formats (PNG sequence, .npz, etc.) drop into the same slot.
