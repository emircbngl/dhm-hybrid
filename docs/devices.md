# Lab device control

Companion to `docs/cameras.md`. The `core.devices` package
provides a registry + Protocols for the **non-camera** instruments
that round out a microscopy session: motorised stages, shutters,
LED controllers, and generic serial devices.

## Why a separate module from cameras

Cameras have one Protocol (`grab` / `start` / `stop`); other
devices diverge sharply on what makes sense:

| kind | actions |
|------|---------|
| stage | `move_to(x_um, y_um, z_um)`, `home()`, `position_um` |
| shutter | `open()`, `close()`, `is_open` |
| LED | `on()`, `off()`, `set_intensity(percent)`, `is_on` |
| generic | `send_command(str, expect_reply=True)` |

A single Protocol with everything optional pushes `hasattr` calls
into every consumer. Per-kind Protocols (StageDevice / ShutterDevice
/ LEDDevice) keep type checks honest at the right granularity.

## Discovery

```python
from core.devices import all_backends, available_backends, make_device

# Every registered backend, including those whose SDK isn't installed
print([(b.name, b.kind.value) for b in all_backends()])

# Only the ones whose SDK probe succeeded — drives the dialog dropdown
print([b.name for b in available_backends()])

stage = make_device("mock_stage")
stage.connect()
stage.move_to(100, 200, 50)
```

`make_device(name)` raises:
- `ValueError` when the name isn't registered.
- `RuntimeError` when the backend exists but its SDK isn't
  installed (e.g. `serial_generic` without pyserial).

## Adding a new backend

Drop `src/core/devices/<vendor>.py` with:

```python
from . import DeviceBackendInfo, DeviceKind

BACKEND = DeviceBackendInfo(
    name="thorlabs_kdc101",
    display_name="Thorlabs KDC101",
    kind=DeviceKind.STAGE,
    summary="USB-CDC controller for KCube DC servo motors.",
    requires_sdk=("thorlabs_kinesis",),
    capabilities={"xy": False, "z": True, "home": True},
)

def is_available() -> bool:
    try:
        import thorlabs_kinesis  # noqa: F401
        return True
    except Exception:
        return False

class _ThorlabsStage:
    # ... implement StageDevice protocol ...
    pass

def make(**kwargs):
    return _ThorlabsStage(**kwargs)
```

Discovery picks it up automatically — no central list to update.

## Multi-device acquisition

The lab pattern that motivated this module: **one acquire call
that coordinates camera + stage + shutter + LED**. Use
`core.devices.orchestrator`:

```python
from core.devices import make_device
from core.devices.orchestrator import (
    AcquisitionPlan, StagePosition, run_plan,
)

cam = ...  # any CameraSource
stage = make_device("mock_stage")
shutter = make_device("mock_shutter")
led = make_device("mock_led")

plan = AcquisitionPlan(
    positions=[
        StagePosition(0, 0, label="A"),
        StagePosition(120, 0, label="B"),
        StagePosition(240, 0, label="C"),
    ],
    led_intensity_percent=50.0,
    shutter_per_frame=True,
    settle_time_s=0.1,
)
result = run_plan(plan, camera=cam, stage=stage,
                  shutter=shutter, led=led)
for frame in result.frames:
    print(frame.position.label, frame.frame.shape if frame.frame else frame.error)
```

The orchestrator returns a `PlanResult` with one `FrameRecord` per
position (or per cancellation point). All exceptions are caught and
recorded per-frame; the run never raises mid-plan, so a missing
sample at one position doesn't kill the rest.

## CI / dev laptop story

Mocks are first-class:

- `mock_stage` — XYZ positioner with soft limits + optional
  settle delay
- `mock_shutter` — binary state with optional latency
- `mock_led` — 0–100 % intensity (clamped, no raise on overflow)

Vendor backends sit alongside as integration scaffolds — same
shape as the camera vendor stubs. SDK detection via import probe;
without the SDK, registry filters them out and `make_device`
raises a clear RuntimeError.
