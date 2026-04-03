"""
NI-IMAQdx camera abstraction.
Provides a clean interface independent of the specific NI SDK version.
Falls back gracefully if NI drivers are not installed.
"""
import numpy as np

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
    """

    def __init__(self, camera_name: str):
        if not NI_AVAILABLE:
            raise CameraError("NI-IMAQdx drivers not installed")
        self.camera_name = camera_name
        self.session = None
        self.is_open = False
        self.is_acquiring = False
        self._resolution = (0, 0)
        self._bit_depth = "mono16"

    @staticmethod
    def enumerate_cameras() -> list[dict]:
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
        if not NI_AVAILABLE:
            raise CameraError("NI drivers missing")
        try:
            self.session = niimaqdx.IMAQdxOpenCamera(self.camera_name, niimaqdx.IMAQdxCameraControlModeController)
            self.is_open = True
            
            # Read resolution to cache it
            width = niimaqdx.IMAQdxGetAttribute(self.session, b"SensorWidth", niimaqdx.IMAQdxAttributeTypeU32)
            height = niimaqdx.IMAQdxGetAttribute(self.session, b"SensorHeight", niimaqdx.IMAQdxAttributeTypeU32)
            if width and height:
                self._resolution = (height, width)
        except Exception as e:
            raise CameraError(f"Failed to open camera: {e}")

    def configure(self, exposure_us: float = 5000, gain_db: float = 0.0, bit_depth: str = "mono16"):
        if not self.is_open:
            raise CameraError("Camera is not open")
        try:
            # Note: actual NI-IMAQdx attribute names may vary by camera model (e.g. "ExposureTimeAbs")
            # We attempt standard generic names
            niimaqdx.IMAQdxSetAttribute(self.session, b"ExposureTimeRaw", niimaqdx.IMAQdxAttributeTypeU32, int(exposure_us))
            niimaqdx.IMAQdxSetAttribute(self.session, b"GainRaw", niimaqdx.IMAQdxAttributeTypeU32, int(gain_db))
            
            self._bit_depth = bit_depth
            if bit_depth.lower() == "mono16":
                niimaqdx.IMAQdxSetAttribute(self.session, b"PixelFormat", niimaqdx.IMAQdxAttributeTypeEnumItem, niimaqdx.IMAQdxPixelFormatMono16)
            else:
                niimaqdx.IMAQdxSetAttribute(self.session, b"PixelFormat", niimaqdx.IMAQdxAttributeTypeEnumItem, niimaqdx.IMAQdxPixelFormatMono8)
        except Exception as e:
            pass # ignore if unsupported by specific camera model

    def start_acquisition(self, mode: str = "continuous"):
        if not self.is_open:
            raise CameraError("Camera not open")
        try:
            if mode == "continuous":
                niimaqdx.IMAQdxConfigureGrab(self.session)
            niimaqdx.IMAQdxStartAcquisition(self.session)
            self.is_acquiring = True
            # pre-allocate image object
            self._img = niimaqdx.IMAQ_Create()
        except Exception as e:
            raise CameraError(f"Failed to start acquisition: {e}")

    def grab_frame(self) -> np.ndarray:
        if not self.is_acquiring:
            raise CameraError("Not acquiring")
        try:
            niimaqdx.IMAQdxGrab(self.session, self._img, waitForNextBuffer=True)
            # IMAQ_ImageToArray converts the IMAQ image to a numpy array
            # The python bindings might differ slightly. Using a hypothetical standard bindings method.
            # Usually niimaq returns a tuple (img_array, ...)
            arr = niimaqdx.IMAQ_ImageToArray(self._img)
            # Depending on bindings, arr might be the array itself
            if isinstance(arr, tuple):
                arr = arr[0]
            if arr is None:
                raise CameraError("Empty frame")
            return np.array(arr, copy=True)
        except Exception as e:
            raise CameraError(f"Failed to grab frame: {e}")

    def stop_acquisition(self):
        if self.is_acquiring:
            try:
                niimaqdx.IMAQdxStopAcquisition(self.session)
            except Exception:
                pass
            self.is_acquiring = False

    def close(self):
        self.stop_acquisition()
        if self.is_open and self.session:
            try:
                niimaqdx.IMAQdxCloseCamera(self.session)
            except Exception:
                pass
        self.is_open = False
        self.session = None

    def get_attribute(self, attr_name: str):
        # Generic wrapper
        pass

    def set_attribute(self, attr_name: str, value):
        pass

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def dtype(self) -> str:
        return "uint16" if "16" in self._bit_depth else "uint8"
