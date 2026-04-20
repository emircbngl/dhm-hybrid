"""
Shared utilities for all test tools.
"""
import sys
import time
import numpy as np
from pathlib import Path

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Default test samples
SAMPLE_DIR = ROOT / "labtest"
DEFAULT_SAMPLE = SAMPLE_DIR / "1-100_25um_stdSlide_withoutcovSlip_2.png"
DEFAULT_REF = None  # No reference for labtest samples

SAMPLE_DIR_3UM = ROOT / "Test Samples"
SAMPLE_3UM = SAMPLE_DIR_3UM / "1-1000_3um_imgplane.png"
REF_3UM = SAMPLE_DIR_3UM / "Ref_1-1000_3um_imgplane.png"

# Default optical parameters (50x oil-immersion)
DEFAULT_WAVELENGTH_M = 632.8e-9
DEFAULT_PIXEL_SIZE_M = 0.088e-6  # 4.4µm / 50x
DEFAULT_MAGNIFICATION = 50.0
DEFAULT_N_MEDIUM = 1.0
DEFAULT_MASK_RADIUS = 80
DEFAULT_Z_MM = 0.0


def find_sample(path_str=None):
    """Find a sample image. Falls back to default."""
    if path_str:
        p = Path(path_str)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
    if DEFAULT_SAMPLE.exists():
        return DEFAULT_SAMPLE
    # Try Test Samples
    if SAMPLE_3UM.exists():
        return SAMPLE_3UM
    raise FileNotFoundError(
        f"No sample found. Place images in {SAMPLE_DIR} or {SAMPLE_DIR_3UM}"
    )


def load_sample(path=None):
    """Load a sample image and return (array, path)."""
    from core.ingestion import load_any
    p = find_sample(path)
    loaded = load_any(p)
    arr = np.asarray(loaded.array)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32), p


def extract_field(img, mask_radius=DEFAULT_MASK_RADIUS):
    """Run off-axis extraction on an image, return (complex_field, center, spectrum, mask)."""
    from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug

    norm = img.copy()
    if np.max(np.abs(norm)) > 0:
        norm = norm / float(np.max(np.abs(norm)))

    params = OffAxisParams(radius=mask_radius)
    return extract_complex_field_offaxis_debug(norm, params)


class Timer:
    """Simple context-manager timer."""
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"  [{self.label}] {self.elapsed*1000:.1f} ms")


def section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg):
    print(f"  ✓ {msg}")


def fail(msg):
    print(f"  ✗ {msg}")


def info(msg):
    print(f"  ℹ {msg}")
