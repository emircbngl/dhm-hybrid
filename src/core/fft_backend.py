from __future__ import annotations

from functools import lru_cache
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


class FFTBackendName(str, Enum):
    PYFFTW = "pyfftw"
    SCIPY = "scipy"
    MLX = "mlx"
    NUMPY = "numpy"
    TORCH = "torch"  # v2.1.0 — CUDA / Metal / CPU via torch.fft


@dataclass(frozen=True)
class FFTBackend:
    name: FFTBackendName

    def fft2(self, x: Any) -> Any:
        raise NotImplementedError

    def ifft2(self, x: Any) -> Any:
        raise NotImplementedError

    def fftshift2(self, x: Any) -> Any:
        raise NotImplementedError

    def ifftshift2(self, x: Any) -> Any:
        raise NotImplementedError

    def to_numpy(self, x: Any) -> np.ndarray:
        raise NotImplementedError

    def from_numpy(self, x: np.ndarray) -> Any:
        raise NotImplementedError

    # v2.1.0 — batched ops over a (B, ny, nx) stack of 2-D arrays.
    # Default impl loops fft2/ifft2 over the batch axis. GPU
    # backends (torch) override with a single batched kernel call,
    # which is where the multi-z autofocus win comes from.
    def fft2_batched(self, x: Any) -> Any:
        """fft2 over the last two axes of an (..., ny, nx) input."""
        arr = np.asarray(x)
        if arr.ndim < 3:
            return self.fft2(arr)
        out = np.empty_like(arr, dtype=np.complex64)
        for i in range(arr.shape[0]):
            out[i] = self.fft2(arr[i]).astype(np.complex64, copy=False)
        return out

    def ifft2_batched(self, x: Any) -> Any:
        """ifft2 over the last two axes of an (..., ny, nx) input."""
        arr = np.asarray(x)
        if arr.ndim < 3:
            return self.ifft2(arr)
        out = np.empty_like(arr, dtype=np.complex64)
        for i in range(arr.shape[0]):
            out[i] = self.ifft2(arr[i]).astype(np.complex64, copy=False)
        return out

    @property
    def supports_batched(self) -> bool:
        """``True`` when the backend overrides the batched ops with a
        true single-call implementation (vs. the default loop)."""
        return False


class NumpyFFTBackend(FFTBackend):
    def __init__(self) -> None:
        super().__init__(name=FFTBackendName.NUMPY)

    def fft2(self, x: np.ndarray) -> np.ndarray:
        return np.fft.fft2(x)

    def ifft2(self, x: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(x)

    def fftshift2(self, x: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(x)

    def ifftshift2(self, x: np.ndarray) -> np.ndarray:
        return np.fft.ifftshift(x)

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return x

    def from_numpy(self, x: np.ndarray) -> np.ndarray:
        return x


class ScipyFFTBackend(FFTBackend):
    def __init__(self) -> None:
        super().__init__(name=FFTBackendName.SCIPY)
        import scipy.fft
        scipy.fft.set_workers(-1)
        self._fft = scipy.fft

    def fft2(self, x: np.ndarray) -> np.ndarray:
        return self._fft.fft2(x)

    def ifft2(self, x: np.ndarray) -> np.ndarray:
        return self._fft.ifft2(x)

    def fftshift2(self, x: np.ndarray) -> np.ndarray:
        return self._fft.fftshift(x)

    def ifftshift2(self, x: np.ndarray) -> np.ndarray:
        return self._fft.ifftshift(x)

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return x

    def from_numpy(self, x: np.ndarray) -> np.ndarray:
        return x


class PyFFTWBackend(FFTBackend):
    def __init__(self) -> None:
        super().__init__(name=FFTBackendName.PYFFTW)
        import pyfftw
        pyfftw.interfaces.cache.enable()
        self._fft = pyfftw.interfaces.scipy_fft

    def fft2(self, x: np.ndarray) -> np.ndarray:
        return self._fft.fft2(x, workers=-1)

    def ifft2(self, x: np.ndarray) -> np.ndarray:
        return self._fft.ifft2(x, workers=-1)

    def fftshift2(self, x: np.ndarray) -> np.ndarray:
        import scipy.fft
        return scipy.fft.fftshift(x)

    def ifftshift2(self, x: np.ndarray) -> np.ndarray:
        import scipy.fft
        return scipy.fft.ifftshift(x)

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return x

    def from_numpy(self, x: np.ndarray) -> np.ndarray:
        return x


class MLXFFTBackend(FFTBackend):
    def __init__(self) -> None:
        super().__init__(name=FFTBackendName.MLX)
        # ``mlx.core`` can abort the whole interpreter while initialising
        # Metal (rather than raising an ImportError) on a machine with an
        # unusable Metal device.  Check it in a child process first so an
        # unavailable MLX runtime is a recoverable backend-selection error.
        if not _mlx_runtime_available():
            raise RuntimeError("MLX/Metal runtime is unavailable")
        import mlx.core as mx

        self._mx = mx
        self._fft = mx.fft

    def _ensure_mx_array(self, x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return self._mx.array(x)
        return x

    def fft2(self, x: Any) -> Any:
        result = self._fft.fft2(self._ensure_mx_array(x))
        return np.array(result)

    def ifft2(self, x: Any) -> Any:
        result = self._fft.ifft2(self._ensure_mx_array(x))
        return np.array(result)

    def fftshift2(self, x: Any) -> Any:
        return self._fft.fftshift(self._ensure_mx_array(x))

    def ifftshift2(self, x: Any) -> Any:
        return self._fft.ifftshift(self._ensure_mx_array(x))

    def to_numpy(self, x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    def from_numpy(self, x: np.ndarray) -> Any:
        return self._mx.array(x)


@lru_cache(maxsize=1)
def _mlx_runtime_available() -> bool:
    """Return whether this Python environment can initialise MLX safely.

    Importing ``mlx.core`` itself may terminate Python with SIGABRT, which a
    normal ``try``/``except`` cannot contain.  The explicit MLX backend is
    therefore probed once in a child process; an abort or timeout simply makes
    the backend unavailable to the parent process.
    """
    probe = "import mlx.core as mx; mx.random.key(0)"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


class TorchFFTBackend(FFTBackend):
    """v2.1.0 — torch.fft backend with optional CUDA / Metal MPS.

    Selects the best device available at construction time:

    1. ``cuda`` if ``torch.cuda.is_available()`` (NVIDIA box).
    2. ``mps`` if ``torch.backends.mps.is_available()`` (Apple
       Silicon).
    3. ``cpu`` otherwise.

    The device is captured once and never changes per-instance —
    callers wanting a different device construct a new backend.

    All public ops (``fft2``, ``ifft2``, ``fft2_batched``,
    ``ifft2_batched``) accept either ``np.ndarray`` or
    ``torch.Tensor`` and return ``np.ndarray`` so existing call
    sites don't have to know about torch. Heavy users can stay
    in tensor-land via :meth:`fft2_tensor` / :meth:`ifft2_tensor`
    to amortise host↔device transfers across many calls.
    """

    def __init__(self, *, device: Optional[str] = None) -> None:
        super().__init__(name=FFTBackendName.TORCH)
        import torch  # noqa: WPS433 — lazy / optional dep
        self._torch = torch
        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif (hasattr(torch.backends, "mps")
              and torch.backends.mps.is_available()):
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def supports_batched(self) -> bool:
        return True

    # ── tensor ↔ numpy conversion ─────────────────────────────
    def _to_tensor(self, x: Any):
        torch = self._torch
        if isinstance(x, torch.Tensor):
            return x.to(self._device, dtype=torch.complex64)
        return torch.from_numpy(
            np.asarray(x).astype(np.complex64, copy=False),
        ).to(self._device)

    def from_numpy(self, x: np.ndarray) -> Any:
        return self._to_tensor(x)

    def to_numpy(self, x: Any) -> np.ndarray:
        torch = self._torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.complex64,
                                                   copy=False)
        return np.asarray(x).astype(np.complex64, copy=False)

    # ── single-shot ops ────────────────────────────────────────
    def fft2(self, x: Any) -> np.ndarray:
        t = self._to_tensor(x)
        return self.to_numpy(self._torch.fft.fft2(t))

    def ifft2(self, x: Any) -> np.ndarray:
        t = self._to_tensor(x)
        return self.to_numpy(self._torch.fft.ifft2(t))

    def fftshift2(self, x: Any) -> np.ndarray:
        t = self._to_tensor(x)
        return self.to_numpy(self._torch.fft.fftshift(t, dim=(-2, -1)))

    def ifftshift2(self, x: Any) -> np.ndarray:
        t = self._to_tensor(x)
        return self.to_numpy(self._torch.fft.ifftshift(t, dim=(-2, -1)))

    # ── batched ops — the v2.1.0 win ──────────────────────────
    def fft2_batched(self, x: Any) -> np.ndarray:
        t = self._to_tensor(x)
        return self.to_numpy(self._torch.fft.fft2(t, dim=(-2, -1)))

    def ifft2_batched(self, x: Any) -> np.ndarray:
        t = self._to_tensor(x)
        return self.to_numpy(self._torch.fft.ifft2(t, dim=(-2, -1)))

    # ── tensor-native fast paths (skip host↔device when caller
    # is ready to stay in torch land) ─────────────────────────
    def fft2_tensor(self, t: Any) -> Any:
        return self._torch.fft.fft2(t.to(self._device), dim=(-2, -1))

    def ifft2_tensor(self, t: Any) -> Any:
        return self._torch.fft.ifft2(t.to(self._device), dim=(-2, -1))


def get_best_fft_backend(prefer: Optional[FFTBackendName] = None) -> FFTBackend:
    if prefer == FFTBackendName.NUMPY:
        return NumpyFFTBackend()

    if prefer == FFTBackendName.TORCH:
        try:
            return TorchFFTBackend()
        except Exception:
            return NumpyFFTBackend()

    if prefer == FFTBackendName.MLX:
        try:
            return MLXFFTBackend()
        except Exception:
            return NumpyFFTBackend()

    if prefer == FFTBackendName.SCIPY:
        try:
            return ScipyFFTBackend()
        except Exception:
            return NumpyFFTBackend()

    if prefer == FFTBackendName.PYFFTW:
        try:
            return PyFFTWBackend()
        except Exception:
            try:
                return ScipyFFTBackend()
            except Exception:
                return NumpyFFTBackend()

    # Default fallback: PyFFTW -> Scipy -> NumPy.  MLX and Torch are explicit
    # opt-ins: both initialise native GPU runtimes, and MLX can terminate the
    # interpreter while probing an unusable Metal device.
    try:
        return PyFFTWBackend()
    except Exception:
        pass

    try:
        return ScipyFFTBackend()
    except Exception:
        pass

    return NumpyFFTBackend()
