from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


class FFTBackendName(str, Enum):
    PYFFTW = "pyfftw"
    SCIPY = "scipy"
    MLX = "mlx"
    NUMPY = "numpy"


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


def get_best_fft_backend(prefer: Optional[FFTBackendName] = None) -> FFTBackend:
    if prefer == FFTBackendName.NUMPY:
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

    # Default fallback sequence: PyFFTW -> MLX -> Scipy -> NumPy
    try:
        return PyFFTWBackend()
    except Exception:
        pass

    try:
        return MLXFFTBackend()
    except Exception:
        pass

    try:
        return ScipyFFTBackend()
    except Exception:
        pass

    return NumpyFFTBackend()
