"""Track C inference: classical poly5 reffree → CNN residual correction.

This is the production-facing module. Call ``ReffreeReconstructor.from_checkpoint``
to load a trained model, then ``reconstruct(hologram_path)`` to get the
final unwrapped phase. No reference hologram needed at runtime.

Pipeline at runtime::

    raw hologram (uint16 PNG / TIFF)
       ↓ demodulate (Fourier sideband)
       ↓ autofocus (or fixed z if known)
       ↓ propagate at z (ASM)
       ↓ unwrap + polynomial bg fit (order 5)
       → ϕ_classical (with structured residual)
       ↓ tile to 768x768 patches with 64-pixel overlap
       ↓ run UNetLite per patch
       ↓ blend tiles with cosine window
       → ϕ_clean
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# The classical demod/autofocus/propagate/unwrap chain lives in core so
# shipped code never imports from scripts/ (2026-07-05 review fixed that
# layering inversion). Optical parameters ride in an OpticalConfig instead
# of module-level globals baked for one microscope.
from core.ingestion import load_any
from core.pipelines.reffree_hybrid import (
    LAB_OPTICS,
    OpticalConfig,
    demodulate,
    reconstruct_in_memory,
)

from recon_dl.unet_lite import UNetLite

_LOG = logging.getLogger(__name__)


@dataclass
class ReffreeResult:
    phase_clean: np.ndarray
    phase_classical: np.ndarray
    predicted_residual: np.ndarray
    z_m: float
    inference_ms: float


def _cosine_window(h: int, w: int, edge: int = 64) -> np.ndarray:
    """A 2D cosine taper that smoothly goes 1 → 0 over the last `edge`
    pixels on each side. Used to blend overlapping tiles."""
    win = np.ones((h, w), dtype=np.float32)
    if edge <= 0:
        return win
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge, dtype=np.float32)))
    win[:edge, :] *= ramp[:, None]
    win[-edge:, :] *= ramp[::-1, None]
    win[:, :edge] *= ramp[None, :]
    win[:, -edge:] *= ramp[None, ::-1]
    return win


class ReffreeReconstructor:
    """End-to-end reference-free DHM reconstructor.

    The classical pipeline (demod → autofocus → propagate → unwrap →
    poly bg) is reused from the existing codebase; the CNN residual
    corrector is loaded from a checkpoint.
    """

    def __init__(self,
                 model: UNetLite,
                 device: torch.device,
                 *,
                 tile_size: int = 768,
                 tile_overlap: int = 64,
                 bg_method: str = "polynomial",
                 bg_polynomial_order: int = 5,
                 bg_n_terms: int = 15,
                 optics: OpticalConfig = LAB_OPTICS) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.bg_method = bg_method
        self.bg_polynomial_order = bg_polynomial_order
        self.bg_n_terms = bg_n_terms
        self.optics = optics

    @classmethod
    def from_checkpoint(cls, ckpt_path: "str | Path",
                        device: Optional[torch.device] = None,
                        **kwargs) -> "ReffreeReconstructor":
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})
        base_c = args.get("base_channels", 32)
        model = UNetLite(base_c=base_c)
        model.load_state_dict(ckpt["model_state_dict"])
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        _LOG.info("loaded UNetLite (base_c=%d) from %s, device=%s",
                  base_c, ckpt_path, device)
        return cls(model, device, **kwargs)

    def _classical_phase(self, hologram_path: "str | Path",
                         fixed_z_m: Optional[float] = None
                         ) -> tuple[np.ndarray, float]:
        raw = np.asarray(load_any(Path(hologram_path)).array)
        holo_fc = demodulate(raw, self.optics)
        # reconstruct_in_memory runs autofocus itself when fixed_z_m is None.
        unwrapped, z_used, _ = reconstruct_in_memory(
            holo_fc, None, self.optics,
            bg_method=self.bg_method,
            bg_n_terms=self.bg_n_terms,
            bg_polynomial_order=self.bg_polynomial_order,
            fixed_z_m=fixed_z_m,
        )
        return unwrapped, z_used

    def _cnn_correct(self, phi_classical: np.ndarray) -> np.ndarray:
        """Tile the input, run the model, blend back to a single image.

        Per tile: predict residual, subtract from input. Blend overlapping
        tiles with a cosine window so seams disappear.
        """
        h, w = phi_classical.shape
        ts = self.tile_size
        ov = self.tile_overlap
        step = ts - ov
        if h < ts or w < ts:
            # Pad to at least one tile.
            pad_h = max(0, ts - h)
            pad_w = max(0, ts - w)
            phi_classical = np.pad(phi_classical,
                                   ((0, pad_h), (0, pad_w)), mode="edge")
            h, w = phi_classical.shape
        # Tile coordinates that cover the entire image.
        ys = list(range(0, max(h - ts, 0) + 1, step))
        if ys[-1] + ts < h:
            ys.append(h - ts)
        xs = list(range(0, max(w - ts, 0) + 1, step))
        if xs[-1] + ts < w:
            xs.append(w - ts)

        accum = np.zeros_like(phi_classical, dtype=np.float32)
        weight = np.zeros_like(phi_classical, dtype=np.float32)
        win = _cosine_window(ts, ts, edge=min(ov, ts // 4))

        with torch.no_grad():
            for y in ys:
                for x in xs:
                    tile = phi_classical[y:y + ts, x:x + ts]
                    t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0)
                    t = t.to(self.device, dtype=torch.float32)
                    corrected, _ = self.model(t)
                    out = corrected.squeeze().detach().cpu().numpy()
                    accum[y:y + ts, x:x + ts] += out * win
                    weight[y:y + ts, x:x + ts] += win
        accum /= np.where(weight > 0, weight, 1.0)
        return accum

    def reconstruct(self, hologram_path: "str | Path",
                    *, fixed_z_m: Optional[float] = None) -> ReffreeResult:
        import time
        phi_classical, z_used = self._classical_phase(hologram_path,
                                                       fixed_z_m=fixed_z_m)
        t0 = time.monotonic()
        phi_clean = self._cnn_correct(phi_classical)
        # Ensure shape matches if we padded earlier.
        h, w = phi_classical.shape
        ho, wo = phi_clean.shape
        phi_clean = phi_clean[:h, :w]
        residual = phi_classical - phi_clean
        inf_ms = (time.monotonic() - t0) * 1000.0
        _LOG.info("CNN inference %.1f ms (%dx%d)", inf_ms, h, w)
        return ReffreeResult(
            phase_clean=phi_clean,
            phase_classical=phi_classical,
            predicted_residual=residual,
            z_m=z_used,
            inference_ms=inf_ms,
        )
