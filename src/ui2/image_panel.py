"""Zoomable image panel for the ui2 app (v2.0.1).

Wraps a ``mvPlot`` + ``mvImageSeries`` so the user can mouse-wheel-zoom
and middle-drag-pan the texture exactly like the plots in the rest of
the Dear PyGui ecosystem. The default ``add_image`` widget is a
static blit — good enough for the MVP, but the lab's first feedback
was "I can't zoom."

The panel also renders a *drop-zone affordance* when no image has
been pushed yet: a muted outlined rectangle with "Drop a hologram
here" text. Once :meth:`set_image` is called the affordance fades
(dpg-side ``configure_item(show=False)``).
"""
from __future__ import annotations

import uuid
from typing import Optional, Tuple

import dearpygui.dearpygui as dpg
import numpy as np

from .theme import ScientificTheme


class ZoomableImagePanel:
    """One titled image preview with pan/zoom via ``mvPlot``."""

    def __init__(
        self,
        title: str,
        *,
        size: int = 512,
        colormap: str = "gray",
    ) -> None:
        self.title = title
        self.size = int(size)
        self.colormap = colormap
        self._suffix = uuid.uuid4().hex[:6]
        self.tex_tag = f"tex_{self._suffix}"
        self.plot_tag = f"plot_{self._suffix}"
        self.x_axis_tag = f"xaxis_{self._suffix}"
        self.y_axis_tag = f"yaxis_{self._suffix}"
        self.series_tag = f"series_{self._suffix}"
        self.dropzone_tag = f"drop_{self._suffix}"
        # v2.0.5 — tag the outer child_window so the app shell can
        # hide/show a panel for Ctrl+1/2/3/4 maximize behaviour.
        self.container_tag = f"panelbox_{self._suffix}"
        self._has_image = False

    # ---- texture setup -------------------------------------------------

    def register_texture(self) -> None:
        """Call once inside a ``texture_registry`` context before
        building the plot."""
        blank = np.zeros(self.size * self.size * 4, dtype=np.float32)
        dpg.add_dynamic_texture(
            width=self.size, height=self.size,
            default_value=list(blank),
            tag=self.tex_tag,
        )

    # ---- UI ------------------------------------------------------------

    def build(self, parent: int | str) -> None:
        with dpg.child_window(parent=parent,
                              tag=self.container_tag,
                              width=self.size + 16,
                              height=self.size + 40,
                              border=True):
            dpg.add_text(self.title)
            dpg.add_separator()
            with dpg.plot(
                tag=self.plot_tag,
                width=self.size, height=self.size,
                no_menus=True, no_box_select=True,
                no_mouse_pos=True, no_title=True,
                equal_aspects=True,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, tag=self.x_axis_tag,
                                  no_tick_labels=True,
                                  no_gridlines=True, no_tick_marks=True)
                dpg.add_plot_axis(dpg.mvYAxis, tag=self.y_axis_tag,
                                  no_tick_labels=True,
                                  no_gridlines=True, no_tick_marks=True,
                                  invert=True)
                dpg.add_image_series(
                    self.tex_tag,
                    bounds_min=[0, 0],
                    bounds_max=[self.size, self.size],
                    parent=self.y_axis_tag,
                    tag=self.series_tag,
                )

            # Overlay drop-zone affordance — one big label centred on
            # the plot area. Hidden as soon as an image lands.
            dpg.add_text(
                "Drop a hologram here\n(or File → Load hologram, Ctrl+O)",
                tag=self.dropzone_tag,
                color=ScientificTheme.TEXT_MUTED,
                pos=(16, self.size // 2),
                wrap=self.size - 32,
            )

    # ---- data ---------------------------------------------------------

    def set_image(self, data: np.ndarray,
                  *, contrast: str = "minmax") -> None:
        """Push a 2-D float array into the texture. Resamples to
        ``size`` × ``size`` for the preview (full-res always through
        core/).

        ``contrast="percentile"`` enables the 1–99% clip useful for
        reference-divided amplitude images where outliers would
        otherwise flatten the histogram."""
        from .app import _to_rgba  # lazy to avoid circular import
        rgba = _to_rgba(_resize(data, self.size, self.size),
                        colormap=self.colormap,
                        contrast=contrast)
        dpg.set_value(self.tex_tag, rgba.flatten())
        if not self._has_image:
            self._has_image = True
            try:
                dpg.configure_item(self.dropzone_tag, show=False)
            except Exception:
                pass
        # Force the plot to fit the image bounds on first set — after
        # that the user's zoom/pan state is preserved across updates.
        try:
            dpg.fit_axis_data(self.x_axis_tag)
            dpg.fit_axis_data(self.y_axis_tag)
        except Exception:
            pass

    def reset_zoom(self) -> None:
        try:
            dpg.fit_axis_data(self.x_axis_tag)
            dpg.fit_axis_data(self.y_axis_tag)
        except Exception:
            pass

    # ---- responsive sizing --------------------------------------------

    def resize(self, new_size: int) -> None:
        """Grow or shrink the panel in place. Rebuilds the dynamic
        texture at the new pixel budget and resizes the wrapping
        child_window + plot so the layout reflow picks up the change.

        Dear PyGui doesn't expose a texture-resize API, so we delete the
        old texture inside its registry and re-register at the new
        dimensions. The ``tex_tag`` is preserved so every ``set_value``
        call site keeps working."""
        new_size = int(new_size)
        if new_size == self.size or new_size < 64:
            return
        self.size = new_size
        # Rebuild the texture at the new dimensions.
        try:
            registry = dpg.get_item_parent(self.tex_tag)
        except Exception:
            registry = None
        try:
            dpg.delete_item(self.tex_tag)
        except Exception:
            pass
        blank = np.zeros(new_size * new_size * 4, dtype=np.float32)
        if registry is not None:
            dpg.add_dynamic_texture(
                width=new_size, height=new_size,
                default_value=list(blank),
                tag=self.tex_tag,
                parent=registry,
            )
        else:
            # Fall back to a fresh registry — only happens if the parent
            # handle went stale (shouldn't under normal teardown).
            with dpg.texture_registry(show=False):
                dpg.add_dynamic_texture(
                    width=new_size, height=new_size,
                    default_value=list(blank),
                    tag=self.tex_tag,
                )
        # Reconfigure plot size + series bounds. Image series needs the
        # upper bound refreshed or the texture stretches.
        try:
            dpg.configure_item(self.plot_tag, width=new_size, height=new_size)
            dpg.configure_item(self.series_tag,
                               bounds_min=[0, 0],
                               bounds_max=[new_size, new_size])
        except Exception:
            pass
        # If the panel had a live image, the texture is now blank until
        # the next set_image — flag it so drop-zone affordance returns.
        self._has_image = False
        try:
            dpg.configure_item(self.dropzone_tag, show=True,
                               pos=(16, new_size // 2),
                               wrap=new_size - 32)
        except Exception:
            pass


def _resize(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
    h, w = arr.shape
    if (h, w) == (target_h, target_w):
        return arr.astype(np.float32, copy=False)
    y_idx = np.linspace(0, h - 1, target_h).astype(np.int32)
    x_idx = np.linspace(0, w - 1, target_w).astype(np.int32)
    return arr[y_idx][:, x_idx].astype(np.float32, copy=False)


__all__ = ["ZoomableImagePanel"]
