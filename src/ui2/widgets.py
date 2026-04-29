"""Reusable Dear PyGui helpers shared by the app shell.

Nothing here owns application state — each helper takes callbacks or
config and returns a tag the caller can configure later. Keeping it
stateless makes the ``app`` module a pure wiring script.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Iterable, List, Optional, Tuple

import dearpygui.dearpygui as dpg

from .theme import ScientificTheme


# ---------------------------------------------------------------------------
# Toast notifications — stacked right-top, auto-dismiss
# ---------------------------------------------------------------------------

class ToastStack:
    """A column of short-lived notification windows at the top-right."""

    TOAST_WIDTH = 320
    TOAST_HEIGHT = 56
    TOAST_MARGIN = 12
    DEFAULT_TTL = 4.5  # seconds

    def __init__(self) -> None:
        self._active: List[Tuple[str, float]] = []  # (tag, expires_at)

    def show(self, message: str, *, level: str = "info",
             ttl: Optional[float] = None) -> str:
        color = {
            "info":   ScientificTheme.TEXT,
            "ok":     ScientificTheme.SUCCESS,
            "warn":   ScientificTheme.WARN,
            "danger": ScientificTheme.DANGER,
        }.get(level, ScientificTheme.TEXT)
        tag = f"toast_{uuid.uuid4().hex[:8]}"
        with dpg.window(tag=tag, no_title_bar=True, no_resize=True,
                        no_move=True, no_collapse=True,
                        no_scrollbar=True, no_focus_on_appearing=True,
                        width=self.TOAST_WIDTH, height=self.TOAST_HEIGHT,
                        pos=(0, 0)):
            dpg.add_text(message, color=color, wrap=self.TOAST_WIDTH - 20)

        self._active.append(
            (tag, time.monotonic() + (ttl or self.DEFAULT_TTL)),
        )
        self._relayout()
        return tag

    def tick(self) -> None:
        """Call once per frame — dismisses expired toasts."""
        now = time.monotonic()
        survivors: List[Tuple[str, float]] = []
        for tag, expires in self._active:
            if now >= expires:
                try:
                    dpg.delete_item(tag)
                except Exception:
                    pass
            else:
                survivors.append((tag, expires))
        if len(survivors) != len(self._active):
            self._active = survivors
            self._relayout()

    def _relayout(self) -> None:
        try:
            vw = dpg.get_viewport_client_width()
        except Exception:
            return
        x = max(0, vw - self.TOAST_WIDTH - self.TOAST_MARGIN)
        y = self.TOAST_MARGIN + 30  # below menu bar
        for tag, _ in self._active:
            try:
                dpg.configure_item(tag, pos=(x, y))
            except Exception:
                pass
            y += self.TOAST_HEIGHT + 8


# ---------------------------------------------------------------------------
# Command palette — fuzzy ⌘K-style launcher
# ---------------------------------------------------------------------------

class CommandPalette:
    """Lightweight command runner. Host registers (id, title, callback,
    hint) tuples; opening the palette pops a modal with a live-filter
    input. Enter fires the top match."""

    TAG_WINDOW = "cmd_palette_window"
    TAG_INPUT = "cmd_palette_input"
    TAG_LIST = "cmd_palette_list"

    def __init__(self) -> None:
        self._commands: List[dict] = []
        self._filtered: List[dict] = []

    # ---- registration -------------------------------------------------
    def register(self, cmd_id: str, title: str, callback: Callable[[], Any],
                 *, hint: str = "", category: str = "") -> None:
        self._commands.append({
            "id": cmd_id, "title": title, "callback": callback,
            "hint": hint, "category": category,
        })

    def clear(self) -> None:
        self._commands.clear()

    # ---- UI -----------------------------------------------------------
    def build(self) -> None:
        with dpg.window(label="Commands", tag=self.TAG_WINDOW, modal=True,
                        show=False, no_resize=True, no_collapse=True,
                        width=520, height=420):
            dpg.add_input_text(tag=self.TAG_INPUT, hint="Search commands…",
                               width=-1,
                               callback=lambda s, a: self._refresh(a),
                               on_enter=True)
            dpg.add_separator()
            dpg.add_child_window(tag=self.TAG_LIST, autosize_x=True,
                                 autosize_y=True, border=False)

    def show(self) -> None:
        if not dpg.does_item_exist(self.TAG_WINDOW):
            self.build()
        dpg.set_value(self.TAG_INPUT, "")
        self._refresh("")
        dpg.configure_item(self.TAG_WINDOW, show=True)
        dpg.focus_item(self.TAG_INPUT)

    def hide(self) -> None:
        try:
            dpg.configure_item(self.TAG_WINDOW, show=False)
        except Exception:
            pass

    def invoke_top(self) -> None:
        """Fire the top match, or surface a quiet hint if there isn't one.

        v2.0.1 did nothing on empty — the user pressed Enter, got zero
        feedback, and was left to figure out whether the palette had
        frozen. Now we update the 'No matches' line colour so the
        Enter acknowledgement is visible."""
        if self._filtered:
            self._invoke(self._filtered[0])
            return
        # No matches — flash the help line so Enter feels like it
        # *did* something, even if nothing launched.
        try:
            dpg.delete_item(self.TAG_LIST, children_only=True)
            dpg.add_text(
                "No matches — try a different query or press Esc to close.",
                parent=self.TAG_LIST,
                color=ScientificTheme.WARN,
            )
        except Exception:
            pass

    # ---- internals ----------------------------------------------------
    def _refresh(self, query: str) -> None:
        q = (query or "").strip().lower()
        if not q:
            self._filtered = list(self._commands)
        else:
            def score(cmd):
                hay = (cmd["title"] + " " + cmd["hint"] + " "
                       + cmd["category"]).lower()
                return hay.find(q), cmd["title"].lower().find(q)

            matches = [c for c in self._commands
                       if q in (c["title"] + " " + c["hint"]
                                + " " + c["category"]).lower()]
            matches.sort(key=score)
            self._filtered = matches

        try:
            dpg.delete_item(self.TAG_LIST, children_only=True)
        except Exception:
            return
        if not self._filtered:
            dpg.add_text("No matches.", parent=self.TAG_LIST,
                         color=ScientificTheme.TEXT_MUTED)
            return
        for cmd in self._filtered[:40]:
            cat = cmd["category"]
            label = f"{cat:<10}  {cmd['title']}" if cat else cmd["title"]
            dpg.add_button(
                label=label, parent=self.TAG_LIST,
                width=-1, height=28,
                callback=lambda s, a, u=cmd: self._invoke(u),
            )
            if cmd["hint"]:
                dpg.add_text(f"    {cmd['hint']}", parent=self.TAG_LIST,
                             color=ScientificTheme.TEXT_MUTED)

    def _invoke(self, cmd: dict) -> None:
        self.hide()
        try:
            cmd["callback"]()
        except Exception as exc:  # noqa: BLE001
            dpg.configure_item(self.TAG_WINDOW, show=False)
            raise


# ---------------------------------------------------------------------------
# Help overlay — list visible commands/controls
# ---------------------------------------------------------------------------

def show_help_overlay(palette: CommandPalette, parent_tag: str = "help_overlay") -> None:
    """Render a simple modal listing every registered command's title
    + hint. Cheap DPG equivalent of the PySide HelpOverlay."""
    if dpg.does_item_exist(parent_tag):
        dpg.configure_item(parent_tag, show=True)
        return
    with dpg.window(label="Help — available commands", modal=True,
                    tag=parent_tag, width=560, height=440,
                    no_collapse=True):
        dpg.add_text(
            f"{len(palette._commands)} command(s) registered. "
            "Press Ctrl+K to open the palette; type to filter.",
            color=ScientificTheme.TEXT_MUTED, wrap=520,
        )
        dpg.add_separator()
        with dpg.child_window(autosize_x=True, autosize_y=True,
                              border=False):
            for cmd in palette._commands:
                cat = cmd["category"] or "General"
                dpg.add_text(f"[{cat}]  {cmd['title']}")
                if cmd["hint"]:
                    dpg.add_text(f"    {cmd['hint']}",
                                 color=ScientificTheme.TEXT_MUTED,
                                 wrap=520)


# ---------------------------------------------------------------------------
# Validation indicator — small coloured dot
# ---------------------------------------------------------------------------

def make_validation_dot(tag: str) -> str:
    """Add a fixed-size text label that we repaint as green/red dots."""
    dpg.add_text("•", tag=tag, color=ScientificTheme.TEXT_MUTED)
    return tag


def set_validation_state(tag: str, valid: Optional[bool],
                         reason: str = "") -> None:
    try:
        if valid is True:
            dpg.configure_item(tag, color=ScientificTheme.SUCCESS)
            with dpg.tooltip(tag):
                pass
        elif valid is False:
            dpg.configure_item(tag, color=ScientificTheme.DANGER)
        else:
            dpg.configure_item(tag, color=ScientificTheme.TEXT_MUTED)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Preset chips — exclusive radio selection row
# ---------------------------------------------------------------------------

class PresetChips:
    """One button per preset — exclusive selection with highlight.

    v2.0.6 adds :meth:`rebuild` so the host app can swap the label set
    at runtime (user-defined presets appear as extra chips). The
    ``_group_tag`` caches the parent group so rebuilds don't need to
    know where the chips live."""

    def __init__(self, labels: Iterable[str],
                 on_select: Callable[[str], None]) -> None:
        self._labels = list(labels)
        self._on_select = on_select
        self._buttons: List[str] = []
        self._active: Optional[str] = None
        self._group_tag: Optional[str] = None

    def build(self, parent: int | str) -> None:
        self._group_tag = f"preset_chips_group_{uuid.uuid4().hex[:6]}"
        with dpg.group(horizontal=True, parent=parent,
                       tag=self._group_tag):
            self._build_buttons()
        if self._labels:
            self.set_active(self._labels[0])

    def rebuild(self, labels: Iterable[str]) -> None:
        """Replace the chip set. Preserves the active selection if
        the new label set still contains it."""
        if self._group_tag is None:
            # ``build`` was never called — nothing to rebuild on.
            self._labels = list(labels)
            return
        prev_active = self._active
        self._labels = list(labels)
        for tag in self._buttons:
            try:
                dpg.delete_item(tag)
            except Exception:
                pass
        self._buttons.clear()
        # Re-add under the same parent group so the row stays in place.
        # DPG widget mutations already serialise through the render
        # loop; no extra locking needed from the UI thread.
        self._build_buttons(parent=self._group_tag)
        # Restore selection when possible.
        if prev_active and prev_active in self._labels:
            self.set_active(prev_active)
        elif self._labels:
            self.set_active(self._labels[0])
        else:
            self._active = None

    def _build_buttons(self, parent: Optional[str] = None) -> None:
        for label in self._labels:
            tag = (f"preset_chip_{label.replace(' ', '_').lower()}_"
                   f"{uuid.uuid4().hex[:6]}")
            kwargs = dict(
                label=label, tag=tag,
                callback=lambda s, a, u=label: self._click(u),
            )
            if parent is not None:
                kwargs["parent"] = parent
            dpg.add_button(**kwargs)
            self._buttons.append(tag)

    def set_active(self, label: str) -> None:
        self._active = label
        for tag in self._buttons:
            is_active = dpg.get_item_label(tag) == label
            try:
                dpg.configure_item(
                    tag,
                    enabled=not is_active,  # disabled-looking = "selected"
                )
            except Exception:
                pass

    def _click(self, label: str) -> None:
        self.set_active(label)
        self._on_select(label)

    @property
    def active(self) -> Optional[str]:
        return self._active

    @property
    def labels(self) -> List[str]:
        return list(self._labels)


__all__ = [
    "ToastStack", "CommandPalette",
    "show_help_overlay",
    "make_validation_dot", "set_validation_state",
    "PresetChips",
]
