"""Command registry — single source of truth for every user-invocable action.

Before v1.0.1 the app spelled its commands in four incompatible dialects:

* ``QShortcut`` instances in ``main_window.py`` (lines 354–378)
* ``QAction`` instances in ``toolbar.py``
* ``QAction`` instances in ``main_window._init_menus`` (Tools / Help)
* raw ``button.clicked.connect`` calls in every sidebar tab

That is fine for a v1.0.0 — the app worked — but it blocks everything the
lab asked for on review: a ⌘K palette needs **one** list; user-customizable
keybindings need **one** place to resolve them; a command log needs one
vocabulary of ids. This module is that list.

Design decisions
----------------
* The ``Command`` dataclass is **pure Python** — no Qt imports at the top
  of the module. This lets the core registry be unit-tested without a
  ``QApplication`` and keeps the semantics (id, title, category, shortcut)
  separable from the widget plumbing.
* Qt wiring lives in :func:`bind_to_qt` at the bottom; call it once per
  command, typically from ``main_window._install_commands``. The binding
  returns a ``QAction`` the caller then attaches to menus / toolbars.
* Command ids are dotted strings — ``<category>.<subtopic>.<verb>`` as a
  soft rule, e.g. ``reconstruct.run``, ``view.maximize.phase``,
  ``autofocus.run``. Treat them as stable: keybinding customization and
  audit logs key off them.
* One global :class:`CommandRegistry` singleton via :func:`get_registry`;
  tests instantiate :class:`CommandRegistry` directly when they need
  isolation.

Rams #4 (understandable) applies: every command needs a ``title`` in
plain English and a short ``hint`` the palette can render as the second
line. Skip the hint for trivially obvious commands.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

CommandCallback = Callable[[], None]
EnabledPredicate = Callable[[], bool]


@dataclass(frozen=True)
class Command:
    """A single user-invocable action.

    Fields
    ------
    id: stable dotted identifier — used for keybinding persistence, audit
        logs, and palette filtering. Must not contain whitespace.
    title: short human-readable label (< 40 chars). Shown in menus, the
        palette, and the toolbar when there's room.
    category: section label for menu grouping and palette filtering. Free
        text, but prefer the constants in :data:`CATEGORIES`.
    callback: zero-argument function invoked when the command runs.
    shortcut: default key sequence string (Qt form, e.g. ``"Ctrl+R"``) or
        ``""`` for no default. Users can rebind — the registry stores the
        default, never the override.
    hint: optional one-line context shown as a second line in the palette.
        Keep under 80 chars.
    when: optional zero-arg predicate that returns True iff the command
        is currently actionable. Menus grey the action out when it
        returns False. Defaults to "always available".
    visible_in_palette: set False for commands that only make sense as
        menu entries (e.g. opening a sub-menu). Defaults to True.
    """

    id: str
    title: str
    category: str
    callback: CommandCallback
    shortcut: str = ""
    hint: str = ""
    when: EnabledPredicate | None = None
    visible_in_palette: bool = True

    def __post_init__(self) -> None:
        if not self.id or any(ch.isspace() for ch in self.id):
            raise ValueError(f"Command id must be non-empty whitespace-free: {self.id!r}")
        if not self.title:
            raise ValueError(f"Command {self.id} has no title")


# ---------------------------------------------------------------------------
# Category constants — pick from these when registering, or pass a custom
# string if the command truly doesn't fit. Keeping this list tight beats
# letting it drift into 20 near-duplicates.
# ---------------------------------------------------------------------------

class Categories:
    FILE = "File"
    VIEW = "View"
    RECONSTRUCT = "Reconstruct"
    AUTOFOCUS = "Autofocus"
    QPI = "QPI"
    CAMERA = "Camera"
    RECORD = "Record"
    TOOLS = "Tools"
    NAVIGATE = "Navigate"
    HELP = "Help"


CATEGORIES = (
    Categories.FILE, Categories.VIEW, Categories.RECONSTRUCT,
    Categories.AUTOFOCUS, Categories.QPI, Categories.CAMERA,
    Categories.RECORD, Categories.TOOLS, Categories.NAVIGATE,
    Categories.HELP,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CommandRegistry:
    """Ordered, thread-safe store of :class:`Command` instances.

    Not a heavyweight container — the app has O(30) commands. The lock is
    there because the registry is written during ``main_window`` setup
    (main thread) and read from worker threads that log the id into the
    audit trail; both uses are short, so contention is negligible.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Command] = {}
        self._order: list[str] = []  # preserves registration order
        self._lock = threading.RLock()

    # -- write --------------------------------------------------------------

    def register(self, cmd: Command, *, replace: bool = False) -> None:
        """Add a command. Raises ``KeyError`` on duplicate id unless
        ``replace=True`` (useful for hot-reload during development)."""
        with self._lock:
            if cmd.id in self._by_id and not replace:
                raise KeyError(f"command already registered: {cmd.id}")
            if cmd.id not in self._by_id:
                self._order.append(cmd.id)
            self._by_id[cmd.id] = cmd
        _LOG.debug("commands: registered %s (%s)", cmd.id, cmd.title)

    def unregister(self, cmd_id: str) -> None:
        with self._lock:
            if cmd_id not in self._by_id:
                return
            del self._by_id[cmd_id]
            self._order.remove(cmd_id)

    def clear(self) -> None:
        """Drop every command. Primarily for tests."""
        with self._lock:
            self._by_id.clear()
            self._order.clear()

    # -- read ---------------------------------------------------------------

    def get(self, cmd_id: str) -> Command | None:
        with self._lock:
            return self._by_id.get(cmd_id)

    def __contains__(self, cmd_id: object) -> bool:
        return isinstance(cmd_id, str) and cmd_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Command]:
        """Iterate commands in registration order."""
        with self._lock:
            ids = list(self._order)
        for cid in ids:
            c = self._by_id.get(cid)
            if c is not None:
                yield c

    def all(self) -> list[Command]:
        """Snapshot of every command in registration order."""
        return list(iter(self))

    def by_category(self, category: str) -> list[Command]:
        return [c for c in self if c.category == category]

    def search(self, query: str, *, include_hidden: bool = False) -> list[Command]:
        """Naive substring/prefix match over ``id`` and ``title`` — the
        palette uses this for live filtering. Deliberately dumb: we rank
        title-matches before id-matches, but stop there. A smarter fuzzy
        matcher can drop in later without changing the API."""
        q = query.strip().lower()
        if not q:
            return [c for c in self if include_hidden or c.visible_in_palette]

        title_hits: list[Command] = []
        id_hits: list[Command] = []
        for c in self:
            if not include_hidden and not c.visible_in_palette:
                continue
            if q in c.title.lower():
                title_hits.append(c)
            elif q in c.id.lower() or q in c.hint.lower():
                id_hits.append(c)
        return title_hits + id_hits

    # -- invocation ---------------------------------------------------------

    def invoke(self, cmd_id: str) -> bool:
        """Run a command by id. Returns True if invoked, False if the id
        is unknown or the ``when`` predicate blocks it. Exceptions from
        the callback propagate — the caller decides whether to surface
        them through the error bus."""
        cmd = self.get(cmd_id)
        if cmd is None:
            _LOG.warning("commands: unknown id %s", cmd_id)
            return False
        if cmd.when is not None and not cmd.when():
            _LOG.debug("commands: %s blocked by when-predicate", cmd_id)
            return False
        cmd.callback()
        return True


# ---- singleton --------------------------------------------------------------

_REGISTRY: CommandRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> CommandRegistry:
    """Return the process-wide :class:`CommandRegistry`. Lazy so tests
    that never import the GUI avoid the allocation."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = CommandRegistry()
    return _REGISTRY


def reset_registry_for_tests() -> None:
    """Replace the singleton with a fresh registry. Use only in tests."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = CommandRegistry()


# ---------------------------------------------------------------------------
# Qt binding — kept at the bottom so the top of the module can be imported
# (and tested) without a QApplication. Lazy imports inside the function keep
# pytest collection fast for non-GUI suites.
# ---------------------------------------------------------------------------

def bind_to_qt(parent, cmd: Command):
    """Build a ``QAction`` wired to ``cmd``'s callback and default shortcut.

    The caller attaches the returned action to a menu / toolbar. We do not
    retain a reference — Qt parents do, via ``parent``.

    Returned QAction is already connected; calling ``action.trigger()``
    runs ``cmd.callback`` exactly once.
    """
    from PySide6.QtGui import QAction, QKeySequence

    action = QAction(cmd.title, parent)
    action.setObjectName(f"cmd_{cmd.id.replace('.', '_')}")
    if cmd.shortcut:
        action.setShortcut(QKeySequence(cmd.shortcut))
    if cmd.hint:
        action.setStatusTip(cmd.hint)
        action.setToolTip(cmd.hint)
    action.triggered.connect(cmd.callback)

    if cmd.when is not None:
        # Snapshot enablement at trigger time — re-evaluate on show via a
        # parent-level ``aboutToShow`` hook if the menu wants live greying.
        action.setEnabled(bool(cmd.when()))
    return action


def bind_many(parent, commands: Iterable[Command]) -> dict[str, object]:
    """Convenience: bind a batch and return ``{id: QAction}``."""
    out: dict[str, object] = {}
    for cmd in commands:
        out[cmd.id] = bind_to_qt(parent, cmd)
    return out


__all__ = [
    "Categories",
    "CATEGORIES",
    "Command",
    "CommandRegistry",
    "bind_many",
    "bind_to_qt",
    "get_registry",
    "reset_registry_for_tests",
]
