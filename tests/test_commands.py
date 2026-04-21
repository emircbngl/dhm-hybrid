"""Command registry — registration, lookup, search, invoke semantics.

These tests drive the pure-Python core. The Qt binding (``bind_to_qt``)
gets exercised indirectly when ``main_window`` wires itself up in the
GUI smoke tests — there's no point spinning up a ``QApplication`` here
just to verify that ``QAction`` holds a shortcut we already set.
"""
from __future__ import annotations

import pytest

from gui.commands import (
    Categories,
    Command,
    CommandRegistry,
    get_registry,
    reset_registry_for_tests,
)


def _noop() -> None:
    return None


def _cmd(cid: str = "a.b", title: str = "Do the thing", **kw) -> Command:
    return Command(
        id=cid,
        title=title,
        category=kw.pop("category", Categories.TOOLS),
        callback=kw.pop("callback", _noop),
        **kw,
    )


# ---- Command dataclass ------------------------------------------------------

def test_command_is_frozen():
    c = _cmd()
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        c.title = "other"  # type: ignore[misc]


def test_command_rejects_empty_id():
    with pytest.raises(ValueError):
        Command(id="", title="X", category="A", callback=_noop)


def test_command_rejects_whitespace_id():
    with pytest.raises(ValueError):
        Command(id="a b", title="X", category="A", callback=_noop)


def test_command_rejects_empty_title():
    with pytest.raises(ValueError):
        Command(id="a.b", title="", category="A", callback=_noop)


# ---- Registry: register / unregister / get ---------------------------------

def test_register_and_get():
    r = CommandRegistry()
    c = _cmd("reconstruct.run", "Reconstruct")
    r.register(c)
    assert r.get("reconstruct.run") is c
    assert "reconstruct.run" in r
    assert len(r) == 1


def test_register_duplicate_without_replace_raises():
    r = CommandRegistry()
    r.register(_cmd("x.y"))
    with pytest.raises(KeyError):
        r.register(_cmd("x.y", title="Different"))


def test_register_duplicate_with_replace_overwrites():
    r = CommandRegistry()
    r.register(_cmd("x.y", title="First"))
    r.register(_cmd("x.y", title="Second"), replace=True)
    got = r.get("x.y")
    assert got is not None and got.title == "Second"
    # Order preserved: still one entry, still in the same slot
    assert [c.id for c in r.all()] == ["x.y"]


def test_unregister_removes_command():
    r = CommandRegistry()
    r.register(_cmd("x.y"))
    r.unregister("x.y")
    assert r.get("x.y") is None
    assert "x.y" not in r


def test_unregister_unknown_is_silent():
    r = CommandRegistry()
    r.unregister("never.was")  # no raise


def test_clear_drops_everything():
    r = CommandRegistry()
    r.register(_cmd("a.1"))
    r.register(_cmd("a.2"))
    r.clear()
    assert len(r) == 0
    assert r.all() == []


# ---- Ordering & iteration ---------------------------------------------------

def test_iteration_preserves_registration_order():
    r = CommandRegistry()
    for cid in ("a", "c", "b", "d"):
        r.register(_cmd(cid))
    assert [c.id for c in r] == ["a", "c", "b", "d"]


def test_by_category_filters_by_category():
    r = CommandRegistry()
    r.register(_cmd("a.1", category=Categories.VIEW))
    r.register(_cmd("a.2", category=Categories.RECONSTRUCT))
    r.register(_cmd("a.3", category=Categories.VIEW))
    ids = [c.id for c in r.by_category(Categories.VIEW)]
    assert ids == ["a.1", "a.3"]


# ---- Search -----------------------------------------------------------------

def test_empty_query_returns_all_visible():
    r = CommandRegistry()
    r.register(_cmd("v.1", title="Visible"))
    r.register(_cmd("v.2", title="Hidden", visible_in_palette=False))
    out = r.search("")
    assert [c.id for c in out] == ["v.1"]


def test_empty_query_with_include_hidden_returns_all():
    r = CommandRegistry()
    r.register(_cmd("v.1", title="Visible"))
    r.register(_cmd("v.2", title="Hidden", visible_in_palette=False))
    out = r.search("", include_hidden=True)
    assert {c.id for c in out} == {"v.1", "v.2"}


def test_search_prefers_title_matches_over_id_matches():
    r = CommandRegistry()
    # "run" appears in id of first, title of second — title should come first
    r.register(_cmd("run.thing", title="Launch"))
    r.register(_cmd("other.thing", title="Run the thing"))
    out = r.search("run")
    assert [c.id for c in out] == ["other.thing", "run.thing"]


def test_search_matches_hint_when_title_and_id_miss():
    r = CommandRegistry()
    r.register(_cmd("a.b", title="Compute", hint="triggers reconstruction"))
    out = r.search("reconstruction")
    assert [c.id for c in out] == ["a.b"]


def test_search_is_case_insensitive():
    r = CommandRegistry()
    r.register(_cmd("a.b", title="Reconstruct"))
    assert r.search("RECONSTRUCT") and r.search("reconstruct")


# ---- Invocation -------------------------------------------------------------

def test_invoke_runs_callback():
    calls = {"n": 0}

    def inc() -> None:
        calls["n"] += 1

    r = CommandRegistry()
    r.register(_cmd("x.y", callback=inc))
    assert r.invoke("x.y") is True
    assert calls["n"] == 1


def test_invoke_unknown_id_returns_false():
    r = CommandRegistry()
    assert r.invoke("ghost") is False


def test_invoke_respects_when_predicate():
    calls = {"n": 0}

    def inc() -> None:
        calls["n"] += 1

    gate = {"open": False}
    r = CommandRegistry()
    r.register(_cmd("x.y", callback=inc, when=lambda: gate["open"]))
    assert r.invoke("x.y") is False
    assert calls["n"] == 0

    gate["open"] = True
    assert r.invoke("x.y") is True
    assert calls["n"] == 1


def test_invoke_propagates_exceptions():
    def boom() -> None:
        raise RuntimeError("boom")

    r = CommandRegistry()
    r.register(_cmd("x.y", callback=boom))
    with pytest.raises(RuntimeError, match="boom"):
        r.invoke("x.y")


# ---- Singleton --------------------------------------------------------------

def test_singleton_returns_same_instance():
    reset_registry_for_tests()
    a = get_registry()
    b = get_registry()
    assert a is b


def test_reset_replaces_singleton():
    a = get_registry()
    reset_registry_for_tests()
    b = get_registry()
    assert a is not b
