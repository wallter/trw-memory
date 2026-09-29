"""Census: every process-local registry in a fork-aware module is reset by its fork handler (B71-133 (d), Q2).

A forked child inherits every module global, but not the threads, locks or
descriptors they describe. A module that registers an ``os.register_at_fork``
child handler (found by scan, plus the modules :data:`REQUIRED` names) must
reset in that handler each mutable registry it holds: a module-level container
or lock, and each such attribute ``__init__`` gives a singleton whose method is
the handler. "Reset" means the handler's own body assigns it anew, clears it,
or calls a reset/reinit on it: reading or appending to it is not. Anything
else is listed in :data:`NOT_RESET` with the reason it is safe to inherit. A new
registry, or a listed one that no longer exists or is now reset, fails by name.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

import trw_memory

_SRC = Path(trw_memory.__file__).parent

#: Modules that hold process-local state and must keep a fork handler, found or not.
REQUIRED = ("_live_stores.py", "_store_lock.py", "_graph_worker_pool.py", "_graph_threads.py", "daemon/_lane.py")

#: "module:registry" (a module global, or "Class.attr" for a singleton's attribute) -> why it is not reset.
NOT_RESET = {
    "_live_stores.py:_LEASE_RELEASED": "a Condition over FD_LOCK, which the handler releases",
    "_live_stores.py:_OPEN": "emptied by the handler's release of every inherited connection (_apply_release)",
    "_live_stores.py:_LEASES": "reader descriptors are inherited open, so their leases still hold in the child",
    "_live_stores.py:_PARKED": "parked descriptors are inherited open; each closes with its store's last connection",
    "_live_stores.py:_FACTORIES": "a cache of connection classes, no process state",
    "_live_stores.py:_READ_LOCKS": "a stale count only defers a reader close (a descriptor, never a lock: none is inherited)",
    "_live_stores.py:_DEFERRED_CLOSES": "the descriptors those stale counts defer; closed if the count ever drops",
    "_live_stores.py:_LOCK_FILES": "emptied by _store_lock's handler (track_lock_file on each inherited lock file)",
    "_live_stores.py:_FINALIZED": "queued releases are idempotent (a done flag), and the handler released them all",
    "_live_stores.py:FD_LOCK": "held across the fork by the 'before' hook, and released (not replaced) by the handler",
    "_live_stores.py:_CONNECTIONS": "the handler releases or quarantines every entry; the weak map drops them as collected",
    "_live_stores.py:_KEPT_AFTER_FORK": "the handler appends each quarantined connection: it must stay referenced forever",
    "_graph_worker_pool.py:_GraphWorkerPool._abandoned": "the parent's workers, kept referenced and never closed in the child",
    "_store_lock.py:WAITS": "a constant table",
    "_store_lock.py:_HOW": "a constant table",
}

_CONTAINERS = {
    "dict",
    "set",
    "list",
    "deque",
    "defaultdict",
    "OrderedDict",
    "WeakKeyDictionary",
    "WeakValueDictionary",
    "WeakSet",
    "Lock",
    "RLock",
    "Condition",
}


def _is_registry(value: ast.expr | None) -> bool:
    if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        return name in _CONTAINERS
    return False


def _assigned(node: ast.stmt) -> list[tuple[ast.expr, ast.expr | None]]:
    if isinstance(node, ast.Assign):
        return [(target, node.value) for target in node.targets]
    if isinstance(node, ast.AnnAssign):
        return [(node.target, node.value)]
    return []


def _key(node: ast.AST) -> str | None:
    """A module global (``X``) or a ``self`` attribute (``self.X``) as ``X``; anything else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _resets(body: ast.AST) -> set[str]:
    """What the handler RESETS: assigns anew (``X = ...``, ``self.X = ...``), empties (``X.clear()``) or
    re-initializes (``X.reset()``, ``X.reinit()``). Reading or appending to a registry is not a reset."""
    keys: set[str | None] = set()
    for node in ast.walk(body):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            keys |= {_key(target) for target, _ in _assigned(node)}
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "clear" or "reset" in node.func.attr or "reinit" in node.func.attr:
                keys.add(_key(node.func.value))
    return {key for key in keys if key is not None}


def _handlers(tree: ast.Module) -> list[ast.expr]:
    return [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register_at_fork"
        for keyword in node.keywords
        if keyword.arg == "after_in_child"
    ]


def _module_census(module: str, source: str) -> tuple[set[str], set[str], bool]:
    """*module*'s registries, the ones its fork handlers reset, and whether it registers a handler at all."""
    tree = ast.parse(source)
    top = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    singletons = {
        target.id: value.func.id
        for node in tree.body
        for target, value in _assigned(node)
        if isinstance(target, ast.Name) and isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
        if isinstance(top.get(value.func.id), ast.ClassDef)
    }
    registries = {
        target.id
        for node in tree.body
        for target, value in _assigned(node)
        if isinstance(target, ast.Name) and not target.id.startswith("__") and _is_registry(value)
    }
    module_globals, reset = set(registries), set()
    for handler in _handlers(tree):
        if isinstance(handler, ast.Name):  # a module function: resets module globals
            reset |= _resets(top[handler.id]) & module_globals
            continue
        assert isinstance(handler, ast.Attribute) and isinstance(handler.value, ast.Name), ast.dump(handler)
        cls = top[singletons[handler.value.id]]
        assert isinstance(cls, ast.ClassDef)
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        attrs = {
            target.attr
            for node in ast.walk(methods["__init__"])
            if isinstance(node, ast.stmt)
            for target, value in _assigned(node)
            if isinstance(target, ast.Attribute) and _is_registry(value)
        }
        registries |= {f"{cls.name}.{attr}" for attr in attrs}
        reset |= {f"{cls.name}.{attr}" for attr in _resets(methods[handler.attr]) & attrs}
    return registries, reset, bool(_handlers(tree))


@cache
def _census() -> tuple[dict[str, set[str]], set[str], list[str]]:
    """{module: its registries}, the registries some handler of their module resets, and the modules with none."""
    registries: dict[str, set[str]] = {}
    reset: set[str] = set()
    unhandled: list[str] = []
    found = {str(p.relative_to(_SRC)) for p in _SRC.rglob("*.py") if "register_at_fork" in p.read_text()}
    for module in sorted(found | set(REQUIRED)):
        registries[module], module_reset, handled = _module_census(module, (_SRC / module).read_text())
        reset |= {f"{module}:{name}" for name in module_reset}
        if not handled:
            unhandled.append(module)
    return registries, reset, unhandled


def test_every_registry_in_a_fork_aware_module_is_reset_or_listed() -> None:
    registries, reset, unhandled = _census()
    assert not unhandled, f"these hold process-local state but register no fork handler: {unhandled}"
    every = {f"{module}:{name}" for module, names in registries.items() for name in names}
    unlisted = sorted(every - reset - set(NOT_RESET))
    assert not unlisted, f"reset these in the module's fork handler, or list them in NOT_RESET with why: {unlisted}"
    stale = sorted(set(NOT_RESET) - (every - reset))
    assert not stale, f"no longer a registry, or now reset: remove from NOT_RESET: {stale}"


def test_the_census_sees_the_registries_it_is_about() -> None:
    """The scan finds the singletons' registries, not just module globals (a guard on the census itself)."""
    registries, reset, _ = _census()
    assert "_GraphWorkerPool._owner_pending" in registries["_graph_worker_pool.py"]
    assert "_graph_threads.py:_GraphThreadRegistry._guard" in reset
    assert "daemon/_lane.py:_queue" in reset


_READ_ONLY_HANDLER = """
import os, threading

class _Pool:
    def __init__(self):
        self._lock = threading.Lock()
        self._owner_pending = {}

    def after_fork_in_child(self):
        self._lock = threading.Lock()
        if len(self._owner_pending):  # read, not reset
            pass

_POOL = _Pool()
os.register_at_fork(after_in_child=_POOL.after_fork_in_child)
"""


def test_a_handler_that_only_reads_a_registry_does_not_reset_it() -> None:
    """sol r1 P2: naming a registry in the handler used to count as resetting it."""
    registries, reset, handled = _module_census("synthetic.py", _READ_ONLY_HANDLER)
    assert handled
    assert registries == {"_Pool._lock", "_Pool._owner_pending"}
    assert reset == {"_Pool._lock"}
