"""Census: every live-store destroyer in trw-memory runs under an EXCLUSIVE ``store_access`` (PRD-CORE-306).

The scan finds each function that renames or moves a file onto a path, unlinks a
``-wal``/``-shm``/``-journal`` sidecar, copies into a database with the backup API,
or drops a table. Each one found must be listed below: wrapped (an entry point
whose ``with store_access(..., <destructive op>)`` body reaches it), pending on
slice S3 (schema migrations, which run at OPEN), or not a store (one line why).
A new site, or a listed one that no longer exists, fails by name.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

import trw_memory
from trw_memory._store_lock import _SHARED

_SRC = Path(trw_memory.__file__).parent
_SIDECARS = ("-wal", "-shm", "-journal")
_RECOVER, _RESTORE = ("storage/_recovery.py:recover_db", "recover"), ("", "restore")
_S3 = "S3: pending -- schema migrations run inside ensure_schema under the OPEN hold; S3 gives them MIGRATE"

#: site -> (entry point, op) when wrapped ("" = the site itself), else the reason it is not wrapped.
CENSUS: dict[str, tuple[str, str] | str] = {
    "storage/_corrupt_backup.py:rotate_corrupt_backup": _RECOVER,
    "storage/_corrupt_backup.py:_prune_one": _RECOVER,
    "storage/_connection.py:clear_journal_sidecars": _RECOVER,
    "storage/_snapshot.py:restore_from_snapshot": _RESTORE,
    "storage/_schema_migrations.py:_migrate_v7_retire_wiki_refs": _S3,
    "storage/_schema_v5.py:_rebuild_memories": _S3,
    "storage/_schema_v5.py:_rebuild_graph_edges": _S3,
    "storage/_schema_v5.py:_rebuild_wiki_refs": _S3,
    "storage/_schema_v5.py:_rebuild_vec_index": _S3,
    "storage/_schema_v5.py:_apply_rebuilds": _S3,
    "storage/_schema_backup.py:snapshot_before_migration": "not a store: backs the store up into a new file",
    "storage/_snapshot.py:create_snapshot": "not a store: VACUUM INTO a new file, renamed onto the snapshot",
    "daemon/_paths.py:write_secret_file": "not a store: the daemon's finalize rename of a new secret file",
    "lifecycle/tiers/_warm.py:_replace_sidecar": "not a store: the warm tier's own sidecar file",
    "security/audit.py:compact": "not a store: the JSONL audit log's compaction",
    "storage/_recovery_preflight.py:_write_json_atomic": "not a store: <db>.recovery.json and <db>.recovering",
    "storage/persistence.py:write_yaml": "not a store: a YAML entry file (YAML refusals are slice S4)",
    "sync/retry_queue.py:_write_all": "not a store: the sync retry queue's file",
    # PRD-CORE-337: safe_fs publishes checkout config/state/hook files (callers: trw_mcp._checkout_write,
    # bootstrap, credentials); trw_memory.storage never writes a store through it.
    "safe_fs.py:_publish_at": "not a store: safe_fs's descriptor-relative publish of a checkout file",
    "safe_fs.py:_write_best_effort": "not a store: safe_fs's best-effort (no dir_fd) publish of a checkout file",
}

#: Staticmethod aliases a call reaches its site through.
_ALIASES = {"_rotate_corrupt_backup": "rotate_corrupt_backup", "_prune_corrupt_backups": "prune_corrupt_backups"}


def _own_nodes(fn: ast.AST) -> list[ast.AST]:
    """*fn*'s nodes, without those of functions nested in it (each is its own site)."""
    nodes, todo = [], list(ast.iter_child_nodes(fn))
    while todo:
        node = todo.pop()
        nodes.append(node)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            todo.extend(ast.iter_child_nodes(node))
    return nodes


def _destroys(nodes: list[ast.AST]) -> bool:
    calls = [n.func for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    owner = {id(f): (f.value.id if isinstance(f.value, ast.Name) else "") for f in calls}
    one_arg = {id(n.func) for n in nodes if isinstance(n, ast.Call) and len(n.args) == 1 and not n.keywords}
    texts = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    sidecar = any(t.endswith(_SIDECARS) for t in texts) or any(
        isinstance(n, ast.Name) and n.id == "_SIDECAR_SUFFIXES" for n in nodes
    )
    return (
        any(t.lstrip().upper().startswith("DROP TABLE") for t in texts)
        or any(f.attr in ("rename", "backup") or (f.attr, owner[id(f)]) == ("move", "shutil") for f in calls)
        or any(f.attr == "replace" and (owner[id(f)] == "os" or id(f) in one_arg) for f in calls)
        or (sidecar and any(f.attr in ("unlink", "remove") for f in calls))
    )


@cache
def _functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    found = {}
    for path in sorted(_SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[f"{path.relative_to(_SRC).as_posix()}:{node.name}"] = node
    return found


def _sites() -> set[str]:
    return {key for key, fn in _functions().items() if _destroys(_own_nodes(fn))}


def _locked(fn: ast.AST) -> list[tuple[str, list[ast.stmt]]]:
    """``(op, body)`` of each ``with store_access(..., <op>)`` in *fn* whose op is EXCLUSIVE."""
    return [
        (op.value, node.body)
        for node in ast.walk(fn)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(call := item.context_expr, ast.Call)
        and getattr(call.func, "id", "") == "store_access"
        and isinstance(op := call.args[-1], ast.Constant)
        and op.value not in _SHARED
    ]


def _reaches(body: list[ast.stmt], site: str, functions: dict[str, ast.AST]) -> bool:
    """Whether the calls in *body* reach *site*, following calls by name through the package."""
    by_name: dict[str, list[str]] = {}
    for key in functions:
        by_name.setdefault(key.rsplit(":", 1)[1], []).append(key)
    seen: set[str] = set()
    todo = [n for stmt in body for n in ast.walk(stmt)]
    while todo:
        node = todo.pop()
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for key in by_name.get(_ALIASES.get(name, name), []):
            if key == site:
                return True
            if key not in seen:
                seen.add(key)
                todo.extend(ast.walk(functions[key]))
    return False


def test_the_scanner_finds_a_known_destroyer() -> None:
    """Guard the guard: a scan that finds nothing would pass every other assertion."""
    sites = _sites()
    assert {"storage/_snapshot.py:restore_from_snapshot", "storage/_corrupt_backup.py:rotate_corrupt_backup"} <= sites
    assert "storage/_schema_v5.py:_rebuild_memories" in sites


def test_every_destroyer_is_listed_and_every_listing_is_live() -> None:
    sites = _sites()
    assert sorted(sites - set(CENSUS)) == [], "a new store destroyer: wrap it in store_access, or list why not"
    assert sorted(set(CENSUS) - sites) == [], "a listed site no longer destroys anything: drop its entry"


def test_every_wrapped_destroyer_runs_under_an_exclusive_op() -> None:
    functions = _functions()
    unwrapped = []
    for site, entry in CENSUS.items():
        if isinstance(entry, str):
            continue
        where, op = entry
        bodies = [body for held, body in _locked(functions[where or site]) if held == op]
        if where:
            ok = any(_reaches(body, site, functions) for body in bodies)
        else:  # the site itself: its destroying calls sit inside the with
            ok = _destroys([n for body in bodies for stmt in body for n in ast.walk(stmt)])
        if not ok:
            unwrapped.append(site)
    assert unwrapped == []


def test_the_pending_and_unwrapped_entries_are_reasoned() -> None:
    reasons = [entry for entry in CENSUS.values() if isinstance(entry, str)]
    assert all(r.startswith(("S3: pending", "not a store: ")) for r in reasons)
