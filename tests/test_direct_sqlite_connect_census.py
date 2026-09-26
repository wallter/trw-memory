"""PRD-SEC-016 round-4 finding 3 -- every direct ``sqlite3.connect``/``dbapi.connect`` call is accounted for.

``storage/_connection.py::connect`` is described as "the ONE function every
SQLite open path... funnels through" (its own docstring, and PRD-SEC-016's
round-2 finding-4 evidence). That claim was FALSE: ``_resilient_fetch.py``
and ``_temporal_fetch.py`` reopened the live store directly after a UTF-8
decode error, bypassing the identity check entirely (round-4 finding 3;
fixed alongside this test). A grep-only claim like that rots silently the
next time someone adds a new direct connect somewhere else in the tree, so
this is an AST census, not a one-off assertion: every direct connect call in
``trw_memory/src`` must be either (a) inside ``storage/_connection.py``
itself (the checked helper's own implementation), or (b) named in
``_AUDITED_EXCEPTIONS`` below with a reason. A new, unlisted call site fails
the test by name.

Round-4 audit (2026-09-24): the two round-4 gaps are now fixed and removed
from the exception list. The remaining entries were read and classified, not
fixed, in that session -- each reason stated why, and each was a NEW residual
risk that session's version of this test made visible rather than a claim
that they were safe.

Round-5 audit (2026-09-24): closed all 4 of round-4's residual entries.
``storage/_schema_backup.py``'s LIVE-store connection is now routed through
``storage._connection.connect`` (removed from the exceptions entirely).
``lifecycle/tiers/_warm.py`` and ``storage/_integrity_scheduler.py`` open
with ``uri=True``/``mode=ro`` -- a shape ``connect()``'s current signature
does not accept -- so they call ``connect_registered`` directly, which runs
the same pinned before/after identity check ``connect()`` gets; they remain
in this exception list (still direct AST-visible connect calls) but are no
longer UNCHECKED. ``storage/_schema_backup.py``'s SECOND connect (the backup
SNAPSHOT target, not the live store) is reclassified precisely: it creates a
brand-new file that does not exist yet, so there is no prior identity to
compare against -- the same fresh-creation exemption ``connect()`` itself
applies -- rather than being left as an unexplained "residual."
"""

from __future__ import annotations

import ast
from pathlib import Path

import trw_memory

#: (relative path from trw_memory/src/trw_memory, enclosing function qualname, 1-based ordinal of the
#: call within that function) -> why this direct connect is
#: accounted for rather than routed through storage._connection.connect. Every entry here is a
#: REPORTED residual, not a verified-safe exception -- see the module docstring above.
_AUDITED_EXCEPTIONS: dict[tuple[str, str, int], str] = {
    ("cli_storage.py", "handle_restore", 1): (
        "CLI restore command's rebuild-from-cold branch (trw-memory-storage restore): opens the "
        "operator-named db off the daemon transport, matching the PRD's in-process-SDK non-goal. NOT audited for the "
        "checkout-boundary race (no caller-supplied path; the CLI operator names the path)."
    ),
    ("lifecycle/tiers/_warm.py", "WarmTierStore.discovery_entries", 1): (
        "Round-5: no longer an unchecked gap. A read-only (mode=ro) connection to a warm-tier "
        "sidecar db, opened during the daemon's own background tier-promotion sweep -- reachable "
        "during live daemon operation. Cannot route through storage._connection.connect (its "
        "signature has no uri=True/read-only mode), so it calls connect_registered directly, which "
        "runs the same pinned before/after identity check. Its StorageError on a mismatch degrades "
        "to 'vectors unavailable' (this is a ranking enhancement, not a data path)."
    ),
    ("storage/_integrity_scheduler.py", "IntegrityScheduler._probe", 1): (
        "Round-5: no longer an unchecked gap. A read-only (mode=ro) periodic integrity check that "
        "runs INSIDE the live daemon process on the store's own db_path. Same signature blocker as "
        "_warm.py above -- connect_registered runs the pinned identity check, and its StorageError "
        "is reported as a genuine regression signal (False, 'db identity changed during open') "
        "through the scheduler's own (ok, detail) contract, since surfacing exactly this kind of "
        "anomaly is the scheduler's whole purpose."
    ),
    ("storage/_schema_backup.py", "snapshot_before_migration", 1): (
        "The BACKUP SNAPSHOT TARGET, not the live store -- snapshot_before_migration() creates it "
        "fresh at a timestamped path under a per-store backup directory; the file does not exist "
        "before this call, so there is no prior identity to compare against (the same fresh-creation "
        "exemption storage._connection.connect() itself applies for a brand-new file). The sibling "
        "SOURCE connection in the same function (_open_snapshot_source, opening the LIVE db_path) IS "
        "routed through connect() -- see the docstring there."
    ),
    ("storage/_memory_model_v2.py", "_snapshot_backup", 1): (
        "Backup-API snapshot during the v1->v2 schema migration CLI utility, off the daemon "
        "transport (operator-invoked migration, not a served request)."
    ),
    ("storage/_memory_model_v2.py", "restore_from_backup", 2): (
        "Same v1->v2 migration CLI utility as the _snapshot_backup entry."
    ),
    ("storage/_memory_model_v2.py", "restore_from_backup", 1): (
        "Same v1->v2 migration CLI utility as the _snapshot_backup entry."
    ),
    ("storage/_memory_model_v2.py", "run_memory_model_v2_cutover", 1): (
        "Same v1->v2 migration CLI utility as the _snapshot_backup entry."
    ),
    ("storage/_corrupt_backup.py", "salvage_via_recover_cli", 1): (
        "RESIDUAL, not fixed this session: reachable from the corrupt-recovery branch of "
        "_init_helpers.open_connection_with_recovery, which DOES run during normal SQLiteBackend "
        "construction when quick_check fails -- but it connects to a TEMPORARY recovery db under "
        "tempfile.TemporaryDirectory(), not the live store path, so the identity-swap threat this "
        "PRD defends against (a principal redirecting an ALREADY-VERIFIED live store) does not "
        "apply the same way. Left unfixed pending a decision on whether recovery-scratch files "
        "warrant the same check; flagged rather than silently accepted."
    ),
    ("storage/_snapshot.py", "create_snapshot", 1): (
        "A read-write connection used by the maintenance/export snapshot CLI path (VACUUM INTO), "
        "off the daemon's per-request transport."
    ),
}


def _is_direct_connect(node: ast.AST) -> bool:
    """A ``connect_registered(...)`` call, or ``X.connect(...)`` where X looks like a DB-API module."""
    if not isinstance(node, ast.Call):
        return False
    # C15: a direct connect now opens through the lock registry; it still
    # bypasses connect()'s identity check, so it is counted here all the same.
    if isinstance(node.func, ast.Name) and node.func.id == "connect_registered":
        return True
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "connect"):
        return False
    receiver = node.func.value
    receiver_name = receiver.id if isinstance(receiver, ast.Name) else None
    return receiver_name in {"sqlite3", "dbapi", "pysqlite3"} or bool(
        receiver_name and receiver_name.endswith("_dbapi")
    )


def _direct_connect_call_sites() -> list[tuple[str, str, int]]:
    """Every direct connect call in ``trw_memory/src/trw_memory/**/*.py``, keyed by where it lives.

    AST-based, not a grep: matches ``connect_registered(...)`` and a bare ``.connect(`` call whose
    receiver name suggests a DB-API module or connection factory (``sqlite3``, ``dbapi``,
    ``pysqlite3``, or anything ending ``_dbapi``) -- narrow enough to skip unrelated ``.connect(``
    calls (network clients, signal/slot connections) without needing a type checker.

    Each site is keyed by (path, enclosing function qualname, 1-based ordinal within that function)
    rather than by line number, so an unrelated edit above a call no longer makes the audited
    entry look stale (the rc7 and rc8 gate failures).
    """
    package_root = Path(trw_memory.__file__).parent
    sites: list[tuple[str, str, int]] = []
    for path in sorted(package_root.rglob("*.py")):
        if path.name == "_connection.py" and path.parent.name == "storage":
            continue  # the checked helper's own implementation
        if path.name == "_live_stores.py" and path.parent.name == "trw_memory":
            continue  # connect_registered's own implementation (C15); its callers are counted below
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # trw-fail-silent-allow: no source file under trw_memory/src is expected to be unparseable; a genuinely broken file would fail mypy/ruff long before this census runs, so skipping it here does not hide a connect() call the rest of the pipeline missed
            continue
        sites.extend(_ordered_sites(tree, str(path.relative_to(package_root))))
    return sites


def _ordered_sites(tree: ast.Module, relative: str) -> list[tuple[str, str, int]]:
    """(path, qualname, ordinal) for every direct connect in ``tree``, numbered in source order."""
    located: list[tuple[str, int, int]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_scope = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_scope = child.name if scope == "<module>" else f"{scope}.{child.name}"
            if isinstance(child, ast.Call) and _is_direct_connect(child):
                located.append((scope, child.lineno, child.col_offset))
            visit(child, child_scope)

    visit(tree, "<module>")
    ordinals: dict[str, int] = {}
    ordered: list[tuple[str, str, int]] = []
    for scope, _line, _col in sorted(located, key=lambda item: (item[1], item[2])):
        ordinals[scope] = ordinals.get(scope, 0) + 1
        ordered.append((relative, scope, ordinals[scope]))
    return ordered


def test_every_direct_sqlite_connect_is_accounted_for() -> None:
    """PRD-SEC-016 round-4 finding 3: a new, unaudited direct connect call fails this test by name."""
    found = set(_direct_connect_call_sites())
    known = set(_AUDITED_EXCEPTIONS)

    unaudited = found - known
    assert unaudited == set(), (
        f"new direct sqlite3/dbapi .connect() call(s) not in _AUDITED_EXCEPTIONS: {sorted(unaudited)} -- "
        "route through storage._connection.connect, or add a reasoned entry."
    )


def test_the_two_round_4_fallback_reopens_no_longer_call_dbapi_connect_directly() -> None:
    """Regression trap for the exact bug this finding names: neither fallback module bypasses connect() now."""
    package_root = Path(trw_memory.__file__).parent
    for relative in ("storage/_resilient_fetch.py", "storage/_temporal_fetch.py"):
        source = (package_root / relative).read_text()
        assert "dbapi.connect(" not in source, f"{relative} still calls dbapi.connect() directly"


def test_every_audited_exception_still_exists_at_its_recorded_site() -> None:
    """The allowlist must track the source, not fossilize: a moved/removed call site is stale, not safe."""
    found = set(_direct_connect_call_sites())
    stale = set(_AUDITED_EXCEPTIONS) - found
    assert stale == set(), (
        f"_AUDITED_EXCEPTIONS entries no longer match any call site (moved or removed): {sorted(stale)}"
    )
