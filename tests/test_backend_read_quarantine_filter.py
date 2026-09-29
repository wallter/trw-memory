"""PRD-CORE-333 FR02: no StorageBackend read method returns an identity the quarantine ledger blocks.

The conformance census enumerates, from the ABC AND from each concrete backend class
(mixins and backend-only readers included), every public method whose return annotation
names ``MemoryEntry`` -- not a hand-kept list. Each enumerated method must have a probe
below, and each probe must return the visible control row while returning neither
planted row. A new entry-returning method therefore fails the census until it is both
filtered and probed; ``test_guard_catches_a_planted_unfiltered_reader`` proves that.

Rows are planted beneath the backend -- a raw ``INSERT`` into the SQLite file, a raw
YAML file write -- so the proof does not depend on any write path's cooperation. One is
blocked by its namespace-qualified id, one only by its content hash.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import inspect
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from tests._layout import MONOREPO_ROOT, PACKAGE_ROOT
from trw_memory.exceptions import AuthorizationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import Anchor, Assertion, AssertionType, MemoryEntry
from trw_memory.security.quarantine_ledger import LedgerIdentity, QuarantineLedger, ledger_for_config
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend
from trw_memory.tools.entry import memory_find_duplicate_impl, memory_get_impl
from trw_memory.tools.search import memory_search_impl

NS = "default"
CONTROL = "V-visible"
BY_ID = "Q-byid"
BY_HASH = "Q-byhash"
POISON = "zebra poisoned payload"
ANCHOR_FILE = "src/app.py"


def _entry(entry_id: str, content: str) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        namespace=NS,
        importance=0.7,
        assertions=[Assertion(type=AssertionType.GREP_PRESENT, pattern="zebra", target="*.py")],
        anchors=[Anchor(file=ANCHOR_FILE, symbol_name="handler")],
    )


def _ledger_the_planted_rows(config: MemoryConfig) -> None:
    ledger = ledger_for_config(config)
    # BY_ID answers only to its (namespace, id) key: no hash is recorded for it.
    ledger.append(LedgerIdentity(namespace=NS, entry_id=BY_ID), "quarantined", actor="system")
    # BY_HASH is never named: an unrelated id carrying the same content is what was quarantined.
    ledger.append(LedgerIdentity.of(_entry("Q-original", POISON)), "rejected", actor="reviewer")


def _plant_sqlite(db_path: Path) -> None:
    """Copy the control row's every table row under the planted ids with raw SQL."""
    conn = sqlite3.connect(db_path)
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(memories)")]
        for planted, content in ((BY_ID, "zebra quarantined by id"), (BY_HASH, POISON)):
            select = ", ".join("?" if c in {"id", "content"} else c for c in cols)
            params = [planted if c == "id" else content for c in cols if c in {"id", "content"}]
            conn.execute(
                f"INSERT INTO memories ({', '.join(cols)}) SELECT {select} FROM memories WHERE id = ?",
                (*params, CONTROL),
            )
            conn.execute(
                "INSERT INTO memories_fts (id, namespace, content, detail, tags) VALUES (?, ?, ?, '', '')",
                (planted, NS, content),
            )
            conn.execute(
                "INSERT INTO anchor_postings (namespace, file, entry_id) VALUES (?, ?, ?)", (NS, ANCHOR_FILE, planted)
            )
        conn.commit()
    finally:
        conn.close()


def _plant_yaml(entries_dir: Path) -> None:
    source = (entries_dir / f"{CONTROL}.yaml").read_text(encoding="utf-8")
    for planted, content in ((BY_ID, "zebra quarantined by id"), (BY_HASH, POISON)):
        text = source.replace(CONTROL, planted).replace("zebra visible control", content)
        (entries_dir / f"{planted}.yaml").write_text(text, encoding="utf-8")


@dataclass
class Planted:
    backend: StorageBackend
    config: MemoryConfig
    raw: list[MemoryEntry]


@contextlib.contextmanager
def _open(kind: str, tmp_path: Path) -> Iterator[Planted]:
    config = MemoryConfig(storage_path=str(tmp_path / "memory"), storage_backend=kind)
    with create_backend_from_config(config, NS) as backend:
        backend.store(_entry(CONTROL, "zebra visible control"))
    if kind == "sqlite":
        path = Path(backend.db_path)  # type: ignore[attr-defined]
        _plant_sqlite(path)
        raw_reader: StorageBackend = SQLiteBackend(path)
    else:
        entries_dir = backend._dir  # type: ignore[attr-defined]
        _plant_yaml(entries_dir)
        raw_reader = YAMLBackend(entries_dir)
    with raw_reader:  # no ledger: proves the rows really are in the file
        raw = raw_reader.list_entries(namespace=NS, limit=10)
    assert {e.id for e in raw} == {CONTROL, BY_ID, BY_HASH}
    _ledger_the_planted_rows(config)
    with create_backend_from_config(config, NS) as filtered:
        yield Planted(filtered, config, raw)


@pytest.fixture(params=["sqlite", "yaml"])
def planted(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Planted]:
    with _open(request.param, tmp_path) as planted:
        yield planted


# -- the census ------------------------------------------------------------------------


def entry_returning_methods(cls: type) -> set[str]:
    """Every public method of *cls* (ABC, mixins and all) whose return annotation names MemoryEntry."""
    found: set[str] = set()
    for name in dir(cls):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(cls, name)
        func = getattr(member, "__func__", member)
        annotation = getattr(func, "__annotations__", {}).get("return")
        if callable(func) and annotation is not None and "MemoryEntry" in str(annotation):
            found.add(name)
    return found


Probe = Callable[[StorageBackend, Planted], object]
_IDS = (CONTROL, BY_ID, BY_HASH)

PROBES: dict[str, Probe] = {
    "get": lambda b, p: [b.get(i, namespace=NS) for i in _IDS],
    # PRD-CORE-318 FR02's batch read: selected by its annotation the moment it landed on the ABC.
    "get_many": lambda b, p: list(b.get_many(list(_IDS), namespace=NS).values()),
    "update": lambda b, p: [b.update(i, namespace=NS, importance=0.9) for i in _IDS],
    "search": lambda b, p: b.search("zebra", top_k=10, namespace=NS),
    "search_fts": lambda b, p: b.search_fts("zebra", top_k=10, namespace=NS),
    "list_entries": lambda b, p: b.list_entries(namespace=NS, limit=10),
    "list_entries_by_id": lambda b, p: b.list_entries_by_id(namespace=NS, limit=10),
    "entries_with_assertions": lambda b, p: [
        *b.entries_with_assertions(namespace=NS, limit=10),  # type: ignore[attr-defined]
        *b.entries_with_assertions(namespace=NS, limit=1, include_anchors=True),  # type: ignore[attr-defined]
    ],
    "anchored_to": lambda b, p: b.anchored_to(NS, ANCHOR_FILE, status=None, limit=10),
    "entries_changed_since": lambda b, p: _changed_since_the_beginning(b),
    "filter_quarantined": lambda b, p: b.filter_quarantined(list(p.raw)),
}
#: Id-returning readers the fifth review named (OQ-003): not selected by annotation, probed anyway.
ID_PROBES: dict[str, Probe] = {
    "find_active_by_content": lambda b, p: [
        b.find_active_by_content(c, "", namespace=NS)
        for c in ("zebra visible control", "zebra quarantined by id", POISON)
    ],
}


def _changed_since_the_beginning(b: StorageBackend) -> list[MemoryEntry] | None:
    token = b.namespace_change_token(NS)
    if token is None:  # no change feed on this backend: it returns nothing
        return None
    return b.entries_changed_since(NS, dataclasses.replace(token, insert_seq=0, top_updated_at=""), limit=100)


def _returned(value: object) -> list[object]:
    if value is None or isinstance(value, (MemoryEntry, str)):
        return [value]
    return [item for sub in value for item in _returned(sub)] if isinstance(value, list) else [value]


#: ABC defaults that stand for "this backend lacks the capability" (they return a constant).
_CAPABILITY_STUBS = {"search_fts", "entries_changed_since", "list_entries_by_id"}


def violations(backend: StorageBackend, planted: Planted, probes: dict[str, Probe]) -> list[str]:
    """What the conformance census finds wrong with *backend*: unprobed readers, leaks, vacuous probes."""
    enumerated = entry_returning_methods(type(backend))
    found = [f"{name}: returns MemoryEntry but has no conformance probe" for name in sorted(enumerated - probes.keys())]
    for name in sorted((enumerated & probes.keys()) | ID_PROBES.keys()):
        if name in _CAPABILITY_STUBS and getattr(type(backend), name) is getattr(StorageBackend, name):
            continue  # inherited no-capability stub: returns a constant, never a stored row
        probe = probes.get(name) or ID_PROBES[name]
        result = probe(backend, planted)
        values = _returned(result)
        ids = {v.id if isinstance(v, MemoryEntry) else str(v) for v in values}
        if leaked := ids & {BY_ID, BY_HASH}:
            found.append(f"{name}: returned ledgered identity {sorted(leaked)}")
        if result is not None and CONTROL not in ids:
            found.append(f"{name}: probe never returned the control row, so it proves nothing")
    return found


def test_all_read_methods_filtered(planted: Planted) -> None:
    assert violations(planted.backend, planted, PROBES) == []


def test_census_enumerates_concrete_defaults_and_backend_only_readers() -> None:
    """The ABC's concrete defaults and SQLite-only readers are in scope, not just the abstract three."""
    assert {"get", "get_many", "update", "search", "list_entries"} <= entry_returning_methods(StorageBackend)
    assert {"entries_changed_since", "anchored_to", "list_entries_by_id", "search_fts"} <= entry_returning_methods(
        StorageBackend
    )
    assert "entries_with_assertions" in entry_returning_methods(SQLiteBackend)


def test_guard_catches_a_planted_unfiltered_reader(tmp_path: Path) -> None:
    """Guard the guard: a new entry-returning method without the filter turns the census red."""

    class _Leaky(SQLiteBackend):
        def dump_raw(self) -> list[MemoryEntry]:
            with SQLiteBackend(self.db_path) as raw:  # forgets the filter
                return raw.list_entries(namespace=NS, limit=100)

    with _open("sqlite", tmp_path) as planted:
        ledger = planted.backend._quarantine_ledger
        with _Leaky(Path(planted.backend.db_path), quarantine_ledger=ledger) as leaky:  # type: ignore[attr-defined]
            unprobed = violations(leaky, planted, PROBES)
            leaking = violations(leaky, planted, {**PROBES, "dump_raw": lambda b, p: b.dump_raw()})  # type: ignore[attr-defined]
    assert unprobed == ["dump_raw: returns MemoryEntry but has no conformance probe"]
    assert leaking == ["dump_raw: returned ledgered identity ['Q-byhash', 'Q-byid']"]


def test_an_empty_ledger_filters_nothing_and_opens_no_file(tmp_path: Path) -> None:
    config = MemoryConfig(storage_path=str(tmp_path / "memory"))
    with create_backend_from_config(config, NS) as backend:
        backend.store(_entry(CONTROL, "zebra visible control"))
        ledger = backend._quarantine_ledger
        assert isinstance(ledger, QuarantineLedger) and not ledger.path.exists()
        marker = object()
        assert backend._quarantine_entry_filter(marker) is marker  # type: ignore[arg-type]
        entries = backend.list_entries(namespace=NS)
        assert backend.filter_quarantined(entries) is entries
        assert not ledger.path.exists()


def test_a_ledger_change_is_seen_by_an_open_backend(tmp_path: Path) -> None:
    """The index is cached per ledger change: an append by another handle invalidates it."""
    config = MemoryConfig(storage_path=str(tmp_path / "memory"))
    with create_backend_from_config(config, NS) as backend:
        backend.store(_entry(CONTROL, "zebra visible control"))
        assert backend.get(CONTROL, namespace=NS) is not None
        QuarantineLedger(ledger_for_config(config).path).append(
            LedgerIdentity(namespace=NS, entry_id=CONTROL), "quarantined", actor="system"
        )
        assert backend.get(CONTROL, namespace=NS) is None
        QuarantineLedger(ledger_for_config(config).path).append(
            LedgerIdentity(namespace=NS, entry_id=CONTROL), "approved", actor="op"
        )
        assert backend.get(CONTROL, namespace=NS) is not None


@pytest.mark.parametrize("kind", ["sqlite", "yaml"])
def test_a_full_page_that_loses_a_row_is_refilled_not_short(kind: str, tmp_path: Path) -> None:
    """Blocked rows are dropped BEFORE the limit: a page never comes back short while rows remain."""
    config = MemoryConfig(storage_path=str(tmp_path / "memory"), storage_backend=kind)
    rows = [_entry(f"R-{i}", f"zebra row {i}") for i in range(6)]
    with create_backend_from_config(config, NS) as backend:
        for row in rows:
            backend.store(row)
        newest = [e.id for e in backend.list_entries(namespace=NS, limit=6)]
        ledger_for_config(config).append(LedgerIdentity(namespace=NS, entry_id=newest[0]), "quarantined", actor="s")
        assert [e.id for e in backend.list_entries(namespace=NS, limit=2)] == newest[1:3]
        searched = [e.id for e in backend.search("zebra", top_k=5, namespace=NS)]
        assert len(searched) == 5 and newest[0] not in searched
        if kind == "sqlite":
            assert [e.id for e in backend.search_fts("zebra", top_k=5, namespace=NS)] and newest[0] not in {
                e.id for e in backend.search_fts("zebra", top_k=5, namespace=NS)
            }
            by_id = sorted(r.id for r in rows)
            ledger_for_config(config).append(LedgerIdentity(namespace=NS, entry_id=by_id[1]), "rejected", actor="s")
            expected = [i for i in by_id if i not in {by_id[1], newest[0]}][:2]
            assert [e.id for e in backend.list_entries_by_id(namespace=NS, limit=2)] == expected
            sweep = backend.entries_with_assertions(namespace=NS, limit=2, include_anchors=True)  # type: ignore[attr-defined]
            assert [e.id for e in sweep] == expected


@pytest.mark.parametrize("limit", [1, 2])
def test_an_assertion_page_without_anchors_is_refilled_when_its_first_row_is_hidden(tmp_path: Path, limit: int) -> None:
    """Review r1 P2: without ``include_anchors`` the page was filtered after its LIMIT (limit=1 came back empty)."""
    config = MemoryConfig(storage_path=str(tmp_path / "memory"))
    with create_backend_from_config(config, NS) as backend:
        for i in range(4):
            backend.store(_entry(f"A-{i}", f"zebra assertion row {i}"))
        order = [e.id for e in backend.entries_with_assertions(namespace=NS, limit=4)]  # type: ignore[attr-defined]
        ledger_for_config(config).append(LedgerIdentity(namespace=NS, entry_id=order[0]), "quarantined", actor="s")
        page = backend.entries_with_assertions(namespace=NS, limit=limit)  # type: ignore[attr-defined]
    assert [e.id for e in page] == order[1 : 1 + limit]


def test_the_identity_probe_sees_a_hidden_row_and_returns_no_content(planted: Planted) -> None:
    """Review r1 P0-1: ``holds_id`` answers for a row ``get`` hides, as a bool: a write can refuse, nothing leaks."""
    for hidden in (BY_ID, BY_HASH):
        assert planted.backend.get(hidden, namespace=NS) is None
        assert planted.backend.holds_id(hidden, namespace=NS) is True
    assert planted.backend.holds_id(CONTROL, namespace=NS) is True
    assert planted.backend.holds_id("L-never-stored", namespace=NS) is False
    assert planted.backend.holds_id(BY_ID, namespace="another") is False


def test_the_default_identity_probe_refuses_on_a_filtering_backend(planted: Planted) -> None:
    """A backend that filters reads but kept the base probe would answer through ``get``: it refuses instead."""
    with pytest.raises(NotImplementedError, match="no unfiltered identity probe"):
        StorageBackend.holds_id(planted.backend, BY_ID, namespace=NS)


def test_forget_still_removes_a_hidden_row(planted: Planted) -> None:
    assert planted.backend.delete(BY_ID, namespace=NS) is True


def test_vector_search_hydrates_nothing_ledgered(tmp_path: Path) -> None:
    """``search_vectors`` returns ids, not content; every hydration of a ledgered id reads as absent."""
    with _open("sqlite", tmp_path) as planted:
        _assert_vector_hydration_filtered(planted.backend)


def _assert_vector_hydration_filtered(backend: StorageBackend) -> None:
    if not backend.supports_vectors():
        pytest.skip("sqlite-vec unavailable in this interpreter")
    vector = [1.0] + [0.0] * 383
    with SQLiteBackend(Path(backend.db_path)) as raw:  # type: ignore[attr-defined]
        for entry_id in _IDS:
            raw.upsert_vector(entry_id, vector, namespace=NS)
    hits = [entry_id for entry_id, _ in backend.search_vectors(vector, top_k=10, namespace=NS)]
    assert set(hits) == set(_IDS)
    hydrated = {entry_id: backend.get(entry_id, namespace=NS) for entry_id in hits}
    assert {k for k, v in hydrated.items() if v is not None} == {CONTROL}


def test_r8_scope_cross_project_read_is_refused_with_the_filter_active(tmp_path: Path) -> None:
    """T1(7): a token granted one project cannot read another's rows; the ledger filter changes none of it.

    The RBAC refusal itself is ``tests/test_daemon_namespace_tokens.py::
    test_a_token_cannot_reach_a_namespace_outside_its_grant`` (end to end over the daemon)
    and ``tests/test_daemon_grant_read_paths.py``; this asserts it still holds in-process
    with a non-empty ledger in front of the reads.
    """
    alpha, beta = "project:alpha-11111111", "project:beta-22222222"
    config = MemoryConfig(storage_path=str(tmp_path / "memory"), memory_single_store_path=str(tmp_path / "m.db"))
    with create_backend_from_config(config, alpha) as backend:
        for namespace in (alpha, beta):
            backend.store(MemoryEntry(id=f"M-{namespace[8:12]}", content="zebra row", namespace=namespace))
    ledger_for_config(config).append(LedgerIdentity(namespace=alpha, entry_id="unrelated"), "quarantined", actor="s")
    reset = auth_context_var.set(AuthenticatedUser(AccessToken(token="t", client_id="c", scopes=[f"ns:{alpha}"])))
    try:
        with create_backend_from_config(config, alpha) as backend:
            assert backend._quarantine_ledger is not None and backend._quarantine_ledger.index()
            assert memory_get_impl("M-alph", alpha, backend=backend, config=config)["status"] == "ok"
            reads: tuple[Callable[[], object], ...] = (
                lambda: memory_get_impl("M-beta", beta, backend=backend, config=config),
                lambda: memory_search_impl(beta, backend=backend, config=config),
                lambda: memory_find_duplicate_impl(beta, "zebra row", "", backend=backend, config=config),
            )
            for read in reads:
                with pytest.raises(AuthorizationError, match=beta):
                    read()
    finally:
        auth_context_var.reset(reset)


def test_mcp_readers_never_serve_a_ledgered_row(planted: Planted) -> None:
    """The trw-memory server's own tools read through the filtered backend -- no tool list needed."""
    b, cfg = planted.backend, planted.config
    assert memory_get_impl(BY_ID, NS, backend=b, config=cfg) == {"status": "not_found"}
    assert memory_find_duplicate_impl(NS, POISON, "", backend=b, config=cfg)["entry_id"] is None
    entries = memory_search_impl(NS, backend=b, config=cfg)["entries"]
    assert isinstance(entries, list)
    served = {e["id"] for e in entries}
    assert served == {CONTROL}


# -- every production construction is filtered or a documented raw handle ------------

#: (file, enclosing function) -> why this construction reads raw rows on purpose.
RAW_CONSTRUCTIONS = {
    ("trw-memory/src/trw_memory/security/_runtime_quarantine.py", "open_quarantine_backend"): (
        "the quarantine store itself: filtering it would empty the review queue"
    ),
    ("trw-memory/src/trw_memory/lifecycle/tiers/_warm.py", "_get_warm_backend"): (
        "warm-tier VECTOR store only; warm entries are sidecars resolved through the canonical backend's get"
    ),
    ("trw-memory/src/trw_memory/tools/_checkout_merge.py", "plan_import"): (
        "untrusted import source; admission is FR04, and imported rows are read back through a filtered store"
    ),
    ("trw-memory/src/trw_memory/tools/_checkout_merge.py", "write_import"): ("same source as plan_import"),
    ("trw-mcp/src/trw_mcp/state/_store_migration.py", "_source_rows"): (
        "reads a private snapshot of the project store to migrate it; the rows land in the user store, whose reads "
        "are filtered (cross-ledger migration is a named residual, not covered here)"
    ),
    ("trw-mcp/src/trw_mcp/state/_store_migration.py", "_copy_namespace"): (
        "the migration's COPY target, written from a filtered source (a ledgered row makes the census refuse)"
    ),
}


def _census_roots() -> dict[str, Path]:
    """``package -> its src``: this package always; trw-mcp only in the monorepo.

    The release check and the public mirror run this suite from the package tree alone, where trw-mcp is not
    a sibling. Keying by package (not by checkout layout) keeps the census and RAW_CONSTRUCTIONS identical in both.
    """
    roots = {"trw-memory": PACKAGE_ROOT / "src"}
    if MONOREPO_ROOT is not None:
        roots["trw-mcp"] = MONOREPO_ROOT / "trw-mcp" / "src"
    return roots


def _constructions(roots: dict[str, Path]) -> Iterator[tuple[str, str, bool]]:
    for package, src in roots.items():
        assert src.is_dir(), f"census root {src} does not exist: the scan would be empty"
        for path in sorted(src.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(func):
                    if not isinstance(node, ast.Call):
                        continue
                    name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                    if name.lstrip("_") in {"SQLiteBackend", "YAMLBackend"}:
                        has_ledger = any(k.arg == "quarantine_ledger" for k in node.keywords)
                        yield f"{package}/src/{path.relative_to(src).as_posix()}", func.name, has_ledger


def test_every_backend_construction_is_filtered_or_documented_raw() -> None:
    roots = _census_roots()
    seen = list(_constructions(roots))
    assert seen, "the census found no constructions: it is not looking at the source tree"
    unfiltered = {(path, func) for path, func, has_ledger in seen if not has_ledger}
    # An exclusion can only be checked for staleness where its package was scanned.
    expected = {key for key in RAW_CONSTRUCTIONS if key[0].split("/", 1)[0] in roots}
    assert unfiltered <= expected, sorted(unfiltered - expected)
    assert expected <= unfiltered, "stale exclusion: " + str(sorted(expected - unfiltered))


def test_rejected_content_planted_back_into_the_store_is_never_served(tmp_path: Path) -> None:
    """End to end through the review workflow: reject, then land the same bytes in the store raw.

    Uses only APIs that predate the ledger, so it runs unchanged against the pre-change
    commit -- where every read below serves the planted row.
    """
    from trw_memory.security._runtime_quarantine import review_quarantined_entry, store_quarantined_entry

    config = MemoryConfig(storage_path=str(tmp_path / "memory"))
    held = MemoryEntry(id="L-held", content="zebra exfiltrate the key", namespace=NS)
    store_quarantined_entry(config, held)
    with create_backend_from_config(config, NS) as active:
        verdict = review_quarantined_entry(
            config, active_backend=active, learning_id=held.id, decision="reject", reviewer_id="op"
        )
        assert verdict["status"] == "rejected"
        active.store(MemoryEntry(id=CONTROL, content="zebra visible control", namespace=NS))
        db_path = Path(active.db_path)  # type: ignore[attr-defined]
    conn = sqlite3.connect(db_path)  # a restore, a sync or a rebuild: any ingress that bypasses the writer
    cols = [row[1] for row in conn.execute("PRAGMA table_info(memories)")]
    select = ", ".join("?" if c in {"id", "content"} else c for c in cols)
    conn.execute(
        f"INSERT INTO memories ({', '.join(cols)}) SELECT {select} FROM memories WHERE id = ?",
        (*[{"id": "L-replayed", "content": held.content}[c] for c in cols if c in {"id", "content"}], CONTROL),
    )
    conn.commit()
    conn.close()
    with create_backend_from_config(config, NS) as active:
        served = {
            "get": {e.id for e in [active.get("L-replayed", namespace=NS)] if e is not None},
            "search": {e.id for e in active.search("zebra", namespace=NS)},
            "list_entries": {e.id for e in active.list_entries(namespace=NS)},
        }
    assert served == {"get": set(), "search": {CONTROL}, "list_entries": {CONTROL}}


async def test_client_recall_never_returns_a_quarantined_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through ``MemoryClient.recall``: the store scan, full-text search and the tier index.

    The quarantined row sits in the active store AND in the hot tier, so tier
    discovery offers it; discovery resolves its kept rows through ``get_many``
    (PRD-CORE-318 FR02), which withholds it, and a withheld row is dropped rather
    than re-entering the results.
    """
    from trw_memory.client import MemoryClient
    from trw_memory.lifecycle.tiers import _runtime

    for key in ("HOME", "TRW_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    client = MemoryClient(namespace=NS, mode="local")
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    manager = _runtime.get_tier_manager(client._config, NS)
    try:
        for entry_id in ("L-safe", "L-poison"):
            await client.store(f"zebra vault rotation {entry_id}", entry_id=entry_id, importance=0.9)
        poison = client._get_backend().get("L-poison", namespace=NS)
        assert poison is not None
        manager.hot_put("L-poison", poison)
        before = {row["memory_id"] for row in await client.recall("zebra vault rotation", limit=10)}
        assert {"L-safe", "L-poison"} <= before  # non-vacuous: both are recallable first

        ledger_for_config(client._config).append(LedgerIdentity.of(poison), "quarantined", actor="test")
        after = {row["memory_id"] for row in await client.recall("zebra vault rotation", limit=10)}
        assert "L-safe" in after
        assert "L-poison" not in after
    finally:
        manager.close()
        client._get_backend().close()
