"""PRD-CORE-245 FR08 — every production writer stamps causality.

The consumer is live. ``resolve_conflict`` runs on every org-shared pull, and the
REMOTE side of that comparison always has a clock (coerced from the peer
payload). With an empty local clock ``compare_clocks`` returns ``"b_wins"`` and
the resolver returns the remote entry outright: **a local edit that strictly
postdates the remote one is discarded, not merged, with no error.** Measured on
the reference store 2026-09-03, 9,311 of 9,366 rows (99.4%) had the empty clock
that produces exactly that.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.entry_factory import local_node_id_for, new_entry, revise_entry
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync.conflict import resolve_conflict
from trw_memory.tools.store import memory_store_impl

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]

#: Files allowed to build a ``MemoryEntry`` directly: the helper itself, the row
#: mappers that rebuild entries FROM storage (they are deserialisers, not
#: writers, and a deserialiser must reproduce the stored clock verbatim), and
#: the security canary seeder whose rows are fixtures with pinned hashes.
_CONSTRUCTION_ALLOWLIST = {
    "trw-memory/src/trw_memory/models/entry_factory.py",
    "trw-memory/src/trw_memory/storage/_row_mapper.py",
    "trw-memory/src/trw_memory/storage/_yaml_row_mapper.py",
    "trw-memory/src/trw_memory/security/_runtime_canary.py",
    "trw-memory/src/trw_memory/sync/_remote_admission.py",
    # Deserialisers of a REMOTE or LEGACY payload. Both must reproduce the clock
    # the payload carries verbatim: stamping a local one over a peer's clock is
    # precisely the causality corruption FR08 exists to prevent, only inverted.
    "trw-memory/src/trw_memory/migration/from_trw.py",
    "trw-mcp/src/trw_mcp/sync/pull.py",
}


async def test_every_writer_populates_the_vector_clock(tmp_path: Path) -> None:
    """All three production writers yield a non-empty clock on round trip."""
    from trw_memory.client import MemoryClient

    cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path), embeddings_enabled=False)

    # Writer 1 — MemoryClient.store (the SDK path).
    client = MemoryClient(namespace="project:default", db_path=tmp_path / "client.db")
    try:
        stored = await client.store("clock through the client", importance=0.5)
        entry = client._get_backend().get(str(stored["memory_id"]), namespace="project:default")
        assert entry is not None
        assert entry.vector_clock, "MemoryClient.store must stamp a vector clock"
    finally:
        await client.close()

    # Writer 2 — trw_memory.tools.store (what trw-memory-server writes through).
    backend = SQLiteBackend(tmp_path / "tool.db")
    try:
        result = memory_store_impl(
            "clock through the tool surface",
            "project:default",
            backend=backend,
            config=cfg,
            entry_id="M-tool-clock",
        )
        assert result["status"] == "stored"
        tool_entry = backend.get("M-tool-clock", namespace="project:default")
        assert tool_entry is not None
        assert tool_entry.vector_clock, "the tool store surface must stamp a vector clock"
    finally:
        backend.close()

    # Writer 3 — trw-mcp's learning_to_entry (the flagship consumer's write path).
    transforms = pytest.importorskip("trw_mcp.state._memory_transforms")
    mcp_entry = transforms._learning_to_memory_entry("L-mcp-clock", "clock through trw-mcp", "detail")
    assert mcp_entry.vector_clock, "the trw-mcp write path must stamp a vector clock"


def test_a_newer_local_edit_survives_a_stale_remote_one() -> None:
    """The production shape: empty-local vs populated-remote used to discard the local row."""
    node = local_node_id_for("/tmp/whatever")
    base = new_entry(
        entry_id="M-conflict",
        content="local content",
        namespace="project:default",
        local_node_id=node,
    )
    # The local row is edited again AFTER the remote snapshot was taken.
    local = revise_entry(base, local_node_id=node, fields={"content": "newer local content"})
    remote = base.model_copy(
        update={
            "content": "stale remote content",
            "updated_at": base.updated_at - timedelta(hours=1),
        }
    )

    assert resolve_conflict(local, remote) is local


def test_an_unstamped_local_row_is_what_used_to_lose() -> None:
    """Regression witness: this is the exact shape 99.4% of live rows had."""
    unstamped = MemoryEntry(id="M-old", content="newer local content", namespace="project:default")
    remote = MemoryEntry(
        id="M-old",
        content="stale remote content",
        namespace="project:default",
        vector_clock={"peer": 1},
        updated_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    # Documents the defect rather than asserting it is acceptable: with no local
    # clock there is nothing to order by, so the remote wins on causality alone.
    # FR08 removes the shape by making every writer stamp one.
    assert resolve_conflict(unstamped, remote) is remote


def test_no_bare_constructor_outside_the_helper() -> None:
    """FR08: a grep-absent assertion over both source trees, by AST not by regex."""
    offenders: list[str] = []
    for tree in ("trw-memory/src/trw_memory", "trw-mcp/src/trw_mcp"):
        root = _REPO / tree
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            relative = path.relative_to(_REPO).as_posix()
            if relative in _CONSTRUCTION_ALLOWLIST:
                continue
            module = ast.parse(path.read_text())
            offenders.extend(
                f"{relative}:{node.lineno}"
                for node in ast.walk(module)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MemoryEntry"
            )
    assert offenders == [], (
        "production code must build entries through trw_memory.models.entry_factory, "
        f"which is the one place that stamps namespace + vector_clock: {offenders}"
    )
