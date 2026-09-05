"""Tests that actor-scoped forget is atomic (TOCTOU closure re-audit #4, #5).

The actor branch previously called ``backend.count(...)`` and
``backend.list_entries(...)`` as two separate, uncovered backend ops. A
concurrent write landing between them yields a wrong count / partial delete.
The fix wraps the count + scan + delete in a single ``backend.transaction()``
(BEGIN IMMEDIATE snapshot).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.forget import memory_forget_impl


class _TxnSpyBackend:
    """Wraps a real SQLiteBackend, recording whether count/list/delete ran
    inside an open transaction() context."""

    def __init__(self, inner: SQLiteBackend) -> None:
        self._inner = inner
        self._txn_depth = 0
        self.count_in_txn: list[bool] = []
        self.list_in_txn: list[bool] = []

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._inner, name)

    def transaction(self):  # type: ignore[no-untyped-def]
        outer = self

        class _Ctx:
            def __enter__(self_ctx):  # type: ignore[no-untyped-def]
                outer._cm = outer._inner.transaction()
                txn = outer._cm.__enter__()
                outer._txn_depth += 1
                return txn

            def __exit__(self_ctx, *exc):  # type: ignore[no-untyped-def]
                outer._txn_depth -= 1
                return outer._cm.__exit__(*exc)

        return _Ctx()

    def count(self, *a, **k):  # type: ignore[no-untyped-def]
        self.count_in_txn.append(self._txn_depth > 0)
        return self._inner.count(*a, **k)

    def list_entries(self, *a, **k):  # type: ignore[no-untyped-def]
        self.list_in_txn.append(self._txn_depth > 0)
        return self._inner.list_entries(*a, **k)


def _entry(entry_id: str, actor: str, namespace: str = "project:a") -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"content {entry_id}",
        namespace=namespace,
        source_identity=actor,
    )


class TestToolForgetActorAtomicity:
    def test_actor_count_and_scan_run_inside_transaction(self, tmp_path: Path) -> None:
        """#5: tool-path actor forget covers count+scan with one transaction."""
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        inner = SQLiteBackend(tmp_path / "mem.db", dim=cfg.embedding_dim)
        try:
            inner.store(_entry("A-1", "alice"))
            inner.store(_entry("A-2", "alice"))
            inner.store(_entry("B-1", "bob"))
            spy = _TxnSpyBackend(inner)

            result = memory_forget_impl(
                None,
                None,
                "project:a",
                backend=spy,  # type: ignore[arg-type]
                config=cfg,
                actor="alice",
            )

            assert result["deleted"] == 2
            # If count() ran at all, it must have been inside the txn.
            assert all(spy.count_in_txn), spy.count_in_txn
            assert spy.list_in_txn and all(spy.list_in_txn), spy.list_in_txn
            # bob's entry survives.
            assert inner.get("B-1", namespace="project:a") is not None
            assert inner.get("A-1", namespace="project:a") is None
        finally:
            inner.close()


class TestClientForgetActorAtomicity:
    async def test_client_actor_count_and_scan_run_inside_transaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#4: client-path actor forget covers count+scan with one transaction."""
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="project:a", mode="local")
        await client.store("alice one", source_identity="alice")
        await client.store("alice two", source_identity="alice")
        await client.store("bob one", source_identity="bob")

        real_backend = client._get_backend()
        spy = _TxnSpyBackend(real_backend)
        client._backend = spy  # type: ignore[assignment]

        result = await client.forget(actor="alice")

        assert result["entries_deleted"] == 2
        assert all(spy.count_in_txn), spy.count_in_txn
        assert spy.list_in_txn and all(spy.list_in_txn), spy.list_in_txn
