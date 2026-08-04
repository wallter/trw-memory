from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

_CLI = "trw_memory.cli"


def _real_import_target(tmp_path: Path) -> tuple[Any, Any]:
    """Return a ``(config, backend)`` pair the ``import`` command can really write to.

    ``handle_import`` runs the SEC-001 store gate, which reads real config fields
    and reference entries off the backend. A ``MagicMock`` config/backend cannot
    stand in for that any more, and substituting one would only re-create the
    bypass these tests are meant to cover.
    """
    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    storage_root = tmp_path / "cli_import_store"
    config = MemoryConfig(storage_path=str(storage_root))
    backend = SQLiteBackend(db_path=storage_root / "default" / "memory.db", dim=config.embedding_dim)
    return config, backend


def _reopen_import_target(tmp_path: Path) -> Any:
    """Reopen the store :func:`_real_import_target` wrote to.

    ``handle_import`` closes the backend in its ``finally``, so assertions about
    what landed must run against a fresh handle.
    """
    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    storage_root = tmp_path / "cli_import_store"
    config = MemoryConfig(storage_path=str(storage_root))
    return SQLiteBackend(db_path=storage_root / "default" / "memory.db", dim=config.embedding_dim)


def _make_store_result(
    memory_id: str = "M-abc12345",
    namespace: str = "default",
) -> dict[str, str]:
    return {
        "memory_id": memory_id,
        "namespace": namespace,
        "status": "stored",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def _make_recall_result(
    memory_id: str = "M-abc12345",
    content: str = "test content",
    score: float = 0.85,
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "content": content,
        "detail": "",
        "tags": ["test"],
        "importance": 0.7,
        "score": score,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "namespace": "default",
    }


def _make_forget_result(memory_id: str = "M-abc12345") -> dict[str, str]:
    return {
        "memory_id": memory_id,
        "status": "deleted",
        "namespace": "default",
    }


def _mock_client() -> MagicMock:
    """Create a mock MemoryClient with async method stubs."""
    client = MagicMock()
    client.store = AsyncMock(return_value=_make_store_result())
    client.recall = AsyncMock(return_value=[_make_recall_result()])
    client.search = AsyncMock(return_value=[_make_recall_result()])
    client.forget = AsyncMock(return_value=_make_forget_result())
    client.close = AsyncMock()
    return client


def _mock_entry(
    entry_id: str = "M-001",
    content: str = "test",
    tags: list[str] | None = None,
) -> MagicMock:
    """Create a mock MemoryEntry for export tests."""
    mock = MagicMock()
    mock.id = entry_id
    mock.content = content
    mock.detail = ""
    _tags = tags or ["py"]
    mock.tags = _tags
    mock.importance = 0.5
    mock.status = "active"
    mock.namespace = "default"
    mock.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mock.updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mock.metadata = {}

    _full: dict[str, object] = {
        "id": entry_id,
        "content": content,
        "detail": "",
        "tags": list(_tags),
        "evidence": [],
        "importance": 0.5,
        "status": "active",
        "recurrence": 1,
        "namespace": "default",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "last_accessed_at": None,
        "access_count": 0,
        "q_value": 0.5,
        "q_observations": 0,
        "source": "agent",
        "source_identity": "",
        "merged_from": [],
        "consolidated_from": [],
        "consolidated_into": None,
        "metadata": {},
        "vector_clock": {},
        "remote_id": None,
        "published_to_platform": False,
        "pending_delete": False,
        "cross_validated": False,
        "outcome_history": [],
        "assertions": [],
    }

    def _to_dict(*, fields: set[str] | None = None) -> dict[str, object]:
        if fields is not None:
            return {k: v for k, v in _full.items() if k in fields}
        return dict(_full)

    mock.to_dict = _to_dict
    return mock
