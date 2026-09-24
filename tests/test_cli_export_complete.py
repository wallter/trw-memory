"""``trw-memory export`` over the daemon: complete across real keyset pages, all or nothing.

The daemon side is the real ``memory_list_page_impl`` over a real backend, so
the pages, their order and the resume cursor are the ones a live daemon serves.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from trw_memory import cli, cli_client
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend
from trw_memory.tools.listing import memory_list_page_impl


def _daemon_over(backend: StorageBackend, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Serve ``list_page`` from *backend*; returns the ``(namespace, after)`` of every call."""
    calls: list[object] = []

    async def list_page(namespace: str, limit: int, after: dict[str, str] | None) -> dict[str, object]:
        calls.append((namespace, after))
        return memory_list_page_impl(namespace, limit, after, backend=backend, config=config)

    client = MagicMock()
    client.list_page = list_page
    monkeypatch.setattr(cli_client, "daemon_client", lambda: client)
    return calls


@pytest.mark.parametrize("count", [0, 2, 4, 5, 1001])
def test_sqlite_cli_export_all_rows_ties_and_namespace(tmp_path, monkeypatch, count):
    backend = SQLiteBackend(tmp_path / "memory.db")
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    entries = [
        MemoryEntry(
            id=f"M-{i:05d}",
            namespace="project:export",
            content=f"fact {i}",
            detail="retained detail",
            updated_at=stamp,
            created_at=stamp,
        )
        for i in range(count)
    ]
    backend.store_many([*entries, MemoryEntry(id="M-foreign", namespace="project:other", content="foreign")])
    if count <= 5:
        monkeypatch.setattr(cli_client, "_EXPORT_PAGE_SIZE", 2)
    calls = _daemon_over(backend, MemoryConfig(storage_path=str(tmp_path)), monkeypatch)
    output = tmp_path / "export.json"

    assert cli.main(["export", "--namespace", "project:export", "--output", str(output)]) == 0

    rows = json.loads(output.read_text())
    assert [row["id"] for row in rows] == [entry.id for entry in reversed(entries)]
    assert all(row["namespace"] == "project:export" and row["detail"] == "retained detail" for row in rows)
    assert len(calls) == count // cli_client._EXPORT_PAGE_SIZE + 1
    assert calls[0] == ("project:export", None)
    assert all(namespace == "project:export" for namespace, _after in calls)  # type: ignore[misc]
    backend.close()


def test_yaml_backend_and_yaml_output_page_completely(tmp_path, monkeypatch):
    from ruamel.yaml import YAML

    backend = YAMLBackend(tmp_path / "entries")
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        backend.store(MemoryEntry(id=f"M-{i}", namespace="default", content=f"fact {i}", updated_at=stamp))
    monkeypatch.setattr(cli_client, "_EXPORT_PAGE_SIZE", 2)
    _daemon_over(backend, MemoryConfig(storage_path=str(tmp_path)), monkeypatch)
    output = tmp_path / "export.yaml"

    assert cli.main(["export", "--namespace", "default", "--format", "yaml", "--output", str(output)]) == 0
    assert [row["id"] for row in YAML(typ="safe").load(output)] == [f"M-{i}" for i in reversed(range(5))]


@pytest.mark.parametrize("failure", ["exception", "nonadvancing"])
@pytest.mark.parametrize("file_output", [True, False])
def test_late_page_failure_does_not_emit_partial_success(tmp_path, monkeypatch, capsys, failure, file_output):
    row = MemoryEntry(id="M-1", content="fact").model_dump(mode="json")
    first = {"status": "ok", "entries": [row], "next": {"updated_at": "2026-01-01", "entry_id": "M-1"}}
    client = MagicMock()
    client.list_page = AsyncMock(side_effect=[first, OSError("page read failed") if failure == "exception" else first])
    monkeypatch.setattr(cli_client, "daemon_client", lambda: client)
    output = tmp_path / "existing.json"
    output.write_text("previous export")
    args = ["export", "--namespace", "default"] + (["--output", str(output)] if file_output else [])

    assert cli.main(args) != 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Exported" not in captured.err
    assert "Error:" in captured.err
    assert output.read_text() == "previous export"
