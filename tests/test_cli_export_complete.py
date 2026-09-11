"""Actual CLI export completeness across real backend keyset pages."""

import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from trw_memory import cli, cli_storage
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend


@pytest.mark.parametrize("count", [0, 2, 4, 5, 10001])
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
    config = MemoryConfig(storage_path=str(tmp_path))
    monkeypatch.setattr(cli, "MemoryConfig", lambda: config)
    monkeypatch.setattr(cli, "_create_local_backend", lambda config, namespace: backend)
    if count <= 5:
        monkeypatch.setattr(cli_storage, "_EXPORT_PAGE_SIZE", 2, raising=False)
    listing = Mock(wraps=backend.list_entries)
    monkeypatch.setattr(backend, "list_entries", listing)
    close = Mock(wraps=backend.close)
    monkeypatch.setattr(backend, "close", close)
    output = tmp_path / "export.json"
    assert cli.main(["export", "--namespace", "project:export", "--output", str(output)]) == 0
    rows = json.loads(output.read_text())
    assert [row["id"] for row in rows] == [entry.id for entry in reversed(entries)]
    assert all(row["namespace"] == "project:export" and row["detail"] == "retained detail" for row in rows)
    close.assert_called_once()
    assert listing.call_count == count // cli_storage._EXPORT_PAGE_SIZE + 1
    assert listing.call_args_list[0].kwargs["after"] is None
    assert all(call.kwargs["namespace"] == "project:export" for call in listing.call_args_list)


def test_yaml_backend_and_yaml_output_page_completely(tmp_path, monkeypatch):
    from ruamel.yaml import YAML

    backend = YAMLBackend(tmp_path / "entries")
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        backend.store(MemoryEntry(id=f"M-{i}", namespace="default", content=f"fact {i}", updated_at=stamp))
    monkeypatch.setattr(cli_storage, "_EXPORT_PAGE_SIZE", 2, raising=False)
    monkeypatch.setattr(cli, "MemoryConfig", lambda: MemoryConfig(storage_path=str(tmp_path)))
    monkeypatch.setattr(cli, "_create_local_backend", lambda config, namespace: backend)
    output = tmp_path / "export.yaml"
    assert cli.main(["export", "--format", "yaml", "--output", str(output)]) == 0
    assert [row["id"] for row in YAML(typ="safe").load(output)] == [f"M-{i}" for i in reversed(range(5))]


@pytest.mark.parametrize("failure", ["exception", "nonadvancing"])
@pytest.mark.parametrize("file_output", [True, False])
def test_late_page_failure_does_not_emit_partial_success(tmp_path, monkeypatch, capsys, failure, file_output):
    backend = Mock()
    entries = [MemoryEntry(id=f"M-{i}", content=f"fact {i}") for i in range(2)]
    backend.list_entries.side_effect = [entries, OSError("page read failed") if failure == "exception" else entries]
    monkeypatch.setattr(cli_storage, "_EXPORT_PAGE_SIZE", 2, raising=False)
    monkeypatch.setattr(cli, "MemoryConfig", lambda: MemoryConfig(storage_path=str(tmp_path)))
    monkeypatch.setattr(cli, "_create_local_backend", lambda config, namespace: backend)
    output = tmp_path / "existing.json"
    output.write_text("previous export")
    args = ["export"] + (["--output", str(output)] if file_output else [])
    assert cli.main(args) != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Exported" not in captured.err
    assert "Error:" in captured.err
    assert output.read_text() == "previous export"
    backend.close.assert_called_once()
