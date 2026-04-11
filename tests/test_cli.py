"""Tests for trw_memory.cli."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trw_memory.cli import main
from trw_memory.cli_parser import build_parser

# Patch targets — module-level imports in trw_memory.cli
_CLI = "trw_memory.cli"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    # Support MemoryEntry.to_dict() interface used by entry_to_export_dict.
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


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_no_command_returns_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_store_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["store", "--summary", "test"])
        assert args.command == "store"
        assert args.summary == "test"
        assert args.importance == 0.5
        assert args.namespace == "default"

    def test_store_all_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "store",
                "--summary",
                "test",
                "--detail",
                "details",
                "--tags",
                "py",
                "--tags",
                "test",
                "--importance",
                "0.9",
                "--namespace",
                "myns",
            ]
        )
        assert args.summary == "test"
        assert args.detail == "details"
        assert args.tags == ["py", "test"]
        assert args.importance == 0.9
        assert args.namespace == "myns"

    def test_recall_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recall", "my query"])
        assert args.command == "recall"
        assert args.query == "my query"
        assert args.limit == 10
        assert args.fmt == "table"

    def test_recall_with_format(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recall", "q", "--format", "json"])
        assert args.fmt == "json"

    def test_search_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["search", "--tags", "py", "--min-importance", "0.5"])
        assert args.command == "search"
        assert args.tags == ["py"]
        assert args.min_importance == 0.5

    def test_consolidate_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["consolidate", "--dry-run"])
        assert args.command == "consolidate"
        assert args.dry_run is True

    def test_export_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "yaml", "--output", "/tmp/out.yaml"])
        assert args.command == "export"
        assert args.fmt == "yaml"
        assert args.output == "/tmp/out.yaml"

    def test_import_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["import", "/tmp/data.json", "--merge"])
        assert args.command == "import"
        assert args.path == "/tmp/data.json"
        assert args.merge is True

    def test_status_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "--format", "json"])
        assert args.command == "status"
        assert args.fmt == "json"

    def test_forget_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["forget", "M-abc123"])
        assert args.command == "forget"
        assert args.memory_id == "M-abc123"


# ---------------------------------------------------------------------------
# main() — no command
# ---------------------------------------------------------------------------


class TestMainNoCommand:
    def test_no_args_returns_1(self) -> None:
        assert main([]) == 1

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([])
        captured = capsys.readouterr()
        assert "trw-memory" in captured.out


# ---------------------------------------------------------------------------
# store subcommand
# ---------------------------------------------------------------------------


class TestStoreCommand:
    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_store_success(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["store", "--summary", "Test content"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Stored:" in captured.out
        assert "M-abc12345" in captured.out
        client.store.assert_awaited_once()
        client.close.assert_awaited_once()

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_store_with_tags(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["store", "--summary", "test", "--tags", "py", "--tags", "sql"])
        call_kwargs = client.store.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("tags") == ["py", "sql"]

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_store_with_importance(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["store", "--summary", "test", "--importance", "0.9"])
        call_kwargs = client.store.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("importance") == 0.9

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_store_error(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.store = AsyncMock(side_effect=ValueError("empty content"))
        mock_cls.return_value = client
        ret = main(["store", "--summary", ""])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_store_missing_summary(self) -> None:
        with pytest.raises(SystemExit):
            main(["store"])


# ---------------------------------------------------------------------------
# recall subcommand
# ---------------------------------------------------------------------------


class TestRecallCommand:
    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_table(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["recall", "test query"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "M-abc12345" in captured.out
        client.recall.assert_awaited_once()

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_json(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["recall", "test query", "--format", "json"])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_compact(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["recall", "q", "--format", "compact"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "score=" in captured.out

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_with_tags(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["recall", "q", "--tags", "py"])
        call_kwargs = client.recall.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("tags") == ["py"]

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_with_limit(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["recall", "q", "--limit", "5"])
        call_kwargs = client.recall.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("limit") == 5

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_error(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.recall = AsyncMock(side_effect=RuntimeError("backend down"))
        mock_cls.return_value = client
        ret = main(["recall", "q"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err


# ---------------------------------------------------------------------------
# search subcommand
# ---------------------------------------------------------------------------


class TestSearchCommand:
    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_success(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["search", "--tags", "py"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "M-abc12345" in captured.out

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_with_since(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["search", "--since", "2026-01-01T00:00:00"])
        assert ret == 0

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_invalid_since(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["search", "--since", "not-a-date"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_json_format(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["search", "--format", "json"])
        assert ret == 0
        parsed = json.loads(capsys.readouterr().out)
        assert isinstance(parsed, list)

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_with_min_importance(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["search", "--min-importance", "0.8"])
        call_kwargs = client.search.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("min_importance") == 0.8


# ---------------------------------------------------------------------------
# consolidate subcommand
# ---------------------------------------------------------------------------


class TestConsolidateCommand:
    @patch(f"{_CLI}.consolidate_cycle")
    @patch(f"{_CLI}.get_local_embedder")
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_consolidate_success(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        mock_get_local_embedder: MagicMock,
        mock_cycle: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend
        mock_get_local_embedder.return_value = MagicMock()
        mock_cycle.return_value = {"status": "no_clusters", "consolidated_count": 0}

        ret = main(["consolidate"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "no_clusters" in captured.out
        mock_backend.close.assert_called_once()

    @patch(f"{_CLI}.consolidate_cycle")
    @patch(f"{_CLI}.get_local_embedder")
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_consolidate_dry_run(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        mock_get_local_embedder: MagicMock,
        mock_cycle: MagicMock,
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend_fn.return_value = MagicMock()
        mock_get_local_embedder.return_value = MagicMock()
        mock_cycle.return_value = {"dry_run": True, "clusters": [], "consolidated_count": 0}
        ret = main(["consolidate", "--dry-run"])
        assert ret == 0
        call_kwargs = mock_cycle.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("dry_run")

    @patch(f"{_CLI}.consolidate_cycle")
    @patch(f"{_CLI}.get_local_embedder")
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_consolidate_passes_resolved_embedder(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        mock_get_local_embedder: MagicMock,
        mock_cycle: MagicMock,
    ) -> None:
        fake_config = MagicMock()
        fake_config.embedding_model = "test-model"
        fake_config.embedding_dim = 123
        mock_config_cls.return_value = fake_config
        mock_backend_fn.return_value = MagicMock()
        fake_embedder = MagicMock()
        mock_get_local_embedder.return_value = fake_embedder
        mock_cycle.return_value = {"status": "no_clusters", "consolidated_count": 0}

        ret = main(["consolidate"])

        assert ret == 0
        mock_get_local_embedder.assert_called_once_with(model_name="test-model", dim=123)
        kwargs = mock_cycle.call_args.kwargs
        assert kwargs["embedder"] is fake_embedder

    @patch(f"{_CLI}.MemoryConfig", side_effect=RuntimeError("config fail"))
    def test_consolidate_error(
        self,
        mock_config_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        ret = main(["consolidate"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err


# ---------------------------------------------------------------------------
# export subcommand
# ---------------------------------------------------------------------------


class TestExportCommand:
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_export_json_stdout(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = [_mock_entry()]
        mock_backend_fn.return_value = mock_backend

        ret = main(["export"])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "M-001"

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_export_json_to_file(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = [_mock_entry()]
        mock_backend_fn.return_value = mock_backend

        out_path = str(tmp_path / "out.json")
        ret = main(["export", "--output", out_path])
        assert ret == 0
        data = json.loads(Path(out_path).read_text())
        assert len(data) == 1

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_export_empty(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = []
        mock_backend_fn.return_value = mock_backend

        ret = main(["export"])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == []

    @patch(f"{_CLI}.MemoryConfig", side_effect=RuntimeError("fail"))
    def test_export_error(
        self,
        mock_config_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        ret = main(["export"])
        assert ret == 1
        assert "Error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# import subcommand
# ---------------------------------------------------------------------------


class TestImportCommand:
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_json_success(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        data = [
            {"content": "Entry 1", "tags": ["a"], "importance": 0.7},
            {"content": "Entry 2", "tags": ["b"], "importance": 0.5},
        ]
        fpath = tmp_path / "import.json"
        fpath.write_text(json.dumps(data))

        ret = main(["import", str(fpath)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 2" in captured.out
        assert mock_backend.store.call_count == 2
        mock_backend.close.assert_called_once()

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_skips_empty_content(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        data = [{"content": ""}, {"content": "valid"}]
        fpath = tmp_path / "import.json"
        fpath.write_text(json.dumps(data))

        ret = main(["import", str(fpath)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 1" in captured.out
        assert "skipped 1" in captured.out

    def test_import_file_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["import", "/nonexistent/file.json"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "file not found" in captured.err

    def test_import_invalid_json(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fpath = tmp_path / "bad.json"
        fpath.write_text("{not valid json")
        ret = main(["import", str(fpath)])
        assert ret == 1
        assert "Error:" in capsys.readouterr().err

    def test_import_not_a_list(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fpath = tmp_path / "obj.json"
        fpath.write_text('{"key": "value"}')
        ret = main(["import", str(fpath)])
        assert ret == 1
        assert "expected a JSON/YAML array" in capsys.readouterr().err

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_merge_mode_skip_existing(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        # backend.get returns an entry for existing ID, None for others
        mock_backend.get.side_effect = lambda eid: _mock_entry() if eid == "M-existing" else None
        mock_backend_fn.return_value = mock_backend

        data = [
            {"id": "M-existing", "content": "old"},
            {"content": "new entry"},
        ]
        fpath = tmp_path / "import.json"
        fpath.write_text(json.dumps(data))

        ret = main(["import", str(fpath), "--merge"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "skipped 1" in captured.out

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_merge_mode_no_id(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Entries without an id field are always imported in merge mode."""
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        data = [{"content": "no id entry"}]
        fpath = tmp_path / "import.json"
        fpath.write_text(json.dumps(data))

        ret = main(["import", str(fpath), "--merge"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 1" in captured.out

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_handles_non_dict_entries(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        data = ["not a dict", {"content": "valid"}]
        fpath = tmp_path / "import.json"
        fpath.write_text(json.dumps(data))

        ret = main(["import", str(fpath)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 1" in captured.out

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_non_list_tags_ignored(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        data = [{"content": "test", "tags": "not-a-list"}]
        fpath = tmp_path / "import.json"
        fpath.write_text(json.dumps(data))

        ret = main(["import", str(fpath)])
        assert ret == 0
        # Verify backend.store was called with an entry that has empty tags
        call_args = mock_backend.store.call_args
        assert call_args is not None
        stored_entry = call_args[0][0]
        assert list(stored_entry.tags) == []


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------


class TestStatusCommand:
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_status_table(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = MagicMock()
        config.storage_backend = "sqlite"
        config.storage_path = ".memory"
        mock_config_cls.return_value = config

        mock_backend = MagicMock()
        mock_backend.count.return_value = 42
        mock_backend_fn.return_value = mock_backend

        ret = main(["status"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "42" in captured.out
        assert "Memory System Status" in captured.out

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_status_json(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = MagicMock()
        config.storage_backend = "sqlite"
        config.storage_path = ".memory"
        mock_config_cls.return_value = config

        mock_backend = MagicMock()
        mock_backend.count.return_value = 5
        mock_backend_fn.return_value = mock_backend

        ret = main(["status", "--format", "json"])
        assert ret == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["entry_count"] == 5

    @patch(f"{_CLI}.MemoryConfig", side_effect=RuntimeError("fail"))
    def test_status_error(
        self,
        mock_config_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        ret = main(["status"])
        assert ret == 1
        assert "Error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# forget subcommand
# ---------------------------------------------------------------------------


class TestForgetCommand:
    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_forget_success(
        self,
        mock_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["forget", "M-abc123"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Deleted:" in captured.out
        client.forget.assert_awaited_once_with("M-abc123")

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_forget_not_found(
        self,
        mock_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from trw_memory.exceptions import MemoryNotFoundError

        client = _mock_client()
        client.forget = AsyncMock(side_effect=MemoryNotFoundError("not found"))
        mock_cls.return_value = client
        ret = main(["forget", "M-nonexistent"])
        assert ret == 1
        assert "Error:" in capsys.readouterr().err

    def test_forget_missing_id(self) -> None:
        with pytest.raises(SystemExit):
            main(["forget"])


# ---------------------------------------------------------------------------
# Export/Import round-trip
# ---------------------------------------------------------------------------


class TestYamlExport:
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_export_yaml_to_file(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
    ) -> None:
        from ruamel.yaml import YAML

        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = [_mock_entry()]
        mock_backend_fn.return_value = mock_backend

        out_path = str(tmp_path / "out.yaml")
        ret = main(["export", "--format", "yaml", "--output", out_path])
        assert ret == 0
        yaml = YAML()
        data = yaml.load(Path(out_path))
        assert len(data) == 1
        assert data[0]["id"] == "M-001"

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_export_yaml_stdout(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from io import StringIO

        from ruamel.yaml import YAML

        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = [_mock_entry()]
        mock_backend_fn.return_value = mock_backend

        ret = main(["export", "--format", "yaml"])
        assert ret == 0
        captured = capsys.readouterr()
        yaml = YAML()
        data = yaml.load(StringIO(captured.out))
        assert len(data) == 1
        assert data[0]["id"] == "M-001"


class TestYamlImport:
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_yaml_success(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from ruamel.yaml import YAML

        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        yaml = YAML()
        data = [
            {"content": "YAML Entry 1", "tags": ["a"], "importance": 0.7},
            {"content": "YAML Entry 2", "tags": ["b"], "importance": 0.5},
        ]
        fpath = tmp_path / "import.yaml"
        with open(fpath, "w") as f:
            yaml.dump(data, f)

        ret = main(["import", str(fpath)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 2" in captured.out
        assert mock_backend.store.call_count == 2

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_yml_extension(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from ruamel.yaml import YAML

        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend_fn.return_value = mock_backend

        yaml = YAML()
        fpath = tmp_path / "import.yml"
        with open(fpath, "w") as f:
            yaml.dump([{"content": "yml test"}], f)

        ret = main(["import", str(fpath)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 1" in captured.out


class TestExportImportRoundTrip:
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_json_round_trip(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Setup export backend
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = [
            _mock_entry(entry_id="M-round", content="roundtrip test", tags=["rt"])
        ]
        mock_backend_fn.return_value = mock_backend

        # Export
        out_path = str(tmp_path / "roundtrip.json")
        ret = main(["export", "--output", out_path])
        assert ret == 0

        # Verify export file
        data = json.loads(Path(out_path).read_text())
        assert len(data) == 1
        assert data[0]["content"] == "roundtrip test"

        # Clear capsys
        capsys.readouterr()

        # Import (reuses same mocked _create_local_backend and MemoryConfig)
        ret = main(["import", out_path])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 1" in captured.out
        # Verify backend.store was called with the roundtrip content
        store_calls = [
            c
            for c in mock_backend.store.call_args_list
            if hasattr(c[0][0], "content") and c[0][0].content == "roundtrip test"
        ]
        assert len(store_calls) == 1

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_yaml_round_trip(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from ruamel.yaml import YAML

        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = [_mock_entry(entry_id="M-yaml", content="yaml roundtrip", tags=["yr"])]
        mock_backend_fn.return_value = mock_backend

        # Export YAML
        out_path = str(tmp_path / "roundtrip.yaml")
        ret = main(["export", "--format", "yaml", "--output", out_path])
        assert ret == 0

        yaml = YAML()
        data = yaml.load(Path(out_path))
        assert len(data) == 1
        assert data[0]["content"] == "yaml roundtrip"

        capsys.readouterr()

        # Import YAML
        ret = main(["import", out_path])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 1" in captured.out
