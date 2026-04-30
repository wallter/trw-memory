"""Tests for trw_memory.cli maintenance commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.cli import main

from ._test_cli_support import _CLI


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

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_consolidate_rejects_invalid_namespace_before_backend_creation(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()

        ret = main(["consolidate", "--namespace", "../../escape"])

        assert ret == 1
        captured = capsys.readouterr()
        assert "Invalid namespace" in captured.err
        mock_backend_fn.assert_not_called()


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
