"""Tests for trw_memory.cli YAML import/export flows."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ruamel.yaml import YAML

from trw_memory.cli import main

from ._test_cli_support import _CLI, _mock_entry, _real_import_target, _reopen_import_target


class TestYamlExport:
    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_export_yaml_to_file(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
    ) -> None:
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
        config, backend = _real_import_target(tmp_path)
        mock_config_cls.return_value = config
        mock_backend_fn.return_value = backend

        yaml = YAML()
        data = [
            {"content": "YAML Entry 1", "tags": ["a"], "importance": 0.7},
            {"content": "YAML Entry 2", "tags": ["b"], "importance": 0.5},
        ]
        fpath = tmp_path / "import.yaml"
        with fpath.open("w") as handle:
            yaml.dump(data, handle)

        ret = main(["import", str(fpath)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 2" in captured.out
        with _reopen_import_target(tmp_path) as reopened:
            stored = {entry.content for entry in reopened.list_entries(namespace="default", limit=10)}
        assert stored == {"YAML Entry 1", "YAML Entry 2"}

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_import_yml_extension(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config, backend = _real_import_target(tmp_path)
        mock_config_cls.return_value = config
        mock_backend_fn.return_value = backend

        yaml = YAML()
        fpath = tmp_path / "import.yml"
        with fpath.open("w") as handle:
            yaml.dump([{"content": "yml test"}], handle)

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
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = [
            _mock_entry(entry_id="M-round", content="roundtrip test", tags=["rt"])
        ]
        mock_backend_fn.return_value = mock_backend

        out_path = str(tmp_path / "roundtrip.json")
        ret = main(["export", "--output", out_path])
        assert ret == 0

        data = json.loads(Path(out_path).read_text())
        assert len(data) == 1
        assert data[0]["content"] == "roundtrip test"

        capsys.readouterr()

        # The import half writes through the real store gate, so swap the mocked
        # export backend for a real one before replaying the exported file.
        import_config, import_backend = _real_import_target(tmp_path)
        mock_config_cls.return_value = import_config
        mock_backend_fn.return_value = import_backend

        ret = main(["import", out_path])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 1" in captured.out
        with _reopen_import_target(tmp_path) as reopened:
            stored = [entry.content for entry in reopened.list_entries(namespace="default", limit=10)]
        assert stored == ["roundtrip test"]

    @patch(f"{_CLI}._create_local_backend")
    @patch(f"{_CLI}.MemoryConfig")
    def test_yaml_round_trip(
        self,
        mock_config_cls: MagicMock,
        mock_backend_fn: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_config_cls.return_value = MagicMock()
        mock_backend = MagicMock()
        mock_backend.list_entries.return_value = [_mock_entry(entry_id="M-yaml", content="yaml roundtrip", tags=["yr"])]
        mock_backend_fn.return_value = mock_backend

        out_path = str(tmp_path / "roundtrip.yaml")
        ret = main(["export", "--format", "yaml", "--output", out_path])
        assert ret == 0

        yaml = YAML()
        data = yaml.load(Path(out_path))
        assert len(data) == 1
        assert data[0]["content"] == "yaml roundtrip"

        capsys.readouterr()

        import_config, import_backend = _real_import_target(tmp_path)
        mock_config_cls.return_value = import_config
        mock_backend_fn.return_value = import_backend

        ret = main(["import", out_path])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Imported 1" in captured.out
        with _reopen_import_target(tmp_path) as reopened:
            stored = [entry.content for entry in reopened.list_entries(namespace="default", limit=10)]
        assert stored == ["yaml roundtrip"]
