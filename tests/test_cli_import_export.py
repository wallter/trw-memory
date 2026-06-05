"""Tests for trw_memory.cli JSON import and export commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.cli import main

from ._test_cli_support import _CLI, _mock_entry


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

    def test_import_yaml_rejects_python_object_tag(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fpath = tmp_path / "payload.yaml"
        fpath.write_text('!!python/object/apply:os.system ["id"]\n')
        ret = main(["import", str(fpath)])
        assert ret == 1
        err = capsys.readouterr().err
        assert "Error:" in err

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

    def test_import_invalid_json_reports_structural_reason_without_payload(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fpath = tmp_path / "bad.json"
        fpath.write_text('{"leaky-secret-key": not-json}')
        ret = main(["import", str(fpath)])
        assert ret == 1
        err = capsys.readouterr().err
        assert "is not valid JSON" in err
        # Structural diagnostics only — never echo the offending payload content.
        assert "leaky-secret-key" not in err

    def test_import_non_utf8_fails_closed_without_byte_dump(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fpath = tmp_path / "binary.json"
        fpath.write_bytes(b"\xff\xfe\x00\x01not utf-8 bytes")
        ret = main(["import", str(fpath)])
        assert ret == 1
        err = capsys.readouterr().err
        assert "not valid UTF-8" in err
        # The undecodable bytes must not be leaked into the diagnostic.
        assert "\\xff" not in err

    def test_import_directory_path_fails_closed(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        target = tmp_path / "a_directory.json"
        target.mkdir()
        ret = main(["import", str(target)])
        assert ret == 1
        err = capsys.readouterr().err
        assert "Error:" in err
        assert "directory" in err

    def test_import_yaml_invalid_reports_structural_reason(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fpath = tmp_path / "bad.yaml"
        fpath.write_text("key: [unterminated\n")
        ret = main(["import", str(fpath)])
        assert ret == 1
        err = capsys.readouterr().err
        assert "is not valid YAML" in err

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
        call_args = mock_backend.store.call_args
        assert call_args is not None
        stored_entry = call_args[0][0]
        assert list(stored_entry.tags) == []
