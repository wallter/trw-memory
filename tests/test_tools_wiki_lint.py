from __future__ import annotations

import json
from pathlib import Path

import pytest

from trw_memory.cli import main
from trw_memory.tools.wiki_lint import memory_wiki_lint_impl


def test_memory_wiki_lint_impl_returns_bounded_summary() -> None:
    result = memory_wiki_lint_impl(
        [
            {
                "kind": "topic",
                "slug": "topic/a",
                "title": "A",
                "confidence": "medium",
                "outbound_refs": [{"target_slug": "topic/missing"}],
            }
        ],
        top_limit=1,
    )

    assert result == {
        "summary": {"missing_target": 1, "provenance_gap": 1},
        "top_findings": [
            {
                "code": "missing_target",
                "page_slug": "topic/a",
                "severity": "error",
                "target_slug": "topic/missing",
            }
        ],
        "total": 2,
    }


def test_wiki_lint_cli_prints_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "pages.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "kind": "topic",
                    "slug": "topic/a",
                    "title": "A",
                    "confidence": "medium",
                    "outbound_refs": [{"target_slug": "topic/missing"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert main(["wiki-lint", str(input_path), "--top-limit", "1"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["missing_target"] == 1
    assert output["total"] == 2


def test_wiki_lint_cli_missing_file_fails_closed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["wiki-lint", "/nonexistent/wiki-pages.json"]) == 1
    err = capsys.readouterr().err
    assert "Error: file not found" in err
    # Structural error, not a leaked OSError errno string.
    assert "Errno" not in err


def test_wiki_lint_cli_invalid_json_reports_structural_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "pages.json"
    input_path.write_text('[{"slug": broken}]')
    assert main(["wiki-lint", str(input_path)]) == 1
    err = capsys.readouterr().err
    assert "is not valid JSON" in err
    assert "broken" not in err


def test_wiki_lint_cli_non_list_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "pages.json"
    input_path.write_text('{"slug": "topic/a"}')
    assert main(["wiki-lint", str(input_path)]) == 1
    err = capsys.readouterr().err
    assert "must be a JSON array, got object" in err


def test_wiki_lint_cli_non_object_item_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "pages.json"
    input_path.write_text('["not-an-object"]')
    assert main(["wiki-lint", str(input_path)]) == 1
    err = capsys.readouterr().err
    assert "item 0 must be a JSON object, got string" in err


def test_wiki_lint_cli_non_utf8_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "pages.json"
    input_path.write_bytes(b"\xff\xfe[]")
    assert main(["wiki-lint", str(input_path)]) == 1
    err = capsys.readouterr().err
    assert "not valid UTF-8" in err
