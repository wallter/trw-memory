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
