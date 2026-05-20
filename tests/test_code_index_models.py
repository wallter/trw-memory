"""Tests for typed code-index schema models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.code_index.models import (
    CodeChunk,
    CodeFile,
    CodeSymbol,
    EmbeddingMetadata,
    content_sha256,
    make_bounded_snippet,
    stable_code_chunk_id,
    stable_code_file_id,
    stable_code_symbol_id,
    validate_code_path,
)


def test_code_models_round_trip_all_required_schema_fields() -> None:
    content_hash = content_sha256("def alpha() -> str:\n    return 'ok'\n")
    code_file = CodeFile(
        namespace="project",
        path="src/app.py",
        language="python",
        content_hash=content_hash,
        size_bytes=36,
        embedding=EmbeddingMetadata(provider="code-embed", model="mini", dimensions=3, available=True),
    )
    symbol = CodeSymbol(
        namespace="project",
        file_id=code_file.id,
        path=code_file.path,
        name="alpha",
        kind="function",
        language="python",
        start_line=1,
        end_line=2,
    )
    chunk = CodeChunk(
        namespace="project",
        file_id=code_file.id,
        path=code_file.path,
        language="python",
        start_line=1,
        end_line=2,
        content_hash=content_hash,
        snippet="def alpha() -> str:\n    return 'ok'",
        symbols=[symbol.name],
        embedding=EmbeddingMetadata(provider="", model="", dimensions=0, available=False, reason="not configured"),
    )

    restored = CodeChunk.model_validate_json(chunk.model_dump_json())

    assert code_file.id == stable_code_file_id("project", "src/app.py", content_hash)
    assert symbol.id == stable_code_symbol_id("project", "src/app.py", "alpha", "function", 1, 2)
    assert chunk.id == stable_code_chunk_id("project", "src/app.py", 1, 2, content_hash)
    assert restored.namespace == "project"
    assert restored.path == "src/app.py"
    assert restored.symbols == ["alpha"]
    assert restored.embedding.reason == "not configured"


@pytest.mark.parametrize("path", ["../escape.py", "/absolute.py", "src/../secret.py", "src/app.py;rm"])
def test_code_path_rejects_traversal_absolute_and_shell_like_paths(path: str) -> None:
    with pytest.raises(ValueError):
        validate_code_path(path)


def test_code_models_reject_unknown_fields_and_invalid_ranges() -> None:
    with pytest.raises(ValidationError) as extra_error:
        CodeFile(namespace="n", path="src/app.py", language="python", content_hash="0" * 64, unexpected="field")

    with pytest.raises(ValidationError) as range_error:
        CodeChunk(
            namespace="n",
            file_id="code-file:abc",
            path="src/app.py",
            language="python",
            start_line=5,
            end_line=4,
            content_hash="0" * 64,
            snippet="bad",
        )

    assert extra_error.value.errors()[0]["type"] == "extra_forbidden"
    assert range_error.value.errors()[0]["type"] == "value_error"


def test_bounded_snippet_never_returns_full_large_content() -> None:
    text = "\n".join(f"line {index}" for index in range(1, 50))

    snippet = make_bounded_snippet(text, start_line=10, end_line=40, max_lines=4, max_chars=40)

    assert snippet.startswith("line 10")
    assert len(snippet) <= 41
    assert "line 20" not in snippet
