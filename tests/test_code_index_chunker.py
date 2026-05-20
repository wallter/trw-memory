"""Tests for deterministic code chunking."""

from __future__ import annotations

from trw_memory.code_index.chunker import CodeChunker


def test_python_chunking_uses_ast_symbols_when_available() -> None:
    source = "class Service:\n    def run(self) -> str:\n        return 'ok'\n\ndef helper() -> int:\n    return 1\n"

    result = CodeChunker().chunk_text(
        namespace="project",
        path="src/service.py",
        text=source,
    )

    assert result.diagnostic.language == "python"
    assert result.diagnostic.strategy == "python_ast"
    assert [(symbol.name, symbol.kind, symbol.start_line) for symbol in result.symbols] == [
        ("Service", "class", 1),
        ("run", "method", 2),
        ("helper", "function", 5),
    ]
    assert [chunk.start_line for chunk in result.chunks] == [1, 5]
    assert result.chunks[0].symbols == ["Service", "run"]


def test_typescript_chunking_uses_dependency_optional_syntax_fallback() -> None:
    source = "export class Widget {\n  render() { return 'x'; }\n}\nexport function mount() { return new Widget(); }\n"

    result = CodeChunker().chunk_text(namespace="project", path="src/widget.ts", text=source)

    assert result.diagnostic.language == "typescript"
    assert result.diagnostic.parser_available is False
    assert result.diagnostic.strategy == "syntax_fallback"
    assert result.diagnostic.reason == "optional parser dependency unavailable"
    assert [(symbol.name, symbol.kind) for symbol in result.symbols] == [("Widget", "class"), ("mount", "function")]
    assert result.chunks[0].snippet.startswith("export class Widget")


def test_unsupported_language_falls_back_to_deterministic_text_windows() -> None:
    source = "\n".join(f"line {index}" for index in range(1, 18))

    result = CodeChunker(max_chunk_lines=5).chunk_text(namespace="project", path="README.md", text=source)

    assert result.diagnostic.language == "text"
    assert result.diagnostic.strategy == "text_fallback"
    assert [(chunk.start_line, chunk.end_line) for chunk in result.chunks] == [(1, 5), (6, 10), (11, 15), (16, 17)]
    assert [chunk.snippet for chunk in result.chunks] == [
        "line 1\nline 2\nline 3\nline 4\nline 5",
        "line 6\nline 7\nline 8\nline 9\nline 10",
        "line 11\nline 12\nline 13\nline 14\nline 15",
        "line 16\nline 17",
    ]
