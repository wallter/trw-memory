"""Deterministic code chunking with dependency-optional parser fallback."""

from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from trw_memory.code_index.models import (
    ChunkingDiagnostic,
    CodeChunk,
    CodeFile,
    CodeSymbol,
    content_sha256,
    make_bounded_snippet,
    validate_code_path,
)

__all__ = ["ChunkingResult", "CodeChunker", "detect_language"]

_TS_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:(class|function)\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=)",
    re.MULTILINE,
)
_PYTHON_EXTENSIONS = frozenset({".py", ".pyi"})
_TYPESCRIPT_EXTENSIONS = frozenset({".ts", ".tsx"})
_JAVASCRIPT_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs"})


class ChunkingResult(BaseModel):
    """Chunking output plus structured parser/fallback diagnostic."""

    model_config = ConfigDict(extra="forbid", strict=True)

    file: CodeFile
    chunks: list[CodeChunk] = Field(default_factory=list)
    symbols: list[CodeSymbol] = Field(default_factory=list)
    diagnostic: ChunkingDiagnostic


class CodeChunker:
    """Deterministic code chunker for Python, TypeScript/JavaScript, and text fallback."""

    def __init__(self, *, max_chunk_lines: int = 80, max_snippet_chars: int = 400) -> None:
        if max_chunk_lines < 1:
            raise ValueError("max_chunk_lines must be >= 1")
        if max_snippet_chars < 40:
            raise ValueError("max_snippet_chars must be >= 40")
        self._max_chunk_lines = max_chunk_lines
        self._max_snippet_chars = max_snippet_chars

    def chunk_text(self, *, namespace: str, path: str, text: str) -> ChunkingResult:
        """Chunk source text into bounded snippets and symbols."""

        safe_path = validate_code_path(path)
        language = detect_language(safe_path)
        digest = content_sha256(text)
        code_file = CodeFile(
            namespace=namespace,
            path=safe_path,
            language=language,
            content_hash=digest,
            size_bytes=len(text.encode("utf-8")),
        )
        if language == "python":
            return self._chunk_python(code_file=code_file, text=text)
        if language in {"typescript", "javascript"}:
            return self._chunk_typescript_or_javascript(code_file=code_file, text=text)
        return self._chunk_text_fallback(
            code_file=code_file,
            text=text,
            diagnostic=ChunkingDiagnostic(
                language=language,
                strategy="text_fallback",
                parser_available=False,
                reason="unsupported language",
            ),
        )

    def _chunk_python(self, *, code_file: CodeFile, text: str) -> ChunkingResult:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self._chunk_text_fallback(
                code_file=code_file,
                text=text,
                diagnostic=ChunkingDiagnostic(
                    language="python",
                    strategy="text_fallback",
                    parser_available=True,
                    reason="python ast parse failed",
                ),
            )

        symbols = _python_symbols(tree, code_file=code_file)
        top_level_ranges = _python_top_level_ranges(tree, text)
        chunks = [
            _make_chunk(
                code_file=code_file,
                text=text,
                start_line=start_line,
                end_line=end_line,
                symbols=_symbols_in_range(symbols, start_line=start_line, end_line=end_line),
                max_snippet_chars=self._max_snippet_chars,
            )
            for start_line, end_line in top_level_ranges
        ]
        if not chunks:
            chunks = self._text_window_chunks(code_file=code_file, text=text)
        return ChunkingResult(
            file=code_file,
            chunks=chunks,
            symbols=symbols,
            diagnostic=ChunkingDiagnostic(language="python", strategy="python_ast", parser_available=True),
        )

    def _chunk_typescript_or_javascript(self, *, code_file: CodeFile, text: str) -> ChunkingResult:
        symbols = []
        for match in _TS_SYMBOL_RE.finditer(text):
            kind_match = match.group(1)
            function_name = match.group(2)
            variable_name = match.group(3)
            name = function_name or variable_name or ""
            kind = kind_match or "variable"
            line = text.count("\n", 0, match.start()) + 1
            symbols.append(
                CodeSymbol(
                    namespace=code_file.namespace,
                    file_id=code_file.id,
                    path=code_file.path,
                    name=name,
                    kind=kind,
                    language=code_file.language,
                    start_line=line,
                    end_line=_find_block_end(text, line),
                )
            )
        chunks = [
            _make_chunk(
                code_file=code_file,
                text=text,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbols=[symbol.name],
                max_snippet_chars=self._max_snippet_chars,
            )
            for symbol in symbols
        ]
        if not chunks:
            chunks = self._text_window_chunks(code_file=code_file, text=text)
        return ChunkingResult(
            file=code_file,
            chunks=chunks,
            symbols=symbols,
            diagnostic=ChunkingDiagnostic(
                language=code_file.language,
                strategy="syntax_fallback",
                parser_available=False,
                reason="optional parser dependency unavailable",
            ),
        )

    def _chunk_text_fallback(
        self,
        *,
        code_file: CodeFile,
        text: str,
        diagnostic: ChunkingDiagnostic,
    ) -> ChunkingResult:
        return ChunkingResult(
            file=code_file,
            chunks=self._text_window_chunks(code_file=code_file, text=text),
            symbols=[],
            diagnostic=diagnostic,
        )

    def _text_window_chunks(self, *, code_file: CodeFile, text: str) -> list[CodeChunk]:
        line_count = max(1, len(text.splitlines()))
        chunks = []
        for start_line in range(1, line_count + 1, self._max_chunk_lines):
            end_line = min(line_count, start_line + self._max_chunk_lines - 1)
            chunks.append(
                _make_chunk(
                    code_file=code_file,
                    text=text,
                    start_line=start_line,
                    end_line=end_line,
                    symbols=[],
                    max_snippet_chars=self._max_snippet_chars,
                )
            )
        return chunks


def detect_language(path: str) -> str:
    """Detect language from file extension for dependency-free operation."""

    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _PYTHON_EXTENSIONS:
        return "python"
    if suffix in _TYPESCRIPT_EXTENSIONS:
        return "typescript"
    if suffix in _JAVASCRIPT_EXTENSIONS:
        return "javascript"
    return "text"


def _python_symbols(tree: ast.AST, *, code_file: CodeFile) -> list[CodeSymbol]:
    symbols = []
    parents = _parent_map(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append(_python_symbol(code_file=code_file, node=node, kind="class"))
        elif isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            parent = parents.get(node)
            kind = "method" if isinstance(parent, ast.ClassDef) else "function"
            symbols.append(_python_symbol(code_file=code_file, node=node, kind=kind))
    return sorted(symbols, key=lambda symbol: (symbol.start_line, symbol.kind, symbol.name))


def _python_symbol(code_file: CodeFile, node: ast.ClassDef | ast.AsyncFunctionDef | ast.FunctionDef, kind: str) -> CodeSymbol:
    return CodeSymbol(
        namespace=code_file.namespace,
        file_id=code_file.id,
        path=code_file.path,
        name=node.name,
        kind=kind,
        language="python",
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
    )


def _python_top_level_ranges(tree: ast.AST, text: str) -> list[tuple[int, int]]:
    line_count = max(1, len(text.splitlines()))
    ranges = []
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef | ast.AsyncFunctionDef | ast.FunctionDef):
            end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
            ranges.append((node.lineno, end_line))
    return ranges or [(1, line_count)]


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _symbols_in_range(symbols: list[CodeSymbol], *, start_line: int, end_line: int) -> list[str]:
    return [symbol.name for symbol in symbols if start_line <= symbol.start_line <= end_line]


def _make_chunk(
    *,
    code_file: CodeFile,
    text: str,
    start_line: int,
    end_line: int,
    symbols: list[str],
    max_snippet_chars: int,
) -> CodeChunk:
    return CodeChunk(
        namespace=code_file.namespace,
        file_id=code_file.id,
        path=code_file.path,
        language=code_file.language,
        start_line=start_line,
        end_line=end_line,
        content_hash=code_file.content_hash,
        snippet=make_bounded_snippet(text, start_line=start_line, end_line=end_line, max_chars=max_snippet_chars),
        symbols=symbols,
    )


def _find_block_end(text: str, start_line: int) -> int:
    lines = text.splitlines()
    if start_line >= len(lines):
        return start_line
    for index in range(start_line, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith(("export class ", "export function ", "class ", "function ", "const ", "let ", "var ")):
            return index
    return min(len(lines), start_line + 20)
