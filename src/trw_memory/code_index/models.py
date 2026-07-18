"""Strong typed code-index schema models.

These models do not reuse ``MemoryEntry`` and are safe for pure in-memory or
file-backed code-index storage.
"""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ChunkingDiagnostic",
    "CodeChunk",
    "CodeFile",
    "CodeSearchResult",
    "CodeSymbol",
    "EmbeddingMetadata",
    "SymbolLookupResult",
    "content_sha256",
    "make_bounded_snippet",
    "stable_code_chunk_id",
    "stable_code_file_id",
    "stable_code_symbol_id",
    "validate_code_path",
]

_SHELL_METACHARS = frozenset(";|&`$<>\\")
_MAX_SNIPPET_LINES = 40
_MAX_SNIPPET_CHARS = 400


class EmbeddingMetadata(BaseModel):
    """Embedding status for code-index objects, independent from memory embeddings."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    provider: str = Field(default="", max_length=128)
    model: str = Field(default="", max_length=128)
    dimensions: int = Field(default=0, ge=0)
    available: bool = False
    reason: str = Field(default="", max_length=240)

    @field_validator("provider", "model", "reason")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class ChunkingDiagnostic(BaseModel):
    """Structured parser/fallback status for a chunking operation."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    language: str
    strategy: Literal["python_ast", "parser_ast", "syntax_fallback", "text_fallback"]
    parser_available: bool
    reason: str = ""


class CodeFile(BaseModel):
    """A code file record stored outside ``MemoryEntry``."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    namespace: str = Field(default="default", min_length=1, max_length=128)
    path: str
    language: str = Field(min_length=1, max_length=64)
    content_hash: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(default=0, ge=0)
    embedding: EmbeddingMetadata = Field(default_factory=EmbeddingMetadata)
    id: str = ""

    @field_validator("namespace", "language")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("code-index text fields must not be blank")
        return stripped

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return validate_code_path(value)

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, value: str) -> str:
        if not all(char in "0123456789abcdef" for char in value):
            raise ValueError("content_hash must be lowercase sha256 hex")
        return value

    @model_validator(mode="after")
    def _set_id(self) -> CodeFile:
        expected_id = stable_code_file_id(self.namespace, self.path, self.content_hash)
        if self.id and self.id != expected_id:
            raise ValueError("code file id does not match stable id fields")
        self.id = expected_id
        return self


class CodeSymbol(BaseModel):
    """A symbol record with disambiguating source location."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    namespace: str = Field(default="default", min_length=1, max_length=128)
    file_id: str = Field(min_length=1)
    path: str
    name: str = Field(min_length=1, max_length=240)
    kind: str = Field(min_length=1, max_length=64)
    language: str = Field(min_length=1, max_length=64)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    id: str = ""

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return validate_code_path(value)

    @field_validator("namespace", "file_id", "name", "kind", "language")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("symbol fields must not be blank")
        return stripped

    @model_validator(mode="after")
    def _validate_range_and_id(self) -> CodeSymbol:
        if self.end_line < self.start_line:
            raise ValueError("symbol end_line must be >= start_line")
        expected_id = stable_code_symbol_id(
            self.namespace, self.path, self.name, self.kind, self.start_line, self.end_line
        )
        if self.id and self.id != expected_id:
            raise ValueError("code symbol id does not match stable id fields")
        self.id = expected_id
        return self


class CodeChunk(BaseModel):
    """A bounded source snippet and line range for code search."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    namespace: str = Field(default="default", min_length=1, max_length=128)
    file_id: str = Field(min_length=1)
    path: str
    language: str = Field(min_length=1, max_length=64)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content_hash: str = Field(min_length=64, max_length=64)
    snippet: str = Field(max_length=_MAX_SNIPPET_CHARS + 1)
    symbols: list[str] = Field(default_factory=list)
    embedding: EmbeddingMetadata = Field(default_factory=EmbeddingMetadata)
    id: str = ""

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return validate_code_path(value)

    @field_validator("namespace", "file_id", "language")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("chunk fields must not be blank")
        return stripped

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, value: str) -> str:
        if not all(char in "0123456789abcdef" for char in value):
            raise ValueError("content_hash must be lowercase sha256 hex")
        return value

    @field_validator("symbols")
    @classmethod
    def _dedupe_symbols(cls, value: list[str]) -> list[str]:
        return sorted({symbol.strip() for symbol in value if symbol.strip()})

    @model_validator(mode="after")
    def _validate_range_and_id(self) -> CodeChunk:
        if self.end_line < self.start_line:
            raise ValueError("chunk end_line must be >= start_line")
        expected_id = stable_code_chunk_id(self.namespace, self.path, self.start_line, self.end_line, self.content_hash)
        if self.id and self.id != expected_id:
            raise ValueError("code chunk id does not match stable id fields")
        self.id = expected_id
        return self


class CodeSearchResult(BaseModel):
    """Explicit code search result with bounded snippet and line range."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    file: str
    symbol: str = ""
    language: str
    line_range: tuple[int, int]
    score: float
    snippet: str = Field(max_length=_MAX_SNIPPET_CHARS + 1)
    embedding: EmbeddingMetadata = Field(default_factory=EmbeddingMetadata)


class SymbolLookupResult(BaseModel):
    """Symbol lookup result with duplicate disambiguation metadata."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    name: str
    kind: str
    file: str
    path: str
    language: str
    line_range: tuple[int, int]
    disambiguation: str


def content_sha256(content: str | bytes) -> str:
    """Return a lowercase sha256 hex digest for stable change detection."""

    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def stable_code_file_id(namespace: str, path: str, content_hash: str) -> str:
    """Return a stable file id scoped by namespace, path, and content hash."""

    return _stable_id("code-file", namespace, path, content_hash)


def stable_code_symbol_id(namespace: str, path: str, name: str, kind: str, start_line: int, end_line: int) -> str:
    """Return a stable symbol id scoped by namespace, symbol, and location."""

    return _stable_id("code-symbol", namespace, path, name, kind, str(start_line), str(end_line))


def stable_code_chunk_id(namespace: str, path: str, start_line: int, end_line: int, content_hash: str) -> str:
    """Return a stable chunk id scoped by namespace, location, and file hash."""

    return _stable_id("code-chunk", namespace, path, str(start_line), str(end_line), content_hash)


def validate_code_path(value: str) -> str:
    """Validate a relative POSIX code path without traversal or shell syntax."""

    path = value.strip()
    if not path:
        raise ValueError("code path must not be empty")
    if any(char in path for char in _SHELL_METACHARS):
        raise ValueError("code path must not contain shell metacharacters")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute():
        raise ValueError("code path must be relative")
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ValueError("code path must not contain traversal segments")
    return pure_path.as_posix()


def make_bounded_snippet(
    text: str,
    *,
    start_line: int,
    end_line: int,
    max_lines: int = _MAX_SNIPPET_LINES,
    max_chars: int = _MAX_SNIPPET_CHARS,
) -> str:
    """Return a line/character bounded snippet; never returns large full-file dumps."""

    lines = text.splitlines()
    safe_start = max(1, start_line)
    safe_end = max(safe_start, end_line)
    selected = lines[safe_start - 1 : min(safe_end, safe_start + max_lines - 1)]
    snippet = "\n".join(selected)
    if len(snippet) <= max_chars:
        return snippet
    return f"{snippet[: max(0, max_chars - 1)]}…"


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(part.strip() for part in parts)
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
