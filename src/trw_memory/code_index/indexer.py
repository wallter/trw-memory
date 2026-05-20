"""Pure code-index storage and filesystem indexer."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from trw_memory.code_index.chunker import CodeChunker
from trw_memory.code_index.models import CodeChunk, CodeFile, CodeSymbol, content_sha256, validate_code_path

__all__ = ["CodeIndexStats", "CodeIndexer", "InMemoryCodeIndex"]

_DEFAULT_EXCLUDED_DIRS = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "build",
    "dist",
    "__pycache__",
})
_DEFAULT_EXCLUDED_NAMES = frozenset({".env", ".env.local", "secrets.env"})
_SECRET_LIKE_NAME_TOKENS = frozenset({"credential", "password", "secret", "token"})
_BINARY_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar", ".so", ".dll", ".exe"})
_SOURCE_EXTENSIONS = frozenset({".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".md", ".txt"})


class CodeIndexStats(BaseModel):
    """Indexing summary for structured callers and tests."""

    model_config = ConfigDict(extra="forbid", strict=True)

    indexed_files: int = 0
    skipped_unchanged: int = 0
    skipped_excluded: int = 0
    deleted_files: int = 0


class InMemoryCodeIndex:
    """Small deterministic code-index store separate from memory storage."""

    def __init__(self) -> None:
        self._files: dict[tuple[str, str], CodeFile] = {}
        self._chunks: dict[tuple[str, str], list[CodeChunk]] = {}
        self._symbols: dict[tuple[str, str], list[CodeSymbol]] = {}

    def upsert_file(self, code_file: CodeFile, *, chunks: Iterable[CodeChunk], symbols: Iterable[CodeSymbol]) -> None:
        key = (code_file.namespace, code_file.path)
        self._files[key] = code_file
        self._chunks[key] = sorted(chunks, key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.id))
        self._symbols[key] = sorted(symbols, key=lambda symbol: (symbol.path, symbol.start_line, symbol.kind, symbol.name))

    def get_file(self, *, namespace: str, path: str) -> CodeFile | None:
        return self._files.get((namespace, validate_code_path(path)))

    def delete_file(self, *, namespace: str, path: str) -> bool:
        key = (namespace, validate_code_path(path))
        existed = key in self._files
        self._files.pop(key, None)
        self._chunks.pop(key, None)
        self._symbols.pop(key, None)
        return existed

    def delete_missing(self, *, namespace: str, paths: set[str]) -> int:
        deleted = 0
        for stored_namespace, stored_path in list(self._files):
            if stored_namespace == namespace and stored_path not in paths and self.delete_file(namespace=namespace, path=stored_path):
                deleted += 1
        return deleted

    def list_files(self, *, namespace: str | None = None) -> list[CodeFile]:
        return sorted(
            (code_file for code_file in self._files.values() if namespace is None or code_file.namespace == namespace),
            key=lambda code_file: (code_file.namespace, code_file.path),
        )

    def list_chunks(
        self,
        *,
        namespace: str | None = None,
        path_glob: str | None = None,
        language: str | None = None,
    ) -> list[CodeChunk]:
        chunks = [chunk for chunk_list in self._chunks.values() for chunk in chunk_list]
        return sorted(
            (
                chunk
                for chunk in chunks
                if (namespace is None or chunk.namespace == namespace)
                and (path_glob is None or fnmatch.fnmatch(chunk.path, path_glob))
                and (language is None or chunk.language == language)
            ),
            key=lambda chunk: (chunk.path, chunk.start_line, chunk.end_line),
        )

    def list_symbols(
        self,
        *,
        namespace: str | None = None,
        name: str | None = None,
        kind: str | None = None,
        path: str | None = None,
    ) -> list[CodeSymbol]:
        symbols = [symbol for symbol_list in self._symbols.values() for symbol in symbol_list]
        return sorted(
            (
                symbol
                for symbol in symbols
                if (namespace is None or symbol.namespace == namespace)
                and (name is None or symbol.name == name)
                and (kind is None or symbol.kind == kind)
                and (path is None or symbol.path == path)
            ),
            key=lambda symbol: (symbol.path, symbol.start_line, symbol.end_line, symbol.kind, symbol.name),
        )


class CodeIndexer:
    """Filesystem indexer with content-hash incremental behavior and default excludes."""

    def __init__(
        self,
        *,
        root: Path,
        store: InMemoryCodeIndex,
        namespace: str = "default",
        chunker: CodeChunker | None = None,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        self._root = root
        self._store = store
        self._namespace = namespace
        self._chunker = chunker or CodeChunker()
        self._max_file_bytes = max_file_bytes

    def index(self) -> CodeIndexStats:
        """Index source files below root, skipping unchanged and deleting missing entries."""

        indexed = 0
        skipped_unchanged = 0
        skipped_excluded = 0
        seen_paths: set[str] = set()
        for source_path in sorted(path for path in self._root.rglob("*") if path.is_file()):
            relative_path = source_path.relative_to(self._root).as_posix()
            if self._is_excluded(source_path, relative_path):
                skipped_excluded += 1
                continue
            seen_paths.add(validate_code_path(relative_path))
            content = source_path.read_text(encoding="utf-8")
            digest = content_sha256(content)
            existing = self._store.get_file(namespace=self._namespace, path=relative_path)
            if existing is not None and existing.content_hash == digest:
                skipped_unchanged += 1
                continue
            result = self._chunker.chunk_text(namespace=self._namespace, path=relative_path, text=content)
            self._store.upsert_file(result.file, chunks=result.chunks, symbols=result.symbols)
            indexed += 1
        deleted = self._store.delete_missing(namespace=self._namespace, paths=seen_paths)
        return CodeIndexStats(
            indexed_files=indexed,
            skipped_unchanged=skipped_unchanged,
            skipped_excluded=skipped_excluded,
            deleted_files=deleted,
        )

    def _is_excluded(self, source_path: Path, relative_path: str) -> bool:
        parts = set(Path(relative_path).parts)
        if parts & _DEFAULT_EXCLUDED_DIRS:
            return True
        lowered_name = source_path.name.lower()
        if source_path.name in _DEFAULT_EXCLUDED_NAMES or any(token in lowered_name for token in _SECRET_LIKE_NAME_TOKENS):
            return True
        if source_path.suffix.lower() in _BINARY_EXTENSIONS:
            return True
        if source_path.suffix.lower() not in _SOURCE_EXTENSIONS:
            return True
        return source_path.stat().st_size > self._max_file_bytes
