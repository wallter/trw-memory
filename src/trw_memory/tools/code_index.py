"""MCP-compatible tools for explicit trw-memory code index/search APIs."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from trw_memory.code_index.indexer import CodeIndexer, InMemoryCodeIndex
from trw_memory.code_index.search import code_search
from trw_memory.code_index.symbols import lookup_symbols
from trw_memory.tools._types import McpServer

_STORES: dict[tuple[str, str], InMemoryCodeIndex] = {}


def memory_code_index_impl(root: str, *, namespace: str = "default") -> dict[str, object]:
    """Index code under ``root`` into an explicit code-index store."""

    try:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            return {"status": "failed", "error_code": "invalid_root", "error": f"{root} is not a directory"}
        store = _store_for(root_path, namespace)
        stats = CodeIndexer(root=root_path, store=store, namespace=namespace).index()
        return {"status": "ok", "root": str(root_path), "namespace": namespace, "stats": stats.model_dump(mode="json")}
    except (OSError, ValueError) as exc:
        return {"status": "failed", "error_code": "index_failed", "error": str(exc)}


def memory_code_search_impl(
    root: str,
    query: str,
    *,
    namespace: str = "default",
    path_glob: str | None = None,
    language: str | None = None,
    limit: int = 10,
) -> dict[str, object]:
    """Search explicit code-index chunks with bounded snippets."""

    try:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            return {"status": "failed", "error_code": "invalid_root", "error": f"{root} is not a directory"}
        store = _indexed_store_for(root_path, namespace)
        results = code_search(
            store,
            query=query,
            namespace=namespace,
            path_glob=path_glob,
            language=language,
            limit=limit,
        )
        return {
            "status": "ok",
            "root": str(root_path),
            "namespace": namespace,
            "results": [cast("dict[str, object]", result.model_dump(mode="json")) for result in results],
        }
    except (OSError, ValueError) as exc:
        return {"status": "failed", "error_code": "search_failed", "error": str(exc), "results": []}


def memory_code_symbol_impl(
    root: str,
    name: str,
    *,
    namespace: str = "default",
    kind: str | None = None,
    path: str | None = None,
) -> dict[str, object]:
    """Look up explicit code-index symbols."""

    try:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            return {"status": "failed", "error_code": "invalid_root", "error": f"{root} is not a directory"}
        store = _indexed_store_for(root_path, namespace)
        results = lookup_symbols(store, namespace=namespace, name=name, kind=kind, path=path)
        return {
            "status": "ok",
            "root": str(root_path),
            "namespace": namespace,
            "results": [cast("dict[str, object]", result.model_dump(mode="json")) for result in results],
        }
    except (OSError, ValueError) as exc:
        return {"status": "failed", "error_code": "symbol_failed", "error": str(exc), "results": []}


def register_code_index_tools(mcp: McpServer) -> None:
    @mcp.tool()
    async def memory_code_index(root: str, namespace: str = "default") -> dict[str, object]:
        """Index code into the explicit trw-memory code index."""

        return memory_code_index_impl(root, namespace=namespace)

    @mcp.tool()
    async def memory_code_search(
        root: str,
        query: str,
        namespace: str = "default",
        path_glob: str | None = None,
        language: str | None = None,
        limit: int = 10,
    ) -> dict[str, object]:
        """Search explicit code-index chunks with bounded snippets."""

        return memory_code_search_impl(
            root,
            query,
            namespace=namespace,
            path_glob=path_glob,
            language=language,
            limit=limit,
        )

    @mcp.tool()
    async def memory_code_symbol(
        root: str,
        name: str,
        namespace: str = "default",
        kind: str | None = None,
        path: str | None = None,
    ) -> dict[str, object]:
        """Look up symbols in the explicit code index."""

        return memory_code_symbol_impl(root, name, namespace=namespace, kind=kind, path=path)


def _store_for(root: Path, namespace: str) -> InMemoryCodeIndex:
    return _STORES.setdefault((str(root), namespace), InMemoryCodeIndex())


def _indexed_store_for(root: Path, namespace: str) -> InMemoryCodeIndex:
    """Return the namespaced store, populating it on first use."""
    store = _store_for(root, namespace)
    if not store.list_files(namespace=namespace):
        CodeIndexer(root=root, store=store, namespace=namespace).index()
    return store
