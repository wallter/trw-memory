"""Symbol lookup APIs for code-index stores."""

from __future__ import annotations

from trw_memory.code_index.indexer import InMemoryCodeIndex
from trw_memory.code_index.models import SymbolLookupResult

__all__ = ["lookup_symbols"]


def lookup_symbols(
    store: InMemoryCodeIndex,
    *,
    namespace: str,
    name: str | None = None,
    kind: str | None = None,
    path: str | None = None,
) -> list[SymbolLookupResult]:
    """Return all matching symbols, including duplicates with disambiguation metadata."""

    return [
        SymbolLookupResult(
            name=symbol.name,
            kind=symbol.kind,
            file=symbol.path,
            path=symbol.path,
            language=symbol.language,
            line_range=(symbol.start_line, symbol.end_line),
            disambiguation=(
                f"{symbol.language}:{symbol.kind}:{symbol.path}:{symbol.start_line}-{symbol.end_line}"
            ),
        )
        for symbol in store.list_symbols(namespace=namespace, name=name, kind=kind, path=path)
    ]
