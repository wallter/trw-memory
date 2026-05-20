"""trw-memory CLI — command-line interface for the trw-memory package."""

from __future__ import annotations

import argparse
import asyncio
import functools
import inspect
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import ParamSpec, TypeVar, cast, overload

import structlog

from trw_memory.cli_client import handle_forget, handle_recall, handle_search, handle_store
from trw_memory.cli_formatters import (
    StatusDict,
    entry_to_export_dict,
    format_export_summary,
    format_import_summary,
    format_results,
    format_status,
    format_store_result,
)
from trw_memory.cli_parser import build_parser
from trw_memory.cli_storage import (
    handle_consolidate,
    handle_export,
    handle_import,
    handle_restore,
    handle_snapshot,
    handle_status,
)
from trw_memory.client import MemoryClient, _create_local_backend
from trw_memory.embeddings import get_local_embedder
from trw_memory.lifecycle.consolidation import consolidate_cycle
from trw_memory.models.config import MemoryConfig
from trw_memory.tools.code_index import memory_code_index_impl, memory_code_search_impl, memory_code_symbol_impl
from trw_memory.tools.wiki_lint import memory_wiki_lint_impl

__all__ = ["main"]

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


@overload
def _cli_error_boundary(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]: ...


@overload
def _cli_error_boundary(fn: Callable[P, R]) -> Callable[P, R]: ...


def _cli_error_boundary(fn: Callable[P, object]) -> Callable[P, object]:
    async_fn = cast("Callable[P, Awaitable[object]]", fn)
    sync_fn = fn

    @functools.wraps(fn)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
        try:
            return await async_fn(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("cli_command_failed", command=fn.__name__, error=str(exc), exc_info=True)
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    @functools.wraps(fn)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
        try:
            return sync_fn(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("cli_command_failed", command=fn.__name__, error=str(exc), exc_info=True)
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    if asyncio.iscoroutinefunction(fn):
        return cast("Callable[P, object]", async_wrapper)
    return cast("Callable[P, object]", sync_wrapper)


@_cli_error_boundary
async def _handle_store(args: argparse.Namespace) -> int:
    return await handle_store(
        args,
        client_cls=MemoryClient,
        format_store_result=format_store_result,
    )


@_cli_error_boundary
async def _handle_recall(args: argparse.Namespace) -> int:
    return await handle_recall(args, client_cls=MemoryClient, format_results=format_results)


@_cli_error_boundary
async def _handle_search(args: argparse.Namespace) -> int:
    return await handle_search(args, client_cls=MemoryClient, format_results=format_results)


@_cli_error_boundary
def _handle_consolidate(args: argparse.Namespace) -> int:
    return handle_consolidate(
        args,
        config_cls=MemoryConfig,
        backend_factory=_create_local_backend,
        embedder_factory=get_local_embedder,
        consolidate_fn=consolidate_cycle,
    )


@_cli_error_boundary
def _handle_export(args: argparse.Namespace) -> int:
    return handle_export(
        args,
        config_cls=MemoryConfig,
        backend_factory=_create_local_backend,
        entry_to_export_dict=entry_to_export_dict,
        export_summary=format_export_summary,
    )


@_cli_error_boundary
def _handle_import(args: argparse.Namespace) -> int:
    from trw_memory.integrations._backend import make_entry

    return handle_import(
        args,
        config_cls=MemoryConfig,
        backend_factory=_create_local_backend,
        make_entry=make_entry,
        import_summary=format_import_summary,
    )


@_cli_error_boundary
def _handle_status(args: argparse.Namespace) -> int:
    return handle_status(
        args,
        config_cls=MemoryConfig,
        backend_factory=_create_local_backend,
        status_dict_cls=StatusDict,
        format_status=format_status,
    )


@_cli_error_boundary
async def _handle_forget(args: argparse.Namespace) -> int:
    return await handle_forget(args, client_cls=MemoryClient)


@_cli_error_boundary
def _handle_restore(args: argparse.Namespace) -> int:
    return handle_restore(args, config_cls=MemoryConfig)


@_cli_error_boundary
def _handle_snapshot(args: argparse.Namespace) -> int:
    return handle_snapshot(args, config_cls=MemoryConfig)


@_cli_error_boundary
def _handle_wiki_lint(args: argparse.Namespace) -> int:
    raw_pages = json.loads(Path(args.path).read_text(encoding="utf-8"))
    if not isinstance(raw_pages, list):
        raise TypeError("wiki-lint input must be a JSON list")
    pages: list[dict[str, object]] = []
    for index, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            raise TypeError(f"wiki-lint item {index} must be an object")
        pages.append({str(key): value for key, value in raw_page.items()})
    print(json.dumps(memory_wiki_lint_impl(pages, top_limit=args.top_limit), sort_keys=True))
    return 0


@_cli_error_boundary
def _handle_code_index(args: argparse.Namespace) -> int:
    print(json.dumps(memory_code_index_impl(args.root, namespace=args.namespace), sort_keys=True))
    return 0


@_cli_error_boundary
def _handle_code_search(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            memory_code_search_impl(
                args.root,
                args.query,
                namespace=args.namespace,
                path_glob=args.path_glob,
                language=args.language,
                limit=args.limit,
            ),
            sort_keys=True,
        )
    )
    return 0


@_cli_error_boundary
def _handle_code_symbol(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            memory_code_symbol_impl(
                args.root,
                args.name,
                namespace=args.namespace,
                kind=args.kind,
                path=args.path,
            ),
            sort_keys=True,
        )
    )
    return 0


async def _dispatch(args: argparse.Namespace) -> int:
    handlers: dict[str, Callable[..., object]] = {
        "store": _handle_store,
        "recall": _handle_recall,
        "search": _handle_search,
        "consolidate": _handle_consolidate,
        "export": _handle_export,
        "import": _handle_import,
        "status": _handle_status,
        "forget": _handle_forget,
        "restore": _handle_restore,
        "snapshot": _handle_snapshot,
        "wiki-lint": _handle_wiki_lint,
        "code-index": _handle_code_index,
        "code-search": _handle_code_search,
        "code-symbol": _handle_code_symbol,
    }
    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        logger.error("cli_unknown_command", command=args.command)
        return 1
    try:
        coro = handler(args)
        rc = await coro if inspect.isawaitable(coro) else coro
        return rc if isinstance(rc, int) else 1
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def main(argv: list[str] | None = None) -> int:
    from trw_memory._logging import configure_logging

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    verbosity = args.verbose
    if args.quiet:
        verbosity = -1

    configure_logging(verbosity=verbosity, log_level=args.log_level)
    return asyncio.run(_dispatch(args))


if __name__ == "__main__":
    sys.exit(main())
