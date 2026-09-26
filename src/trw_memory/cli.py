"""trw-memory CLI — command-line interface for the trw-memory package."""

from __future__ import annotations

import argparse
import asyncio
import functools
import inspect
import sys
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import ParamSpec, TypeVar, cast, overload

import structlog

from trw_memory.cli_client import handle_daemon_verb, handle_reembed
from trw_memory.cli_formatters import (
    format_import_summary,
)
from trw_memory.cli_namespace import handle_namespace
from trw_memory.cli_parser import build_parser
from trw_memory.cli_storage import (
    handle_import,
    handle_restore,
    handle_snapshot,
)
from trw_memory.client import MemoryClient, _create_local_backend
from trw_memory.models.config import MemoryConfig

__all__ = ["main"]

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _run_async(coro: Coroutine[object, object, R]) -> R:
    """Run CLI dispatch without replacing a caller-owned event loop.

    ``asyncio.run`` clears the current thread's dormant policy loop. That is
    harmless in a fresh console process but leaks an embedding host's loop and
    self-pipe sockets when ``main`` is invoked as a library entry point. A
    short-lived worker thread gives dispatch an isolated loop while preserving
    both dormant and running loops owned by the caller.
    """
    results: list[R] = []
    errors: list[BaseException] = []

    def _runner() -> None:
        try:
            results.append(asyncio.run(coro))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_runner, name="trw-memory-cli-dispatch")
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return results[0]


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
            logger.exception("cli_command_failed", command=fn.__name__, error=str(exc))
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    @functools.wraps(fn)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
        try:
            return sync_fn(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            logger.exception("cli_command_failed", command=fn.__name__, error=str(exc))
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    if asyncio.iscoroutinefunction(fn):
        return cast("Callable[P, object]", async_wrapper)
    return cast("Callable[P, object]", sync_wrapper)


@_cli_error_boundary
async def _handle_daemon_verb(args: argparse.Namespace) -> int:
    return await handle_daemon_verb(args)


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
async def _handle_reembed(args: argparse.Namespace) -> int:
    return await handle_reembed(args, client_cls=MemoryClient)


@_cli_error_boundary
def _handle_restore(args: argparse.Namespace) -> int:
    return handle_restore(args, config_cls=MemoryConfig)


@_cli_error_boundary
def _handle_snapshot(args: argparse.Namespace) -> int:
    return handle_snapshot(args, config_cls=MemoryConfig)


@_cli_error_boundary
async def _handle_namespace(args: argparse.Namespace) -> int:
    return await handle_namespace(args)


async def _dispatch(args: argparse.Namespace) -> int:
    handlers: dict[str, Callable[..., object]] = {
        "store": _handle_daemon_verb,
        "recall": _handle_daemon_verb,
        "search": _handle_daemon_verb,
        "consolidate": _handle_daemon_verb,
        "export": _handle_daemon_verb,
        "import": _handle_import,
        "status": _handle_daemon_verb,
        "forget": _handle_daemon_verb,
        "reembed": _handle_reembed,
        "restore": _handle_restore,
        "snapshot": _handle_snapshot,
        "namespace": _handle_namespace,
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
    return _run_async(_dispatch(args))


if __name__ == "__main__":
    sys.exit(main())
