"""FastMCP server entry point for trw-memory.

Registers MCP tools and exposes a ``main()`` callable for the
``trw-memory-server`` console script defined in pyproject.toml.

Two live modes, selected by the first argument (PRD-CORE-253 FR03):

``serve stdio`` (the default)
    Today's behaviour -- one server process per MCP client, spoken over
    standard input and output. This is what a client that spawns the server
    itself uses.
``serve http``
    The loopback daemon. One process per OS user serves every consumer of that
    user's store over ``streamable-http`` on 127.0.0.1 with a per-user bearer
    token, an ephemeral port published in a 0600 discovery file, a
    single-instance claim and an idle shutdown. See :mod:`trw_memory.daemon`.

Both modes are reachable and both serve the same registered tools, which is
what makes this a mode selector rather than a dormant flag: neither is a
disabled path waiting for a switch.

fastmcp is an optional dependency (``pip install trw-memory[mcp]``).
Importing this module without fastmcp installed will raise ImportError.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from fastmcp import FastMCP

mcp = FastMCP("trw-memory")

#: The two live transports. ``stdio`` stays the default so a client that spawns
#: the server itself is unaffected by the daemon's arrival.
SERVE_MODES = ("stdio", "http")

#: The published tool surface, in one place. Both transports serve exactly this
#: set, and ``tests/test_server.py`` asserts the live registry equals it in both
#: directions -- so adding a tool is one edit here rather than a hunt through
#: test files, and a tool that appears without an edit fails loudly.
REGISTERED_TOOL_NAMES: tuple[str, ...] = (
    "memory_audit",
    "memory_code_index",
    "memory_code_search",
    "memory_code_symbol",
    "memory_consolidate",
    "memory_forget",
    "memory_namespace_diagnose",
    "memory_namespace_merge",
    "memory_namespace_rename",
    "memory_quarantine_list",
    "memory_recall",
    "memory_review",
    "memory_search",
    "memory_status",
    "memory_store",
    "memory_wiki_lint",
)


def _register_tools() -> None:
    from trw_memory.tools.audit import register_audit_tool
    from trw_memory.tools.code_index import register_code_index_tools
    from trw_memory.tools.consolidate import register_consolidate_tool
    from trw_memory.tools.forget import register_forget_tool
    from trw_memory.tools.namespace_admin import register_namespace_admin_tools
    from trw_memory.tools.recall import register_recall_tool
    from trw_memory.tools.review import register_quarantine_list_tool, register_review_tool
    from trw_memory.tools.search import register_search_tool
    from trw_memory.tools.status import register_status_tool
    from trw_memory.tools.store import register_store_tool
    from trw_memory.tools.wiki_lint import register_wiki_lint_tool

    register_store_tool(mcp)
    register_recall_tool(mcp)
    register_audit_tool(mcp)
    register_review_tool(mcp)
    register_quarantine_list_tool(mcp)
    register_namespace_admin_tools(mcp)
    register_forget_tool(mcp)
    register_consolidate_tool(mcp)
    register_search_tool(mcp)
    register_status_tool(mcp)
    register_wiki_lint_tool(mcp)
    register_code_index_tools(mcp)


_register_tools()


def _preflight(config: object) -> None:
    """Fail before serving when a required runtime dependency is missing."""
    from trw_memory.embeddings import get_local_embedder
    from trw_memory.storage.sqlite_backend import _import_sqlcipher_driver

    if getattr(config, "encryption_enabled", False):
        _import_sqlcipher_driver()
    if getattr(config, "local_only", False):
        get_local_embedder(
            model_name=getattr(config, "embedding_model", ""),
            dim=getattr(config, "embedding_dim", 0),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trw-memory-server", description=__doc__)
    subcommands = parser.add_subparsers(dest="command")
    serve = subcommands.add_parser("serve", help="Serve the memory tool surface")
    serve.add_argument(
        "mode",
        nargs="?",
        default="stdio",
        choices=SERVE_MODES,
        help="stdio (default) for one process per client, http for the loopback daemon",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=None,
        help="Loopback port for http mode; omit to use memory_daemon_port (0 = ephemeral)",
    )
    serve.add_argument(
        "--idle-shutdown-seconds",
        type=float,
        default=None,
        help=(
            "Seconds without a request before an http daemon exits. Omit to use "
            "memory_daemon_idle_shutdown_seconds. An explicit value is an operator's "
            "deliberate short-lived daemon and is not bound by that field's 60s floor."
        ),
    )
    return parser


def _serve_http(port: int | None, idle_shutdown_seconds: float | None) -> None:
    """Run the loopback daemon, exiting quietly if one is already running."""
    from trw_memory.daemon._serve import DaemonServeOptions, serve_loopback
    from trw_memory.exceptions import ConfigError, DaemonAlreadyRunningError
    from trw_memory.models.config import MemoryConfig

    config = MemoryConfig()
    _preflight(config)
    if config.encryption_enabled:
        # The daemon pins memory_single_store_path, and that combination is
        # refused (PRD-CORE-253 FR09): SQLCipher keys a file while this package
        # derives a per-namespace key, so every namespace after the first could
        # not decrypt the shared store. Refusing HERE, before the pin, turns a
        # confusing second-namespace decrypt failure inside a served tool call
        # into one startup error that names the reason.
        raise ConfigError(
            "refusing to start the loopback daemon with encryption_enabled: the daemon serves every "
            "namespace from one file, and this package derives a per-namespace SQLCipher key, so only "
            "the first namespace could decrypt it. Single-file encryption keys are PRD-CORE-253 FR09. "
            "Run `trw-memory-server serve stdio` for an encrypted per-namespace store, or disable "
            "encryption for the daemon."
        )
    options = DaemonServeOptions.from_config(config, port=port, idle_shutdown_seconds=idle_shutdown_seconds)
    try:
        asyncio.run(serve_loopback(options))
    except DaemonAlreadyRunningError as exc:
        # Not a failure: this is the second start declining to bind. Reported
        # on stderr and exited 0 so an auto-start race is not an error either.
        print(str(exc), file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point for ``trw-memory-server``."""
    from trw_memory.models.config import MemoryConfig

    args = _build_parser().parse_args(argv)
    if getattr(args, "mode", "stdio") == "http":
        _serve_http(args.port, args.idle_shutdown_seconds)
        return
    _preflight(MemoryConfig())
    mcp.run()


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
