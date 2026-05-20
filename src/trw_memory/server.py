"""FastMCP server entry point for trw-memory.

Registers MCP tools and exposes a ``main()`` callable for the
``trw-memory`` console script defined in pyproject.toml.

fastmcp is an optional dependency (``pip install trw-memory[mcp]``).
Importing this module without fastmcp installed will raise ImportError.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("trw-memory")


def _register_tools() -> None:
    from trw_memory.tools.audit import register_audit_tool
    from trw_memory.tools.consolidate import register_consolidate_tool
    from trw_memory.tools.forget import register_forget_tool
    from trw_memory.tools.recall import register_recall_tool
    from trw_memory.tools.review import register_review_tool
    from trw_memory.tools.search import register_search_tool
    from trw_memory.tools.status import register_status_tool
    from trw_memory.tools.store import register_store_tool
    from trw_memory.tools.wiki_lint import register_wiki_lint_tool

    register_store_tool(mcp)
    register_recall_tool(mcp)
    register_audit_tool(mcp)
    register_review_tool(mcp)
    register_forget_tool(mcp)
    register_consolidate_tool(mcp)
    register_search_tool(mcp)
    register_status_tool(mcp)
    register_wiki_lint_tool(mcp)


_register_tools()


def main() -> None:
    from trw_memory.embeddings import get_local_embedder
    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import _import_sqlcipher_driver

    cfg = MemoryConfig()
    if cfg.encryption_enabled:
        _import_sqlcipher_driver()
    if cfg.local_only:
        get_local_embedder(model_name=cfg.embedding_model, dim=cfg.embedding_dim)
    mcp.run()
