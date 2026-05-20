"""MCP tool: memory_wiki_lint — validate wiki page references."""

from __future__ import annotations

from trw_memory.tools._types import McpServer
from trw_memory.wiki.lint import lint_wiki_pages, summarize_lint


def memory_wiki_lint_impl(pages: list[dict[str, object]], *, top_limit: int = 20) -> dict[str, object]:
    report = lint_wiki_pages(pages, top_limit=top_limit)
    return summarize_lint(report, top_limit=top_limit)


def register_wiki_lint_tool(mcp: McpServer) -> None:
    @mcp.tool()
    async def memory_wiki_lint(pages: list[dict[str, object]], top_limit: int = 20) -> dict[str, object]:
        """Lint explicit wiki pages for missing targets, backlinks, and provenance gaps."""

        return memory_wiki_lint_impl(pages, top_limit=top_limit)
