"""Wave 14: coverage gap-fill for tools/wiki_lint.py (line 19) and wiki/lint.py (lines 156-158)."""

from __future__ import annotations

from unittest.mock import MagicMock

from trw_memory.tools.wiki_lint import memory_wiki_lint_impl, register_wiki_lint_tool


class TestRegisterWikiLintTool:
    async def test_registered_function_delegates_to_impl(self) -> None:
        """register_wiki_lint_tool wires memory_wiki_lint to impl (line 19)."""
        registered: dict[str, object] = {}
        mock_mcp = MagicMock()

        def _capture(f):
            registered["fn"] = f
            return f

        mock_mcp.tool.return_value = _capture
        register_wiki_lint_tool(mock_mcp)

        assert "fn" in registered

        result = await registered["fn"]([], 20)  # type: ignore[operator]

        assert isinstance(result, dict)
        assert "total" in result or "findings_count" in result or "status" in result


class TestWikiLintCoercePagesValidationError:
    def test_invalid_page_dict_produces_lint_finding(self) -> None:
        """WikiPage.model_validate fails on bad dict → ValidationError → finding (lines 156-158)."""
        # A dict missing required fields (kind, title) should fail model_validate
        invalid_page: dict[str, object] = {
            "slug": "topic/bad-page",
            # Missing required 'kind' and 'title' fields
        }

        result = memory_wiki_lint_impl([invalid_page], top_limit=20)

        findings_list = list(result.get("findings", result.get("top_findings", [])))  # type: ignore[arg-type]
        assert len(findings_list) >= 1 or int(str(result.get("total", 0))) >= 1
        assert any("invalid_slug" in str(f.get("code", "")) for f in findings_list)

    def test_page_with_invalid_slug_format_produces_validation_finding(self) -> None:
        """Page with invalid slug triggers ValidationError finding."""
        invalid_page: dict[str, object] = {
            "kind": "topic",
            "slug": "INVALID SLUG!!",
            "title": "Bad Page",
        }

        result = memory_wiki_lint_impl([invalid_page], top_limit=20)

        assert int(str(result.get("total", result.get("findings_count", 0)))) >= 1
