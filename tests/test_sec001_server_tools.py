from __future__ import annotations

from trw_memory.tools.audit import register_audit_tool
from trw_memory.tools.review import register_review_tool


class _FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):  # type: ignore[no-untyped-def]
        def decorator(fn):  # type: ignore[no-untyped-def]
            self.tools[fn.__name__] = fn
            return fn

        return decorator


def test_sec001_public_tools_register() -> None:
    mcp = _FakeMCP()
    register_audit_tool(mcp)
    register_review_tool(mcp)
    assert "memory_audit" in mcp.tools
    assert "memory_review" in mcp.tools
