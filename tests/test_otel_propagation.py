"""PRD-CORE-343 FR08/FR10: trace context crosses the daemon's direct JSON path; the daemon's setup call site."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("fastmcp")

from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests import test_otel_emitter as _emitter

mem_spans = _emitter.mem_spans  # the fixture, registered in this module by name


async def test_direct_path_memory_span_is_a_descendant_of_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mem_spans: InMemorySpanExporter
) -> None:
    """FastMCP extracts ``_meta.traceparent`` on the stateless JSON path, so no daemon-side fallback exists."""
    import httpx
    from fastmcp import FastMCP

    from trw_memory.daemon._direct import _inject_trace_context
    from trw_memory.tools.recall import register_recall_tool

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_EMBEDDINGS_ENABLED", "false")
    mcp = FastMCP("otel-propagation")
    register_recall_tool(mcp)
    app: Any = mcp.http_app(transport="streamable-http", stateless_http=True, json_response=True)
    params: dict[str, Any] = {"name": "memory_recall", "arguments": {"query": "q", "namespace": "project:default"}}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://daemon") as http:
            with trace.get_tracer("client").start_as_current_span("caller") as caller:
                _inject_trace_context(params)
                reply = await http.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params},
                    headers={"Accept": "application/json, text/event-stream"},
                )
    assert reply.status_code == 200
    spans = {s.name: s for s in mem_spans.get_finished_spans()}
    server, memory = spans["tools/call memory_recall"], spans["search_memory"]
    assert server.parent is not None and memory.parent is not None
    assert server.parent.span_id == caller.get_span_context().span_id
    assert memory.parent.span_id == server.context.span_id
    assert memory.context.trace_id == caller.get_span_context().trace_id


@pytest.mark.parametrize(("flag", "enabled"), [(None, False), ("false", False), ("true", True)])
def test_daemon_setup_reads_the_opt_in_and_drops_traceparent(
    flag: str | None, enabled: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory import otel_setup
    from trw_memory.daemon import _serve
    from trw_memory.daemon._paths import DaemonPaths

    calls: list[tuple[str, Path | None, bool]] = []
    monkeypatch.setattr(otel_setup, "configure_tracing", lambda s, d, enabled: calls.append((s, d, enabled)) or enabled)
    monkeypatch.setenv("TRACEPARENT", "00-" + "1" * 32 + "-" + "2" * 16 + "-01")
    if flag is None:
        monkeypatch.delenv("TRW_OTEL_ENABLED", raising=False)
    else:
        monkeypatch.setenv("TRW_OTEL_ENABLED", flag)
    assert _serve._configure_tracing(DaemonPaths(user_memory_dir=tmp_path)) is enabled
    assert calls == [("trw-memory", tmp_path / "traces", enabled)]
    import os

    assert "TRACEPARENT" not in os.environ
