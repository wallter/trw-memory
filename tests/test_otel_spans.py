"""PRD-CORE-343 FR03/FR04/FR05/NFR03: memory spans from both entry families on a real SQLite store."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests import test_otel_emitter as _emitter
from tests._otel_support import assert_no_canary
from trw_memory import _otel_keys as k
from trw_memory.models.config import MemoryConfig

mem_spans = _emitter.mem_spans  # the fixture, registered in this module by name
CANARY = "CANARY-5e1d-do-not-export"
NS = "project:default"


@pytest.fixture()
def tools_env(tmp_path: Path) -> Iterator[tuple[MemoryConfig, Any]]:
    from trw_memory.integrations._backend import create_backend_from_config

    cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path), embeddings_enabled=False)
    with create_backend_from_config(cfg, NS) as backend:
        yield cfg, backend


def _attrs(exporter: InMemorySpanExporter) -> list[tuple[str, dict[str, Any]]]:
    return [(s.name, dict(s.attributes or {})) for s in exporter.get_finished_spans()]


class TestToolFamily:
    def test_create_then_engine_decided_update(self, tools_env: Any, mem_spans: InMemorySpanExporter) -> None:
        from trw_memory.tools.store import memory_store_impl

        cfg, backend = tools_env
        first = memory_store_impl(f"fact {CANARY}", NS, backend=backend, config=cfg, tags=[CANARY])
        memory_store_impl("fact two", NS, backend=backend, config=cfg, entry_id=first["memory_id"])
        memory_store_impl("fact three", NS, backend=backend, config=cfg, entry_id="M-brandnew1")
        (n1, a1), (n2, a2), (n3, a3) = _attrs(mem_spans)
        assert (n1, a1[k.WRITE_OUTCOME], a1[k.RECORD_ID], a1[k.RECORD_COUNT]) == (
            "create_memory",
            "created",
            first["memory_id"],
            1,
        )
        assert (n2, a2[k.WRITE_OUTCOME]) == ("upsert_memory", "updated")
        assert (n3, a3[k.WRITE_OUTCOME]) == ("upsert_memory", "created")  # outcome from the engine, not the id
        assert a1[k.STORE_ID] == NS
        assert_no_canary(mem_spans.get_finished_spans(), CANARY)

    def test_refused_write_has_error_type_and_no_outcome(self, tools_env: Any, mem_spans: InMemorySpanExporter) -> None:
        from trw_memory.tools.store import memory_store_impl

        cfg, backend = tools_env
        assert memory_store_impl("x", "bad namespace!", backend=backend, config=cfg)["status"] == "invalid"
        ((_name, attrs),) = _attrs(mem_spans)
        assert attrs[k.ERROR_TYPE] == "invalid"
        assert k.WRITE_OUTCOME not in attrs
        assert k.STORE_ID not in attrs  # fails the ID pattern: omitted

    def test_update_and_both_delete_targets(self, tools_env: Any, mem_spans: InMemorySpanExporter) -> None:
        from trw_memory.lifecycle.correction import LearningPatch
        from trw_memory.tools.forget import memory_forget_impl
        from trw_memory.tools.store import memory_store_impl
        from trw_memory.tools.update import memory_update_impl

        cfg, backend = tools_env
        mid = memory_store_impl(f"doomed {CANARY}", NS, backend=backend, config=cfg)["memory_id"]
        memory_update_impl(mid, LearningPatch(status="obsolete"), NS, backend=backend, config=cfg)
        memory_forget_impl(None, CANARY, NS, backend=backend, config=cfg, actor=CANARY)
        memory_forget_impl(mid, None, NS, backend=backend, config=cfg)
        spans = _attrs(mem_spans)[1:]
        assert [n for n, _ in spans] == ["update_memory", "delete_memory", "delete_memory"]
        assert spans[0][1][k.WRITE_OUTCOME] == "updated"
        assert spans[1][1][k.DELETE_TARGET] == "selector"
        assert k.RECORD_ID not in spans[1][1]
        assert (spans[2][1][k.DELETE_TARGET], spans[2][1][k.RECORD_ID]) == ("record", mid)
        assert_no_canary(mem_spans.get_finished_spans(), CANARY)

    def test_recall_threshold_and_empty_recall(self, tools_env: Any, mem_spans: InMemorySpanExporter) -> None:
        from trw_memory.tools.recall import memory_recall_impl
        from trw_memory.tools.store import memory_store_impl

        cfg, backend = tools_env
        for i in range(3):
            memory_store_impl(f"pytest fixture scoping rule {i}", NS, backend=backend, config=cfg)
        mem_spans.clear()
        memory_recall_impl(f"fixture {CANARY}", NS, backend=backend, config=cfg, limit=5, min_score=0.9)
        memory_recall_impl("zzzqqq", NS, backend=backend, config=cfg, limit=5, graph_depth=1)
        (_n1, hit), (_n2, empty) = _attrs(mem_spans)
        assert hit[k.RECALL_METHOD] == "keyword"  # embeddings disabled: the keyword fallback
        assert hit[k.RECALL_SCORE_KIND] == "other"
        assert k.RECALL_THRESHOLD not in hit  # no direction on an "other" scale (MEM-13.3)
        assert hit[k.TOP_K] == 5
        assert empty[k.RECORD_COUNT] == 0
        assert k.TOP_K not in empty  # graph_depth > 0
        assert all(s.status.status_code.name != "ERROR" for s in mem_spans.get_finished_spans())
        assert_no_canary(mem_spans.get_finished_spans(), CANARY)


class TestClientFamily:
    async def test_client_store_recall_forget_and_nested_bulk(
        self, client: Any, mem_spans: InMemorySpanExporter
    ) -> None:
        stored = await client.store(f"client fact {CANARY}", tags=[CANARY])
        await client.store_many([{"content": "bulk a"}, {"content": "bulk b"}])
        await client.recall(f"client {CANARY}", limit=3)
        await client.forget(stored["memory_id"])
        names = [n for n, _ in _attrs(mem_spans)]
        assert names == ["create_memory", "create_memory", "search_memory", "delete_memory"]
        many = _attrs(mem_spans)[1][1]
        assert (many[k.RECORD_COUNT], many[k.WRITE_OUTCOME]) == (2, "created")  # one span, not a nested bulk span
        assert _attrs(mem_spans)[0][1][k.STORE_ID] == "default"
        assert_no_canary(mem_spans.get_finished_spans(), CANARY)

    async def test_client_host_exception_is_reraised(self, client: Any, mem_spans: InMemorySpanExporter) -> None:
        from trw_memory.exceptions import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError):
            await client.forget("M-missing")
        ((_n, attrs),) = _attrs(mem_spans)
        assert attrs[k.ERROR_TYPE] == "not_found"
