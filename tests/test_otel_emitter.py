"""PRD-CORE-343 FR01/FR07/FR08/NFR01/NFR02/NFR05: the memory-span emitter, unit level."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests._otel_support import assert_keys_registered, assert_no_unkeyed_digest
from trw_memory import _otel
from trw_memory import _otel_keys as k


def assert_memory_conformance(spans: list[ReadableSpan]) -> None:
    """``trw_memory`` spans use only ``_otel_keys`` keys, types and closed enums (NFR05), no events or text."""
    mine = [s for s in spans if s.instrumentation_scope is not None and s.instrumentation_scope.name == "trw_memory"]
    assert_keys_registered(mine, k.REGISTRY)
    assert_no_unkeyed_digest(mine)
    for span in mine:
        assert span.name in k.OPERATIONS
        for key, value in (span.attributes or {}).items():
            typ, enum = k.REGISTRY[key]
            assert type(value) is typ, (key, value)
            assert enum is None or value in enum, (key, value)
        assert not span.events
        assert not span.status.description


@pytest.fixture()
def mem_spans(otel_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """The shared session exporter (PRD-CORE-342 FR09), with the registry check on teardown."""
    monkeypatch.setattr(_otel, "_SDK_DISABLED", False)
    yield otel_spans
    assert_memory_conformance(list(otel_spans.get_finished_spans()))


@_otel.memory_op("store")
def _fake_store(content: str, namespace: str, *, entry_id: str | None = None, status: str = "stored") -> dict[str, Any]:
    return {"memory_id": entry_id or "M-new", "namespace": namespace, "status": status}


@_otel.memory_op("forget")
def _fake_forget(memory_id: str | None, query: str | None, namespace: str) -> dict[str, Any]:
    raise KeyError(f"nothing for {query}")


class TestScopeAndSwitch:
    def test_tracer_scope_version_and_semconv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trw_memory._version import __version__

        seen: list[tuple[object, ...]] = []
        monkeypatch.setattr(trace, "get_tracer", lambda *a, **kw: seen.append((*a, kw)) or "tracer")
        assert _otel._probe() == "tracer"
        assert seen == [("trw_memory", __version__, {"attributes": {k.SCOPE_SEMCONV_VERSION: k.SEMCONV_VERSION}})]

    def test_span_is_internal_named_by_operation(self, mem_spans: InMemorySpanExporter) -> None:
        _fake_store("c", "project:x")
        (span,) = mem_spans.get_finished_spans()
        assert span.name == "create_memory"
        assert span.kind is trace.SpanKind.INTERNAL
        assert span.instrumentation_scope is not None

    @pytest.mark.parametrize("state", ["sdk_disabled", "no_tracer", "not_recording"])
    def test_disabled_states_emit_nothing_and_keep_results(
        self, state: str, mem_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if state == "sdk_disabled":
            monkeypatch.setattr(_otel, "_SDK_DISABLED", None)
            monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        elif state == "no_tracer":
            monkeypatch.setattr(_otel, "_TRACER", None)  # below-floor or missing API
        else:
            monkeypatch.setattr(_otel, "_TRACER", trace.NoOpTracer())
        spy: list[object] = []
        monkeypatch.setattr(_otel, "clean_id", lambda v: spy.append(v))
        assert _fake_store("c", "project:x", entry_id="M-1") == {
            "memory_id": "M-1",
            "namespace": "project:x",
            "status": "stored",
        }
        assert mem_spans.get_finished_spans() == ()
        assert spy == []  # nothing marshaled (NFR01)

    def test_below_floor_probe_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.metadata

        monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.20.0")
        assert _otel._probe() is None


class TestFaults:
    def test_host_exception_is_same_object_with_error_type_only(self, mem_spans: InMemorySpanExporter) -> None:
        with pytest.raises(KeyError) as info:
            _fake_forget(None, "CANARY-q", "project:x")
        (span,) = mem_spans.get_finished_spans()
        assert span.attributes == {k.OPERATION_NAME: "delete_memory", k.ERROR_TYPE: "_OTHER"}
        assert span.status.status_code is trace.StatusCode.ERROR
        assert "CANARY-q" in str(info.value)

    def test_raising_setter_never_changes_the_result(
        self, mem_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("telemetry fault")

        monkeypatch.setattr(_otel, "_describe", boom)
        assert _fake_store("c", "project:x")["status"] == "stored"
        assert len(mem_spans.get_finished_spans()) == 1

    def test_raising_tracer_never_changes_the_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Broken:
            def start_span(self, *_a: object, **_k: object) -> None:
                raise RuntimeError("tracer fault")

        monkeypatch.setattr(_otel, "_TRACER", Broken())
        monkeypatch.setattr(_otel, "_SDK_DISABLED", False)
        assert _fake_store("c", "project:x")["status"] == "stored"

    def test_nested_operation_emits_one_span(self, mem_spans: InMemorySpanExporter) -> None:
        @_otel.memory_op("store_many")
        def outer(entries: list[dict[str, str]]) -> int:
            return sum(_fake_store(e["content"], "project:x")["status"] == "stored" for e in entries)

        assert outer([{"content": "a"}, {"content": "b"}]) == 2
        (span,) = mem_spans.get_finished_spans()
        assert span.name == "create_memory"
        assert span.attributes is not None
        assert span.attributes[k.RECORD_COUNT] == 2
        assert span.attributes[k.WRITE_OUTCOME] == "created"


class TestIds:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("L-abc123", "L-abc123"),
            ("M-1#hype2", "M-1"),
            (42, "42"),
            ("x" * 129, None),
            ("x" * 5000, None),
            ("bad id with spaces", None),
            ("idh1:abc", None),
            ("M-\ud800", None),
            (True, None),
            (1.5, None),
            (1 << 64, None),
            (None, None),
        ],
    )
    def test_clean_id(self, value: object, expected: str | None) -> None:
        assert _otel.clean_id(value) == expected

    def test_export_store_id_is_identity_locally(self) -> None:
        assert _otel.export_store_id("project:default") == "project:default"

    def test_ids_are_deduplicated_then_capped(self, mem_spans: InMemorySpanExporter) -> None:
        @_otel.memory_op("search")
        def many(namespace: str, limit: int) -> list[dict[str, object]]:
            return [{"id": f"L-{i}"} for i in range(limit)] + [{"id": "L-0"}]

        many("project:x", 10_000)
        (span,) = mem_spans.get_finished_spans()
        attrs = span.attributes or {}
        assert attrs[k.RECORD_COUNT] == 10_001
        assert len(attrs[k.RECORD_IDS]) == _otel.ARRAY_CAP  # type: ignore[arg-type]
        assert attrs[k.RECORD_TRUNCATED] is True

    def test_sanitizer_work_is_bounded(self, mem_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[object] = []
        real = _otel.clean_id
        monkeypatch.setattr(_otel, "clean_id", lambda v: calls.append(v) or real(v))

        @_otel.memory_op("search")
        def many(namespace: str, limit: int) -> list[dict[str, object]]:
            return [{"id": f"L-{i}"} for i in range(limit)]

        many("project:x", 10_000)
        assert len(calls) <= 1 + _otel.ARRAY_CAP + 1  # store.id plus at most 33 ids


class TestPropagation:
    def test_direct_path_injects_traceparent_only(self, mem_spans: InMemorySpanExporter) -> None:
        from trw_memory.daemon._direct import _inject_trace_context

        params: dict[str, Any] = {"name": "memory_recall", "arguments": {}}
        _inject_trace_context(params)
        assert "_meta" not in params  # no current span: nothing injected
        with trace.get_tracer("client").start_as_current_span("caller") as caller:
            _inject_trace_context(params)
            trace_id = format(caller.get_span_context().trace_id, "032x")
        assert list(params["_meta"]) == ["traceparent"]
        assert params["_meta"]["traceparent"].split("-")[1] == trace_id
