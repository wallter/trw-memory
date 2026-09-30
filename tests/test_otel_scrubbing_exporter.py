"""ScrubbingSpanExporter: no exception text reaches any exporter (PRD-CORE-344 FR05)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from tests._otel_support import assert_no_canary
from trw_memory import otel_setup

CANARY = "CANARY-7f3e-secret-text"


class _Recorder:
    def __init__(self) -> None:
        self.batches: list[Sequence[ReadableSpan]] = []
        self.shutdowns = 0
        self.flushes: list[int] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.batches.append(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.shutdowns += 1

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self.flushes.append(timeout_millis)
        return True


def _raising_span() -> ReadableSpan:
    mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    tracer = provider.get_tracer("fastmcp")
    with pytest.raises(ValueError), tracer.start_as_current_span("tools/call boom") as span:
        span.set_attribute("error.type", "ValueError")
        span.add_event("kept", {"k": 1, "exception.message": CANARY})
        span.set_attribute("exception.stacktrace", CANARY)
        raise ValueError(CANARY)
    (finished,) = mem.get_finished_spans()
    assert any(e.name == "exception" for e in finished.events)
    assert CANARY in (finished.status.description or "")
    return finished


def test_exception_events_and_status_description_are_removed() -> None:
    rec = _Recorder()
    original = _raising_span()
    assert otel_setup.ScrubbingSpanExporter(rec).export([original]) is SpanExportResult.SUCCESS
    (out,) = rec.batches[0]
    assert_no_canary([out], CANARY)
    assert [e.name for e in out.events] == ["kept"]
    assert dict(out.events[0].attributes or {}) == {"k": 1}
    assert "exception.stacktrace" not in (out.attributes or {})
    assert out.status.status_code is StatusCode.ERROR
    assert out.status.description is None
    assert out.attributes["error.type"] == "ValueError"
    assert out.context == original.context
    assert (out.name, out.start_time, out.end_time, out.kind) == (
        original.name,
        original.start_time,
        original.end_time,
        original.kind,
    )
    assert out.instrumentation_scope == original.instrumentation_scope


def test_scrub_fault_returns_failure_without_delegating(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder()

    def boom(_span: ReadableSpan) -> ReadableSpan:
        raise RuntimeError("x")

    monkeypatch.setattr(otel_setup, "_scrub", boom)
    assert otel_setup.ScrubbingSpanExporter(rec).export([_raising_span()]) is SpanExportResult.FAILURE
    assert rec.batches == []


def test_shutdown_and_force_flush_delegate() -> None:
    rec = _Recorder()
    scrubber = otel_setup.ScrubbingSpanExporter(rec)
    assert scrubber.force_flush(123) is True
    scrubber.shutdown()
    assert (rec.flushes, rec.shutdowns) == ([123], 1)


def test_scrubbed_spans_encode_through_otlp_and_json(tmp_path: Path) -> None:
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    path = tmp_path / "traces-x-1-20260929.jsonl"
    scrubber = otel_setup.ScrubbingSpanExporter(otel_setup.OtlpJsonFileExporter(path))
    assert scrubber.export([_raising_span()]) is SpanExportResult.SUCCESS
    request = encode_spans([otel_setup._scrub(_raising_span())])
    assert request.resource_spans[0].scope_spans[0].spans[0].status.message == ""
    text = path.read_text()
    assert CANARY not in text
    assert '"code":2' in text


def test_link_attributes_are_scrubbed_like_span_and_event_attributes() -> None:
    """A Link carrying exception.* (a third party put it there) must not reach any exporter."""
    from opentelemetry.trace import Link

    original = _raising_span()
    linked = ReadableSpan(
        name="linked",
        context=original.context,
        attributes={"k": 1},
        links=(Link(original.context, {"exception.message": CANARY, "keep": "yes"}),),
    )
    rec = _Recorder()

    assert otel_setup.ScrubbingSpanExporter(rec).export([linked]) is SpanExportResult.SUCCESS

    (out,) = rec.batches[0]
    assert_no_canary([out], CANARY)
    assert dict(out.links[0].attributes or {}) == {"keep": "yes"}
    assert out.links[0].context == original.context
    assert CANARY not in otel_setup.encode_otlp_json_line([out]).decode()
