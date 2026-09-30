"""Shared OTel test fixture and privacy helpers (PRD-CORE-342 FR09).

trw-mcp/tests/_otel_support.py and trw-memory/tests/_otel_support.py are byte-identical copies (a test
asserts it): trw-memory tests cannot import trw-mcp's. ``set_tracer_provider`` works once per process,
so one session provider with an in-memory exporter is installed on first use; tests that need "no
provider" run in a subprocess.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_EXPORTER = InMemorySpanExporter()
_HEX64 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def session_exporter() -> InMemorySpanExporter:
    """Install the process-wide in-memory provider once and return its exporter."""
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
        trace.set_tracer_provider(provider)
    return _EXPORTER


@pytest.fixture
def otel_spans() -> Iterator[InMemorySpanExporter]:
    """The session in-memory exporter, cleared before and after the test."""
    exporter = session_exporter()
    exporter.clear()
    yield exporter
    exporter.clear()


def _attr_maps(span: ReadableSpan) -> Iterator[dict[str, Any]]:
    yield dict(span.attributes or {})
    for link in span.links:
        yield dict(link.attributes or {})
    for event in span.events:
        yield dict(event.attributes or {})


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def assert_no_canary(spans: Iterable[ReadableSpan], canary: str) -> None:
    """Fail if ``canary`` is in any attribute, link/event attribute, event name or status description."""
    for span in spans:
        assert canary not in span.name, span.name
        assert canary not in (span.status.description or ""), span.name
        for event in span.events:
            assert canary not in event.name, span.name
        for attrs in _attr_maps(span):
            for key, value in attrs.items():
                for text in _strings(value):
                    assert canary not in text, f"{span.name}: {key}"


def _is_digest(text: str) -> bool:
    return text.startswith("sha256:") or bool(_HEX64.search(text))


def assert_no_unkeyed_digest(spans_or_lines: Iterable[ReadableSpan] | Iterable[str]) -> None:
    """Fail on an attribute value that is ``sha256:``-prefixed or holds a run of exactly 64 hex chars.

    Only attribute values are inspected (span, link and event attributes; for OTLP/JSON lines, every
    ``attributes`` list), never trace or span ids.
    """
    for item in spans_or_lines:
        if isinstance(item, str):
            for value in _json_attribute_values(json.loads(item)):
                assert not _is_digest(value), value
            continue
        for attrs in _attr_maps(item):
            for key, value in attrs.items():
                for text in _strings(value):
                    assert not _is_digest(text), f"{item.name}: {key}"


def _json_attribute_values(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "attributes" and isinstance(value, list):
                for kv in value:
                    yield from _strings(_any_value(kv.get("value", {})))
            else:
                yield from _json_attribute_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _json_attribute_values(item)


def _any_value(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "arrayValue" in value:
        return [_any_value(v) for v in value["arrayValue"].get("values", [])]
    return None


def assert_keys_registered(spans: Iterable[ReadableSpan], registry: Iterable[str]) -> None:
    """Fail if a span attribute key is outside ``registry``."""
    allowed = set(registry)
    for span in spans:
        extra = set((span.attributes or {}).keys()) - allowed
        assert not extra, f"{span.name}: unregistered {sorted(extra)}"
