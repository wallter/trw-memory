"""The bounded OTLP/JSON-lines file exporter (PRD-CORE-342 FR03)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from tests._otel_support import assert_no_unkeyed_digest
from trw_memory import otel_setup


def _spans(n: int = 1) -> list[ReadableSpan]:
    mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("parent", kind=SpanKind.SERVER):
        for i in range(n):
            with tracer.start_as_current_span(f"child{i}") as span:
                span.set_attribute("com.trwframework.run.id", "run-1")
    return list(mem.get_finished_spans())


def test_line_is_otlp_json_with_hex_ids_and_integer_enums(tmp_path: Path) -> None:
    path = tmp_path / "traces-x-1-20260929.jsonl"
    exporter = otel_setup.OtlpJsonFileExporter(path)
    spans = _spans()
    assert exporter.export(spans) is SpanExportResult.SUCCESS
    (line,) = path.read_text().splitlines()
    doc = json.loads(line)
    span_docs = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    by_name = {s["name"]: s for s in span_docs}
    child, parent = by_name["child0"], by_name["parent"]
    assert child["traceId"] == format(spans[0].context.trace_id, "032x")
    assert child["spanId"] == format(spans[0].context.span_id, "016x")
    assert child["parentSpanId"] == parent["spanId"]
    assert parent["kind"] == 2  # SERVER, integer enum
    assert all(len(s["spanId"]) == 16 and s["spanId"] == s["spanId"].lower() for s in span_docs)
    assert path.stat().st_mode & 0o777 == 0o600
    assert_no_unkeyed_digest([line])


def test_cap_is_never_exceeded_and_no_line_is_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "traces-x-1-20260929.jsonl"
    one = len(otel_setup.encode_otlp_json_line(_spans()))
    monkeypatch.setattr(otel_setup, "FILE_CAP_BYTES", one * 2 + one // 2)
    exporter = otel_setup.OtlpJsonFileExporter(path)
    results = [exporter.export(_spans()) for _ in range(5)]
    assert results == [SpanExportResult.SUCCESS] * 5
    lines = path.read_bytes().split(b"\n")
    assert lines[-1] == b""
    assert len(lines) - 1 == 2
    assert path.stat().st_size <= otel_setup.FILE_CAP_BYTES
    for line in lines[:-1]:
        json.loads(line)


def test_oversized_single_batch_is_dropped_with_one_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(otel_setup.logger, "warning", lambda event, **_: warnings.append(event))
    monkeypatch.setattr(otel_setup, "FILE_CAP_BYTES", 10)
    path = tmp_path / "traces-x-1-20260929.jsonl"
    exporter = otel_setup.OtlpJsonFileExporter(path)
    assert exporter.export(_spans()) is SpanExportResult.SUCCESS
    assert exporter.export(_spans()) is SpanExportResult.SUCCESS
    assert not path.exists()
    assert warnings == ["otel_file_cap_reached"]


@pytest.mark.skipif(os.getuid() == 0, reason="root ignores directory permissions")
def test_unwritable_directory_returns_failure(tmp_path: Path) -> None:
    ro = tmp_path / "ro"
    ro.mkdir(mode=0o500)
    try:
        exporter = otel_setup.OtlpJsonFileExporter(ro / "traces-x-1-20260929.jsonl")
        assert exporter.export(_spans()) is SpanExportResult.FAILURE
    finally:
        ro.chmod(0o700)


def test_prune_respects_budget_pattern_symlinks_and_current_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(otel_setup, "DIR_BUDGET_BYTES", 250)
    d = tmp_path / "otel"
    d.mkdir()
    for i, name in enumerate(["traces-a-1-20260101.jsonl", "traces-b-2-20260102.jsonl", "traces-c-3-20260103.jsonl"]):
        (d / name).write_bytes(b"x" * 100)
        os.utime(d / name, (1000 + i, 1000 + i))
    (d / "other.jsonl").write_bytes(b"x" * 1000)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"x" * 1000)
    (d / "traces-link-9-20260101.jsonl").symlink_to(outside)
    for oldest in (d / "other.jsonl", outside):  # the oldest by mtime: a wrong pattern would delete them first
        os.utime(oldest, (1, 1))
    os.utime(d / "traces-link-9-20260101.jsonl", (1, 1), follow_symlinks=False)
    keep = d / "traces-a-1-20260101.jsonl"
    otel_setup._prune(d, keep=keep)
    remaining = sorted(p.name for p in d.iterdir())
    # oldest eligible file (b) removed; total now 200 < 250; others untouched
    assert remaining == [
        "other.jsonl",
        "traces-a-1-20260101.jsonl",
        "traces-c-3-20260103.jsonl",
        "traces-link-9-20260101.jsonl",
    ]
    assert outside.exists()


def test_install_tightens_a_loose_user_owned_directory(tmp_path: Path) -> None:
    d = tmp_path / "otel"
    d.mkdir(mode=0o755)
    d.chmod(0o755)
    exporter = otel_setup._install_file_exporter("trw-mcp", d)
    assert d.stat().st_mode & 0o777 == 0o700
    assert exporter.path.parent == d
    assert f"-{os.getpid()}-" in exporter.path.name


def test_install_prunes_an_over_budget_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel_setup, "DIR_BUDGET_BYTES", 150)
    d = tmp_path / "otel"
    d.mkdir(mode=0o700)
    for i in range(3):
        (d / f"traces-old-{i}-20260101.jsonl").write_bytes(b"x" * 100)
        os.utime(d / f"traces-old-{i}-20260101.jsonl", (100 + i, 100 + i))
    otel_setup._install_file_exporter("trw-mcp", d)
    assert sorted(p.name for p in d.iterdir()) == ["traces-old-2-20260101.jsonl"]


def test_a_failed_write_leaves_no_partial_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "traces-x-1-20260929.jsonl"
    exporter = otel_setup.OtlpJsonFileExporter(path)
    assert exporter.export(_spans()) is SpanExportResult.SUCCESS
    before = path.read_bytes()
    real_write = os.write

    def short_write(fd: int, data: bytes) -> int:
        return real_write(fd, data[: len(data) // 2])

    monkeypatch.setattr(otel_setup.os, "write", short_write)
    assert exporter.export(_spans()) is SpanExportResult.FAILURE
    assert path.read_bytes() == before


def test_prune_caps_the_file_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel_setup, "MAX_FILES", 3)
    d = tmp_path / "otel"
    d.mkdir()
    for i in range(5):
        (d / f"traces-cli-{i}-20260101.jsonl").write_bytes(b"x")
        os.utime(d / f"traces-cli-{i}-20260101.jsonl", (100 + i, 100 + i))
    otel_setup._prune(d, keep=d / "traces-new-9-20260101.jsonl")
    # two remain plus the current process's file about to be created = the cap
    assert sorted(p.name for p in d.iterdir()) == ["traces-cli-3-20260101.jsonl", "traces-cli-4-20260101.jsonl"]


def test_the_budget_holds_after_files_grow_post_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install saw a directory under budget; a sibling grows afterwards; the next write prunes it (oldest first)."""
    monkeypatch.setattr(otel_setup, "DIR_BUDGET_BYTES", 5_000)
    d = tmp_path / "otel"
    d.mkdir(mode=0o700)
    sibling = d / "traces-other-1-20260101.jsonl"
    sibling.write_bytes(b"x" * 100)
    os.utime(sibling, (100, 100))
    exporter = otel_setup._install_file_exporter("trw-mcp", d)
    assert sibling.exists()  # under budget at install

    sibling.write_bytes(b"x" * 6_000)  # grows past the whole budget afterwards
    os.utime(sibling, (100, 100))
    assert exporter.export(_spans()) is SpanExportResult.SUCCESS

    assert not sibling.exists()
    assert exporter.path.exists() and exporter.path.stat().st_size > 0  # the live file is never pruned


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_a_directory_swapped_for_a_symlink_never_reaches_files_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Install, then replace the export dir with a symlink to an outside dir holding an over-budget victim."""
    monkeypatch.setattr(otel_setup, "DIR_BUDGET_BYTES", 500)
    d = tmp_path / "otel"
    d.mkdir(mode=0o700)
    exporter = otel_setup._install_file_exporter("trw-mcp", d)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "traces-victim.jsonl"
    victim.write_bytes(b"v" * 10_000)
    os.utime(victim, (100, 100))
    d.rmdir()
    d.symlink_to(outside, target_is_directory=True)

    result = exporter.export(_spans())

    assert result in (SpanExportResult.SUCCESS, SpanExportResult.FAILURE)
    assert victim.read_bytes() == b"v" * 10_000  # not pruned
    assert [p.name for p in outside.iterdir()] == ["traces-victim.jsonl"]  # and nothing written there either


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_prune_refuses_a_symlinked_directory_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel_setup, "DIR_BUDGET_BYTES", 10)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "traces-victim.jsonl"
    victim.write_bytes(b"v" * 100)
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)

    otel_setup._prune(link, keep=link / "traces-live.jsonl")

    assert victim.exists()


@pytest.mark.parametrize("mutation", ["replaced_by_file", "removed", "unreadable"])
def test_prune_and_export_survive_a_hostile_directory_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """The export dir is replaced by a file, deleted, or made unreadable after install: no raise, no outside effect."""
    monkeypatch.setattr(otel_setup, "DIR_BUDGET_BYTES", 10)
    d = tmp_path / "otel"
    d.mkdir(mode=0o700)
    exporter = otel_setup._install_file_exporter("trw-mcp", d)
    bystander = tmp_path / "traces-bystander.jsonl"  # a sibling of the directory, never inside it
    bystander.write_bytes(b"b" * 1_000)
    if mutation == "replaced_by_file":
        d.rmdir()
        d.write_bytes(b"not a directory")
    elif mutation == "removed":
        d.rmdir()
    else:
        d.chmod(0)
    try:
        assert exporter.export(_spans()) in (SpanExportResult.SUCCESS, SpanExportResult.FAILURE)
        otel_setup._prune(d, keep=d / "traces-live.jsonl")  # and the direct call is a quiet no-op
    finally:
        if mutation == "unreadable":
            d.chmod(0o700)
    assert bystander.read_bytes() == b"b" * 1_000
