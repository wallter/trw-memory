"""Opt-in OpenTelemetry SDK setup for TRW application entrypoints (PRD-CORE-342, PRD-CORE-344 FR05).

This is the only runtime importer of ``opentelemetry.sdk`` in TRW. Libraries use ``opentelemetry-api``
alone; an application entrypoint (the trw-mcp server/CLI, the memory daemon) calls
:func:`configure_tracing` once. With ``enabled`` false or ``OTEL_SDK_DISABLED=true`` it returns before
importing the SDK and writes nothing to disk.

Exporters (``OTEL_TRACES_EXPORTER``): unset or ``trw_file`` gives the bounded local OTLP/JSON-lines file
when ``file_dir`` is set, else OTLP/HTTP; ``otlp``, ``console`` and ``none`` behave as standard. Every
exporter is wrapped in :class:`ScrubbingSpanExporter`, so no exported span carries exception text
(OTEL-CONVENTIONS C-3).
"""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from trw_memory._version import __version__

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = structlog.get_logger(__name__)

FILE_CAP_BYTES = 52_428_800  # 50 MiB per file; a line that does not fit is dropped, never truncated
DIR_BUDGET_BYTES = 268_435_456  # 256 MiB across traces-*.jsonl, enforced at install and after every write (oldest first, never the live file)
MAX_FILES = 512  # one file per process: short-lived CLI verbs would otherwise grow the directory unboundedly
_ID_KEYS = frozenset({"traceId", "spanId", "parentSpanId"})
_lock = threading.Lock()
_installed = False


def configure_tracing(
    service_name: str, file_dir: Path | None, enabled: bool, *, service_version: str | None = None
) -> bool:
    """Install one SDK ``TracerProvider`` for this process; return whether one is installed.

    Never raises. A second call after a successful install is a no-op that returns True.
    ``service_version`` is the process's package version (default: trw-memory's, for the daemon).
    """
    global _installed
    if not enabled or os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False
    with _lock:
        if _installed:
            return True
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:  # trw-fail-silent-allow: False IS the contract ("no provider installed"); logged
            logger.debug("otel_sdk_unavailable", service=service_name)
            return False
        if isinstance(trace.get_tracer_provider(), TracerProvider):
            logger.debug("otel_provider_already_set", service=service_name)
            return False  # another owner installed one; the API allows exactly one per process
        try:
            exporter = _select_exporter(service_name, file_dir)
            os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.namespace": "trwframework",
                    "service.version": service_version or __version__,
                    "service.instance.id": str(uuid.uuid4()),
                }
            )
            provider = TracerProvider(resource=resource)
            if exporter is not None:
                provider.add_span_processor(BatchSpanProcessor(cast("SpanExporter", ScrubbingSpanExporter(exporter))))
            trace.set_tracer_provider(provider)
        # trw-fail-silent-allow: False = no provider installed; telemetry never blocks startup (NFR01); logged
        except Exception as exc:
            logger.warning("otel_setup_failed", error_type=type(exc).__name__)
            return False
        _installed = True
        return True


def _select_exporter(service_name: str, file_dir: Path | None) -> SpanExporter | OtlpJsonFileExporter | None:
    choice = os.environ.get("OTEL_TRACES_EXPORTER", "").strip().lower()
    if choice in ("", "trw_file") and file_dir is not None:
        return _install_file_exporter(service_name, file_dir)
    if choice in ("", "trw_file", "otlp"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter()
    if choice == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter(out=sys.stderr)  # stdout is the MCP JSON-RPC channel for a stdio server
    if choice == "none":
        return None
    logger.warning("otel_exporter_unsupported", value=choice[:32])
    raise ValueError("unsupported OTEL_TRACES_EXPORTER")


def _install_file_exporter(service_name: str, file_dir: Path) -> OtlpJsonFileExporter:
    file_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = file_dir.lstat()
    if st.st_uid == os.getuid() and st.st_mode & 0o077:
        file_dir.chmod(0o700)
    path = file_dir / f"traces-{service_name}-{os.getpid()}-{time.strftime('%Y%m%d', time.gmtime())}.jsonl"
    identity = _dir_identity(file_dir)
    _prune(file_dir, keep=path, identity=identity)
    return OtlpJsonFileExporter(path, identity)


def _dir_identity(file_dir: Path) -> tuple[int, int] | None:
    """``(st_dev, st_ino)`` of the export directory itself, never of what a symlink points at."""
    try:
        st = file_dir.lstat()
    except (
        OSError
    ):  # trw-fail-silent-allow: a missing or unreadable directory has no identity; callers treat None as unbound
        return None
    return (st.st_dev, st.st_ino) if stat.S_ISDIR(st.st_mode) else None


def _prune(file_dir: Path, keep: Path, identity: tuple[int, int] | None = None) -> None:
    """Delete the oldest ``traces-*.jsonl`` files until the directory is under budget (never ``keep``).

    Every step goes through one directory descriptor opened ``O_NOFOLLOW``: if the directory was swapped for a
    symlink (or, given *identity*, for anything but the directory recorded at install), nothing is scanned or
    unlinked, so a prune can never reach a file outside the export directory.
    """
    try:
        fd = os.open(file_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        logger.debug("otel_prune_skipped", reason="directory_unopenable")
        return
    try:
        opened = os.fstat(fd)
        if identity is not None and (opened.st_dev, opened.st_ino) != identity:
            logger.warning("otel_prune_skipped", reason="directory_replaced")
            return
        files: list[tuple[float, int, str]] = []
        total = 0
        with os.scandir(fd) as entries:
            for entry in entries:
                if not (entry.name.startswith("traces-") and entry.name.endswith(".jsonl")):
                    continue
                try:  # a concurrent process may prune the same directory: skip what vanished
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        continue
                    st = entry.stat(follow_symlinks=False)
                except OSError:  # trw-fail-silent-allow: a file another process pruned mid-scan is simply absent
                    continue
                total += st.st_size
                if entry.name != keep.name:
                    files.append((st.st_mtime, st.st_size, entry.name))
        count = len(files) + 1  # the current process's file is about to exist
        for _mtime, size, name in sorted(files):
            if total < DIR_BUDGET_BYTES and count <= MAX_FILES:
                break
            try:
                os.unlink(name, dir_fd=fd)
            except FileNotFoundError:  # trw-fail-silent-allow: another process already pruned this file
                pass
            except OSError:
                logger.debug("otel_prune_skipped", error_type="OSError")
                continue
            total -= size
            count -= 1
    finally:
        os.close(fd)


def _hex_ids(node: Any) -> None:
    """Rewrite protobuf-JSON base64 trace/span ids to OTLP/JSON lowercase hex, in place."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _ID_KEYS and isinstance(value, str):
                node[key] = base64.b64decode(value).hex()
            else:
                _hex_ids(value)
    elif isinstance(node, list):
        for item in node:
            _hex_ids(item)


def encode_otlp_json_line(spans: Sequence[ReadableSpan]) -> bytes:
    """One OTLP/JSON ``ExportTraceServiceRequest`` line: hex ids, integer enums, lowerCamelCase keys."""
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    doc = MessageToDict(encode_spans(spans), use_integers_for_enums=True)
    _hex_ids(doc)
    return (json.dumps(doc, separators=(",", ":")) + "\n").encode("utf-8")


def _result(name: str) -> SpanExportResult:
    from opentelemetry.sdk.trace.export import SpanExportResult

    return SpanExportResult[name]


class OtlpJsonFileExporter:
    """Bounded OTLP/JSON-lines ``SpanExporter`` (duck-typed so importing this module needs no SDK).

    One file per process. The whole encoded line is checked against :data:`FILE_CAP_BYTES` before it is
    written, so no line is ever truncated; once a line does not fit, this and every later batch is dropped
    with one ``otel_file_cap_reached`` warning and ``SUCCESS`` (the SDK must not retry). I/O errors return
    ``FAILURE``; nothing raises into the host.
    """

    def __init__(self, path: Path, dir_identity: tuple[int, int] | None = None) -> None:
        self._path = path
        self._dir_identity = dir_identity if dir_identity is not None else _dir_identity(path.parent)
        self._capped = False
        self._write_lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            line = encode_otlp_json_line(spans)
        except Exception as exc:  # an unencodable batch is lost, loudly
            logger.warning("otel_file_encode_failed", error_type=type(exc).__name__)
            return _result("FAILURE")
        with self._write_lock:
            if self._capped:
                return _result("SUCCESS")
            try:
                size = self._path.stat().st_size if self._path.exists() else 0
                if size + len(line) > FILE_CAP_BYTES:
                    self._capped = True
                    logger.warning("otel_file_cap_reached", cap_bytes=FILE_CAP_BYTES)
                    return _result("SUCCESS")
                if _dir_identity(self._path.parent) != self._dir_identity:  # swapped since install: write nowhere else
                    logger.warning("otel_file_write_skipped", reason="directory_replaced")
                    return _result("FAILURE")
                flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(self._path, flags, 0o600)
                try:
                    written = os.write(fd, line)
                    if written != len(line):  # never leave a truncated line behind
                        os.ftruncate(fd, size)
                        raise OSError("short write")
                except OSError:
                    os.ftruncate(fd, size)
                    raise
                finally:
                    os.close(fd)
                _prune(
                    self._path.parent, self._path, self._dir_identity
                )  # files grow after install: hold the budget on every write
            except OSError as exc:
                logger.warning("otel_file_write_failed", error_type=type(exc).__name__)
                return _result("FAILURE")
        return _result("SUCCESS")

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class ScrubbingSpanExporter:
    """Drop ``exception`` events and blank status descriptions before delegating (PRD-CORE-344 FR05).

    Each span is rebuilt through the public ``ReadableSpan`` constructor with the same fields, filtered
    events and a description-free status; ``error.type`` and every other attribute are kept, and ``exception.*`` keys are dropped from span, event and link attributes alike. A
    batch that cannot be scrubbed returns ``FAILURE`` without reaching the delegate: exporting unscrubbed
    spans is worse than losing them.
    """

    def __init__(self, delegate: SpanExporter | OtlpJsonFileExporter) -> None:
        self._delegate = delegate

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            scrubbed = [_scrub(span) for span in spans]
        except Exception as exc:
            logger.warning("otel_scrub_failed", error_type=type(exc).__name__)
            return _result("FAILURE")
        return self._delegate.export(scrubbed)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def _scrub(span: ReadableSpan) -> ReadableSpan:
    from opentelemetry.sdk.trace import Event
    from opentelemetry.sdk.trace import ReadableSpan as _ReadableSpan
    from opentelemetry.trace import Link
    from opentelemetry.trace.status import Status

    return _ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_without_exception_keys(span.attributes),
        events=tuple(
            Event(event.name, _without_exception_keys(event.attributes), event.timestamp)
            for event in span.events
            if event.name != "exception"
        ),
        links=tuple(Link(link.context, _without_exception_keys(link.attributes)) for link in span.links),
        kind=span.kind,
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _without_exception_keys(attributes: Any) -> Any:
    """Drop ``exception.*`` keys (message, stacktrace) wherever a third party put them."""
    if not attributes or not any(str(key).startswith("exception.") for key in attributes):
        return attributes
    return {key: value for key, value in attributes.items() if not str(key).startswith("exception.")}
