"""OpenTelemetry memory spans (PRD-CORE-343): one ``gen_ai.memory.client`` span per operation.

API only (OTEL-CONVENTIONS PKG-2): a host that installs an SDK provider sees the spans;
with none, ``OTEL_SDK_DISABLED=true`` or an API below the floor every span is a no-op
and nothing is marshaled. Values are engine ids, enums, counts and scores -- never
content, query text, tags, actor, paths or exception text (Part C, C-1/C-2).

The interface is :func:`memory_op` (a decorator on each operation entry function),
:func:`note` (diagnostics from inside a running operation) and :func:`export_store_id`
(the hosted-service mapping hook). Keys live in :mod:`trw_memory._otel_keys`.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, TypeVar, cast

import structlog

from trw_memory import _otel_keys as k
from trw_memory._hype_ids import hype_parent_id
from trw_memory._version import __version__

logger = structlog.get_logger(__name__)

ARRAY_CAP = 32
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_API_FLOOR = (1, 38)
#: A refusal status (a returned value, never a raise, MEM-10.2) -> its ``error.type`` (MEM-11.2).
_REFUSALS = {s: s for s in ("invalid", "not_found", "conflict", "rate_limited", "blocked")} | {
    "error": "storage_error",
    "unavailable": "_OTHER",
}
_EXCEPTIONS = {
    "MemoryNotFoundError": "not_found",
    "AuthorizationError": "unauthorized",
    "ValueError": "invalid",
    "SchemaValidationError": "invalid",
    "RateLimitError": "rate_limited",
    "PIIBlockError": "blocked",
    "PoisoningError": "blocked",
    "StorageError": "storage_error",
}

F = TypeVar("F", bound=Callable[..., Any])


def _probe() -> Any:
    """The ``trw_memory`` tracer, or None when the API is missing or below the floor (PKG-4, CONF-05)."""
    try:
        from importlib.metadata import version

        from opentelemetry import trace

        if tuple(int(p) for p in version("opentelemetry-api").split(".")[:2]) < _API_FLOOR:
            return None
        return trace.get_tracer("trw_memory", __version__, attributes={k.SCOPE_SEMCONV_VERSION: k.SEMCONV_VERSION})
    except (
        Exception
    ):  # trw-fail-silent-allow: None = telemetry off (CONF-05); a broken API never reaches the host; logged
        logger.debug("otel_probe_failed")
        return None


_TRACER: Any = _probe()
_SDK_DISABLED: bool | None = None
#: The recorder of the operation running in this context: it blocks nested spans (CONF-17.6).
_CURRENT: contextvars.ContextVar[_Recorder | None] = contextvars.ContextVar("trw_memory_otel_span", default=None)


def export_store_id(namespace: object) -> str | None:
    """The exported ``gen_ai.memory.store.id`` for *namespace*: identity locally (FR07).

    A hosted service replaces this with an opaque-id map or a keyed tenant-scoped HMAC;
    an unkeyed hash is never an allowed implementation (amendment 0.2 item 3).
    """
    return clean_id(namespace)


def clean_id(value: object) -> str | None:
    """*value* as an exportable identifier (SEC-07.1, OTEL-CONVENTIONS C-4), or None to omit it."""
    if type(value) is int:
        return str(value) if value.bit_length() <= 63 else None
    if type(value) is not str or len(value) > 4096:
        return None
    value = hype_parent_id(value) or value
    if value.startswith("idh1:") or not _ID.fullmatch(value):
        return None
    return value


def _disabled() -> bool:
    """``OTEL_SDK_DISABLED``, read at first emit and cached (CONVENTIONS §2)."""
    global _SDK_DISABLED
    if _SDK_DISABLED is None:
        _SDK_DISABLED = os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true"
    return _SDK_DISABLED


class _Recorder:
    """Attributes for one span; every setter is a no-op unless the span records."""

    __slots__ = ("notes", "recording", "span")

    def __init__(self, span: Any) -> None:
        self.span = span
        self.recording = span is not None and span.is_recording()
        self.notes: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        if self.recording and value is not None:
            self.span.set_attribute(key, value)

    def ids(self, values: Iterable[object], *, single: bool = False) -> None:
        """Deduplicate, then cap at ``ARRAY_CAP`` with ``record.truncated`` (SEC-07.5/07.6)."""
        seen: dict[str, None] = {}
        for raw in values:
            ident = clean_id(raw)
            if ident is not None:
                seen[ident] = None
                if len(seen) > ARRAY_CAP:
                    break
        kept = tuple(seen)[:ARRAY_CAP]
        if single and len(kept) == 1:
            self.set(k.RECORD_ID, kept[0])
        elif kept:
            self.set(k.RECORD_IDS, kept)
        if len(seen) > ARRAY_CAP:
            self.set(k.RECORD_TRUNCATED, True)

    def refuse(self, error_type: str) -> None:
        self.set(k.ERROR_TYPE, error_type if error_type in k.ERROR_TYPES else "_OTHER")
        if self.recording:
            from opentelemetry.trace import Status, StatusCode

            self.span.set_status(Status(StatusCode.ERROR))


def note(**values: object) -> None:
    """Record diagnostics the running operation already computed (method, threshold, reranked...)."""
    rec = _CURRENT.get()
    if rec is not None and rec.recording:
        rec.notes.update(values)


# ---------------------------------------------------------------- operation mapping


def _write_op(args: Mapping[str, Any]) -> str:
    return "upsert_memory" if args.get("entry_id") else "create_memory"


_OP_NAMES: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "store": _write_op,
    "bulk_store": lambda _a: "upsert_memory",
    "store_many": lambda _a: "create_memory",
    "update": lambda _a: "update_memory",
    "forget": lambda _a: "delete_memory",
    "recall": lambda _a: "search_memory",
    "search": lambda _a: "search_memory",
    "similar": lambda _a: "search_memory",
}


def _status(rec: _Recorder, result: object, outcomes: Mapping[str, str]) -> None:
    status = result.get("status") if isinstance(result, dict) else None
    if isinstance(status, str) and status in outcomes:
        rec.set(k.WRITE_OUTCOME, outcomes[status])
    elif isinstance(status, str) and status in _REFUSALS:
        rec.refuse(_REFUSALS[status])


def _describe(kind: str, args: Mapping[str, Any], result: object, rec: _Recorder) -> None:
    """Set the operation's attributes from its arguments and its result (FR03-FR05)."""
    res: Mapping[str, Any] = result if isinstance(result, dict) else {}
    client = args.get("client")
    namespace = res.get("namespace") or args.get("namespace") or getattr(client, "_namespace", None)
    rec.set(k.STORE_ID, export_store_id(namespace))
    if kind == "store":
        rec.set(k.RECORD_COUNT, 1)
        rec.ids([res.get("memory_id")], single=True)
        _status(rec, res, {"stored": "created", "updated": "updated"})
    elif kind == "update":
        rec.set(k.RECORD_COUNT, 1)
        rec.ids([args.get("entry_id")], single=True)
        _status(rec, res, {"updated": "updated", "no_changes": "noop"})
    elif kind == "bulk_store":
        total, stored, updated = (getattr(result, n, 0) for n in ("total", "stored", "updated"))
        rec.set(k.RECORD_COUNT, total)
        rec.set(k.FAILED_COUNT, getattr(result, "rejected", 0))
        if stored + updated:
            rec.set(k.WRITE_OUTCOME, "created" if stored == total else "updated" if updated == total else "mixed")
        rec.ids(getattr(item, "memory_id", None) for item in getattr(result, "items", ()))
    elif kind == "store_many":
        total = len(args.get("entries") or ())
        rec.set(k.RECORD_COUNT, total)
        if isinstance(result, int) and result:
            rec.set(k.WRITE_OUTCOME, "created" if result == total else "mixed")
    elif kind == "forget":
        memory_id = args.get("memory_id")
        rec.set(k.DELETE_TARGET, "record" if memory_id else "selector")
        if memory_id:
            rec.set(k.RECORD_COUNT, 1)
            rec.ids([memory_id], single=True)
        elif isinstance(res.get("deleted", res.get("entries_deleted")), int):
            rec.set(k.RECORD_COUNT, res.get("deleted", res.get("entries_deleted")))
        _status(rec, res, {})
    elif kind == "similar":
        _status(rec, res, {})
        if res.get("status") == "ok":
            rec.set(k.RECALL_METHOD, "vector")
            rec.set(k.RECORD_COUNT, 1 if res.get("existing_id") else 0)
            rec.ids([res.get("existing_id")], single=True)
    else:
        _describe_search(args, result, res, rec)


def _describe_search(args: Mapping[str, Any], result: object, res: Mapping[str, Any], rec: _Recorder) -> None:
    rows = result if isinstance(result, list) else res.get("memories", res.get("entries"))
    if not isinstance(rows, list):
        _status(rec, res, {})
        return
    rec.set(k.RECORD_COUNT, len(rows))
    if not (args.get("graph_depth") or args.get("include_graph_expansion")) and type(args.get("limit")) is int:
        rec.set(k.TOP_K, args.get("limit"))
    rec.ids(row.get("id") if isinstance(row, dict) else getattr(row, "id", None) for row in rows[: ARRAY_CAP + 1])
    notes = rec.notes
    method = notes.get("method")
    rec.set(k.RECALL_METHOD, method)
    if "reranked" in notes:
        rec.set(k.RECALL_RERANKED, bool(notes["reranked"]))
    if method is None:
        return
    # Scores are fused RRF unless a rerank or prior reordered them into rank positions (MEM-13.3).
    kind = "rrf" if method == "hybrid" and not notes.get("rank_scores") else "other"
    rec.set(k.RECALL_SCORE_KIND, kind)
    threshold = notes.get("threshold")
    if isinstance(threshold, float) and kind != "other":
        rec.set(k.RECALL_THRESHOLD, threshold)
        rec.set(k.RECALL_FILTERED, notes.get("filtered"))
    top = rows[0].get("score") if rows and isinstance(rows[0], dict) else None
    if kind != "other" and type(top) is float:
        rec.set(k.RECALL_TOP_SCORE, top)


# ---------------------------------------------------------------- the span itself


class _Span:
    """Start, describe and end one span around a host call; telemetry faults never reach the host."""

    __slots__ = ("args", "kind", "rec", "span", "token")

    def __init__(self, kind: str, sig: inspect.Signature, a: tuple[Any, ...], kw: dict[str, Any]) -> None:
        self.kind, self.token = kind, None
        self.span: Any = None
        self.args: Mapping[str, Any] = {}
        self.rec = _Recorder(None)
        if _CURRENT.get() is not None or _TRACER is None or _disabled():
            return
        try:
            op = _OP_NAMES[kind](kw)
            from opentelemetry.trace import SpanKind

            self.span = _TRACER.start_span(op, kind=SpanKind.INTERNAL, attributes={k.OPERATION_NAME: op})
            self.rec = _Recorder(self.span)
            self.token = _CURRENT.set(self.rec)
            if self.rec.recording:
                self.args = sig.bind_partial(*a, **kw).arguments
        except Exception:  # trw:intentional telemetry fault: the host call proceeds as traced so far
            logger.debug("otel_span_start_failed", operation=kind)

    def use(self) -> Any:
        from opentelemetry import trace

        return trace.use_span(self.span, end_on_exit=False, record_exception=False, set_status_on_exception=False)

    def finish(self, result: object = None, exc: BaseException | None = None) -> None:
        """Reset the nesting guard, describe the outcome and end the span; each step is guarded apart."""
        if self.span is None:
            return
        for step in (self._reset, lambda: self._describe(result, exc), self.span.end):
            try:
                step()
            except Exception:  # trw:intentional telemetry fault never changes a host result
                logger.debug("otel_span_finish_failed", operation=self.kind)

    def _reset(self) -> None:
        if self.token is not None:
            _CURRENT.reset(self.token)

    def _describe(self, result: object, exc: BaseException | None) -> None:
        if isinstance(exc, Exception):
            self.rec.refuse(_exception_type(exc))
        elif exc is None and self.rec.recording:
            _describe(self.kind, self.args, result, self.rec)


def _exception_type(exc: Exception) -> str:
    for cls in type(exc).__mro__:
        if cls.__name__ in _EXCEPTIONS:
            return _EXCEPTIONS[cls.__name__]
    return "_OTHER"


def memory_op(kind: str) -> Callable[[F], F]:
    """Wrap an operation entry function (sync or async) in one memory span (FR01)."""

    def decorate(fn: F) -> F:
        sig = inspect.signature(fn)
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def run_async(*a: Any, **kw: Any) -> Any:
                span = _Span(kind, sig, a, kw)
                if span.span is None:
                    return await cast("Callable[..., Awaitable[Any]]", fn)(*a, **kw)
                try:
                    with span.use():
                        result = await cast("Callable[..., Awaitable[Any]]", fn)(*a, **kw)
                except BaseException as exc:
                    span.finish(exc=exc)
                    raise
                span.finish(result)
                return result

            return cast("F", run_async)

        @functools.wraps(fn)
        def run(*a: Any, **kw: Any) -> Any:
            span = _Span(kind, sig, a, kw)
            if span.span is None:
                return fn(*a, **kw)
            try:
                with span.use():
                    result = fn(*a, **kw)
            except BaseException as exc:
                span.finish(exc=exc)
                raise
            span.finish(result)
            return result

        return cast("F", run)

    return decorate
