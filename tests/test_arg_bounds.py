"""``ArgumentBounds`` (``trw_memory.daemon._arg_bounds``) refuses an oversized tool call
before the tool body runs (rc9 sweep round 4).

Drives the served surface in-process, over the real MCP wire (``fastmcp.Client``
against ``trw_memory.server.mcp``), so every assertion covers the actual
registered middleware and tool schemas rather than a unit-level stand-in.
``tests/_trw_home.py``'s autouse ``isolated_trw_home`` fixture (loaded by
``conftest.py``) redirects ``HOME``/``XDG_DATA_HOME``/``TRW_USER_DIR`` to a throwaway
tmp dir for every test in this module, so nothing here touches the operator's
real ``~`` or ``~/.trw``.

Three groups of tests:

1. A census over every served tool's parameter schema: any ``string``/``array``/
   ``object`` argument (including one only reachable through ``anyOf``, e.g.
   ``list[str] | None``) must have a bound (``_arg_bounds.bound(tool, arg)`` is
   not ``None``), and every ``OVERRIDES``/``ARGUMENTS`` entry must still name a
   real tool or parameter. This is the test that fails when a new tool ships
   with an unbounded argument.
2. Four failing-first sweep findings: build an argument exactly one item past
   its bound and assert the call is refused with ``argument_too_large`` -- and,
   separately, that the tool's ``_impl`` function was never reached (proving the
   refusal came from the middleware, not the tool body).
3. Bound edges and shapes: exactly-at-the-limit is admitted, nested defaults
   (``TEXT``/``ITEMS``/``NAME``) apply inside an object argument and the refusal
   still names the top-level argument, the whole request is capped at
   ``MAX_REQUEST_VALUES``/``MAX_REQUEST_CHARS``, a wrapped-output tool's refusal
   is still unwrapped by the client's ``.data``, a plain top-level string
   argument is bounded too, and an HTTP body past ``MAX_BODY_BYTES`` is answered
   413 before the app reads it.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client
from fastmcp.tools.tool import Tool

from trw_memory.daemon import _arg_bounds as ab
from trw_memory.server import mcp
from trw_memory.tools import recall_support
from trw_memory.tools import sync as sync_mod

pytest.importorskip("fastmcp")

#: A namespace scoped to this test module; never the daemon's bare defaults.
NS = "project:test-arg-bounds"

#: JSON Schema types that ``bound()`` must cover (characters for a string, items
#: for an array or object).
_BOUNDED_TYPES = {"string", "array", "object"}


async def _tools() -> dict[str, Tool]:
    """Every tool the served ``mcp`` instance publishes, by name."""
    return {tool.name: tool for tool in await mcp.list_tools()}


def _schema_types(schema: dict[str, Any], defs: dict[str, Any]) -> set[str]:
    """The JSON type(s) *schema* admits, minus ``null``: through ``$ref`` into *defs* and any
    ``anyOf``/``oneOf``/``allOf`` nesting. A schema that names no type at all (``Any``) counts
    as every bounded type, so it cannot slip past the census.
    """
    if "$ref" in schema:
        return _schema_types(defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    types = {schema["type"]} if isinstance(schema.get("type"), str) else set(schema.get("type") or ())
    for key in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(key, ()):
            types |= _schema_types(branch, defs)
    if not types and not any(key in schema for key in ("anyOf", "oneOf", "allOf", "enum", "const")):
        types = set(_BOUNDED_TYPES)
    types.discard("null")
    return types


async def _call(tool: str, **arguments: object) -> Any:
    """Call *tool* on the real served surface over the wire; never raises on a tool error."""
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments, raise_on_error=False)
        return result.data


def _assert_argument_too_large(data: object, argument: str, limit: int) -> None:
    assert isinstance(data, dict), f"expected a refusal dict, got {data!r}"
    assert data.get("status") == "invalid", data
    assert data.get("error") == "argument_too_large", data
    assert data.get("argument") == argument, data
    assert data.get("limit") == limit, data


# ---------------------------------------------------------------------------
# 1. Census: every bounded-shape argument of every served tool has a bound,
#    and every _arg_bounds entry still names something real.
# ---------------------------------------------------------------------------


async def test_every_string_array_or_object_argument_has_a_bound() -> None:
    """A served tool's ``str``/``list``/``dict`` argument with no bound can ship an
    unbounded MCP surface without anyone deciding it -- this is the guard.
    """
    tools = await _tools()
    missing: list[tuple[str, str]] = []
    for tool_name, tool in tools.items():
        properties = tool.parameters.get("properties", {})
        for argument, schema in properties.items():
            defs = tool.parameters.get("$defs", {})
            if _schema_types(schema, defs) & _BOUNDED_TYPES and ab.bound(tool_name, argument) is None:
                missing.append((tool_name, argument))
    assert not missing, (
        "these served tool arguments accept a string, array or object but have no bound "
        "in trw_memory.daemon._arg_bounds -- add one to ARGUMENTS (keyed by argument name) "
        f"or OVERRIDES (keyed by tool, for a tool-specific limit): {missing}"
    )


async def test_overrides_name_real_tools_and_parameters() -> None:
    """A stale ``OVERRIDES`` entry (a renamed/removed tool or argument) would silently
    stop bounding what it once bounded; this catches the drift.
    """
    tools = await _tools()
    bad: list[str] = []
    for tool_name, per_argument in ab.OVERRIDES.items():
        tool = tools.get(tool_name)
        if tool is None:
            bad.append(f"OVERRIDES names tool {tool_name!r}, which is not served")
            continue
        properties = tool.parameters.get("properties", {})
        bad.extend(
            f"OVERRIDES[{tool_name!r}] names argument {argument!r}, which is not a parameter of it"
            for argument in per_argument
            if argument not in properties
        )
    assert not bad, "\n".join(bad)


async def test_argument_names_are_real_parameters_of_at_least_one_tool() -> None:
    """A stale ``ARGUMENTS`` key (an argument name no tool uses any more) is dead
    weight that also hides a real gap were the name ever reused differently.
    """
    tools = await _tools()
    all_parameter_names: set[str] = set()
    for tool in tools.values():
        all_parameter_names |= set(tool.parameters.get("properties", {}))
    stale = sorted(name for name in ab.ARGUMENTS if name not in all_parameter_names)
    assert not stale, f"these ARGUMENTS keys in trw_memory.daemon._arg_bounds name no served tool's parameter: {stale}"


# ---------------------------------------------------------------------------
# 2. Failing-first sweep findings: one item past the bound is refused BEFORE
#    the tool body runs.
# ---------------------------------------------------------------------------


async def test_sync_mark_synced_acks_over_bound_is_refused() -> None:
    limit = ab.bound("memory_sync_mark_synced", "acks")
    assert limit is not None
    acks = {f"id{i}": 1 for i in range(limit + 1)}
    data = await _call("memory_sync_mark_synced", namespace=NS, acks=acks)
    _assert_argument_too_large(data, "acks", limit)


async def test_sync_mark_synced_over_bound_never_reaches_the_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal must come from the middleware, not a permission check inside the tool body.

    ``register_sync_tools`` closes over ``memory_sync_mark_synced_impl`` by its
    module-global name and looks it up again at call time (it is referenced from
    inside a ``lambda`` the tool wrapper builds fresh per call), so replacing the
    module attribute here is visible to a call made through the registered tool.
    """
    calls: list[object] = []

    def _stub(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"marked": 0}

    monkeypatch.setattr(sync_mod, "memory_sync_mark_synced_impl", _stub)
    limit = ab.bound("memory_sync_mark_synced", "acks")
    assert limit is not None
    acks = {f"id{i}": 1 for i in range(limit + 1)}
    await _call("memory_sync_mark_synced", namespace=NS, acks=acks)
    assert calls == [], "memory_sync_mark_synced_impl ran despite an over-bound acks argument"


async def test_admit_shared_results_over_bound_is_refused() -> None:
    limit = ab.bound("memory_admit_shared", "results")
    assert limit is not None
    results = [{"id": f"r{i}"} for i in range(limit + 1)]
    data = await _call("memory_admit_shared", namespace=NS, results=results)
    _assert_argument_too_large(data, "results", limit)


async def test_admit_shared_over_bound_never_reaches_the_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def _stub(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"status": "ok", "admitted": [], "refused": 0, "gate_errors": 0}

    monkeypatch.setattr(recall_support, "memory_admit_shared_impl", _stub)
    limit = ab.bound("memory_admit_shared", "results")
    assert limit is not None
    results = [{"id": f"r{i}"} for i in range(limit + 1)]
    await _call("memory_admit_shared", namespace=NS, results=results)
    assert calls == [], "memory_admit_shared_impl ran despite an over-bound results argument"


async def test_vectors_ids_over_bound_is_refused() -> None:
    limit = ab.bound("memory_vectors", "ids")
    assert limit is not None
    ids = [f"id{i}" for i in range(limit + 1)]
    data = await _call("memory_vectors", namespace=NS, ids=ids)
    _assert_argument_too_large(data, "ids", limit)


async def test_vectors_over_bound_never_reaches_the_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def _stub(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr(recall_support, "memory_vectors_impl", _stub)
    limit = ab.bound("memory_vectors", "ids")
    assert limit is not None
    ids = [f"id{i}" for i in range(limit + 1)]
    await _call("memory_vectors", namespace=NS, ids=ids)
    assert calls == [], "memory_vectors_impl ran despite an over-bound ids argument"


async def test_sync_find_ids_over_bound_is_refused() -> None:
    limit = ab.bound("memory_sync_find", "ids")
    assert limit is not None
    ids = [f"id{i}" for i in range(limit + 1)]
    data = await _call("memory_sync_find", namespace=NS, remote_id="r", ids=ids)
    _assert_argument_too_large(data, "ids", limit)


async def test_sync_find_over_bound_never_reaches_the_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def _stub(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"status": "not_found"}

    monkeypatch.setattr(sync_mod, "memory_sync_find_impl", _stub)
    limit = ab.bound("memory_sync_find", "ids")
    assert limit is not None
    ids = [f"id{i}" for i in range(limit + 1)]
    await _call("memory_sync_find", namespace=NS, remote_id="r", ids=ids)
    assert calls == [], "memory_sync_find_impl ran despite an over-bound ids argument"


# ---------------------------------------------------------------------------
# 3. Bound edges and shapes.
# ---------------------------------------------------------------------------


async def test_exactly_at_the_limit_is_admitted() -> None:
    """*Exactly* the limit (not one past it) must reach the tool body -- the middleware
    is an ``>`` check, not ``>=``. The tool may still refuse for its own reasons
    (e.g. a namespace permission), so this only asserts the refusal was not
    ``argument_too_large``.
    """
    limit = ab.bound("memory_sync_mark_synced", "acks")
    assert limit is not None
    acks = {f"id{i}": 1 for i in range(limit)}
    data = await _call("memory_sync_mark_synced", namespace=NS, acks=acks)
    assert not (isinstance(data, dict) and data.get("error") == "argument_too_large"), data


async def test_nested_object_value_over_text_default_is_refused() -> None:
    """``metadata`` has no ``OVERRIDES``/``ARGUMENTS`` entry of its own for its VALUES
    (only for the object itself, via ``ITEMS``) -- a value nested inside it falls
    back to the ``TEXT`` default, per the module's docstring. The refusal names
    the TOP-LEVEL argument (``metadata``), never a nested path, even though the
    breach is inside it.
    """
    data = await _call("memory_store", namespace=NS, content="c", metadata={"k": "x" * (ab.TEXT + 1)})
    _assert_argument_too_large(data, "metadata", ab.TEXT)


async def test_nested_dict_key_over_name_default_is_refused() -> None:
    """A dict key nested inside an argument is bounded at ``NAME``, regardless of
    what the value's own bound is; the refusal still names the top-level argument.
    """
    data = await _call("memory_store", namespace=NS, content="c", metadata={"k" * (ab.NAME + 1): "v"})
    _assert_argument_too_large(data, "metadata", ab.NAME)


async def test_whole_request_over_max_chars_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-request character cap, exercised with a small ``MAX_REQUEST_CHARS`` so the
    test stays fast -- a real 8 MiB request is not worth building just to cross it. Both
    ``namespace`` and ``content`` individually fit their own per-argument bounds, so the
    refusal can only come from the request-level total.
    """
    monkeypatch.setattr(ab, "MAX_REQUEST_CHARS", 5)
    data = await _call("memory_store", namespace=NS, content="hello world")
    _assert_argument_too_large(data, "(request)", 5)


async def test_whole_request_over_max_values_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-request value-count cap, the same way: each top-level argument (and each
    item/key walked inside one) counts as one value.
    """
    monkeypatch.setattr(ab, "MAX_REQUEST_VALUES", 2)
    data = await _call("memory_status", namespace=NS, security_settings_only=False, extra="x")
    _assert_argument_too_large(data, "(request)", 2)


async def test_wrapped_output_tool_refusal_is_still_unwrapped_by_data() -> None:
    """``memory_update``'s return type is wrapped on the wire (``x-fastmcp-wrap-result``
    on its output schema, since it returns a union). The middleware wraps its own
    refusal to match (``{"result": refusal}``) so a ``fastmcp.Client``'s ``.data``
    still unwraps it to the plain refusal dict any other tool's refusal already is.
    """
    limit = ab.bound("memory_update", "patch")
    assert limit is not None
    patch = {f"k{i}": "v" for i in range(limit + 1)}
    async with Client(mcp) as client:
        result = await client.call_tool(
            "memory_update", {"entry_id": "e1", "namespace": NS, "patch": patch}, raise_on_error=False
        )
        expected = {"status": "invalid", "error": "argument_too_large", "argument": "patch", "limit": limit}
        assert result.structured_content == {"result": expected}
        _assert_argument_too_large(result.data, "patch", limit)


async def test_top_level_string_over_name_bound_is_refused() -> None:
    limit = ab.bound("memory_get", "memory_id")
    assert limit is not None
    data = await _call("memory_get", namespace=NS, memory_id="x" * (limit + 1))
    _assert_argument_too_large(data, "memory_id", limit)


async def _drive(
    app: Any, messages: list[dict[str, Any]], *, method: str = "POST", length: int | None = None
) -> list[dict[str, Any]]:
    """Run *app* behind the daemon's ``_IdleTracker`` on a request whose receive yields *messages*; what was sent."""
    from trw_memory.daemon._serve import _IdleTracker

    incoming = list(messages)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return incoming.pop(0) if incoming else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    headers = [] if length is None else [(b"content-length", str(length).encode())]
    scope = {"type": "http", "method": method, "path": "/mcp", "headers": headers, "query_string": b""}
    await _IdleTracker(app)(scope, receive, send)
    return sent


def _chunks(*bodies: bytes) -> list[dict[str, Any]]:
    return [{"type": "http.request", "body": b, "more_body": i < len(bodies) - 1} for i, b in enumerate(bodies)]


class _Recorder:
    """A stand-in app that reads messages until the body ends or the client goes, then answers 200."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            self.received.append(message)
            if message["type"] != "http.request" or not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    @property
    def body(self) -> bytes:
        return b"".join(m.get("body", b"") for m in self.received)


async def test_a_declared_body_past_the_cap_is_answered_413_before_anything_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ab, "MAX_BODY_BYTES", 10)
    app = _Recorder()
    sent = await _drive(app, _chunks(b"12345678901"), length=11)
    assert (sent[0]["status"], app.received) == (413, [])


async def test_an_undeclared_body_past_the_cap_ends_as_a_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab, "MAX_BODY_BYTES", 10)
    app = _Recorder()
    await _drive(app, _chunks(b"1234567", b"890", b"x", b"y"))
    assert app.body == b"1234567890"
    assert app.received[-1] == {"type": "http.disconnect"}


async def test_bodies_at_the_cap_and_bodiless_requests_reach_the_app_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab, "MAX_BODY_BYTES", 10)
    posted = _Recorder()
    assert (await _drive(posted, _chunks(b"1234567", b"890"), length=10))[0]["status"] == 200
    assert posted.body == b"1234567890"
    for method in ("GET", "DELETE"):
        app = _Recorder()
        await _drive(app, _chunks(b""), method=method)
        assert app.received == _chunks(b"")
    gone = _Recorder()
    await _drive(gone, [{"type": "http.disconnect"}])
    assert gone.received == [{"type": "http.disconnect"}]


async def test_a_malformed_or_huge_declared_length_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric length is left to the app; a numeric one too long to convert is simply over the cap."""
    from trw_memory.daemon._serve import _IdleTracker

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    for header, status in ((b"9" * 5000, 413), (b"abc", 200), (b"-1", 200)):
        sent.clear()
        scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": [(b"content-length", header)]}
        await _IdleTracker(_Recorder())(scope, receive, send)
        assert sent[0]["status"] == status, header[:10]
