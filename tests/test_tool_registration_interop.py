"""Interop tests for registered MCP tool wrappers and MemoryClient."""

from __future__ import annotations

import ast
import importlib
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from trw_memory.client import MemoryClient
from trw_memory.tools.recall import register_recall_tool
from trw_memory.tools.search import register_search_tool
from trw_memory.tools.store import register_store_tool


class _FakeMCP:
    """Minimal FastMCP-like registry used to capture registered tool callables."""

    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., Coroutine[Any, Any, dict[str, object]]]] = {}

    def tool(
        self,
    ) -> Callable[
        [Callable[..., Coroutine[Any, Any, dict[str, object]]]],
        Callable[..., Coroutine[Any, Any, dict[str, object]]],
    ]:
        def _decorator(
            fn: Callable[..., Coroutine[Any, Any, dict[str, object]]],
        ) -> Callable[..., Coroutine[Any, Any, dict[str, object]]]:
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


async def test_registered_store_tool_writes_visible_to_client(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Tool wrapper and client must resolve the same namespace-scoped backend."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    mcp = _FakeMCP()
    register_store_tool(mcp)

    await mcp.tools["memory_store"](
        content="stored via tool wrapper",
        namespace="project:interop",
        tags=["interop"],
    )

    client = MemoryClient(namespace="project:interop", mode="local")
    try:
        results = await client.search(tags=["interop"])
    finally:
        await client.close()

    assert [result["content"] for result in results] == ["stored via tool wrapper"]


async def test_registered_search_tool_reads_entries_stored_by_client(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Tool wrapper must see entries created through the local client."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    client = MemoryClient(namespace="project:interop", mode="local")
    try:
        await client.store("stored via client", tags=["interop"])
    finally:
        await client.close()

    mcp = _FakeMCP()
    register_search_tool(mcp)
    result = await mcp.tools["memory_search"](
        namespace="project:interop",
        tags=["interop"],
    )

    entries = result["entries"]
    assert isinstance(entries, list)
    assert [entry["content"] for entry in entries] == ["stored via client"]


async def test_registered_recall_tool_accepts_source_aware_args(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Registered recall tool must expose the same source-aware policy args as the client."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    client = MemoryClient(namespace="project:interop", mode="local")
    try:
        await client.store(
            "durable rule",
            metadata={"source_kind": "instruction_rule"},
            entry_id="M-durable",
        )
        await client.store(
            "ephemeral bulletin",
            metadata={"source_kind": "lifecycle"},
            expires="2020-01-01T00:00:00+00:00",
            entry_id="M-expired",
        )
    finally:
        await client.close()

    mcp = _FakeMCP()
    register_recall_tool(mcp)
    result = await mcp.tools["memory_recall"](
        query="",
        namespace="project:interop",
        include_source_kinds=["instruction_rule", "lifecycle"],
        exclude_expired=True,
    )

    memories = result["memories"]
    assert isinstance(memories, list)
    assert [entry["id"] for entry in memories] == ["M-durable"]


# --- PRD-SEC-016 FR05: no served tool body touches the filesystem directly ---
#
# Round-2 finding 3 (closed by AST call-graph reachability): a substring
# search for an opener marker anywhere in a module's TEXT let a comment, or a
# compliant sibling tool sharing the module, clear a non-compliant one.
# Round-2 finding 5 (closed by per-PARAMETER argument tracking): calling an
# opener on any value, not the tainted one, still cleared a tool whose real
# path parameter reached a raw sink directly.
#
# Round-4 finding 2: per-parameter tracking is STILL evadable by an alias --
#
#     async def memory_planted(root: str) -> str:
#         checkout_path(root, "op", within=True)  # satisfies "root reaches an opener"
#         p = root                                # ... but the raw value survives under a new name
#         with open(p) as handle:                 # and THIS bare-name check never sees `p`
#             return handle.read()
#
# Bare-argument alias tracking is unbounded in general (any number of
# reassignments, attribute copies, tuple unpacking, ...); chasing it adds
# scope without closing the gap for good. The evasion-proof, and simpler,
# rule instead: a served tool's OWN function body may never call a raw
# filesystem primitive AT ALL, regardless of which value it is given. Real
# filesystem access happens in a NAMED module-level helper (which calls
# ``checkout_path``/``open_checkout_file_fd``), never inline in the wrapper.
# This also means the scan no longer needs to know which parameter NAMES are
# path-shaped -- it checks every served tool, unconditionally.

#: Raw filesystem primitives a served tool's own body must never call, period -- builtin ``open``;
#: ``Path`` read/write methods; ``os.open`` (bypasses the dir_fd-anchored walk entirely); and the
#: ``shutil`` copy/move/tree family, which round-2 finding 3's planted-bypass tests named.
_SINK_NAME_CALLS = frozenset({"open"})
_SINK_ATTR_CALLS = frozenset({"read_bytes", "read_text", "write_bytes", "write_text"})
_SINK_DOTTED_CALLS = frozenset(
    {"os.open"} | {f"shutil.{name}" for name in ("copy", "copy2", "copyfile", "copytree", "move", "rmtree")}
)


def _call_target(call: ast.Call) -> str | None:
    """A name for *call*'s target: ``"open"``, ``"os.open"``, ``"read_bytes"``, or ``None`` if unresolvable."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
        return func.attr
    return None


def _calls_a_raw_sink(node: ast.AST) -> bool:
    """Whether *node*'s subtree (a served tool's own body, nested lambdas included) calls a raw sink."""
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        target = _call_target(call)
        if target in _SINK_NAME_CALLS or target in _SINK_DOTTED_CALLS:
            return True
        if isinstance(call.func, ast.Attribute) and call.func.attr in _SINK_ATTR_CALLS:
            return True
    return False


def _opener_violations(module_path: Path) -> list[str]:
    """Names of every served ``async def`` in *module_path* that calls a raw filesystem sink directly.

    No alias, argument, or parameter-name tracking (round-4 finding 2): a
    function is flagged the moment ANY sink call appears anywhere in its own
    body, independent of which value -- or which name that value currently
    goes by -- is passed to it.
    """
    tree = ast.parse(module_path.read_text())
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and _calls_a_raw_sink(node)]


def _served_tool_module_paths() -> list[Path]:
    """Every module ``trw_memory.server._register_tools`` imports a ``register_*`` function from."""
    import trw_memory.server as server_module

    tree = ast.parse(Path(server_module.__file__).read_text())
    module_names = sorted(
        {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("trw_memory.tools.")
        }
    )
    return [Path(importlib.import_module(name).__file__) for name in module_names]  # type: ignore[arg-type]


def test_no_served_tool_calls_a_raw_filesystem_sink_directly() -> None:
    """PRD-SEC-016 FR05 (round-4 finding 2's simplification): the scan passes on the current tree.

    Every served tool wrapper delegates its actual filesystem access to a
    named module-level helper (which itself calls ``checkout_path``/
    ``open_checkout_file_fd``); none call ``open``/``os.open``/``shutil.*``/
    ``Path.{read,write}_{bytes,text}`` inline. (code-index's unbounded root
    is separately gone entirely -- FR01.)
    """
    violations = {str(module): names for module in _served_tool_module_paths() if (names := _opener_violations(module))}

    assert violations == {}


def test_a_planted_tool_without_the_opener_fails_the_scan(tmp_path: Path) -> None:
    """The scan mechanism itself is not vacuous: a tool reaching the filesystem without the opener fails."""
    planted = tmp_path / "planted_tool.py"
    planted.write_text(
        "async def memory_planted(root: str) -> str:\n    with open(root) as handle:\n        return handle.read()\n"
    )

    violations = _opener_violations(planted)

    assert violations == ["memory_planted"]


def test_a_planted_tool_that_calls_checkout_path_passes_the_scan(tmp_path: Path) -> None:
    """A tool that DOES route its path parameter through the opener is not flagged."""
    planted = tmp_path / "planted_tool_ok.py"
    planted.write_text(
        "from trw_memory.tools.entry import checkout_path\n\n"
        "async def memory_planted_ok(root: str) -> str | dict[str, object] | None:\n"
        "    return checkout_path(root, 'memory_planted_ok', within=False)\n"
    )

    assert _opener_violations(planted) == []


def test_a_comment_only_mention_of_the_opener_does_not_clear_the_scan(tmp_path: Path) -> None:
    """Round-2 finding 3, part 1: an opener name in a COMMENT is not a call and must not clear the tool."""
    planted = tmp_path / "planted_tool_comment.py"
    planted.write_text(
        "# TODO: route this through checkout_path( ) and open_checkout_file_fd( ) one day\n"
        "async def memory_planted(root: str) -> str:\n    with open(root) as handle:\n        return handle.read()\n"
    )

    assert _opener_violations(planted) == ["memory_planted"]


@pytest.mark.parametrize(
    ("bypass_name", "bypass_body"),
    [
        ("memory_bypass_open", "    with open(root) as handle:\n        return handle.read()\n"),
        ("memory_bypass_read_bytes", "    from pathlib import Path\n\n    return Path(root).read_bytes()\n"),
        ("memory_bypass_os_open", "    import os\n\n    fd = os.open(root, os.O_RDONLY)\n    return str(fd)\n"),
        (
            "memory_bypass_shutil",
            "    import shutil\n    import tempfile\n\n    dest = tempfile.mktemp()\n"
            "    shutil.copyfile(root, dest)\n    return dest\n",
        ),
    ],
)
def test_a_second_noncompliant_tool_in_a_compliant_module_still_fails_the_scan(
    tmp_path: Path, bypass_name: str, bypass_body: str
) -> None:
    """Round-2 finding 3, part 2: one compliant tool in a module must not launder a sibling bypass tool.

    Each parametrization plants a DIFFERENT raw filesystem primitive
    (``open``, ``Path.read_bytes``, ``os.open``, ``shutil.copyfile``) as the
    bypass, alongside a genuinely compliant sibling tool in the SAME module
    -- reproducing the exact shape the module-wide substring scan missed.
    """
    planted = tmp_path / f"planted_mixed_{bypass_name}.py"
    planted.write_text(
        "from trw_memory.tools.entry import checkout_path\n\n"
        "async def memory_compliant(root: str) -> str | dict[str, object] | None:\n"
        "    return checkout_path(root, 'memory_compliant', within=False)\n\n\n"
        f"async def {bypass_name}(root: str) -> str:\n{bypass_body}"
    )

    violations = _opener_violations(planted)

    assert violations == [bypass_name], f"the compliant sibling must not clear {bypass_name}"


def test_an_opener_call_on_a_different_value_still_fails_because_the_sink_is_present(tmp_path: Path) -> None:
    """PRD-SEC-016 round-2 finding 5's original reproduction still flags -- now for the simpler reason.

    The function calls ``checkout_path`` (on `other`, a value unrelated to
    the eventual sink), AND separately calls a raw ``open()`` on `root`. The
    round-4 rule does not care which parameter the opener call used, or
    which parameter the sink call used -- ANY sink call in a served tool's
    own body is a violation on its own, full stop.
    """
    planted = tmp_path / "planted_wrong_arg.py"
    planted.write_text(
        "from trw_memory.tools.entry import checkout_path\n\n"
        "async def memory_planted(root: str, other: str) -> str:\n"
        "    checkout_path(other, 'memory_planted', within=True)\n"
        "    with open(root) as handle:\n"
        "        return handle.read()\n"
    )

    violations = _opener_violations(planted)

    assert violations == ["memory_planted"]


def test_an_alias_of_the_path_parameter_still_fails_the_scan(tmp_path: Path) -> None:
    """PRD-SEC-016 round-4 finding 2: the exact evasion a per-parameter bare-argument scan could not see.

    ``p = root`` renames the tainted value; a scan that only checks whether
    the ORIGINAL parameter name is a bare argument to a sink call never
    notices `p`. The round-4 rule does not track values or names at all --
    it flags the ``open(...)`` call itself, so no alias depth evades it.
    """
    planted = tmp_path / "planted_alias.py"
    planted.write_text(
        "from trw_memory.tools.entry import checkout_path\n\n"
        "async def memory_planted(root: str) -> str:\n"
        "    checkout_path(root, 'memory_planted', within=True)\n"
        "    p = root\n"
        "    with open(p) as handle:\n"
        "        return handle.read()\n"
    )

    assert _opener_violations(planted) == ["memory_planted"]


def test_a_tool_that_never_calls_a_raw_sink_is_not_flagged(tmp_path: Path) -> None:
    """The ordinary, correct case: a tool that only calls the opener (or nothing filesystem-related) passes."""
    planted = tmp_path / "planted_correct.py"
    planted.write_text(
        "from trw_memory.tools.entry import checkout_path\n\n"
        "async def memory_planted(root: str) -> str | dict[str, object] | None:\n"
        "    validated = checkout_path(root, 'memory_planted', within=True)\n"
        "    return validated\n"
    )

    assert _opener_violations(planted) == []


def test_a_tool_with_no_path_shaped_parameter_name_is_still_scanned(tmp_path: Path) -> None:
    """Round-4 simplification: the scan no longer depends on a `_PATH_PARAM_NAMES` allowlist.

    A parameter named ``db_file`` -- not ``source_path``/``project_root``/
    ``root``/``path`` -- would have been INVISIBLE to the round-2/round-3
    per-parameter scan entirely. The round-4 rule checks every served async
    tool's body for a raw sink call regardless of any parameter's name.
    """
    planted = tmp_path / "planted_unnamed_param.py"
    planted.write_text(
        "async def memory_planted(db_file: str) -> str:\n    with open(db_file) as handle:\n        return handle.read()\n"
    )

    assert _opener_violations(planted) == ["memory_planted"]


def test_a_tool_with_no_filesystem_touching_parameter_is_not_flagged(tmp_path: Path) -> None:
    """A parameter that is only compared/logged, never opened, triggers no violation."""
    planted = tmp_path / "planted_unused_path.py"
    planted.write_text("async def memory_planted(root: str) -> bool:\n    return root == '/tmp'\n")

    assert _opener_violations(planted) == []
