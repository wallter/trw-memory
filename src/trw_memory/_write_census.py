"""The raw-write matcher both PRD-CORE-337 FR05 censuses share (test support, imported by no runtime module).

``trw-memory/tests/test_no_raw_checkout_writes_census.py`` and
``trw-mcp/tests/test_no_raw_checkout_writes_census.py`` each keep their own audited allowlist but
walk their trees with this one matcher, so a matcher fix lands in both at once (review round 1 found
the same blind spot in two hand-copied matchers). It lives in the package rather than under
``tests/`` for the reason ``trw_memory.storage.probe_fixtures`` does: both packages' test trees are
named ``tests``, so trw-mcp's tests cannot import trw-memory's, but they can import trw-memory.

A raw write is ``X.write_text(...)``, ``X.write_bytes(...)``, or an opener whose mode writes
(``w``, ``a``, ``x`` or ``+``): builtin ``open``, ``builtins.open``, ``io.open``, ``gzip``/``bz2``/
``lzma``/``codecs``/``tarfile``'s ``open``, ``os.fdopen``, a ``Path``-style ``X.open(mode)`` and ``shutil.copy``/``copy2``/``copyfile`` (a copy writes its
destination by name and follows a symlink there; PRD-CORE-337 found the live cursor hook scripts written this way).
``shutil.copytree``/``move`` stay uncounted: tree operations, audited by their callers.
Anything the AST cannot resolve counts as a write -- a non-literal mode, a ``*args`` or a
``**kwargs`` that could carry the mode -- because a false positive costs one reasoned allowlist row
while a false negative is an unguarded writer. ``os.open`` is not counted: it takes flags, not a
mode string, and it is how ``trw_memory.safe_fs`` itself writes.
"""

from __future__ import annotations

import ast
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

__all__ = ["CLASS_TAGS", "Site", "census", "is_raw_write", "ordered_sites", "raw_write_sites", "report"]

#: (relative path, enclosing qualname, 1-based ordinal of the write within that scope).
Site = tuple[str, str, int]

#: The class tag every allowlist row carries (PRD-CORE-337 FR05; charter Q2 "class before site"):
#:   migrates-in-FR06 -- a bootstrap generator file FR06 enumerates; Slice B deletes the row as it migrates.
#:   migrates-in-FR07 -- a channel writer file FR07 enumerates; Slice C deletes the row.
#:   migrates-in-FR08 -- one of R12's leaf-symlink-exposed ``.trw/`` state writers; Slice D deletes the row.
#:   migrates-in-FR09 -- the credentials create-then-chmod window; Slice E deletes the row.
#:   migrates-in-FR10 -- a ``.write_text`` whose receiver is trw-mcp's ``FileStateWriter`` (leaf-atomic,
#:     parent-exposed), or that helper's own writes; Slice F delegates it to ``safe_fs`` or re-tags it.
#:   unscheduled-checkout-write -- a write into a checkout path no FR06-FR10 slice enumerates. A REPORTED
#:     residual, not a safe exception: it needs its own follow-up (the PRD's non-goals call these stragglers).
#:   own-state-stays -- a write that never lands in a checkout (a private temp dir, a test-fixture builder).
#:   operator-named-path -- the destination is an explicit operator CLI argument, not a path TRW chooses.
CLASS_TAGS = frozenset(
    {
        "migrates-in-FR06",
        "migrates-in-FR07",
        "migrates-in-FR08",
        "migrates-in-FR09",
        "migrates-in-FR10",
        "unscheduled-checkout-write",
        "own-state-stays",
        "operator-named-path",
    }
)

#: ``M.open(path, mode)`` receivers: the mode is the SECOND argument.
_PATH_FIRST_OPENERS = frozenset({"builtins", "io", "gzip", "bz2", "lzma", "codecs", "tarfile"})
#: ``M.open`` receivers that are not a file opener with a mode string (``os.open`` takes flags).
_NOT_A_MODE_OPENER = frozenset({"os", "webbrowser", "zipfile"})
_WRITE_MODE_CHARS = frozenset("wax+")
#: ``shutil`` calls that write one destination file (``copytree``/``move`` are tree operations, named below).
_SHUTIL_FILE_COPIES = frozenset({"copy", "copy2", "copyfile"})


def _mode_writes(call: ast.Call, positional_index: int) -> bool:
    """Whether *call*'s mode (keyword ``mode`` or argument *positional_index*) can write."""
    mode: ast.expr | None = None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    if mode is None:
        unpacked = any(isinstance(arg, ast.Starred) for arg in call.args)
        if unpacked or any(keyword.arg is None for keyword in call.keywords):
            return True  # a *args/**kwargs could carry the mode; the AST cannot prove it reads
        if len(call.args) <= positional_index:
            return False  # no mode at all: the default "r"
        mode = call.args[positional_index]
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return bool(_WRITE_MODE_CHARS & set(mode.value))
    return True  # a non-literal mode


def is_raw_write(call: ast.Call) -> bool:
    """True when *call* is a ``write_text``/``write_bytes`` or an opener whose mode can write."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "open" and _mode_writes(call, 1)
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr in {"write_text", "write_bytes"}:
        return True
    receiver = func.value.id if isinstance(func.value, ast.Name) else None
    if receiver == "shutil" and func.attr in _SHUTIL_FILE_COPIES:
        return True  # writes its destination by name and follows a symlinked destination
    if func.attr == "fdopen":
        return receiver == "os" and _mode_writes(call, 1)
    if func.attr != "open" or receiver in _NOT_A_MODE_OPENER:
        return False
    return _mode_writes(call, 1 if receiver in _PATH_FIRST_OPENERS else 0)


def ordered_sites(tree: ast.Module, relative: str) -> list[Site]:
    """Every raw write in *tree*, keyed ``(relative, qualname, ordinal)`` and numbered in source order per scope."""
    located: list[tuple[str, int, int]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_scope = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_scope = child.name if scope == "<module>" else f"{scope}.{child.name}"
            if isinstance(child, ast.Call) and is_raw_write(child):
                located.append((scope, child.lineno, child.col_offset))
            visit(child, child_scope)

    visit(tree, "<module>")
    ordinals: dict[str, int] = {}
    ordered: list[Site] = []
    for scope, _line, _col in sorted(located, key=lambda item: (item[1], item[2])):
        ordinals[scope] = ordinals.get(scope, 0) + 1
        ordered.append((relative, scope, ordinals[scope]))
    return ordered


def raw_write_sites(root: Path, trees: Sequence[str], exclude: Collection[str] = ()) -> list[Site]:
    """Every raw write in the *trees* under *root* (paths relative to *root*), except the files in *exclude*.

    There is no ``SyntaxError`` escape hatch: an unparseable file must fail the census, not hide its writes.
    """
    sites: list[Site] = []
    for tree_name in trees:
        for path in sorted((root / tree_name).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative not in exclude:
                sites.extend(ordered_sites(ast.parse(path.read_text(encoding="utf-8")), relative))
    return sites


def census(found: Sequence[Site], audited: Mapping[Site, tuple[str, str]]) -> tuple[list[Site], list[Site]]:
    """(unlisted sites, stale rows): a write with no row, and a row with no write."""
    return sorted(set(found) - set(audited)), sorted(set(audited) - set(found))


def report(unlisted: Sequence[Site], stale: Sequence[Site]) -> str:
    """One line per failure, naming the path, qualname and ordinal."""
    lines = [f"unlisted raw checkout write: {path} :: {qualname} #{ordinal}" for path, qualname, ordinal in unlisted]
    lines += [
        f"stale allowlist row (no such write): {path} :: {qualname} #{ordinal}" for path, qualname, ordinal in stale
    ]
    return "\n".join(lines)
