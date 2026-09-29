"""Skip-reason category registry and the static skip-site census (PRD-QUAL-146 FR10, backlog B80-67).

Every ``pytest.skip`` / ``pytest.xfail`` / ``pytest.importorskip`` call and every
``pytest.mark.skip`` / ``skipif`` / ``xfail`` marker under this tests directory must
map to exactly one registered category. ``test_skip_census.py`` fails on any site
that does not, so a rotting skip cannot land without saying why it is legitimate.

Classification, in order:

1. a ``# skip-category: <name>`` comment anywhere in the site's source span
   (the only way to classify a reason that is not a literal);
2. ``importorskip`` is ``optional-dependency`` by construction;
3. the literal reason text (a string constant, an f-string's literal parts, an
   implicit concatenation, or a module-level string constant it names) matched
   against ``CATEGORIES`` in order.

Soundness scope: proves each static site has a reason that maps to a registered
category. It does not prove the skip condition is correct, and it does not see
skips spelled outside the ``pytest.<name>`` call shapes above.

This module is duplicated verbatim (engine) in trw-mcp/tests and trw-memory/tests:
the two packages publish as separate mirrors and neither test tree may import the
other's.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

_I = re.IGNORECASE

# Ordered: the first pattern that matches the reason text wins.
CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("census-drained", re.compile(r"every audited write has migrated", _I)),
    ("tracked-defect", re.compile(r"PRD-[A-Z]+-\d+|\bB\d{2}-\d+|strict=True means", _I)),
    (
        "opt-in",
        re.compile(
            r"timing job|host-resource budget|fixture not selected|live learning corpus|large corpus"
            r"|live \.trw/runs|valid real PRDs|developer requirements\.lock|no clean measurement",
            _I,
        ),
    ),
    (
        "layout",
        re.compile(
            r"monorepo|mirror|standalone|public repo|exported tree|packaged|package data|bundled"
            r"|not present|absent|not built|in this checkout|not a git checkout|No \.md|No \.sh"
            r"|No skill|No files in skill|No agents|no tracked in-scope file|not found in package",
            _I,
        ),
    ),
    ("privilege", re.compile(r"\broot\b|non-root|unreadable|mode 000|as this user", _I)),
    (
        "platform",
        re.compile(
            r"windows|win32|posix|linux|darwin|macos|\bBSD\b|\bfork\b|fifo|SIGUSR1|SIGTERM|flock|advisory.?lock"
            r"|/proc|O_PATH|openat|dir_fd|symlink|mode bits|mode/owner|owner/mode|inode|Python 3\.\d+|3\.\d+\+"
            r"|kernel|sandbox-exec|write-confinement|process.group|zombies|Metal|RLIMIT|AF_UNIX|reparenting"
            r"|descriptor exhaustion|--relative-paths|FTS5|this platform|os\.fork",
            _I,
        ),
    ),
    (
        "host-tool",
        re.compile(
            r"\b(git|jq|sh|dash|bash|gpg|timeout|grep|python3?|ssh-keygen|uv|strings)\b|Codex CLI"
            r"|Codex Rust binary|codex binary|interpreter|launcher|entry point|required tool",
            _I,
        ),
    ),
    (
        "optional-dependency",
        re.compile(
            r"not installed|\[embeddings\]|sqlite-vec|xdist|sentence-transformers|numpy|alongside trw-memory"
            r"|model loader|did not load",
            _I,
        ),
    ),
)
CATEGORY_NAMES: frozenset[str] = frozenset(name for name, _ in CATEGORIES)

_SKIP_CALLS = frozenset({"skip", "xfail", "importorskip"})
_SKIP_MARKS = frozenset({"skip", "skipif", "xfail"})
_COMMENT = re.compile(r"#\s*skip-category:\s*([a-z-]+)")


@dataclass(frozen=True)
class SkipSite:
    path: str
    line: int
    kind: str
    reason: str
    category: str | None


def _pytest_chain(node: ast.expr) -> list[str] | None:
    """``pytest.mark.skipif`` -> ["mark", "skipif"]; None when the chain is not rooted at ``pytest``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "pytest":
        return parts[::-1]
    return None


def _module_strings(tree: ast.Module) -> dict[str, str]:
    consts: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            text = _literal(stmt.value, {})
            if text:
                consts[stmt.targets[0].id] = text
    return consts


def _literal(node: ast.expr | None, consts: dict[str, str]) -> str:
    """The literal text of a reason expression; '' when none is statically visible."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_literal(v, consts) for v in node.values if isinstance(v, ast.Constant))
    if isinstance(node, ast.Name):
        return consts.get(node.id, "")
    if isinstance(node, ast.BoolOp):  # `_REASON or ""`
        return " ".join(_literal(v, consts) for v in node.values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _literal(node.func.value, consts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal(node.left, consts) + _literal(node.right, consts)
    return ""


def _reason_node(call: ast.Call, positional_reason: bool) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg in {"reason", "msg"}:
            return kw.value
    if positional_reason and call.args:
        return call.args[0]
    return None


def classify(reason: str, kind: str, comment: str | None) -> str | None:
    if comment is not None:
        return comment if comment in CATEGORY_NAMES else None
    if kind == "importorskip":
        return "optional-dependency"
    for name, pattern in CATEGORIES:
        if name == "tracked-defect" and not kind.endswith("xfail"):
            continue  # an id in a skip reason is provenance; only an xfail tracks a defect
        if reason and pattern.search(reason):
            return name
    return None


def scan_source(source: str, path: str) -> list[SkipSite]:
    tree = ast.parse(source)
    lines = source.splitlines()
    consts = _module_strings(tree)
    sites: list[SkipSite] = []
    seen: set[int] = set()
    for node in ast.walk(tree):
        call = node if isinstance(node, ast.Call) else None
        target = call.func if call is not None else node
        if not isinstance(target, ast.expr) or id(target) in seen:
            continue
        chain = _pytest_chain(target)
        if chain is None:
            continue
        if len(chain) == 1 and chain[0] in _SKIP_CALLS and call is not None:
            kind, positional = chain[0], chain[0] != "importorskip"
        elif len(chain) == 2 and chain[0] == "mark" and chain[1] in _SKIP_MARKS:
            kind, positional = f"mark.{chain[1]}", chain[1] == "skip"
        else:
            continue
        seen.add(id(target))
        span_node = call if call is not None else target
        span = lines[span_node.lineno - 1 : (span_node.end_lineno or span_node.lineno)]
        match = next((m for m in (_COMMENT.search(line) for line in span) if m), None)
        reason = _literal(_reason_node(call, positional), consts) if call is not None else ""
        category = classify(reason, kind, match.group(1) if match else None)
        sites.append(SkipSite(path, span_node.lineno, kind, reason, category))
    return sites


def scan_tree(root: Path) -> list[SkipSite]:
    sites: list[SkipSite] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        sites.extend(scan_source(path.read_text(encoding="utf-8"), str(path.relative_to(root))))
    return sites
