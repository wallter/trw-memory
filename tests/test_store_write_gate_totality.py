"""Totality guard: no production ``backend.store()`` may skip the SEC-001 gate.

Five shipped write surfaces (the LangChain / CrewAI / LlamaIndex / VSCode
adapters and the ``trw-memory import`` CLI) each called ``backend.store(entry)``
directly, so caller-supplied content reached the store without the injection
gate, PII scan, anomaly scoring or provenance signature that
``prepare_entry_for_store`` applies. Fixing those five call sites does not stop a
sixth adapter from being written the same way, so the durable fix is this test:
it DERIVES every ``.store(...)`` call site from the source tree and requires each
one to be guarded, internal to ``security/``, or named in a justified exclusion
set. Hand-listing the call sites here would reproduce the subset-registry defect
(``docs/documentation/wiring-defect-patterns.md`` P11) inside its own fix.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import trw_memory

SRC_ROOT = Path(trw_memory.__file__).resolve().parent

#: Names that constitute "the entry passed the store gate" when called earlier in
#: the same function as the ``.store(...)``.
GATE_CALLS = frozenset({"prepare_entry_for_store", "guarded_store", "guarded_store_or_raise"})

#: Reason classes an exclusion may claim. Anything outside this vocabulary is a
#: new justification that needs review, not a free-text escape hatch.
REASON_CLASSES = frozenset(
    {
        # The backend's own machinery — this IS the storage layer, not a caller.
        "backend_internal",
        # Re-persists a MemoryEntry that already passed the gate on its way in.
        "rewrite_of_persisted_entry",
        # Writes an entry synthesised only from rows already in the store.
        "derived_from_gated_entries",
    }
)

#: Call sites that legitimately bypass the gate, keyed on
#: ``(path relative to trw_memory/, enclosing qualified name)``.
JUSTIFIED_INTERNAL_STORES: dict[tuple[str, str], str] = {
    ("storage/interface.py", "StorageBackend.store_many"): (
        "backend_internal: the default bulk-write fallback fans out to this "
        "backend's own store(); gating here would double-gate every caller."
    ),
    ("lifecycle/consolidation.py", "_restore_originals"): (
        "rewrite_of_persisted_entry: rolls a failed consolidation back by "
        "re-writing the exact entries it had just read out of the store."
    ),
    ("lifecycle/tiers/_runtime.py", "tier_candidates._restore_entry"): (
        "rewrite_of_persisted_entry: rehydrates a hot-tier row back into the "
        "backend it was evicted from; the content already passed intake."
    ),
    ("lifecycle/consolidation.py", "_create_consolidated_entry"): (
        "derived_from_gated_entries: content is summarised from a cluster of "
        "rows that each passed the gate at write time; no new caller input."
    ),
    ("tools/consolidate.py", "_promote_team_memories"): (
        "derived_from_gated_entries: republishes a stored team entry into the "
        "project namespace via model_copy; content is unchanged."
    ),
}


@dataclass(frozen=True)
class StoreCall:
    """One ``<receiver>.store(<entry>)`` call site found in the source tree."""

    module: str
    qualname: str
    lineno: int
    gate: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.module, self.qualname)

    @property
    def in_security_package(self) -> bool:
        return self.module.startswith("security/")


def _qualname(stack: list[ast.AST]) -> str:
    return ".".join(
        node.name for node in stack if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _is_store_call(node: ast.AST) -> bool:
    """Structural predicate — deliberately not keyed on receiver NAME.

    A name-based match ("call it a bypass if the receiver is called ``backend``")
    is the very indexing mistake this guard exists to prevent. The shape that
    identifies a persistence call is instead ``X.store(<one positional arg>)``
    with no keywords: ``MemoryClient.store(content, tags=..., importance=...)``
    passes keywords and is excluded structurally.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "store"):
        return False
    # `backend.store(entry)` — the ordinary shape.
    if len(node.args) == 1 and not node.keywords:
        return True
    # `backend.store(entry=e)` — same call, zero positional args. Requiring
    # "exactly one positional and no keywords" made this shape INVISIBLE to the
    # scanner: not a reported bypass, not even a candidate. An adapter written that
    # way escaped the guard entirely.
    return not node.args and len(node.keywords) == 1 and node.keywords[0].arg == "entry"


def _store_argument(node: ast.Call) -> ast.expr:
    """The single entry expression of a store call, positional or keyword."""
    return node.args[0] if node.args else node.keywords[0].value


def _gated_entry_names(func: ast.AST, before: int) -> dict[str, str]:
    """Map each expression holding the GATE'S OUTPUT to the gate that produced it.

    Keys are either ``"<result>.entry"`` or a plain local name rebound from one.
    Only assignments above line *before* count.

    Proximity is not the contract. The original rule — "some gate call appears
    earlier in this function" — clears a caller that runs the gate and then stores
    the ORIGINAL entry, discarding ``decision.entry``. That caller loses every
    mutation the pipeline applied (PII handling, provenance signature) and loses
    the quarantine verdict entirely, while scanning as fully guarded. Every
    current call site rebinds correctly (``entry = decision.entry``), so this
    tracks dataflow to keep it that way rather than to fix a live bypass.

    Four shapes, all of which occur in the production tree:
      1. ``d = gate(...)``            -> ``store(d.entry)``
      2. ``e = d.entry``              -> ``store(e)``
      3. ``e = <already-gated name>`` -> ``store(e)``
      4. ``acc.append(d.entry)`` then ``for ... e ... in <expr mentioning acc>``
         -> ``store(e)``   (``_client_bulk_store.bulk_store_impl``)

    Shape 4 resolves ``enumerate`` and ``zip`` POSITIONALLY. An earlier version
    marked every Name bound by the ``for`` target, which was permissive enough to
    clear a real bypass::

        for gated, raw in zip(accepted, raw_entries):
            backend.store(raw)          # ungated caller input, scanned as gated

    Only the target position fed by the gated accumulator is marked now, so that
    snippet is reported. ``test_scanner_rejects_a_positionally_ungated_loop_var``
    pins it.
    """
    results: dict[str, str] = {}
    entries: dict[str, str] = {}
    containers: dict[str, str] = {}

    for sub in ast.walk(func):
        if isinstance(sub, ast.Assign) and sub.lineno < before and len(sub.targets) == 1:
            target = sub.targets[0]
            value = sub.value
            if not isinstance(target, ast.Name):
                continue
            # d = prepare_entry_for_store(...)
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in GATE_CALLS:
                results[target.id] = value.func.id
            # e = d.entry
            elif isinstance(value, ast.Attribute) and value.attr == "entry" and isinstance(value.value, ast.Name):
                if value.value.id in results:
                    entries[target.id] = results[value.value.id]
            # e = <already-gated name>
            elif isinstance(value, ast.Name) and value.id in entries:
                entries[target.id] = entries[value.id]
        # acc.append(d.entry)
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "append"
            and sub.lineno < before
        ):
            if not isinstance(sub.func.value, ast.Name) or len(sub.args) != 1:
                continue
            appended = sub.args[0]
            if (
                isinstance(appended, ast.Attribute)
                and appended.attr == "entry"
                and isinstance(appended.value, ast.Name)
                and appended.value.id in results
            ):
                containers[sub.func.value.id] = results[appended.value.id]

    # for <target> in <iter>: bind positionally through enumerate()/zip().
    for sub in ast.walk(func):
        if isinstance(sub, ast.For) and sub.lineno < before:
            _bind_loop_target(sub.target, sub.iter, containers, entries)

    return {**{f"{name}.entry": gate for name, gate in results.items()}, **entries}


def _bind_loop_target(
    target: ast.expr,
    iterated: ast.expr,
    containers: dict[str, str],
    entries: dict[str, str],
) -> None:
    """Mark only the target POSITION that a gated accumulator actually feeds.

    Handles ``for e in acc``, ``for i, e in enumerate(acc)`` and
    ``for a, b in zip(x, acc)`` — including the nested
    ``for j, (i, e) in enumerate(zip(idx, acc))`` shape that
    ``_client_bulk_store.bulk_store_impl`` uses. Anything else is ignored rather
    than guessed at: an unrecognised iterable leaves the target unmarked, which is
    the conservative direction (the call site reports as a bypass and a human
    looks at it).
    """
    # for e in acc
    if isinstance(iterated, ast.Name):
        gate = containers.get(iterated.id, "")
        if gate and isinstance(target, ast.Name):
            entries[target.id] = gate
        return

    if not isinstance(iterated, ast.Call) or not isinstance(iterated.func, ast.Name):
        return

    # for i, e in enumerate(X)  ->  recurse on (e, X)
    if iterated.func.id == "enumerate" and iterated.args:
        if isinstance(target, ast.Tuple) and len(target.elts) == 2:
            _bind_loop_target(target.elts[1], iterated.args[0], containers, entries)
        return

    # for a, b in zip(X, Y)  ->  position i of the target pairs with arg i
    if iterated.func.id == "zip" and isinstance(target, ast.Tuple):
        for element, argument in zip(target.elts, iterated.args, strict=False):
            _bind_loop_target(element, argument, containers, entries)


def _gate_for(func: ast.AST | None, call: ast.Call) -> str:
    """Return the gate whose OUTPUT is the argument of *call*, or ``""``."""
    if func is None:
        return ""
    gated = _gated_entry_names(func, call.lineno)
    arg = _store_argument(call)
    if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
        return gated.get(f"{arg.value.id}.{arg.attr}", "")
    if isinstance(arg, ast.Name):
        return gated.get(arg.id, "")
    return ""


def scan_source(root: Path, tree: ast.AST, module: str) -> list[StoreCall]:
    """Collect every store call site in one parsed module."""
    found: list[StoreCall] = []
    stack: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        scoped = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if scoped:
            stack.append(node)
        if _is_store_call(node):
            enclosing = next(
                (n for n in reversed(stack) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                None,
            )
            assert isinstance(node, ast.Call)
            found.append(
                StoreCall(
                    module=module,
                    qualname=_qualname(stack),
                    lineno=node.lineno,
                    gate=_gate_for(enclosing, node),
                )
            )
        for child in ast.iter_child_nodes(node):
            visit(child)
        if scoped:
            stack.pop()

    visit(tree)
    return found


def scan_tree(root: Path) -> list[StoreCall]:
    """Derive every store call site under *root* (production source only)."""
    calls: list[StoreCall] = []
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        calls.extend(scan_source(root, ast.parse(path.read_text(encoding="utf-8")), module))
    return calls


def _bypasses(calls: list[StoreCall]) -> list[StoreCall]:
    return [
        call
        for call in calls
        if not call.in_security_package and not call.gate and call.key not in JUSTIFIED_INTERNAL_STORES
    ]


@pytest.fixture(scope="module")
def store_calls() -> list[StoreCall]:
    return scan_tree(SRC_ROOT)


class TestStoreGateTotality:
    """Every production store call site is guarded, internal, or justified."""

    def test_no_unguarded_store_call_sites(self, store_calls: list[StoreCall]) -> None:
        offenders = [f"{call.module}:{call.lineno} ({call.qualname})" for call in _bypasses(store_calls)]
        assert offenders == [], (
            "backend.store() call sites bypass the SEC-001 store gate. Route them "
            "through trw_memory.security.write_gate.guarded_store, or add a "
            "justified entry to JUSTIFIED_INTERNAL_STORES:\n  " + "\n  ".join(offenders)
        )

    def test_scanner_is_non_vacuous(self, store_calls: list[StoreCall]) -> None:
        """Control: the scanner must still see call sites of each known class."""
        keys = {call.key for call in store_calls}
        assert ("tools/store.py", "memory_store_impl") in keys
        assert ("_client_store.py", "store_impl") in keys
        assert ("security/keys.py", "rotate_master_key") in keys
        assert len(store_calls) >= 10, f"only {len(store_calls)} store call sites found — scanner likely broken"

    def test_known_guarded_sites_are_classified_guarded(self, store_calls: list[StoreCall]) -> None:
        """Precision: the pre-existing correct writers are not reported."""
        by_key = {call.key: call for call in store_calls}
        assert by_key[("tools/store.py", "memory_store_impl")].gate == "prepare_entry_for_store"
        assert by_key[("_client_store.py", "store_impl")].gate == "prepare_entry_for_store"
        assert by_key[("_client_bulk_store.py", "bulk_store_impl")].gate == "prepare_entry_for_store"

    def test_security_package_sites_are_not_reported(self, store_calls: list[StoreCall]) -> None:
        """Precision: key rotation re-encrypts in place and must not be flagged."""
        rotation = next(call for call in store_calls if call.key == ("security/keys.py", "rotate_master_key"))
        assert rotation.in_security_package
        assert rotation not in _bypasses(store_calls)

    def test_repaired_write_surfaces_are_no_longer_bypasses(self, store_calls: list[StoreCall]) -> None:
        """Attribution anchor: reverting any adapter puts it back in this list."""
        repaired = {
            ("integrations/langchain.py", "TRWChatMessageHistory.add_messages"),
            ("integrations/crewai.py", "TRWCrewStorage.save"),
            ("integrations/llamaindex.py", "TRWChatStore.add_message"),
            ("integrations/vscode.py", "LocalMemoryAdapter.store_selection"),
            ("cli_storage.py", "handle_import"),
        }
        assert repaired & {call.key for call in _bypasses(store_calls)} == set()


class TestScannerContracts:
    """The scanner itself must detect a bypass and clear a guarded caller."""

    def _scan_snippet(self, source: str) -> list[StoreCall]:
        return scan_source(SRC_ROOT, ast.parse(source), "synthetic/adapter.py")

    def test_detects_a_new_unguarded_adapter(self) -> None:
        calls = self._scan_snippet("def save(backend, entry):\n    backend.store(entry)\n")
        assert [call.gate for call in calls] == [""]
        assert _bypasses(calls) == calls

    def test_clears_a_guarded_caller(self) -> None:
        calls = self._scan_snippet(
            "def save(backend, entry):\n"
            "    decision = prepare_entry_for_store(entry, backend=backend, config=cfg)\n"
            "    backend.store(decision.entry)\n"
        )
        assert [call.gate for call in calls] == ["prepare_entry_for_store"]
        assert _bypasses(calls) == []

    def test_scanner_rejects_a_discarded_gate(self) -> None:
        """The gate ran, and its result was thrown away.

        This is the shape the old proximity rule ("some gate call appears earlier
        in this function") could not see. Storing the ORIGINAL entry discards every
        mutation the pipeline applied — PII handling, provenance signature — and
        discards the quarantine verdict entirely, while scanning as fully guarded.
        """
        calls = self._scan_snippet(
            "def save(backend, entry):\n"
            "    prepare_entry_for_store(entry, backend=backend, config=cfg)\n"
            "    backend.store(entry)\n"
        )
        assert [call.gate for call in calls] == [""]
        assert _bypasses(calls) == calls

    def test_scanner_clears_a_rebound_gated_entry(self) -> None:
        """Precision: ``entry = decision.entry`` is how all three real callers do it."""
        calls = self._scan_snippet(
            "def save(backend, entry):\n"
            "    decision = prepare_entry_for_store(entry, backend=backend, config=cfg)\n"
            "    entry = decision.entry\n"
            "    backend.store(entry)\n"
        )
        assert [call.gate for call in calls] == ["prepare_entry_for_store"]

    def test_scanner_clears_an_accumulated_gated_entry(self) -> None:
        """Precision: the bulk path accumulates gated entries and loops over them."""
        calls = self._scan_snippet(
            "def save_many(backend, requests):\n"
            "    accepted = []\n"
            "    for req in requests:\n"
            "        decision = prepare_entry_for_store(req, backend=backend, config=cfg)\n"
            "        accepted.append(decision.entry)\n"
            "    for i, entry in enumerate(accepted):\n"
            "        backend.store(entry)\n"
        )
        assert [call.gate for call in calls] == ["prepare_entry_for_store"]

    def test_scanner_rejects_a_positionally_ungated_loop_var(self) -> None:
        """Only the target POSITION the accumulator feeds may be cleared.

        Marking every name bound by the loop target cleared this real bypass: the
        gated entries and the raw caller input are zipped together and the RAW one
        is stored.
        """
        calls = self._scan_snippet(
            "def save_many(backend, requests, raw_entries):\n"
            "    accepted = []\n"
            "    for req in requests:\n"
            "        decision = prepare_entry_for_store(req, backend=backend, config=cfg)\n"
            "        accepted.append(decision.entry)\n"
            "    for gated, raw in zip(accepted, raw_entries):\n"
            "        backend.store(raw)\n"
        )
        assert [call.gate for call in calls] == [""]
        assert _bypasses(calls) == calls

    def test_scanner_clears_the_zipped_gated_position(self) -> None:
        """Precision companion: the gated position in the SAME zip is cleared."""
        calls = self._scan_snippet(
            "def save_many(backend, requests, raw_entries):\n"
            "    accepted = []\n"
            "    for req in requests:\n"
            "        decision = prepare_entry_for_store(req, backend=backend, config=cfg)\n"
            "        accepted.append(decision.entry)\n"
            "    for raw, gated in zip(raw_entries, accepted):\n"
            "        backend.store(gated)\n"
        )
        assert [call.gate for call in calls] == ["prepare_entry_for_store"]

    def test_scanner_sees_a_keyword_store_call(self) -> None:
        """``backend.store(entry=e)`` was invisible — not even a candidate."""
        calls = self._scan_snippet("def save(backend, entry):\n    backend.store(entry=entry)\n")
        assert len(calls) == 1
        assert _bypasses(calls) == calls

    def test_a_later_gate_cannot_clear_an_earlier_store(self) -> None:
        """Line ordering applies to the accumulator and loop rules too.

        The append/for passes had no ``lineno`` guard, so a gated accumulator built
        BELOW a bypass retroactively cleared it.
        """
        calls = self._scan_snippet(
            "def save(backend, entry, requests):\n"
            "    backend.store(entry)\n"
            "    accepted = []\n"
            "    for req in requests:\n"
            "        decision = prepare_entry_for_store(req, backend=backend, config=cfg)\n"
            "        accepted.append(decision.entry)\n"
            "    for entry in accepted:\n"
            "        pass\n"
        )
        assert [call.gate for call in calls] == [""]

    def test_gate_called_after_the_store_does_not_count(self) -> None:
        """Ordering matters: a gate call below the write guards nothing."""
        calls = self._scan_snippet(
            "def save(backend, entry):\n"
            "    backend.store(entry)\n"
            "    prepare_entry_for_store(entry, backend=backend, config=cfg)\n"
        )
        assert [call.gate for call in calls] == [""]

    def test_client_store_with_keywords_is_not_a_backend_write(self) -> None:
        """Precision: ``MemoryClient.store(content, tags=...)`` is not a row write."""
        calls = self._scan_snippet("async def run(client):\n    await client.store(content, tags=tags)\n")
        assert calls == []


class TestExclusionSetIsJustified:
    """The exclusion set is the mechanism, so it needs its own contract."""

    def test_every_exclusion_resolves_to_a_live_unguarded_call_site(self, store_calls: list[StoreCall]) -> None:
        """A stale exclusion silently re-widens the gate — fail on it.

        "Live" means the site still exists AND still needs the exclusion: an
        entry whose call site has since been guarded or moved into ``security/``
        is dead weight that would keep covering a future rewrite of that
        function.
        """
        needs_exclusion = {call.key for call in store_calls if not call.gate and not call.in_security_package}
        stale = sorted(key for key in JUSTIFIED_INTERNAL_STORES if key not in needs_exclusion)
        assert stale == [], f"exclusions no longer cover an unguarded call site: {stale}"

    def test_every_exclusion_declares_a_known_reason_class(self) -> None:
        for key, reason in JUSTIFIED_INTERNAL_STORES.items():
            reason_class, _, rationale = reason.partition(": ")
            assert reason_class in REASON_CLASSES, f"{key}: unknown reason class {reason_class!r}"
            assert len(rationale.split()) >= 8, f"{key}: rationale is too thin to review"

    def test_no_exclusion_covers_a_security_package_site(self, store_calls: list[StoreCall]) -> None:
        """``security/`` sites are cleared by location; listing one hides drift."""
        assert not any(key[0].startswith("security/") for key in JUSTIFIED_INTERNAL_STORES)
