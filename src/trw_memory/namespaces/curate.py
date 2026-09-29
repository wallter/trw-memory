"""Namespace rename, merge and moved-checkout detection -- FR01 + FR05.

The FR01 identity is a digest over a checkout's canonical root, which is stable
while the path is stable and changes when the path does. That is the right
trade -- it separates two clones an operator may want kept apart -- but it means
a **moved or renamed checkout orphans its rows** under the old namespace. These
are the repair verbs that make that recoverable, and the detector that tells an
operator the repair is needed.

Nothing here runs automatically. A silent auto-merge on a path change would be
indistinguishable from two genuinely different projects that happened to occupy
the same path over time, so the detector reports and the operator decides.

Both write verbs are deliberately conservative about loss:

* ``rename`` refuses a destination that already has rows. That case is a merge,
  and making the caller say so prevents an accidental silent union.
* ``merge`` keeps the DESTINATION row when an id exists in both namespaces and
  leaves the source row in place, so a conflict is reported rather than
  resolved by overwriting something the operator never named.
* a row's dense vector travels with it, so a re-key does not silently demote
  moved rows to keyword-only retrieval.

Both verbs take a SOURCE and a DESTINATION backend. Under the layout that ships
today two namespaces live in two SQLite files, so a move crosses stores; once
FR02 consolidates them into the single user-space file the caller passes the
same backend twice and the whole move runs inside one transaction. Handling
both is why the pair is explicit rather than assumed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from trw_memory.exceptions import ConfigError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.interface import EntryCursor, StorageBackend, is_transactional

__all__ = [
    "MoveProgress",
    "MovedCheckoutObservation",
    "NamespaceCurateResult",
    "NamespaceRowCount",
    "NamespaceStores",
    "detect_moved_checkout",
    "merge_namespace",
    "move_batch",
    "rename_namespace",
    "store_census",
]

logger = structlog.get_logger(__name__)

#: Rows hydrated per pass. Large enough that a typical namespace moves in one
#: pass, small enough that a 12k-row store never holds the whole corpus twice.
_BATCH_LIMIT = 2_000

#: The namespace scope project identities live under (FR01).
_PROJECT_SCOPE = "project:"


class NamespaceCurateResult(BaseModel):
    """What a rename or merge actually did."""

    source: str = Field(description="Namespace rows were taken from")
    destination: str = Field(description="Namespace rows were written to")
    source_rows: int = Field(ge=0, description="Rows found in the source before the operation")
    moved: int = Field(ge=0, description="Rows re-labelled onto the destination")
    skipped: int = Field(ge=0, description="Rows left in the source because the destination already held that id")
    status: Literal["renamed", "merged", "noop", "moving"] = Field(description="Outcome class")
    complete: bool = Field(default=True, description="False: the source is not drained yet; call again to resume")


class NamespaceRowCount(BaseModel):
    """A namespace and how many rows it holds."""

    namespace: str
    rows: int = Field(ge=0)


class MovedCheckoutObservation(BaseModel):
    """Evidence that a checkout was moved or renamed. A report, never a repair."""

    current_namespace: str = Field(description="The identity this checkout resolves to now")
    current_rows: int = Field(ge=0, description="Rows under the current identity; the signal requires 0")
    candidates: list[NamespaceRowCount] = Field(description="Populated same-slug namespaces with a different digest")
    repair_command: str = Field(description="The exact command an operator runs to carry the rows forward")


@dataclass(frozen=True)
class NamespaceStores:
    """The backends a curate verb reads from and writes to.

    ``source is destination`` when both namespaces live in one file, which is
    the FR02 end state and the only shape in which the move is a single
    transaction. When they differ the move is per-row atomic across two stores;
    the counts a verb returns are what the caller reconciles against.
    """

    source: StorageBackend
    destination: StorageBackend

    @classmethod
    def shared(cls, backend: StorageBackend) -> NamespaceStores:
        """Both namespaces in one store -- one backend, one transaction."""
        return cls(source=backend, destination=backend)

    @contextlib.contextmanager
    def destination_transaction(self) -> Iterator[None]:
        """Open the destination's transaction unless it IS the source's."""
        if self.destination is self.source:
            yield
            return
        with self.destination.transaction():
            yield


def _slug_of(namespace: str) -> str | None:
    """Return the slug half of ``project:<slug>-<digest8>``, or None."""
    if not namespace.startswith(_PROJECT_SCOPE):
        return None
    remainder = namespace[len(_PROJECT_SCOPE) :]
    slug, separator, _digest = remainder.rpartition("-")
    return slug if separator and slug else None


@dataclass(frozen=True)
class MoveProgress:
    """Where one rename or merge stands between its batches (B71-96).

    *cursor* resumes the keyset scan; *skipped* counts the conflicts this pass left in the source and
    *pass_moved* the rows it moved; *moved* counts every row the move put in the destination, which is
    how a resumed rename tells its own half-moved destination from one holding rows it did not move.
    """

    merge: bool
    started: bool = False
    complete: bool = False
    source_rows: int = 0
    moved: int = 0
    skipped: int = 0
    pass_moved: int = 0
    cursor: EntryCursor | None = None

    def result(self, source: str, destination: str) -> NamespaceCurateResult:
        """The reply: ``moving`` until the source is verified drained, then ``merged``/``renamed`` (``noop``)."""
        done = ("merged" if self.merge else "renamed") if self.source_rows else "noop"
        return NamespaceCurateResult(
            source=source,
            destination=destination,
            source_rows=self.source_rows,
            moved=self.moved,
            skipped=self.skipped,
            status=done if self.complete else "moving",
            complete=self.complete,
        )


def move_batch(
    stores: NamespaceStores, source: str, destination: str, progress: MoveProgress, *, limit: int
) -> MoveProgress:
    """Move the next *limit* rows of *source* after ``progress.cursor`` (with their edges and vectors) onto
    *destination* in one transaction; return the progress to resume from.

    Paging is by KEYSET cursor, not by a repeated ``limit``-window read: a skipped conflict stays in the
    source, so a window read re-serves it every pass, and the loop that de-duplicated it in memory once
    broke on the empty window and stranded every row ranked below it. The pass that reaches the end runs
    the drained-source check (:func:`_end_of_pass`).

    Raises:
        ConfigError: If either namespace is invalid, the two are equal, or a rename's destination holds
            a row this rename did not move there (that case is a merge, and the caller must say so).
        StorageError: If a page failed to advance the cursor, or the source cannot drain. Raised INSIDE
            the transaction, so a transactional backend (SQLite) rolls the batch back; a backend whose
            ``transaction()`` is the ABC's no-op default (YAML) keeps what it wrote.
    """
    source, destination = _validate_pair(source, destination)
    # B71-61: a rename's destination check only holds the write lock through the move on a backend whose
    # transaction() is a real BEGIN IMMEDIATE (SQLite); YAML's writers take no shared lock, so a rename onto
    # a non-transactional destination refuses instead of racing (as quarantine approval does, B71-32).
    if not progress.merge and not is_transactional(stores.destination):
        raise ConfigError(
            f"refusing to rename {source!r} onto {destination!r}: the destination backend does not "
            "support transactions, so the destination check and the move cannot be made atomic. "
            "Merge instead, or rename onto a transactional backend."
        )
    with stores.source.transaction(), stores.destination_transaction():
        if not progress.started:  # an empty source is a noop, checked BEFORE the destination
            progress = replace(progress, started=True, source_rows=stores.source.count(namespace=source))
            if not progress.source_rows:
                return replace(progress, complete=True)
        batch = stores.source.list_entries(namespace=source, limit=limit, after=progress.cursor)
        if not batch:
            return _end_of_pass(stores, source, destination, progress)
        # Checked under the write lock with every batch (rc5 C12): between two batches another tenant can land a row.
        if not progress.merge and (held := stores.destination.count(namespace=destination)) != progress.moved:
            raise ConfigError(
                f"refusing to rename {source!r} onto {destination!r}: the destination holds {held} rows and this "
                f"rename moved {progress.moved} there. Use merge if folding them together is what you mean (it "
                "also finishes a rename a daemon restart interrupted)."
            )
        cursor = EntryCursor.from_entry(batch[-1])
        if cursor == progress.cursor:
            raise StorageError(
                f"refusing to loop moving {source!r} onto {destination!r}: the keyset cursor "
                f"at {cursor.entry_id!r} did not advance"
            )
        ids = {entry.id for entry in batch}
        # Edges first: deleting a moved row purges its edges from the source.
        edges = [edge for edge in stores.source.graph_edges(source) if {edge.source_id, edge.target_id} & ids]
        stores.destination.add_graph_edges(destination, edges)
        moved, skipped = _move_entries(stores, source, destination, batch, skip_conflicts=progress.merge)
        progress = replace(
            progress,
            cursor=cursor,
            moved=progress.moved + moved,
            skipped=progress.skipped + skipped,
            pass_moved=progress.pass_moved + moved,
        )
        # A short page is the end of the scan: check it in this batch rather than one more job.
        return _end_of_pass(stores, source, destination, progress) if len(batch) < limit else progress


def _move_entries(
    stores: NamespaceStores, source: str, destination: str, batch: list[MemoryEntry], *, skip_conflicts: bool
) -> tuple[int, int]:
    """Re-label *batch* onto *destination*, each row's dense vector with it; return (moved, skipped)."""
    records = (
        stores.source.get_vector_records([entry.id for entry in batch], namespace=source)
        if stores.source.supports_vectors()
        else {}
    )
    # Preserve unknown legacy bytes for backends without record reads,
    # but never widen a scoped copy to an unqualified ID lookup.
    missing = [entry.id for entry in batch if entry.id not in records]
    legacy_vectors = (
        stores.source.get_stored_embeddings(missing, namespace=source)
        if missing and stores.source.supports_vectors()
        else {}
    )
    # One bulk read, write and delete per page (PRD-CORE-309 B71-13): per-row calls made a 20k-row import
    # take minutes (every per-row delete scanned the fts index).
    taken = stores.destination.existing_ids([entry.id for entry in batch], namespace=destination)
    if (
        taken and not skip_conflicts
    ):  # a resumed rename's count check cannot see a row swapped in under this id (sol r3)
        raise ConfigError(
            f"refusing to rename {source!r} onto {destination!r}: the destination already holds {min(taken)!r},"
            " which this rename did not move there. Use merge if folding them together is what you mean."
        )
    moving = [entry for entry in batch if entry.id not in taken]
    skipped = len(batch) - len(moving)
    stores.destination.store_many([entry.model_copy(update={"namespace": destination}) for entry in moving])
    for entry in moving:
        record = records.get(entry.id)
        embedding = list(record.embedding) if record is not None else legacy_vectors.get(entry.id)
        if embedding is not None:
            proof = record.provenance if record is not None else None
            if proof is not None and not proof.matches(proof.space, f"{entry.content} {entry.detail}", embedding):
                proof = None
            stores.destination.upsert_vector(
                entry.id,
                embedding,
                namespace=destination,
                **({"provenance": proof} if proof is not None else {}),
            )
    # Deleting the source rows drops their vectors too, which is why the destination vectors are written first.
    stores.source.delete_many([entry.id for entry in moving], namespace=source)
    return len(moving), skipped


def _end_of_pass(stores: NamespaceStores, source: str, destination: str, progress: MoveProgress) -> MoveProgress:
    """Complete the move once the source holds exactly the rows this pass chose to skip.

    The counts a verb returns are self-reported: they say what the loop did, not what the store now
    holds. This is the independent check, and it is the difference between a stranded-row bug that
    surfaces as an error and one that surfaces as ``status="merged"`` months before anybody notices.
    A pass that moved rows and still left extra ones behind scans again: between two batches another
    tenant can write a source row behind the cursor. A pass that moved nothing cannot drain it.
    """
    remaining = stores.source.count(namespace=source)
    if remaining == progress.skipped:
        stores.destination.add_graph_edges(destination, stores.source.graph_edges(source))  # edges no moved row carried
        return replace(progress, complete=True)
    if progress.pass_moved:
        return replace(progress, cursor=None, skipped=0, pass_moved=0)
    raise StorageError(
        f"incomplete move of {source!r} onto {destination!r}: the source still holds {remaining} rows "
        f"but only {progress.skipped} were deliberately skipped as conflicts. Raised instead of reporting a "
        f"successful merge; on a transactional backend the batch is rolled back."
    )


def _validate_pair(source: str, destination: str) -> tuple[str, str]:
    source = validate_namespace(source)
    destination = validate_namespace(destination)
    if source == destination:
        raise ConfigError(f"source and destination are the same namespace ({source!r}); nothing to do")
    return source, destination


def _move_all(stores: NamespaceStores, source: str, destination: str, *, merge: bool) -> NamespaceCurateResult:
    """Every batch of one move, inside one transaction (in-process callers hold no daemon lane)."""
    source, destination = _validate_pair(source, destination)
    progress = MoveProgress(merge=merge)
    with stores.source.transaction(), stores.destination_transaction():
        while not progress.complete:
            progress = move_batch(stores, source, destination, progress, limit=_BATCH_LIMIT)
    logger.info("namespace_moved", source=source, destination=destination, moved=progress.moved, merge=merge)
    return progress.result(source, destination)


def rename_namespace(stores: NamespaceStores, source: str, destination: str) -> NamespaceCurateResult:
    """Re-label every row of *source* onto *destination*, which must not already hold rows.

    An empty source is ``noop`` with zero moved -- checked BEFORE the destination, which is what makes
    a re-run of a completed rename a no-op rather than a refusal against the rows it moved there.

    Raises:
        ConfigError: If either namespace is invalid, the two are equal, or the destination already
            holds rows (that case is a merge, and the caller must say so).
    """
    return _move_all(stores, source, destination, merge=False)


def merge_namespace(stores: NamespaceStores, source: str, destination: str) -> NamespaceCurateResult:
    """Fold *source* into *destination*, keeping the destination on a conflict.

    Skipped rows stay in the source: a merge never deletes a row it did not copy. ``status="merged"``
    is reported only once the source is verified to hold EXACTLY the skipped rows.

    Raises:
        ConfigError: If either namespace is invalid or the two are equal.
        StorageError: If the source did not drain to the skipped rows (the move is rolled back on a
            transactional backend).
    """
    return _move_all(stores, source, destination, merge=True)


def store_census(config: MemoryConfig) -> dict[str, int]:
    """Return ``{namespace: row_count}`` across every store *config* reaches.

    The single source of truth for "what namespaces exist and how big are they",
    shared by the ``memory_namespace_diagnose`` tool and trw-mcp's session-start
    advisory. Both need the same answer, and a second implementation is how the
    two would come to disagree about whether a checkout looks moved.

    Under ``memory_single_store_path`` this is one file; otherwise it spans every
    discovered per-namespace store. Either way the namespaces come from the
    stores themselves, never from directory names, which are a lossy encoding.

    This is the ONLY census. A single-backend sibling (``namespace_census``) was
    exported alongside it until 0.16.0 and read by nothing but this module's own
    tests: it answered for one open store, so under the default split layout it
    silently under-counted every namespace living in another file. Two exported
    census functions with different blind spots is exactly how the diagnose tool
    and the session-start advisory would come to disagree, so the narrower one
    was removed rather than kept as a convenience.

    Over the daemon it holds only the token's granted namespaces (PRD-CORE-298
    FR02), so a moved-checkout sibling another checkout owns is not reported.
    """
    from trw_memory.integrations._backend import discover_namespace_backends
    from trw_memory.security.rbac import within_grant

    census: dict[str, int] = {}
    with discover_namespace_backends(config) as stores:
        for namespaces, backend in stores:
            for namespace in filter(within_grant, namespaces):
                census[namespace] = census.get(namespace, 0) + backend.count(namespace=namespace)
    return census


def detect_moved_checkout(namespace: str, row_counts: Mapping[str, int]) -> MovedCheckoutObservation | None:
    """Report the signal a moved or renamed checkout leaves behind.

    Takes a census rather than a backend because the namespaces being compared
    do not necessarily share a store: under the layout that ships today each
    namespace is its own SQLite file, so a caller assembles the census across
    every store (``discover_namespace_backends``) and this stays a pure
    function of it.

    The signal is deliberately narrow -- an EMPTY current project namespace
    plus at least one populated ``project:<same-slug>-*`` sibling -- because
    that is the exact shape of a path change, and a fresh clone of a
    differently named project produces none of it.

    Args:
        namespace: The identity the caller resolves to now.
        row_counts: Namespace to row count over every store in scope.

    Returns:
        The observation, or ``None`` when there is nothing to report. Never
        writes and never merges.
    """
    slug = _slug_of(namespace)
    if slug is None or row_counts.get(namespace, 0):
        return None
    candidates = [
        NamespaceRowCount(namespace=candidate, rows=rows)
        for candidate, rows in sorted(row_counts.items())
        if candidate != namespace and _slug_of(candidate) == slug and rows
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.rows)
    logger.info("moved_checkout_detected", current=namespace, candidates=[item.namespace for item in candidates])
    return MovedCheckoutObservation(
        current_namespace=namespace,
        current_rows=0,
        candidates=candidates,
        repair_command=f"trw-memory namespace rename {best.namespace} {namespace}",
    )
