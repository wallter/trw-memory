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
from dataclasses import dataclass
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from trw_memory.exceptions import ConfigError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.interface import EntryCursor, StorageBackend

__all__ = [
    "MovedCheckoutObservation",
    "NamespaceCurateResult",
    "NamespaceRowCount",
    "NamespaceStores",
    "detect_moved_checkout",
    "merge_namespace",
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
    status: Literal["renamed", "merged", "noop"] = Field(description="Outcome class")


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


def _move_rows(
    stores: NamespaceStores,
    source: str,
    destination: str,
    *,
    skip_conflicts: bool,
) -> tuple[int, int]:
    """Re-label rows from *source* to *destination*; return (moved, skipped).

    Paging is by KEYSET cursor, not by a repeated ``limit``-window read. A
    skipped conflict stays in the source, so a window read re-serves the same
    rows every pass: the loop then either spins forever or -- as it did before
    -- de-duplicates in memory, finds the whole window already seen, sees an
    empty batch and BREAKS, silently stranding every non-conflicting row ranked
    below that window while still reporting a clean merge. A cursor is a
    position rather than a count, so the window advances past the skipped rows
    on its own and no in-memory ``seen`` set is needed.

    Raises:
        StorageError: If the source did not drain to exactly the rows this pass
            deliberately left behind, or if a page failed to advance the
            cursor. Both are raised INSIDE the transaction, so a transactional
            backend (SQLite) rolls the move back whole. A backend whose
            ``transaction()`` is the ABC's no-op default (YAML) keeps what it
            already wrote; there the error is the operator's signal to
            reconcile, which is still strictly better than a partial re-key
            reported as a success.
    """
    moved = 0
    skipped = 0
    cursor: EntryCursor | None = None
    with stores.source.transaction(), stores.destination_transaction():
        while True:
            batch = stores.source.list_entries(namespace=source, limit=_BATCH_LIMIT, after=cursor)
            if not batch:
                break
            next_cursor = EntryCursor.from_entry(batch[-1])
            if next_cursor == cursor:
                raise StorageError(
                    f"refusing to loop moving {source!r} onto {destination!r}: the keyset cursor "
                    f"at {next_cursor.entry_id!r} did not advance"
                )
            cursor = next_cursor
            embeddings = (
                stores.source.get_stored_embeddings([entry.id for entry in batch])
                if stores.source.supports_vectors()
                else {}
            )
            for entry in batch:
                if skip_conflicts and stores.destination.get(entry.id, namespace=destination) is not None:
                    skipped += 1
                    continue
                stores.destination.store(entry.model_copy(update={"namespace": destination}))
                embedding = embeddings.get(entry.id)
                if embedding is not None:
                    stores.destination.upsert_vector(entry.id, embedding, namespace=destination)
                # Deleting the source row drops its vector too, which is why the
                # destination vector is written first.
                stores.source.delete(entry.id, namespace=source)
                moved += 1
        _assert_source_drained(stores, source, destination, skipped=skipped)
    return moved, skipped


def _assert_source_drained(
    stores: NamespaceStores,
    source: str,
    destination: str,
    *,
    skipped: int,
) -> None:
    """Fail loudly unless the source holds exactly the rows we chose to skip.

    The counts a verb returns are self-reported: they say what the loop did,
    not what the store now holds. This is the independent check, and it is the
    difference between a stranded-row bug that surfaces as an error and one
    that surfaces as ``status="merged"`` months before anybody notices the rows
    are gone from both namespaces' point of view.
    """
    remaining = stores.source.count(namespace=source)
    if remaining == skipped:
        return
    raise StorageError(
        f"incomplete move of {source!r} onto {destination!r}: the source still holds {remaining} rows "
        f"but only {skipped} were deliberately skipped as conflicts. Raised instead of reporting a "
        f"successful merge; on a transactional backend the move is rolled back."
    )


def _validate_pair(source: str, destination: str) -> tuple[str, str]:
    source = validate_namespace(source)
    destination = validate_namespace(destination)
    if source == destination:
        raise ConfigError(f"source and destination are the same namespace ({source!r}); nothing to do")
    return source, destination


def rename_namespace(stores: NamespaceStores, source: str, destination: str) -> NamespaceCurateResult:
    """Re-label every row of *source* onto *destination*.

    Args:
        stores: Source and destination backends.
        source: Namespace to empty.
        destination: Namespace to fill. Must not already hold rows.

    Returns:
        Counts and outcome. An empty source is ``noop`` with zero moved --
        checked BEFORE the destination, which is what makes a re-run of a
        completed rename a no-op rather than a "destination already populated"
        refusal against the rows it just moved there itself.

    Raises:
        ConfigError: If either namespace is invalid, the two are equal, or the
            destination already holds rows (that case is a merge, and the
            caller must say so).
    """
    source, destination = _validate_pair(source, destination)
    source_rows = stores.source.count(namespace=source)
    if not source_rows:
        return NamespaceCurateResult(
            source=source, destination=destination, source_rows=0, moved=0, skipped=0, status="noop"
        )
    destination_rows = stores.destination.count(namespace=destination)
    if destination_rows:
        raise ConfigError(
            f"refusing to rename {source!r} onto {destination!r}: the destination already holds "
            f"{destination_rows} rows. Use merge if folding them together is what you mean."
        )
    moved, skipped = _move_rows(stores, source, destination, skip_conflicts=False)
    logger.info("namespace_renamed", source=source, destination=destination, moved=moved)
    return NamespaceCurateResult(
        source=source, destination=destination, source_rows=source_rows, moved=moved, skipped=skipped, status="renamed"
    )


def merge_namespace(stores: NamespaceStores, source: str, destination: str) -> NamespaceCurateResult:
    """Fold *source* into *destination*, keeping the destination on a conflict.

    Args:
        stores: Source and destination backends.
        source: Namespace to fold in.
        destination: Namespace that wins every id collision.

    Returns:
        Counts and outcome, including how many rows were skipped because the
        destination already held that id. Skipped rows stay in the source: a
        merge never deletes a row it did not copy. ``status="merged"`` is
        reported only once the source is verified to hold EXACTLY the skipped
        rows -- a partial merge raises instead of reporting success.

    Raises:
        ConfigError: If either namespace is invalid or the two are equal.
        StorageError: If the source did not drain to the skipped rows (the
            move is rolled back on a transactional backend).
    """
    source, destination = _validate_pair(source, destination)
    source_rows = stores.source.count(namespace=source)
    if not source_rows:
        return NamespaceCurateResult(
            source=source, destination=destination, source_rows=0, moved=0, skipped=0, status="noop"
        )
    moved, skipped = _move_rows(stores, source, destination, skip_conflicts=True)
    logger.info("namespace_merged", source=source, destination=destination, moved=moved, skipped=skipped)
    return NamespaceCurateResult(
        source=source, destination=destination, source_rows=source_rows, moved=moved, skipped=skipped, status="merged"
    )


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
    """
    from trw_memory.integrations._backend import discover_namespace_backends

    census: dict[str, int] = {}
    with discover_namespace_backends(config) as stores:
        for namespaces, backend in stores:
            for namespace in namespaces:
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
    prefix = f"{_PROJECT_SCOPE}{slug}-"
    candidates = [
        NamespaceRowCount(namespace=candidate, rows=rows)
        for candidate, rows in sorted(row_counts.items())
        if candidate != namespace and candidate.startswith(prefix) and rows
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
