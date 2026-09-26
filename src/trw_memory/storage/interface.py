"""StorageBackend ABC — defines the contract for all storage implementations.

Any class that subclasses :class:`StorageBackend` can be used as a drop-in
replacement.  Two implementations are provided:

- :class:`~trw_memory.storage.sqlite_backend.SQLiteBackend` — primary, fast
- :class:`~trw_memory.storage.yaml_backend.YAMLBackend` — portable fallback
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple

from trw_memory.models.memory import MemoryEntry, MemoryStatus

if TYPE_CHECKING:
    from trw_memory.embeddings.provenance import EmbeddingSpace, StoredVector, VectorProvenance
    from trw_memory.retrieval.temporal_selection import TemporalSelection


class GraphEdge(NamedTuple):
    """One knowledge-graph edge, apart from the namespace it is filed under."""

    source_id: str
    target_id: str
    edge_type: str
    weight: float
    created_at: str
    metadata: str = "{}"


@dataclass(frozen=True)
class EntryCursor:
    """A keyset position in the ``(updated_at DESC, id DESC)`` listing order.

    ``list_entries`` pages with ``after=`` rather than an OFFSET because its
    callers MUTATE the rows they page over. A namespace merge deletes what it
    moved and leaves what it skipped, so an offset window either re-reads the
    skipped rows forever or -- if the caller de-duplicates in memory -- stops
    advancing the moment a whole window is skipped, silently stranding every
    row ranked below it. A cursor is a position, not a count, so the window
    advances regardless of what the caller did to the rows it just saw.

    ``updated_at`` is the STORED text encoding (``datetime.isoformat()``, the
    same encoder every write path uses), not a ``datetime``: the ORDER BY
    compares that column as TEXT, so the cursor has to compare in the same
    space. Re-encoding a parsed ``datetime`` at query time would let a format
    difference put the cursor on the wrong side of its own boundary row.
    """

    updated_at: str
    entry_id: str

    @classmethod
    def from_entry(cls, entry: MemoryEntry) -> EntryCursor:
        """Return the cursor that resumes listing immediately AFTER *entry*."""
        return cls(updated_at=entry.updated_at.isoformat(), entry_id=entry.id)


@dataclass(frozen=True)
class NamespaceChangeToken:
    """A cheap position in one namespace's write history.

    Equal tokens mean nothing a :meth:`StorageBackend.entries_changed_since`
    feed would report has happened in between. ``store`` identifies the
    database, so instances opened on one file compare equal.
    """

    store: str
    insert_seq: int
    top_updated_at: str
    delete_epoch: int


class StorageBackend(ABC):
    """Abstract base class for memory storage backends.

    All mutating operations are synchronous.  Backends are responsible for
    their own thread-safety; callers must not share backend instances across
    threads without external synchronisation.
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> None:
        """Persist a new entry (or replace an existing one with the same id).

        Args:
            entry: The memory entry to persist.

        Raises:
            StorageError: If the write fails.
        """
        ...

    @abstractmethod
    def get(self, entry_id: str, *, namespace: str) -> MemoryEntry | None:
        """Retrieve the entry identified by ``(namespace, entry_id)``.

        PRD-CORE-245 FR03: a memory row's identity is composite, so *namespace*
        is required and has no default. A default would silently reinstate the
        ambiguity the composite key exists to remove: one database file holds
        many namespaces, and an id is only unique within one of them.

        Args:
            entry_id: The entry's ``id`` field.
            namespace: The namespace that owns the entry.

        Returns:
            The :class:`MemoryEntry`, or ``None`` if not found.

        Raises:
            StorageError: If the read fails.
        """
        ...

    @abstractmethod
    def update(self, entry_id: str, *, namespace: str, **fields: object) -> MemoryEntry | None:
        """Apply a partial update to the ``(namespace, entry_id)``-identified entry.

        PRD-CORE-245 FR03: required, not defaulted — see :meth:`get`.

        Args:
            entry_id: Target entry identifier.
            namespace: The namespace that owns the entry.
            **fields: Field names and new values.  Only supplied fields are
                changed; all others retain their current values.

        Returns:
            The updated :class:`MemoryEntry`, or ``None`` if not found.

        Raises:
            StorageError: If the update fails.
        """
        ...

    @abstractmethod
    def delete(self, entry_id: str, *, namespace: str) -> bool:
        """Remove the ``(namespace, entry_id)``-identified entry from storage.

        PRD-CORE-245 FR03: required, not defaulted — see :meth:`get`.

        Args:
            entry_id: The entry to remove.
            namespace: The namespace that owns the entry.

        Returns:
            ``True`` if the entry existed and was deleted, ``False`` otherwise.

        Raises:
            StorageError: If the deletion fails.
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 25,
        tags: list[str] | None = None,
        status: MemoryStatus | None = None,
        min_importance: float = 0.0,
        namespace: str | None = None,
        temporal_selection: TemporalSelection | None = None,
        entry_filter: Callable[[MemoryEntry], bool] | None = None,
    ) -> list[MemoryEntry]:
        """Keyword search over content and detail fields.

        Args:
            query: Free-text search string.
            top_k: Maximum number of results to return.
            tags: If provided, entries must contain ALL of these tags.
            status: If provided, filter to entries with this status.
            min_importance: Lower bound on ``importance`` (inclusive).
            namespace: If provided, restrict to this namespace.
            entry_filter: Pure full-entry predicate applied before result caps; may
                be replayed after decoding recovery. Exceptions propagate unchanged.
                None preserves unfiltered behavior and does not imply temporal policy.
            temporal_selection: Optional eligibility-before-limit policy; None
                preserves raw maintenance visibility.

        Returns:
            Up to *top_k* matching entries, ordered by relevance (descending).

        Raises:
            StorageError: If the query fails.
        """
        ...

    @abstractmethod
    def count(self, namespace: str | None = None) -> int:
        """Return the total number of stored entries.

        Args:
            namespace: If provided, count only entries in this namespace.

        Returns:
            Number of entries.

        Raises:
            StorageError: If the count query fails.
        """
        ...

    @abstractmethod
    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
        exclude_superseded: bool = False,
        tags: list[str] | None = None,
        after: EntryCursor | None = None,
        temporal_selection: TemporalSelection | None = None,
        entry_filter: Callable[[MemoryEntry], bool] | None = None,
    ) -> list[MemoryEntry]:
        """Return entries with optional filters.

        Args:
            status: If provided, only return entries with this status.
            namespace: If provided, only return entries in this namespace.
            min_importance: If > 0.0, only return entries whose importance is
                >= this value. Pushes the importance threshold into the storage
                layer so callers that only want high-importance rows do not
                hydrate the full namespace into memory first. Default 0.0 keeps
                the legacy behaviour (no importance filter).
            entry_filter: Pure full-entry predicate applied before result caps; may
                be replayed after decoding recovery. Exceptions propagate unchanged.
                None preserves unfiltered behavior and does not imply temporal policy.
            temporal_selection: Optional eligibility-before-limit policy; cannot
                be combined with a raw-order after cursor.
            limit: Maximum number of entries to return.
            exclude_superseded: When True, exclude entries that have a non-null
                ``invalid_from`` value (bi-temporal superseded entries).  Only
                effective for the common case where ``as_of`` is not set; callers
                doing point-in-time queries should leave this False and rely on
                ``apply_validity_prior`` post-fusion instead.
            tags: If provided, only return entries containing ALL of these tags.
                The predicate is applied BEFORE the limit so tagged entries past
                the row limit are not truncated away (the recall silent-drop
                bug). When omitted the legacy behaviour (no tag filter) holds.
            after: Keyset position from a previous page — only entries ranked
                strictly BELOW it are returned. This is the only safe way to
                page over rows the caller is mutating; see :class:`EntryCursor`.

        Returns:
            Up to *limit* entries ordered by ``updated_at`` descending, ``id``
            descending. The ``id`` tiebreak makes the order TOTAL, which is
            what lets consecutive ``after=`` pages be disjoint and complete
            even when many rows share an ``updated_at``.

        Raises:
            StorageError: If the query fails.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any held resources (file handles, DB connections, etc.).

        Safe to call multiple times.
        """
        ...

    # -- Non-abstract extension points (safe defaults) ----------------------

    def list_namespaces(self, required_namespaces: list[str] | None = None) -> list[str]:
        """Return distinct namespaces that have stored entries.

        Subclasses that support multi-namespace storage should override this
        to query their underlying store.  The default returns an empty list,
        which is safe for single-namespace or in-memory backends.

        Args:
            required_namespaces: When provided, scope the result to this
                authorized set so enumeration never leaks the existence of
                other tenants' namespaces (trw-memory-11). ``None`` returns
                every namespace (admin/single-tenant behaviour).

        Returns:
            Sorted list of unique namespace strings.  Empty if the backend
            does not track namespaces or has no entries.
        """
        return []

    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all entries belonging to *namespace*.

        Subclasses that support bulk-delete should override this for
        efficiency.  The default returns ``0`` (no entries deleted) and
        performs no I/O.

        Args:
            namespace: The namespace whose entries should be removed.

        Returns:
            Number of entries actually deleted.  ``0`` if the namespace
            does not exist or the backend does not support this operation.
        """
        return 0

    def increment_session_counts(
        self, entry_ids: list[str], *, namespace: str, updated_at: datetime | None = None
    ) -> int:
        """Increment ``session_count`` of *namespace*'s rows among *entry_ids*.

        Backends that support bulk mutation should override this to perform the
        work in a single transaction. The default is a safe no-op.

        Args:
            entry_ids: Distinct entry ids to increment.
            namespace: The namespace whose rows are counted (PRD-CORE-245 FR03:
                a bare id does not identify a row).
            updated_at: Optional timestamp to stamp onto updated rows.

        Returns:
            Number of rows updated.
        """
        return 0

    def increment_recall_access(
        self, entry_ids: list[str], *, namespace: str, accessed_at: datetime | None = None
    ) -> int:
        """Increment ``access_count`` and ``recall_count`` of *namespace*'s recalled rows.

        F-008: backends that support bulk mutation should override this to do
        the work in a single statement / commit. The default falls back to a
        per-entry get+update loop so non-SQLite backends keep correct
        semantics (each distinct id incremented once).

        Args:
            entry_ids: Entry ids that were surfaced by recall (may contain dups).
            namespace: The namespace whose rows are counted; a twin elsewhere is left alone.
            accessed_at: Timestamp to stamp onto ``last_accessed_at``.

        Returns:
            Number of entries updated.
        """
        updated = 0
        for entry_id in dict.fromkeys(entry_ids):
            entry = self.get(entry_id, namespace=namespace)
            if entry is None:
                continue
            self.update(
                entry_id,
                namespace=namespace,
                access_count=entry.access_count + 1,
                recall_count=entry.recall_count + 1,
                last_accessed_at=accessed_at,
            )
            updated += 1
        return updated

    @contextlib.contextmanager
    def transaction(self) -> Iterator[StorageBackend]:
        """Optional batching context — backends that support transactions
        should override.

        PRD-FIX-088 FR02: callers wrap a series of writes in
        ``with backend.transaction(): ...`` to collapse N implicit
        per-call commits into one explicit commit.  The default
        implementation is a no-op pass-through so callers don't need
        ``hasattr`` guards; non-supporting backends still see N implicit
        commits, which is correct (just slower).
        """
        yield self

    def checkpoint_wal(self, mode: str = "PASSIVE") -> Mapping[str, object]:
        """Checkpoint a write-ahead log, if the backend keeps one.

        This is an optional maintenance seam mirroring the :meth:`supports_vectors`
        capability pattern: backends that maintain a WAL (e.g.
        :class:`~trw_memory.storage.sqlite_backend.SQLiteBackend`) override this to
        run ``PRAGMA wal_checkpoint`` and report frame counts. Backends without a
        WAL (e.g. :class:`~trw_memory.storage.yaml_backend.YAMLBackend`) inherit
        this safe no-op so callers can invoke maintenance uniformly across the
        backend seam without a capability guard.

        Args:
            mode: Requested checkpoint mode (``PASSIVE``/``FULL``/``RESTART``/
                ``TRUNCATE``). Ignored by the no-op default.

        Returns:
            An empty mapping for the no-op default; overriding backends return a
            structured result describing the checkpoint outcome (e.g.
            :class:`~trw_memory.storage._wal_checkpoint.CheckpointResult`).
        """
        return {}

    def namespace_change_token(self, namespace: str) -> NamespaceChangeToken | None:
        """Return a token that changes whenever *namespace* is written, or ``None``.

        Optional capability for callers that keep a view derived from a
        namespace's recent rows. ``None`` (this default) means "no cheap token":
        such callers re-read what they need on every call.
        """
        return None

    def entries_changed_since(
        self, namespace: str, token: NamespaceChangeToken, *, limit: int
    ) -> list[MemoryEntry] | None:
        """Return the rows of *namespace* (any status) written since *token*, newest first.

        ``None`` when the backend has no change feed, or when more than *limit*
        rows changed: the caller must then re-read rather than trust a partial
        feed. Only meaningful for a token this backend's
        :meth:`namespace_change_token` returned.
        """
        return None

    def graph_edges(self, namespace: str) -> list[GraphEdge]:
        """Every knowledge-graph edge filed under *namespace*; a backend without a graph holds none."""
        return []

    def graph_edge_count(self, namespace: str) -> int:
        """How many edges :meth:`graph_edges` would return, without reading them."""
        return len(self.graph_edges(namespace))

    def ids_by_source(self, namespace: str, source_identity: str, limit: int) -> list[str]:
        """Up to *limit* ids of *namespace*'s rows written by *source_identity*, any status."""
        found = self.list_entries(
            namespace=namespace, limit=limit, entry_filter=lambda e: e.source_identity == source_identity
        )
        return [entry.id for entry in found]

    def add_graph_edges(self, namespace: str, edges: Sequence[GraphEdge]) -> None:
        """File *edges* under *namespace*, keeping any already there.

        Raises:
            StorageError: If this backend keeps no graph and *edges* is not empty -- they would be lost.
        """
        if edges:
            from trw_memory.exceptions import StorageError

            raise StorageError(f"{type(self).__name__} keeps no knowledge graph; {len(edges)} edges would be lost")

    def find_active_by_content(self, content: str, detail: str, *, namespace: str = "default") -> str | None:
        """Id of an ACTIVE entry of *namespace* whose content and detail match exactly, or ``None``.

        A scan over the namespace's active rows; SQLite answers it with one indexed query.
        """
        match = self.list_entries(
            status=MemoryStatus.ACTIVE,
            namespace=namespace,
            limit=1_000_000,
            entry_filter=lambda entry: (entry.content, entry.detail) == (content, detail),
        )
        return match[0].id if match else None

    def supports_vectors(self) -> bool:
        """Return whether this backend can persist and search dense vectors.

        This is the explicit capability signal for the vector seam. The vector
        extension methods (:meth:`upsert_vector`, :meth:`search_vectors`,
        :meth:`get_stored_embeddings`, ...) silently no-op on backends without
        vector support, so a caller that only computes an embedding in order to
        persist it can consult this method first and skip the (expensive)
        embedding-model call entirely when it would be wasted.

        The default is ``False`` — backends gain vector support by overriding
        this to report their real runtime state (e.g. whether ``sqlite-vec``
        loaded and the virtual table exists). Returning ``False`` here is always
        safe: it only ever suppresses work the no-op methods would discard.

        Returns:
            ``True`` when ``upsert_vector`` / ``search_vectors`` actually
            persist and query vectors; ``False`` when they no-op.
        """
        return False

    def upsert_vector(  # noqa: B027 -- optional vector capability default
        self, entry_id: str, embedding: list[float], *, namespace: str, provenance: VectorProvenance | None = None
    ) -> None:
        """Insert or update the dense vector for ``(namespace, entry_id)``.

        Provenance, when supplied, must match vector bytes and be committed
        atomically with them. An unqualified replacement clears previous proof.

        Backends that support vector search (e.g. via ``sqlite-vec``) should
        override this.  The default is a silent no-op so that callers do not
        need to guard against missing vector support.

        Args:
            entry_id: The memory entry id to associate the vector with.
            embedding: Dense float vector.  Length must match the backend's
                configured dimensionality.
            namespace: The namespace that owns the entry. ``vec_index`` is keyed
                ``UNIQUE (namespace, entry_id)`` under schema 5 (PRD-CORE-245
                FR02), so a bare id no longer identifies one vector.

        Raises:
            StorageError: If the upsert fails (only in overriding backends).
        """

    def delete_vector(self, entry_id: str, *, namespace: str) -> bool:
        """Delete the dense vector for ``(namespace, entry_id)``.

        PRD-CORE-245 FR03: required, not defaulted — see :meth:`get`.
        Backends without vector support return ``False``.
        """
        return False

    def vector_exists(self, entry_id: str, *, namespace: str) -> bool:
        """Return whether a dense vector currently exists for ``(namespace, entry_id)``."""
        return False

    def existing_vector_ids(self, namespace: str | None = None) -> set[str]:
        """Return the set of entry IDs that currently have a stored vector.

        Default returns an empty set so callers can opt into batch backfill
        skipping without branching on backend capabilities.

        Args:
            namespace: When provided, scope the lookup to that namespace so a
                shared multi-namespace store does not return every tenant's
                vector ids. ``None`` keeps the legacy full lookup.
        """
        return set()

    def search_vectors(
        self,
        query_embedding: list[float],
        top_k: int = 25,
        namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        """KNN search over stored dense vectors.

        Backends that support vector search should override this.  The default
        returns an empty list so that callers can always call this method
        without checking for vector support.

        Args:
            query_embedding: Query vector.  Length must match the backend's
                configured dimensionality.
            top_k: Maximum number of nearest neighbours to return.
            namespace: When provided, scope results to that namespace so a
                shared multi-namespace store does not surface another tenant's
                vectors. ``None`` keeps the legacy unscoped search.

        Returns:
            List of ``(entry_id, distance)`` tuples sorted by distance
            ascending (closest first).  Empty if the backend has no vector
            support or no vectors are stored.

        Raises:
            StorageError: If the search fails (only in overriding backends).
        """
        return []

    def get_stored_embeddings(self, entry_ids: list[str], *, namespace: str | None = None) -> dict[str, list[float]]:
        """Return stored dense vectors for the requested entry IDs.

        ``namespace=None`` retains the legacy unscoped lookup. An explicit
        namespace, including ``""``, restricts storage selection to that exact
        value before decoding or building the ID-keyed result mapping.

        Backends with vector persistence should override this. The default
        returns an empty mapping so callers can opt into dense retrieval
        without branching on backend capabilities.
        """
        return {}

    def get_vector_records(self, entry_ids: list[str], *, namespace: str) -> dict[str, StoredVector]:
        """Return scoped vectors and validated provenance; missing proof is None.

        Backends without this capability return no records rather than upgrading
        unqualified legacy vectors into trusted evidence.
        """
        return {}

    def vector_space_census(self, *, namespace: str) -> dict[EmbeddingSpace | None, int] | None:
        """Count *namespace*'s stored vectors by claimed embedding space, reading no vector blobs.

        ``None`` key: NULL or malformed provenance. ``None`` result: this backend
        cannot take a census -- callers must treat that as unknown, never as
        "every vector is in the loaded space".
        """
        return None

    def recent_vector_records(self, *, namespace: str, limit: int) -> dict[str, StoredVector]:
        """Vectors of the *limit* most recently updated ACTIVE entries of *namespace*.

        The candidate set is exactly ``list_entries(status=ACTIVE,
        namespace=namespace, limit=limit)``; entries without a vector are absent.
        Backends that can select those ids without decoding every row override
        this (graph enrichment reads it for every namespace in the store).
        """
        entries = self.list_entries(status=MemoryStatus.ACTIVE, namespace=namespace, limit=limit)
        return self.get_vector_records([entry.id for entry in entries], namespace=namespace)

    def hype_sibling_ids(self, parent_id: str, *, namespace: str) -> list[str]:
        """Enumerate legacy derived vectors; check supports_vectors() first."""
        raise NotImplementedError("legacy vector cleanup unavailable for this backend")

    def delete_hype_siblings(self, parent_id: str, *, namespace: str) -> int:
        """Delete namespace-owned derived vectors; unavailable is not a purge."""
        raise NotImplementedError("legacy vector cleanup unavailable for this backend")

    def store_many(self, entries: list[MemoryEntry]) -> int:
        """Bulk-insert entries in a single transaction.

        Backends that support efficient batch writes should override this.
        The default falls back to per-entry ``store()`` calls (correct, but
        slower for large batches).

        Args:
            entries: Entries to persist (INSERT OR REPLACE semantics).

        Returns:
            Number of entries written (== ``len(entries)``).
        """
        for entry in entries:
            self.store(entry)
        return len(entries)

    def search_fts(
        self,
        query: str,
        *,
        top_k: int = 25,
        status: MemoryStatus | None = None,
        min_importance: float = 0.0,
        namespace: str | None = None,
        tags: list[str] | None = None,
        temporal_selection: TemporalSelection | None = None,
        entry_filter: Callable[[MemoryEntry], bool] | None = None,
    ) -> list[MemoryEntry]:
        """Full-text search using a backend-native FTS index (e.g. FTS5).

        Backends that provide an FTS index should override this.  The default
        returns an empty list so callers can always call this method without
        checking for FTS support.

        Args:
            query: Free-text search string.
            top_k: Maximum number of results to return.
            status: If provided, filter to entries with this status.
            min_importance: Lower bound on importance (inclusive).
            namespace: If provided, restrict to this namespace.
            tags: If provided, every returned entry carries ALL of them, applied
                before *top_k*, exactly as ``list_entries`` applies it.

        Returns:
            Up to *top_k* matching entries. Empty if the backend has no FTS
            support or no matches are found.
        """
        return []

    # -- Context manager (non-abstract) ------------------------------------

    def __enter__(self) -> StorageBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
