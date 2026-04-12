"""NamespaceManager — high-level namespace operations backed by a storage backend."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from trw_memory.namespaces.validation import validate_namespace

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend


class NamespaceManager:
    """Manage namespace lifecycle (list, register, delete) against a storage backend.

    Args:
        backend: The SQLiteBackend to operate on.
    """

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend

    def register(self, ns: str) -> str:
        """Validate and register a namespace.

        This is idempotent — registering an existing namespace is a no-op.

        Args:
            ns: Namespace to register.

        Returns:
            The validated namespace string.

        Raises:
            ConfigError: If the namespace pattern is invalid.
        """
        return validate_namespace(ns)

    def list_namespaces(self) -> list[str]:
        """Return all distinct namespaces with stored entries.

        Returns:
            Sorted list of namespace strings.
        """
        return self._backend.list_namespaces()

    def delete(self, ns: str) -> int:
        """Delete all entries in a namespace.

        Args:
            ns: Namespace to clear.

        Returns:
            Number of entries deleted.

        Raises:
            ConfigError: If the namespace pattern is invalid.
        """
        validate_namespace(ns)
        return self._backend.delete_by_namespace(ns)

    def count(self, ns: str) -> int:
        """Return the number of entries in a namespace.

        Args:
            ns: Namespace to count.

        Returns:
            Entry count.

        Raises:
            ConfigError: If the namespace pattern is invalid.
        """
        validate_namespace(ns)
        return self._backend.count(namespace=ns)

    def ensure_team_namespace(self, ns: str, *, created_at: datetime | None = None) -> None:
        """Ensure a ``team:`` namespace has a lifecycle row in ``memory_namespaces``."""
        validate_namespace(ns)
        if not ns.startswith("team:"):
            return

        created = created_at or datetime.now(timezone.utc)
        team_id = ns.split(":", 1)[1]

        with self._backend._lock:
            self._backend._conn.execute(
                """
                INSERT INTO memory_namespaces (namespace_id, team_id, created_at, expires_at, status)
                VALUES (?, ?, ?, NULL, 'active')
                ON CONFLICT(namespace_id) DO NOTHING
                """,
                (ns, team_id, created.isoformat()),
            )
            self._backend._conn.commit()

    def mark_team_namespace_completed(
        self,
        ns: str,
        *,
        completed_at: datetime | None = None,
        ttl_hours: int = 24,
    ) -> str | None:
        """Set a team namespace expiry timestamp relative to completion time."""
        validate_namespace(ns)
        if not ns.startswith("team:"):
            return None

        completed = completed_at or datetime.now(timezone.utc)
        expires_at = completed + timedelta(hours=ttl_hours)
        self.ensure_team_namespace(ns, created_at=completed)

        with self._backend._lock:
            self._backend._conn.execute(
                """
                UPDATE memory_namespaces
                SET expires_at = ?, status = 'active'
                WHERE namespace_id = ?
                """,
                (expires_at.isoformat(), ns),
            )
            self._backend._conn.commit()

        return expires_at.isoformat()

    def team_namespace_expired(
        self,
        ns: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Return whether a team namespace has passed its expiry timestamp."""
        validate_namespace(ns)
        if not ns.startswith("team:"):
            return False

        with self._backend._lock:
            row = self._backend._conn.execute(
                "SELECT expires_at, status FROM memory_namespaces WHERE namespace_id = ?",
                (ns,),
            ).fetchone()

            if row is None:
                return False

            raw_expires_at = row[0]
            status = str(row[1] or "active")
            if raw_expires_at in (None, ""):
                return status == "expired"

            expires_at = datetime.fromisoformat(str(raw_expires_at))
            expired = expires_at <= (now or datetime.now(timezone.utc))
            if expired and status != "expired":
                self._backend._conn.execute(
                    "UPDATE memory_namespaces SET status = 'expired' WHERE namespace_id = ?",
                    (ns,),
                )
                self._backend._conn.commit()
            return expired
