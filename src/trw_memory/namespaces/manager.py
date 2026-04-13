"""NamespaceManager — high-level namespace operations backed by a storage backend."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.persistence import lock_for_rmw, read_yaml, write_yaml

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

_LIFECYCLE_METADATA_FILE = "namespace_lifecycle.yaml"


class NamespaceManager:
    """Manage namespace lifecycle (list, register, delete) against a storage backend."""

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    def _lifecycle_metadata_path(self) -> Path | None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend
        from trw_memory.storage.yaml_backend import YAMLBackend

        if isinstance(self._backend, SQLiteBackend):
            return self._backend._db_path.parent / _LIFECYCLE_METADATA_FILE
        if isinstance(self._backend, YAMLBackend):
            return self._backend._dir.parent / _LIFECYCLE_METADATA_FILE
        return None

    def _read_lifecycle_metadata(self, ns: str) -> dict[str, object] | None:
        metadata_path = self._lifecycle_metadata_path()
        if metadata_path is None or not metadata_path.exists():
            return None

        with lock_for_rmw(metadata_path) as locked_path:
            raw = read_yaml(locked_path)

        if not isinstance(raw, dict):
            raise ValueError(f"Lifecycle metadata must be a mapping for {ns!r}")
        if str(raw.get("namespace_id", ns)) != ns:
            raise ValueError(f"Lifecycle metadata namespace mismatch for {ns!r}")
        return raw

    def _write_lifecycle_metadata(self, ns: str, payload: dict[str, object]) -> None:
        metadata_path = self._lifecycle_metadata_path()
        if metadata_path is None:
            return
        with lock_for_rmw(metadata_path) as locked_path:
            write_yaml(locked_path, payload)

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
        existing_metadata = self._read_lifecycle_metadata(ns)
        if existing_metadata is None:
            self._write_lifecycle_metadata(
                ns,
                {
                    "namespace_id": ns,
                    "team_id": team_id,
                    "created_at": created.isoformat(),
                    "expires_at": None,
                    "status": "active",
                },
            )

        from trw_memory.storage.sqlite_backend import SQLiteBackend

        if not isinstance(self._backend, SQLiteBackend):
            return

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
        team_id = ns.split(":", 1)[1]
        self.ensure_team_namespace(ns, created_at=completed)

        existing_metadata = self._read_lifecycle_metadata(ns) or {}
        self._write_lifecycle_metadata(
            ns,
            {
                "namespace_id": ns,
                "team_id": str(existing_metadata.get("team_id", team_id)),
                "created_at": str(existing_metadata.get("created_at", completed.isoformat())),
                "expires_at": expires_at.isoformat(),
                "status": "completed",
            },
        )

        from trw_memory.storage.sqlite_backend import SQLiteBackend

        if not isinstance(self._backend, SQLiteBackend):
            return expires_at.isoformat()

        with self._backend._lock:
            self._backend._conn.execute(
                """
                UPDATE memory_namespaces
                SET expires_at = ?, status = 'completed'
                WHERE namespace_id = ?
                """,
                (expires_at.isoformat(), ns),
            )
            self._backend._conn.commit()

        return expires_at.isoformat()

    def team_namespace_completed(self, ns: str) -> bool:
        """Return whether a team namespace has already entered its completion window."""
        validate_namespace(ns)
        if not ns.startswith("team:"):
            return False

        from trw_memory.storage.sqlite_backend import SQLiteBackend

        if isinstance(self._backend, SQLiteBackend):
            with self._backend._lock:
                row = self._backend._conn.execute(
                    "SELECT expires_at, status FROM memory_namespaces WHERE namespace_id = ?",
                    (ns,),
                ).fetchone()
                if row is not None:
                    raw_expires_at = row[0]
                    status = str(row[1] or "active")
                    return raw_expires_at not in (None, "") or status in {"completed", "expired"}

        metadata = self._read_lifecycle_metadata(ns)
        if metadata is None:
            return False
        raw_expires_at = metadata.get("expires_at")
        status = str(metadata.get("status") or "active")
        return raw_expires_at not in (None, "") or status in {"completed", "expired"}

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

        from trw_memory.storage.sqlite_backend import SQLiteBackend

        if isinstance(self._backend, SQLiteBackend):
            with self._backend._lock:
                row = self._backend._conn.execute(
                    "SELECT expires_at, status FROM memory_namespaces WHERE namespace_id = ?",
                    (ns,),
                ).fetchone()

                if row is not None:
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

        metadata = self._read_lifecycle_metadata(ns)
        if metadata is None:
            return False

        raw_expires_at = metadata.get("expires_at")
        status = str(metadata.get("status") or "active")
        if raw_expires_at in (None, ""):
            return status == "expired"

        expires_at = datetime.fromisoformat(str(raw_expires_at))
        expired = expires_at <= (now or datetime.now(timezone.utc))
        if expired and status != "expired":
            updated_metadata = dict(metadata)
            updated_metadata["status"] = "expired"
            self._write_lifecycle_metadata(ns, updated_metadata)
            return expired
        return expired
