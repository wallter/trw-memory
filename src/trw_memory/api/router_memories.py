"""Memory CRUD endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from trw_memory.api.deps import get_backend
from trw_memory.exceptions import ConfigError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.namespace import validate_namespace
from trw_memory.storage.sqlite_backend import SQLiteBackend

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateMemoryRequest(BaseModel):
    """Request body for creating a new memory entry."""

    content: str
    detail: str = ""
    tags: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    importance: float = 0.5
    namespace: str = "default"
    source: str = "api"
    metadata: dict[str, str] = Field(default_factory=dict)


class UpdateMemoryRequest(BaseModel):
    """Request body for partially updating a memory entry."""

    content: str | None = None
    detail: str | None = None
    tags: list[str] | None = None
    importance: float | None = None
    status: str | None = None
    metadata: dict[str, str] | None = None


class SearchRequest(BaseModel):
    """Request body for searching memories."""

    query: str = ""
    tags: list[str] | None = None
    status: str | None = None
    min_importance: float = 0.0
    namespace: str | None = None
    limit: int = 25


class MemoryResponse(BaseModel):
    """Serialised memory entry returned by the API."""

    id: str
    content: str
    detail: str
    tags: list[str]
    evidence: list[str]
    importance: float
    status: str
    recurrence: int
    namespace: str
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None
    access_count: int
    q_value: float
    q_observations: int
    source: str
    source_identity: str
    merged_from: list[str]
    consolidated_from: list[str]
    consolidated_into: str | None
    metadata: dict[str, str]


def _entry_to_response(entry: MemoryEntry) -> MemoryResponse:
    """Convert a MemoryEntry to a MemoryResponse."""
    return MemoryResponse.model_validate(entry.model_dump())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=201, response_model=MemoryResponse)
def create_memory(
    body: CreateMemoryRequest,
    backend: SQLiteBackend = Depends(get_backend),
) -> MemoryResponse:
    """Create a new memory entry."""
    try:
        validate_namespace(body.namespace)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    now = datetime.now(timezone.utc)
    entry_id = f"M-{uuid.uuid4().hex[:12]}"
    entry = MemoryEntry(
        id=entry_id,
        content=body.content,
        detail=body.detail,
        tags=body.tags,
        evidence=body.evidence,
        importance=body.importance,
        namespace=body.namespace,
        source=body.source,
        metadata=body.metadata,
        created_at=now,
        updated_at=now,
    )
    backend.store(entry)
    return _entry_to_response(entry)


@router.get("/{memory_id}", response_model=MemoryResponse)
def get_memory(
    memory_id: str,
    backend: SQLiteBackend = Depends(get_backend),
) -> MemoryResponse:
    """Retrieve a single memory entry by ID."""
    entry = backend.get(memory_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _entry_to_response(entry)


@router.patch("/{memory_id}", response_model=MemoryResponse)
def update_memory(
    memory_id: str,
    body: UpdateMemoryRequest,
    backend: SQLiteBackend = Depends(get_backend),
) -> MemoryResponse:
    """Partially update an existing memory entry."""
    fields: dict[str, object] = {}
    if body.content is not None:
        fields["content"] = body.content
    if body.detail is not None:
        fields["detail"] = body.detail
    if body.tags is not None:
        fields["tags"] = body.tags
    if body.importance is not None:
        fields["importance"] = body.importance
    if body.status is not None:
        fields["status"] = body.status
    if body.metadata is not None:
        fields["metadata"] = body.metadata

    updated = backend.update(memory_id, **fields)
    if updated is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _entry_to_response(updated)


@router.delete("/{memory_id}", status_code=204)
def delete_memory(
    memory_id: str,
    backend: SQLiteBackend = Depends(get_backend),
) -> None:
    """Delete a memory entry."""
    deleted = backend.delete(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")


@router.post("/search", response_model=list[MemoryResponse])
def search_memories(
    body: SearchRequest,
    backend: SQLiteBackend = Depends(get_backend),
) -> list[MemoryResponse]:
    """Search memory entries with filters."""
    status_filter: MemoryStatus | None = None
    if body.status is not None:
        try:
            status_filter = MemoryStatus(body.status)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid status: {body.status}"
            ) from exc

    results = backend.search(
        query=body.query,
        top_k=body.limit,
        tags=body.tags,
        status=status_filter,
        min_importance=body.min_importance,
        namespace=body.namespace,
    )
    return [_entry_to_response(e) for e in results]
