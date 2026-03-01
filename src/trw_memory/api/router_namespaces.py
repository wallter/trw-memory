"""Namespace management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from trw_memory.api.deps import get_backend
from trw_memory.exceptions import ConfigError
from trw_memory.namespace import validate_namespace
from trw_memory.storage.sqlite_backend import SQLiteBackend

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateNamespaceRequest(BaseModel):
    """Request body for registering a namespace."""

    namespace: str


class NamespaceInfo(BaseModel):
    """Namespace details returned by the API."""

    namespace: str
    entry_count: int = 0


class NamespaceListResponse(BaseModel):
    """List of namespaces."""

    namespaces: list[NamespaceInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=201, response_model=NamespaceInfo)
def create_namespace(
    body: CreateNamespaceRequest,
    backend: SQLiteBackend = Depends(get_backend),
) -> NamespaceInfo:
    """Register a namespace (validates the pattern)."""
    try:
        validate_namespace(body.namespace)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    count = backend.count(namespace=body.namespace)
    return NamespaceInfo(namespace=body.namespace, entry_count=count)


@router.get("", response_model=NamespaceListResponse)
def list_namespaces(
    backend: SQLiteBackend = Depends(get_backend),
) -> NamespaceListResponse:
    """List all distinct namespaces that have entries."""
    ns_list = backend.list_namespaces()
    infos = [
        NamespaceInfo(namespace=ns, entry_count=backend.count(namespace=ns))
        for ns in ns_list
    ]
    return NamespaceListResponse(namespaces=infos)


@router.get("/{ns}", response_model=NamespaceInfo)
def get_namespace(
    ns: str,
    backend: SQLiteBackend = Depends(get_backend),
) -> NamespaceInfo:
    """Get namespace details (entry count)."""
    try:
        validate_namespace(ns)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    count = backend.count(namespace=ns)
    if count == 0:
        raise HTTPException(status_code=404, detail="Namespace not found or empty")
    return NamespaceInfo(namespace=ns, entry_count=count)


@router.delete("/{ns}", status_code=204)
def delete_namespace(
    ns: str,
    backend: SQLiteBackend = Depends(get_backend),
) -> None:
    """Delete all entries in a namespace."""
    try:
        validate_namespace(ns)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    backend.delete_by_namespace(ns)
