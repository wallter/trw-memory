"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from trw_memory._version import __version__
from trw_memory.api.deps import get_backend
from trw_memory.storage.sqlite_backend import SQLiteBackend

router = APIRouter()


class ComponentStatus(BaseModel):
    """Status of an individual component."""

    database: str = "ok"
    version: str = __version__


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    components: ComponentStatus = ComponentStatus()


@router.get("/health", response_model=HealthResponse)
def health_check(
    backend: SQLiteBackend = Depends(get_backend),
) -> HealthResponse:
    """Return health status with component checks."""
    db_status = "ok"
    try:
        backend.count()
    except Exception:
        db_status = "degraded"

    return HealthResponse(
        status="healthy" if db_status == "ok" else "degraded",
        components=ComponentStatus(database=db_status, version=__version__),
    )
