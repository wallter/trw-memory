"""FastAPI application factory for trw-memory REST API."""

from __future__ import annotations

from fastapi import FastAPI

from trw_memory.api.auth import api_key_middleware
from trw_memory.api.router_health import router as health_router
from trw_memory.api.router_jobs import router as jobs_router
from trw_memory.api.router_memories import router as memories_router
from trw_memory.api.router_namespaces import router as namespaces_router


def create_app() -> FastAPI:
    """Build and return the trw-memory FastAPI application."""
    app = FastAPI(title="trw-memory", version="0.1.0", root_path="/v1")

    # Middleware
    app.middleware("http")(api_key_middleware)

    # Routers
    app.include_router(memories_router, prefix="/memories", tags=["memories"])
    app.include_router(namespaces_router, prefix="/namespaces", tags=["namespaces"])
    app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
    app.include_router(health_router, tags=["health"])

    return app
