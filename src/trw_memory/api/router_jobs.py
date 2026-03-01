"""Background job management endpoints."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from trw_memory.api.deps import get_backend
from trw_memory.storage.sqlite_backend import SQLiteBackend

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Lifecycle status of a background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    """Representation of a background job."""

    id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime | None = None
    result: dict[str, str] = Field(default_factory=dict)
    error: str = ""


class CreateJobRequest(BaseModel):
    """Request body for submitting a background job."""

    job_type: str  # "consolidation" or "tier_sweep"


class JobResponse(BaseModel):
    """Serialised job returned by the API."""

    id: str
    job_type: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    result: dict[str, str]
    error: str


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def _run_job(job_id: str, backend: SQLiteBackend) -> None:
    """Execute a job in a background thread."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.RUNNING

    try:
        if job.job_type == "consolidation":
            count = backend.count()
            result = {"entries_scanned": str(count), "action": "consolidation_noop"}
        elif job.job_type == "tier_sweep":
            count = backend.count()
            result = {"entries_scanned": str(count), "action": "tier_sweep_noop"}
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

        with _lock:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result = result

    except Exception as exc:  # noqa: BLE001
        with _lock:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error = str(exc)


def _job_to_response(job: Job) -> JobResponse:
    """Convert a Job to a JobResponse."""
    return JobResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status.value if isinstance(job.status, JobStatus) else str(job.status),
        created_at=job.created_at,
        completed_at=job.completed_at,
        result=job.result,
        error=job.error,
    )


def reset_jobs() -> None:
    """Clear the in-memory job store (for testing)."""
    with _lock:
        _jobs.clear()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=201, response_model=JobResponse)
def create_job(
    body: CreateJobRequest,
    backend: SQLiteBackend = Depends(get_backend),
) -> JobResponse:
    """Submit a background job (consolidation or tier_sweep)."""
    if body.job_type not in ("consolidation", "tier_sweep"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid job_type: {body.job_type}. Must be 'consolidation' or 'tier_sweep'.",
        )

    job_id = f"J-{uuid.uuid4().hex[:12]}"
    job = Job(id=job_id, job_type=body.job_type)

    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run_job, args=(job_id, backend), daemon=True)
    thread.start()
    # Wait briefly for the thread to complete (jobs are fast in the stub impl)
    thread.join(timeout=2.0)

    return _job_to_response(job)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    """Poll a job's status."""
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.get("", response_model=list[JobResponse])
def list_jobs() -> list[JobResponse]:
    """List all recent jobs."""
    with _lock:
        all_jobs = list(_jobs.values())
    # Return most recent first
    all_jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [_job_to_response(j) for j in all_jobs[:50]]
