"""Regenerate ``memory.db`` with the RELEASED trw-memory 2.0.0 (PRD-CORE-293 R1 fixture).

Run under an interpreter that has exactly ``trw-memory==2.0.0`` installed, never
this checkout:  uv venv v200 && uv pip install -p v200/bin/python trw-memory==2.0.0
&& v200/bin/python make_store.py
"""

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

assert importlib.metadata.version("trw-memory") == "2.0.0", "must be the released 2.0.0 wheel"

NS = "project/legacy"
WHEN = datetime(2026, 9, 20, tzinfo=timezone.utc)
out = Path(__file__).with_name("memory.db")
out.unlink(missing_ok=True)
with SQLiteBackend(out) as backend:
    backend.store(
        MemoryEntry(
            id="L-default",
            content="row left at the legacy column defaults",
            namespace=NS,
            created_at=WHEN,
            updated_at=WHEN,
        )
    )
    backend.store(
        MemoryEntry(
            id="L-populated",
            content="row with populated legacy reward values",
            namespace=NS,
            created_at=WHEN,
            updated_at=WHEN,
            q_value=0.9,
            q_observations=7,
            helpful_count=4,
            unhelpful_count=2,
        )
    )
