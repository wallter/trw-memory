"""Tests for read-time row quarantining (P2 — auto-recovery layer).

Strategy: insert good rows via the backend, then directly inject a bad-UTF-8
row using a raw sqlite3 connection with text_factory=bytes. This simulates the
exact incident scenario (a corrupt inode yielding undecoded bytes).

All tests use in-memory SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import structlog.testing

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(entry_id: str, detail: str = "clean detail") -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content="content",
        detail=detail,
        namespace="default",
        source="agent",  # type: ignore[arg-type]
    )


def _inject_bad_utf8_row(db_path: Path | str, entry_id: str) -> None:
    """Directly insert a row with invalid UTF-8 bytes in the `detail` column.

    We use a normal text connection to write the SQL but use a custom text_factory
    on a second connection to verify.  The trick: SQLite stores blobs verbatim, so
    we write the bad bytes as a BLOB then re-read via text_factory=bytes.

    Simpler alternative used here: write via executescript with hex literal so we
    bypass Python's str codec entirely.
    """
    # Write the row using a regular str connection with explicit hex blob.
    # \x80\x81 are bare UTF-8 continuation bytes with no lead byte — invalid.
    bad_hex = "bad\x80\x81bytes"
    # We cannot embed raw non-UTF8 bytes in a Python str literal.
    # Instead, use sqlite3's X'...' hex blob syntax to inject the raw bytes.
    bad_bytes_hex = "6261648081627974 6573"  # "bad\x80\x81bytes" in hex (spaces ignored)
    # Build via bytes object:
    bad_bytes = b"bad\x80\x81bytes"
    bad_hex_str = bad_bytes.hex()  # "6261648081627974657300..." — no spaces

    conn = sqlite3.connect(str(db_path))
    # Minimal INSERT with only required columns; schema allows NULLs for the rest.
    conn.execute(
        """
        INSERT OR REPLACE INTO memories (
            id, content, detail, tags, evidence, importance, status,
            recurrence, namespace, created_at, updated_at, access_count,
            session_count, q_value, q_observations, source,
            source_identity, client_profile, model_id,
            merged_from, consolidated_from, outcome_history,
            assertions, anchors, anchor_validity,
            type, nudge_line, expires_at, confidence,
            task_type, domain, phase_origin, phase_affinity,
            team_origin, protection_tier, sessions_surfaced,
            outcome_correlation, sync_hash, sync_seq,
            recall_count, helpful_count, unhelpful_count,
            vector_clock, metadata,
            published_to_platform, pending_delete, cross_validated
        ) VALUES (
            ?, ?, CAST(X'""" + bad_hex_str + """' AS TEXT),
            '[]', '[]', 0.5, 'active',
            1, 'default', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00', 0,
            0, 0.5, 0, 'agent',
            '', '', '',
            '[]', '[]', '[]',
            '[]', '[]', 1.0,
            'fact', '', '', 'medium',
            '', '[]', '', '[]',
            '', 'standard', 0,
            '', '', 0,
            0, 0, 0,
            '{}', '{}',
            0, 0, 0
        )
        """,
        (entry_id, "content"),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Test: bad row quarantined, good rows returned
# ---------------------------------------------------------------------------


def test_list_entries_skips_bad_utf8_row(tmp_path: Path) -> None:
    """Bad-UTF-8 rows are skipped; good rows are returned; quarantine counter increments."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)

    # Insert a good entry via the backend
    good = _make_entry("M-good-001", "perfectly valid detail")
    backend.store(good)

    backend.close()

    # Inject a bad row directly
    _inject_bad_utf8_row(db_path, "M-bad-001")

    # Reopen and read
    backend2 = SQLiteBackend(db_path)
    with structlog.testing.capture_logs() as logs:
        results = backend2.list_entries(limit=100)

    ids = [e.id for e in results]
    assert "M-good-001" in ids, "Good row must be returned"
    assert "M-bad-001" not in ids, "Bad row must be quarantined"
    assert backend2.quarantine_count_utf8 >= 1

    quarantine_events = [
        log for log in logs if log.get("action") == "memory_row_utf8_quarantined"
    ]
    assert len(quarantine_events) >= 1, "Expected at least one quarantine log event"
    backend2.close()


# ---------------------------------------------------------------------------
# Test: all bad rows → empty list, no raise
# ---------------------------------------------------------------------------


def test_list_entries_returns_empty_list_when_all_bad(tmp_path: Path) -> None:
    """When every row is corrupted, list_entries returns [] without raising."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()

    # Inject 2 bad rows
    _inject_bad_utf8_row(db_path, "M-bad-001")
    _inject_bad_utf8_row(db_path, "M-bad-002")

    backend2 = SQLiteBackend(db_path)
    results = backend2.list_entries(limit=100)
    assert results == []
    assert backend2.quarantine_count_utf8 >= 2
    backend2.close()


# ---------------------------------------------------------------------------
# Test: clean DB leaves quarantine counter at 0
# ---------------------------------------------------------------------------


def test_list_entries_on_clean_db_no_quarantine() -> None:
    """Happy path: clean DB, quarantine counter stays at 0."""
    backend = SQLiteBackend(Path(":memory:"))
    entry = _make_entry("M-clean-001", "clean entry")
    backend.store(entry)

    results = backend.list_entries(limit=100)
    assert any(e.id == "M-clean-001" for e in results)
    assert backend.quarantine_count_utf8 == 0
    backend.close()
