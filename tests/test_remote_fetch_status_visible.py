"""W13 — "no remote matches" and "every remote item was refused" are different answers.

``fetch_shared_memories`` returned a bare list. Six distinct outcomes left it as
``[]``: sync off, an unusable platform URL, a non-200, an unparseable body, a
transport error, and a peer whose every item the admission gate refused. The
refusal count existed -- ``admit_remote_results`` computed it -- and was spent on
a log line before the return.

These tests drive the real fetch with a stubbed HTTP transport (the network
boundary, not the unit under test) and a real SQLite-backed admission gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync import fetch_shared_memories

pytestmark = pytest.mark.integration

_ITEMS: list[dict[str, object]] = [
    {"source_learning_id": "R-1", "summary": "a peer learning", "detail": "d1"},
    {"source_learning_id": "R-2", "summary": "another peer learning", "detail": "d2"},
]


def _config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(
        storage_backend="sqlite",
        storage_path=str(tmp_path),
        sync_enabled=True,
        platform_url="https://api.test.invalid",
        local_only=False,
    )


def _transport(mock_cls: MagicMock, payload: object) -> None:
    """Stub the httpx client so the platform answers 200 with *payload*."""
    client = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = json.loads(json.dumps(payload))
    client.post.return_value = response
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    mock_cls.return_value = client


def test_every_item_refused_is_not_the_same_as_an_empty_corpus(tmp_path: Path) -> None:
    """Both return no results; only one of them says the peer sent anything."""
    backend = SQLiteBackend(tmp_path / "refused.db")
    try:
        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            _transport(mock_cls, _ITEMS)
            refusal = MagicMock()
            refusal.quarantined = True
            refusal.entry = MagicMock()
            with (
                patch("trw_memory.security.runtime.prepare_entry_for_store", return_value=refusal),
                patch("trw_memory.security.runtime.store_quarantined_entry"),
            ):
                refused = fetch_shared_memories("query", _config(tmp_path), backend=backend)

        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            _transport(mock_cls, [])
            empty = fetch_shared_memories("query", _config(tmp_path), backend=backend)
    finally:
        backend.close()

    assert refused.results == empty.results == []
    assert (refused.status, refused.fetched, refused.refused) == ("refused", 2, 2)
    assert (empty.status, empty.fetched, empty.refused) == ("ok", 0, 0)


def test_a_partial_refusal_is_reported_as_partial(tmp_path: Path) -> None:
    """Results came back AND something was dropped: both facts survive."""
    backend = SQLiteBackend(tmp_path / "partial.db")
    verdicts = [MagicMock(quarantined=True, entry=MagicMock()), MagicMock(quarantined=False)]
    try:
        with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
            _transport(mock_cls, _ITEMS)
            with (
                patch("trw_memory.security.runtime.prepare_entry_for_store", side_effect=verdicts),
                patch("trw_memory.security.runtime.store_quarantined_entry"),
            ):
                fetched = fetch_shared_memories("query", _config(tmp_path), backend=backend)
    finally:
        backend.close()

    assert len(fetched.results) == 1
    assert (fetched.status, fetched.fetched, fetched.refused) == ("partial", 2, 1)


async def test_the_recall_consumer_reads_the_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A field nobody reads would be a new defect, so assert the caller reads it.

    ``MemoryClient.recall(include_shared=True)`` is the production consumer. It
    must report a refused fetch rather than merging an empty list in silence --
    which is exactly what it did while the status did not exist.
    """
    from trw_memory.client import MemoryClient
    from trw_memory.sync import SharedFetchResult

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
    monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
    monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.invalid")
    monkeypatch.setenv("MEMORY_PLATFORM_API_KEY", "test-key")

    client = MemoryClient(namespace="default", mode="local")
    await client.store("local entry", importance=0.8)
    refused = SharedFetchResult([], "refused", 2, 2)
    with (
        patch("trw_memory.client.fetch_shared_memories", return_value=refused),
        patch("trw_memory.client.logger") as mock_logger,
    ):
        results = await client.recall("entry", include_shared=True)
    await client.close()

    # The local half of the answer is untouched -- this reports, it does not fail.
    assert [result["source"] for result in results] == ["local"]
    degraded = [
        call for call in mock_logger.warning.call_args_list if call.args[:1] == ("memory_shared_fetch_degraded",)
    ]
    assert len(degraded) == 1
    assert degraded[0].kwargs["outcome"] == "refused"
    assert degraded[0].kwargs["refused"] == 2
