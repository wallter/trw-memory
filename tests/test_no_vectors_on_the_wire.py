"""PRD-CORE-302 FR04: no public package sends vectors to the platform.

Both SDK egress paths run for real against a stubbed HTTP transport (the
network boundary). The client holds a working embedder throughout, so a
regression that re-attaches a query or entry vector would reach the body.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from trw_memory.client import MemoryClient

pytestmark = pytest.mark.integration


def _transport(mock_cls: MagicMock, answer: object) -> MagicMock:
    client = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = answer
    client.post.return_value = response
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    mock_cls.return_value = client
    return client


def _bodies(http: MagicMock, path: str) -> list[dict[str, object]]:
    return [json.loads(json.dumps(c.kwargs["json"])) for c in http.post.call_args_list if c.args[0].endswith(path)]


@pytest.fixture()
def synced_client(client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    client._config = client._config.model_copy(
        update={"sync_enabled": True, "platform_url": "https://api.test.invalid"}
    )
    embedder = Mock()
    embedder.available.return_value = True
    embedder.embed.return_value = [1.0, 0.0]
    embedder.embed_query.return_value = [1.0, 0.0]
    embedder.embed_batch.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(client, "_get_embedder", lambda: embedder)
    return client


async def test_org_shared_recall_sends_no_embedding(synced_client: MemoryClient) -> None:
    with patch("trw_memory.sync._remote_fetch.httpx.Client") as mock_cls:
        http = _transport(mock_cls, [])
        await synced_client.recall("deploy rollback", include_shared=True, include_org_memories=False)

    (body,) = _bodies(http, "/v1/learnings/search")
    assert body["query"] == "deploy rollback"
    assert "embedding" not in body


def test_publish_sends_no_embedding(synced_client: MemoryClient) -> None:
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.sync.remote import publish_memory_result

    with patch("trw_memory.sync._remote_publish.httpx.Client") as mock_cls:
        http = _transport(mock_cls, {"id": "7"})
        result = publish_memory_result(
            MemoryEntry(id="L-1", content="a learning", importance=0.9), synced_client._config
        )

    assert result["success"] is True
    (body,) = _bodies(http, "/v1/learnings")
    assert body["summary"] == "a learning"
    assert "embedding" not in body


def test_retry_drain_strips_a_vector_queued_before_the_change(synced_client: MemoryClient) -> None:
    from trw_memory.sync.remote import drain_retry_queue

    queue = synced_client._retry_queue
    assert queue.enqueue("L-2", {"summary": "queued", "source_learning_id": "L-2", "embedding": [0.1, 0.2]})
    with patch("trw_memory.sync._remote_publish.httpx.Client") as mock_cls:
        http = _transport(mock_cls, {"id": "8"})
        assert drain_retry_queue(queue, synced_client._config)["drained"] == 1

    (body,) = _bodies(http, "/v1/learnings")
    assert body["summary"] == "queued"
    assert "embedding" not in body
