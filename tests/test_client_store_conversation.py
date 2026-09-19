"""``MemoryClient.store_conversation``: turns become context-carrying entries."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trw_memory._client_conversation import ConversationMessage, conversation_requests


def _turns() -> list[ConversationMessage]:
    return [
        {"role": "user", "speaker": "Caroline", "content": "I went to a LGBTQ support group yesterday."},
        {"role": "assistant", "speaker": "Melanie", "content": "That's great! What did it look like?"},
        {"role": "user", "speaker": "Caroline", "content": "A circle of chairs, very welcoming."},
    ]


class TestConversationRequests:
    def test_detail_carries_preceding_turns_and_content_stays_verbatim(self) -> None:
        reqs = conversation_requests(_turns(), context_turns=1)
        assert [r.content for r in reqs] == [
            "Caroline: I went to a LGBTQ support group yesterday.",
            "Melanie: That's great! What did it look like?",
            "Caroline: A circle of chairs, very welcoming.",
        ]
        assert reqs[0].detail == ""  # no preceding turn, no observed_at
        assert reqs[1].detail == "Caroline: I went to a LGBTQ support group yesterday."
        assert reqs[2].detail == "Melanie: That's great! What did it look like?"

    def test_context_window_widens_with_context_turns(self) -> None:
        reqs = conversation_requests(_turns(), context_turns=2)
        assert reqs[2].detail == (
            "Caroline: I went to a LGBTQ support group yesterday. | Melanie: That's great! What did it look like?"
        )

    def test_zero_context_stores_bare_turns(self) -> None:
        assert all(r.detail == "" for r in conversation_requests(_turns(), context_turns=0))

    def test_preceding_seeds_the_window_for_chunked_feeds(self) -> None:
        reqs = conversation_requests(_turns()[1:2], context_turns=1, preceding=["Caroline: earlier turn"])
        assert reqs[0].detail == "Caroline: earlier turn"

    def test_metadata_carries_role_speaker_observed_at_and_turn_index(self) -> None:
        when = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
        reqs = conversation_requests(_turns(), observed_at=when, session_id="s1", metadata={"dia_id": "D1:3"})
        assert reqs[1].metadata == {
            "dia_id": "D1:3",
            "turn_index": "1",
            "role": "assistant",
            "speaker": "Melanie",
            "observed_at": "2023-05-08T13:56:00+00:00",
            "session_id": "s1",
        }
        # The conversation id is row metadata, never the rate-limited writer session (L-8hyp).
        assert reqs[1].session_id is None
        assert reqs[1].source == "human"

    def test_per_turn_observed_at_overrides_call_level(self) -> None:
        turns = _turns()
        turns[0]["observed_at"] = "2023-01-01T00:00:00+00:00"
        reqs = conversation_requests(turns, observed_at="2023-05-08T00:00:00+00:00")
        assert reqs[0].metadata is not None and reqs[0].metadata["observed_at"] == "2023-01-01T00:00:00+00:00"
        assert reqs[1].metadata is not None and reqs[1].metadata["observed_at"] == "2023-05-08T00:00:00+00:00"

    def test_blank_turns_are_skipped_and_speaker_prefix_not_doubled(self) -> None:
        reqs = conversation_requests(
            [{"content": "   "}, {"speaker": "Ann", "content": "Ann: already prefixed"}], context_turns=1
        )
        assert [r.content for r in reqs] == ["Ann: already prefixed"]

    def test_negative_context_rejected(self) -> None:
        with pytest.raises(ValueError, match="context_turns"):
            conversation_requests(_turns(), context_turns=-1)


class TestStoreConversation:
    async def test_stores_turns_and_recall_finds_context_dependent_reply(self, memory_client) -> None:
        summary = await memory_client.store_conversation(_turns(), observed_at="2023-05-08T13:56:00+00:00")
        assert summary.succeeded == 3
        rows = await memory_client.recall("support group chairs welcoming", limit=3, include_org_memories=False)
        contents = [r["content"] for r in rows]
        assert "Caroline: A circle of chairs, very welcoming." in contents
        by_content = {r["content"]: r for r in rows}
        row = by_content["Caroline: A circle of chairs, very welcoming."]
        assert row["detail"] == "Melanie: That's great! What did it look like? | May 2023"
        assert row["metadata"]["observed_at"] == "2023-05-08T13:56:00+00:00"

    async def test_session_month_makes_a_turn_findable_by_date(self, memory_client) -> None:
        await memory_client.store_conversation(
            [{"speaker": "Ann", "content": "We finally repainted the kitchen."}],
            observed_at="2023-05-08T13:56:00+00:00",
        )
        await memory_client.store_conversation(
            [{"speaker": "Ann", "content": "We finally repainted the garage."}],
            observed_at="2023-11-02T09:00:00+00:00",
        )
        rows = await memory_client.recall("what did Ann repaint in November 2023", limit=1, include_org_memories=False)
        assert rows[0]["content"] == "Ann: We finally repainted the garage."

    async def test_empty_conversation_is_a_noop_summary(self, memory_client) -> None:
        summary = await memory_client.store_conversation([])
        assert (summary.total, summary.succeeded) == (0, 0)


def test_observed_month_is_carried_in_detail_and_bad_dates_are_ignored() -> None:
    reqs = conversation_requests(_turns()[:2], observed_at="2023-05-08T13:56:00+00:00", context_turns=1)
    assert reqs[0].detail == "May 2023"
    assert reqs[1].detail == "Caroline: I went to a LGBTQ support group yesterday. | May 2023"
    bad = conversation_requests(_turns()[:1], observed_at="last tuesday", context_turns=1)
    assert bad[0].detail == ""
    assert bad[0].metadata is not None and bad[0].metadata["observed_at"] == "last tuesday"


def test_z_dates_are_normalized_for_python310_parser(monkeypatch) -> None:
    from trw_memory import _client_conversation as conversation

    class Python310Datetime:
        @staticmethod
        def fromisoformat(value):
            assert not value.endswith("Z"), "Python 3.10 cannot parse the UTC Z suffix"
            return datetime.fromisoformat(value)

    monkeypatch.setattr(conversation, "datetime", Python310Datetime)
    assert conversation._date_words("2023-05-08T13:56:00Z") == "May 2023"
    assert conversation._date_words("2023-05-08T13:56:00+00:00") == "May 2023"
    assert conversation._date_words("not a date") == ""
