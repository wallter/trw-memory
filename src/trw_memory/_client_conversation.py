"""Conversation ingestion for ``MemoryClient`` -- turns become context-carrying entries.

Belongs to ``client.py``; delegated from ``MemoryClient.store_conversation``.

A chat turn on its own is a poor memory: "Cool! What did it look like?" says
nothing without the turn before it, so both BM25 and the embedder rank it at
random. ``store_conversation`` stores every turn verbatim as ``content`` (so a
row still maps 1:1 to what was said and when) and carries the preceding
``context_turns`` turns of the same conversation in ``detail``, which both
retrieval paths index. On LOCOMO evidence retrieval that one change lifted
hit@10 from 73.5% to 78.2% (385 questions, ``benchmarks/locomo``).

No LLM runs at ingest time. That is deliberate: mem0-style fact extraction
costs one to two model calls per turn, can hallucinate, and resolves relative
dates against the wrong "today". Keeping the raw turn plus its date and
neighbourhood preserves the evidence; the reader model does the inference at
recall time, when it knows the question.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from typing_extensions import NotRequired, TypedDict

from trw_memory._client_bulk_store import BulkStoreRequest, BulkStoreSummary

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient


#: One call is one bulk_store batch; larger feeds should be chunked with ``preceding=``.
MAX_CONVERSATION_MESSAGES = 5000


class ConversationMessage(TypedDict):
    """One turn. ``speaker`` (when given) is prefixed to the stored content so
    attribution survives retrieval; ``observed_at`` overrides the call-level
    timestamp for that turn."""

    content: str
    role: NotRequired[str]
    speaker: NotRequired[str]
    observed_at: NotRequired[datetime | str]


def _iso(value: datetime | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip


def _date_words(observed_iso: str) -> str:
    """``"2023-05-08T13:56:00+00:00"`` -> ``"May 2023"`` for the lexical index.

    Temporal questions name months and years ("what did she do in May 2023?")
    that never appear in the turn text. Indexing the session's month and year in
    ``detail`` (which BM25 and the embedder both see) lifted LOCOMO evidence
    hit@10 on temporal questions 71.0% -> 73.8% and overall 74.5% -> 76.2%
    before re-ranking (1,540 questions, benchmarks/locomo, 2026-09-18). A value
    that does not parse contributes nothing rather than a wrong date.
    """
    try:
        dt = datetime.fromisoformat(
            observed_iso.removesuffix("Z") + "+00:00" if observed_iso.endswith("Z") else observed_iso
        )
    except ValueError:  # trw-fail-silent-allow: a free-text observed_at ("last tuesday") has no month to index; it stays verbatim in metadata and adds no date words, which is the documented behaviour
        return ""
    return f"{_MONTHS[dt.month - 1]} {dt.year}"


def _turn_text(message: ConversationMessage) -> str:
    content = str(message.get("content", "")).strip()
    speaker = str(message.get("speaker", "")).strip()
    if content and speaker and not content.startswith(f"{speaker}:"):
        return f"{speaker}: {content}"
    return content


def conversation_requests(
    messages: Sequence[ConversationMessage],
    *,
    context_turns: int = 1,
    preceding: Sequence[str] = (),
    observed_at: datetime | str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    importance: float = 0.5,
    metadata: dict[str, str] | None = None,
    source: Literal["human", "agent", "tool", "consolidated"] = "human",
) -> list[BulkStoreRequest]:
    """Shape *messages* into ``BulkStoreRequest`` rows with rolling context.

    Pure and synchronous so callers (and tests) can inspect exactly what will
    be stored. ``preceding`` seeds the context window for callers that feed a
    conversation in chunks: pass the texts of the turns stored just before
    ``messages[0]`` and the first new turn still sees its neighbours.

    ``session_id`` identifies the RECORDED conversation: it is stamped into each
    row's ``metadata["session_id"]`` (and so into its signed provenance), and the
    rows' writer ``BulkStoreRequest.session_id`` stays ``None``. Conversation
    ingest is therefore not metered by the per-session write-rate limiter
    (``max_memory_writes_per_minute``), which exists to stop an agent flooding
    memory with its own writes -- a 600-turn transcript, or one turn per call at
    20 calls/second, stores every turn.
    """
    if context_turns < 0:
        raise ValueError(f"context_turns must be >= 0, got {context_turns}")
    if len(messages) > MAX_CONVERSATION_MESSAGES:
        raise ValueError(
            f"store_conversation accepts at most {MAX_CONVERSATION_MESSAGES} messages per call, got {len(messages)}"
        )
    window: list[str] = [t for t in preceding if t][-context_turns:] if context_turns else []
    default_observed = _iso(observed_at)
    requests: list[BulkStoreRequest] = []
    for index, message in enumerate(messages):
        text = _turn_text(message)
        if not text:
            continue
        turn_meta: dict[str, str] = dict(metadata or {})
        turn_meta["turn_index"] = str(index)
        role = str(message.get("role", "")).strip()
        if role:
            turn_meta["role"] = role
        speaker = str(message.get("speaker", "")).strip()
        if speaker:
            turn_meta["speaker"] = speaker
        observed = _iso(message.get("observed_at")) or default_observed
        if observed:
            turn_meta["observed_at"] = observed
        if session_id:
            turn_meta["session_id"] = session_id
        context = [*window]
        when = _date_words(observed) if observed else ""
        if when:
            context.append(when)
        requests.append(
            BulkStoreRequest(
                content=text,
                detail=" | ".join(context),
                tags=list(tags) if tags else None,
                importance=importance,
                metadata=turn_meta,
                source=source,
                # The conversation id lives in metadata (provenance reads it
                # from there), NOT in the writer-session slot: that slot keys
                # the write-rate limiter, which meters an agent's decisions to
                # write. A transcript's volume is set by the conversation, so
                # metering it dropped every turn past the tenth (L-8hyp).
                session_id=None,
            )
        )
        if context_turns:
            window = [*window, text][-context_turns:]
    return requests


async def store_conversation_impl(
    client: MemoryClient,
    messages: Sequence[ConversationMessage],
    *,
    context_turns: int = 1,
    preceding: Sequence[str] = (),
    observed_at: datetime | str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    importance: float = 0.5,
    metadata: dict[str, str] | None = None,
    source: Literal["human", "agent", "tool", "consolidated"] = "human",
) -> BulkStoreSummary:
    """Async impl for :meth:`MemoryClient.store_conversation`."""
    requests = conversation_requests(
        messages,
        context_turns=context_turns,
        preceding=preceding,
        observed_at=observed_at,
        session_id=session_id,
        tags=tags,
        importance=importance,
        metadata=metadata,
        source=source,
    )
    if not requests:
        return BulkStoreSummary(total=0, stored=0, updated=0, quarantined=0, rejected=0, duration_ms=0.0)
    return await client.bulk_store(requests)


__all__ = ["ConversationMessage", "conversation_requests", "store_conversation_impl"]
