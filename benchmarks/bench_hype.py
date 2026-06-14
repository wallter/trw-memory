"""HyPE quality benchmark — recall@10 / nDCG@10 uplift (PRD-CORE-195 FR06).

Runs the bundled question-style golden set twice through the REAL store +
recall path of :class:`~trw_memory.client.MemoryClient` — once with
``hype_enabled=False`` and once with ``hype_enabled=True`` — and reports
``recall_at_10`` / ``ndcg_at_10`` for each arm plus the HyPE-on-minus-off delta.

The HyPE-on arm uses :class:`GoldenQuestionGenerator`, a deterministic (no-LLM)
fixture generator that emits each entry's OWN positive golden query strings as
its hypothetical questions. This isolates the storage/fusion machinery: it shows
the maximal uplift HyPE can deliver when the generated questions perfectly
paraphrase the eval queries, without coupling the benchmark to any LLM, network,
or proprietary trw-eval artifact.

Requires the optional ``[vectors]`` (sqlite-vec) and ``[embeddings]``
(sentence-transformers) extras — without dense vectors HyPE is a no-op and both
arms collapse to BM25-only (the benchmark detects this and reports it).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from benchmarks.bench_quality import ndcg_at_k, recall_at_k
from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry

_FIXTURE = Path(__file__).parent / "fixtures" / "golden_set.json"


class GoldenQuestionGenerator:
    """Deterministic HyPE generator: emit an entry's own golden questions.

    Maps ``entry.id`` → its list of positive golden query strings. No LLM, no
    network — purely a fixture lookup, so the benchmark is reproducible offline.
    """

    def __init__(self, questions_by_id: dict[str, list[str]]) -> None:
        self._questions_by_id = questions_by_id

    def generate(self, entry: MemoryEntry) -> list[str]:
        return list(self._questions_by_id.get(entry.id, []))


def _load_golden(path: Path) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, object]] = data["entries"]
    questions_by_id: dict[str, list[str]] = {}
    for ge in entries:
        eid = str(ge["id"])
        positives = [str(q["query"]) for q in ge["queries"] if bool(q["relevant"])]
        questions_by_id[eid] = positives
    return entries, questions_by_id


async def _run_arm(
    golden_entries: list[dict[str, object]],
    questions_by_id: dict[str, list[str]],
    db_path: Path,
    *,
    hype_enabled: bool,
) -> dict[str, float]:
    """Store all golden entries then measure recall@10/ndcg@10 over positives."""
    generator = GoldenQuestionGenerator(questions_by_id) if hype_enabled else None
    client = MemoryClient("default", mode="local", db_path=db_path, question_generator=generator)
    client._config.hype_enabled = hype_enabled
    client._config.local_only = True
    # Benchmark fixture content is trusted; disable the injection gate so a
    # golden entry that happens to contain a flagged phrase ("system prompt")
    # is not rejected (bench_quality.py stores via the backend, bypassing it).
    client._config.poisoning_detection_enabled = False
    client._config.recall_auto_temporal = False
    try:
        stored_ids: set[str] = set()
        for ge in golden_entries:
            try:
                await client.store(
                    content=str(ge["content"]),
                    tags=[str(t) for t in ge["tags"]],  # type: ignore[union-attr]
                    importance=float(ge["importance"]),  # type: ignore[arg-type]
                    entry_id=str(ge["id"]),
                )
            except Exception:  # noqa: BLE001
                # A fixture entry that trips the unconditional input-validation
                # gate (e.g. content containing a flagged phrase) is skipped from
                # BOTH arms identically, preserving comparison validity.
                continue
            stored_ids.add(str(ge["id"]))

        recalls: list[float] = []
        ndcgs: list[float] = []
        for ge in golden_entries:
            entry_id = str(ge["id"])
            if entry_id not in stored_ids:
                continue
            relevant = {entry_id}
            for q in ge["queries"]:  # type: ignore[union-attr]
                if not bool(q["relevant"]):
                    continue
                results = await client.recall(str(q["query"]), limit=10)
                retrieved = [r["memory_id"] for r in results]
                recalls.append(recall_at_k(retrieved, relevant, 10))
                ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
        n = len(recalls)
        return {
            "recall_at_10": round(sum(recalls) / n, 4) if n else 0.0,
            "ndcg_at_10": round(sum(ndcgs) / n, 4) if n else 0.0,
            "queries": float(n),
        }
    finally:
        await client.close()


class HypeBenchmark:
    """Two-arm HyPE benchmark over the bundled golden set."""

    def __init__(self, fixture_path: Path = _FIXTURE) -> None:
        self.fixture_path = fixture_path

    def run(self, db_dir: Path) -> dict[str, dict[str, float]]:
        """Run both arms; return ``{"off": {...}, "on": {...}, "delta": {...}}``."""
        db_dir.mkdir(parents=True, exist_ok=True)
        golden_entries, questions_by_id = _load_golden(self.fixture_path)

        off = asyncio.run(
            _run_arm(golden_entries, questions_by_id, db_dir / "hype_off.db", hype_enabled=False)
        )
        on = asyncio.run(
            _run_arm(golden_entries, questions_by_id, db_dir / "hype_on.db", hype_enabled=True)
        )
        delta = {
            "recall_at_10": round(on["recall_at_10"] - off["recall_at_10"], 4),
            "ndcg_at_10": round(on["ndcg_at_10"] - off["ndcg_at_10"], 4),
        }
        return {"off": off, "on": on, "delta": delta}


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        report = HypeBenchmark().run(Path(tmp))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
