# EngMem — a benchmark for engineering memory

Most memory benchmarks ask whether a system can recall what two people said to each other. That is
not what a memory system for engineers does. It carries learnings, incidents, conventions and
decisions across sessions, in a corpus where **facts get superseded** — and where handing an agent a
retired convention is worse than handing it nothing at all.

EngMem measures that job. It is LLM-free, deterministic under a seed, and runs in seconds on a
laptop.

```bash
# free structural baselines, no embedding model needed
python -m benchmarks.engmem.run --synth --no-trw

# the full suite
python -m benchmarks.engmem.run --synth --distractors 1000 --fresh

# quality AND cost as the store grows
python -m benchmarks.engmem.scale --sizes 1000,5000,20000
```

## Two metrics the conversational benchmarks cannot express

**`complete@k`** — every required row present, not one hit out of three. `hit@k` and MRR both score a
single hit as success. On a conversational benchmark that hid a multi-hop bottleneck from us for an
entire round of work: mean recall@10 read 51.8% while `complete@10` was 31.7%. Averaging was doing
the hiding.

**`forbidden@k`** — retired or out-of-scope rows that reached the agent. This is the metric that
separates a memory system from a search engine. A system that answers well and occasionally hands
over a superseded instruction is *worse* than one that stays quiet, and no metric in the
hit@k/recall@k family can say so.

## Why `forbidden@k` matters, in one measurement

At 1,000 rows, identical corpus, identical queries:

| arm | hit@10 | **forbidden@10** |
|---|---|---|
| hybrid retrieval with lifecycle filtering | 100.0% | **0.0%** |
| substring scan ("grep") | 25.0% | **50.0%** |

Grep finds a quarter of the answers, and a retired convention appears in its ranked list on half of
all queries.

Read that narrowly. EngMem scores ranked **identifiers** -- no model reads the results -- so this
measures that stale records are *retrieved*, not that they produced a wrong answer. It is staleness,
not fabrication; a retired record is real and was once true. And the grep arm here holds
`(id, timestamp, text)` with no status field, so it *cannot* filter retired rows: a real engineer
greping notes could archive superseded files or filter a `status:` marker and do better.

What survives is narrower and still worth measuring: grep has no intrinsic notion that one fact
replaced another. You can filter an explicit marker, but maintaining that marker is itself the
memory-system function -- the difference is whether the store enforces it or a human does, forever.

This is not a tuning gap that a better embedding model closes. Independent work
([MemStrata, arXiv:2606.26511](https://arxiv.org/pdf/2606.26511)) measures that cosine similarity
separates a *contradicted* fact from a mere *duplicate* at **AUROC 0.59 — near chance** — because a
superseding fact is usually *more* embedding-similar to what it replaces than a paraphrase is.
Supersession has to be a deterministic property of the data model. It cannot be delegated to a
similarity score.

## Design

**Time order is load-bearing.** Events replay in timestamp order into a fresh store and every query
carries an `as_of`. A learning written after the commit it would have helped is leakage, not
evidence, and the replay makes that structurally impossible rather than filtering it afterwards —
because retrieving from a full store and filtering by timestamp after the fact still leaks through
BM25 statistics and the candidate pool. `leakage_check()` runs on every suite and refuses to score
one whose gold postdates its own query.

**Gold is planted, not inferred.** `synth.py` generates a seeded corpus with known answers, so a
scorer bug shows up as an impossible number rather than a plausible one. Distractors are drawn from
a different slot space than gold, so generation can never accidentally emit a duplicate of the
answer.

**Baselines are mandatory.** Every table includes recency (ignore the query, return newest) and grep
(substring scan). A memory engine that cannot beat a sorted list and a substring match is not
earning its complexity, and publishing without those rows hides that.

**Cost is reported beside quality.** `scale.py` reports p50/p95 per retrieval, replay wall time and
bytes on disk at each store size. A retrieval policy that holds its recall while going superlinear
in time has not scaled; reporting quality alone would call that a success.

## Tasks

The suite is the public half of a ten-task design (C1-C10). Implemented here: **C1** before-edit
recall, **C3** current-above-superseded. The remaining tasks — as-of time travel, cross-project
isolation, dedup quality, contradiction detection, provenance poisoning, forgetting stale knowledge,
session-start relevance within budget — share the same schema and replay engine.

## Writing your own arm

An arm returns ranked `(id, chars)` best-first:

```python
class MyArm:
    name = "mine"
    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        ...
```

`as_of` is enforced by the replay — the store only contains events up to the query's instant — so an
arm must not filter by time itself. Doing so masks a leak rather than preventing one.

**One warning, learned the expensive way.** If your arm *reimplements* the code path you are trying
to measure, it will diverge from it, and the divergence will look like a finding. Ours diverged three
times in a row — a missing `limit` that silently ranked 100 rows, a document encoder used to embed a
query, and a missing status filter that surfaced rows the product would never return. Each produced a
plausible, publishable-looking number, and each was wrong. **Call the real function, or accept that
you are measuring your own copy.**

## Reproducibility

Every run prints the gold SHA-256 and the leakage audit. `holdout()` keeps a stable 20% by hash of
the query id, so adding queries never reshuffles the existing split. Generation is deterministic
under `--seed`.
