# LongMemEval: LLM-free retrieval benchmark for trw-memory

This scores trw-memory's retrieval on [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(Wu et al., ICLR 2025, [arXiv:2410.10813](https://arxiv.org/abs/2410.10813), MIT).
It uses no answerer and no judge. Each question marks where its answer lives in
two ways: the answer sessions (`answer_session_ids`) and, inside them, the turns
flagged `has_answer`. The benchmark checks whether that evidence comes back in
`MemoryClient.recall`'s top-k. Runs are deterministic apart from nondeterminism
in the embedder, and the same metric shape as `../locomo/retrieval_eval.py`
makes the two benchmarks directly comparable.

## Dataset

Use the **cleaned** release: [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
The original `xiaowu0162/longmemeval` repo is marked deprecated upstream,
because it contained noisy history sessions that interfered with answer
correctness.

```bash
mkdir -p scratch/longmemeval   # repo-root scratch/ is gitignored; never commit the data
curl -L -o scratch/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json
shasum -a 256 scratch/longmemeval/longmemeval_s_cleaned.json
# d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442  (277,383,467 bytes)
```

`longmemeval_s_cleaned.json` contains 500 questions. Each has 38-62 sessions
(mean 47.7) and 396-616 turns (mean 493.5), for about 247k turns in total.
Thirty questions are abstention questions (`question_id` ends in `_abs`). They
have nothing to retrieve, so the benchmark excludes them, which leaves 470
scored questions.

## What it does

- **One store per question.** Every question gets a fresh directory,
  `<store>/<question_id>/memory`, and its own namespace. Two reasons:
  1. The haystacks share filler sessions, and trw-memory's cross-project graph
     pass reads sibling `project:` stores in the same storage directory and
     writes back to them.
  2. Security state is kept beside `storage_path`: the write-rate limiter, the
     size-anomaly baselines and the audit log. If questions shared a parent
     directory, one question's ingest would shift quarantine decisions for the
     next.
- **Product ingest.** Each session goes through `conversation_requests`, the
  shaping behind `MemoryClient.store_conversation`, then `bulk_store` (one call
  per session):
  - `context_turns=1`.
  - The session's `haystack_dates` entry becomes `observed_at`, which also adds
    the session's month and year to `detail`.
  - `{"content", "role"}` per turn, with no speaker prefix.
  - Each row carries `session_id`, `session_pos` (its position in the
    haystack), `session_date` and `turn_index` in metadata.
- **Query.** `recall(question, limit=max(k), include_org_memories=False)` with
  every other setting at its default. That default includes the 0.19.0
  cross-encoder rerank and the confidence floor, so recall can return fewer
  than `limit` rows.
- **Idempotent ingest.** A manifest (`lme_ingest.json`) records the row count
  after ingest. A later run reuses the store when the count still matches, so
  you can sweep `MEMORY_*` retrieval knobs without re-embedding.
  `--reingest` forces a rebuild.

## Metrics (per `question_type` and overall)

| metric | meaning |
|---|---|
| `hit@k` | at least one `has_answer` turn is in the top-k |
| `recall@k` | fraction of the question's `has_answer` turns in the top-k |
| `mrr` | reciprocal rank of the first `has_answer` turn |
| `session_recall@k` | fraction of `answer_session_ids` with at least one turn in the top-k (`session_hit@k` is in the JSON too) |
| `session_mrr` | reciprocal rank of the first turn from any answer session |

Turn identity is `(session_pos, turn_index)`, not the session id. Thirteen
haystacks list the same session id twice.

`--out` writes per-question JSON in the LOCOMO shape: `conv` is the question
id, `q` is 0, `category` is the `question_type`, and the `hit@k`/`mrr` keys
are the same. `scratch/paired_retr.py`-style paired McNemar comparisons
therefore work unchanged. Each record also includes `ingest_s`, `query_s`,
`rows`, `n_evidence`, `n_answer_sessions` and `n_evidence_dropped`.

## Running

```bash
PY=../../../.venv/bin/python   # from this directory; put THIS checkout's src first
export PYTHONPATH=$(cd ../../src && pwd)

# inner loop: a deterministic stratified 20-question sample
$PY retrieval_eval.py --store /tmp/lme-store --limit-questions 20 --workers 2

# one ability only
$PY retrieval_eval.py --store /tmp/lme-store --question-types multi-session,temporal-reasoning

# the full scored set, saved for paired comparison
$PY retrieval_eval.py --store /tmp/lme-store --workers 4 --label base --out /tmp/lme-base.json
MEMORY_RECALL_RERANK=false $PY retrieval_eval.py --store /tmp/lme-store --workers 4 \
  --label norerank --out /tmp/lme-norerank.json   # reuses the stores; no re-embedding
```

`--limit-questions N` draws a stratified sample. Each question type gets a
share proportional to its size (largest remainder, at least one per type).
Within a type, questions are ordered by a hash of their id. The sample is
therefore stable across runs, and a larger N contains every question from a
smaller N whenever the per-type allocation does not shrink.

`--workers N` runs questions in separate spawned processes. Each process loads
its own embedder and reranker, and the stores never overlap.

## Caveats

- **`question_date` is not used.** trw-memory has no "evaluate as of this
  conversation date" knob. `recall(as_of=...)` filters by each row's
  *bitemporal validity window*, which starts at ingest wall-clock time, so
  passing the 2023 question date would hide every row. Temporal-reasoning
  questions are therefore scored without the reference date that the reader
  model would get.
- **Poisoning guards drop some turns.** A small fraction of turns are rejected
  at ingest by `validate_store_inputs`. They are mostly in coding sessions:
  entries over 10,240 bytes and blocked patterns such as `<script` and
  `javascript:`. This is product behaviour and is kept deliberately.
  `n_evidence_dropped` counts `has_answer` turns lost this way, and the report
  prints a warning when any are lost.
- **Do not pass `session_id=` to `store_conversation` for long sessions.** That
  argument also keys the write rate limiter (10 writes per minute per session
  by default). In one call it silently rejects every turn after the tenth.
  This harness therefore puts the session id in metadata instead.
- **By default, `@50` is really "whatever recall returned".** The rerank
  confidence floor (`MEMORY_RECALL_RERANK_MIN_SCORE`, default -8) truncates
  results. On `longmemeval_s_cleaned`, recall returned 5-44 rows (median 10),
  so the default `hit@50` measures the product's actual answer set, not a
  top-50 list. To score a true top-k, pass
  `MEMORY_RECALL_RERANK_MIN_SCORE=-1000`. The stores are reused, so that run
  takes about two minutes. `n_returned` is recorded per question.
- Retrieval quality is not QA accuracy. The paper's end-to-end numbers need an
  answerer and a GPT-4o judge, and this benchmark measures neither.
