# LOCOMO: trw-memory vs mem0, on mem0's own benchmark suite

This directory runs [mem0's `memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks)
LOCOMO pipeline **unmodified** against two backends behind one local REST shim:

| backend | what it is |
|---|---|
| `mem0` | the in-process `mem0ai` SDK (`Memory.from_config`), Ollama for fact extraction, local on-disk Qdrant |
| `trw`  | `trw_memory.MemoryClient`, one `project:` namespace per benchmark user, `store_conversation` ingestion |

Same dataset parsing, same one-turn-per-request chunking, same answerer prompt,
same judge prompt, same cutoffs, same metrics, and the **same embedding model**
(`all-MiniLM-L6-v2`) on both sides. Any gap is the memory system's design:
extraction, storage, retrieval.

## Files

| file | role |
|---|---|
| `bootstrap.sh` | clones `memory-benchmarks` at a pinned commit, applies `upstream-llm_client.patch`, fetches `locomo10.json` |
| `upstream-llm_client.patch` | the one upstream change: `LLM_EXTRA_BODY` env is merged into every chat call so Ollama reasoning models can be told `reasoning_effort=none` |
| `server.py` | the mem0-OSS-compatible shim (`POST /memories`, `POST /search`, `DELETE /memories`) over either backend |
| `run.sh` | starts the shim, runs `python -m benchmarks.locomo.run` against it with Ollama as answerer + judge |
| `retrieval_eval.py` | **LLM-free inner loop**: ingests turns into trw-memory and scores evidence hit@k / recall@k / MRR against LOCOMO's per-question evidence ids |
| `component_eval.py` | offline ranker comparison on an ingested store (bm25 / dense / rrf / combmax / stemming / cross-encoder) |

## Running

```bash
# judged end-to-end (answerer + judge = a local Ollama model; both backends get the same one)
BENCH_PY=../../../.venv/bin/python LLM_MODEL=llama3.1:32k \
  ./run.sh trw  my-run --conversations 0 --top-k 50 --top-k-cutoffs 10,50 --max-workers 1
BENCH_PY=../../../.venv/bin/python LLM_MODEL=llama3.1:32k \
  ./run.sh mem0 my-run --conversations 0 --top-k 50 --top-k-cutoffs 10,50 --max-workers 1

# retrieval-only inner loop (seconds, deterministic)
python retrieval_eval.py --store /tmp/locomo-store --context 1 --k 10,50
python component_eval.py --store /tmp/locomo-store --conversations 0,1,2 --by-category
```

`run.sh` puts *this checkout's* `trw-memory/src` on `PYTHONPATH`; the shared
venv's editable install points at the main tree otherwise.

Create the 32k-context Ollama variant once (`ollama` defaults to a 4k window,
which silently truncates a top-50 prompt):

```
printf 'FROM llama3.1:latest\nPARAMETER num_ctx 32768\n' > /tmp/Modelfile && ollama create llama3.1:32k -f /tmp/Modelfile
```

## Fairness notes (read before quoting numbers)

* **Timestamps.** mem0's OSS SDK rejects `timestamp=`; upstream's docker shim
  silently drops it, so every OSS memory is dated at ingest time. The shim passes
  the session date through `metadata.created_at`, which mem0 honours, so the
  answerer sees real conversation dates for both backends (as it would on Mem0 Cloud).
* **Relative dates.** mem0's extraction prompt resolves "yesterday" against the
  wall clock of the machine running the benchmark, not the session date. That is
  mem0's behaviour, not a shim artefact; trw-memory stores the verbatim turn plus
  its session date and leaves the inference to the answerer.
* **Judge.** Both backends are judged by the same local model. A small local judge
  is more lenient and noisier than the GPT-class judges in mem0's published table,
  so compare the two columns below with each other, not with mem0's website.
* **Retrieval scores.** trw-memory's `score` is rank-based after fusion; the
  answerer prompt never sees scores, so this does not leak into the comparison.
* **Statistical significance.** Per `CLAUDE.md`, report N and run count; a
  comparison needs non-overlapping Wilson 95% CIs or a McNemar test. One
  conversation is a smoke result, not a verdict.

## What the inner loop found (trw-memory, conversations 0-2, 385 questions)

Evidence hit@10 / hit@50, all-MiniLM-L6-v2, `component_eval.py`:

| change | hit@10 | hit@50 | MRR |
|---|---|---|---|
| raw turns, rrf@5 (the old default) | 73.5 | 89.6 | 40.7 |
| + preceding turn carried as `detail` (`store_conversation`, context_turns=1) | 78.2 | 90.6 | 45.5 |
| + BM25 query stopwords + suffix stemming | 79.7 | 92.5 | 46.1 |
| + cross-encoder rerank of the fused top-50 (now the default) | 85.5 | 92.5 | 61.2 |

Each of those is a product change in `trw_memory`, not a benchmark tweak.

Full product recall path on all 10 conversations (1,540 questions, `retrieval_eval.py`):
hit@10 84.0%, hit@50 89.4%, recall@50 83.0%, MRR 59.5%; single-hop 90.4 / 94.3,
multi-hop 78.7 / 89.0, temporal 81.9 / 85.4, open-domain 50.0 / 61.5 (hit@10 / hit@50).

## Results

Conversation 0, 152 questions per side, `llama3.1:32k` answerer + judge, paired:

| cutoff | mem0 (OSS) | trw-memory | McNemar p |
|---|---|---|---|
| top_10 | 88.2% [82.1, 92.4] | 91.4% [85.9, 94.9] | 0.38 |
| top_50 | 92.1% [86.7, 95.4] | 91.4% [85.9, 94.9] | 1.00 |

Statistical parity at this sample size (overlapping Wilson 95% intervals, no
significant McNemar), reached with no ingest-time LLM (mem0: ~1.5 h of
extraction calls for this conversation; trw-memory: ~75 s). Per-question
files live under `results/locomo/predicted_<project>/` in the
`memory-benchmarks` checkout; `compare.py A B` reproduces the table.
