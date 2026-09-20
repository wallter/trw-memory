#!/usr/bin/env bash
# One entry point per phase of the hosted-judge LOCOMO benchmark plan (trw-memory vs mem0).
#
#   phases.sh preflight                                             # no spend
#   phases.sh predict-trw  <phase> <run> <embed> <port>             # local, no LLM; blocks until done
#   phases.sh predict-mem0 <phase> <run> <parallel> <base-port> [--fresh] <conv>...   # blocks until done
#   phases.sh verify <pred-dir> <conv>...                           # every expected prediction present, 0 failed chunks
#   phases.sh judge <phase> <pred-dir> <tag> <model> <cutoff> [batch_judge args...]
#   phases.sh stock-check <phase> <pred-dir> <n-per-conv>           # the unmodified evaluator, for P2
#   phases.sh mark <phase> <label>                                  # key-usage snapshot into the ledger (C6)
#   phases.sh sample <n> <seed> <out-file>                          # frozen ids, stratified conversation x category
#   phases.sh compare <dirA> <dirB> <labelA> <labelB> <cutoff> [compare.py args]   # default --expected 1540
#
# Paid or long commands refuse to start unless preflight passed in the last 24 h with the pinned
# dataset hash, no STOP file exists, and `watch.py --phase <phase>` has a heartbeat under 3 min old.
# Predict commands run their jobs as process-group leaders (set -m), record them in
# ~/.cache/trw-bench/<phase>.pids for watch.py, wait for them, remove them, and exit non-zero if any
# job failed or `verify` finds a gap. Run them in the background from the harness; they block.
#
# Env: PY (python), OPENROUTER_API_KEY or TRW_BENCH_ENV_FILE (default: the main checkout's .env),
#      MEM0_EXTRACT_MODEL (default openai/gpt-5-mini = mem0 2.0.20's default), MEM0_EXTRACT_REASONING.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HOME/.cache/trw-bench"
export BENCH_DIR="$ROOT/memory-benchmarks"
PY="${PY:-python}"
export TRW_BENCH_ENV_FILE="${TRW_BENCH_ENV_FILE:-$HERE/../../../.env}"
MEM0_EXTRACT_MODEL="${MEM0_EXTRACT_MODEL:-openai/gpt-5-mini}"
DATASET_SHA="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"  # locomo10.json, 2026-09-19
PREDS="$BENCH_DIR/results/locomo"
mkdir -p "$ROOT/state"

key() {
  if [ -n "${OPENROUTER_API_KEY:-}" ]; then echo "$OPENROUTER_API_KEY"; return; fi
  sed -n 's/^[[:space:]]*OPENROUTER_API_KEY[[:space:]]*=[[:space:]]*["'\'']*\([^"'\'' ]*\).*/\1/p' "$TRW_BENCH_ENV_FILE" | head -1
}
die() { echo "phases.sh: $*" >&2; exit 1; }
age() { echo $(( $(date +%s) - $(stat -f %m "$1" 2>/dev/null || echo 0) )); }

guard() {  # the preconditions of every paid or long-running command
  local phase="$1"
  [ ! -e "$ROOT/STOP" ] || die "STOP file present: $(cat "$ROOT/STOP")"
  [ -f "$ROOT/preflight.ok" ] && [ "$(age "$ROOT/preflight.ok")" -lt 86400 ] || die "run 'phases.sh preflight' (valid 24 h)"
  grep -q "$DATASET_SHA" "$ROOT/preflight.ok" || die "preflight recorded a different dataset hash"
  local hb="$ROOT/watch.$phase.heartbeat"
  [ -f "$hb" ] && [ "$(age "$hb")" -lt 180 ] || die "no live watcher for phase $phase: start watch.py --phase $phase first"
}

# Job control: each background job is its own process group, so watch.py can stop exactly it.
PIDS=(); NAMES=()
start_job() {  # start_job <phase> <name> <log> <cmd...>
  local phase="$1" name="$2" log="$3"; shift 3
  set -m
  "$@" >>"$log" 2>&1 &
  local pid=$!
  set +m
  echo "$pid" >> "$ROOT/$phase.pids"
  PIDS+=("$pid"); NAMES+=("$name")
  echo "started $name pid $pid (log $log)"
}
# ${A[@]+"${A[@]}"}: bash 3.2 (macOS /bin/bash) treats an empty array as unbound under set -u.
running() { local n=0 p; for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo "$n"; }
finish_jobs() {  # finish_jobs <phase>: wait for every job, un-record it, fail if any failed
  local phase="$1" i failed=0
  for i in ${PIDS[@]+"${!PIDS[@]}"}; do
    if wait "${PIDS[$i]}"; then echo "done ${NAMES[$i]}"; else echo "FAILED ${NAMES[$i]} (exit $?)"; failed=1; fi
    sed -i '' "/^${PIDS[$i]}\$/d" "$ROOT/$phase.pids" 2>/dev/null || true
  done
  PIDS=(); NAMES=()
  return "$failed"
}

verify() {  # verify <pred-dir> <conv>...
  local pred="$1"; shift
  "$PY" - "$BENCH_DIR" "$PREDS/$pred" "$@" <<'EOF'
import json, os, sys
bench, pred, convs = sys.argv[1], sys.argv[2], [int(c) for c in sys.argv[3:]]
sys.path.insert(0, bench)
from benchmarks.locomo import prompts, run
items = run.expected_locomo_question_items(run.load_dataset(f"{bench}/datasets/locomo/locomo10.json"),
                                           convs, prompts.CATEGORIES_TO_EVALUATE, None)
ok, missing = run.locomo_predict_outputs_complete(pred, items)
problems = [f"{len(missing)} predictions missing, e.g. {missing[:5]}"] if not ok else []
for c in convs:
    path = os.path.join(pred, f"_ingestion_{c}.json")
    if not os.path.exists(path):
        problems.append(f"conversation {c}: no ingestion checkpoint")
    elif json.load(open(path)).get("total_chunks_failed", 0):
        problems.append(f"conversation {c}: {json.load(open(path))['total_chunks_failed']} failed chunks")
    if os.path.exists(os.path.join(pred, f"_progress_{c}.json")):
        problems.append(f"conversation {c}: partial-ingest checkpoint still present")
print(f"verify {os.path.basename(pred)}: {len(items)} expected, " + ("OK" if not problems else "; ".join(problems)))
sys.exit(1 if problems else 0)
EOF
}

cmd="${1:?command}"; shift
case "$cmd" in
  preflight)
    rm -f "$ROOT/preflight.ok"
    [ -n "$(key)" ] || die "no OPENROUTER_API_KEY in the environment or $TRW_BENCH_ENV_FILE"
    [ ! -e "$ROOT/STOP" ] || die "STOP file present: $(cat "$ROOT/STOP")"
    "$HERE/bootstrap.sh" "$BENCH_DIR"
    sha=$(shasum -a 256 "$BENCH_DIR/datasets/locomo/locomo10.json" | cut -d' ' -f1)
    [ "$sha" = "$DATASET_SHA" ] || die "dataset hash $sha != pinned $DATASET_SHA"
    [ ! -e "$BENCH_DIR/.env" ] || die "$BENCH_DIR/.env exists; the runner loads it with override=True"
    if env | grep -q '^MEMORY_'; then echo "note: inherited MEMORY_* variables (the shim overrides storage/embedder): $(env | grep '^MEMORY_' | cut -d= -f1 | tr '\n' ' ')"; fi
    {
      echo "dataset_sha $sha"
      echo "harness $(git -C "$BENCH_DIR" rev-parse HEAD)"
      echo "commit $(git -C "$HERE" rev-parse HEAD)"
      "$PY" -c "import importlib.metadata as m; print('mem0ai', m.version('mem0ai'))"
      PYTHONPATH="$HERE/../../src" "$PY" -c "import trw_memory; print('trw-memory source', trw_memory.__file__)"
      grep -m1 '^version' "$HERE/../../pyproject.toml" | sed 's/^/trw-memory pyproject /'
      OPENROUTER_API_KEY="$(key)" "$PY" -c "
import sys; sys.path.insert(0, '$HERE'); import batch_judge as b
i = b.key_info() or sys.exit('key lookup failed')
print(f\"key limit {i.get('limit')} remaining {i.get('limit_remaining')} used {i.get('usage')}\")"
    } | tee "$ROOT/preflight.tmp"
    mv "$ROOT/preflight.tmp" "$ROOT/preflight.ok"
    ;;

  predict-trw)
    phase="${1:?phase}"; run="${2:?run}"; embed="${3:?embed model}"; port="${4:?port}"
    guard "$phase"
    [ ! -e "$ROOT/state/$run" ] || die "store $ROOT/state/$run exists; remove it for a fresh run"
    BENCH_PY="$PY" BENCH_PORT="$port" BENCH_EMBED_MODEL="$embed" BENCH_DATA_DIR="$ROOT/state/$run" \
      BENCH_SKIP_BOOTSTRAP=1 start_job "$phase" "$run" "$ROOT/logs-$run.txt" \
      "$HERE/run.sh" trw "$run" --run-id "$run" --top-k 50 --predict-only --max-workers 1
    finish_jobs "$phase"
    verify "predicted_$run" 0 1 2 3 4 5 6 7 8 9
    ;;

  predict-mem0)
    phase="${1:?phase}"; run="${2:?run}"; parallel="${3:?parallel}"; base="${4:?base port}"; shift 4
    fresh=""; if [ "${1:-}" = "--fresh" ]; then fresh=1; shift; fi
    [ $# -gt 0 ] || die "no conversations given"
    guard "$phase"
    reasoning="${MEM0_EXTRACT_REASONING:-$([[ "$MEM0_EXTRACT_MODEL" == *gpt-5* ]] && echo 1 || echo 0)}"
    for conv in "$@"; do  # refuse or clean before spending anything
      store="$ROOT/state/$run-c$conv"
      if [ -e "$store" ] || [ -e "$PREDS/predicted_$run/_progress_$conv.json" ] || [ -e "$PREDS/predicted_$run/_ingestion_$conv.json" ]; then
        [ -n "$fresh" ] || die "conversation $conv has a store or checkpoint; pass --fresh to rebuild it from scratch"
        rm -rf "$store"
        rm -f "$PREDS/predicted_$run/_ingestion_$conv.json" "$PREDS/predicted_$run/_progress_$conv.json"
        rm -f "$PREDS/predicted_$run"/conv"$conv"_q*.json
      fi
    done
    for conv in "$@"; do
      while [ "$(running)" -ge "$parallel" ]; do sleep 15; done
      [ ! -e "$ROOT/STOP" ] || { echo "STOP file present; not starting conversation $conv"; break; }
      BENCH_PY="$PY" BENCH_PORT=$((base + conv)) BENCH_DATA_DIR="$ROOT/state/$run-c$conv" BENCH_SKIP_BOOTSTRAP=1 \
        LLM_MODEL="$MEM0_EXTRACT_MODEL" BENCH_EXTRACT_BASE_URL=https://openrouter.ai/api/v1 \
        BENCH_EXTRACT_API_KEY="$(key)" BENCH_EXTRACT_REASONING="$reasoning" \
        start_job "$phase" "$run-c$conv" "$ROOT/logs-$run-c$conv.txt" \
        "$HERE/run.sh" mem0 "$run" --run-id "$run" --conversations "$conv" --top-k 50 --predict-only --max-workers 1
    done
    status=0; finish_jobs "$phase" || status=1
    verify "predicted_$run" "$@" || status=1
    exit "$status"
    ;;

  verify)
    verify "${1:?pred dir}" "${@:2}"
    ;;

  judge)
    phase="${1:?phase}"; pred="${2:?pred dir}"; tag="${3:?tag}"; model="${4:?model}"; cutoff="${5:?cutoff}"; shift 5
    guard "$phase"
    OPENROUTER_API_KEY="$(key)" "$PY" "$HERE/batch_judge.py" --bench-dir "$BENCH_DIR" --pred "$pred" --tag "$tag" \
      --model "$model" --cutoff "$cutoff" --phase "$phase" "$@"
    ;;

  stock-check)  # P2: the unmodified evaluator answers + judges the first <n> questions per conversation
    phase="${1:?phase}"; pred="${2:?pred dir}"; n="${3:?questions per conversation}"
    guard "$phase"
    copy="${pred}-stock"
    [ ! -e "$PREDS/$copy" ] || die "$PREDS/$copy exists; remove it to re-run the stock check"
    cp -R "$PREDS/$pred" "$PREDS/$copy"   # --evaluate-only writes judgments into its predict dir
    # --evaluate-only never contacts a memory backend, so no shim: call the pinned evaluator directly,
    # as a managed job so watch.py can stop it (the stock evaluator does not read the STOP file).
    cd "$BENCH_DIR"
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY="$(key)" \
      start_job "$phase" stock-check "$ROOT/logs-stockcheck.txt" \
      "$PY" -m benchmarks.locomo.run --project-name "${copy#predicted_}" --provider openai \
        --answerer-model openai/gpt-4o-mini --judge-model openai/gpt-4o-mini \
        --evaluate-only --max-questions "$n" --top-k-cutoffs 10 --max-workers 4
    finish_jobs "$phase"
    cd "$HERE"
    grep -l '"cutoff_results"' "$PREDS/$copy"/conv*_q*.json | xargs -n1 basename | sed 's/\.json$//' \
      > "$ROOT/stockcheck-$pred.qids"
    echo "stock evaluator judged $(wc -l < "$ROOT/stockcheck-$pred.qids" | tr -d ' ') questions into $PREDS/$copy" \
         "(ids: $ROOT/stockcheck-$pred.qids)"
    ;;

  mark)
    phase="${1:?phase}"; label="${2:?label}"
    OPENROUTER_API_KEY="$(key)" "$PY" -c "
import sys; sys.path.insert(0, '$HERE'); import batch_judge as b
b.ledger({'phase': '$phase', 'event': 'mark', 'label': '$label', 'key': b.key_info()})
print('marked', '$phase', '$label')"
    ;;

  sample)
    n="${1:?n}"; seed="${2:?seed}"; out="${3:?out file}"
    "$PY" - "$BENCH_DIR" "$n" "$seed" "$out" <<'EOF'
import random, sys
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
from benchmarks.locomo import prompts, run
bench, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
items = run.expected_locomo_question_items(run.load_dataset(f"{bench}/datasets/locomo/locomo10.json"),
                                           list(range(10)), prompts.CATEGORIES_TO_EVALUATE, None)
strata = defaultdict(list)
for qid, conv, _, qa in items:
    strata[(conv, qa["category"])].append(qid)
rng = random.Random(seed)
keys = sorted(strata)
picked = []
for k in keys:  # proportional allocation, at least one per stratum
    take = max(1, round(n * len(strata[k]) / len(items)))
    picked += rng.sample(sorted(strata[k]), min(take, len(strata[k])))
picked = sorted(rng.sample(sorted(picked), n) if len(picked) > n else picked)
open(out, "w").write("\n".join(picked) + "\n")
print(f"{len(picked)} ids from {len(keys)} strata -> {out}")
EOF
    ;;

  compare)
    a="${1:?}"; b="${2:?}"; la="${3:?}"; lb="${4:?}"; cut="${5:?}"; shift 5
    extra=("$@"); [ ${#extra[@]} -gt 0 ] || extra=(--expected 1540)
    "$PY" "$HERE/compare.py" "$PREDS/$a" "$PREDS/$b" --label-a "$la" --label-b "$lb" --cutoff "$cut" \
      --json "$ROOT/compare-$a-vs-$b-$cut.json" "${extra[@]}"
    ;;

  *) die "unknown command $cmd" ;;
esac
