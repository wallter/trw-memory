#!/usr/bin/env bash
# Run mem0's LOCOMO benchmark against one backend through the local shim.
#
#   run.sh <mem0|trw> <project-name> [extra runner args...]
#
# Env (all optional):
#   BENCH_DIR        memory-benchmarks checkout   (default: ../../../scratch/memory-benchmarks)
#   BENCH_PY         python interpreter           (default: python)
#   LLM_MODEL        answerer + judge model        (default: llama3.1:latest)
#   ANSWERER_MODEL / JUDGE_MODEL   override either role (default: LLM_MODEL)
#   LLM_BASE_URL / LLM_API_KEY     hosted OpenAI-compatible endpoint for answerer + judge (default: local Ollama)
#   BENCH_EXTRACT_BASE_URL / BENCH_EXTRACT_API_KEY  hosted endpoint for mem0's extraction LLM (model: LLM_MODEL)
#   OLLAMA_BASE_URL  (default: http://localhost:11434)
#   BENCH_PORT       shim port                    (default: 8888)
#   BENCH_DATA_DIR   backend state dir            (default: <BENCH_DIR>/bench-state/<backend>)
#
# The upstream runner is invoked UNMODIFIED (see bootstrap.sh for the single
# client-side patch). Results land in <BENCH_DIR>/results/locomo/.
set -euo pipefail
BACKEND="${1:?mem0|trw}"; shift
PROJECT="${1:?project name}"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${BENCH_DIR:-$HERE/../../../scratch/memory-benchmarks}"
BENCH_PY="${BENCH_PY:-python}"
LLM_MODEL="${LLM_MODEL:-llama3.1:latest}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
BENCH_PORT="${BENCH_PORT:-8888}"
export BENCH_BACKEND="$BACKEND" BENCH_PORT
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$BENCH_DIR/bench-state/$BACKEND}"
export BENCH_LLM_MODEL="$LLM_MODEL" OLLAMA_BASE_URL
# Run against THIS checkout of trw-memory, not whatever editable install the venv points at.
export PYTHONPATH="$HERE/../../src${PYTHONPATH:+:$PYTHONPATH}"

"$HERE/bootstrap.sh" "$BENCH_DIR" >/dev/null

mkdir -p "$BENCH_DIR/logs"
"$BENCH_PY" "$HERE/server.py" >"$BENCH_DIR/logs/shim-$BACKEND-$PROJECT.log" 2>&1 &
SHIM=$!
trap 'kill $SHIM 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$BENCH_PORT/health" >/dev/null && break
  sleep 1
done
curl -sf "http://127.0.0.1:$BENCH_PORT/health" >/dev/null || { echo "shim failed to start; see logs/shim-$BACKEND-$PROJECT.log"; exit 1; }

cd "$BENCH_DIR"
# Answerer + judge endpoint. Default: local Ollama. For a hosted OpenAI-compatible
# provider (OpenRouter, OpenAI, vLLM, ...) set LLM_BASE_URL + LLM_API_KEY, and
# optionally ANSWERER_MODEL / JUDGE_MODEL (both default to LLM_MODEL), e.g.
#   LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_API_KEY=$OPENROUTER_API_KEY \
#   ANSWERER_MODEL=meta-llama/llama-3.3-70b-instruct JUDGE_MODEL=openai/gpt-4o ./run.sh ...
if [ -n "${LLM_BASE_URL:-}" ]; then
  export OPENAI_BASE_URL="$LLM_BASE_URL" OPENAI_API_KEY="${LLM_API_KEY:?LLM_API_KEY is required with LLM_BASE_URL}"
else
  export OPENAI_BASE_URL="$OLLAMA_BASE_URL/v1" OPENAI_API_KEY="ollama"
  # Ollama only: make thinking models answer directly.
  if [ -z "${LLM_EXTRA_BODY:-}" ]; then export LLM_EXTRA_BODY='{"reasoning_effort":"none"}'; fi
fi
export MEM0_HOST="http://127.0.0.1:$BENCH_PORT" MEM0_BACKEND=oss
"$BENCH_PY" -m benchmarks.locomo.run \
  --project-name "$PROJECT" \
  --answerer-model "${ANSWERER_MODEL:-$LLM_MODEL}" --judge-model "${JUDGE_MODEL:-$LLM_MODEL}" --provider openai \
  --mem0-host "$MEM0_HOST" \
  "$@"
curl -s "http://127.0.0.1:$BENCH_PORT/health"; echo
