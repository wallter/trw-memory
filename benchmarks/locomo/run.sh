#!/usr/bin/env bash
# Run mem0's LOCOMO benchmark against one backend through the local shim.
#
#   run.sh <mem0|trw> <project-name> [extra runner args...]
#
# Env (all optional):
#   BENCH_DIR        memory-benchmarks checkout   (default: ~/.cache/trw-bench/memory-benchmarks,
#                    outside any worktree so results survive worktree cleanup)
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
BENCH_DIR="${BENCH_DIR:-$HOME/.cache/trw-bench/memory-benchmarks}"
BENCH_PY="${BENCH_PY:-python}"
LLM_MODEL="${LLM_MODEL:-llama3.1:latest}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
BENCH_PORT="${BENCH_PORT:-8888}"
export BENCH_BACKEND="$BACKEND" BENCH_PORT
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$BENCH_DIR/bench-state/$BACKEND}"
export BENCH_LLM_MODEL="$LLM_MODEL" OLLAMA_BASE_URL
# Run against THIS checkout of trw-memory, not whatever editable install the venv points at.
export PYTHONPATH="$HERE/../../src${PYTHONPATH:+:$PYTHONPATH}"

[ -n "${BENCH_SKIP_BOOTSTRAP:-}" ] || "$HERE/bootstrap.sh" "$BENCH_DIR" >/dev/null

# Never benchmark a stale shim: the port must be free, and /health must come from the child we start.
if curl -sf "http://127.0.0.1:$BENCH_PORT/health" >/dev/null 2>&1; then
  echo "port $BENCH_PORT already serves a shim; stop it or pick another BENCH_PORT"; exit 1
fi
mkdir -p "$BENCH_DIR/logs" "$BENCH_DATA_DIR"
SHIM_LOG="$BENCH_DIR/logs/shim-$BACKEND-$PROJECT-$BENCH_PORT.log"
"$BENCH_PY" "$HERE/server.py" >>"$SHIM_LOG" 2>&1 &
SHIM=$!
trap 'kill $SHIM 2>/dev/null || true' EXIT
DATA_REAL="$(cd "$BENCH_DATA_DIR" && pwd -P)"
for _ in $(seq 1 180); do
  kill -0 "$SHIM" 2>/dev/null || { echo "shim exited during startup; see $SHIM_LOG"; exit 1; }
  health="$(curl -sf "http://127.0.0.1:$BENCH_PORT/health" 2>/dev/null || true)"
  [ -n "$health" ] && break
  sleep 1
done
"$BENCH_PY" - "$health" "$SHIM" "$BACKEND" "$DATA_REAL" <<'EOF' || exit 1
import json, os, sys
raw, pid, backend, data_dir = sys.argv[1:]
h = json.loads(raw or "{}")
want = {"pid": int(pid), "backend": backend, "data_dir": os.path.realpath(data_dir)}
got = {"pid": h.get("pid"), "backend": h.get("backend"), "data_dir": os.path.realpath(h.get("data_dir") or "")}
if got != want:
    sys.exit(f"shim identity mismatch: want {want}, got {got}")
print(f"shim ok: {backend} pid {pid} embed={h.get('embed_model')} llm={h.get('llm_model')} data={data_dir}")
EOF

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
