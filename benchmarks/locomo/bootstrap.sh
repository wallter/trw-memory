#!/usr/bin/env bash
# Fetch mem0's memory-benchmarks suite at a pinned commit and apply the one
# TRW patch (LLM_EXTRA_BODY env passthrough so Ollama reasoning models can be
# told not to think). Idempotent. Usage: bootstrap.sh <dest-dir>
set -euo pipefail
DEST="${1:?dest dir}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="4b61c5d31b9c668a12b4f5e78064248a02c82d2b"   # memory-benchmarks main, 2026-09-16
if [ ! -d "$DEST/.git" ]; then
  git clone --quiet https://github.com/mem0ai/memory-benchmarks.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$PIN" || true
git -C "$DEST" checkout --quiet "$PIN"
if ! grep -q "_openai_extra_body" "$DEST/benchmarks/common/llm_client.py"; then
  git -C "$DEST" apply "$HERE/upstream-llm_client.patch"
fi
mkdir -p "$DEST/datasets/locomo"
if [ ! -s "$DEST/datasets/locomo/locomo10.json" ]; then
  curl -sSL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
    -o "$DEST/datasets/locomo/locomo10.json"
fi
echo "memory-benchmarks ready at $DEST (pinned $PIN)"
