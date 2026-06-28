#!/usr/bin/env bash
# run-local.sh — run LLMViz locally with REAL white-box models (GPT-2 family) enabled.
# This is what makes the "Inside a block" view fully live: genuine Q/K/V + attention that
# change every step. Without LLMVIZ_LIVE the app falls back to the scripted DEMO (static-looking).
#
# Usage:  ./run-local.sh            # NANO + MICRO live (fit comfortably on a Mac)
#         LLMVIZ_LIVE_TIERS=nano,micro,small ./run-local.sh   # also run gpt2-medium (SMALL)
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || { echo "no venv at $PY — create it and: $PY -m pip install -r requirements.txt torch transformers"; exit 1; }

export LLMVIZ_LIVE=1
export LLMVIZ_LIVE_TIERS="${LLMVIZ_LIVE_TIERS:-nano,micro}"
PORT="${PORT:-8810}"

echo "LLMViz · live tiers: $LLMVIZ_LIVE_TIERS · http://127.0.0.1:$PORT"
exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
