#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

set -a
if [ -f ~/.zshenv.local ]; then
  source ~/.zshenv.local
fi
set +a

export PYTHONPATH=python
export CUDA_VISIBLE_DEVICES=5
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

LOG_DIR="jetspec/profiles/matrix_linear_jetspec_w1"
SERVER_LOG="$LOG_DIR/nsys_server_31161.log"
PID_FILE="$LOG_DIR/nsys_server_31161.pid"

python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path JetSpec/jetspec-qwen3-8b \
  --speculative-num-draft-tokens 16 \
  --speculative-dflash-tree-width 1 \
  --reasoning-parser qwen3 \
  --attention-backend fa4 \
  --page-size 16 \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --trust-remote-code \
  --max-running-requests 1 \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-backend-decode full \
  --host 0.0.0.0 \
  --port 31161 \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 360); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "nsys server exited before health"
    tail -n 160 "$SERVER_LOG" || true
    exit 1
  fi
  if curl -fsS http://127.0.0.1:31161/health >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

curl -fsS http://127.0.0.1:31161/health >/dev/null

for _ in $(seq 1 3); do
  curl -fsS http://127.0.0.1:31161/generate \
    -H 'Content-Type: application/json' \
    -d '{"text":"hello","sampling_params":{"max_new_tokens":256,"temperature":0}}' \
    >/dev/null
done
