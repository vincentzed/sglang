#!/usr/bin/env bash
set -euo pipefail

cd /sgl-workspace/sglang

if [ -f "$HOME/.zshenv.local" ]; then
  source "$HOME/.zshenv.local"
fi
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=python

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "-$SERVER_PID" 2>/dev/null || kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$SERVER_PID" 2>/dev/null || return 0
      sleep 1
    done
    kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

setsid bash -lc '
  cd /sgl-workspace/sglang
  if [ -f "$HOME/.zshenv.local" ]; then source "$HOME/.zshenv.local"; fi
  if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi
  if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"; fi
  export CUDA_VISIBLE_DEVICES=4
  export PYTHONPATH=python
  exec python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path JetSpec/jetspec-qwen3-8b \
    --speculative-num-draft-tokens 16 \
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
    --port 31160
' &
SERVER_PID=$!
echo "nsys server pid=$SERVER_PID"

for i in $(seq 1 1800); do
  if curl -fsS http://127.0.0.1:31160/health >/dev/null 2>&1; then
    echo "nsys server healthy after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "nsys server exited before health" >&2
    exit 1
  fi
  if [ "$i" -eq 1800 ]; then
    echo "timed out waiting for nsys server health" >&2
    exit 1
  fi
  sleep 1
done

for i in $(seq 1 5); do
  curl -fsS \
    -H 'Content-Type: application/json' \
    -d '{"text":"hello","sampling_params":{"max_new_tokens":256,"temperature":0}}' \
    http://127.0.0.1:31160/generate >/dev/null
  echo "generate request ${i} complete"
done
