#!/usr/bin/env bash
set -euo pipefail

cd /sgl-workspace/sglang

[ -f ~/.zshenv.local ] && source ~/.zshenv.local || true
export PYTHONPATH=python
export CUDA_VISIBLE_DEVICES=7
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
export SGLANG_DFLASH_TREE_PAGED_FA4_VERIFY=1

PROFILE_DIR=jetspec/profiles/matrix_paged_tree_w8_b16
SERVER_LOG="$PROFILE_DIR/nsys_server_31163.log"
: > "$SERVER_LOG"

setsid bash -lc '
  set -euo pipefail
  cd /sgl-workspace/sglang
  [ -f ~/.zshenv.local ] && source ~/.zshenv.local || true
  export PYTHONPATH=python
  export CUDA_VISIBLE_DEVICES=7
  if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi
  if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"; fi
  export SGLANG_DFLASH_TREE_PAGED_FA4_VERIFY=1
  exec python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path JetSpec/jetspec-qwen3-8b \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-tree-width 8 \
    --speculative-dflash-tree-budget 16 \
    --speculative-dflash-tree-draft top2gap \
    --speculative-dflash-top2gap-beta 1.0 \
    --speculative-dflash-top2gap-g0 1.0 \
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
    --port 31163
' >> "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PROFILE_DIR/nsys_server_31163.pid"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "-$SERVER_PID" 2>/dev/null || kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$SERVER_PID" 2>/dev/null || return 0
      sleep 1
    done
    kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for i in $(seq 1 420); do
  if curl -fsS http://127.0.0.1:31163/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "nsys server exited before health"
    tail -200 "$SERVER_LOG" || true
    exit 1
  fi
  if [ "$i" -eq 420 ]; then
    echo "timed out waiting for nsys server health"
    tail -200 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 1
done

for i in 1 2 3; do
  curl -fsS \
    -H 'Content-Type: application/json' \
    -d '{"text":"hello","sampling_params":{"max_new_tokens":256,"temperature":0}}' \
    http://127.0.0.1:31163/generate \
    > "$PROFILE_DIR/nsys_generate_${i}.json"
done
