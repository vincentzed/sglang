#!/usr/bin/env bash
set -euo pipefail

cd /sgl-workspace/sglang

PROFILE_ROOT="jetspec/profiles/matrix_compact_tree_w8_b16"
SERVER_LOG="${PROFILE_ROOT}/nsys_server_31162.log"

if [ -f ~/.zshenv.local ]; then
  set -a
  source ~/.zshenv.local
  set +a
fi

export PYTHONPATH=python
export CUDA_VISIBLE_DEVICES=6
unset SGLANG_DFLASH_TREE_PAGED_FA4_VERIFY

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

python -m sglang.launch_server \
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
  --port 31162 \
  > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "${SERVER_PID}" 2>/dev/null || return 0
      sleep 1
    done
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for i in $(seq 1 1800); do
  if curl -fsS --max-time 2 http://127.0.0.1:31162/health >/dev/null 2>&1; then
    echo "nsys_server_healthy_after_s=${i}"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "nsys_server_exited_before_health"
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
  if [ "${i}" -eq 1800 ]; then
    echo "nsys_server_health_timeout"
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 1
done

for _ in $(seq 1 4); do
  curl -fsS http://127.0.0.1:31162/generate \
    -H 'Content-Type: application/json' \
    -d '{"text":"hello","sampling_params":{"max_new_tokens":256,"temperature":0}}' \
    >/dev/null
done

cleanup
