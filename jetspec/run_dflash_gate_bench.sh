#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-jetspec/runs/dflash_gate_bench_${STAMP}}"
LOG_DIR="${LOG_DIR:-jetspec/logs/dflash_gate_bench_${STAMP}}"
PROFILE_DIR="${PROFILE_DIR:-jetspec/profiles/dflash_gate_bench_${STAMP}}"
PORT_BASE="${PORT_BASE:-31950}"
GATE_SAMPLES="${GATE_SAMPLES:-5}"
BENCH_SAMPLES="${BENCH_SAMPLES:-80}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-900}"
MODE="${1:-all}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-JetSpec/jetspec-qwen3-8b}"
HOST="${HOST:-0.0.0.0}"
GPU="${CUDA_VISIBLE_DEVICES:-7}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
PROFILE_MEM_FRACTION_STATIC="${PROFILE_MEM_FRACTION_STATIC:-0.35}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$PROFILE_DIR"

source ~/.zshenv.local >/dev/null 2>&1 || true
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export PYTHONPATH=python

RUNNING_PID=""

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

cleanup() {
  if [[ -n "${RUNNING_PID:-}" ]] && kill -0 "$RUNNING_PID" >/dev/null 2>&1; then
    log "stopping server pid=$RUNNING_PID"
    kill -INT "-$RUNNING_PID" >/dev/null 2>&1 || kill -INT "$RUNNING_PID" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$RUNNING_PID" >/dev/null 2>&1; then
        RUNNING_PID=""
        return
      fi
      sleep 1
    done
    kill -KILL "-$RUNNING_PID" >/dev/null 2>&1 || kill -KILL "$RUNNING_PID" >/dev/null 2>&1 || true
    RUNNING_PID=""
  fi
}
trap cleanup EXIT INT TERM

wait_health() {
  local port="$1"
  local log_path="$2"
  local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
  local url="http://127.0.0.1:${port}/health"
  local last_error=""
  while (( SECONDS < deadline )); do
    if ! kill -0 "$RUNNING_PID" >/dev/null 2>&1; then
      log "server died before health; log tail follows"
      tail -200 "$log_path" >&2 || true
      return 1
    fi
    if python - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

urllib.request.urlopen(sys.argv[1], timeout=2).read()
PY
    then
      log "healthy port=$port"
      return 0
    fi
    sleep 2
  done
  log "timed out waiting for ${url}; log tail follows"
  tail -200 "$log_path" >&2 || true
  return 1
}

launch_linear() {
  local port="$1"
  local log_path="$2"
  log "launch linear width=1 port=$port log=$log_path"
  python -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --dtype bfloat16 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-tree-width 1 \
    --reasoning-parser qwen3 \
    --attention-backend fa4 \
    --page-size 16 \
    --tp-size 1 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --trust-remote-code \
    --max-running-requests 1 \
    --cuda-graph-max-bs-decode 1 \
    --cuda-graph-backend-decode full \
    --host "$HOST" \
    --port "$port" \
    >"$log_path" 2>&1 &
  RUNNING_PID="$!"
  wait_health "$port" "$log_path"
}

launch_tree() {
  local port="$1"
  local log_path="$2"
  local width="$3"
  local budget="$4"
  local beta="$5"
  local g0="$6"
  local mem_fraction="${7:-$MEM_FRACTION_STATIC}"
  log "launch top2gap tree w${width}/b${budget} beta=${beta} g0=${g0} port=$port log=$log_path"
  python -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --dtype bfloat16 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-tree-width "$width" \
    --speculative-dflash-tree-budget "$budget" \
    --speculative-dflash-tree-draft top2gap \
    --speculative-dflash-top2gap-beta "$beta" \
    --speculative-dflash-top2gap-g0 "$g0" \
    --reasoning-parser qwen3 \
    --attention-backend fa4 \
    --page-size 16 \
    --tp-size 1 \
    --mem-fraction-static "$mem_fraction" \
    --trust-remote-code \
    --max-running-requests 1 \
    --cuda-graph-max-bs-decode 1 \
    --cuda-graph-backend-decode full \
    --host "$HOST" \
    --port "$port" \
    >"$log_path" 2>&1 &
  RUNNING_PID="$!"
  wait_health "$port" "$log_path"
}

run_bench() {
  local port="$1"
  local dataset="$2"
  local mode="$3"
  local out_path="$4"
  local run_name="$5"
  local samples="$6"
  shift 6
  log "bench dataset=$dataset mode=$mode samples=$samples out=$out_path"
  python jetspec/bench_paper_sglang.py \
    --base-url "http://127.0.0.1:${port}" \
    --out "$out_path" \
    --run-name "$run_name" \
    --mode "$mode" \
    --dataset "$dataset" \
    --num-samples "$samples" \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --flush-cache-before-run \
    --health-timeout-s "$HEALTH_TIMEOUT_S" \
    --request-timeout-s "$REQUEST_TIMEOUT_S" \
    "$@"
}

summarize() {
  python - "$OUT_DIR/summary.ndjson" "$@" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
rows = []
for raw in sys.argv[2:]:
    path = Path(raw)
    data = json.loads(path.read_text())
    summary = data["summary"]
    steps = summary["total_spec_verify_ct"] or (
        summary["total_completion_tokens"] / summary["accept_length"]
    )
    wall_s = summary["total_wall_latency_s"]
    loss = data.get("losslessness") or {}
    nodes = data.get("tree_node_stats") or {}
    rows.append(
        {
            "accept_len": summary["accept_length"],
            "dataset": data["dataset"],
            "exact": loss.get("token_exact"),
            "mean_tree_nodes": nodes.get("mean_num_nodes"),
            "mismatches": len(loss.get("mismatches") or []),
            "mode": data["mode"],
            "ms_per_step": wall_s / steps * 1000.0,
            "path": str(path),
            "run_name": data["run_name"],
            "steps_per_s": steps / wall_s,
            "tok_s": summary["throughput_wall_tok_s"],
        }
    )
summary_path.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
)
for row in rows:
    print(json.dumps(row, sort_keys=True))
PY
}

run_gate_and_bench() {
  local linear_port=$((PORT_BASE))
  local tree_port=$((PORT_BASE + 1))
  local width=8
  local budget=16
  local beta="$1"
  local g0="$2"
  local slug="top2gap_w${width}_b${budget}_beta${beta//./p}_g0${g0//./p}"

  local gsm_oracle_gate="$OUT_DIR/gate_gsm8k_linear_w1_fa4p16.json"
  local math_oracle_gate="$OUT_DIR/gate_math500_linear_w1_fa4p16.json"
  local gsm_linear_full="$OUT_DIR/bench_gsm8k_linear_w1_fa4p16.json"
  local math_linear_full="$OUT_DIR/bench_math500_linear_w1_fa4p16.json"

  if [[ "$MODE" == "dry-run" ]]; then
    log "dry-run output dir: $OUT_DIR"
    log "linear port: $linear_port; tree port: $tree_port; slug: $slug"
    return
  fi

  launch_linear "$linear_port" "$LOG_DIR/linear_w1_fa4p16_${linear_port}.log"
  run_bench "$linear_port" gsm8k linear "$gsm_oracle_gate" gate-gsm8k-linear-w1-fa4p16 "$GATE_SAMPLES" \
    --tree-width 1 --tree-budget 16 --flush-cache-between-prompts
  run_bench "$linear_port" math500 linear "$math_oracle_gate" gate-math500-linear-w1-fa4p16 "$GATE_SAMPLES" \
    --tree-width 1 --tree-budget 16 --flush-cache-between-prompts
  run_bench "$linear_port" gsm8k linear "$gsm_linear_full" bench-gsm8k-linear-w1-fa4p16 "$BENCH_SAMPLES" \
    --tree-width 1 --tree-budget 16
  run_bench "$linear_port" math500 linear "$math_linear_full" bench-math500-linear-w1-fa4p16 "$BENCH_SAMPLES" \
    --tree-width 1 --tree-budget 16
  cleanup

  launch_tree "$tree_port" "$LOG_DIR/${slug}_${tree_port}.log" "$width" "$budget" "$beta" "$g0"
  local gsm_tree_gate="$OUT_DIR/gate_gsm8k_${slug}.json"
  local math_tree_gate="$OUT_DIR/gate_math500_${slug}.json"
  local gsm_tree_full="$OUT_DIR/bench_gsm8k_${slug}.json"
  local math_tree_full="$OUT_DIR/bench_math500_${slug}.json"
  run_bench "$tree_port" gsm8k tree "$gsm_tree_gate" "gate-gsm8k-${slug}" "$GATE_SAMPLES" \
    --tree-width "$width" --tree-budget "$budget" --tree-draft top2gap \
    --top2gap-beta "$beta" --top2gap-g0 "$g0" \
    --compare-to "$gsm_oracle_gate" --flush-cache-between-prompts
  run_bench "$tree_port" math500 tree "$math_tree_gate" "gate-math500-${slug}" "$GATE_SAMPLES" \
    --tree-width "$width" --tree-budget "$budget" --tree-draft top2gap \
    --top2gap-beta "$beta" --top2gap-g0 "$g0" \
    --compare-to "$math_oracle_gate" --flush-cache-between-prompts
  run_bench "$tree_port" gsm8k tree "$gsm_tree_full" "bench-gsm8k-${slug}" "$BENCH_SAMPLES" \
    --tree-width "$width" --tree-budget "$budget" --tree-draft top2gap \
    --top2gap-beta "$beta" --top2gap-g0 "$g0" \
    --compare-to "$gsm_linear_full"
  run_bench "$tree_port" math500 tree "$math_tree_full" "bench-math500-${slug}" "$BENCH_SAMPLES" \
    --tree-width "$width" --tree-budget "$budget" --tree-draft top2gap \
    --top2gap-beta "$beta" --top2gap-g0 "$g0" \
    --compare-to "$math_linear_full"
  cleanup

  summarize \
    "$gsm_oracle_gate" "$math_oracle_gate" "$gsm_linear_full" "$math_linear_full" \
    "$gsm_tree_gate" "$math_tree_gate" "$gsm_tree_full" "$math_tree_full"
}

run_profile() {
  local port=$((PORT_BASE + 2))
  if [[ "$MODE" == "dry-run" ]]; then
    log "profile command will use port=$port profile_dir=$PROFILE_DIR"
    return
  fi
  launch_tree "$port" "$LOG_DIR/profile_top2gap_w8_b16_${port}.log" 8 16 1.0 1.0 "$PROFILE_MEM_FRACTION_STATIC"
  python3 /root/.claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
    --framework sglang \
    --url "http://127.0.0.1:${port}" \
    --output-dir "$PROFILE_DIR/top2gap_w8_b16_decode" \
    --profile-workload decode \
    --warmup-steps 10 \
    --num-steps 5 \
    --profile-by-stage \
    >"$PROFILE_DIR/top2gap_w8_b16_decode_analysis.txt"
  cleanup
}

case "$MODE" in
  all)
    run_gate_and_bench 1.0 1.0
    ;;
  lean-b16-pair)
    run_gate_and_bench 1.0 1.0
    run_gate_and_bench 2.0 0.5
    ;;
  profile)
    run_profile
    ;;
  dry-run)
    run_gate_and_bench 1.0 1.0
    run_profile
    ;;
  *)
    printf 'usage: %s [all|lean-b16-pair|profile|dry-run]\n' "$0" >&2
    exit 2
    ;;
esac
