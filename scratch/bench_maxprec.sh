#!/usr/bin/env bash
# VALUE: the *perf* half of the comparison — fixed 512-in/512-out bench_serving
# (--random-range-ratio 1.0 => every request identical length => deterministic) across
# the 4 servers, showing fp16 is perf-equivalent to bf16 and fp32's extra checkpoint
# bandwidth is negligible on this MoE-heavy model. Requires launch_maxprec.sh up.
set +e
cd /sgl-workspace/sglang
IN=${IN:-512}; OUT=${OUT:-512}
declare -A PORT=( [recurrent]=31010 [bf16]=31011 [fp16]=31012 [fp32]=31013 )
for C in 16 32; do
  for label in recurrent bf16 fp16 fp32; do
    p=${PORT[$label]}
    echo "===== $label  concurrency=$C  in=$IN out=$OUT ====="
    python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $p \
      --dataset-name random --random-input-len $IN --random-output-len $OUT \
      --random-range-ratio 1.0 --num-prompts $((C*4)) --max-concurrency $C \
      2>&1 | grep -iE "Output token throughput|Median TPOT|Mean TPOT|Median E2E|Successful requests" | sed 's/^/  /'
  done
done
echo "===== PERF DONE ====="
