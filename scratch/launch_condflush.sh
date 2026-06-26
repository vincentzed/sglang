#!/usr/bin/env bash
# VALUE: A/B the algorithmic conditioning-triggered EARLY FLUSH (lever #2) vs the fp16-RNE
# baseline. When a verify chunk is ill-conditioned (row-L1(A) > thresh), the verify kernel
# flags the slot and the cursor kernel forces an early flush -> less accumulation in the bad
# replay basis. Env: SGLANG_REPLAYSSM_COND_FLUSH=1, SGLANG_REPLAYSSM_COND_THRESH=<float>.
# Sweeps two thresholds (aggressive 1.5 / moderate 2.5). All fp16-RNE, CUDA graph ON.
set +e
cd /sgl-workspace/sglang
M=Qwen/Qwen3.6-35B-A3B
SPEC="--model-path $M --linear-attn-decode-backend triton --reasoning-parser qwen3 \
--attention-backend triton --page-size 1 \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --trust-remote-code"
RS="--enable-gdn-replayssm-spec --gdn-replayssm-spec-cache-len 16"
FP16="--mamba-ssm-dtype float16"
C="SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1"

# GPU4: recurrent exact reference
env $C CUDA_VISIBLE_DEVICES=4 setsid nohup sglang serve $SPEC --port 31010 > scratch/c_recur.log 2>&1 < /dev/null &
# GPU5: fp16 RNE baseline (no cond flush)
env $C CUDA_VISIBLE_DEVICES=5 setsid nohup sglang serve $SPEC $RS $FP16 --port 31011 > scratch/c_base.log 2>&1 < /dev/null &
# GPU6: fp16 RNE + cond flush, aggressive threshold 1.5
env $C SGLANG_REPLAYSSM_COND_FLUSH=1 SGLANG_REPLAYSSM_COND_THRESH=1.5 CUDA_VISIBLE_DEVICES=6 setsid nohup sglang serve $SPEC $RS $FP16 --port 31012 > scratch/c_cf15.log 2>&1 < /dev/null &
# GPU7: fp16 RNE + cond flush, moderate threshold 2.5
env $C SGLANG_REPLAYSSM_COND_FLUSH=1 SGLANG_REPLAYSSM_COND_THRESH=2.5 CUDA_VISIBLE_DEVICES=7 setsid nohup sglang serve $SPEC $RS $FP16 --port 31013 > scratch/c_cf25.log 2>&1 < /dev/null &
echo "launched recur=31010 base=31011 cf1.5=31012 cf2.5=31013 (GPUs 4-7, cond-flush A/B)"
