#!/usr/bin/env bash
# VALUE: brings up the exact 4 servers (recurrent / bf16 / fp16 / fp32 checkpoint) that
# rigorous_suite.py and bench_maxprec.sh target on ports 31010-31013. Run this FIRST.
# recurrent = replayssm flag OFF (exact verify); bf16 = current PR; fp16/fp32 exercise
# the h0.dtype kernel patch via --mamba-ssm-dtype {float16,float32}.
# baseline(bf16) vs fp16 vs max-precision(fp32 storage + TF32 dots), + recurrent ref.
# GPUs 4-7. Uses the kernel patch (h0.dtype reconstruction).
set +e
cd /sgl-workspace/sglang
M=Qwen/Qwen3.6-35B-A3B
SPEC="--model-path $M --linear-attn-decode-backend triton --reasoning-parser qwen3 \
--attention-backend triton --page-size 1 \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --trust-remote-code"
RS="--enable-gdn-replayssm-spec --gdn-replayssm-spec-cache-len 16"

# GPU4: recurrent (exact reference)
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=4 setsid nohup sglang serve $SPEC --port 31010 > scratch/m_recur.log 2>&1 < /dev/null &
# GPU5: replayssm bf16 (BASELINE / current PR)
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=5 setsid nohup sglang serve $SPEC $RS --port 31011 > scratch/m_bf16.log 2>&1 < /dev/null &
# GPU6: replayssm fp16 checkpoint (2B)
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=6 setsid nohup sglang serve $SPEC $RS --mamba-ssm-dtype float16 --port 31012 > scratch/m_fp16.log 2>&1 < /dev/null &
# GPU7: replayssm fp32 storage + TF32 dots (MAX PRECISION, 4B)
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=7 setsid nohup sglang serve $SPEC $RS --mamba-ssm-dtype float32 --port 31013 > scratch/m_fp32.log 2>&1 < /dev/null &
echo "launched recur=31010 bf16=31011 fp16=31012 fp32=31013 (GPUs 4-7)"
