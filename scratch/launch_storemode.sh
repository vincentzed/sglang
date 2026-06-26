#!/usr/bin/env bash
# VALUE: A/B the checkpoint STORE precision modes in production: fp16 RNE (current) vs
# fp16+SR (stochastic rounding, SOTA in TRT-LLM/flashinfer/vLLM) vs fp16+EF (error-feedback).
# Store mode is env-selected (SGLANG_REPLAYSSM_STORE_MODE) in gdn_replayssm_spec_decode.py.
# CUDA graph ON (production): SR's per-step variation comes from b_cache_base in-kernel,
# so no host seed sync is needed and the run is ~3x faster than graph-off.
set +e
cd /sgl-workspace/sglang
M=Qwen/Qwen3.6-35B-A3B
SPEC="--model-path $M --linear-attn-decode-backend triton --reasoning-parser qwen3 \
--attention-backend triton --page-size 1 \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --trust-remote-code"
RS="--enable-gdn-replayssm-spec --gdn-replayssm-spec-cache-len 16"
FP16="--mamba-ssm-dtype float16"
COMMON="SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1"

# GPU4: recurrent exact reference (replayssm off)
env $COMMON CUDA_VISIBLE_DEVICES=4 setsid nohup sglang serve $SPEC --port 31010 > scratch/s_recur.log 2>&1 < /dev/null &
# GPU5: fp16 RNE (current PR / best so far)
env $COMMON SGLANG_REPLAYSSM_STORE_MODE=rne CUDA_VISIBLE_DEVICES=5 setsid nohup sglang serve $SPEC $RS $FP16 --port 31011 > scratch/s_rne.log 2>&1 < /dev/null &
# GPU6: fp16 + stochastic rounding
env $COMMON SGLANG_REPLAYSSM_STORE_MODE=sr CUDA_VISIBLE_DEVICES=6 setsid nohup sglang serve $SPEC $RS $FP16 --port 31012 > scratch/s_sr.log 2>&1 < /dev/null &
# GPU7: fp16 + error-feedback
env $COMMON SGLANG_REPLAYSSM_STORE_MODE=ef CUDA_VISIBLE_DEVICES=7 setsid nohup sglang serve $SPEC $RS $FP16 --port 31013 > scratch/s_ef.log 2>&1 < /dev/null &
echo "launched recur=31010 rne=31011 sr=31012 ef=31013 (GPUs 4-7, cuda-graph OFF)"
