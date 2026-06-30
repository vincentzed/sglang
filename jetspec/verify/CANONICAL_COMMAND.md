# Canonical DFlash MoE command — VERIFIED working (2026-06-28, B300 SM103, GPU4)

The user's canonical DFlash launch for `Qwen/Qwen3.6-35B-A3B` + draft
`z-lab/Qwen3.6-35B-A3B-DFlash` WORKS for **linear DFlash**, with ONE required
addition on SM100+: `--mamba-ssm-dtype bfloat16` (without it, arg validation
raises `--linear-attn-decode-backend flashinfer on SM100+ requires
--mamba-ssm-dtype bfloat16`).

Verified-working command:

```
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash \
  --speculative-dflash-block-size 8 \
  --speculative-draft-attention-backend fa4 \
  --attention-backend trtllm_mha \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \          # REQUIRED on SM100+; missing from the original
  --mamba-scheduler-strategy extra_buffer \
  --tp-size 1 --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 --cuda-graph-backend-prefill tc_piecewise \
  --enable-flashinfer-allreduce-fusion --mem-fraction-static 0.8 \
  --host 0.0.0.0 --port <free>
```

Live evidence (sample greedy request): coherent output, `spec_verify_ct=11`,
`accept_length=5.82`. Server came up healthy in ~3.5 min (load + cuda-graph capture).

## Confirmed backend facts
- `--speculative-draft-attention-backend fa4` is a REAL CLI flag (auto-generated
  from the `speculative_draft_attention_backend` field) and the DFlash draft uses fa4.
- `--attention-backend trtllm_mha` is valid on Blackwell but **forces page_size -> 64**
  ("TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing from None to 64").
- linear-attn here runs the **FlashInfer GDN kernel** (decode=extend=verify=FlashInferGDNKernel).
  This is the kernel that ASSERTS on tree-verify sizes (`cache_steps >= T`), which is why the
  TREE feature must use triton/cutedsl GDN, not flashinfer GDN. So canonical-linear (flashinfer GDN)
  and the tree feature (triton/cutedsl GDN) legitimately diverge on the linear-attn backend.
- Deprecation warnings (non-fatal): `--mamba-scheduler-strategy` -> use `--mamba-radix-cache-strategy`;
  `--enable-flashinfer-allreduce-fusion` -> `--flashinfer-allreduce-fusion-backend=auto`.
- Benign: `z-lab/Qwen3.6-35B-A3B-DFlash` has no `generation_config.json` (proceeds without).

## Implication for the tree feature / next codex MoE run
Benchmark MoE tree-vs-linear against THIS exact config (trtllm_mha target @ page_size 64,
fa4 draft, mamba-ssm-dtype bfloat16) — but the tree verify's linear-attn GDN must be
triton/cutedsl (flashinfer GDN asserts on tree sizes). Dense 8B uses fa4 + page_size 16
per `test/registered/attention/unittests/dense/test_fa4.py` (`runner_fa4_dflash_verify_chain`).
