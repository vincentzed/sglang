# JetSpec DFlash live GPU validation

Date: 2026-06-27 UTC

Hardware/env:
- GPU: `CUDA_VISIBLE_DEVICES=7` (`NVIDIA B300 SXM6 AC`, SM100, 275 GB)
- Required env used for every server: `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Repo/env: local develop install, `python -m sglang.launch_server`, `PYTHONPATH=python`
- Decode settings: BF16/default dtype, greedy `temperature=0`, `--tp-size 1`, `max_new_tokens=96`
- Harness: `jetspec/run_fixed_prompts.py`, 10 fixed prompts including GSM-style arithmetic, sequence, code, translation, and summary prompts.

## Caveats

- `--attention-backend fa3` is not usable on this B300/SM100 host in this checkout: launch fails with `FlashAttention v3 Backend requires SM>=80 and SM<=90. Please use --attention-backend flashinfer.` The 8B smoke pair was therefore run with `--attention-backend flashinfer`.
- Port `30000` was already occupied by an unrelated server reporting `nvidia/GLM-5.2-NVFP4`, so the MoE runs used ports `31106`, `31107`, and `31108`.
- The user's original MoE command already had the invalid `--speculative-draft-attention-backend fa4` removed. No working draft-specific FA4 equivalent exists in this checkout.
- The MoE command needed one additional required flag on SM100: `--mamba-ssm-dtype bfloat16`. Without it, validation fails with `--linear-attn-decode-backend flashinfer on SM100+ requires --mamba-ssm-dtype bfloat16, got None`.
- The MoE server accepts `--attention-backend trtllm_mha` for the target, but the DFlash draft worker logs: `DFLASH draft worker does not support 'trtllm_mha' because the draft path requires per-layer DFlash attention. Falling back to 'flashinfer'.`
- `z-lab/Qwen3.6-35B-A3B-DFlash` config was fetched under `jetspec/_hf_configs/` and confirmed `architectures: ["DFlashDraftModel"]`.

## 8B smoke pair

Models:
- Target: `Qwen/Qwen3-8B`
- Draft: `JetSpec/jetspec-qwen3-8b`

Linear baseline command:

```bash
CUDA_VISIBLE_DEVICES=7 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 PYTHONPATH=python \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --dtype bfloat16 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path JetSpec/jetspec-qwen3-8b \
  --speculative-num-draft-tokens 16 \
  --speculative-dflash-tree-width 1 \
  --reasoning-parser qwen3 \
  --attention-backend flashinfer \
  --tp-size 1 --mem-fraction-static 0.8 --trust-remote-code \
  --host 0.0.0.0 --port 31085
```

Tree commands used the same launch plus:

```bash
--speculative-dflash-tree-width {4,7} --speculative-dflash-tree-budget {64,128}
```

Results:

| run | artifact | token exact vs linear | mean accept length | aggregate tok/s | mean prompt tok/s |
|---|---|---:|---:|---:|---:|
| linear width=1 | `jetspec/baseline_linear.json` | oracle | 3.5505 | 492.38 | 622.20 |
| tree w4 b64 | `jetspec/runs/8b_tree_w4_b64_branch_reverify_31103.json` | PASS | 4.4799 | 187.56 | 226.39 |
| tree w4 b128 | `jetspec/runs/8b_tree_w4_b128_branch_reverify_31104.json` | PASS | 4.7291 | 168.59 | 202.52 |
| tree w7 b64 | `jetspec/runs/8b_tree_w7_b64_branch_reverify_31102.json` | PASS | 3.9621 | 181.35 | 207.82 |
| tree w7 b128 | `jetspec/runs/8b_tree_w7_b128_branch_reverify_31105.json` | PASS | 4.7845 | 175.76 | 206.82 |

8B gate status: PASS. All four tree runs have `losslessness.token_exact=true` and zero mismatches against `jetspec/baseline_linear.json`.

8B performance status: tree accept length materially exceeds linear, with the best observed mean accept length `4.7845` vs linear `3.5505`. Throughput is lower because the current correctness fix performs a causal branch reverify after the tree pass.

## MoE before/after

Models:
- Target: `Qwen/Qwen3.6-35B-A3B`
- Draft: `z-lab/Qwen3.6-35B-A3B-DFlash`

Required MoE linear command used:

```bash
source ~/.zshenv.local >/dev/null 2>&1 || true
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export CUDA_VISIBLE_DEVICES=7
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export PYTHONPATH=python
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash \
  --speculative-dflash-block-size 8 \
  --attention-backend trtllm_mha \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-ssm-dtype bfloat16 \
  --tp-size 1 \
  --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 \
  --cuda-graph-backend-prefill tc_piecewise \
  --enable-flashinfer-allreduce-fusion \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 --port 31106
```

MoE tree commands used the same flags and changed only port plus tree knobs:

```bash
# w4/b64
--host 0.0.0.0 --port 31107 \
--speculative-dflash-tree-width 4 --speculative-dflash-tree-budget 64

# w7/b128
--host 0.0.0.0 --port 31108 \
--speculative-dflash-tree-width 7 --speculative-dflash-tree-budget 128
```

Results:

| run | artifact | token exact vs linear | mean accept length | aggregate tok/s | mean prompt tok/s |
|---|---|---:|---:|---:|---:|
| linear before | `jetspec/runs/moe_linear_zlab_31106.json` | oracle | 4.3034 | 553.56 | 574.88 |
| tree after w4 b64 | `jetspec/runs/moe_tree_w4_b64_zlab_31107_retry.json` | PASS | 4.2969 | 100.64 | 124.02 |
| tree after w7 b128 | `jetspec/runs/moe_tree_w7_b128_zlab_31108.json` | PASS | 4.2717 | 101.37 | 123.53 |

MoE gate status: PASS for the two live tree shapes run. Both tree artifacts have `losslessness.token_exact=true` and zero mismatches against `jetspec/runs/moe_linear_zlab_31106.json`.

MoE performance status: FAIL for speedup/accept-length improvement. The hybrid-Mamba fallback is live and lossless, but it chooses a branch from the draft tree and then causally reverifies one block-sized branch. That avoids the FlashInfer GDN `cache_steps >= T` assertion for tree budgets larger than the DFlash block size, but it does not gain accept length over linear DFlash and adds overhead.

## Fixes made during validation

- Added `ForwardBatch.allow_cuda_graph` and propagated it through DFlash verify prep so custom tree verification can force eager verify without globally disabling graph capture.
- Fixed tree custom-mask sizing for exact total verify tokens.
- Fixed tree verify cache-location dtype assignment.
- Converted DFlash tree containers from `@dataclass` to `msgspec.Struct`.
- Reworked tree correctness to select a branch from the tree pass, then causally reverify that branch before computing accept/bonus/KV commit. This fixed the 8B token divergence.
- Implemented hybrid-Mamba tree commit support by calling the same target Mamba-state update path used by linear DFlash after the causal branch verify.
- Added a hybrid-Mamba fallback that skips the unsafe full target tree verify and uses the draft tree only for branch selection before causal branch reverify.

## CUDA graph status

- Existing server graph capture remains enabled for the required MoE launch flags; logs show target prefill, target verify, and draft verify CUDA graph capture completing during startup.
- The new tree speculative decode path still returns `can_run_cuda_graph=False`.
- Custom-mask tree target verification and causal branch reverify are explicitly guarded with `allow_cuda_graph=False`. Enabling replay for tree decode still requires fixed tree capture buckets/padded tree-budget graph runners.

## Validation commands

```bash
PYTHONPATH=python python test/registered/unit/spec/test_dflash_tree_construction.py
```

Result: 11 tests passed.

Representative harness commands:

```bash
PYTHONPATH=python python jetspec/run_fixed_prompts.py \
  --base-url http://127.0.0.1:31085 \
  --out jetspec/runs/8b_linear_flashinfer_31085.json \
  --run-name 8b-linear-flashinfer-overlap \
  --target-model Qwen/Qwen3-8B \
  --draft-model JetSpec/jetspec-qwen3-8b \
  --tree-width 1 --tree-budget 16 --max-new-tokens 96 \
  --health-timeout-s 1800 --request-timeout-s 240

PYTHONPATH=python python jetspec/run_fixed_prompts.py \
  --base-url http://127.0.0.1:31108 \
  --out jetspec/runs/moe_tree_w7_b128_zlab_31108.json \
  --run-name moe-tree-w7-b128-zlab \
  --target-model Qwen/Qwen3.6-35B-A3B \
  --draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --tree-width 7 --tree-budget 128 --max-new-tokens 96 \
  --attention-backend trtllm_mha \
  --compare-to jetspec/runs/moe_linear_zlab_31106.json \
  --health-timeout-s 1800 --request-timeout-s 600
```
