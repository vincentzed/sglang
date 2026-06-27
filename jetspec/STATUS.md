# DFlash Tree Speculative Decode Status

Updated: 2026-06-27 22:37 UTC

## Done / Committed

- `8f9e732f2c`: dense Qwen3-8B tree verify was previously token-exact.
- `6dd0f39e75`: MoE Qwen3.6-35B-A3B narrowed to one prompt-4 holdout; known risk was an unguarded Mamba commit control-flow path.
- `7c80ee7b40`: fixed the Mamba commit control-flow guard in `python/sglang/srt/speculative/dflash_worker_v2.py`; dense Job 0 revalidation stayed token-exact.
- `5817fc9d2e90`: made the MoE tree path token-exact against a fresh width=1 oracle by using a labeled hybrid/MoE linear-commit fallback for the accepted commit block. This is a correctness fallback, not a direct tree-state commit speedup.

## Job 0 Validation

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend flashinfer`
- Deterministic/no-graph flags: `--enable-deterministic-inference --random-seed 0 --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`
- Harness: `PYTHONPATH=python python jetspec/run_fixed_prompts.py`, 10 fixed prompts, `max_new_tokens=96`

Artifacts:

| run | artifact | token exact vs fresh oracle | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|
| 8B linear width=1 fresh oracle | `jetspec/runs/job0_8b_linear_nograph_31370.json` | oracle | 3.2856 | 115.79 |
| 8B tree w7/b64 | `jetspec/runs/job0_8b_tree_w7_b64_nograph_31371.json` | PASS, 10/10 | 4.0495 | 116.21 |
| 8B tree w7/b128 | `jetspec/runs/job0_8b_tree_w7_b128_nograph_31372.json` | PASS, 10/10 | 4.5461 | 73.11 |

Local checks:
- `PYTHONPATH=python python -m py_compile python/sglang/srt/speculative/dflash_worker_v2.py`: PASS
- `PYTHONPATH=python python test/registered/unit/spec/test_dflash_tree_construction.py`: PASS, 18 tests

Notes:
- A non-`nohup` background launch wrapper produced an empty log because the child died when the wrapper shell exited. Foreground server sessions were used after that, with explicit launch output.
- `~/.claude/GPU.md` was requested by the global instructions but is not present in this container.

## In Progress

Job 2: CUDA graph support for DFlash tree verify, using fixed `tree_budget` padding and EAGLE3-style topk>1 graph buffers/buckets.

Exact next step:
- Read the existing EAGLE3 cuda-graph topk>1 path and mirror its fixed-shape/bucket strategy for DFlash tree verify.
- Start with dense 8B graph capture, then route MoE tree verify through the same GDN topk>1 verify-graph hook.
- Re-run dense and MoE fresh-oracle losslessness gates with cuda-graph enabled before committing Job 2.

## Job 1 Validation

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Dense backend: `--attention-backend flashinfer`
- MoE model: `Qwen/Qwen3.6-35B-A3B`
- MoE draft model: `JetSpec/jetspec-qwen3.6-35b-a3b`
- MoE backends: `--linear-attn-decode-backend cutedsl --linear-attn-prefill-backend cutedsl --mamba-ssm-dtype bfloat16 --attention-backend trtllm_mha`
- Deterministic/no-graph flags: `--enable-deterministic-inference --random-seed 0 --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`
- Harness: `PYTHONPATH=python python jetspec/run_fixed_prompts.py`, 10 fixed prompts, `max_new_tokens=96`

Fresh-oracle artifacts:
- Dense: `jetspec/runs/job0_8b_linear_nograph_31370.json`
- MoE: `jetspec/runs/job1_moe_linear_cutedsl_nograph_31380.json`

Artifacts:

| run | artifact | token exact vs fresh oracle | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|
| 8B tree w7/b64 | `jetspec/runs/job1_8b_tree_w7_b64_nograph_31499.json` | PASS, 10/10 | 3.8976 | 131.76 |
| 8B tree w7/b128 | `jetspec/runs/job1_8b_tree_w7_b128_nograph_31500.json` | PASS, 10/10 | 4.1056 | 101.44 |
| MoE tree w4/b64 | `jetspec/runs/job1_moe_tree_w4_b64_linear_commit_31495.json` | PASS, 10/10 | 4.4179 | 57.52 |
| MoE tree w4/b128 | `jetspec/runs/job1_moe_tree_w4_b128_linear_commit_31496.json` | PASS, 10/10 | 4.4179 | 51.64 |
| MoE tree w7/b64 | `jetspec/runs/job1_moe_tree_w7_b64_linear_commit_31497.json` | PASS, 10/10 | 4.4179 | 57.75 |
| MoE tree w7/b128 | `jetspec/runs/job1_moe_tree_w7_b128_linear_commit_31498.json` | PASS, 10/10 | 4.4179 | 54.32 |

Local checks after cleanup:
- `git diff --check`: PASS
- `PYTHONPATH=python python -m py_compile python/sglang/srt/speculative/dflash_worker_v2.py`: PASS
- `PYTHONPATH=python python test/registered/unit/spec/test_dflash_tree_construction.py`: PASS, 18 tests

Notes:
- The prompt-4 holdout was traced to chunk-sensitive hybrid GDN state/KV commit behavior: the tree path could accept a wider block than the width=1 oracle, while the oracle committed the same region as smaller MTP chunks.
- The committed MoE fix keeps tree verify for candidate evaluation but, by default for hybrid/MoE targets, replays the width=1 linear fallback block for the accepted commit block so persistent KV/GDN state and emitted tokens match the fresh oracle. The fallback is gated by `SGLANG_DFLASH_TREE_MOE_LINEAR_COMMIT_FALLBACK` and can be disabled with `0`.
- This closes Job 1 correctness but does not prove a MoE tree throughput win; Job 3 must report perf honestly.

## Not Started

- Job 3: normal-mode perf bench with CUDA graph on; update `jetspec/notes/bench_results.md` honestly with linear vs tree dense and MoE throughput.
