# DFlash Tree Speculative Decode Status

Updated: 2026-06-27 19:55 UTC

## Done / Committed

- `8f9e732f2c`: dense Qwen3-8B tree verify was previously token-exact.
- `6dd0f39e75`: MoE Qwen3.6-35B-A3B narrowed to one prompt-4 holdout; known risk was an unguarded Mamba commit control-flow path.
- Pending Job 0 commit: fixed the Mamba commit control-flow guard in `python/sglang/srt/speculative/dflash_worker_v2.py`.

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

Job 1: close the MoE prompt-4 holdout and make MoE fully lossless.

Exact next step:
- Launch a fresh MoE width=1 oracle under the required MoE flags:
  `--linear-attn-decode-backend cutedsl --linear-attn-prefill-backend cutedsl --mamba-ssm-dtype bfloat16 --attention-backend trtllm_mha`.
- Compare tree `w4/b64`, `w4/b128`, `w7/b64`, and `w7/b128` against that fresh oracle.
- If prompt 4 still diverges after at most 1-2 targeted tries, instrument prompt 4 per layer for committed KV plus GDN conv/SSM state deltas against a clean width=1 linear forward of the same accepted tokens.

## Not Started

- Job 2: CUDA graph support for DFlash tree verify, using fixed `tree_budget` padding and EAGLE3-style topk>1 graph buffers/buckets.
- Job 3: normal-mode perf bench with CUDA graph on; update `jetspec/notes/bench_results.md` honestly with linear vs tree dense and MoE throughput.
