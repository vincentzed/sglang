# DFlash Tree Speculative Decode Status

Updated: 2026-06-28 00:35 UTC

## Done / Committed

- `8f9e732f2c`: dense Qwen3-8B tree verify was previously token-exact.
- `6dd0f39e75`: MoE Qwen3.6-35B-A3B narrowed to one prompt-4 holdout; known risk was an unguarded Mamba commit control-flow path.
- `7c80ee7b40`: fixed the Mamba commit control-flow guard in `python/sglang/srt/speculative/dflash_worker_v2.py`; dense Job 0 revalidation stayed token-exact.
- `5817fc9d2e90`: made the MoE tree path token-exact against a fresh width=1 oracle by using a labeled hybrid/MoE linear-commit fallback for the accepted commit block. This is a correctness fallback, not a direct tree-state commit speedup.
- `5c2120c5a91c`: recorded Job 1 status and validation artifacts.
- `11c65c076d`: enabled DFlash tree target-verify CUDA graphs for dense FlashInfer expanded-causal verify and MoE hybrid-GDN custom-mask verify; dense and MoE cuda-graph gates are token-exact against fresh oracles.
- `ff275594dd`: recorded Job 2 status and validation artifacts.
- `e8d533a19c`: recorded Job 3 normal-mode CUDA graph performance results and verdict in `jetspec/notes/bench_results.md`.
- Perf pass 2026-06-28: attempted dense direct accepted-path commit, reverted it after token-exactness failed, then swept dense CUDA-graph tree width/budget. Results are recorded in `jetspec/notes/bench_results.md`.

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

- None.

Exact next step:
- If continuing performance work, the dense blocker is acceptance/amortization rather than correctness: the best valid swept dense tree throughput was w2/b16 at 173.77 tok/s but it did not beat linear accept, and the best throughput while beating linear accept was w4/b32 at 163.84 tok/s, still 0.32x linear. A direct dense accepted-path commit is not safe with the current FlashInfer expanded-causal verifier because tree KV/hidden state is not causal-equivalent to the accepted replay branch.
- MoE direct exact KV/GDN-state commit remains a follow-up. It was not attempted in this pass.

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

## Job 2 Validation

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Dense backend: `--attention-backend flashinfer`
- MoE model: `Qwen/Qwen3.6-35B-A3B`
- MoE draft model: `z-lab/Qwen3.6-35B-A3B-DFlash`
- MoE backends: `--linear-attn-decode-backend cutedsl --linear-attn-prefill-backend cutedsl --mamba-ssm-dtype bfloat16 --attention-backend trtllm_mha`
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`
- Harness: `PYTHONPATH=python python jetspec/run_fixed_prompts.py`, 10 fixed prompts, `max_new_tokens=96`

Fresh-oracle artifacts:
- Dense: `jetspec/runs/job0_8b_linear_nograph_31370.json`
- MoE: `jetspec/runs/job1_moe_linear_cutedsl_nograph_31380.json`

Artifacts:

| run | artifact | token exact vs fresh oracle | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|
| 8B tree w7/b64 cuda graph | `jetspec/runs/job2_8b_tree_w7_b64_cudagraph_rows_31518.json` | PASS, 10/10 | 4.0495 | 74.71 |
| 8B tree w7/b128 cuda graph | `jetspec/runs/job2_8b_tree_w7_b128_cudagraph_rows_31519.json` | PASS, 10/10 | 4.5399 | 58.64 |
| MoE tree w4/b64 cuda graph | `jetspec/runs/job2_moe_tree_w4_b64_cudagraph_31520.json` | PASS, 10/10 | 4.4179 | 80.64 |
| MoE tree w4/b128 cuda graph | `jetspec/runs/job2_moe_tree_w4_b128_cudagraph_31521.json` | PASS, 10/10 | 4.4179 | 66.40 |
| MoE tree w7/b64 cuda graph | `jetspec/runs/job2_moe_tree_w7_b64_cudagraph_31522.json` | PASS, 10/10 | 4.4179 | 78.78 |
| MoE tree w7/b128 cuda graph | `jetspec/runs/job2_moe_tree_w7_b128_cudagraph_31523.json` | PASS, 10/10 | 4.4179 | 67.34 |

CUDA graph evidence:
- Dense 8B w7/b64 and w7/b128 captured DFlash target-verify CUDA graphs with FlashInfer expanded-causal rows and decode logs reported `cuda graph: True`.
- MoE w4/w7, budget 64/128 captured DFlash target-verify CUDA graphs with the hybrid-GDN path and a graph-capable `flashinfer` full-attention replacement for the periodic full-attention layers; decode logs reported `cuda graph: True`.

Local checks after Job 2:
- `git diff --check`: PASS
- `PYTHONPATH=python python -m py_compile python/sglang/srt/model_executor/runner/eager_runner.py python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py python/sglang/srt/speculative/dflash_worker_v2.py python/sglang/srt/layers/attention/flashinfer_backend.py`: PASS
- `PYTHONPATH=python python test/registered/unit/spec/test_dflash_tree_construction.py`: PASS, 18 tests

Notes:
- Dense FlashInfer tree verify uses a graph shape of `tree_budget` expanded rows by `speculative_num_draft_tokens` tokens, with fixed buckets capped to the active request count.
- MoE hybrid-GDN tree verify uses fixed `tree_budget` tokens per request and reuses the graph-captured full-attention replacement at runtime so target-verify graph replay and eager fallback share the same backend semantics.
- The MoE correctness path still relies on the labeled accepted-path linear replay fallback introduced in Job 1 for persistent KV/GDN state commit. Job 2 proves cuda graph does not change emitted tokens.

## Job 3 Validation

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Normal greedy mode: no deterministic flags, harness `temperature=0`
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`
- Harness: `PYTHONPATH=python python jetspec/run_fixed_prompts.py`, 10 fixed prompts, `max_new_tokens=96`

Dense artifacts:

| run | artifact | token exact vs fresh graph-linear | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|
| 8B linear width=1 | `jetspec/runs/job3_8b_linear_cudagraph_31530.json` | oracle | 3.7569 | 510.53 |
| 8B tree w7/b64 | `jetspec/runs/job3_8b_tree_w7_b64_cudagraph_31531.json` | PASS, 10/10 | 4.5441 | 131.52 |
| 8B tree w7/b128 | `jetspec/runs/job3_8b_tree_w7_b128_cudagraph_31532.json` | PASS, 10/10 | 5.2161 | 85.69 |

MoE artifacts:

| run | artifact | token exact vs fresh graph-linear | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|
| MoE linear width=1 | `jetspec/runs/job3_moe_linear_cudagraph_31540.json` | oracle | 4.4179 | 499.06 |
| MoE tree w4/b64 | `jetspec/runs/job2_moe_tree_w4_b64_cudagraph_31520.json` | PASS, 10/10 | 4.4179 | 80.64 |
| MoE tree w4/b128 | `jetspec/runs/job2_moe_tree_w4_b128_cudagraph_31521.json` | PASS, 10/10 | 4.4179 | 66.40 |
| MoE tree w7/b64 | `jetspec/runs/job2_moe_tree_w7_b64_cudagraph_31522.json` | PASS, 10/10 | 4.4179 | 78.78 |
| MoE tree w7/b128 | `jetspec/runs/job2_moe_tree_w7_b128_cudagraph_31523.json` | PASS, 10/10 | 4.4179 | 67.34 |

Verdict:
- Tree does not beat linear on tok/s in these normal-mode graph runs.
- Dense tree gains accept length but loses throughput because fixed-shape tree verify and accepted-path replay/commit overhead dominate.
- MoE tree keeps the same mean accept length as linear under the accepted-path linear replay fallback, so it adds overhead without an acceptance gain.

## 2026-06-28 Performance Pass

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Normal greedy mode: no deterministic flags, harness `temperature=0`
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Dense backend: `--attention-backend flashinfer`

Lever 1:
- Attempted to remove dense CUDA-graph accepted-path replay and directly gather accepted expanded-causal tree KV.
- Reverted the code change because `jetspec/runs/perf1_8b_tree_w7_b64_direct_commit_cudagraph_31611.json` mismatched `5/10` prompts against the fresh graph-linear oracle despite improving throughput to `158.50 tok/s`.
- Diagnostic artifact `jetspec/runs/perf1_debug_prompt4_tree_w7_b64_direct_commit_compare_31612.json` reproduced prompt 4 divergence. Compare logs showed same-shape tree state reproduced itself, but accepted tree rows differed from a fresh causal branch replay, so direct commit is not lossless with the current verifier.

Lever 2:

Fresh oracle:
- `jetspec/runs/perf1_8b_linear_cudagraph_31610.json`: mean accept `3.7569`, aggregate `511.38 tok/s`.

Valid dense sweep artifacts:

| run | token exact vs fresh graph-linear | mean accept length | aggregate tok/s |
|---|---:|---:|---:|
| w2/b16 `jetspec/runs/perf1_sweep_8b_tree_w2_b16_cudagraph_31623.json` | PASS, 10/10 | 3.7477 | 173.77 |
| w4/b16 `jetspec/runs/perf1_sweep_8b_tree_w4_b16_cudagraph_31624.json` | PASS, 10/10 | 3.2309 | 169.18 |
| w7/b16 `jetspec/runs/perf1_sweep_8b_tree_w7_b16_cudagraph_31625.json` | PASS, 10/10 | 2.9500 | 159.22 |
| w2/b32 `jetspec/runs/perf1_sweep_8b_tree_w2_b32_cudagraph_31626.json` | PASS, 10/10 | 4.4810 | 155.15 |
| w4/b32 `jetspec/runs/perf1_sweep_8b_tree_w4_b32_cudagraph_31627.json` | PASS, 10/10 | 4.1024 | 163.84 |
| w7/b32 `jetspec/runs/perf1_sweep_8b_tree_w7_b32_cudagraph_31628.json` | PASS, 10/10 | 3.4931 | 154.34 |
| w2/b64 `jetspec/runs/perf1_sweep_8b_tree_w2_b64_cudagraph_31629.json` | PASS, 10/10 | 4.5582 | 111.86 |
| w4/b64 `jetspec/runs/perf1_sweep_8b_tree_w4_b64_cudagraph_31630.json` | PASS, 10/10 | 4.8405 | 127.94 |
| w7/b64 `jetspec/runs/perf1_sweep_8b_tree_w7_b64_cudagraph_31631.json` | PASS, 10/10 | 4.5441 | 130.57 |

Invalid dense sweep configs:
- w2/b8, w4/b8, and w7/b8 failed at launch because `tree_budget=8` is below the DFlash block size 16. Logs: `jetspec/logs/perf1_sweep_8b_tree_w{2,4,7}_b8_cudagraph_3162{0,1,2}_server.log`.

Verdict:
- No dense tree config beats linear throughput.
- Best valid dense tree throughput: w2/b16 at `173.77 tok/s` (`0.34x` linear), but it does not beat linear accept.
- Best throughput while beating linear accept: w4/b32 at `163.84 tok/s` (`0.32x` linear).
- Best swept acceptance: w4/b64 at mean accept `4.8405`; prior Job 3 w7/b128 reached `5.2161` but dropped to `85.69 tok/s`.
- Dense acceptance plateaus around 5 with this draft head, well short of the 7-9 range needed to amortize fixed tree verify plus replay/commit overhead.

## Not Started

- MoE direct exact KV/GDN-state commit after accepted tree verify. The current MoE path remains lossless through the linear-commit fallback and has not been changed in this performance pass.
