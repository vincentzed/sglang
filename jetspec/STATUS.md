# DFlash Tree Speculative Decode Status

Updated: 2026-06-28 05:01 UTC

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
- `dd1cc2947f`: Job A1 dense FlashInfer tree verify now defaults to the compact custom-mask verifier and uses a causal accepted-branch reverify/commit for losslessness. For w7/b64, target tree-verify graph shape drops from expanded `64 * 16 = 1024` positions/request to compact `64` positions/request plus a `16`-position branch reverify.
- Job B 2026-06-28: canonical MT-bench rerun completed for width=1 linear and compact tree w7/b64. Tree raised accept length from `4.11` to `5.08`, but throughput fell from `774.91` to `283.90 tok/s`, so compact tree is still `0.37x` linear on the real benchmark.
- Direct-commit causal-equivalence probe 2026-06-28: dense no-reverify direct commit remains blocked. The fresh flushed w7/b64 gate failed `6/10` prompts, and per-layer diagnostics show the first accepted-path discrepancy at layer-0 attention output while layer-0 K/V are still identical to the clean causal branch.
- Realignment Job 1 2026-06-28: linear DFlash on dense Qwen3-8B now runs on the canonical FA4 target backend with `page_size=16`. The first unmodified launch rewrote `--page-size 16` to `128`; `python/sglang/srt/server_args.py` now keeps explicit FA4 page16 for DFlash while preserving the generic FA4 page128 auto-force. Fresh flushed width=1 oracle: `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json`, mean accept `3.5827`, aggregate `561.04 tok/s`.
- Realignment Job 2 2026-06-28: dense FA4 page16 tree verify already routes through the multi-token custom-mask/FULL_MASK shape, not the FlashInfer compact path. No-reverify direct commit is still not lossless on FA4 page16; the first diagnostic shows exact layer-0 K/V but a layer-0 hidden delta, so the residual is in the FA4 tree/custom-mask verify execution shape rather than DFlash draft KV materialization. Retained-reverify w7/b64 and w7/b128 are token-exact against the fresh FA4 page16 oracle.
- Realignment Job 3 2026-06-28: canonical MT-bench rerun completed on the aligned FA4 page16 backend. Width=1 linear reached `763.948 tok/s` with accept length `4.07`; dense tree w7/b64 with retained reverify reached `199.983 tok/s` with accept length `3.695`, so tree is `0.26x` linear (`3.82x` slower). No-reverify was not landed.

## Realignment Job 1 - FA4 Page16 Linear Baseline

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend fa4 --page-size 16`
- Normal greedy mode: no deterministic flags, harness `temperature=0`
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph left at the normal `tc_piecewise` default.
- Harness: `PYTHONPATH=python python jetspec/run_fixed_prompts.py`, 10 fixed prompts, `max_new_tokens=96`, `--flush-cache-before-run --flush-cache-between-prompts`

Launch correction:
- Before this pass, FA4 non-MLA server-arg normalization rewrote DFlash `--page-size 16` to `128`, so the requested canonical baseline could not actually run as page16.
- The guard now exempts explicit DFlash FA4 page16 from that rewrite. Generic FA4 non-MLA topk=1 decode still keeps the existing page128 auto-force.
- Evidence: server log `jetspec/logs/job1_fa4p16_linear_w1_31811_server.log` shows `page_size=16`, `attention_backend='fa4'`, DFlash block size `16`, and target verify graph capture with `num_tokens_per_bs=16`.

Artifact:

| run | artifact | server backend/page | token exact | mean accept length | aggregate tok/s |
|---|---|---|---:|---:|---:|
| 8B linear width=1 fresh flushed oracle | `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json` | `fa4`, `page_size=16` | oracle | 3.5827 | 561.04 |

Notes:
- A background `nohup` launch on port `31810` died with a zero-byte log, so the server was relaunched in a foreground tool session with output also saved through `tee`.
- A foreground pre-patch launch confirmed the failure mode: SGLang accepted `--attention-backend fa4 --page-size 16` but rewrote runtime `page_size` to `128`. No oracle from that misaligned launch is used.
- This FA4 page16 oracle is the baseline for the dense tree losslessness gates in Job 2.

## Realignment Job 2 - FA4 Page16 Tree Verify

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend fa4 --page-size 16`
- Normal greedy mode: no deterministic flags, harness `temperature=0`
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph left at the normal `tc_piecewise` default.
- Harness: `PYTHONPATH=python python jetspec/run_fixed_prompts.py`, 10 fixed prompts, `max_new_tokens=96`, `--flush-cache-before-run --flush-cache-between-prompts`
- Fresh flushed oracle: `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json`

Shape check:
- No extra DFlash-worker reroute was needed for FA4. The old compact/expanded branches in `dflash_worker_v2.py` are gated to `FlashInferAttnBackend`; with `--attention-backend fa4`, the active backend is `FlashAttentionBackend`, so dense tree verify uses the generic multi-token custom-mask path.
- That path builds an ancestor-only tree mask with prefix attention in `build_tree_custom_mask`, matching the EAGLE FULL_MASK contract at the Python mask level.
- Evidence: `jetspec/logs/job2_fa4p16_tree_w7_b64_reverify_31814_server.log` captured target verify with `num_tokens_per_bs=64`, and `jetspec/logs/job2_fa4p16_tree_w7_b128_reverify_31815_server.log` captured target verify with `num_tokens_per_bs=128`; both are compact multi-token tree verifies on `fa4`, `page_size=16`.

No-reverify direct-commit gate:

| run | artifact | token exact vs FA4 page16 oracle | mismatches | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|---:|
| 8B tree w7/b64, dense accepted-path reverify disabled | `jetspec/runs/job2_fa4p16_tree_w7_b64_noreverify_flush_31812.json` | FAIL | 10/10 | 2.9558 | 281.32 |

Diagnostic:
- Diagnostic server log: `jetspec/logs/job2_fa4p16_tree_w7_b64_noreverify_compare_31813_server.log`.
- Representative first compare: prefix `[6]`, commit length `[2]`, accepted local nodes `[[0,3]]`, branch candidates `[[12095,13]]`.
- Tree and clean causal branch predictions agreed on that step: `tree_predict=[[13,576]]`, `branch_predict=[[13,576]]`.
- Accepted-path hidden still diverged: `hidden_max_abs=40.0`.
- First per-layer hidden discrepancy was at layer 0: `first_layer_hidden_delta=(0, 0.1357421875)`.
- Layer-0 K/V were exact: layer-0 `kv_max_abs=(0.0, 0.0)`.
- Later layer K/V diverged after the layer-0 hidden delta propagated; for example layer 1 had `kv_max_abs=(2.75, 0.03076171875)`.
- The diagnostic process hit a CUDA error after logging this compare during the debug replay cleanup, so this log is evidence for causal-equivalence triage only, not a successful serving gate.

Conclusion:
- The no-reverify EAGLE-style direct commit cannot be landed on dense FA4 page16 yet.
- The residual does not point at positions, RoPE, or DFlash draft K/V materialization; those are still consistent through layer-0 K/V.
- The residual is in the FA4 tree/custom-mask target-verify execution shape: the first mismatch is the layer-0 attention output for the accepted path.
- Therefore the dense accepted-path reverify remains required. MoE was not changed.

Retained-reverify gates:

| run | artifact | token exact vs FA4 page16 oracle | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|
| 8B tree w7/b64 + accepted-path reverify | `jetspec/runs/job2_fa4p16_tree_w7_b64_reverify_flush_31814.json` | PASS, 10/10 | 3.2339 | 138.73 |
| 8B tree w7/b128 + accepted-path reverify | `jetspec/runs/job2_fa4p16_tree_w7_b128_reverify_flush_31815.json` | PASS, 10/10 | 3.2728 | 138.04 |

Notes:
- The no-reverify speed number is invalid as a benchmark because the output is not lossless and includes token-0 divergence in several prompts.
- The retained-reverify gates are intentionally slower than linear; they are correctness checks on the canonical backend before Job 3 MT-bench, not the final benchmark verdict.

## Realignment Job 3 - FA4 Page16 MT-bench

Date/time: 2026-06-28 04:36-04:59 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend fa4 --page-size 16`
- Normal greedy mode: canonical driver sends `temperature=0`
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph left at the normal `tc_piecewise` default.
- Harness: `benchmark/mtbench/bench_sglang_eagle.py`, 80 MT-bench questions, `--parallel 1`, `max_new_tokens=2048`.
- Tree pass: dense w7/b64 with the retained accepted-path reverify. The benchmark process disabled only `sglang.global_config.global_config.enable_precache_with_tracing`, matching the earlier tree MT-bench methodology; server-side cache behavior and generation path stayed normal.

Artifacts:

| run | result artifact | answers | accept length | throughput | latency | speed vs linear |
|---|---|---:|---:|---:|---:|---:|
| 8B linear width=1 | `jetspec/runs/mtbench_fa4p16_linear_31820_result.jsonl` | 80/80 | 4.070 | 763.948 tok/s | 334.886 s | 1.00x |
| 8B tree w7/b64 + retained accepted-path reverify | `jetspec/runs/mtbench_fa4p16_tree_w7_b64_reverify_noprecache_31821_result.jsonl` | 80/80 | 3.695 | 199.983 tok/s | 1279.286 s | 0.26x |

Answer files:
- `jetspec/runs/mtbench_fa4p16_linear_31820_answers.jsonl`: 80 rows
- `jetspec/runs/mtbench_fa4p16_tree_w7_b64_reverify_noprecache_31821_answers.jsonl`: 80 rows
- `jetspec/runs/mtbench_question.jsonl`: 80 rows

Raw result lines:

```json
{"task": "mtbench", "backend": "srt", "num_gpus": 1, "latency": 334.886, "throughput": 763.948, "accept_length": 4.07, "num_requests": 80, "other": {"num_questions": 80, "parallel": 1}}
{"task": "mtbench", "backend": "srt", "num_gpus": 1, "latency": 1279.286, "throughput": 199.983, "accept_length": 3.695, "num_requests": 80, "other": {"num_questions": 80, "parallel": 1}}
```

Verdict:
- On DFlash's canonical dense backend, dense tree does not beat or match width=1 linear.
- Tree is `0.26x` linear throughput, or `3.82x` slower wall-clock, with a throughput gap of `563.965 tok/s`.
- The aligned FA4 page16 tree run also did not improve acceptance on MT-bench: `3.695` vs linear `4.070` (`0.91x`).
- The remaining measured cost is the 64-node FA4 custom-mask target verify plus the accepted-branch reverify and tree/draft bookkeeping. Because Job 2 showed FA4 page16 tree verify is still not causal-exact for accepted-path state, the accepted-path reverify remains required.
- MoE was not changed. If the dense FA4 causal-equivalence issue is solved later, the MoE equivalent follow-up remains target `trtllm_mha` verify plus direct GDN-state/KV commit, not a change to the DFlash draft backend fallback.

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

- MoE direct exact KV/GDN-state commit remains a follow-up. It was not attempted in this pass.

## Job A1 Efficient Dense Verify

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend flashinfer`
- Normal greedy mode: no deterministic flags, harness `temperature=0`
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph disabled during this short gate to keep launch/validation tight.
- Harness: `PYTHONPATH=python python jetspec/run_fixed_prompts.py`, 10 fixed prompts, `max_new_tokens=96`, `--flush-cache-before-run --flush-cache-between-prompts`

Artifacts:

| run | artifact | token exact vs fresh flushed oracle | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|
| 8B linear width=1 flushed oracle | `jetspec/runs/a1_fresh_linear_oracle_all_flush_31712.json` | oracle | 3.6342 | 531.98 |
| 8B compact tree w7/b64 + branch reverify | `jetspec/runs/a1_dense_compact_reverify_w7_b64_all_flush_31711.json` | PASS, 10/10 | 4.3495 | 212.57 |

Verify-position count:
- Before A1, dense FlashInfer tree verify defaulted to expanded-causal rows. For w7/b64 with block size 16, the target verify graph processed `64 * 16 = 1024` positions/request.
- After A1, dense tree verify defaults to the custom-mask compact graph. The captured target verify graph reported `num_tokens_per_bs=64`, `rows_per_request=1`, so the tree pass processes `64` positions/request. The lossless commit path adds one causal accepted-branch reverify of `16` positions/request, for roughly `80` target positions/request.
- Position-count drop for w7/b64: `1024 -> ~80` positions/request, about `12.8x` fewer target verify positions.

Notes:
- Initial custom-mask-only runs reproduced the prior drift: prompt 5 passed after branch reverify, and prompt 9 exposed that unflushed width=1 output can itself differ when radix cache supplies a shared prefix. The A1 losslessness gate therefore uses a fresh flushed oracle per prompt.
- This is an efficiency win against the expanded verifier, not a full throughput win over linear. Fixed-prompt compact w7/b64 improved over prior expanded w7/b64 throughput (`~130 tok/s` in Job 3/perf sweep) to `212.57 tok/s`, but remains `0.40x` the flushed linear oracle.

Local checks after A1:
- `PYTHONPATH=python python -m py_compile python/sglang/srt/speculative/dflash_worker_v2.py python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`: PASS
- `PYTHONPATH=python python test/registered/unit/spec/test_dflash_tree_construction.py`: PASS, 18 tests
- `git diff --check`: PASS

## Job B Canonical MT-bench

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend flashinfer`
- Normal greedy mode: canonical driver sets `temperature=0`
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph left in normal default `tc_piecewise` mode.
- Harness: `benchmark/mtbench/bench_sglang_eagle.py`, 80 MT-bench questions, `--parallel 1`, `max_new_tokens=2048`.

Canonical-driver note:
- The unmodified driver succeeded for width=1 linear, but tree mode rejected the driver's frontend traced-prefix-cache warmup before generation because that warmup sends a cache-prefix request without the benchmark's greedy `temperature=0` sampling params.
- The fair pair below disables only `sglang.global_config.global_config.enable_precache_with_tracing` before calling the canonical driver. Server-side radix cache remains enabled, and generation still uses the same canonical MT-bench request path.

Artifacts:

| run | artifact | num questions | accept length | throughput | latency | speed vs linear |
|---|---|---:|---:|---:|---:|---:|
| 8B linear width=1 | `jetspec/runs/mtbench_a1_linear_noprecache_31722_result.jsonl` | 80 | 4.11 | 774.91 tok/s | 332.93 s | 1.00x |
| 8B compact tree w7/b64 + branch reverify | `jetspec/runs/mtbench_a1_tree_w7_b64_noprecache_31723_result.jsonl` | 80 | 5.08 | 283.90 tok/s | 909.50 s | 0.37x |

Answer files:
- `jetspec/runs/mtbench_a1_linear_noprecache_31722_answers.jsonl`: 80 rows
- `jetspec/runs/mtbench_a1_tree_w7_b64_noprecache_31723_answers.jsonl`: 80 rows
- `jetspec/runs/mtbench_question.jsonl`: 80 rows

Verdict:
- Longer canonical outputs do amortize some fixed overhead and improve tree's measured accept length to `5.08`, which is `1.24x` the linear accept length.
- The efficient compact verify still does not make tree beat or match linear. Tree is `2.73x` slower end-to-end on MT-bench (`283.90` vs `774.91 tok/s`).
- The remaining measured cost is no longer the expanded `tree_budget * block_size` verify blowup. For w7/b64 that is fixed from `1024` target positions/request to about `80`. The remaining cost is the extra 64-node compact target verify, the 16-token causal accepted-branch reverify required for lossless commit, and tree/draft bookkeeping that still dominates the accept-length gain.

## Direct-Commit Causal-Equivalence Probe

Date/time: 2026-06-28 03:30-03:55 UTC.

Goal:
- Make the dense accepted tree-verify state causal-exact so DFlash can drop the dense accepted-path reverify and directly gather accepted KV, matching EAGLE's no-reverify tree-commit model.
- Scope was dense `Qwen/Qwen3-8B` only. MoE direct commit was not changed.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend flashinfer`
- Normal greedy mode: no deterministic flags, harness `temperature=0`
- CUDA graph enabled for the gate: `--cuda-graph-max-bs-decode 1`, decode backend `full`
- Harness: `PYTHONPATH=python python jetspec/run_fixed_prompts.py`, 10 fixed prompts, `max_new_tokens=96`, `--flush-cache-before-run --flush-cache-between-prompts`

Fresh flushed oracle:
- `jetspec/runs/job1_fresh_linear_all_w1_31747.json`: mean accept `3.7569`, aggregate `516.20 tok/s`.

Direct-commit gate:

| run | artifact | token exact vs fresh flushed oracle | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|
| 8B compact tree w7/b64, dense reverify disabled | `jetspec/runs/job1_compactkv_direct_commit_w7_b64_all_flush_31748.json` | FAIL, 6/10 mismatched | 4.5164 | 245.73 |

Mismatch notes:
- The first mismatching prompt diverged at token index `17`, expected `... 17,18,14 ...` and produced `... 17,19,14 ...`.
- Other first diffs were prompt 3 at `51`, prompt 5 at `36`, prompt 6 at `32`, prompt 8 at `57`, and prompt 9 at `14`.
- Throughput from this invalid run is not a usable speed result. It is reported only to show that the no-reverify path did run and failed the losslessness gate.

Per-layer causal-equivalence diagnostic:
- Instrumented the compact FlashInfer w7/b64 path to compare accepted-path tree hidden/K/V against a clean width=1 causal forward of the same accepted tokens.
- Diagnostic log: `jetspec/logs/job1_compactkv_exactlen_nograph_compare_w7_b64_31751_server.log`.
- Representative step: prefix `[6]`, commit length `[2]`, accepted local nodes `[[0,3]]`, branch candidates `[[12095,13]]`.
- Tree and branch logits still agreed on the next-token predictions for that step: `tree_predict=[[13,576]]`, `branch_predict=[[13,576]]`.
- The final accepted-path hidden delta was large: `hidden_max_abs=8.0`.
- The first per-layer hidden discrepancy was already at layer 0: `first_layer_hidden_delta=(0, 0.0078125)`.
- Layer-0 K/V were still identical: `kv_max_abs` for layer 0 was `(0.0, 0.0)`.
- Layer-1 K/V then diverged, for example `(0.03125, 0.0009765625)`, and deeper hidden/KV deltas grew from there.

Conclusion:
- The first observed discrepancy is not position IDs or RoPE/K/V projection. Those are consistent through layer-0 K/V.
- The first observed discrepancy is the layer-0 attention output in the target verifier path. That discrepancy is enough to propagate into layer-1 K/V and eventually flip tokens under direct commit.
- Dense direct commit is therefore still unsafe. The accepted-path reverify remains required for token-exact dense output until the target tree-verify attention path is made causal-exact or replaced with a one-forward verifier that produces identical accepted-path attention output.
- Job 2 was not landed and no commit was made for no-reverify direct commit. Job 3 canonical no-reverify MT-bench was not run because the fresh flushed losslessness gate failed.

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
