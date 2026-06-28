# JetSpec DFlash live GPU validation

Date: 2026-06-27 UTC

Hardware/env:
- GPU: `CUDA_VISIBLE_DEVICES=7` (`NVIDIA B300 SXM6 AC`, SM100, 275 GB)
- Required env used for every server: `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Repo/env: local develop install, `python -m sglang.launch_server`, `PYTHONPATH=python`
- Decode settings: BF16/default dtype, greedy `temperature=0`, `--tp-size 1`, `max_new_tokens=96`
- Harness: `jetspec/run_fixed_prompts.py`, 10 fixed prompts including GSM-style arithmetic, sequence, code, translation, and summary prompts.

## Mask root-cause Job 1 FA4 page16 attended-set dump

Date/time: 2026-06-28 05:13-05:20 UTC.

Mode:
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`.
- Canonical backend/page: `--attention-backend fa4 --page-size 16`.
- Tree configuration: `--speculative-dflash-tree-width 7 --speculative-dflash-tree-budget 64`.
- Diagnostic server: accepted-path reverify disabled; causal layer/KV compare and attended-set dump enabled.

Artifact:
- `jetspec/logs/job1_maskdump_fa4p16_tree_w7_b64_31932_server.log`

Exact finding:
- First failing accepted path: `prefix=[6]`, `commit_lens=[2]`, accepted local nodes `[[0, 3]]`.
- DFlash's generated custom tree mask for accepted node `3` is correct: prefix `[0,6)`, expected/allowed tree cols `[0, 3]`, no extras, no missing, physical tree slots `[22, 25]`.
- FA4's effective verifier is wrong: `effective_mode=linear_causal_no_custom_mask`, allowed tree cols for node `3` are `[0, 1, 2, 3]`.
- Wrong extras: non-ancestor tree cols `[1, 2]`, logical KV positions `[7, 8]`, physical slots `[23, 24]`.
- Missing positions: none.
- The same step reproduced the existing diagnostic signature: `first_layer_hidden_delta=(0, 0.1357421875)` with layer-0 K/V exact, `kv_max_abs=(0.0, 0.0)`.

Root cause:
- DFlash passes `DFlashVerifyInput.topk=7` plus a 64-node ancestor mask, but `FlashAttentionBackend` chooses its target-verify custom-mask/cascade path from backend `self.topk`, initialized from `server_args.speculative_eagle_topk=1`.
- EAGLE works because its server-side `speculative_eagle_topk` drives the FA4 FULL_MASK path.
- DFlash therefore falls through to FA4's `topk<=1` causal verifier, which ignores the custom mask and treats the first `--speculative-num-draft-tokens 16` BFS nodes as a linear causal chain.

## Mask root-cause Job 2 FA4 page16 mask fix and no dense reverify

Date/time: 2026-06-28 05:36-06:02 UTC.

Mode:
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`.
- Canonical backend/page: `--attention-backend fa4 --page-size 16`.
- Tree configurations: w7/b64 and w7/b128.
- Normal greedy serving mode: no deterministic flags, harness `temperature=0`.
- CUDA graph enabled for normal decode: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph left at the normal `tc_piecewise` default.
- Fresh flushed oracle: `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json`.

Fix:
- The Job 1 bug was not in DFlash's Python tree mask: DFlash's row for accepted node `3` already allowed only `[0, 3]`.
- The first fix is to make FA4 target verify consult DFlash's per-verify tree shape rather than backend `self.topk`, which is `1` for DFlash.
- FA4 page16 still cannot consume a sparse ancestor set by treating token IDs as page IDs; page-size conversion turns sparse slots into contiguous physical pages.
- Dense FA4 DFlash therefore now supplies exact per-query `compact_kv_indices`: prefix slots followed by the accepted node's ancestor chain ending at self. FA4 reads those gathered K/V rows through varlen attention.
- Dense accepted-path reverify is removed for this compact dense path. Accepted tree KV is committed directly by copying accepted source slots into the canonical next-prefix slots. MoE was left on its current lossless path.

Diagnostic evidence:
- Diagnostic log: `jetspec/logs/job2_compact_direct_diag_nograph_fa4p16_tree_w7_b64_31940_server.log`.
- Original failing node after the fix: `prefix=[6]`, `commit_lens=[2]`, accepted local path `[[0, 3]]`, branch candidates `[[12095, 13]]`.
- Correct causal set: prefix `[0,6)` plus tree cols `[0, 3]`.
- Compact verifier set: allowed tree cols `[0, 3]`, extras `[]`, missing `[]`, physical tree slots `[22, 25]`.
- Layer-0 signature: `first_layer_hidden_delta=(0, 0.001953125)` and layer-0 `kv_max_abs=(0.0, 0.0)`. This removes the prior mask-scale layer-0 delta `0.1357421875`.
- The later-layer hook/KV list is not a clean bf16 equivalence proof because compact FA4 varlen and the clean paged causal replay use different kernel shapes; token-exact flushed serving is the losslessness gate for this milestone.

Fixed-prompt losslessness gates:

| run | artifact | token exact vs FA4 page16 oracle | mismatches | mean accept length | aggregate tok/s | speed vs oracle |
|---|---|---:|---:|---:|---:|---:|
| 8B linear width=1 flushed oracle | `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json` | oracle | 0 | 3.5827 | 561.04 | 1.00x |
| 8B tree w7/b64, no dense accepted-path reverify | `jetspec/runs/job2_maskfix_noreverify_fa4p16_tree_w7_b64_flush_31941.json` | PASS | 0/10 | 4.2004 | 217.44 | 0.39x |
| 8B tree w7/b128, no dense accepted-path reverify | `jetspec/runs/job2_maskfix_noreverify_fa4p16_tree_w7_b128_flush_31942.json` | PASS | 0/10 | 4.7738 | 181.38 | 0.32x |

Verdict:
- The attended-set root cause is fixed for dense FA4 page16: accepted nodes no longer see sibling/cousin KV positions.
- Dense accepted-path reverify is gone in the compact FA4 dense path and the fresh flushed oracle gates pass for both required budgets.
- Short fixed-prompt throughput is still well below linear because compact tree verify is eager and performs a ragged prefix+ancestor gather for every tree node. Job 3 must measure the real MT-bench tradeoff.

## Caveats

- `--attention-backend fa3` is not usable on this B300/SM100 host in this checkout: launch fails with `FlashAttention v3 Backend requires SM>=80 and SM<=90. Please use --attention-backend flashinfer.` The 8B smoke pair was therefore run with `--attention-backend flashinfer`.
- Port `30000` was already occupied by an unrelated server reporting `nvidia/GLM-5.2-NVFP4`, so the MoE runs used ports `31106`, `31107`, and `31108`.
- The user's original MoE command already had the invalid `--speculative-draft-attention-backend fa4` removed. No working draft-specific FA4 equivalent exists in this checkout.
- The MoE command needed one additional required flag on SM100: `--mamba-ssm-dtype bfloat16`. Without it, validation fails with `--linear-attn-decode-backend flashinfer on SM100+ requires --mamba-ssm-dtype bfloat16, got None`.
- The MoE server accepts `--attention-backend trtllm_mha` for the target, but the DFlash draft worker logs: `DFLASH draft worker does not support 'trtllm_mha' because the draft path requires per-layer DFlash attention. Falling back to 'flashinfer'.`
- `z-lab/Qwen3.6-35B-A3B-DFlash` config was fetched under `jetspec/_hf_configs/` and confirmed `architectures: ["DFlashDraftModel"]`.

## Realignment Job 1 FA4 page16 linear baseline

Date/time: 2026-06-28 04:11-04:13 UTC.

Mode:
- Normal greedy serving mode: no deterministic flags, harness `temperature=0`.
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph left at the normal `tc_piecewise` default.
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`.
- Canonical backend/page: `--attention-backend fa4 --page-size 16`.
- Harness: `jetspec/run_fixed_prompts.py`, 10 prompts, `max_new_tokens=96`, `--flush-cache-before-run --flush-cache-between-prompts`.

Launch correction:
- A pre-patch foreground launch proved the existing FA4 non-MLA guard rewrote `--page-size 16` to runtime `page_size=128`, which would have recreated the non-canonical baseline problem.
- `python/sglang/srt/server_args.py` now keeps explicit DFlash FA4 page16 while leaving the generic FA4 non-MLA page128 rewrite in place.
- The accepted launch log `jetspec/logs/job1_fa4p16_linear_w1_31811_server.log` shows `page_size=16`, `attention_backend='fa4'`, DFlash block size `16`, and target verify graph capture with `num_tokens_per_bs=16`.

Fixed-prompt result:

| run | artifact | backend/page | mean accept length | aggregate tok/s | mean prompt tok/s |
|---|---|---|---:|---:|---:|
| 8B linear width=1 flushed oracle | `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json` | `fa4`, `page_size=16` | 3.5827 | 561.04 | 687.74 |

Raw summary:

```json
{"aggregate_tok_per_s": 561.0427603854891, "mean_accept_length": 3.5827345795966488, "mean_per_prompt_tok_per_s": 687.7366106441112, "num_prompts": 10, "total_completion_tokens": 952, "total_e2e_latency": 1.696840360877104}
```

Verdict:
- Linear DFlash is healthy on the aligned FA4 page16 backend.
- This fresh flushed oracle replaces the old flashinfer/page_size=1 oracles for the dense tree losslessness gates in the realignment pass.

## Realignment Job 2 FA4 page16 dense tree verify

Date/time: 2026-06-28 04:18-04:27 UTC.

Mode:
- Normal greedy serving mode: no deterministic flags, harness `temperature=0`.
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph left at the normal `tc_piecewise` default.
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`.
- Canonical backend/page: `--attention-backend fa4 --page-size 16`.
- Harness: `jetspec/run_fixed_prompts.py`, 10 prompts, `max_new_tokens=96`, `--flush-cache-before-run --flush-cache-between-prompts`.
- Fresh flushed oracle: `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json`.

Shape result:
- The dense FA4 tree path is already the multi-token custom-mask path. The FlashInfer compact/expanded branches are only active for `FlashInferAttnBackend`; `fa4` uses `FlashAttentionBackend`.
- The target verify graph captures confirm the aligned shape:
  - `w7/b64`: `num_tokens_per_bs=64`, `rows_per_request=1`, log `jetspec/logs/job2_fa4p16_tree_w7_b64_reverify_31814_server.log`.
  - `w7/b128`: `num_tokens_per_bs=128`, `rows_per_request=1`, log `jetspec/logs/job2_fa4p16_tree_w7_b128_reverify_31815_server.log`.

No-reverify direct-commit result:

| run | artifact | token exact vs FA4 page16 oracle | mismatches | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|---:|
| 8B tree w7/b64, accepted-path reverify disabled | `jetspec/runs/job2_fa4p16_tree_w7_b64_noreverify_flush_31812.json` | FAIL | 10/10 | 2.9558 | 281.32 |

Per-layer diagnostic:
- Diagnostic artifact: `jetspec/logs/job2_fa4p16_tree_w7_b64_noreverify_compare_31813_server.log`.
- Representative compare: prefix `[6]`, commit length `[2]`, accepted local nodes `[[0,3]]`, branch candidates `[[12095,13]]`.
- Tree and clean causal branch predictions agreed at that step: `tree_predict=[[13,576]]`, `branch_predict=[[13,576]]`.
- Accepted-path hidden diverged: `hidden_max_abs=40.0`.
- First per-layer hidden mismatch was layer 0: `first_layer_hidden_delta=(0, 0.1357421875)`.
- Layer-0 K/V were exact: layer-0 `kv_max_abs=(0.0, 0.0)`.
- Layer-1 K/V then diverged, for example `(2.75, 0.03076171875)`, after the layer-0 hidden delta propagated.

Retained-reverify correctness gates:

| run | artifact | token exact vs FA4 page16 oracle | mean accept length | aggregate tok/s | speed vs FA4 page16 linear |
|---|---|---:|---:|---:|---:|
| 8B linear width=1 flushed oracle | `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json` | oracle | 3.5827 | 561.04 | 1.00x |
| 8B tree w7/b64 + accepted-path reverify | `jetspec/runs/job2_fa4p16_tree_w7_b64_reverify_flush_31814.json` | PASS | 3.2339 | 138.73 | 0.25x |
| 8B tree w7/b128 + accepted-path reverify | `jetspec/runs/job2_fa4p16_tree_w7_b128_reverify_flush_31815.json` | PASS | 3.2728 | 138.04 | 0.25x |

Job 2 verdict:
- FA4 page16 tree verify did not make the accepted-path state causal-exact. The accepted-path reverify cannot be dropped.
- The residual is not DFlash draft K/V materialization, positions, or RoPE: layer-0 K/V are exact, and the first mismatch is the layer-0 attention output.
- The remaining correctness issue is the FA4 tree/custom-mask target-verify execution shape for this DFlash tree path.
- The retained-reverify path remains lossless on the canonical backend for both w7/b64 and w7/b128, but it is roughly `0.25x` the fixed-prompt linear throughput. Job 3 must therefore benchmark the honest aligned configuration: FA4 page16 tree with retained reverify.

## Realignment Job 3 FA4 page16 canonical MT-bench

Date/time: 2026-06-28 04:36-04:59 UTC.

Mode:
- Canonical SGLang spec-decode benchmark: `benchmark/mtbench/bench_sglang_eagle.py`, MT-bench 80 questions, `--parallel 1`, `max_new_tokens=2048`.
- Normal greedy serving mode: the benchmark sends `temperature=0`.
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; dense prefill graph left at normal default `tc_piecewise`.
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`.
- Canonical backend/page: `--attention-backend fa4 --page-size 16`.
- Tree configuration: `--speculative-dflash-tree-width 7 --speculative-dflash-tree-budget 64`.
- Tree correctness mode: retained accepted-path reverify. The no-reverify direct commit from Job 2 did not pass losslessness and was not benchmarked as a valid speed result.
- The tree benchmark process disabled only `sglang.global_config.global_config.enable_precache_with_tracing`, as in the earlier canonical tree pass. The linear pass used the unmodified benchmark driver.

Launch knobs:

```bash
--speculative-algorithm DFLASH \
--speculative-draft-model-path JetSpec/jetspec-qwen3-8b \
--speculative-num-draft-tokens 16 \
--reasoning-parser qwen3 \
--attention-backend fa4 \
--page-size 16 \
--tp-size 1 --mem-fraction-static 0.8 --trust-remote-code \
--max-running-requests 1 \
--cuda-graph-max-bs-decode 1 \
--cuda-graph-backend-decode full
```

Canonical results:

| run | result artifact | answers | accept length | throughput | latency | speed vs linear |
|---|---|---:|---:|---:|---:|---:|
| 8B linear width=1 | `jetspec/runs/mtbench_fa4p16_linear_31820_result.jsonl` | 80/80 | 4.070 | 763.948 tok/s | 334.886 s | 1.00x |
| 8B tree w7/b64 + retained accepted-path reverify | `jetspec/runs/mtbench_fa4p16_tree_w7_b64_reverify_noprecache_31821_result.jsonl` | 80/80 | 3.695 | 199.983 tok/s | 1279.286 s | 0.26x |

Raw result lines:

```json
{"task": "mtbench", "backend": "srt", "num_gpus": 1, "latency": 334.886, "throughput": 763.948, "accept_length": 4.07, "num_requests": 80, "other": {"num_questions": 80, "parallel": 1}}
{"task": "mtbench", "backend": "srt", "num_gpus": 1, "latency": 1279.286, "throughput": 199.983, "accept_length": 3.695, "num_requests": 80, "other": {"num_questions": 80, "parallel": 1}}
```

Benchmark verdict:
- On DFlash's real dense backend (`fa4`, `page_size=16`), dense tree does not beat or match linear.
- Tree reaches only `0.26x` linear throughput, or `3.82x` slower wall-clock (`199.983` vs `763.948 tok/s`).
- Unlike the earlier FlashInfer compact benchmark, FA4 page16 dense tree does not show an acceptance gain on MT-bench: `3.695` vs linear `4.070`, or `0.91x` linear acceptance.
- The remaining cost is the 64-node FA4 custom-mask target verify plus the accepted-path causal reverify and tree/draft bookkeeping. Job 2 showed the FA4 page16 custom-mask verify state is not causal-exact for direct commit, so the accepted-path reverify cannot be removed without breaking losslessness.
- MoE was left untouched. The equivalent future MoE direction remains target `trtllm_mha` verify plus direct GDN-state/KV commit if the dense FA4 causal-equivalence problem is solved.

## Job A1 compact dense verify

Date/time: 2026-06-28 02:16-02:24 UTC.

Mode:
- Normal greedy serving mode: no deterministic flags, harness `temperature=0`.
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; prefill graph disabled for this short fixed-prompt gate.
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`, `--attention-backend flashinfer`.
- Harness: `jetspec/run_fixed_prompts.py`, 10 prompts, `max_new_tokens=96`, `--flush-cache-before-run --flush-cache-between-prompts`.
- Fresh flushed oracle for this pass: `jetspec/runs/a1_fresh_linear_oracle_all_flush_31712.json`.

Change:
- Dense FlashInfer tree verify now defaults away from the expanded-causal verifier. `SGLANG_DFLASH_TREE_EXPANDED_CAUSAL=1` remains the opt-in escape hatch.
- CUDA graph capture uses the same default, so w7/b64 captured `num_tokens_per_bs=64`, `rows_per_request=1`.
- The compact tree logits/KV are still not assumed to be bit-exact causal branch state. The accepted branch is causally reverified for one DFlash block before computing the final accept/bonus/commit state.

Fixed-prompt results:

| run | artifact | token exact vs fresh flushed linear | mean accept length | aggregate tok/s | speed vs flushed linear |
|---|---|---:|---:|---:|---:|
| 8B linear width=1 flushed oracle | `jetspec/runs/a1_fresh_linear_oracle_all_flush_31712.json` | oracle | 3.6342 | 531.98 | 1.00x |
| 8B expanded tree w7/b64 before A1 | `jetspec/runs/perf1_sweep_8b_tree_w7_b64_cudagraph_31631.json` | PASS in prior gate | 4.5441 | 130.57 | 0.25x |
| 8B compact tree w7/b64 + branch reverify | `jetspec/runs/a1_dense_compact_reverify_w7_b64_all_flush_31711.json` | PASS | 4.3495 | 212.57 | 0.40x |

Verify-position count:
- Before A1, dense expanded-causal w7/b64 processed `tree_budget * block_size = 64 * 16 = 1024` target verify positions/request.
- After A1, dense compact w7/b64 processes `64` target tree-verify positions/request plus one `16`-position causal branch reverify, for roughly `80` target positions/request.
- That is a `12.8x` verify-position reduction for w7/b64. The short fixed-prompt throughput improved by about `1.63x` versus the prior expanded w7/b64 artifact, but still trails linear by about `2.5x`.

Losslessness notes:
- A custom-mask-only diagnostic reproduced the prior verifier-equivalence issue; prompt 5 and prompt 9 were used as drift probes.
- Prompt 9 also exposed a pre-existing comparison hazard: unflushed width=1 DFlash can produce a different output when radix cache supplies a shared prefix. The token-exact A1 gate therefore compares both width=1 and tree with cache flushed before each prompt, matching the "fresh deterministic oracle" requirement.

Verdict:
- A1 landed the main efficiency lever for dense verify: the target verify graph is compact and lossless with branch reverify.
- It is not enough by itself to beat linear on the short 96-token harness. The remaining measured cost is the extra 64-node tree verify, the 16-token branch reverify, and tree/draft bookkeeping overhead. Job B must use the canonical MT-bench to see whether longer outputs amortize this enough to close the gap.

## Job B canonical MT-bench after compact verify

Date/time: 2026-06-28 02:47-03:03 UTC.

Mode:
- Canonical SGLang spec-decode benchmark: `benchmark/mtbench/bench_sglang_eagle.py`, MT-bench 80 questions, `--parallel 1`, `max_new_tokens=2048`.
- Normal greedy serving mode: the benchmark sends `temperature=0`.
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`; dense prefill graph left at normal default `tc_piecewise`.
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`, `--attention-backend flashinfer`.
- Server-side cache behavior is normal. The only benchmark-side adjustment was disabling `sglang.global_config.global_config.enable_precache_with_tracing` before calling the canonical driver, because tree mode correctly rejects the frontend cache-prefix warmup as non-greedy when it is sent without the benchmark's `temperature=0` params.

Launch knobs:

```bash
--speculative-algorithm DFLASH \
--speculative-draft-model-path JetSpec/jetspec-qwen3-8b \
--speculative-num-draft-tokens 16 \
--reasoning-parser qwen3 \
--attention-backend flashinfer \
--tp-size 1 --mem-fraction-static 0.8 --trust-remote-code \
--max-running-requests 1 \
--cuda-graph-max-bs-decode 1 \
--cuda-graph-backend-decode full
```

Tree adds:

```bash
--speculative-dflash-tree-width 7 \
--speculative-dflash-tree-budget 64
```

Canonical results:

| run | result artifact | answers | accept length | throughput | latency | speed vs linear |
|---|---|---:|---:|---:|---:|---:|
| 8B linear width=1 | `jetspec/runs/mtbench_a1_linear_noprecache_31722_result.jsonl` | 80/80 | 4.11 | 774.91 tok/s | 332.93 s | 1.00x |
| 8B compact tree w7/b64 + branch reverify | `jetspec/runs/mtbench_a1_tree_w7_b64_noprecache_31723_result.jsonl` | 80/80 | 5.08 | 283.90 tok/s | 909.50 s | 0.37x |

Raw result lines:

```json
{"task": "mtbench", "backend": "srt", "num_gpus": 1, "latency": 332.933, "throughput": 774.908, "accept_length": 4.11, "num_requests": 80, "other": {"num_questions": 80, "parallel": 1}}
{"task": "mtbench", "backend": "srt", "num_gpus": 1, "latency": 909.503, "throughput": 283.903, "accept_length": 5.08, "num_requests": 80, "other": {"num_questions": 80, "parallel": 1}}
```

Benchmark verdict:
- Tree's accept length is real and better on the canonical benchmark: `5.08` vs linear `4.11`, a `1.24x` acceptance gain.
- The compact verifier still does not make tree beat or match linear. Tree reaches only `0.37x` linear throughput, or `2.73x` slower wall-clock, despite the 12.8x verify-position reduction from A1.
- The expanded-causal verify blowup is no longer the dominant measured cost for w7/b64. Remaining costs are the extra 64-node compact target verify, the 16-token causal accepted-branch reverify needed for token-exact commit, and tree/draft bookkeeping. The next real lever is removing or shrinking those residual costs without losing causal-branch equivalence; otherwise the current accept gain is not enough to pay for tree verification.

## Dense direct-commit causal-equivalence probe

Date/time: 2026-06-28 03:30-03:55 UTC.

Goal:
- Test whether dense compact tree verify can be made causal-exact enough to remove the accepted-path reverify and directly gather accepted KV, EAGLE-style.
- Scope was dense `Qwen/Qwen3-8B` only. MoE was not changed because its persistent GDN/conv state still requires replay/fallback.

Mode:
- Normal greedy serving mode, harness `temperature=0`.
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`, `--attention-backend flashinfer`.
- Gate used CUDA graph on: `--cuda-graph-max-bs-decode 1`, decode backend `full`.
- Harness: `jetspec/run_fixed_prompts.py`, 10 prompts, `max_new_tokens=96`, `--flush-cache-before-run --flush-cache-between-prompts`.

Fresh flushed oracle:

| run | artifact | mean accept length | aggregate tok/s |
|---|---|---:|---:|
| 8B linear width=1 | `jetspec/runs/job1_fresh_linear_all_w1_31747.json` | 3.7569 | 516.20 |

No-reverify direct-commit gate:

| run | artifact | token exact vs fresh flushed oracle | mismatches | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|---:|
| 8B compact tree w7/b64, dense reverify disabled | `jetspec/runs/job1_compactkv_direct_commit_w7_b64_all_flush_31748.json` | FAIL | 6/10 | 4.5164 | 245.73 |

Mismatch details:
- Prompt 0 first differed at token index `17`: expected `... 17,18,14 ...`, actual `... 17,19,14 ...`.
- Other first diffs: prompt 3 at `51`, prompt 5 at `36`, prompt 6 at `32`, prompt 8 at `57`, prompt 9 at `14`.
- The `245.73 tok/s` number is not a valid speedup result because the output is not lossless.

Per-layer diagnostic:
- Instrumented the accepted path in the compact FlashInfer tree verify and compared hidden/K/V against a clean causal branch forward of the exact same accepted tokens.
- Diagnostic artifact: `jetspec/logs/job1_compactkv_exactlen_nograph_compare_w7_b64_31751_server.log`.
- Representative step: prefix `[6]`, commit length `[2]`, accepted local nodes `[[0,3]]`, branch candidates `[[12095,13]]`.
- Tree and branch next-token predictions still matched for that step: `tree_predict=[[13,576]]`, `branch_predict=[[13,576]]`.
- Final hidden state mismatch was large: `hidden_max_abs=8.0`.
- First per-layer hidden mismatch appeared at layer 0: `first_layer_hidden_delta=(0, 0.0078125)`.
- Layer-0 K/V were identical: layer-0 `kv_max_abs=(0.0, 0.0)`.
- Layer-1 K/V then diverged, for example `(0.03125, 0.0009765625)`, and later layers amplified the difference.

Verdict:
- Position IDs, RoPE, and the layer-0 K/V projection are not the first failing point. The first observed discrepancy is the layer-0 attention output produced by the target verifier path.
- Because this discrepancy flips tokens under the fresh flushed gate, the dense accepted-path reverify cannot be removed yet.
- No canonical MT-bench no-reverify run was performed. The no-reverify path failed correctness before it reached the benchmark stage.
- Current honest performance baseline remains the canonical compact+reverify table above: linear `774.91 tok/s`, compact tree w7/b64 `283.90 tok/s`, tree `2.73x` slower despite accept length `5.08` vs `4.11`.

## Job 3 final CUDA graph perf

Date/time: 2026-06-27 23:41-23:49 UTC.

Mode:
- Normal greedy serving mode: no deterministic flags, `temperature=0` in the harness.
- CUDA graph enabled: decode graph backend `full`; dense prefill graph used the default `tc_piecewise`, while MoE prefill graph auto-disabled and target/draft verify decode graphs captured.
- Single-request fixed-prompt harness: `jetspec/run_fixed_prompts.py`, 10 prompts, `max_new_tokens=96`, `max_running_requests=1`, `cuda_graph_max_bs_decode=1`.
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`, `--attention-backend flashinfer`.
- MoE target: `Qwen/Qwen3.6-35B-A3B`, draft: `z-lab/Qwen3.6-35B-A3B-DFlash`, `--attention-backend trtllm_mha`, `--linear-attn-decode-backend cutedsl`, `--linear-attn-prefill-backend cutedsl`, `--mamba-ssm-dtype bfloat16`.

Dense results:

| run | artifact | token exact vs fresh graph-linear | mean accept length | aggregate tok/s | speed vs linear |
|---|---|---:|---:|---:|---:|
| 8B linear width=1 | `jetspec/runs/job3_8b_linear_cudagraph_31530.json` | oracle | 3.7569 | 510.53 | 1.00x |
| 8B tree w7/b64 | `jetspec/runs/job3_8b_tree_w7_b64_cudagraph_31531.json` | PASS | 4.5441 | 131.52 | 0.26x |
| 8B tree w7/b128 | `jetspec/runs/job3_8b_tree_w7_b128_cudagraph_31532.json` | PASS | 5.2161 | 85.69 | 0.17x |

MoE results:

| run | artifact | token exact vs fresh graph-linear | mean accept length | aggregate tok/s | speed vs linear |
|---|---|---:|---:|---:|---:|
| MoE linear width=1 | `jetspec/runs/job3_moe_linear_cudagraph_31540.json` | oracle | 4.4179 | 499.06 | 1.00x |
| MoE tree w4/b64 | `jetspec/runs/job2_moe_tree_w4_b64_cudagraph_31520.json` | PASS | 4.4179 | 80.64 | 0.16x |
| MoE tree w4/b128 | `jetspec/runs/job2_moe_tree_w4_b128_cudagraph_31521.json` | PASS | 4.4179 | 66.40 | 0.13x |
| MoE tree w7/b64 | `jetspec/runs/job2_moe_tree_w7_b64_cudagraph_31522.json` | PASS | 4.4179 | 78.78 | 0.16x |
| MoE tree w7/b128 | `jetspec/runs/job2_moe_tree_w7_b128_cudagraph_31523.json` | PASS | 4.4179 | 67.34 | 0.13x |

Verdict:
- Losslessness is green in normal mode with CUDA graph on. Dense tree w7/b64 and w7/b128 are token-exact against `job3_8b_linear_cudagraph_31530.json`; MoE tree w4/w7, budget 64/128 artifacts are token-exact against `job3_moe_linear_cudagraph_31540.json`.
- Tree does not beat linear on throughput. Dense tree improves mean accept length from 3.7569 to 4.5441/5.2161, but the fixed-shape tree verify cost and accepted-path replay/commit work swamp that gain. MoE tree does not improve accept length at all under the current accepted-path linear replay fallback, so it only adds tree verify overhead.
- The practical bottleneck is not CUDA graph correctness anymore. The next lever is acceptance and amortization: crossproduct draft scoring/width/budget needs to raise dense acceptance much closer to the paper's 7-9 range, and the MoE direct KV/GDN-state commit path would need to replace the current correctness fallback before tree can plausibly beat width=1 linear on tok/s.

## Perf pass: dense replay removal attempt and width/budget sweep

Date/time: 2026-06-28 00:05-00:26 UTC.

Mode:
- Normal greedy serving mode: no deterministic flags, harness `temperature=0`.
- CUDA graph enabled: `--cuda-graph-max-bs-decode 1`, decode backend `full`.
- Dense target: `Qwen/Qwen3-8B`, draft: `JetSpec/jetspec-qwen3-8b`, `--attention-backend flashinfer`.
- Fresh oracle for this pass: `jetspec/runs/perf1_8b_linear_cudagraph_31610.json`.

Lever 1 result:
- Attempted change: disable the dense CUDA-graph accepted-path reverify and commit the accepted expanded-causal tree KV directly by gathering accepted tree slots. The code change was reverted.
- Full 10-prompt result before revert: `jetspec/runs/perf1_8b_tree_w7_b64_direct_commit_cudagraph_31611.json` was faster than Job 3 w7/b64 (`158.50 tok/s` vs `131.52 tok/s`) but failed the losslessness gate (`5/10` prompts mismatched vs `perf1_8b_linear_cudagraph_31610.json`).
- Diagnostic result: `jetspec/runs/perf1_debug_prompt4_tree_w7_b64_direct_commit_compare_31612.json` reproduced prompt 4 divergence at first diff `11`. The built-in causal compare logged `same_shape_hidden_max_abs=0.0` and `same_shape_kv_max_abs=0.0` for early steps, but nonzero causal-branch deltas such as `hidden_max_abs=4.0` and KV deltas through the stack. That means the expanded-causal tree state is internally reproducible, but it is not token-exact equivalent to the accepted causal replay branch.
- Conclusion: direct dense commit is not safe with the current verifier. The accepted-path replay is not just redundant overhead; it is preserving the causal branch state required for token-exact output.

Lever 2 dense CUDA-graph sweep:

| width | budget | artifact | token exact vs `perf1_8b_linear_cudagraph_31610.json` | mean accept length | aggregate tok/s | speed vs linear | accept > linear |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 16 | `jetspec/runs/perf1_8b_linear_cudagraph_31610.json` | oracle | 3.7569 | 511.38 | 1.00x | oracle |
| 2 | 8 | `jetspec/logs/perf1_sweep_8b_tree_w2_b8_cudagraph_31620_server.log` | invalid launch | - | - | - | - |
| 4 | 8 | `jetspec/logs/perf1_sweep_8b_tree_w4_b8_cudagraph_31621_server.log` | invalid launch | - | - | - | - |
| 7 | 8 | `jetspec/logs/perf1_sweep_8b_tree_w7_b8_cudagraph_31622_server.log` | invalid launch | - | - | - | - |
| 2 | 16 | `jetspec/runs/perf1_sweep_8b_tree_w2_b16_cudagraph_31623.json` | PASS | 3.7477 | 173.77 | 0.34x | no |
| 4 | 16 | `jetspec/runs/perf1_sweep_8b_tree_w4_b16_cudagraph_31624.json` | PASS | 3.2309 | 169.18 | 0.33x | no |
| 7 | 16 | `jetspec/runs/perf1_sweep_8b_tree_w7_b16_cudagraph_31625.json` | PASS | 2.9500 | 159.22 | 0.31x | no |
| 2 | 32 | `jetspec/runs/perf1_sweep_8b_tree_w2_b32_cudagraph_31626.json` | PASS | 4.4810 | 155.15 | 0.30x | yes |
| 4 | 32 | `jetspec/runs/perf1_sweep_8b_tree_w4_b32_cudagraph_31627.json` | PASS | 4.1024 | 163.84 | 0.32x | yes |
| 7 | 32 | `jetspec/runs/perf1_sweep_8b_tree_w7_b32_cudagraph_31628.json` | PASS | 3.4931 | 154.34 | 0.30x | no |
| 2 | 64 | `jetspec/runs/perf1_sweep_8b_tree_w2_b64_cudagraph_31629.json` | PASS | 4.5582 | 111.86 | 0.22x | yes |
| 4 | 64 | `jetspec/runs/perf1_sweep_8b_tree_w4_b64_cudagraph_31630.json` | PASS | 4.8405 | 127.94 | 0.25x | yes |
| 7 | 64 | `jetspec/runs/perf1_sweep_8b_tree_w7_b64_cudagraph_31631.json` | PASS | 4.5441 | 130.57 | 0.26x | yes |

Sweep notes:
- `tree_budget=8` is invalid for this draft head because the server requires `--speculative-dflash-tree-budget >= block_size`; the DFlash block size is 16.
- Best dense tree tok/s in the valid sweep: w2/b16 at `173.77 tok/s`, but its mean accept length `3.7477` is slightly below the width=1 linear oracle `3.7569`.
- Best dense tok/s while beating linear accept: w4/b32 at `163.84 tok/s`, mean accept `4.1024`, only `0.32x` the linear oracle throughput.
- Best dense acceptance in the valid sweep: w4/b64 at mean accept `4.8405`, but only `127.94 tok/s` (`0.25x` linear). The prior Job 3 w7/b128 run reached mean accept `5.2161` at `85.69 tok/s`.
- Acceptance ceiling conclusion: no swept config pushes dense acceptance meaningfully above about 5 toward the paper's 7-9 range. With the current draft head and current correctness replay, tree remains throughput-limited by fixed-shape verify plus accepted-path replay/commit overhead.

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

- Current status: enabled for DFlash tree target verify.
- Dense FlashInfer tree verify captures fixed expanded-causal graph buckets: w7/b64 captured `bs=64`, `tokens_per_bs=16`; w7/b128 captured `bs=128`, `tokens_per_bs=16`. Decode logs reported `cuda graph: True`.
- MoE hybrid-GDN tree verify captures fixed custom-mask target-verify graph buckets using a graph-capable `flashinfer` full-attention replacement for periodic full-attention layers. MoE w4/w7, budget 64/128 decode logs reported `cuda graph: True`.
- Prefill graph behavior differs by model in the measured runs: dense used the default `tc_piecewise` prefill graph, while MoE prefill graph was auto-disabled by the resolved cuda graph config. The throughput verdict above is based on the measured end-to-end fixed-prompt artifacts, not launch-time graph capture.

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
