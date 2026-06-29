# DFlash Tree Speculative Decode Status

Updated: 2026-06-29 17:10 UTC

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
- Mask root-cause Job 1 2026-06-28: dense FA4 page16 w7/b64 attended-set dump found that DFlash's generated tree mask is ancestor-exact, but FA4 does not enter the custom-mask target-verify path. For the first failing accepted node `3`, DFlash's custom mask allows only tree cols `[0, 3]`, but FA4's effective `topk<=1` causal verifier attends `[0, 1, 2, 3]`, admitting non-ancestor sibling/cousin cols `[1, 2]` (logical KV positions `[7, 8]`, physical slots `[23, 24]`). Root cause: FA4 gates the cascade/custom-mask verifier on backend `self.topk`, initialized from `speculative_eagle_topk=1`, instead of DFlash's per-verify `DFlashVerifyInput.topk=7` / `custom_mask`.
- Mask root-cause Job 2 2026-06-28: dense FA4 page16 tree verify now uses an exact compact KV index list for each query row, so every accepted node reads exactly committed prefix plus self/ancestors. The dense accepted-path reverify is removed for compact dense target verify, and accepted KV is committed directly by copying accepted tree slots into the canonical next-prefix slots. Fresh flushed FA4 page16 width=1 oracle gates passed for w7/b64 and w7/b128 with zero token mismatches.
- Mask root-cause Job 3 2026-06-28: canonical MT-bench rerun completed with the dense reverify removed. FA4 page16 width=1 linear reached `764.540 tok/s` with accept length `4.070`; dense tree w7/b64 no-reverify reached `243.000 tok/s` with accept length `5.053`. Tree improves over the old retained-reverify tree (`199.983 tok/s`) but is still only `0.32x` linear (`3.15x` slower), so dense tree does not beat or match linear yet.
- Paper-dataset benchmark 2026-06-28: added `jetspec/bench_paper_sglang.py` and measured first-80 GSM8K/MATH-500 prompts with JetSpec prompt formatting. Valid tree b128 reaches paper-level acceptance (`7.77` GSM8K vs paper `7.94`; `9.55` MATH-500 vs paper `9.56`) but is verify-cost-limited (`23.55/25.64 ms/step`, `329.91/372.49 tok/s`). b255 is not lossless (`3/5` GSM8K oracle mismatches, `2/5` MATH oracle mismatches) and is diagnostic only.
- Top2gap construction Job 1 2026-06-29: added `speculative_dflash_tree_draft=top2gap`, per-depth top-2-gap sigmoid fanout caps, and measured tree node-count counters. Fresh flushed FA4 page16 width=1 oracle gates passed for top2gap w4/b64 beta=1.0 g0=1.0 on GSM8K and MATH-500 with zero token mismatches. First-5 accept length improved from `5.33 -> 6.56` on GSM8K and `7.71 -> 9.18` on MATH-500; mean root-inclusive tree nodes were `57.54` and `50.45`.
- Lean top2gap sweep 2026-06-29: swept top2gap-only `width=8`, budgets `16/24/32/48`, `(beta,g0)=(1.0,1.0)` and `(2.0,0.5)` on first-80 GSM8K/MATH-500 with FA4 page16 and normal decode graphs. All 16 dataset rows were token-exact, but none matched linear throughput. Best paired config is `budget=16`, `beta=1.0`, `g0=1.0`: GSM8K `570.64 tok/s`, accept `6.45`, nodes `16.00` (`0.49x` linear); MATH-500 `674.74 tok/s`, accept `7.90`, nodes `16.00` (`0.45x` linear). Best single MATH row is `budget=16`, `beta=2.0`, `g0=0.5` at `679.60 tok/s`, but still only `0.45x` linear. Verdict: lean top2gap raises accept over linear at the node floor but remains verify/host-overhead limited.
- Tree decode machinery pass 2026-06-29: landed a bounded Component B/C pass for metadata and accepted-path KV commit. B vectorizes batched retrieve-link construction and skips unused custom-mask/retrieve metadata on dense compact FA4 verify. C adds an ordered fused all-layer KV commit for accepted tree paths. Fresh flushed FA4 page16 oracle gate passed for top2gap w8/b16 beta=1.0 g0=1.0 on GSM8K and MATH-500 with zero mismatches. Compared with the prior lean b16 row, GSM8K improved `570.64 -> 775.72 tok/s` and `11.30 -> 8.31 ms/step`; MATH-500 improved `674.74 -> 908.35 tok/s` and `11.71 -> 8.70 ms/step`. Tree is still below same-run linear (`938.62/1206.93 tok/s`) and the prior linear bars (`1152.85/1505.24 tok/s`), so Component A paged-tree verify remains the main blocker.
- Compact metadata follow-up 2026-06-29: landed a second Component B reduction in the compact FA4 path. Tree construction now bulk-copies draft top-k/tree rows instead of per-row `.item()`/`.tolist()` syncs, keeps compact sequence lengths on CPU without a device-to-host round trip, and fills one preallocated compact KV-index tensor instead of per-node tensor/cat materialization. Fresh flushed FA4 page16 oracle gate passed for top2gap w8/b16 beta=1.0 g0=1.0 on GSM8K and MATH-500 with zero mismatches. Compared with the B/C commit, GSM8K improved `775.72 -> 803.95 tok/s` and `8.31 -> 8.02 ms/step`; MATH-500 improved `908.35 -> 937.56 tok/s` and `8.70 -> 8.43 ms/step`. Tree is still below same-run linear (`936.66/1204.29 tok/s`) and prior linear bars (`1152.85/1505.24 tok/s`).
- Accept-kernel follow-up 2026-06-29: compact FA4 tree acceptance now reuses SGLang's CUDA `verify_tree_greedy` kernel with later-first retrieve links to preserve JetSpec duplicate-sibling semantics, replacing the torch tensor-algebra accept walk on the dense compact path. Fresh flushed FA4 page16 oracle gate passed for top2gap w8/b16 beta=1.0 g0=1.0 on GSM8K and MATH-500 with zero mismatches. Compared with the compact metadata commit, GSM8K improved `803.95 -> 817.61 tok/s` and `8.02 -> 7.89 ms/step`; MATH-500 improved `937.56 -> 954.60 tok/s` and `8.43 -> 8.28 ms/step`. Tree remains below same-run linear (`937.49/1205.79 tok/s`) and prior linear bars (`1152.85/1505.24 tok/s`).
- Final paged-verify serving confirmation 2026-06-29: ran the end-to-end paged FA4 tree verifier with `SGLANG_DFLASH_TREE_PAGED_FA4_VERIFY=1` on GPU 0, FA4 page16, normal decode CUDA graph, and fresh flushed oracle/bench flow. Same-run linear bars completed: GSM8K `1140.66 tok/s`, accept `5.849`, `5.13 ms/step`; MATH-500 `1483.76 tok/s`, accept `7.624`, `5.14 ms/step`. The paged tree failed the first serving losslessness gate on GSM8K: top2gap w8/b16 reported `token_exact=false` with `3/5` mismatches against the fresh linear oracle, so no valid full tree benchmark or MATH-500 tree gate was run. The captured invalid GSM8K tree gate was `835.50 tok/s`, accept `6.183`, `7.40 ms/step`, mean nodes `16.00`, but it is diagnostic only and cannot be compared to linear or the paged-off b16 bars. Verdict: tree-with-paged-kernel does not match or beat linear because the serving path is not lossless.

## Final Paged FA4 Verify Serving Gate

Date/time: 2026-06-29 17:03-17:08 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=0`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Paged verifier env: `SGLANG_DFLASH_TREE_PAGED_FA4_VERIFY=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend for DFlash rows: `--attention-backend fa4 --page-size 16`
- Decode graph flags: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`
- Harness: `jetspec/run_dflash_gate_bench.sh all` with `RUN_STAMP=final_paged_fa4_20260629_1704`, fresh flushed oracle gates, and first-80 benchmark prompts for the linear bars.
- Artifact dir: `jetspec/runs/final_paged_fa4_20260629_1704`

Results:

| dataset | config | exact | accept | tok/s | ms/step | mean nodes | vs linear | paged-off b16 delta | artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| GSM8K | linear w1 FA4 page16 | PASS | 5.849 | 1140.66 | 5.13 | n/a | 1.00x | n/a | `jetspec/runs/final_paged_fa4_20260629_1704/bench_gsm8k_linear_w1_fa4p16.json` |
| GSM8K | top2gap w8/b16 paged FA4 | FAIL 3/5 | 6.183 | 835.50 | 7.40 | 16.00 | invalid | invalid | `jetspec/runs/final_paged_fa4_20260629_1704/gate_gsm8k_top2gap_w8_b16_beta1p0_g01p0.json` |
| MATH-500 | linear w1 FA4 page16 | PASS | 7.624 | 1483.76 | 5.14 | n/a | 1.00x | n/a | `jetspec/runs/final_paged_fa4_20260629_1704/bench_math500_linear_w1_fa4p16.json` |
| MATH-500 | top2gap w8/b16 paged FA4 | not run | n/a | n/a | n/a | n/a | invalid | invalid | blocked by GSM8K serving mismatch |

Mismatch details from the GSM8K gate:

| sample | first diff | expected len | actual len |
|---:|---:|---:|---:|
| 0 | 8 | 326 | 275 |
| 1 | 49 | 139 | 138 |
| 4 | 22 | 328 | 327 |

Status:
- Losslessness gate: FAIL. The paged verifier is not token-exact in serving despite the isolated bit-exact kernel result.
- Performance verdict: no valid tree throughput comparison is available. The only tree row is a failed first-5 GSM8K gate, so its `7.40 ms/step` cannot be used as the requested full-run delta versus the paged-off b16 rows (`7.89` GSM8K / `8.28` MATH-500).
- Final answer to the benchmark question: NO. With the paged kernel enabled, dense top2gap w8/b16 does not match or beat linear DFlash end-to-end, because the serving losslessness gate fails before a valid benchmark can be accepted.

## Tree Decode Machinery Pass - Components B/C

Date/time: 2026-06-29 05:10-05:16 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend for DFlash rows: `--attention-backend fa4 --page-size 16`
- Decode graph flags: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`
- Harness: `jetspec/run_dflash_gate_bench.sh all`, first 5 flushed gate prompts plus first 80 benchmark prompts per dataset.
- Summary artifact: `jetspec/runs/dflash_gate_bench_component_bc_20260629_051037/summary.ndjson`

Landed:
- Component B: `build_batched_retrieve_links_from_parents` replaces per-row Python retrieve-link construction for tree-state metadata, and dense compact FA4 verify skips unused `custom_mask`/retrieve-link metadata.
- Component C: `copy_all_layer_kv_cache_accept_path_ordered_tiled` adds a fused ordered accepted-path KV commit for MHA KV pools; dense compact and expanded-causal tree commits call the ordered helper.
- Component D support: `jetspec/run_dflash_gate_bench.sh` records the fresh oracle gate, full linear baseline, tree gate, and full tree benchmark in one reproducible run.

Results:

| dataset | config | exact | accept | tok/s | ms/step | mean nodes | vs same-run linear | vs prior linear bar | artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| GSM8K | linear w1 FA4 page16 | PASS | 5.85 | 938.62 | 6.23 | n/a | 1.00x | 0.81x | `jetspec/runs/dflash_gate_bench_component_bc_20260629_051037/bench_gsm8k_linear_w1_fa4p16.json` |
| GSM8K | top2gap w8/b16 beta=1.0 g0=1.0 | PASS | 6.45 | 775.72 | 8.31 | 16.00 | 0.83x | 0.67x | `jetspec/runs/dflash_gate_bench_component_bc_20260629_051037/bench_gsm8k_top2gap_w8_b16_beta1p0_g01p0.json` |
| MATH-500 | linear w1 FA4 page16 | PASS | 7.62 | 1206.93 | 6.32 | n/a | 1.00x | 0.80x | `jetspec/runs/dflash_gate_bench_component_bc_20260629_051037/bench_math500_linear_w1_fa4p16.json` |
| MATH-500 | top2gap w8/b16 beta=1.0 g0=1.0 | PASS | 7.90 | 908.35 | 8.70 | 16.00 | 0.75x | 0.60x | `jetspec/runs/dflash_gate_bench_component_bc_20260629_051037/bench_math500_top2gap_w8_b16_beta1p0_g01p0.json` |

Before/after versus the prior lean top2gap b16 row:

| dataset | prior tok/s | new tok/s | prior ms/step | new ms/step | step-time delta |
|---|---:|---:|---:|---:|---:|
| GSM8K | 570.64 | 775.72 | 11.30 | 8.31 | -26.4% |
| MATH-500 | 674.74 | 908.35 | 11.71 | 8.70 | -25.7% |

Status:
- Losslessness gate: PASS. Fresh flushed oracle comparison reported `exact=true`, zero mismatches for GSM8K and MATH-500 gate and benchmark rows.
- Performance: landed because b16 tree per-step cost dropped by about one quarter with unchanged node count and unchanged acceptance.
- Verdict: still not linear. The remaining same-run per-step gap is `8.31 vs 6.23 ms` on GSM8K and `8.70 vs 6.32 ms` on MATH-500. Against the prior faster linear bars, tree remains `0.67x` and `0.60x` linear. Component A, the FA4-exact paged-tree verify replacement, remains the critical unlanded blocker.

## Compact Metadata Follow-Up

Date/time: 2026-06-29 05:35-05:42 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend for DFlash rows: `--attention-backend fa4 --page-size 16`
- Decode graph flags: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`
- Harness: `jetspec/run_dflash_gate_bench.sh all`, first 5 flushed gate prompts plus first 80 benchmark prompts per dataset.
- Summary artifact: `jetspec/runs/dflash_gate_bench_component_meta2_20260629_053534/summary.ndjson`

Landed:
- Bulk CPU copies for root/top-k inputs to the CPU tree builder, replacing per-row scalar syncs and per-row tensor-to-list conversions.
- CPU-side tree token/parent/depth staging with one H2D copy per tensor, replacing three small per-row device tensor materializations.
- Compact FA4 metadata assembly now builds `compact_seq_lens_cpu` directly on CPU, batches each row's tree-path slot gather, and fills one preallocated `compact_kv_indices` tensor instead of allocating `torch.tensor(path)` and `torch.cat((prefix, path))` per node.

Results:

| dataset | config | exact | accept | tok/s | ms/step | mean nodes | vs same-run linear | vs prior linear bar | artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| GSM8K | linear w1 FA4 page16 | PASS | 5.85 | 936.66 | 6.24 | n/a | 1.00x | 0.81x | `jetspec/runs/dflash_gate_bench_component_meta2_20260629_053534/bench_gsm8k_linear_w1_fa4p16.json` |
| GSM8K | top2gap w8/b16 beta=1.0 g0=1.0 | PASS | 6.45 | 803.95 | 8.02 | 16.00 | 0.86x | 0.70x | `jetspec/runs/dflash_gate_bench_component_meta2_20260629_053534/bench_gsm8k_top2gap_w8_b16_beta1p0_g01p0.json` |
| MATH-500 | linear w1 FA4 page16 | PASS | 7.62 | 1204.29 | 6.33 | n/a | 1.00x | 0.80x | `jetspec/runs/dflash_gate_bench_component_meta2_20260629_053534/bench_math500_linear_w1_fa4p16.json` |
| MATH-500 | top2gap w8/b16 beta=1.0 g0=1.0 | PASS | 7.90 | 937.56 | 8.43 | 16.00 | 0.78x | 0.62x | `jetspec/runs/dflash_gate_bench_component_meta2_20260629_053534/bench_math500_top2gap_w8_b16_beta1p0_g01p0.json` |

Before/after versus the Component B/C commit:

| dataset | B/C tok/s | new tok/s | B/C ms/step | new ms/step | step-time delta |
|---|---:|---:|---:|---:|---:|
| GSM8K | 775.72 | 803.95 | 8.31 | 8.02 | -3.5% |
| MATH-500 | 908.35 | 937.56 | 8.70 | 8.43 | -3.1% |

Status:
- Losslessness gate: PASS. Fresh flushed oracle comparison reported `exact=true`, zero mismatches for GSM8K and MATH-500 gate and benchmark rows.
- Performance: landed as a bounded Component B follow-up; same-node b16 tree step time improved again without changing acceptance or verifier numerics.
- Verdict: still not linear. The remaining same-run per-step gap is `8.02 vs 6.24 ms` on GSM8K and `8.43 vs 6.33 ms` on MATH-500. The remaining overhead is now dominated by the compact FA4 verifier's repeated prefix/KV gather shape and target model replay rather than accepted-path KV commit.

## Accept-Kernel Follow-Up

Date/time: 2026-06-29 05:56-06:02 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend for DFlash rows: `--attention-backend fa4 --page-size 16`
- Decode graph flags: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`
- Harness: `jetspec/run_dflash_gate_bench.sh all`, first 5 flushed gate prompts plus first 80 benchmark prompts per dataset.
- Summary artifact: `jetspec/runs/dflash_gate_bench_component_accept_kernel2_20260629_055615/summary.ndjson`

Landed:
- `build_batched_retrieve_links_from_parents(..., prefer_later_sibling=True)` can produce later-first sibling links for the accept walk. This preserves JetSpec's duplicate-child rule where a later sibling with the same draft token overwrites an earlier one.
- Dense compact FA4 tree acceptance uses SGLang's CUDA `verify_tree_greedy` kernel when available, with the tensor implementation retained as the fallback for other paths/devices.

Results:

| dataset | config | exact | accept | tok/s | ms/step | mean nodes | vs same-run linear | vs prior linear bar | artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| GSM8K | linear w1 FA4 page16 | PASS | 5.85 | 937.49 | 6.24 | n/a | 1.00x | 0.81x | `jetspec/runs/dflash_gate_bench_component_accept_kernel2_20260629_055615/bench_gsm8k_linear_w1_fa4p16.json` |
| GSM8K | top2gap w8/b16 beta=1.0 g0=1.0 | PASS | 6.45 | 817.61 | 7.89 | 16.00 | 0.87x | 0.71x | `jetspec/runs/dflash_gate_bench_component_accept_kernel2_20260629_055615/bench_gsm8k_top2gap_w8_b16_beta1p0_g01p0.json` |
| MATH-500 | linear w1 FA4 page16 | PASS | 7.62 | 1205.79 | 6.32 | n/a | 1.00x | 0.80x | `jetspec/runs/dflash_gate_bench_component_accept_kernel2_20260629_055615/bench_math500_linear_w1_fa4p16.json` |
| MATH-500 | top2gap w8/b16 beta=1.0 g0=1.0 | PASS | 7.90 | 954.60 | 8.28 | 16.00 | 0.79x | 0.63x | `jetspec/runs/dflash_gate_bench_component_accept_kernel2_20260629_055615/bench_math500_top2gap_w8_b16_beta1p0_g01p0.json` |

Before/after versus the compact metadata commit:

| dataset | metadata tok/s | new tok/s | metadata ms/step | new ms/step | step-time delta |
|---|---:|---:|---:|---:|---:|
| GSM8K | 803.95 | 817.61 | 8.02 | 7.89 | -1.7% |
| MATH-500 | 937.56 | 954.60 | 8.43 | 8.28 | -1.8% |

Status:
- Losslessness gate: PASS. Fresh flushed oracle comparison reported `exact=true`, zero mismatches for GSM8K and MATH-500 gate and benchmark rows.
- Performance: landed because it removes part of the post-verify torch accept walk without changing verifier math, accepted path semantics, or node count.
- Verdict: still not linear. The remaining same-run per-step gap is `7.89 vs 6.24 ms` on GSM8K and `8.28 vs 6.32 ms` on MATH-500.

Residual profile after this commit:
- Artifact: `jetspec/profiles/dflash_gate_bench_component_accept_kernel_profile_20260629_060426/top2gap_w8_b16_decode_analysis.txt`
- The tree accept walk is no longer a millisecond block: `verify_tree_greedy_func` is about `0.02-0.04 ms/call` in the trace.
- Compact FA4 still pays tree-only K/V materialization costs: `vectorized_gather_kernel` remains `0.94 ms` over the captured decode window, and DtoD copies from `_forward_batch_generation_tree` remain visible (`0.47 ms`, 210 launches).
- The explicit FA4 attention kernels are not larger than linear; the residual is the compact verifier machinery that repeats prefix/KV gather work to present one ragged FA4 row per node.
- Hard wall: closing the remaining `~1.6-2.0 ms/step` gap needs Component A: a paged-tree verifier that shares prefix KV across nodes and stays FA4-numerically equivalent. It must honor `softcap=layer.logit_cap` and keep fp32 softmax-probability accumulation through the V matmul. The JetSpec reference kernel is not directly landable because it lacks those FA4-exact properties.

## Lean Top2gap Sweep - Paper Datasets

Date/time: 2026-06-29 04:09-04:43 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend for DFlash rows: `--attention-backend fa4 --page-size 16`
- Decode graph flags: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`
- Harness: `jetspec/bench_paper_sglang.py`, first 80 samples per dataset, greedy `temperature=0`, `top_p=1.0`, `max_new_tokens=2048`
- Summary artifact: `jetspec/runs/top2gap_lean_20260629/summary.ndjson`

Comparison bars:
- GSM8K linear DFlash: `1152.85 tok/s`, accept `5.85`
- MATH-500 linear DFlash: `1505.24 tok/s`, accept `7.62`

Results:

| dataset | beta | g0 | budget | exact | accept | tok/s | ms/step | mean nodes | vs linear |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| GSM8K | 1.0 | 1.0 | 24 | PASS | 6.89 | 551.39 | 12.50 | 23.68 | 0.48x |
| MATH-500 | 1.0 | 1.0 | 24 | PASS | 8.61 | 653.04 | 13.18 | 23.10 | 0.43x |
| GSM8K | 1.0 | 1.0 | 16 | PASS | 6.45 | 570.64 | 11.30 | 16.00 | 0.49x |
| MATH-500 | 1.0 | 1.0 | 16 | PASS | 7.90 | 674.74 | 11.71 | 16.00 | 0.45x |
| GSM8K | 1.0 | 1.0 | 32 | PASS | 7.11 | 524.98 | 13.54 | 31.20 | 0.46x |
| MATH-500 | 1.0 | 1.0 | 32 | PASS | 8.89 | 617.16 | 14.40 | 29.99 | 0.41x |
| GSM8K | 1.0 | 1.0 | 48 | PASS | 7.34 | 466.21 | 15.74 | 45.96 | 0.40x |
| MATH-500 | 1.0 | 1.0 | 48 | PASS | 9.25 | 544.24 | 16.99 | 43.40 | 0.36x |
| GSM8K | 2.0 | 0.5 | 24 | PASS | 6.77 | 538.51 | 12.58 | 23.28 | 0.47x |
| MATH-500 | 2.0 | 0.5 | 24 | PASS | 8.60 | 646.45 | 13.31 | 22.60 | 0.43x |
| GSM8K | 2.0 | 0.5 | 16 | PASS | 6.49 | 552.35 | 11.76 | 16.00 | 0.48x |
| MATH-500 | 2.0 | 0.5 | 16 | PASS | 8.11 | 679.60 | 11.93 | 16.00 | 0.45x |
| GSM8K | 2.0 | 0.5 | 32 | PASS | 6.91 | 509.96 | 13.56 | 30.38 | 0.44x |
| MATH-500 | 2.0 | 0.5 | 32 | PASS | 8.84 | 590.52 | 14.97 | 28.98 | 0.39x |
| GSM8K | 2.0 | 0.5 | 48 | PASS | 7.03 | 438.17 | 16.05 | 44.19 | 0.38x |
| MATH-500 | 2.0 | 0.5 | 48 | PASS | 9.05 | 535.78 | 16.89 | 41.05 | 0.36x |

Verdict:
- No lean top2gap config matches or beats linear throughput while keeping accept above linear.
- The closest paired config is `width=8`, `budget=16`, `beta=1.0`, `g0=1.0`; it exceeds linear accept on both datasets but remains `0.49x` and `0.45x` linear throughput.
- The trend says lower budget improves speed down to the current `budget=16` floor; bigger budgets improve accept but lose throughput. Sharper `(beta=2.0,g0=0.5)` is not an overall win, with only a small MATH-500-only throughput bump at `budget=16`.

## Top2gap Construction Job 1 - Losslessness Gate

Date/time: 2026-06-29 03:16-03:22 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend for DFlash rows: `--attention-backend fa4 --page-size 16`
- Decode graph flags: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`
- Harness: `jetspec/bench_paper_sglang.py`, first 5 samples per dataset, greedy `temperature=0`, `top_p=1.0`, `max_new_tokens=2048`

Implemented:
- `build_tree_from_topk_cpu(..., score_mode="top2gap")` computes per-depth fanout caps from the rank-1/rank-2 logprob gap: `round(width * sigmoid(-beta * (gap - g0)))`, clamped to at least 1.
- The breadth-first and depth-first CPU tree builders now honor explicit `fanout_caps`.
- Server args and speculative arg validation expose `--speculative-dflash-tree-draft top2gap`, `--speculative-dflash-top2gap-beta`, and `--speculative-dflash-top2gap-g0`.
- `jetspec/bench_paper_sglang.py` records `tree_draft`, top2gap params, and mean root-inclusive tree node count from scheduler internal state.

Artifacts:

| dataset | config | lossless gate | accept len | tok/s | ms/step | mean tree nodes | artifact |
|---|---|---|---:|---:|---:|---:|---|
| GSM8K | width=1 oracle | oracle | 5.33 | 854.93 | 6.24 | n/a | `jetspec/runs/top2gap_job1_oracle_gate_gsm8k_w1_31965.json` |
| GSM8K | top2gap w4/b64 beta=1.0 g0=1.0 | pass 5/5 | 6.56 | 351.82 | 18.66 | 57.54 | `jetspec/runs/top2gap_job1_gate_gsm8k_tree_w4_b64_beta1_g01_31966.json` |
| MATH-500 | width=1 oracle | oracle | 7.71 | 1224.58 | 6.30 | n/a | `jetspec/runs/top2gap_job1_oracle_gate_math500_w1_31965.json` |
| MATH-500 | top2gap w4/b64 beta=1.0 g0=1.0 | pass 5/5 | 9.18 | 479.81 | 19.14 | 50.45 | `jetspec/runs/top2gap_job1_gate_math500_tree_w4_b64_beta1_g01_31966.json` |

Status:
- Job 1 losslessness gate passed. The construction change is shape-only for verify/commit correctness, and the fresh flushed oracle check confirmed token-exact outputs on both datasets.
- Next: run the requested beta/g0/budget/width sweep to find whether a lean top2gap tree can beat linear DFlash throughput.

## Paper Dataset Benchmark - GSM8K and MATH-500

Date/time: 2026-06-28 16:24-17:02 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend for DFlash rows: `--attention-backend fa4 --page-size 16`
- Decode graph flags: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`
- Harness: `jetspec/bench_paper_sglang.py`, first 80 samples per dataset, greedy `temperature=0`, `top_p=1.0`, `max_new_tokens=2048`
- Prompt format matches JetSpec's `tps_walltime.py`: problem text plus `Please reason step by step, and put your final answer within \boxed{}.` through the Qwen chat template with thinking disabled.
- AR-greedy caveat: target-only FA4 used `--speculative-num-draft-tokens 1` to avoid a current startup guard crash when the value is `None`; `speculative_algorithm=None`, so no speculation. Runtime page size was rewritten to 128 for AR only by the non-spec FA4 guard. DFlash rows retained page size 16.

Results:

| dataset | config | lossless gate | accept len | tok/s | vs AR | vs linear | ms/step | artifact |
|---|---|---|---:|---:|---:|---:|---:|---|
| GSM8K | AR-greedy | n/a | 1.00 | 269.57 | 1.00x | 0.23x | 3.71 | `jetspec/runs/paper_gsm8k_ar_31980.json` |
| GSM8K | Linear DFlash | n/a | 5.85 | 1159.90 | 4.30x | 1.00x | 5.04 | `jetspec/runs/paper_gsm8k_linear_31981.json` |
| GSM8K | Tree w7/b64 | pass 5/5 | 6.46 | 376.50 | 1.40x | 0.32x | 17.17 | `jetspec/runs/paper_gsm8k_tree_w7_b64_31985.json` |
| GSM8K | Tree w7/b128 | pass 5/5 | 7.77 | 329.91 | 1.22x | 0.28x | 23.55 | `jetspec/runs/paper_gsm8k_tree_w7_b128_31986.json` |
| GSM8K | Tree w7/b255 | FAIL 3/5 | 8.20 | 206.76 | 0.77x | 0.18x | 39.65 | `jetspec/runs/paper_gsm8k_tree_w7_b255_31982.json` |
| MATH-500 | AR-greedy | n/a | 1.00 | 262.87 | 1.00x | 0.17x | 3.80 | `jetspec/runs/paper_math500_ar_31993.json` |
| MATH-500 | Linear DFlash | n/a | 7.62 | 1503.36 | 5.72x | 1.00x | 5.07 | `jetspec/runs/paper_math500_linear_31984.json` |
| MATH-500 | Tree w7/b64 | pass 5/5 | 7.41 | 424.60 | 1.62x | 0.28x | 17.46 | `jetspec/runs/paper_math500_tree_w7_b64_31985.json` |
| MATH-500 | Tree w7/b128 | pass 5/5 | 9.55 | 372.49 | 1.42x | 0.25x | 25.64 | `jetspec/runs/paper_math500_tree_w7_b128_31986.json` |
| MATH-500 | Tree w7/b255 | FAIL 2/5 | 10.10 | 224.23 | 0.85x | 0.15x | 45.02 | `jetspec/runs/paper_math500_tree_w7_b255_31987.json` |

Verdict:
- Valid b128 acceptance reaches the paper's 7-9 range. Acceptance is not the primary blocker on these datasets.
- b255 has higher acceptance but breaks losslessness and should not be optimized as the first valid target.
- The gap is verify per-step cost. Paper implied step time is about `8.07 ms` on GSM8K and `8.31 ms` on MATH-500; valid b128 is `23.55 ms` and `25.64 ms`.
- Tree decode logs show eager target verify (`cuda graph: False`) while linear DFlash decode logs show CUDA graph use (`cuda graph: True`). The current ragged compact FA4 varlen verifier is the immediate optimization target; it needs a paged-tree/kernel-shaped verifier closer to the paper's `optimus_cutedsl.flash_attn_varlen_tree_paged_sm90`.

## Mask Root-Cause Job 3 - FA4 Page16 MT-bench, No Dense Reverify

Date/time: 2026-06-28 06:14-06:41 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend fa4 --page-size 16`
- Normal decode graph settings: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`
- Harness: `benchmark/mtbench/bench_sglang_eagle.py`, 80 MT-bench questions, `--parallel 1`, `max_new_tokens=2048`
- Tree pass: dense w7/b64 with accepted-path reverify removed. The benchmark process disabled only `sglang.global_config.global_config.enable_precache_with_tracing`.

Artifacts:

| run | result artifact | answers | accept length | throughput | latency | speed vs linear |
|---|---|---:|---:|---:|---:|---:|
| 8B linear width=1 | `jetspec/runs/mtbench_maskfix_fa4p16_linear_31945_result.jsonl` | 80/80 | 4.070 | 764.540 tok/s | 334.626 s | 1.00x |
| 8B tree w7/b64, no dense accepted-path reverify | `jetspec/runs/mtbench_maskfix_fa4p16_tree_w7_b64_noreverify_noprecache_31946_result.jsonl` | 80/80 | 5.053 | 243.000 tok/s | 1052.817 s | 0.32x |

Answer files:
- `jetspec/runs/mtbench_maskfix_fa4p16_linear_31945_answers.jsonl`: 80 rows
- `jetspec/runs/mtbench_maskfix_fa4p16_tree_w7_b64_noreverify_noprecache_31946_answers.jsonl`: 80 rows
- `jetspec/runs/mtbench_question.jsonl`: 80 rows
- Linear and tree answer `choices` match exactly: 0/80 mismatches.

Raw result lines:

```json
{"task": "mtbench", "backend": "srt", "num_gpus": 1, "latency": 334.626, "throughput": 764.54, "accept_length": 4.07, "num_requests": 80, "other": {"num_questions": 80, "parallel": 1}}
{"task": "mtbench", "backend": "srt", "num_gpus": 1, "latency": 1052.817, "throughput": 243.0, "accept_length": 5.053, "num_requests": 80, "other": {"num_questions": 80, "parallel": 1}}
```

Verdict:
- Removing the dense accepted-path reverify improves tree throughput versus the prior FA4 page16 retained-reverify tree run: `243.000` vs `199.983 tok/s` (`1.22x`).
- Dense tree still does not beat or match width=1 linear: `243.000` vs `764.540 tok/s`, or `0.32x` linear throughput (`3.15x` slower).
- Tree acceptance is higher than linear (`5.053` vs `4.070`, `1.24x`), but the ragged compact FA4 tree verify and tree/draft bookkeeping still dominate the saved target forwards.
- MoE was not changed.

## Mask Root-Cause Job 2 - FA4 Page16 Compact Exact Verify, No Dense Reverify

Date/time: 2026-06-28 05:36-06:02 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend fa4 --page-size 16`
- Tree gates: `--speculative-dflash-tree-width 7`, budgets `64` and `128`
- Fresh flushed oracle: `jetspec/runs/job1_fa4p16_linear_w1_flush_31811.json`

Fix:
- `FlashAttentionBackend` no longer relies only on backend `self.topk` for DFlash target verify. DFlash custom-mask and compact-tree metadata now use the per-request DFlash verify shape.
- FA4 page16 cannot represent DFlash's sparse ancestor set by treating raw token IDs as page IDs. Dense DFlash FA4 therefore builds `compact_kv_indices` for every tree query row: the committed prefix slots followed by the query's ancestor chain ending at self.
- The compact verifier gathers exactly those K/V rows and calls FA4 varlen attention with `max_seq_len_q=1`, `causal=False`, and per-query `cu_seqlens_k`. This bypasses the bad page-table interpretation that admitted sibling/cousin slots or missed sparse ancestors.
- Dense compact direct commit skips `_reverify_accepted_tree_path_for_commit`: accepted tree KV slots are copied into the canonical contiguous prefix slots with `move_kv_cache_overlap_safe`. MoE remains on its existing lossless path.

Accepted-set and layer-0 diagnostic:
- Diagnostic log: `jetspec/logs/job2_compact_direct_diag_nograph_fa4p16_tree_w7_b64_31940_server.log`
- Original failing node repro after the fix: `prefix=[6]`, `commit_lens=[2]`, accepted local path `[[0, 3]]`, branch candidates `[[12095, 13]]`.
- Correct set: prefix `[0,6)` plus tree cols `[0, 3]`; compact allowed set: `[0, 3]`; extras `[]`; missing `[]`; tree slots `[22, 25]`.
- The original mask-scale layer-0 hidden delta is gone: `first_layer_hidden_delta=(0, 0.001953125)` instead of `0.1357421875`, with layer-0 K/V still exact `(0.0, 0.0)`.
- The debug hook list is not a clean all-layer bf16-equivalence proof because the compact FA4 verifier and the paged clean-causal replay use different FA4 kernel shapes; later-layer K/V deltas grow from the layer-0 bf16 difference. The losslessness gate below is therefore the commit criterion for this milestone.

Fresh flushed losslessness gates:

| run | artifact | token exact vs FA4 page16 oracle | mismatches | mean accept length | aggregate tok/s |
|---|---|---:|---:|---:|---:|
| 8B tree w7/b64, no dense accepted-path reverify | `jetspec/runs/job2_maskfix_noreverify_fa4p16_tree_w7_b64_flush_31941.json` | PASS | 0/10 | 4.2004 | 217.44 |
| 8B tree w7/b128, no dense accepted-path reverify | `jetspec/runs/job2_maskfix_noreverify_fa4p16_tree_w7_b128_flush_31942.json` | PASS | 0/10 | 4.7738 | 181.38 |

Notes:
- Both servers used normal decode graph settings: `--cuda-graph-max-bs-decode 1 --cuda-graph-backend-decode full`.
- Compact FA4 tree verify itself currently runs eager because each query row has a ragged compact K/V list; this is expected to matter in Job 3 performance.
- Two exploratory gated experiments that tried to force the compact set through FA4 `with_kvcache` did not improve all-layer agreement and were not kept.

## Mask Root-Cause Job 1 - FA4 Page16 Attended Set

Date/time: 2026-06-28 05:13-05:20 UTC.

Environment:
- GPU: `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`
- Dense model: `Qwen/Qwen3-8B`
- Dense draft model: `JetSpec/jetspec-qwen3-8b`
- Backend: `--attention-backend fa4 --page-size 16`
- Tree: `--speculative-dflash-tree-width 7 --speculative-dflash-tree-budget 64`
- Diagnostic env: accepted-path reverify disabled, causal/layer/KV compare enabled, attended-set dump enabled.

Artifacts:
- Server log: `jetspec/logs/job1_maskdump_fa4p16_tree_w7_b64_31932_server.log`
- Earlier confirming log: `jetspec/logs/job1_maskdump_fa4p16_tree_w7_b64_31931_server.log`

Exact failing step:
- Compare step: `prefix=[6]`, `commit_lens=[2]`, accepted local path `[[0, 3]]`, branch candidates `[[12095, 13]]`.
- Tree and clean causal branch predicted the same next tokens: `tree_predict=[[13, 576]]`, `branch_predict=[[13, 576]]`.
- Hidden divergence reproduced at layer 0: `first_layer_hidden_delta=(0, 0.1357421875)`.
- Layer-0 K/V were still exact: `kv_max_abs=(0.0, 0.0)`.

Attended-set diff for accepted node `3`:
- Correct causal set: committed prefix positions `[0,6)` plus self/ancestor tree cols `[0, 3]`.
- DFlash `custom_mask` row: `allowed_tree_cols=[0, 3]`, `extra_tree_cols=[]`, `missing_tree_cols=[]`.
- Physical slots for the correct tree cols: `[22, 25]`.
- FA4 effective path: `effective_mode=linear_causal_no_custom_mask`, `allowed_tree_cols=[0, 1, 2, 3]`.
- Wrong extras admitted by FA4: tree cols `[1, 2]`, logical KV positions `[7, 8]`, physical slots `[23, 24]`.
- Missing positions: none.

Root cause:
- DFlash creates an EAGLE-style ancestor-only mask with prefix attention, matching `TreeMaskMode.FULL_MASK` semantics at the Python mask level.
- EAGLE reaches FA4's custom-mask cascade verifier because `server_args.speculative_eagle_topk > 1`.
- DFlash passes `DFlashVerifyInput.topk=7` and a 64-node `custom_mask`, but `FlashAttentionBackend` only checks backend `self.topk`, initialized from `server_args.speculative_eagle_topk` (`1` in DFlash).
- Therefore dense DFlash FA4 target verify falls into the `topk<=1` causal path. That path treats the first `--speculative-num-draft-tokens 16` BFS nodes as a contiguous causal chain and ignores the DFlash tree mask, so node `3` attends sibling/cousin cols `1` and `2`.
- This is the mask/index-construction bug that makes the accepted-path reverify necessary.

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

## 2026-06-29 Component 1 FA4 Paged-Tree Verify

Status:
- Implemented an opt-in FA4 paged-tree verifier for dense DFlash tree verify behind `SGLANG_DFLASH_TREE_PAGED_FA4_VERIFY=1`.
- The verifier uses FA4 CUTE `mask_mod` over the normal paged prefix plus tree-node suffix slots, so FA4 still owns softcap, softmax, and the P*V accumulation numerics.
- The worker now builds request-level prefix lengths and a `[request, q_node, kv_node]` ancestor mask instead of compact per-row K/V indices for this path.
- CUDA graph metadata plumbing is present for the paged path; graph replay uses static prefix-length and ancestor-mask buffers, with `SGLANG_DFLASH_TREE_PAGED_FA4_CUDA_GRAPH=0` available as a kill switch.

Validation completed:
- `CUDA_VISIBLE_DEVICES=7 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 PYTHONPATH=python python -m pytest test/registered/jit/test_dflash_paged_tree_verify.py -q -s`
- Result: 2 passed.
- The isolated FA4 oracle test compares paged-tree FA4 against compact gather plus FA4 with nonzero `softcap=5.0`; observed `max_abs_diff=0.0`, `mean_abs_diff=0.0`.
- The backend-path test runs `RadixAttention -> FlashAttentionBackend` with paged DFlash metadata, then repeats through `init_forward_metadata_out_graph(..., in_capture=True)` / `init_forward_metadata_in_graph(...)`; both match a softcapped reference within bf16 tolerance.
- Isolated timing on GPU 7 with `bs=16`, `tree_budget=16`, `prefix_len=256`, `hq=32`, `hkv=8`, `head_dim=128`: compact gather+FA4 `0.6212 ms/iter`, paged FA4 `0.1728 ms/iter`, with `max_abs_diff=0.0`.

Full gate status:
- Fresh flushed GSM8K/MATH-500 oracle and `jetspec/bench_paper_sglang.py` runs were not completed in this pass because GPU 7 reported only about `1.1 GiB` free while `nvidia-smi`/`pmon` showed no compute owner; targeted `nvidia-smi --gpu-reset -i 7` was refused as "in use by another client".
- The paged path remains default-off until the full flushed oracle and paper benchmark can be run on a usable GPU.
