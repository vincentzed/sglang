# M4 Step-1 result: NO-GO — green-context indexer∥attention overlap does not beat serial

Task: overlap the DSA indexer with the main-attention Q projection via FlashInfer/CUDA
green-context SM partitioning during long-context prefill
(`agent-tasks/M4-green-context-indexer-attn-overlap.md`).

**Verdict: the task is dead at Step 1.** At every prefill shape tested (GLM-5.2 TP4 and
DeepSeek-V3.2 TP8 per-rank shapes, first/middle/last chunks of an 80k prefill, 32k and
80k single-shot), the best green-context partition is **2–4% slower than serial**, and
the naive two-stream fork is a wash (1.00× ± 0.03 run-to-run noise). This is not a
tuning problem; it is arithmetic (see "Why it cannot win" below).

## Step 0 — environment

- 8× NVIDIA B300 SXM6 AC (SM103, **148 SMs**, 275 GB), measurements on GPU 0.
- torch **2.11.0+cu130** (the torch-2.7/2.8 green-ctx CUDA-graph caveat window does not
  apply; prefill is eager anyway), driver **610.43.02** (CUDA ≥13.1, so green-ctx
  workqueue "balanced" scope is available and was tested), flashinfer 0.6.13,
  sglang @ d7dcdf3efd.
- Shapes taken from the actual checkpoints' `config.json`
  (zai-org/GLM-5.2-FP8 & nvidia/GLM-5.2-NVFP4: hidden 6144, 64 heads, qk_head 256,
  q_lora 2048, indexer 32×128 topk 2048; deepseek-ai/DeepSeek-V3.2: hidden 7168,
  128 heads, qk_head 192, q_lora 1536, indexer 64×128).
- On ≥160 GB GPUs sglang defaults `chunked_prefill_size=16384`, so a real 80k prefill
  is 5 forwards of M=16384 rows against growing KV (N up to 80k). The bench covers that
  regime plus single-shot 32k/80k.

## What was measured

Region = one DSA layer's overlap window at eager prefill, per rank, with the
**production kernels**:

- **Side A (main Q-prep)**: `q_b_proj` GEMM (bf16, cuBLAS/TGV path) → q_nope/q_pe split
  → absorbed `bmm` with `w_kc`.
- **Side B (whole indexer)**: `wq_b` + `wk_weights_proj` GEMMs,
  `fused_q_indexer_rope_first_quant`, `fused_k_indexer_norm_rope` + fp8 `act_quant`,
  chunked `deep_gemm.fp8_mqa_logits`, `sgl_kernel.fast_topk_transform_ragged_fused`.

Arms: serial (today's order) | two-stream on one SM pool | green-context two-way SM
split (`sglang.srt.multiplex.green_ctx.split_device_green_ctx_by_sm_count`), A-slice
swept 8→48 SMs, both "wide" (whole indexer on the big slice) and "narrow" (projections
only, logits/topk after the join on the full device). Timing: CUDA events, median of
12–20 iters after warmup. Correctness: top-k compared as sorted sets (the fused top-k
kernel returns the selected set in nondeterministic order run-to-run — pre-existing
behavior, unrelated to this change), identical across all arms.

## Step-1 table (GLM-5.2 TP4 per-rank shapes; median ms; speedup vs serial)

`deep_gemm` grid sized to the slice in green arms (see hazard note below).

| arm | M=16k, N=16k | M=16k, N=49k | M=16k, N=82k | M=32k, N=32k | M=82k, N=82k (chunked logits) |
|---|---|---|---|---|---|
| side A alone (full device) | 0.226 | 0.226 | 0.226 | 0.436 | 1.06 |
| side B alone (full device) | 1.46 | 4.41 | 7.45 | 4.45 | 22.6 |
| serial | 1.67 (1.000×) | 4.69 (1.000×) | 7.70 (1.000×) | 4.89 (1.000×) | 23.8 (1.000×) |
| two-stream (one SM pool) | 0.994× | 1.011× | 0.996× | 0.997× | 0.993× |
| green a8/b140 | — | — | **0.975×** | — | **0.980×** |
| green a16/b132 | 0.829×¹ | 0.715×¹ | 0.934× | 0.773×¹ | 0.959× |
| green a24/b124 | 0.807×¹ | 0.704×¹ | 0.896× | 0.753×¹ | 0.918× |
| green narrow (best over 8–48) | 0.989× | 0.980× | 0.953× | 0.972× | 0.998× |

¹ columns run before the deep_gemm grid fix also include that penalty; with the fix the
best wide split is still <1.0 everywhere (0.975–0.990).

- `workqueue_scope="balanced"` (CUDA 13.1): best arm 0.990× — no rescue.
- **DeepSeek-V3.2 shapes** (M=16k, N=82k): side A 0.154 ms vs side B 12.6 ms (1.2%);
  best green arm 0.976×, two-stream 1.013× (within noise). Same verdict.
- Green-ctx split creation: **0.4 ms** one-time — creation cost is a non-issue; it is
  not why this fails.
- **Co-residency was real**: profiler trace of green a16/b132 shows side A's kernels
  overlap side B's kernels for **94.8%** of side A's busy time on disjoint SM slices.
  The mechanism works; it just costs more than it hides.

## Why it cannot win (the arithmetic that kills it)

1. **The hideable work is tiny.** The only main-side work independent of the indexer
   output is q_b_proj + q split + absorbed bmm ≈ 0.23 ms at M=16384 — **3.0%** of the
   8 ms overlap region at N=80k (1.2% for V3.2). The task brief's "~16%" is the
   indexer's *own* logits+topk share of e2e GPU time; it is on the *indexer* side of
   the overlap and cannot be recovered by this overlap — it is the thing being
   overlapped *against*.
2. **The minimum SM carve-out exceeds the prize.** Both sides are throughput-bound at
   long-prefill sizes (near-linear SM scaling: side A on 16/148 SMs ran 7.7× slower,
   matching the trace). Hiding side A saves ≤3% of the region, but the smallest
   partition the driver allows on B300 is 8 SMs = **5.4%** of the machine, which slows
   the dominant indexer side by ~4–6% of the region. Loss ≥ gain at every split ratio,
   in both wide and narrow variants. deep_gemm's logits kernel is ~55–75% of side B and
   is fp8-tensor-core/SM-bound, so it pays the full proportional tax; the
   bandwidth-bound top-k does not release enough SM-time to change the balance.
3. **Two-stream on one SM pool is free and already a wash** (0.99–1.03× = noise),
   confirming there is no idle-resource window at these kernel sizes for co-scheduling
   to exploit. (The indexer's internal wq_b ∥ wk_weights_proj two-stream overlap in
   `_fused_q_prepare_and_store` already runs at eager prefill today — the projections
   the task proposed to overlap were in fact already partially overlapped.)
4. **e2e ceiling even for a FREE overlap**: 0.226 ms × 5 chunks × 22 full-indexer
   layers (GLM-5.2 `index_topk_freq=4`) ≈ 25 ms ≈ **<1% of an 80k prefill**; ≈2–3% if
   all 78 layers ran full indexers. Measured green-ctx delta is negative, so the real
   number is a regression.

## Integration hazard found on the way (worth keeping in mind for any green-ctx work)

`deep_gemm.fp8_mqa_logits` sizes its persistent grid from `deep_gemm.get_num_sms()`
(=148). Launched on a 140-SM green slice, the 148-CTA grid runs as **two waves → ~2×
slowdown** of the logits kernel (observed: 0.71× overall). Any future green-ctx
integration must re-plumb per-slice SM counts into deep_gemm (and any other
persistent-grid kernel) or it will silently halve logits throughput.

## What would change the verdict

- Hardware/driver allowing sub-8-SM partitions AND a main-side share above the
  minimum-slice fraction (not true for any DSA model here: 1.2–4.7%).
- Overlapping the indexer against a *large* independent consumer (e.g. cross-layer
  MoE↔indexer overlap) — out of scope for M4 and a different dependency structure.

## Artifacts

- `bench_step1.py` — the settling experiment (arms: serial / two-stream / green wide /
  green narrow; presets glm52+dsv32; `--deepgemm-slice-sms`, `--workqueue-scope`).
- `trace_coresidency.py` — co-residency proof (94.8% overlap on green slices).
- `res_*.json` — raw numbers.

Per the task's Step-1 gate: green-ctx does not beat serial ⇒ **STOP; no implementation,
no server-level A/B needed** (Steps 2–5 are moot).
