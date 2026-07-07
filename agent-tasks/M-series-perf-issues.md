# DSA / GLM-5.2 perf ideas — issue bodies (M-series)

Six GitHub-issue-ready bodies. All target the DeepSeek-V3.2 / GLM-5.2 DSA path on
SM100 (Blackwell). Grounded in the b8zhong GLM-5.2-NVFP4 80k profiling (TP4, DSA
attention, `moe_runner=flashinfer_trtllm`, fp8 KV). Each dedups against existing
work. File separately or as a batch.

---

## M3 — Cross-layer indexer/attention pipeline (DSA prefill)

**Motivation.** Each DSA layer runs indexer (select top-k) → sparse-MLA attention
*serially* (attention depends on the top-k). At 80k prefill the indexer-selection
machinery is a measured ~16% of GPU time — `deep_gemm::sm100_fp8_mqa_logits` (8.3%)
+ `topk_transform_prefill` (6.2%) + gather (1.8%) — running in front of the 33%
attention. Under index-share (`index_topk_freq>1`), most layers are `skip_topk`
shared layers that never run their own indexer, so that "slot" is idle.

**Proposal.** Pipeline across layers: while layer L's sparse-MLA attention runs,
precompute/prefetch what L+1's attention needs — start the producer layer's indexer,
or prefetch the reused KV pages for shared layers — on a side stream with event sync,
mirroring the existing dual-stream indexer overlap.

**Prior art / dedup.** #29637 (HiSparse prefetch via IndexShare, hides ~70% host-IO)
and #28523 (IndexCache shared-layer IO overlap) already prefetch shared-layer pages on
a side stream — but for the **HiSparse host-offload** path. This generalizes the idea
to the on-device (non-HiSparse) DSA path as a cross-layer indexer/attention pipeline.

**Risk.** Cross-layer scheduling correctness; CUDA-graph capture stream explosion —
#30025 fixed a 22→2 stream blowup in the indexer dual-stream; this must not
reintroduce it.

**Definition of done.** Long-context prefill TTFT improvement, **bit-identical
outputs**, captured-graph stream count ≤ current.

**References.** `dsa_indexer.py`, `dsa_backend.py`; #29637, #28523, #30025.

---

## M4 — Green-context SM partition: indexer ∥ main-attention projection

**Motivation.** Within a DSA layer, the indexer Q/K projection and the main MLA QKV
projection both read `hidden_states` and are independent, but run serially on one
stream. At 80k prefill the indexer is ~16% of GPU time; overlapping its projection
with the main attention prep recovers exposed serial time.

**Proposal.** Use FlashInfer green contexts (`split_device_green_ctx_by_sm_count`) to
partition SMs and run the indexer projection concurrently with the main MLA projection
on separate SM slices, joining before the logits/attention.

**Prior art / dedup.** FlashInfer green-context SM partitioning (`green_ctx.py`) is an
existing mechanism, not yet applied to the DSA indexer/attention split. Distinct from
the dual-stream indexer Q∥K overlap (#27705), which is *within* the indexer.

**Risk.** SM-partition sizing (too thin a slice starves the indexer); the overlap
gain flips negative at small token counts (decode) → gate to long prefill; green-ctx ×
CUDA-graph interaction.

**Definition of done.** Measurable prefill speedup at ≥32k with bit-identical outputs;
no regression at short context (gated off there).

**References.** `dsa_indexer.py`; FlashInfer `green_ctx.py`; #27705.

---

## M5 — Overlap prefill TP all-reduce with expert compute (AsyncTP / TBO)

**Motivation.** GLM-5.2-NVFP4 80k prefill (TP4) shows `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`
at **13%** — an exposed, unfused TP all-reduce. Decode already uses MNNVL one-shot AR
fusion (9.5%), but prefill (token count ≫ the 2048 fusion cap) falls to plain ring.

**Proposal.** Overlap the prefill all-reduce with adjacent compute: (a) **AsyncTP** —
rewrite AllReduce → ReduceScatter + local-norm + AllGather so the collective overlaps
the next GEMM; or (b) **two-batch overlap (TBO)** — overlap one micro-batch's comm with
the other's expert compute.

**Prior art / dedup.** #28639 (open, `ag_gemm`+`moe_rs` overlap kernels for dsv4
prefill) is direct prior art — validate/adapt for GLM-5.2/V3.2. Complements N3
(AR+RMSNorm fusion, capped at 2048 tokens so it does **not** cover 80k prefill). vLLM
AsyncTP (`fuse_gemm_comms`) is the upstream analog.

**Risk.** Correctness under DP-attention (no DP-attn AR-fusion path exists today);
symmetric-memory workspace size at long context.

**Definition of done.** Long-context prefill throughput improvement (bench vs ring),
bit-identical outputs, **pure-TP first** (DP-attn scoped separately).

**References.** `communicator.py`, `parallel_state.py`; #28639; N3.

---

## M6 — FP4 KV cache for the SM100 sparse-MLA decode path

> **Dedup note:** this is the **SM100/CUDA sparse-MLA decode slice** of the broader
> FP4-KV-cache effort — file it *under* the existing tracking issues, not standalone.

**Motivation.** Decode is KV-bandwidth-bound on already-FP8 KV. On GLM-5.2-NVFP4 80k
decode the attention + FlashMLA-KV read dominate memory traffic; moving the sparse-MLA
KV cache FP8 → FP4 (E2M1) halves the KV read — the decode-bandwidth analog of the
already-shipped FP8-Q win.

**Proposal.** Add an FP4 KV-cache path for the SM100 sparse-MLA decode kernel
(`_forward_flashmla_kv` / trtllm-gen): FP4 quant on store, dequant-in-kernel on read.

**Prior art / dedup.** Track under **#21601** ("[Feature] Add FP4 KV Cache Design +
SM120") and **#26571** ("[WIP] FP4 KV Cache Support"). Distinct from **#21889** (AMD
TileLang FP4-KV for NSA) and **#25555** (MXFP4 blockfp4-hadamard-quant) — this is the
**NVIDIA SM100 sparse-MLA decode** kernel path specifically. Coordinate with the above
to avoid duplication.

**Risk / accuracy.** FP4 KV error compounds autoregressively at decode; needs an
accuracy study (gsm8k/AIME within threshold). Kernel support: the SM100 sparse-MLA
decode kernel must accept FP4 KV (likely a new dequant path).

**Definition of done.** Decode TPOT improvement at long context, accuracy within
threshold, landed under the #21601/#26571 tracking umbrella.

**References.** `dsa_backend.py` (`_forward_flashmla_kv`); #21601, #26571, #21889, #25555.

---

## M7 — Temporal index-share: reuse top-k across target decode steps

**Motivation.** The DSA indexer runs every decode step to select top-k KV, but for a
stable context the top-k drifts slowly step-to-step, so per-step recompute is largely
redundant. Decode trace: `topk_transform_decode` (7.3%) + `fp8_paged_mqa_logits`
(1.3%) per step.

**Proposal.** Reuse the indexer top-k across N consecutive *target* decode steps
(recompute every N, carry indices between), skipping the logits + top-k on N-1 of N
steps, gated by a cheap refresh/acceptance check.

**Prior art / dedup.** #29787 (open) already reuses top-k across MTP *draft-decode*
steps (`index_share_for_mtp_iteration`). This generalizes the *same proven mechanism*
to the target decode loop.

**Risk.** Accuracy — stale top-k on rapidly-shifting context changes selection and
outputs. Needs an accuracy sweep (gsm8k/AIME) over N + a refresh policy; flag-gated.

**Definition of done.** Decode throughput gain with accuracy within threshold across
tested N; refresh policy documented; flag-gated, default-off.

**References.** `dsa_indexer.py`; #29787.

---

## M10 — MXFP4 index-K cache for GLM-5.2 / DeepSeek-V3.2 (port from V4)

**Motivation.** The DSA indexer K cache is stored FP8 (128 B fp8 key + 4 B fp32 scale
= 132 B/token/layer). DeepSeek-V4 has an opt-in FP4 indexer
(`--enable-deepseek-v4-fp4-indexer`, #26209) storing index-K in MXFP4 (E2M1 +
UE8M0/block-32) = 68 B/token/layer — a **~48% cut** — but it's not wired for the
V3.2/GLM dsa path. Stacks with N7 (dropping shared-layer index-K buffers).

**Proposal.** Port the V4 MXFP4 indexer (`dsv4/fp4_indexer.py` quantize + store
kernels, and the `fp8_fp4_paged_mqa_logits` logits path) to the shared DSA (V3.2/GLM)
indexer, behind an opt-in flag like the V4 version.

**Prior art / dedup.** #26209 (V4 FP4 indexer) is the source; this ports it to the dsa
path. **MXFP4 only:** the DeepGEMM `fp4` MQA-logits kernel (`sm100_mqa_logits.cuh`,
verified on the sgl-project/DeepGEMM `dev` branch) hardcodes E2M1 + `float_ue8m0_t` —
there is **no NVFP4 (E4M3-scale) mode**, so a finer-grained FP4 index would require a
net-new logits kernel and is explicitly out of scope here.

**Risk / accuracy.** #26209 measured **GSM8K 0.980 → 0.965** (~1.5-pt) and pass@16
0.944 → 0.934 on V4; the coarse UE8M0 block-scale perturbs top-k selection. Opt-in,
default-off.

**Definition of done.** Index-K cache reduced ~48% (larger `max_total_num_tokens` at
fixed mem-fraction), accuracy within the V4-measured envelope on GLM-5.2/V3.2, opt-in
flag.

**References.** `dsv4/fp4_indexer.py`, `dsa_indexer.py`, `DSATokenToKVPool`
(`memory_pool.py`); #26209; N7.
