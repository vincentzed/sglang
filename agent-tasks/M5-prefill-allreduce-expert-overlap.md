# Task: Overlap the exposed prefill TP all-reduce with compute (AsyncTP / two-batch overlap)

On long-context DSA prefill (DeepSeek-V3.2 / GLM-5.2, tensor-parallel), the TP
all-reduce runs as a plain, **exposed** NCCL ring collective — 13% of GPU time at
80k on TP4. Decode already hides its all-reduce via MNNVL one-shot fusion, but
prefill's token count blows past the fusion cap, so it falls back to serial ring.
Recover that time by overlapping the collective with adjacent compute — either
AsyncTP (rewrite AllReduce → ReduceScatter + local-norm + AllGather so it overlaps
the next GEMM) or two-batch overlap (one micro-batch's comm ∥ the other's experts).

---

## TL;DR for the agent

You are a peer engineer landing fresh on SGLang. You own this: prove the all-reduce
is genuinely exposed serial time, prove a fused/overlapped path actually beats plain
ring at long-context prefill **before writing anything**, then implement and verify.

**One-line opportunity.** The 80k GLM-5.2-NVFP4 prefill trace (TP4) shows
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` at **13% / 414 ms** via
`parallel_state.py:_all_reduce_in_place` — an exposed, unfused TP all-reduce sitting
serially on the critical path. Decode hides its AR with a MNNVL one-shot fusion
kernel (`trtllm_mnnvl…oneshotAllreduceFusionKernel`, ~9.5%), but prefill's per-chunk
token count (chunked-prefill 16384 ≫ the 2048 fusion cap) routes it to plain ring.

**Strong-prior hypothesis (must be proven, not assumed).** Part of that 13% is
recoverable by overlapping the collective with the next GEMM / the other
micro-batch's expert compute. AsyncTP and two-batch-overlap are the two established
mechanisms.

**What we do NOT trust / do NOT anchor on:**
- **Do NOT assume the 13% is fully recoverable.** At large, bandwidth-bound token
  counts a plain ring all-reduce is often already near link-saturation, and fused
  AR paths frequently *lose* there (they win at small, latency-bound decode sizes —
  which is exactly why the fusion cap is 2048). The 13% is *potential*, not banked.
  Step 1 must A/B a real overlap path vs ring at 80k and show a win first.
- **Do NOT reuse the FlashInfer AR+RMSNorm fusion (N3) here.** It is capped at
  `FUSE_ALLREDUCE_MAX_BATCH_SIZE = 2048` (`communicator.py:160,170`) precisely because
  it is a decode-shaped one/two-shot kernel; 80k prefill exceeds it by design. This
  task is a *different* mechanism (overlap, not one-shot fusion).
- **Do NOT trust the cited line numbers verbatim** — grep by symbol; the repo moves.

**The job in one breath:** confirm the AR is exposed serial time in the trace →
A/B an overlap path vs ring at 80k to prove a win exists → pick AsyncTP vs TBO with
evidence → implement pure-TP → verify bit-identical outputs + prefill throughput gain.

**Out of scope:** pure-TP only for the first cut — **do NOT** attempt DP-attention;
there is no DP-attn AR-fusion/overlap path today (`is_dp_attention_enabled()` hard-
disables the existing fusion at `communicator.py:171,187,274`), so DP-attn is a
separate follow-up. Do NOT touch decode (already MNNVL-fused). Do NOT change the
2048 cap (that's N3's territory and a separate tradeoff).

---

## Background you need (no prior context assumed)

### The exposed prefill all-reduce
DSA models (DeepSeek-V3.2 / GLM-5.x) run TP all-reduce after the attention/MoE
output projections. On the current code the fused path
`apply_flashinfer_allreduce_fusion(batch_size)` (`communicator.py:163`) only engages
when **all** of: `batch_size <= FUSE_ALLREDUCE_MAX_BATCH_SIZE` (=2048, `:160,170`),
`not is_dp_attention_enabled()` (`:171`), and `flashinfer_allreduce_fusion_backend is
not None` (`:172`). Long-context prefill (chunked to 16384) fails the size gate, so
the reduce falls to `parallel_state.py:_all_reduce_in_place` (`:773`) → plain NCCL
ring (`RING_LL`) — serial, unoverlapped. That is the 13% in the trace.

### The two overlap mechanisms
- **AsyncTP.** Rewrite `AllReduce(x)` as `ReduceScatter(x) → local op (e.g. RMSNorm on
  the local shard) → AllGather`, so the two collectives can be pipelined with the
  producing/consuming GEMMs (a GEMM's output tiles reduce-scatter as they complete;
  the next GEMM consumes all-gathered tiles as they arrive). The building blocks are
  already present: `parallel_state.py` has `reduce_scatter`/`reduce_scatterv` (`:908,
  917`) and `all_gather`/`all_gatherv` (`:1037,1111`). vLLM's `fuse_gemm_comms`
  (AsyncTP pass, symmetric-memory collectives) is the upstream analog.
- **Two-batch overlap (TBO).** Split the prefill batch into two micro-batches and
  overlap one's communication with the other's attention/expert compute. **#28639**
  ("add `ag_gemm` and `moe_rs` overlap kernels for dsv4 prefill", open) is direct
  prior art — an all-gather-GEMM and MoE-reduce-scatter overlap for the V4 prefill
  path; adapt/validate it for the V3.2/GLM DSA path.

### Why this is prefill-specific
Decode is fully CUDA-graphed and already uses MNNVL one-shot AR fusion; its AR is
hidden. Only prefill (eager, large token count, over the fusion cap) exposes the ring.

---

## Plan (do these in order)

### Step 0 — Environment & access
- Repo: `git@github.com:vincentzed/sglang.git` (or upstream). Branch off latest `main`.
- Hardware: a TP≥4 Blackwell node (B200/B300) with NVLink/MNNVL. GPU required.
- Models: `zai-org/GLM-5.2` or `nvidia/GLM-5.2-NVFP4`, and `deepseek-ai/DeepSeek-V3.2-Exp`.
  If a checkpoint gates/401s, STOP and report.
- **Deliverable:** env note (node, TP size, NVLink topology, checkpoints resolved).

### Step 1 — Settling experiment (the make-or-break; do this first)
Prove there is a win before building anything.
- Capture an 80k prefill trace (pure TP4) and confirm the all-reduce is **serial** on
  the critical path (a `RING_LL` kernel with no overlapping compute in its window),
  matching the 13%.
- **A/B a real overlap path vs ring at 80k.** Cheapest proxy first: force the existing
  FlashInfer AR-fusion on for a large chunk by temporarily raising
  `FUSE_ALLREDUCE_MAX_BATCH_SIZE`, and/or prototype an AsyncTP RS→norm→AG rewrite for
  one layer, and measure prefill time vs ring. Guiding question: does the overlap beat
  ring at this token count, or is ring already ~link-saturated (in which case M5 is
  dead)? Restart the server between measured runs; gate on the confound-free signal
  (per-chunk prefill kernel time, not end-to-end noise).
- **Deliverable:** trace evidence the AR is exposed + an A/B number (overlap vs ring at
  80k). If overlap does not beat ring, STOP and report M5 as not worth it.

### Step 2 — Choose AsyncTP vs TBO
Given Step 1, pick the mechanism with evidence:
- AsyncTP fits when the AR sits between two GEMMs whose tiles can be pipelined; TBO
  fits when there are two independent micro-batches to interleave (better for the
  MoE/expert-heavy path, and #28639 already prototypes it for V4).
- **Deliverable:** a one-paragraph fix-mechanism verdict (AsyncTP vs TBO) with the
  Step-1 evidence and the reason the other was rejected.

### Step 3 — Implement (constraints, not a patch)
- Pure-TP path only. Reuse the existing `reduce_scatter*` / `all_gather*` primitives
  (`parallel_state.py`) rather than new collectives.
- If adapting #28639, keep its op structure; if AsyncTP, mirror vLLM's RS→norm→AG
  rewrite and ensure the local op (RMSNorm) operates on the correct shard.
- **Invariants:** numerically identical result to the plain all-reduce (RS+AG is exact
  in principle — verify no fp accumulation-order regression beyond noise). Do NOT
  change the decode path, the 2048 cap, or any DP-attention code.
- **Deliverable:** the diff, compiling clean, pure-TP.

### Step 4 — Verify: correctness + throughput
- **Correctness:** GSM8K (full 1319, 20-shot) on GLM-5.2 and V3.2 identical to `main`.
- **Throughput:** long-context prefill throughput / TTFT improvement vs `main`
  (`bench_one_batch_server` at input≈8k–80k, and `bench_serving` long-context), pure TP.
- **No DP-attn break:** confirm a DP-attention launch still runs correctly (the new
  path must be gated off under DP-attn).
- **Deliverable:** before/after prefill throughput + TTFT, GSM8K parity table, DP-attn
  smoke pass.

### Step 5 — Hand back
Open a PR to `sgl-project/sglang`: *"[DSA] Overlap long-context prefill TP all-reduce
with compute (AsyncTP / TBO)"*. Body: one-line what+why, the Step-1 A/B proving the
win, before/after throughput, GSM8K parity, and the pure-TP / DP-attn-out-of-scope
note. Reference #28639, N3, vLLM AsyncTP.
- **Deliverable:** the PR URL.

---

## Definition of done
- Trace evidence the prefill AR is exposed serial time (Step 1), **and an A/B showing
  the overlap path beats plain ring at 80k** — if it doesn't, the done-state is a
  written "M5 not worth it: ring is already ~saturated at long context" with the numbers.
- With the change: measurable long-context prefill throughput/TTFT gain, **bit-identical
  (within-noise) GSM8K** on GLM-5.2 and V3.2.
- DP-attention still correct (new path gated off there).
- No change to decode, the 2048 fusion cap, or DP-attn code.

## Deliverables
- Diff (AsyncTP RS→norm→AG rewrite, or the adapted #28639 TBO overlap), pure-TP.
- Step-1 A/B (overlap vs ring at 80k) + trace evidence.
- Before/after prefill throughput/TTFT + GSM8K parity (GLM-5.2 + V3.2).
- Upstream PR.

## Constraints / notes
- Scratch under repo cwd (`scratch/`), never `/tmp`. Run `pre-commit run`; don't skip hooks.
- Restart the server between measured runs; report the hardware-independent signal
  (per-chunk prefill kernel time), not just end-to-end (which is noisy at 80k).
- The 2048 cap is N3's lever, not this one — do not conflate. This is overlap, not
  one-shot fusion.

---

## Testing commands

> M5 targets **DeepSeek-V3.2 / GLM-5.x (DSA)** long-context prefill on SM100. DeepSeek-V4
> is a **separate arch** — the V4 commands are regression-safety only.

### A/B — overlap vs ring at long-context prefill (the decisive test)
```bash
# Baseline (plain ring, current behavior):
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --chunked-prefill-size 16384 --port 30000            # (no --enable-dp-attention)
python3 -m sglang.bench_one_batch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --batch-size 1 --input-len 80000 --output-len 8

# After the overlap change (same launch), compare prefill/TTFT.
# Confound-free: capture --profile-by-stage EXTEND traces both ways and compare the
# AllReduce window (should shrink / overlap with GEMM).
```

### Correctness — GLM-5.2 + DeepSeek-V3.2 (must stay identical)
```bash
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code --tp 4 --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000

python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.2-Exp --trust-remote-code --tp 8 --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
```

### DP-attention smoke (must not break — new path gated off)
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.2-Exp --trust-remote-code \
  --tp 8 --enable-dp-attention --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --num-shots 20 --port 30000
```

### Regression-safety — DeepSeek-V4 (separate arch)
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash --trust-remote-code --tp 4 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001
# PASS = V4 GSM8K + prefill perf unchanged vs main.
```

---

## References
- PR #28639 — `ag_gemm` + `moe_rs` overlap kernels for dsv4 prefill (open) — direct TBO prior art.
- N3 — FlashInfer AR+RMSNorm fusion (capped at `FUSE_ALLREDUCE_MAX_BATCH_SIZE=2048`; does not cover long prefill).
- vLLM AsyncTP — `fuse_gemm_comms` / `SequenceParallelismPass` (RS→norm→AG + symmetric-memory collective overlap).
- Code: `python/sglang/srt/layers/communicator.py` (`apply_flashinfer_allreduce_fusion`, the cap + gates),
  `python/sglang/srt/distributed/parallel_state.py` (`_all_reduce_in_place`, `reduce_scatter*`, `all_gather*`).
