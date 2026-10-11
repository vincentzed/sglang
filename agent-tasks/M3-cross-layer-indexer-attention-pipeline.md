# Task: Pipeline the DSA indexer across layers so it overlaps the previous layer's sparse-MLA attention

Under DSA sparse attention, each layer runs `indexer (select top-k) → sparse-MLA
attention` strictly in series. Overlap them across layers: while layer L's attention
runs, precompute/prefetch what layer L+1's attention needs (start the producer layer's
indexer, or prefetch the reused KV pages for `skip_topk` shared layers) on a side
stream. **First prove the exposed serial gap is real** — this is a speculative overlap,
gated on measurement before any kernel work.

---

## TL;DR for the agent

You are a peer engineer landing fresh on SGLang. You own this end to end: measure
whether there is a real serial gap to close, then (only if there is) design and land a
cross-layer indexer/attention pipeline that preserves outputs and CUDA-graph stream
counts.

**One-line opportunity.** In the DSA path (DeepSeek-V3.2 / GLM-5.x), a layer's
sparse-MLA attention depends on the indexer's top-k, so the two run serially. Profiling
GLM-5.2-NVFP4 at 80k prefill (TP4) shows the indexer-selection machinery —
`deep_gemm::sm100_fp8_mqa_logits` (8.3%) + `topk_transform_prefill` (6.2%) + gather
(1.8%) ≈ **~16% of GPU time** — sitting in front of the ~33% attention. Under
index-share, most layers are `skip_topk` shared layers that run **no** indexer at all,
so their "indexer slot" is idle time that a cross-layer pipeline could fill with
next-layer prefetch/precompute.

**Strong-prior hypothesis (UNPROVEN — this is the point of Step 1).** A meaningful
fraction of that ~16% is *exposed* serial time (the attention kernel waits on the
indexer with no other work on the SM), and overlapping the next layer's indexer/prefetch
with the current layer's attention recovers it. This is a novel overlap, not a
known-good win — do not assume the gap exists until you measure it.

**What we do NOT trust / do NOT anchor on:**
- **Do NOT assume the ~16% is fully recoverable.** Part of it may already overlap the
  attention on the GPU (concurrent kernels), and part is on the critical dependency
  chain and cannot move. Step 1 must quantify the *exposed* serial fraction; if it's
  small, this task is not worth building — say so and stop.
- **Do NOT reintroduce the CUDA-graph stream explosion.** The indexer already uses a
  dual-stream Q‖K overlap; #30025 fixed a bug where its issue order made
  `cudaGraphInstantiate` reserve ~22 side streams (one per layer) instead of 2. A
  cross-layer pipeline adds *more* cross-stream forks — it must reuse a bounded set of
  streams and keep the captured-graph stream count ≤ current.
- **Do NOT trust the file:line anchors verbatim** — grep by symbol; this repo moves fast.

**The job in one breath:** measure the exposed serial indexer→attention gap at long
prefill → confirm the shared-layer slot is idle and identify the prefetch/precompute
target for L+1 → (if the gap is real) implement a bounded-stream cross-layer pipeline →
verify bit-identical outputs, stream count ≤ current, and a real prefill TTFT delta.

**Out of scope:** do NOT change index-share / `skip_topk` selection, the indexer
kernels, or the top-k algorithm. Do NOT touch the DeepSeek-V4 native indexer (separate
arch). Do NOT rebuild the HiSparse prefetch path (see dedup below).

---

## Background you need (no prior context assumed)

### DSA / the lightning indexer
DeepSeek-V3.2 and GLM-5.x use **DSA (DeepSeek Sparse Attention)**, "NSA" in the tree. A
small per-layer **indexer** projects Q/K, applies RoPE (+ optional Hadamard),
FP8-quantizes, computes `fp8_mqa_logits` over the context, then selects the top-k KV
positions; sparse-MLA attention runs over the selected positions. The indexer is the
`Indexer` class in `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (class def
circa `:329`).

### The per-layer serial dependency
Each transformer layer does `indexer → attention`. The attention backend
(`python/sglang/srt/layers/attention/dsa_backend.py`) dispatches `forward_extend`
(circa `:1692`) → `_forward_trtllm` (`:1723`) / `_forward_standard_mha` (`:1762`) and
`forward_decode` (`:1927`), all of which consume the indexer's top-k. So the top-k is a
hard dependency of *this* layer's attention — you cannot overlap the indexer with its
*own* layer's attention, only with a *neighboring* layer's.

### Index-share / `skip_topk` (why there's idle headroom)
`dsa_layer_skips_topk(config, layer_id)` (`configs/model_config.py:180`, driven by
`index_topk_pattern`/`index_topk_freq`) marks some layers `skip_topk=True`; they reuse
the previous producer layer's top-k and carry no indexer weights. The gate
`should_run_indexer` (`forward_mla.py:181`; call sites `forward_mha.py:172`,
`forward_mla.py:377/403`; PR #29959) runs the indexer only on non-shared layers. With
`index_topk_freq>1`, ~48% of layers are shared → their indexer slot is idle, and the
comment at `model_config.py:212` ("those still get a slot") notes shared layers are
still allocated slots.

### The existing dual-stream overlap (and its landmine)
The indexer already overlaps Q‖K within a layer via `alt_stream` (ctor arg
`dsa_indexer.py:356/371`; `_get_q_k_bf16` `:516` with `alt_stream.wait_stream` `:527`,
`with torch.cuda.stream(self.alt_stream)` `:539`, join `:553`; and
`_fused_q_prepare_and_store` `:682`, dual-stream block `:729-744`). PR **#30025**
reordered the issue sequence so graph capture reuses one side stream (22→2). Any
cross-layer pipeline must respect that: reuse a bounded stream set, keep issue order
capture-friendly.

---

## Plan (do these in order)

### Step 0 — Environment & access
- Repo: `git@github.com:vincentzed/sglang.git` (or upstream). Branch off latest `main`.
- Hardware: a Blackwell node (B200/B300), ≥4 GPUs (TP4) for the 80k prefill repro. GPU
  required — DSA kernels are CUDA JIT.
- Models: `zai-org/GLM-5.2` (or `nvidia/GLM-5.2-NVFP4`) and `deepseek-ai/DeepSeek-V3.2-Exp`.
  If a checkpoint gates/401s, STOP and report. Confirm index-share is active (log the
  `skip_topk` count / resolved `index_topk_freq`).
- **Deliverable:** env note + confirmation index-share is on.

### Step 1 — Settling experiment: is the serial gap real? (do this FIRST)
Capture an 80k prefill torch-profiler trace (per the profiler skill; `--profile-by-stage`,
prefill workload) and quantify, per DSA layer, the **exposed** time between the indexer
kernels (`sm100_fp8_mqa_logits`, `topk_transform_prefill`) and the following attention
kernel (`fmhaSm100…`) — i.e. GPU-idle or serialized time on the dependency edge, not
wall-clock of the kernels themselves. Compare producer vs shared (`skip_topk`) layers.
Guiding question: of the ~16% indexer-selection time, how much is on the critical path
with the SM otherwise idle (recoverable) vs already concurrent with other work
(not)? If the exposed fraction is small (say <~3-4% of prefill), **this task is not
worth building — report that and stop.**
- **Deliverable:** a per-layer exposed-gap measurement with a clear verdict: real
  headroom (proceed) or not (stop).

### Step 2 — Confirm the shared-layer slot is idle and pin the prefetch target
Instrument the indexer/attention path to confirm `skip_topk` layers run no indexer
(consistent with `should_run_indexer`) and that their execution window has spare
stream/SM capacity. Identify precisely what L+1's attention needs that can be produced
early: for a producer L+1, the indexer top-k (needs L+1's `hidden_states`, which isn't
ready until L completes — so cross-layer indexer *compute* may be blocked by the
residual stream; verify); for a shared L+1, the *carried* top-k already exists, so the
prefetchable item is the KV pages, which is what HiSparse already does (see dedup).
Guiding question: is the realizable pipeline "compute next indexer early" (likely
blocked by the residual dependency) or "prefetch next layer's KV early" (more likely
feasible)? Let the evidence pick.
- **Deliverable:** a fix-shape verdict — what specifically can be moved earlier, and for
  which layer class — with the dependency that makes the other option infeasible.

### Step 3 — Implement (constraints, not a patch)
- Use a **bounded** side-stream set + CUDA events, mirroring the existing `alt_stream`
  pattern; do NOT allocate a new stream per layer. Issue order must keep
  `cudaGraphInstantiate` at ≤ the current side-stream count.
- Keep it gated behind a flag (default-off) until proven.
- **Invariants:** outputs bit-identical; the producer/attention dependency is never
  violated (never read a top-k / KV before it's produced); GLM-5.2 (already-fused
  indexer) and V3.2 paths both preserved.
- **Deliverable:** the diff (indexer/backend + stream plumbing), compiling clean.

### Step 4 — Verify
- **Correctness:** GSM8K identical to `main` on GLM-5.2 and DeepSeek-V3.2; ideally
  bit-identical per-layer top-k on a fixed prompt.
- **Graph safety:** captured-graph side-stream count ≤ current (the #30025 guardrail) —
  measure it explicitly.
- **Perf:** prefill TTFT improvement at ≥32k that matches the Step-1 recoverable
  estimate; no decode regression.
- **Deliverable:** before/after TTFT, stream-count check, GSM8K parity table.

### Step 5 — Hand back
Open a PR: *"[DSA] Cross-layer indexer/attention prefetch pipeline (prefill)."* Body:
the Step-1 exposed-gap evidence, the fix-shape verdict, before/after TTFT, the
stream-count guardrail result, GSM8K parity. Reference #29637, #28523, #30025, #29959.
- **Deliverable:** PR URL.

---

## Definition of done
- Step-1 measurement states the **exposed** serial indexer→attention fraction with a
  proceed/stop verdict (gate on evidence, not vibes).
- If shipped: prefill TTFT improvement at long context, **outputs bit-identical**,
  captured-graph side-stream count **≤ current** (#30025 not regressed), GSM8K unchanged
  on GLM-5.2 and V3.2, flag-gated.
- No change to index-share, indexer kernels, top-k algorithm, or the V4 indexer.

## Deliverables
- Per-layer exposed-gap measurement + verdict.
- Cross-layer pipeline diff (indexer/backend + bounded-stream plumbing) behind a flag.
- Before/after TTFT, stream-count guardrail evidence, GSM8K parity table.
- Upstream PR.

## Constraints / notes
- Scratch under repo cwd (`scratch/`), never `/tmp`. Run `pre-commit run`; don't skip hooks.
- The indexer kernels are JIT — clear the JIT cache if a stream/plumbing edit doesn't take.
- Coordinate with open HiSparse-prefetch work (#29637, #28523) — they touch the same
  prefetch machinery; don't collide.

---

## Testing commands

> M3 targets the **DeepSeek-V3.2 / GLM-5.x (DSA)** on-device path. DeepSeek-V4 is a
> separate arch — the V4 commands are **regression-safety** only.

### Correctness — GLM-5.2 / DeepSeek-V3.2
```bash
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# DeepSeek-V3.2 (confirm index-share pattern active):
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --trust-remote-code --tp 8 --enable-dp-attention --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# DSA index-cache gate (index-share correctness):
python3 test/registered/8-gpu-models/test_deepseek_v32_indexcache.py
```

### Perf — prefill TTFT at long context (the target regime)
```bash
python3 -m sglang.bench_one_batch_server --model-path zai-org/GLM-5.2 \
  --trust-remote-code --tp 4 --batch-size 1 --input-len 80000 --output-len 8
# compare flag-on vs flag-off; TTFT is the metric.
```

### Regression-safety — DeepSeek-V4 (separate arch; prove untouched)
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tp 4 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001
```

---

## References
- PR #29637 — HiSparse prefetch via IndexShare (hides ~70% host-IO). **Dedup:** M3 is the
  on-device generalization of this idea; #29637 is the HiSparse host-offload path.
- PR #28523 — IndexCache shared-layer IO overlap (side-stream prefetch of shared-layer
  pages). Same dedup — HiSparse path; coordinate on the shared prefetch machinery.
- PR #30025 — DSA indexer dual-stream reorder (22→2 captured streams). The guardrail M3
  must not regress.
- PR #29959 — `should_run_indexer` gate (indexer runs only on non-shared layers).
- PR #27705 — the indexer Q/K fusion + `alt_stream` dual-stream pattern to mirror.
- Code: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`Indexer`,
  `_get_q_k_bf16`, `_fused_q_prepare_and_store`, `alt_stream`);
  `python/sglang/srt/layers/attention/dsa_backend.py` (`forward_extend`,
  `_forward_trtllm`, `_forward_standard_mha`, `forward_decode`);
  `python/sglang/srt/configs/model_config.py` (`dsa_layer_skips_topk`).
