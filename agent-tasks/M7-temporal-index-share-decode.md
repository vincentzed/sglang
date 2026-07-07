# Task: Reuse the DSA indexer top-k across consecutive target decode steps (temporal index-share)

Skip the DSA lightning-indexer's per-step top-k selection on N-1 of every N target
decode steps by carrying the previous step's top-k forward, gated by a cheap
refresh/acceptance check. The indexer select machinery (`topk_transform_decode` +
`fp8_paged_mqa_logits`) is ~8.6% of decode GPU time on GLM-5.2-NVFP4 at long context,
and for a stable context the selection drifts slowly step-to-step — so most of that
recompute is redundant. **This is accuracy-gated: prove the drift is small before
building anything.**

---

## TL;DR for the agent

You are a peer engineer landing fresh on SGLang. You own this end to end: first
*measure* whether temporal reuse is even safe (top-k drift across steps), then — only
if it is — design the refresh policy, implement it flag-gated, and prove decode
throughput up with accuracy unchanged.

**One-line opportunity.** DSA (DeepSeek Sparse Attention; DeepSeek-V3.2 / GLM-5.x) runs
the indexer every decode step to pick the top-k KV positions each query attends to. On
GLM-5.2-NVFP4 80k decode (TP4, from profiling) that select path is
`topk_transform_decode` **7.3%** + `deep_gemm::sm100_fp8_paged_mqa_logits` **1.3%** per
step. For a context that isn't changing much, the selected top-k set barely moves
between adjacent steps, so recomputing it every step is largely wasted work.

**Strong-prior hypothesis (UNPROVEN — Step 1 must confirm it).** The top-k set overlap
between step *t* and step *t+N* is high for small N, so reusing the step-*t* top-k for
the next N-1 steps changes outputs negligibly while removing the indexer logits + top-k
on those steps. If the drift turns out to be large, this idea is dead — do not force it.

**What we do NOT trust / do NOT anchor on:**
- **Do NOT assume a fixed N is safe.** Stale top-k on a rapidly-shifting context (topic
  change, long generation drifting off the prompt) changes *which* KV attention sees →
  changes outputs. The whole idea rests on the empirical drift, which is *content-
  dependent*. Treat "N=4 is fine" as a hypothesis to disprove, not a default.
- **Do NOT anchor on the throughput number before the accuracy gate.** 8.6%/step is the
  ceiling only if reuse is free; a drift-triggered refresh that fires often erodes it.
  Report the *net* gain after the refresh policy, not the raw skip fraction.
- **Do NOT trust PR/file line numbers verbatim** — grep by symbol; the repo moves fast.

**The job in one breath:** measure step-to-step top-k drift on real prompts (settling
experiment) → if small, design a refresh/acceptance policy → implement it flag-gated,
carrying top-k forward the way the layer-level share already does → verify decode TPOT
improves with gsm8k/AIME within threshold across the tested N.

**Out of scope:**
- **Do NOT re-implement MTP/draft top-k reuse.** #29787 (open) already reuses the top-k
  across MTP *draft-decode* steps (`index_share_for_mtp_iteration`). This task is the
  *target* decode loop, a different axis — build on #29787's carry mechanism, don't
  duplicate it, and don't touch the draft path.
- Do NOT change the layer-level index-share (`skip_topk`) logic, the indexer kernels, or
  the prefill path (prefill recomputes per chunk; this is decode-only).

---

## Background you need (no prior context assumed)

### The DSA indexer and its two existing "share" axes
DeepSeek-V3.2 / GLM-5.x use a per-layer **indexer** (`Indexer`,
`python/sglang/srt/layers/attention/dsa/dsa_indexer.py`) that projects Q/K, computes
`fp8_mqa_logits`, and selects the top-`index_topk` KV positions per query; sparse MLA
attention then runs over those positions. There are already **two** reuse axes:

1. **Across layers (`skip_topk` / index-share).** With `index_topk_freq > 1`, "shared"
   layers reuse the *producer* layer's top-k instead of computing their own. The gate is
   `should_run_indexer(...)` (`forward_mla.py:181-200`), and the carried indices flow
   through `prev_topk_indices` (mirrored into shared layers at `forward_mla.py:377-403`).
2. **Across MTP draft steps.** #29787 (open) carries the target's top-k into the MTP
   *draft-decode* iterations (`index_share_for_mtp_iteration`) rather than re-running the
   indexer each draft step.

**M7 adds a third axis: across consecutive *target* decode steps.** No layer or draft
sharing removes the *per-step* recompute in ordinary (non-draft) decode — that's the gap.

### The per-step decode select path (what M7 skips)
On the CUDA decode path, `Indexer._get_topk_paged` (`dsa_indexer.py:775`) runs
`deep_gemm.fp8_paged_mqa_logits` (`:889` / `:901`) then `metadata.topk_transform(logits,
self.index_topk)` (`:913`). `topk_transform` itself (`:297`) is already fused/tuned
(`fast_topk_transform_*`, #26788, #30274) — so the lever is **calling it less often**,
not making it faster. Everything decode runs inside a full CUDA graph
(`full_cuda_graph_backend.py` replay), so the reuse must be graph-capture-safe (a static
"reuse vs recompute" branch, or a counter refreshed in-place on replay — do not add a
data-dependent host branch inside the captured region).

### The carry mechanism to mirror
The layer-level share already carries a top-k tensor between call sites via
`prev_topk_indices` and mirrors it into the KV/metadata; the TP broadcast of the
finalized top-k is `broadcast_indexer_topk_from_rank0_` (`dsa_indexer.py:207-241`,
gated by `SGLANG_DSA_TOPK_BROADCAST`). M7's step-carry should reuse the same tensor
plumbing, one level up (persist the top-k in the attention metadata / a small ring
across decode steps).

---

## Plan (do these in order)

### Step 0 — Environment & access
- Repo: `git@github.com:vincentzed/sglang.git` (or upstream `sgl-project/sglang`), branch
  off latest `main`. GPU required (CUDA DSA decode); a B200/H200 with ≥4 GPUs for TP4.
- Models: `zai-org/GLM-5.2` (or `nvidia/GLM-5.2-NVFP4`) and `deepseek-ai/DeepSeek-V3.2-Exp`
  — both DSA. Confirm decode runs the CUDA `_get_topk_paged` path (not HiSparse/tilelang).
- **Deliverable:** env note + confirmation the decode indexer path is the `deep_gemm`
  paged-MQA-logits one.

### Step 1 — Settling experiment: measure top-k drift (THE make-or-break)
Before touching any runtime code, answer empirically: *how fast does the selected top-k
actually change between decode steps?* Instrument `_get_topk_paged` (read-only capture,
no behavior change) to dump, per decode step and per layer, the selected top-k index set.
Then compute the **Jaccard overlap** (|topk_t ∩ topk_{t+n}| / index_topk) for n = 1..8
across a mix of prompts: stable/long-context (RAG-style), and drifting (open-ended
generation, topic switches). Break it down by layer (early vs late) and by how far into
generation.
- Guiding questions: Is overlap ≥ ~0.95 for n up to some N? Does it collapse on drifting
  prompts or at generation boundaries? Are some layers far less stable than others (→ per-
  layer N)? Where would a stale top-k first change the *sampled token*?
- **Deliverable:** a drift table (overlap vs n, by prompt-type and layer) and a verdict:
  **is temporal reuse safe, and for what N / under what refresh trigger?** If overlap is
  low even at n=2, STOP and report M7 as dead — do not implement.

### Step 2 — Design the refresh / acceptance policy
Using Step 1's data, choose between: (a) **recompute-every-N** (simplest, graph-safe: a
static counter, reuse for N-1 then refresh); (b) **drift-triggered** (refresh when a cheap
signal — e.g. the score of the carried top-k's boundary position, or a periodic full
recompute + compare — indicates the set moved). Prefer the simplest policy Step 1 shows is
safe; graph-capture-safety favors (a).
- **Deliverable:** a one-page policy spec (N, refresh trigger, per-layer vs global,
  graph-capture handling) justified by the Step-1 numbers.

### Step 3 — Implement (constraints, not a patch)
- Persist the finalized per-layer top-k across decode steps (reuse the existing top-k
  tensor plumbing / `prev_topk_indices` style), and skip `fp8_paged_mqa_logits` +
  `topk_transform` on reuse steps in `_get_topk_paged`.
- **Flag-gated, default-off** (new `SGLANG_*` env; follow env-var conventions). Mirror
  #29787's carry mechanism; do not fork it.
- **Graph-capture-safe:** the reuse/recompute decision must be a static or in-place-
  refreshed counter (`.copy_()` on replay), not a data-dependent host branch inside the
  captured region.
- **Invariants:** non-reuse steps behave exactly as today; prefill untouched; MTP draft
  path (#29787) untouched; TP top-k broadcast still consistent across ranks on refresh
  steps.
- **Deliverable:** the diff (indexer step-carry + gate), compiling/importing clean.

### Step 4 — Verify: throughput up, accuracy held
- **Accuracy (the gate):** gsm8k (full 1319) and AIME across the tested N on both GLM-5.2
  and V3.2 — must stay within threshold vs N=1 (no reuse). Sweep N to find the largest N
  that holds accuracy.
- **Throughput:** decode TPOT / output tok/s improvement at long context via
  `bench_one_batch_server`, reporting the **net** gain after the refresh policy (not the
  raw skip fraction).
- **Deliverable:** an N-sweep table (accuracy + TPOT per N) identifying the safe operating
  point, on GLM-5.2 and V3.2.

### Step 5 — Hand back
Open a PR: *"[DSA] Temporal index-share: reuse indexer top-k across N target decode
steps (opt-in)"*. Body: the drift measurement, the policy, the N-sweep accuracy/TPOT
table, the flag. Reference #29787.
- **Deliverable:** PR URL.

---

## Definition of done
- Step-1 drift table exists and shows temporal reuse is safe for some N ≥ 2 (or the task
  is closed as dead with that evidence).
- Decode TPOT improves measurably at the chosen N, with **gsm8k/AIME within threshold vs
  N=1** on GLM-5.2 and DeepSeek-V3.2 (evidence-gated, not vibes).
- Graph-capture-safe; flag-gated default-off; MTP-draft path (#29787) and prefill
  untouched; TP top-k stays consistent across ranks.

## Deliverables
- Drift-measurement instrument + table (Step 1).
- Refresh-policy spec (Step 2).
- Implementation diff + new env flag.
- N-sweep accuracy/TPOT tables (GLM-5.2, V3.2).
- Upstream PR.

## Constraints / notes
- Scratch under repo cwd (`scratch/`), never `/tmp`. Run `pre-commit run`; don't skip hooks.
- New env var must follow the `SGLANG_*` env-var conventions (register in `environ.py`).
- Decode is fully CUDA-graphed — validate reuse under graph replay, not just eager.
- This is **exploratory**: Step 1 gates everything. Do not implement before the drift
  measurement justifies it.

---

## Testing commands

> M7 targets **DeepSeek-V3.2 / GLM-5.2 (DSA)** decode. DeepSeek-V4 is a separate arch —
> the V4 commands are **regression-safety** only.

### Accuracy N-sweep (the gate) — GLM-5.2 / DeepSeek-V3.2
```bash
# Baseline (no reuse, N=1):
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# Then relaunch with the new flag at N=2,4,8 and re-run; accuracy must hold vs N=1.
# SGLANG_DSA_TEMPORAL_TOPK_REUSE=<N> python3 -m sglang.launch_server ... (flag name TBD)
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --trust-remote-code --tp 8 --enable-dp-attention --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# AIME (harder, sensitive to selection drift): nvidia/GLM-5.2-NVFP4, 16 repeats, temp 1.0.
```

### Decode TPOT — net gain after refresh policy
```bash
python3 -m sglang.bench_one_batch_server --model-path zai-org/GLM-5.2 \
  --trust-remote-code --tp 4 --batch-size 1 --input-len 80000 --output-len 1024
# Compare N=1 vs the chosen N; report median TPOT / output tok/s.
```

### Regression-safety — DeepSeek-V4 (separate arch; prove untouched)
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tp 4 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001
# PASS = V4 decode unchanged vs main (M7 must not touch the V4 native indexer/MTP path).
```

---

## References
- PR #29787 (open) — reuses top-k across MTP *draft-decode* steps
  (`index_share_for_mtp_iteration`); the proven carry mechanism M7 generalizes to target
  decode.
- PR #26788 / #30274 — the already-fused/tuned `topk_transform` (why the lever is *fewer
  calls*, not a faster kernel).
- Code: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`_get_topk_paged` ~775,
  `topk_transform` ~297, `fp8_paged_mqa_logits` ~889/901, `broadcast_indexer_topk_*`
  ~207-241); `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
  (`should_run_indexer` ~181, `prev_topk_indices` carry ~377-403); `environ.py` (new flag).
