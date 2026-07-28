# Task: Overlap the DSA indexer projection with the main-attention projection via green-context SM partitioning (prefill)

Run the DSA indexer's Q/K projection concurrently with the main MLA QKV projection on
**separate SM slices** (FlashInfer green contexts) during long-context prefill, instead
of serially. Both read `hidden_states` and are independent up to the point the logits
need the projected Q/K, so the exposed indexer projection time (~16% of 80k prefill)
can be hidden behind the main attention prep.

---

## TL;DR for the agent

You are a peer engineer landing fresh on SGLang. You own this end to end: prove the
overlap is real and worth it on the running system (this is the make-or-break),
size the SM partition, implement it behind a gate, and verify no regression.

**One-line opportunity.** In a DSA layer (DeepSeek-V3.2 / GLM-5.x), the indexer Q/K
projection and the main MLA QKV projection both consume `hidden_states` and are mutually
independent until the indexer logits/attention consume the projected tensors. On
GLM-5.2-NVFP4 80k prefill (TP4) the indexer selection machinery is ~16% of GPU time
(`deep_gemm::sm100_fp8_mqa_logits` 8.3% + `topk_transform_prefill` 6.2% + gather 1.8%),
and its projection runs serially in front of the main attention. Partitioning SMs so the
two projections run **concurrently** recovers that exposed serial time.

**Strong-prior hypothesis (SPECULATIVE — you must prove it before building).** The
indexer projection and main-attention projection are independent and can overlap; on a
big-kernel prefill the win requires **true SM co-residency (green contexts)**, not a
naive two-stream fork (which time-slices on one SM pool and barely helps when both sides
are large). This is unproven — Step 1 gates the whole task.

**What we do NOT trust / do NOT anchor on:**
- **Do NOT assume a plain `alt_stream` two-stream fork suffices.** A decode-only,
  capture-mode two-stream overlap of `q_b_proj` ∥ indexer **already exists**
  (`forward_mla.py`, "overlap q_b_proj and indexer during decode", circa line 363,
  gated on `get_is_capture_mode() and is_decode_or_idle()`). It does *not* cover eager
  prefill, and on prefill's large kernels two streams on one SM pool mostly serialize
  anyway. The new lever is **SM partitioning** for prefill — don't just reuse the
  decode two-stream path and call it done.
- **Do NOT trust that green contexts are free under CUDA graphs.** `server_args.py`
  (circa line 6551) warns CUDA Green Context has known CUDA-graph perf-degradation on
  torch 2.7.x–2.8.x. Verify the interaction on the target torch before wiring it into a
  captured path; prefill is eager so this may be moot there, but confirm.
- **Do NOT trust the cited line numbers verbatim** — grep by symbol; the repo moves fast.

**The job in one breath:** measure the exposed indexer-projection serial time at 32k/80k
prefill → A/B a green-context-partitioned indexer∥attention against serial (is the gain
real, and what SM split doesn't starve the indexer?) → if real, implement behind a gate
for long prefill only → verify bit-identical outputs and a real speedup with no
short-context regression.

**Out of scope:** do NOT touch the DeepSeek-V4 native indexer (separate arch); do NOT
change the existing decode two-stream overlap; do NOT change the indexer's internal Q∥K
dual-stream (#27705); do NOT change index-share / `skip_topk` logic or numerics.

---

## Background you need (no prior context assumed)

### The DSA layer structure
DeepSeek-V3.2 / GLM-5.x use **DSA (DeepSeek Sparse Attention)**. Per layer, a small
**indexer** projects Q/K (`Indexer` in
`python/sglang/srt/layers/attention/dsa/dsa_indexer.py`; `__init__` circa line 339;
projections `wq_b` / `wk_weights_proj` / `wk` / `weights_proj` circa lines 387–416),
applies RoPE + FP8 quant, computes `fp8_mqa_logits`, and selects the top-k KV positions.
Sparse-MLA attention then runs over the selection. The **main** attention path projects
Q/KV from the same `hidden_states` in `forward_absorb_prepare`
(`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`,
circa line 259): `q_b_proj`, `q_proj(hidden_states)` (~416), `kv_a_proj_with_mqa(hidden_states)` (~419),
and the indexer is invoked as `self.indexer(x=hidden_states, ...)` (~378/404).

### Why they can overlap
Both the indexer projection and the main QKV projection are pure functions of
`hidden_states`; nothing in the main projection depends on the indexer output until the
sparse attention consumes the top-k. So the two projection GEMMs are independent and can
execute concurrently, joining before the indexer logits and the attention.

### What already exists (and its limits)
- **Decode two-stream overlap:** `forward_mla.py` (~363) already runs `q_b_proj` on
  `alt_stream` while the indexer runs on the current stream — but only under
  `get_is_capture_mode()` (CUDA graph) and `is_decode_or_idle()`. Eager prefill (where
  the indexer is 16%) is not covered.
- **Indexer-internal Q∥K dual-stream:** PR #27705 overlaps the indexer's own key path
  with its query path on `alt_stream` (distinct — that's *inside* the indexer, not
  indexer-vs-main-attention).
- **Neither uses SM partitioning.** Two streams share one SM pool; on prefill's large
  projection GEMMs they largely time-slice rather than co-reside.

### The mechanism this task introduces
FlashInfer **green contexts** partition the GPU's SMs into disjoint slices with
dedicated streams, so two kernel families genuinely run at once on separate SMs:
`flashinfer/green_ctx.py` — `split_device_green_ctx` (circa line 126),
`split_device_green_ctx_by_sm_count` (circa line 196). It is **not currently wired into
the SRT DSA path** (only referenced in `server_args.py` caveats and
`multiplex/pdmux_context.py` for a different purpose). This task brings it to the
indexer∥attention split for prefill. See also the FlashInfer green-context overlap
family in the profiler fuse/overlap catalog.

---

## Plan (do these in order)

### Step 0 — Environment & access
- Repo: `git@github.com:vincentzed/sglang.git` (or upstream). Branch off latest `main`.
- Hardware: a Blackwell node (B200/B300), TP4, for long-context prefill. Record the
  torch version (the green-ctx CUDA-graph caveat is torch-2.7/2.8-specific).
- Models: `zai-org/GLM-5.2` (or `nvidia/GLM-5.2-NVFP4`) and `deepseek-ai/DeepSeek-V3.2-Exp`.
- **Deliverable:** environment note incl. torch version and a baseline prefill profile
  at 80k confirming the indexer projection + logits share (~16%).

### Step 1 — Settling experiment (prove the overlap is real; do this FIRST)
Before any integration, decide whether SM-partitioned concurrency actually beats serial:
- Build a standalone micro-bench: the indexer Q/K projection GEMM and the main MLA QKV
  projection GEMM at representative prefill shapes (bs1×80k, TP4 per-rank dims), run (a)
  serial, (b) two-stream on one SM pool, (c) green-context partitioned across two SM
  slices swept over split ratios (e.g. indexer gets 10–40% of SMs).
- Guiding questions: does (c) beat (a)? At what SM split is the indexer not starved and
  the main projection not throttled? Does the win survive at 32k, or only 80k+? What is
  the green-context creation/teardown overhead per forward, and can it be amortized
  (created once, reused across layers)?
- **Deliverable:** a table (serial vs two-stream vs green-ctx@ratios, at 32k/80k) with a
  go/no-go verdict and the chosen SM split. **If green-ctx does not beat serial, STOP
  and report — the task is dead.**

### Step 2 — Confirm independence + CUDA-graph compatibility
- Confirm no hidden data dependency between the two projections (both read
  `hidden_states`; the join point is before logits/attention).
- Determine whether prefill runs eager or under a captured graph in the target config,
  and whether green contexts are safe there on the target torch (the `server_args.py`
  caveat). Prefill is typically eager → likely fine; verify.
- **Deliverable:** a dependency/graph-compat note stating exactly where the join barrier
  goes and whether capture is involved.

### Step 3 — Implement (constraints, not a patch)
- Create the green-context SM partition **once** (reused across layers/forwards, not
  per-layer), with the Step-1 split ratio; run the indexer projection on the indexer
  slice and the main QKV projection on the main slice, join before the indexer logits.
- **Gate to long-context prefill only** (a token-count threshold) — below it, fall
  through to the existing serial/decode path unchanged.
- **Invariants:** bit-identical numerics (this is a scheduling change only); the decode
  two-stream path (~363) and the indexer-internal Q∥K dual-stream (#27705) untouched;
  no green context on a captured decode graph unless Step 2 proved it safe.
- **Deliverable:** the diff (green-ctx setup + the prefill projection split + the gate),
  compiling and importing clean.

### Step 4 — Verify: speedup, parity, no short-context regression
- **Speedup:** prefill TTFT / throughput improvement at ≥32k (bench vs `main`).
- **Parity:** GSM8K identical to `main` on GLM-5.2 and V3.2 (scheduling change ⇒
  bit-identical top-k on a fixed prompt is the strong check).
- **No short-context regression:** at small token counts the overlap flips negative
  (green-ctx overhead + under-utilized slices); confirm the gate leaves short prefill
  and decode on the fast serial path with zero delta.
- **Deliverable:** before/after prefill perf at 32k/80k, GSM8K parity table, and a
  short-context no-regression check.

### Step 5 — Hand back
Open a PR: *"[DSA] Green-context SM overlap of indexer projection with main-attention
projection (long prefill)"*. Body: one-line what+why, the Step-1 go/no-go table, the
chosen SM split, parity + perf, and the gate rationale. Reference #27705.
- **Deliverable:** the PR URL.

---

## Definition of done
- Step-1 evidence that green-context partitioning **beats serial** at ≥32k prefill, with
  the chosen SM split (or a documented no-go that kills the task).
- Measurable long-prefill speedup with **bit-identical outputs** on GLM-5.2 and V3.2.
- **No regression at short context / decode** (gated off there, zero delta).
- Existing decode two-stream overlap and indexer-internal Q∥K dual-stream unchanged;
  V4 indexer untouched.

## Deliverables
- Green-ctx overlap diff (setup + prefill projection split + token-count gate).
- Step-1 micro-bench table (serial vs two-stream vs green-ctx@ratios).
- Parity (GSM8K) + long-prefill perf + short-context no-regression evidence.
- Upstream PR.

## Constraints / notes
- Scratch under repo cwd (`scratch/`), never `/tmp`. Run `pre-commit run`; don't skip hooks.
- Create the green context **once and reuse** — per-layer creation will erase the gain.
- Respect the torch-2.7/2.8 green-ctx CUDA-graph caveat (`server_args.py` ~6551): do not
  put a green context inside a captured decode graph unless proven safe on the target torch.
- This is a **scheduling-only** change — any numeric delta means a real bug.

---

## Testing commands

> M4 targets the **DeepSeek-V3.2 / GLM-5.x (DSA)** prefill path on SM100. DeepSeek-V4 is
> a separate arch — the V4 command below is **regression-safety** only.

### Accuracy parity — GLM-5.2 / DeepSeek-V3.2
```bash
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# must equal main; ideally byte-identical per-layer top-k on a fixed prompt.

python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --trust-remote-code --tp 8 --enable-dp-attention --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
```

### Perf — long-context prefill (32k / 80k)
```bash
python3 -m sglang.bench_one_batch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --batch-size 1 --input-len 32768 --output-len 1
python3 -m sglang.bench_one_batch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --batch-size 1 --input-len 80000 --output-len 1
# compare gate-on vs gate-off; expect improvement at 80k, flat/positive at 32k.
```

### Short-context no-regression + decode
```bash
# small prefill + decode must show ZERO delta (gate leaves them on serial):
python3 -m sglang.bench_one_batch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --batch-size 8 --input-len 1024 --output-len 1024
```

### Regression-safety — DeepSeek-V4 (separate arch; prove untouched)
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tp 4 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001
```

---

## References
- PR #27705 — indexer-internal Q∥K dual-stream (distinct from this indexer∥main-attn overlap).
- Existing decode overlap: `forward_mla.py` "overlap q_b_proj and indexer during decode" (~363).
- Green contexts: `flashinfer/green_ctx.py` (`split_device_green_ctx` ~126,
  `split_device_green_ctx_by_sm_count` ~196); torch-2.7/2.8 CUDA-graph caveat in
  `python/sglang/srt/server_args.py` (~6551); prior green-ctx use in
  `python/sglang/srt/multiplex/pdmux_context.py`.
- Code anchors: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (`Indexer.__init__`
  ~339, projections ~387–416, `alt_stream` ~356/371, `_get_q_k_bf16` ~516);
  `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
  (`forward_absorb_prepare` ~259, indexer call ~378/404).
