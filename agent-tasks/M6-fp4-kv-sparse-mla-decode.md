# Task: FP4 KV cache for the SM100 sparse-MLA **decode** path (DeepSeek-V3.2 / GLM-5.2)

Add an FP4 (E2M1) KV-cache path to the SM100 sparse-MLA **decode** kernel so the
per-step KV read is halved on already-FP8 deployments. This is the decode-bandwidth
analog of the FP8-Q win that already shipped on Blackwell.

> **File this UNDER the existing FP4-KV tracking issues — do NOT open a competing
> standalone issue.** This is the **NVIDIA SM100 / CUDA sparse-MLA decode slice** of the
> broader FP4-KV-cache effort tracked in **#21601** ("[Feature] Add FP4 KV Cache Design
> and support SM120 GPUs") and **#26571** ("[WIP] FP4 KV Cache Support"). It is
> **distinct from** #21889 (AMD TileLang FP4-KV for NSA) and #25555 (MXFP4
> blockfp4-hadamard-quant). Coordinate with the #21601/#26571 owners before writing
> code; align the cache layout and quant recipe with whatever those issues settle on.

---

## TL;DR for the agent

You are a peer engineer landing fresh on SGLang. You own this end to end: prove
decode is actually KV-bandwidth-bound (so FP4 KV would help), confirm what the SM100
sparse-MLA decode kernel can consume, implement the FP4 KV path, and verify **both** a
TPOT win **and** that accuracy holds — under the #21601/#26571 umbrella, not as a fork
of it.

**One-line opportunity.** DSA decode reads the sparse-MLA KV per step; the cache is
already FP8 (`kv_cache_dtype=fp8_e4m3`) and the attention is already FP8-Q
(`fmhaSm100…QkvE4m3`). Decode is **KV-bandwidth-bound**, so dropping the KV cache FP8 →
FP4 (E2M1) roughly halves the dominant read and should improve TPOT at long context.

**Strong-prior hypothesis (must confirm on hardware).** On GLM-5.2-NVFP4 80k decode,
the per-step cost is dominated by the KV read of the sparse-MLA kernel (already-FP8),
not by Q-side or compute; halving KV bytes with FP4 buys a real TPOT reduction that
scales with context length. FP4-Q bought nothing at decode (s_q=1); **FP4-KV is the
lever that does**, because the KV read is the bottleneck.

**What we do NOT trust / do NOT anchor on:**
- **Do NOT assume FP4 KV accuracy is free.** Unlike prefill, decode errors **compound
  autoregressively** — a KV cache that quantizes acceptably for a one-shot prefill can
  drift generations over a long decode. Treat accuracy as a **hard gate**, measured on
  a real long-output workload, not a smoke test.
- **Do NOT reuse the FP4 *indexer* cache path for the main KV.** The indexer K cache
  (`dsv4/fp4_indexer.py`, MXFP4/E2M1+UE8M0) is a *different, smaller* buffer used only
  to compute top-k; the main sparse-MLA KV is a separate, larger cache with its own
  kernel ABI. Don't conflate them (that's M10's territory).
- **Do NOT trust line numbers verbatim** — grep by symbol; the repo moves fast.
- **Do NOT pick the FP4 recipe unilaterally.** #21601 is the design issue; match its
  element/scale format (NVFP4 vs MXFP4, block size) rather than inventing one.

**The job in one breath:** confirm decode is KV-bandwidth-bound on FP8 KV (settling
experiment) → determine whether the SM100 sparse-MLA decode kernel can consume FP4 KV
or needs a dequant path → implement FP4 store + in-kernel dequant behind a flag →
verify TPOT gain **and** long-output accuracy → PR under #21601/#26571.

**Out of scope:** prefill KV (this is decode-only); the AMD TileLang path (#21889); the
MXFP4-hadamard indexer-style quant (#25555); the FP4 *indexer* K cache (that's M10);
touching the FP8-Q attention (already shipped).

---

## Background you need (no prior context assumed)

### DSA decode and where the bytes go
DeepSeek-V3.2 / GLM-5.x use DSA (DeepSeek Sparse Attention). At decode the indexer
selects `dsa_index_topk` KV positions per query and the sparse-MLA kernel attends over
them. The whole decode loop replays inside a **full CUDA graph** (per profiling, every
decode kernel's CPU op is `cudaGraphLaunch`), so host-launch latency is already gone —
the remaining cost is pure kernel time, and for attention that is dominated by the
**KV read** (`topk × d_qk` bytes/query). The KV is already stored FP8, and Q/K/V are
already FP8 in the kernel (`fmhaSm100…QkvE4m3`), so the one remaining bandwidth lever
is the KV element width: FP8 → FP4 halves it.

### The SM100 sparse-MLA decode dispatch (what you'll touch)
`python/sglang/srt/layers/attention/dsa_backend.py`:
- `_DSA_IMPL_T` (≈:318) enumerates the five DSA impls: `flashmla_sparse`, `flashmla_kv`,
  `fa3`, `tilelang`, `trtllm`. Decode impl is `self.dsa_decode_impl` from
  `server_args.dsa_decode_backend` (≈:372); the default decode path is **`flashmla_kv`**
  (`use_flashmla_kv` gate ≈:711, and the decode branches ≈:943 / :1074 / :1116 /
  :1173).
- FP8 KV is gated on `model_runner.kv_cache_dtype == fp8_dtype` (≈:416, `self.kv_cache_dtype`
  ≈:434; `float8_e4m3fn` KV path ≈:852). This is the exact seam an FP4 mode extends.
- `self.dsa_index_topk` (≈:352) is the per-query selected count threaded through every
  decode metadata builder.
- There is a benchmarked auto-selector `flashmla_auto` (≈:383) — note it, but this task
  targets the KV element width, not impl selection.

The KV store side lives in the DSA KV pool (`mem_cache/memory_pool.py`, the
`set_mla_kv_buffer` / `quantize_k_cache_separate` family already do the BF16→FP8 store).
An FP4 path adds a quant-on-store + dequant-on-read variant here and in the decode
kernel.

### The FP4-KV tracking landscape (READ THESE FIRST — dedup)
- **#21601** "[Feature] Add FP4 KV Cache Design and support SM120 GPUs" (open, updated
  2026-07-06) — the design/tracking umbrella. Adopt its format decision.
- **#26571** "[WIP] FP4 KV Cache Support" (open) — the in-flight support work.
- **#21889** "[AMD] Enable FP4 (E2M1) KV cache quantization for NSA with TileLang
  backend" (open) — the ROCm/TileLang slice; shows the token-pool capacity win
  (BF16 2.15M → MXFP4 5.2M tokens on that path). **Different backend/arch** — this task
  is the NVIDIA SM100 sparse-MLA decode kernel.
- **#25555** "[MXFP4 KV Cache] MXFP4 KV Cache: support blockfp4-hadamard-quant" (open) —
  a specific quant recipe; not the decode-kernel integration.

Your job is the **SM100 sparse-MLA decode kernel + KV-pool** slice; everything else
above either lives on a different backend or is a design/recipe issue you consume.

---

## Plan (do these in order)

### Step 0 — Environment & access
- Repo: `git@github.com:vincentzed/sglang.git` (or upstream). Branch off latest `main`.
- Hardware: a Blackwell node (B200/B300, sm100/sm103) — the sparse-MLA FP8-Q decode
  kernel is SM100-specific; there is no CPU path.
- Models: `nvidia/GLM-5.2-NVFP4` (primary) and `deepseek-ai/DeepSeek-V3.2-Exp`. If a
  checkpoint gates/401s, STOP and report.
- Read #21601 + #26571 in full and note the chosen FP4 format before writing anything.
- **Deliverable:** env note (node, GPU count, both checkpoints resolved) + a one-paragraph
  summary of the #21601 format decision you will match.

### Step 1 — Settling experiment: is decode actually KV-bandwidth-bound? (do this first)
Before any kernel work, prove FP4-KV would help. Capture a decode trace of GLM-5.2-NVFP4
at long context (input≈80k, output≥1024, TP4) and confirm the sparse-MLA decode kernel's
time tracks KV bytes read (`topk × d_qk × bytes`), not compute or Q-side. A quick proxy:
sweep `dsa_index_topk` (or context length) and show decode kernel time scales with the
KV read volume. Also scope the accuracy gate here: pick the long-output eval (AIME with
large `max_tokens`, plus GSM8K) that will decide the hard gate in Step 4.
- **Deliverable:** a statement — "decode sparse-MLA is KV-bandwidth-bound on FP8 KV
  (evidence: kernel time vs KV-bytes)" or "it is NOT (redirect: <what dominates>)" — and
  the chosen accuracy eval + threshold. If not bandwidth-bound, STOP and report; M6 dies.

### Step 2 — Kernel support check
Determine whether the SM100 sparse-MLA decode kernel (`flashmla_kv` / trtllm-gen path)
can already consume an FP4 KV cache, or whether it needs a new dequant-in-kernel path.
Grep the kernel entry (`flash_mla_with_kvcache` / `trtllm_batch_decode_with_kv_cache_mla`,
`is_fp8_kvcache`) and the KV pool store. Report exactly what changes: a KV-pool FP4
store, a decode-kernel FP4 dequant, and the metadata/layout (block size + scale) matching
#21601.
- **Deliverable:** a fix-shape verdict — "kernel accepts FP4 KV directly" vs "needs a new
  dequant path in <file:kernel>" — with the exact call sites, and the layout that matches
  #21601's format.

### Step 3 — Implement (constraints, not a patch)
- FP4 quant on the KV **store** side in the DSA KV pool, dequant-in-kernel on the
  **read** side of the SM100 sparse-MLA decode kernel. Behind an opt-in flag (mirror
  how `kv_cache_dtype=fp8_e4m3` is threaded), default off, gated to SM100 + the validated
  decode shapes.
- **Match #21601's element/scale format exactly** (block size, NVFP4-vs-MXFP4 scale) —
  do not invent a recipe.
- **Invariants:** FP8 (and BF16) KV paths byte-identical when the flag is off; prefill
  path untouched; the FP4 *indexer* cache (M10) untouched; the FP8-Q attention untouched.
- **Deliverable:** the diff (KV-pool FP4 store + decode-kernel dequant + flag plumbing),
  compiling and importing clean, flag-gated.

### Step 4 — Verify: TPOT win AND accuracy (accuracy is a hard gate)
- **Perf:** decode TPOT improvement at long context (`bench_one_batch_server`, decode-shaped
  workload) FP4-KV vs FP8-KV, on GLM-5.2-NVFP4 and V3.2. Report the delta and confirm it
  grows with context (the bandwidth signature).
- **Accuracy (HARD GATE):** long-output eval — AIME (large `max_tokens`) + GSM8K — must
  stay within the Step-1 threshold. Because decode errors compound, run the *long* output,
  not a 200-token smoke.
- **Deliverable:** before/after TPOT table + the accuracy table; the change ships only if
  accuracy clears the gate.

### Step 5 — Hand back (under the tracking umbrella)
Open a PR that references **#21601 / #26571** and states it is the SM100 sparse-MLA
decode slice. Body: one-line what+why, the KV-bandwidth evidence, the TPOT delta, the
long-output accuracy table, and the flag. Ping the #21601 owners so it lands as part of
the coordinated FP4-KV effort, not a parallel path.
- **Deliverable:** the PR URL, linked to #21601/#26571.

---

## Definition of done
- Step-1 evidence that decode is KV-bandwidth-bound on FP8 KV (else M6 is closed with a
  redirect).
- FP4 KV decode path lands behind an opt-in flag, SM100-gated, FP8/BF16 paths
  byte-identical when off.
- **Accuracy hard gate:** long-output GSM8K + AIME within threshold (decode-compounding
  measured, not smoke-tested).
- Measurable decode TPOT improvement at long context that scales with context length.
- PR filed under #21601/#26571 (not standalone); format matches the tracking-issue decision.
- No change to prefill KV, the FP4 indexer cache, the FP8-Q attention, or the AMD path.

## Deliverables
- Diff: DSA KV-pool FP4 store + SM100 sparse-MLA decode dequant + flag plumbing.
- KV-bandwidth-bound evidence (Step 1) + kernel-support verdict (Step 2).
- TPOT before/after + long-output accuracy tables.
- PR linked to #21601/#26571.

## Constraints / notes
- Scratch under repo cwd (`scratch/`), never `/tmp`. Run `pre-commit run`.
- **Coordinate with #21601/#26571 owners before coding** — adopt their FP4 format and
  cache layout; this issue is a slice, not a competitor.
- Decode runs inside a full CUDA graph — any FP4 metadata must be graph-capture-safe
  (static buffers, `.copy_()` on replay), like the existing FP8 path.

---

## Testing commands

> M6 targets **DeepSeek-V3.2 / GLM-5.2 (DSA)** SM100 decode. DeepSeek-V4 uses a separate
> arch/pool — the V4 commands are **regression-safety** only.

### Decode TPOT — FP4-KV vs FP8-KV (GLM-5.2-NVFP4, then V3.2)
```bash
# FP8 KV baseline:
python3 -m sglang.launch_server --model-path nvidia/GLM-5.2-NVFP4 --trust-remote-code \
  --tp 4 --kv-cache-dtype fp8_e4m3 --port 30000
python3 -m sglang.bench_one_batch_server --model-path nvidia/GLM-5.2-NVFP4 \
  --trust-remote-code --tp 4 --batch-size 1 --input-len 80000 --output-len 1024

# FP4 KV (this change, flag on) — same workload; compare median TPOT, expect it to
# widen with input-len (the bandwidth signature).
```

### Accuracy hard gate — long output (decode-compounding)
```bash
# GSM8K full 1319, 20-shot:
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# AIME with large max_tokens (the long-decode stress that exposes FP4-KV drift):
#   16 repeats, max_tokens 64000, temperature 1.0, top_p 0.95
# PASS = FP4-KV within threshold of FP8-KV on BOTH.
```

### Regression-safety — DeepSeek-V4 (SEPARATE arch/pool; prove untouched)
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tp 4 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001
# PASS = V4 unchanged vs main (V4 must not be affected).
```

---

## References
- Tracking (file under these): #21601, #26571. Adjacent/distinct: #21889 (AMD TileLang
  NSA FP4-KV), #25555 (MXFP4 blockfp4-hadamard-quant).
- Code: `python/sglang/srt/layers/attention/dsa_backend.py` (`_DSA_IMPL_T` ≈:318,
  `dsa_decode_impl` ≈:372, `flashmla_kv` decode ≈:711/:943/:1074/:1116, FP8-KV gate
  ≈:416/:434/:852, `dsa_index_topk` ≈:352); DSA KV pool `mem_cache/memory_pool.py`
  (`set_mla_kv_buffer` / `quantize_k_cache_separate`).
- Context: FP8-Q sparse-MLA already shipped on SM100 (`fmhaSm100…QkvE4m3`); the FP4
  *indexer* cache (`dsv4/fp4_indexer.py`, MXFP4) is M10, a different buffer.
