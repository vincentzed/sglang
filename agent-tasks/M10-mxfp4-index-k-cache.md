# Task: Port the DeepSeek-V4 MXFP4 index-K cache to the GLM-5.2 / DeepSeek-V3.2 (DSA) path

Bring DeepSeek-V4's opt-in MXFP4 indexer K-cache (E2M1 elements + UE8M0/block-32
scale, `#26209`) to the shared DSA (V3.2 / GLM-5.x) indexer, behind an opt-in flag.
The DSA index-K cache is FP8 today (**132 B/token/layer**); MXFP4 stores it in
**68 B/token/layer** — a **~48% cut** of the indexer KV — that stacks with N7
(dropping the shared-layer index-K buffers entirely). Reuse V4's kernels; write no
new ones.

---

## TL;DR for the agent

You are a peer engineer landing fresh on SGLang. You own this: wire the existing V4
MXFP4 indexer kernels into the shared DSA path behind a flag, prove the cache actually
shrinks, and prove accuracy stays within the V4-measured envelope.

**One-line opportunity.** The DSA "lightning indexer" (DeepSeek-V3.2 / GLM-5.x) stores
its per-layer index-K cache as FP8 (`128 B` fp8 key + `4 B` fp32 scale = 132 B/token/
layer). DeepSeek-V4 already ships an **opt-in FP4 indexer** (`--enable-deepseek-v4-fp4-indexer`,
#26209) that stores the same cache as **MXFP4** (E2M1 + UE8M0/block-32) = **68 B/token/
layer** — the size difference is literally coded as `head_dim_with_sf = 68 if
use_fp4_indexer else 132` (`dsv4/indexer.py:719`). The kernels exist; they are just not
wired for the shared V3.2/GLM `dsa` path.

**Strong-prior hypothesis (this is a *port*, not new kernel work).** The V4 quantize/
store kernels (`dsv4/fp4_indexer.py`) and the FP4 logits path
(`deep_gemm.fp8_fp4_paged_mqa_logits`) are complete and validated on V4. The task is
plumbing: add a flag on the DSA path, route the DSA indexer store/read/logits through
the FP4 variants, and size the DSA KV-pool index-K buffer at 68 B instead of 132 B.

**What we do NOT trust / OUT OF SCOPE:**
- **Do NOT attempt NVFP4 (E4M3-scale) for finer accuracy — it is kernel-blocked.** The
  DeepGEMM `fp4` MQA-logits kernel (`sm100_mqa_logits.cuh`, verified on the
  sgl-project/DeepGEMM **`dev`** branch) hardcodes E2M1 + `float_ue8m0_t`; its only
  format knob is a `bool is_fp4` selecting {MXFP4-FP4, FP8} — there is **no
  E4M3-scale/NVFP4 mode on any branch**. "Use NVFP4 for better scale granularity"
  would require writing a **net-new sparse MQA-logits kernel** and is explicitly out of
  scope. This task is MXFP4-only.
- **Do NOT default this on.** #26209 measured a real accuracy cost (below). It must be
  opt-in / default-off, exactly like the V4 flag.
- **Do NOT touch the V4 native indexer path** except to reuse its kernels — it shares
  `dsv4/` code, so verify V4 stays byte-identical (regression-safety).
- **Do NOT trust the file:line numbers verbatim** — grep by symbol; the repo moves fast.

**The job in one breath:** add a DSA FP4-indexer flag → route the DSA indexer
quantize/store, cache-read, and logits through V4's MXFP4 kernels → size the DSA index-K
pool buffer at 68 B → prove `max_total_num_tokens` rises ~48%-of-index-K at fixed
mem-fraction and GLM-5.2/V3.2 accuracy stays within the V4-measured ~1.5-pt GSM8K
envelope → confirm it stacks with N7 and doesn't regress V4.

---

## Background you need (no prior context assumed)

### The DSA indexer and its FP8 K cache
DeepSeek-V3.2 / GLM-5.x use **DSA** (DeepSeek Sparse Attention): a per-layer **indexer**
projects Q/K, applies RoPE (+ optional Hadamard), FP8-quantizes, and computes
`fp8_mqa_logits` to pick the top-k KV positions. The indexer's own K activations are
cached (separate from the main MLA latent KV) so decode/later-layer top-k can be
recomputed. That cache is:
- Stored by `fused_k_indexer_norm_rope_store(...)` →
  `pool.get_index_k_with_scale_buffer(layer_id=...)` (`dsa_indexer.py:652-654`); read at
  `:806` (`get_index_k_with_scale_buffer`) and `:1042` (`get_index_k_scale_buffer`).
- Allocated in **`DSATokenToKVPool`** (`memory_pool.py:3005`): a per-layer list
  `self.index_k_with_scale_buffer = [... for _ in range(layer_num)]` (`:3068`), dtype
  `torch.uint8` (`:3007`), indexed by `layer_id - self.start_layer` (`:3106`). Layout is
  **128 B fp8 key + 4 B fp32 scale = 132 B/token/layer**.

### V4's MXFP4 indexer (the thing to port)
PR **#26209** added an opt-in FP4 indexer for DeepSeek-V4 (`--enable-deepseek-v4-fp4-indexer`):
- **Quantize/store kernels** — `python/sglang/srt/layers/attention/dsv4/fp4_indexer.py`:
  `quantize_fp4_indexer_tensor` and `store_fp4_index_k_cache`. The quant is **MXFP4**:
  `_fp4_e2m1_code` (E2M1, max 6.0; `:27`) + `_ceil_ue8m0_exp` (UE8M0 power-of-2 scale;
  `:18`) with `GROUP_N=32, BLOCK_N=128` (4 scale groups of 32 per 128-dim token). Output
  = `64 B` packed E2M1 + `4 B` (4 packed UE8M0 exponents) = **68 B/token/layer**.
- **Logits path** — `dsv4/indexer.py:660` imports `deep_gemm.fp8_fp4_paged_mqa_logits`
  and dispatches to it when `use_fp4_indexer` (`:643`); the cache row width is
  `head_dim_with_sf = 68 if use_fp4_indexer else 132` (`:719`). The flag resolves from
  `get_global_server_args().enable_deepseek_v4_fp4_indexer` (`:864`).
- **Accuracy (measured on V4-Flash, #26209):** GSM8K **0.980 → 0.965** (~1.5-pt), GPQA
  pass@1 0.835 → 0.833 (flat), pass@16 0.944 → 0.934. Kernel 1.49–1.53×, E2E 1.06–1.08×.

### The MXFP4-only constraint (verified this session)
The DeepGEMM sparse MQA-logits kernel on the **`dev`** branch
(`deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`) uses
`make_instr_desc_block_scaled<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, ...>`
for the FP4 branch — E2M1 + UE8M0. The host wrapper's only format arg is `bool is_fp4`
({MXFP4, FP8}). No NVFP4 path exists. So the FP4 index cache **must** be MXFP4; finer
scaling is not available without a new kernel.

### Relationship to N7
N7 stops allocating the index-K buffer for `skip_topk` shared layers (~48% of layers).
M10 shrinks the *per-token* size of the buffers that remain (producer layers). They
**stack**: N7 removes ~48% of the buffers; M10 halves the survivors → combined index-K
footprint ≈ `52% × (68/132)` ≈ **27%** of today's.

---

## Plan (do these in order)

### Step 0 — Environment & access
- Repo: `git@github.com:vincentzed/sglang.git` (or upstream). Branch off latest `main`.
- Hardware: SM100 (B200/B300), ≥4 GPUs (TP4). CUDA + the `deep_gemm` wheel must be
  present (the FP4 logits path is DeepGEMM; `import deep_gemm; deep_gemm.fp8_fp4_paged_mqa_logits`
  must resolve). If the wheel is absent, STOP and report — FP4 index has no non-DeepGEMM path.
- Models: `zai-org/GLM-5.2` or `nvidia/GLM-5.2-NVFP4`, and `deepseek-ai/DeepSeek-V3.2-Exp`
  (both DSA). Keep `deepseek-ai/DeepSeek-V4-Flash` available for regression-safety.
- **Deliverable:** env note — node, GPU count, deep_gemm importable, all checkpoints resolved.

### Step 1 — Map the port surface
Enumerate exactly what the DSA path must route through the FP4 variants:
- The three DSA index-K **store/read** sites in `dsa_indexer.py` (`:652/654` store, `:806`
  read, `:1042` scale read, plus the `:1530/1548` cache-buffer accesses) — which of these
  the V4 path already handles vs which are DSA-only.
- The DSA **logits** call (the DSA analog of `dsv4/indexer.py:660`'s
  `fp8_fp4_paged_mqa_logits` dispatch) — confirm the shared DSA indexer can reach the
  same DeepGEMM entrypoint and what Q/K SF layout it expects.
- The **pool** buffer: `DSATokenToKVPool.index_k_with_scale_buffer` sizing
  (`memory_pool.py:3068`) — where the `132`-byte row width is set, to switch it to `68`
  under the flag.
- Guiding question: is the V4 FP4 indexer code in `dsv4/` reusable as-is from the `dsa`
  path, or is it entangled with V4-specific compressor/metadata (`dsv4/compressor.py:208`
  gates on `enable_deepseek_v4_fp4_indexer`)? Identify the minimal shared surface.
- **Deliverable:** a call-site map — for store, read, logits, and pool-alloc — of what
  exists on the DSA path vs what must be added, naming the V4 functions to reuse.

### Step 2 — Wire the flag + route through the MXFP4 kernels (constraints, not a patch)
1. Add a DSA-scoped opt-in flag (e.g. reuse/extend the indexer-fp4 server arg for the
   `dsa` backend, or add a sibling) — **default-off**.
2. Under the flag, route the DSA indexer's store/read through `store_fp4_index_k_cache` /
   the MXFP4 read, the logits through `fp8_fp4_paged_mqa_logits`, and size
   `DSATokenToKVPool.index_k_with_scale_buffer` rows at **68 B** (mirror `head_dim_with_sf`).
- **Invariants:** the FP8 path stays the byte-identical default when the flag is off; the
  **V4 native path is untouched** (shared `dsv4/` kernels only *reused*, not modified);
  **no new NVFP4/E4M3 kernel** — MXFP4 only; the `layer_id - start_layer` pool indexing and
  PP `start_layer` slicing stay valid at the new row width.
- **Deliverable:** the diff (flag + DSA store/read/logits routing + pool sizing), compiling
  and importing clean, FP4 path reachable on GLM-5.2.

### Step 3 — Accuracy gate + cache-reduction measurement (the make-or-break)
- **Cache reduction:** at fixed `--mem-fraction-static`, the startup `max_total_num_tokens`
  must rise by ~`(132-68) B × tokens × (#producer layers)`. Show before/after; confirm the
  index-K row width is 68 in the FP4 run.
- **Accuracy:** GSM8K (full 1319, 20-shot) and the `test_deepseek_v32_indexcache.py` gate on
  GLM-5.2 and V3.2, FP4-index vs FP8-index. **PASS = within the V4-measured envelope**
  (~1.5-pt GSM8K drop, pass@16 ≈ −1pt) — not a larger regression. AIME on `nvidia/GLM-5.2-NVFP4`
  for a harder signal.
- Guiding question: does the coarse UE8M0 scale hurt GLM-5.2/V3.2 selection *more* than V4
  (different head dims / index_topk)? If the drop materially exceeds V4's ~1.5-pt, report it
  — that changes the ship/no-ship call.
- **Deliverable:** before/after `max_total_num_tokens`, and a GSM8K (+ index-cache gate)
  table FP8-vs-FP4 for GLM-5.2 and V3.2 with the delta vs the V4 envelope.

### Step 4 — Confirm it stacks with N7 and doesn't regress V4
- With **both** N7 (shared-layer drop) and M10 (68-B producers) on, verify correctness and
  the combined pool-size gain (~27% of today's index-K footprint).
- **V4 regression-safety:** since `dsv4/fp4_indexer.py` is shared, run V4-Flash with and
  without `--enable-deepseek-v4-fp4-indexer` and confirm GSM8K + pool size are unchanged vs `main`.
- **Deliverable:** combined-with-N7 correctness + pool-size result, and a V4 no-regression check.

### Step 5 — Hand back
Open a PR: *"[DSA] Opt-in MXFP4 index-K cache for GLM-5.2 / DeepSeek-V3.2 (port of #26209)."*
Body: one-line what+why, the 68-vs-132 cache-size cut, the FP8-vs-FP4 accuracy table within
the V4 envelope, the N7-stacking result, and the explicit MXFP4-only / NVFP4-out-of-scope note.
- **Deliverable:** the PR URL.

---

## Definition of done
- FP4-index flag on the DSA path, **default-off**, routing store/read/logits through the
  existing V4 MXFP4 kernels (no new kernels; MXFP4 only).
- Index-K row width **68 B** under the flag; `max_total_num_tokens` rises accordingly at
  fixed mem-fraction (before/after logged).
- GLM-5.2 and V3.2 accuracy **within the V4-measured envelope** (~1.5-pt GSM8K; index-cache
  gate passes); reported, not assumed.
- Stacks with N7 (combined pool-size gain shown); **V4 native path byte-identical**.
- No NVFP4/E4M3 kernel introduced.

## Deliverables
- Diff: DSA flag + store/read/logits routing + `DSATokenToKVPool` 68-B sizing.
- Before/after `max_total_num_tokens`; FP8-vs-FP4 GSM8K/index-cache table (GLM-5.2 + V3.2).
- N7-stacking correctness/pool result; V4 no-regression check.
- Upstream PR.

## Constraints / notes
- Scratch under repo cwd (`scratch/`), never `/tmp`. Run `pre-commit run`; don't skip hooks.
- **MXFP4 only** — NVFP4 is kernel-blocked (DeepGEMM `sm100_mqa_logits.cuh` dev branch has
  no E4M3-scale mode); do not attempt it.
- **Default-off** — the ~1.5-pt GSM8K cost makes this opt-in, like the V4 flag.
- FP4 index has **no non-DeepGEMM fallback** — if the wheel is absent the flag must error
  clearly, not silently degrade.

---

## Testing commands

> M10 targets **GLM-5.2 / DeepSeek-V3.2 (dsa)**. V4 commands are **regression-safety**
> (shared `dsv4/` kernels must stay byte-identical), not validation of M10.

### Cache reduction + accuracy — GLM-5.2 / V3.2
```bash
# Baseline (FP8 index) vs FP4 index — compare startup max_total_num_tokens AND gsm8k:
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --mem-fraction-static 0.85 --port 30000
#   grep startup log: "max_total_num_tokens"
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000

python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --mem-fraction-static 0.85 <FP4-INDEX-FLAG> --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
#   PASS: max_total_num_tokens up ~(64B × tokens × producer-layers); gsm8k within ~1.5-pt of FP8.

# DeepSeek-V3.2 (also confirm index-share pattern active):
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --trust-remote-code --tp 8 --enable-dp-attention --mem-fraction-static 0.85 <FP4-INDEX-FLAG> --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
```
Registered DSA index-cache gate (selection correctness under FP4 index):
```bash
python3 test/registered/8-gpu-models/test_deepseek_v32_indexcache.py
```
AIME (harder signal, NVFP4 checkpoint): launch `nvidia/GLM-5.2-NVFP4` tp4 + your AIME-2025 client.

### Regression-safety — DeepSeek-V4 (shared dsv4/ kernels; prove untouched)
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tp 4 --enable-deepseek-v4-fp4-indexer --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001
#   PASS: V4-Flash gsm8k + pool size unchanged vs main, flag on and off.
```

---

## References
- PR **#26209** — DeepSeek-V4 FP4 indexer (source to port); docs #27035.
- Kernels: `python/sglang/srt/layers/attention/dsv4/fp4_indexer.py`
  (`quantize_fp4_indexer_tensor`, `store_fp4_index_k_cache`, `_fp4_e2m1_code`, `_ceil_ue8m0_exp`);
  `dsv4/indexer.py` (`use_fp4_indexer` :643, `fp8_fp4_paged_mqa_logits` :660,
  `head_dim_with_sf = 68/132` :719, flag :864).
- DSA path: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (store :652/654, reads
  :806/:1042); `DSATokenToKVPool` in `python/sglang/srt/mem_cache/memory_pool.py`
  (:3005, alloc :3068, `get_index_k_with_scale_buffer` :3106).
- MXFP4-only constraint: DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`
  (`dev` branch) — E2M1 + `float_ue8m0_t`, `bool is_fp4`, no NVFP4 mode.
- **[[N7-shared-layer-index-k-dealloc]]** — stacks with this; drop shared-layer index-K buffers.
