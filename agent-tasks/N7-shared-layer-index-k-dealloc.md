# Task: Stop allocating the per-layer index-K buffer for shared (skip_topk) DSA layers

Under DSA index-share, most layers reuse a producer layer's top-k and never run
their own indexer — yet the KV pool allocates a full per-layer index-K-with-scale
buffer for every layer, including these shared ones, where it is provably dead.
Reclaim that HBM (~9 GB/rank at the default pattern, ~11% of attention KV) for the
main KV pool.

---

## TL;DR for the agent

You are a peer engineer landing fresh on SGLang. You own this: confirm the buffer is
dead on shared layers on the running system, implement the safe reclamation, and prove
no correctness regression under PD-disagg + HiCache.

**One-line opportunity.** DSA (DeepSeek Sparse Attention; DeepSeek-V3.2 / GLM-5.x)
allocates a per-layer FP8 **index-K-with-scale** buffer (132 B/token/layer). With
**index-share** (`index_topk_freq > 1` / `index_topk_pattern`), ~48% of layers are
`skip_topk` "shared" layers that reuse a producer layer's top-k. Those layers **never
run their indexer**, so their index-K buffer is **never written and never read** — but
it is still allocated. Freeing it recycles HBM into more KV token slots.

**Strong-prior hypothesis (evidenced — confirm on hardware).** The shared-layer
index-K buffer is genuinely dead (write-nobody + read-nobody). The reclamation is real;
the question is *how* to do it safely, because the buffer is a dense per-layer list
that many subsystems index by contiguous `layer_id`.

**What we do NOT trust / do NOT anchor on:**
- **Do NOT assume shared layers can skip the *main* KV buffer.** They still run sparse
  MLA attention over **their own main latent KV** at the carried top-k *positions*. Only
  the small **indexer** K-with-scale buffer (used to *compute* top-k) is dead. If you
  free main KV you corrupt the model. Confirm the distinction before touching anything.
- **Do NOT do a full "skip alloc + layer→slot remap".** That is the V4-style
  `layer_mapping`/`compress_ratios` approach and it generated a PP/HiCache/disagg bug
  cascade (#25477, #29106, #27888, #28941). Prefer the **empty-shared-slot** shape
  (keep the list length = `layer_num`, make shared entries zero-size/aliased, skip them
  in transfer loops) so every `[layer_id]` access and `range(layer_num)` loop still
  resolves structurally.
- **Do NOT conflate the two "slots".** `get_num_indexer_layers`' "those still get a
  slot" comment refers to the tiny **top-k capturer host buffer** (observability), a
  *different* structure from the target `index_k_with_scale_buffer`.

**The job in one breath:** confirm write-nobody/read-nobody on shared layers with
file:line + a runtime check → implement the empty-shared-slot reclamation → prove HBM
freed (bigger `max_total_num_tokens`) with identical accuracy, including under
PD-disagg and HiCache.

**Out of scope:** do NOT change index-share/`skip_topk` selection logic, the indexer
kernels, or the DeepSeek-V4 pools (separate `DeepSeekV4TokenToKVPool`). Do NOT attempt
the layer-remap variant.

---

## Background you need (no prior context assumed)

### Index-share and skip_topk
DSA models select top-k KV per query via a per-layer indexer. **Index-share**
(`dsa_layer_skips_topk(config, layer_id)` in
`python/sglang/srt/configs/model_config.py`, driven by `index_topk_pattern` /
`index_topk_freq` / `index_skip_topk_offset`) marks some layers `skip_topk=True`
(set on the module in `python/sglang/srt/models/deepseek_v2.py`, circa lines 1666-1673).
A shared layer **reuses the previous layer's top-k indices** and carries no indexer
weights in the checkpoint. PR #29959 added the `should_run_indexer` gate so the indexer
only runs when `not skip_topk`.

### The target buffer and why it's dead on shared layers
The DSA KV pool allocates a per-layer index-K buffer:
- `DSATokenToKVPool.__init__` (grep in `python/sglang/srt/mem_cache/memory_pool.py`,
  circa line 3068): `self.index_k_with_scale_buffer = [torch.zeros(...) for _ in
  range(layer_num)]` — **one full buffer per layer, shared layers included.**
- Size: `cols = 64 * (128 + 4)` → **132 B/token/layer** (128 B FP8 index key + 4 B FP32
  scale). For comparison the **main MLA latent** is 576 B/token/layer
  (`kv_lora_rank 512 + qk_rope 64`, FP8) — so index-K is ~23% of the main KV.

Dead on shared layers because:
- **Written by nobody:** all index-K stores live inside the `Indexer` class
  (`_fused_k_prepare_and_store` circa line 654; `_store_index_k_cache` circa
  1530/1548/1572 in `dsa_indexer.py`), and every `self.indexer(...)` call is gated by
  `should_run_indexer` (`forward_mla.py` circa 377/403; `forward_mha.py` circa 172).
  Shared layers never enter the indexer → never store.
- **Read by nobody:** all index-K reads are inside `Indexer` methods reached only when
  it runs (`_get_topk_paged` ~806, `_get_topk_ragged` ~1042, `_get_topk_ragged_with_cp`
  ~1306/1367, `forward_indexer` ~1464). The sparse-MLA attention backend reads the
  **main** latent KV (`kv_buffer`) at the carried top-k positions, **not** index-K.

### Savings (why it's worth it)
- Shared fraction is large: the DeepSeek-V3.2 index-cache test pattern is **29/61 ≈ 48%
  shared**; `index_topk_freq=2` ≈ 49%; `freq=4` ≈ 74%.
- `48% × 23%` ≈ **11% of total attention-KV** freed (~17% at freq=4). On a ~2.4M-token
  TP4 GLM-5.2 pool: `132 B × 2.4M × ~29 shared layers ≈ 9.2 GB/rank`. The *ratio* is
  pool-size-independent, so it's robust.
- #29576 already fixed a *different* per-layer buffer (the RoPE buffer, per-layer ×
  context) down to 0.25 GB/GPU. N7 is a distinct buffer — do not confuse them.

### Where "one slot per layer" is assumed (the risk surface)
The buffer is a dense list addressed by contiguous `layer_id - start_layer` in many
consumers (all in `memory_pool.py` unless noted):
- `move_kv_cache` (~3103) — retract, lockstep over layers.
- `get_cpu_copy` / `load_cpu_copy` (~3193/3215) — offload loops over `range(layer_num)`.
- `get_kv_size_bytes` (~3240).
- **`get_state_buf_infos` (~3226-3236)** — emits per-layer `data_ptrs`/`data_lens`/
  `item_lens` for **PD-disagg state transfer** (NIXL / mooncake).
- **HiCache host pool** `python/sglang/srt/mem_cache/memory_pool_host.py` (~3296-3453) —
  mirrors the buffer per layer and transfers by `[layer_id]` (the biggest coupling).
- PP `start_layer` offsets everywhere the above run.

Disagg gotcha: NIXL's `_send_state_pages_flat` historically **hard-asserted equal
per-entry lengths** (#24888) — heterogeneous/zero `item_lens` per layer must be tolerated
(route through the generic per-buffer transfer, or explicitly skip shared layers).

---

## Plan (do these in order)

### Step 0 — Environment & access
- Repo: `git@github.com:vincentzed/sglang.git` (or upstream). Branch off latest `main`.
- Hardware: ≥4-GPU node (TP4) to see the memory delta; an 8-GPU node if you also test PD
  1P1D + HiCache. GPU required (DSA pools + kernels are CUDA).
- Models: `deepseek-ai/DeepSeek-V3.2-Exp` or `zai-org/GLM-5.2` (both DSA / index-share).
  Confirm the launch reports an index-share pattern (`index_topk_freq > 1` or a pattern
  with `S` entries). If a checkpoint gates/401s, STOP and report.
- **Deliverable:** environment note + confirmation the model is using index-share
  (log `skip_topk` count / the resolved pattern).

### Step 1 — Confirm the buffer is dead on shared layers (settling experiment)
Prove write-nobody + read-nobody empirically, not just by reading code:
- Instrument `DSATokenToKVPool` (or the store/read sites) to record, per `layer_id`,
  whether index-K was ever written / read during a real GSM8K run. Assert that every
  `skip_topk` layer has **zero writes and zero reads**.
- Cross-check against `should_run_indexer`: log the layers where the indexer actually
  runs; it must be exactly the non-`skip_topk` set.
- Guiding question: does any path (retract, offload, disagg send, HiCache mirror) *touch*
  a shared layer's index-K buffer even though attention/indexer don't? Those touches are
  what the empty-slot design must neutralize (Step 3), not correctness reads.
- **Deliverable:** a per-layer write/read table for one run proving shared layers are
  dead for indexer purposes, plus the list of *bookkeeping* touches (transfer/offload)
  that remain.

### Step 2 — Size the win and pick the shape
- Compute actual freed bytes for the target model: `132 B × max_total_num_tokens ×
  (#skip_topk layers)`. Report GB/rank and % of the attention-KV pool.
- Decide the implementation shape. **Prime choice: empty-shared-slot** (keep list length
  = `layer_num`; shared entries are zero-size or a shared 1-page dummy; skip them in
  offload/move/transfer loops). Reject the layer-remap variant in writing, citing the V4
  cascade (#25477/#29106/#27888/#28941), unless you find a blocking reason the empty-slot
  shape can't work.
- **Deliverable:** the freed-GB number + a one-paragraph shape verdict (empty-slot vs
  remap) with the rejected option's risk.

### Step 3 — Implement (constraints, not a patch)
- In `DSATokenToKVPool.__init__`, allocate a full buffer only for **non-`skip_topk`**
  layers; for shared layers store a **zero-size (or aliased 1-page) placeholder** so the
  list stays length `layer_num` and every `[layer_id]` access resolves.
- Gate the loops in `move_kv_cache`, `get_cpu_copy`/`load_cpu_copy`, `get_kv_size_bytes`,
  and `get_state_buf_infos` to **skip zero-size shared entries** (don't offload/transfer
  a dead buffer).
- HiCache host pool (`memory_pool_host.py`): mirror only real layers; the per-`layer_id`
  transfer must no-op on shared layers.
- Disagg: ensure `get_state_buf_infos` emitting zero/absent shared-layer entries is
  tolerated by the transport — align with the generic per-buffer transfer path (the
  fix-shape #24888 used for NIXL), or skip shared layers explicitly on both send + recv.
- **Invariants:** main KV pool unchanged; producer (non-shared) layers behave exactly as
  before; `layer_id` addressing stays valid everywhere; PP `start_layer` slicing still
  correct. Do NOT touch V4 pools or index-share selection.
- **Deliverable:** the diff (pool alloc + the four transfer/offload consumers + HiCache
  host pool + disagg buf-infos), compiling and importing clean.

### Step 4 — Verify: memory freed, accuracy identical, PD/HiCache safe
- **Memory:** at fixed `--mem-fraction-static`, `max_total_num_tokens` must rise by the
  Step-2 amount (server startup logs the pool size). Show before/after.
- **Accuracy:** GSM8K (full 1319, 20-shot) identical to `main` on both V3.2 and GLM-5.2 —
  index-share correctness must be untouched (this is the guard that shared-layer top-k
  reuse still works).
- **PD-disagg:** run a 1P1D disagg config (NIXL and, if available, mooncake) and confirm
  correct output — this is where a mis-handled shared-layer `item_lens` corrupts state.
- **HiCache:** run with `--enable-hierarchical-cache`; confirm a cache commit/load-back
  round-trip preserves accuracy (shared-layer mirror must no-op cleanly).
- **Deliverable:** before/after `max_total_num_tokens`, GSM8K parity table, and green
  PD-disagg + HiCache round-trips.

### Step 5 — Hand back
Open a PR to `sgl-project/sglang`: *"[DSA] Skip index-K allocation for shared
(skip_topk) index-share layers"*. Body: one-line what+why, the freed-GB number, the
per-layer dead-buffer evidence, the accuracy-parity + PD/HiCache results, and the shape
rationale (empty-slot vs remap). Reference #29959 (the gate), #24888 (disagg len shape).
- **Deliverable:** the PR URL.

---

## Definition of done
- Per-layer evidence that shared (`skip_topk`) layers' index-K is write-nobody +
  read-nobody (Step 1 table), with the main-KV-still-needed distinction explicit.
- `max_total_num_tokens` increases by ~`132 B × tokens × #shared_layers` at fixed
  mem-fraction (before/after logged).
- GSM8K **identical** to `main` on V3.2 and GLM-5.2 (index-share correctness intact).
- **PD-disagg (NIXL/mooncake) and HiCache round-trips pass** — no state corruption from
  zero-size shared-layer entries.
- Empty-shared-slot shape (not a layer remap); `layer_id` addressing valid everywhere;
  V4 pools and index-share logic untouched.

## Deliverables
- Diff: `memory_pool.py` (`DSATokenToKVPool` alloc + 4 transfer/offload consumers) +
  `memory_pool_host.py` (HiCache mirror) + disagg buf-infos handling.
- Per-layer dead-buffer evidence + freed-GB measurement.
- GSM8K parity table (V3.2 + GLM-5.2), PD-disagg + HiCache verification.
- Upstream PR.

## Constraints / notes
- Scratch under repo cwd (`scratch/`), never `/tmp`.
- Run `pre-commit run`; don't skip hooks.
- Coordinate with open PRs that touch the HiCache host pool: **#29637** (HiSparse
  prefetch via IndexShare) and **#28523** (IndexCache shared-layer IO overlap) both edit
  `memory_pool_host.py`; **#29340** (RL skip shared indexer *weights*) is orthogonal.
- The empty-slot placeholder must not break `torch.save`/offload serialization — verify
  the offload path handles zero-size tensors (or use a shared 1-page dummy).

---

## Testing commands

> N7 targets the **DeepSeek-V3.2 / GLM-5.x (DSA/NSA)** `DSATokenToKVPool`. DeepSeek-V4
> uses a **separate** `DeepSeekV4TokenToKVPool` and is unaffected — the V4 commands are
> **regression-safety** only.

### Memory delta + accuracy — GLM-5.2 / DeepSeek-V3.2
```bash
# Before (main) and after (this change), same flags — compare startup pool size:
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --mem-fraction-static 0.85 --port 30000
#   -> grep the startup log for "max_total_num_tokens" ; it must rise after the change.
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
#   -> must equal main (index-share correctness unchanged).

# DeepSeek-V3.2 target (also confirm the index-share pattern is active):
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --trust-remote-code --tp 8 --enable-dp-attention --mem-fraction-static 0.85 --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
```
Registered DSA index-cache gate (proves index-share selection still correct):
```bash
python3 test/registered/8-gpu-models/test_deepseek_v32_indexcache.py
```

### PD-disagg round-trip (the highest-risk path — state transfer)
```bash
# Launch a 1P1D disagg pair per the cookbook (NIXL backend), then run GSM8K through the
# router. The change is correct only if disagg output matches non-disagg.
# (Use the DeepSeek-V3.2 / GLM-5.2 disagg recipe from docs/cookbook; key point: the
#  index-K get_state_buf_infos must tolerate zero-size shared-layer entries.)
```

### HiCache round-trip
```bash
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 4 --enable-hierarchical-cache --mem-fraction-static 0.85 --port 30000
# Warm a prompt, evict, re-request (cache hit) -> output must be unchanged.
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --num-shots 20 --port 30000
```

### Regression-safety — DeepSeek-V4 (SEPARATE POOL; prove untouched)
```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code --tp 8 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001

python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tp 4 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001
# PASS = V4 pool size + GSM8K unchanged vs main (V4 must not be affected at all).
```

---

## References
- PR #29959 — `should_run_indexer` gate (indexer runs only on non-shared layers).
- PR #29576 — the *other* per-layer buffer fix (RoPE buffer; do not confuse).
- PR #24888 — NIXL disagg state-transfer len-shape handling (the `_send_state_pages_flat`
  equal-length assertion to respect).
- V4 remap-cascade precedent (why to AVOID the layer-remap shape): #25477, #29106,
  #27888, #28941.
- Open PRs touching the HiCache host pool to coordinate with: #29637, #28523; orthogonal:
  #29340.
- Code: `python/sglang/srt/mem_cache/memory_pool.py` (`DSATokenToKVPool`,
  `move_kv_cache`, `get_cpu_copy`/`load_cpu_copy`, `get_kv_size_bytes`,
  `get_state_buf_infos`), `python/sglang/srt/mem_cache/memory_pool_host.py`;
  `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`;
  `python/sglang/srt/configs/model_config.py` (`dsa_layer_skips_topk`).
