# Task: Re-enable the DSA lightning-indexer Q/K kernel fusion for the NeoX / DeepSeek-V3.2 family

Add a NeoX (non-interleaved) RoPE branch to the indexer's interleaved-only fused
CUDA kernels so DeepSeek-V3.2 (and any `is_neox_style=True` DSA model) can run the
fast fused indexer path that GLM-5.2 already uses. Today V3.2 is forced onto the
slow split path because the fused kernels apply the wrong rotation.

---

## TL;DR for the agent

You are a peer engineer landing fresh on SGLang. You own this problem end to end:
reproduce the accuracy regression, prove the root cause on the running system,
implement the kernel fix, and verify parity.

**One-line problem.** The DSA "lightning indexer" (DeepSeek Sparse Attention, used
by DeepSeek-V3.2-Exp and GLM-5.x) has a fused Q/K kernel path that is **~10%
faster single-stream / ~6% faster at BS=128 decode on B300** (PR #27705). It is
**disabled for `is_neox_style=True` models** (DeepSeek-V3.2) because the fused
kernels hardcode **interleaved / GPT-J RoPE** and take no `is_neox` flag, so on a
NeoX model they rotate the wrong dimension pairs → corrupt the indexer scores →
wrong top-k → a ~2.4-point GSM8K drop. GLM-5.2 (non-NeoX) runs the fused path fine.

**Strong-prior hypothesis (already evidenced — you must still confirm on hardware).**
The entire regression is the RoPE-convention mismatch in the indexer's own CUDA
kernels. The fix is to add a NeoX rotation branch to those kernels and plumb
`is_neox_style` through, then drop the `and not is_neox_style` clause on the fusion
gate.

**What we do NOT trust / do NOT anchor on:**
- **Do NOT anchor on "the Hadamard drop causes the accuracy gap."** PR #27705 drops
  the per-head Hadamard rotation for V3.2. That is a *real but tolerated*
  second-order effect (higher FP8 quant error), **not** the blocker: it is
  symmetric+orthonormal so `q·k` is preserved in exact arithmetic, and it hits
  **both** families — yet GLM-5.2 runs Hadamard-free fusion and *passes*. If
  Hadamard were the cause, GLM-5.2 would regress too. Leave Hadamard dropped; the Q
  kernel already exposes a `kHadamard=true` template arg if a later FP8 sweep ever
  needs it. Prove the RoPE convention is the cause, not the quant.
- **Do NOT trust the PR-body line numbers verbatim** — this repo moves fast; the
  cited lines are circa the investigation HEAD. Grep by symbol name to the exact
  spot before editing.

**The job in one breath:** reproduce the V3.2 fusion-on regression → confirm the
kernel rotation is interleaved-only while the model is NeoX (state it with a
kernel-source diff) → add the NeoX rotation branch + plumb the flag → verify
fusion-on now matches fusion-off on V3.2 GSM8K and that GLM-5.2 is unchanged.

**Out of scope:** do NOT refactor the non-fused split path, do NOT touch the
DeepSeek-V4 native indexer (separate arch, separate kernels), do NOT change the
Hadamard behavior, do NOT change index-share / `skip_topk` logic.

---

## Background you need (no prior context assumed)

### What DSA / the lightning indexer is
DeepSeek-V3.2 and GLM-5.x use **DSA (DeepSeek Sparse Attention)**, a.k.a. NSA in the
SGLang tree. A small per-layer **indexer** projects Q and K, applies RoPE (and,
historically, a per-head Hadamard rotation), FP8-quantizes them, and computes
`fp8_mqa_logits` to select the **top-k KV positions** each query attends to. Sparse
MLA attention then runs over the selected positions. The indexer lives at:
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` — the `Indexer` class.

### The fusion and why it's gated off for V3.2
PR **#27705** fused the indexer Q and K paths into single CUDA kernels (rope + optional
Hadamard + FP8 quant + head-gate on Q; norm + rope + FP8 quant + index-cache store on
K, so K overlaps Q on an alt stream). It landed `+10%` single-stream / `+6%` BS128
decode on B300. It then triggered a bug cascade (memory blowup #29576, stale RoPE cache
#29613, CUDA-graph stream explosion #30025) and an **accuracy regression** on
DeepSeek-V3.2:
- #30088 measured GSM8K **0.955 fusion-OFF vs 0.931 fusion-ON** (full 1319-q 20-shot,
  8×H200, tp8, DP-attn, 5 runs each, `/flush_cache` between) — a consistent ~2.4-pt gap.
- #30018 turned fusion off by default; #30088 kept it off; **#30111** found the true
  cause = NeoX-RoPE incompatibility and re-enabled fusion **globally except when
  `is_neox_style=True`**, restoring V3.2 to 0.955.

The current gate (grep for `use_dsa_indexer_fusion` in `dsa_indexer.py`, `Indexer.__init__`,
circa line 366):
```python
self.use_dsa_indexer_fusion = (
    not envs.SGLANG_DISABLE_DSA_INDEXER_FUSION.get()
    and not is_neox_style          # <-- this clause parks V3.2 on the slow path
)
```
`SGLANG_DISABLE_DSA_INDEXER_FUSION` default is `EnvBool(False)` in
`python/sglang/srt/environ.py` (grep it). So: **fusion is on by default for
non-NeoX (GLM-5.2), off for NeoX (DeepSeek-V3.2).**

### The smoking gun — the kernels are interleaved-only
The indexer has its **own dedicated fused kernels** (distinct from the generic
qk-norm-rope kernel) that hardcode interleaved/GPT-J rotation and take **no `is_neox`
parameter**:

- **K kernel** — `python/sglang/jit_kernel/csrc/deepseek_v32/indexer_k.cuh`
  - `load_rope_first_cos_sin` pairs `pair0 = lane_id*2`, `pair1 = pair0+1` (circa lines
    40-41) → **adjacent** dims.
  - The rotate (circa lines 124-137) treats `(data[0],data[1])`, `(data[2],data[3])`
    as (real,imag) complex pairs — **interleaved**. NeoX instead pairs `(i, i+rope_dim/2)`.
  - `FusedKIndexerNormRopeParams` (circa lines 51-62) has **no** `is_neox` field.
- **Q kernel** — `python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`
  - `fused_q_indexer_rope_hadamard_quant`; same interleaved rotate (circa lines 52-59,
    512-521); template params `<..., kRopeFirst, kHadamard>` (circa line 451) — **no**
    `is_neox`; the freq-cache comment (circa line 441) says "interleaved
    [cos0, sin0, ...]". (Note: this Q kernel lives under the `deepseek_v4/` directory
    but is the shared indexer Q kernel used by V3.2 — do not confuse it with the V4
    native indexer.)
- **Reference kernel that DOES handle NeoX** —
  `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` takes an `is_neox`
  arg (circa lines 257, 300; comment "1 = NeoX style"). **Use this as the template**
  for the NeoX branch.

### Why the non-fused path is correct for V3.2
The split path applies RoPE via `self.rotary_emb(positions, ...)` (`dsa_indexer.py`,
`_get_k_bf16`, circa line 628), and `RotaryEmbedding` honors `is_neox_style`. So the
slow path rotates correctly for NeoX; only the fused kernels are wrong. That is why
#30111's `is_neox` skip fully restores accuracy — empirical proof the kernel rotation
is the *entire* regression.

### The wrapper plumbing you'll thread the flag through
- Python wrappers: grep `fused_q_indexer_rope_first_quant`, `fused_k_indexer_norm_rope`,
  `fused_k_indexer_norm_rope_store`, `can_use_dsa_fused_store` (Python side under
  `python/sglang/jit_kernel/` + call sites in `dsa_indexer.py`
  `_fused_q_prepare_and_store` circa line 682 and `_fused_k_prepare_and_store` circa
  line 634).
- `is_neox_style` is already a constructor arg of `Indexer` (grep `is_neox_style` in
  `dsa_indexer.py.__init__`, circa line 353) — it just isn't passed to the kernels.

---

## Plan (do these in order)

### Step 0 — Environment & access
- Repo: this fork, `git@github.com:vincentzed/sglang.git` (or the upstream
  `https://github.com/sgl-project/sglang.git`). Branch off latest `main`.
- Hardware: a Blackwell node (B200/B300) or H200, ≥8 GPUs for the DP-attn repro. The
  indexer fused kernels are CUDA JIT (`python/sglang/jit_kernel`), so you need a real
  GPU + nvcc; there is no CPU path.
- Models (HF): `deepseek-ai/DeepSeek-V3.2-Exp` (the **target**, NeoX), `zai-org/GLM-5.2`
  or `nvidia/GLM-5.2-NVFP4` (the **must-not-regress** control, non-NeoX). If a
  checkpoint download 401s/gates, STOP and report — do not substitute a different arch.
- Build: `pip install -e "python[all]"` per the repo dev docs; confirm the JIT indexer
  kernels compile (`python -c "import sglang.jit_kernel"` and launch once).
- **Deliverable:** a one-paragraph environment note — node, GPU count, both checkpoints
  resolved, JIT build green.

### Step 1 — Reproduce the regression (the settling experiment; do this first)
The decisive, cheap A/B is the `SGLANG_DISABLE_DSA_INDEXER_FUSION` toggle on V3.2.
Launch DeepSeek-V3.2-Exp twice and run full GSM8K each time, `/flush_cache` between:
```bash
# fusion OFF (correct baseline)
SGLANG_DISABLE_DSA_INDEXER_FUSION=1 python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp --trust-remote-code \
  --tp 8 --enable-dp-attention --port 30000 &
# ... wait for ready, then:
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# restart server with SGLANG_DISABLE_DSA_INDEXER_FUSION=0 (fusion ON) and repeat.
```
Also run the registered index-cache test that #30088 used as the CI gate:
```bash
python3 test/registered/8-gpu-models/test_deepseek_v32_indexcache.py
# class TestDeepseekV32IndexTopkPattern; threshold 0.935
```
Expect: fusion-OFF ≈ 0.955, fusion-ON ≈ 0.931 (straddling the 0.93 gate) — the
regression #30111 dodged by forcing the NeoX skip. To *see* fusion-ON on V3.2 you must
temporarily remove the `and not is_neox_style` clause (Step 3 does this for real).
- **Deliverable:** the two GSM8K numbers on V3.2 (fusion off vs on) reproducing the
  ~2.4-pt gap, plus the index-cache test result.

### Step 2 — Confirm the mechanism is the RoPE convention, not the quant
Prove the kernels rotate interleaved while the model is NeoX, and that this — not the
Hadamard drop — corrupts the top-k.
- Show the interleaved pairing in `indexer_k.cuh` (`load_rope_first_cos_sin`,
  `pair0/pair1`) and `main_norm_rope.cuh` (the rotate) and that neither takes `is_neox`.
- Confirm `Indexer.is_neox_style` is `True` for DeepSeek-V3.2 (log it at construction)
  and `False`/other for GLM-5.2.
- Settling check: instrument the indexer to dump the per-layer top-k index set for a
  fixed prompt under (a) fusion-OFF and (b) fusion-ON on V3.2. If the top-k **sets
  diverge**, the rotation is corrupting selection (root cause). Then dump the same on
  GLM-5.2 — top-k should be stable there (exonerates the Hadamard drop, which is common
  to both). Guiding question: does the divergence appear on the RoPE half of the head
  only, consistent with a wrong dim-pairing?
- **Deliverable:** a fix-layer verdict — "root cause = interleaved RoPE in the indexer
  fused kernels vs a NeoX model" — backed by (i) the kernel source showing interleaved
  pairing + no `is_neox`, and (ii) the top-k divergence on V3.2 that is *absent* on
  GLM-5.2. Explicitly rule out the Hadamard-quant hypothesis with the GLM-5.2 evidence.

### Step 3 — Implement the NeoX branch (constraints, not a patch)
1. Add a NeoX rotation path to **both** indexer fused kernels
   (`indexer_k.cuh` `fused_k_indexer_norm_rope[_store]`, and `main_norm_rope.cuh`
   `fused_q_indexer_rope_hadamard_quant`): when NeoX, pair dim `i` with `i + kRopeDim/2`
   instead of `(2i, 2i+1)`, and **load cos/sin in the NeoX (non-interleaved) layout the
   model's cache uses**. Mirror the existing `is_neox` handling in
   `fused_qknorm_rope.cuh` (the reference kernel) — do not invent a new convention.
2. Plumb the flag: add an `is_neox` param to the kernel params structs + launch
   wrappers, thread `self.rotary_emb.is_neox_style` (or `self.is_neox_style`) through
   `_fused_q_prepare_and_store` / `_fused_k_prepare_and_store` into the launch. Keep it
   a compile-time template switch if that matches the existing kernel style (both are
   templated), to avoid a runtime branch in the hot loop.
3. Relax the gate: drop `and not is_neox_style` from `use_dsa_indexer_fusion`
   (`dsa_indexer.py` circa line 366) so NeoX models take the fused path. Keep
   `SGLANG_DISABLE_DSA_INDEXER_FUSION` as the global kill-switch.
- **Invariants to preserve:** GLM-5.2's (non-NeoX / interleaved) path must be
  byte-identical — the NeoX branch must be additive, selected only when
  `is_neox_style`. Do NOT change the Hadamard behavior (`kHadamard` stays as-is). Do
  NOT touch the DeepSeek-V4 native indexer. The **cos_sin / freqs_cis cache layout
  differs** between interleaved and NeoX — the freq-load must match the model's
  convention exactly, or you silently reintroduce the same score corruption. This is
  the single highest-risk line.
- **Deliverable:** the kernel diff (both files) + the wrapper/gate plumbing, compiling
  clean.

### Step 4 — Verify parity + no collateral
- **V3.2 (target):** fusion-ON GSM8K must now **match fusion-OFF** (≈0.955, and clear
  the 0.935 index-cache gate). Run the same 5×/`flush_cache` protocol as Step 1.
- **GLM-5.2 (control):** GSM8K unchanged vs `main` (byte-identical top-k on a fixed
  prompt is the strong check).
- **Correctness unit:** add a fused-vs-eager top-k checksum test for a NeoX indexer (a
  small offline case that asserts the fused kernel's selected indices equal the
  `rotary_emb`-based reference under `is_neox_style=True`). This is the regression
  guard that would have caught #27705.
- **Perf:** confirm the fused path is actually faster on V3.2 (mirror #27705's numbers)
  with `bench_one_batch_server` (BS=1 single-stream and BS=128 decode). See testing
  section.
- **Deliverable:** before/after GSM8K table for V3.2 and GLM-5.2, the new checksum test
  passing, and a perf delta on V3.2.

### Step 5 — Hand back
Open a PR to upstream `sgl-project/sglang` titled roughly *"[DSA] Add NeoX RoPE branch
to indexer fused kernels; re-enable fusion for DeepSeek-V3.2"*. Body: one-line what+why,
the root-cause statement with the top-k-divergence evidence, the before/after GSM8K
table, the new test, and the perf delta. Reference #27705 / #30088 / #30111.
- **Deliverable:** the PR URL.

---

## Definition of done
- Root cause stated in 1-2 sentences, backed by Step 2's kernel-source diff **and** the
  V3.2-vs-GLM-5.2 top-k divergence evidence (Hadamard hypothesis explicitly ruled out).
- With the NeoX branch, **DeepSeek-V3.2 fusion-ON GSM8K matches fusion-OFF** (≈0.955,
  ≥0.935 gate) across 5 runs — the ~2.4-pt gap is closed, not merely masked.
- **GLM-5.2 unchanged** (ideally byte-identical top-k on a fixed prompt).
- New fused-vs-eager NeoX top-k checksum test passes.
- Measurable V3.2 fused-path speedup (order of #27705's +10%/+6%).
- No change to the split path, the Hadamard behavior, index-share, or the V4 indexer.

## Deliverables
- Kernel diff: `indexer_k.cuh` + `main_norm_rope.cuh` NeoX branches.
- Wrapper/gate plumbing diff in `dsa_indexer.py` (+ Python kernel wrappers).
- New NeoX top-k checksum test.
- Root-cause writeup + before/after GSM8K + perf table.
- Upstream PR.

## Constraints / notes
- Keep any scratch scripts/logs under the repo cwd (a `scratch/` subdir), never `/tmp`.
- Do not skip pre-commit hooks; run `pre-commit run` before the PR.
- The fused kernels are JIT — a stale JIT cache can hide a rebuild; clear
  `~/.cache` JIT dirs if a kernel edit doesn't take effect.
- Toggle for A/B: `SGLANG_DISABLE_DSA_INDEXER_FUSION` (1 = force split path).

---

## Testing commands

> N4 targets the **DeepSeek-V3.2 / GLM-5.x (NSA/DSA)** arch. DeepSeek-V4 is a
> **separate** arch with its own native indexer and kernels — the V4 commands below
> are **regression-safety** only (prove V4 is untouched), not validation of N4.

### A/B accuracy — DeepSeek-V3.2 (the target)
```bash
# Baseline (fusion OFF = correct):
SGLANG_DISABLE_DSA_INDEXER_FUSION=1 python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp --trust-remote-code \
  --tp 8 --enable-dp-attention --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# curl -s localhost:30000/flush_cache  # between runs

# After the NeoX-branch fix (fusion ON, gate clause removed):
SGLANG_DISABLE_DSA_INDEXER_FUSION=0 python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp --trust-remote-code \
  --tp 8 --enable-dp-attention --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# PASS when fusion-ON >= fusion-OFF within noise (~0.955), not the pre-fix 0.931.
```
Registered CI gate (the exact test #30088 tracked):
```bash
python3 test/registered/8-gpu-models/test_deepseek_v32_indexcache.py
```
AIME (harder signal; the harness #29613 used, on the NVFP4 checkpoint):
```bash
python3 -m sglang.launch_server --model-path nvidia/GLM-5.2-NVFP4 --trust-remote-code \
  --tp 4 --port 30000
# then your AIME-2025 client: 16 repeats, max_tokens 64000, temperature 1.0, top_p 0.95
```

### Control — GLM-5.2 must NOT regress
```bash
python3 -m sglang.launch_server --model-path zai-org/GLM-5.2 --trust-remote-code \
  --tp 8 --port 30000
python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --num-shots 20 --port 30000
# Expect identical to main; ideally verify byte-identical per-layer top-k on a fixed prompt.
```

### Perf — DeepSeek-V3.2 fused vs split
```bash
# single-stream (BS=1) and decode (BS=128); compare DISABLE=1 vs DISABLE=0:
python3 -m sglang.bench_one_batch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp --trust-remote-code --tp 8 \
  --batch-size 1 --input-len 1 --output-len 512
python3 -m sglang.bench_one_batch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp --trust-remote-code --tp 8 \
  --batch-size 128 --input-len 1024 --output-len 512
```

### Regression-safety — DeepSeek-V4 (SEPARATE ARCH; prove untouched)
The Q kernel file lives under `csrc/deepseek_v4/`, so a smoke run on V4 guards against
an accidental shared-code break. V4 does NOT exercise N4's gate change.
```bash
# V4 Pro (1.6T, multi-node TP16 in the cookbook; single-node example shown for smoke):
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code --tp 8 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001

# V4 Flash (285B, TP4/TP8):
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tp 4 --port 30001
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --port 30001
# PASS = V4 GSM8K unchanged vs main (any drift means the shared csrc/deepseek_v4 edit leaked).
```
(Use the cookbook's exact per-platform flags for real V4 serving; the above is a smoke
gate. Adjust `--tp`, MoE backend, and mem flags to your node.)

---

## References
- PR #27705 — landed the indexer Q/K fusion (dropped Hadamard for V3.2; +10%/+6% B300).
- PR #29576 / #29613 / #30025 — fusion follow-up fixes (memory, cos_sin cache, streams).
- PR #30018 / #30088 — turned fusion off by default over the V3.2 GSM8K regression
  (0.955 off vs 0.931 on).
- PR #30111 — root-caused to NeoX RoPE; re-enabled fusion except `is_neox_style`
  (the clause this task removes once the kernels are NeoX-correct).
- Kernels: `python/sglang/jit_kernel/csrc/deepseek_v32/indexer_k.cuh`,
  `python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh`; NeoX reference
  `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`.
- Gate/plumbing: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
  (`Indexer.__init__`, `_fused_q_prepare_and_store`, `_fused_k_prepare_and_store`,
  `_get_k_bf16`); env `SGLANG_DISABLE_DSA_INDEXER_FUSION` in
  `python/sglang/srt/environ.py`.
