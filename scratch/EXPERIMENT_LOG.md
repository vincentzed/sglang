# ReplaySSM spec-verify — full experiment log

Everything tested in the investigation of the ReplaySSM GDN spec-verify long-output
regression. Reverse-chronological conclusions up top; full record below.

- **Base:** PR #28695 (`gdn_replayssm_spec_decode`, Yuan Luo) — GDN ReplaySSM ring spec-verify.
  The fp16 fix below also landed upstream as 28695 commit `8c1ddbdb9a` (identical to ours).
- **Model / setup:** `Qwen/Qwen3.6-35B-A3B` (GDN-hybrid, NEXTN MTP, topk=1), 8× B300, TP1 unless
  noted. Eval = AIME-2024 + AIME-2025 (60 problems), greedy (temp 0), max_tokens 32768,
  `--reasoning-parser qwen3`. Loop metric = max single-line repeat in (reasoning+content),
  lines>8 chars; loop≥50 = pathological, loop≥200 = severe/catastrophic.
- **Launch recipe (SM100):** `--linear-attn-decode-backend triton --attention-backend triton
  --page-size 1 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --enable-gdn-replayssm-spec --gdn-replayssm-spec-cache-len 16
  --mamba-radix-cache-strategy no_buffer --disable-overlap-schedule` (+ `--mamba-ssm-dtype float16`),
  env `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`.

## TL;DR — what won and what didn't
| change tested | layer | verdict | evidence |
|---|---|---|---|
| **bf16 → fp16 checkpoint (RNE)** | numerical | ✅ **THE FIX** (free, 1-line) | loops 7.9%→1.7%, acc +6pt (240-sample) |
| fp16 + stochastic rounding (SR) | numerical | ✗ worse (lowers acc) | repro + 180-sample e2e |
| fp16 + error-feedback (EF) | numerical | ✗ no better than RNE | flip-floor; 180-sample e2e |
| fp32 storage (+TF32 dots) | numerical | ✗ worse + 2× mem (slot confound) | 240-sample e2e |
| conditioning-triggered flush (#2) | algorithmic | ✗ **actively hurts** | 180-sample, monotone worse |
| conditioning-adaptive exact fallback (#1) | algorithmic | ✗ **not built** — premise broken | proxy validation |

**Bottom line: fp16-RNE checkpoint is the validated answer. No store trick, no algorithmic
lever beat it. The headroom past it in this setup is error accumulation in the weak-decay
regime, which the exact recurrent verify does NOT fix either (recurrent ≈ fp16-RNE e2e).**

---

## 1. Root cause + the fix (fp16 checkpoint)
- **Symptom:** PR 28695's chunked `(I+A)^-1` verify is "not long-output lossless" — AIME degenerates
  into repetition loops on long outputs.
- **Root cause (initially framed as):** the bf16 round-trip of the ReplaySSM checkpoint every flush
  (bf16 7-bit mantissa dots set ~1e-2 base error, bf16 store accumulates it). **Corrected later**
  (§6): the `(I+A)^-1` is actually well-conditioned; the real driver is fp16-dot error *accumulating*
  in the weak-decay (non-contracting) regime.
- **Fix:** the reconstruction/flush casts were hard-wired to `q.dtype` (bf16); changed to the
  checkpoint dtype `h0.dtype` (no-op for bf16; fp16 with `--mamba-ssm-dtype float16`). 1-line
  `replace_all`. **Landed upstream identically** (28695 `8c1ddbdb9a`).

### Rigorous e2e — 240 samples/config (60 problems × 4 repeats, CUDA-graph on)
| config | acc mean±std | loop≥50 | loop≥200 | trunc | avg_tok |
|---|---|---|---|---|---|
| recurrent (exact) | 0.583±0.012 | 6/240 (2.5%) | 6/240 | 91/240 | 26196 |
| **bf16** (original PR) | 0.588±0.018 | 19/240 (7.9%) | 13/240 (5.4%) | 95/240 | 26255 |
| **fp16 (fix)** | **0.650±0.012** | **4/240 (1.7%)** | 4/240 (1.7%) | 81/240 | 26157 |
| fp32 (store+TF32 dots) | 0.637±0.025 | 13/240 (5.4%) | 9/240 (3.8%) | 82/240 | 26222 |

**Headline business impact (fp16 vs bf16):** accuracy +6.2 pts (+10.5% rel, ~5σ); degenerate-loop
rate ~4.6× fewer (7.9%→1.7%); severe loops ~3.2× fewer; **zero cost** (same 2 bytes, perf-neutral).

---

## 2. Numerical store strategies — the full matrix
Mechanistic repro (`repro_cause.py`): exact recurrence vs chunked `(I+A)^-1` fold over many chunks,
relative state error vs fp64 + next-token-flip rate. Steady-state (1000 folds, nodecay/patho):

| store mode | grid·rule | bytes | state err | flips/1000 | verdict |
|---|---|---|---|---|---|
| bf16 RNE | 7-bit, nearest | 2B | ~2.0e-2 (accumulates) | ~20-29 | original bug |
| **fp16 RNE** | 10-bit, nearest | 2B | ~2.5e-3 | ~1-2 | **best practical** |
| fp16 SR (cvt.rs.f16x2) | 10-bit, stochastic | 2B | ~3.3e-3 (worse) | ~3-4 | adds variance, no bias to remove |
| fp16 EF (residual carry) | 10-bit + fp32 resid | 2B(+resid) | ~1.5e-3 | ~1-2 | hits floor but flips unchanged |
| tf32 (fp32 store) | exact store, 10-bit dots | 4B | ~1.4e-3 | ~1-2 | dot-limited, 2× mem |
| true fp32 | exact, exact dots | 4B | ~3e-7 | ~0 | no tensor cores (slow) |

Key insight: there's a **token-decision flip-floor**. bf16 is *above* it (loops); fp16 is *below* it.
Once below, lowering the error further (SR/EF/tf32/fp32) is invisible to the output — fp16/EF/tf32/fp32
all flip identically. So fp16-RNE is the Goldilocks point (cheapest on both grid+rule that's under the
floor). The error is dominated by the `(I+A)^-1` fold + the fp16 DOTS, not the store — which is why
SR/EF (store-only fixes) can't move the output.

### Store-mode production A/B — 180 samples/config (RNE vs SR vs EF, all fp16, CUDA-graph on)
| mode | acc mean±std | loop≥50 | loop≥200 | trunc | avg_tok |
|---|---|---|---|---|---|
| recurrent | 0.628±0.034 | 4/180 | 4/180 | 63 | 26177 |
| **rne (fp16)** | 0.628±0.034 | 11/180 | 5/180 | 65 | 26243 |
| sr | **0.594**±0.021 | 13/180 | 12/180 | 70 | 26238 |
| ef | 0.628±0.028 | 10/180 | 10/180 | 63 | 25601 |
→ SR is the only mode that *drops accuracy*; EF ties RNE. fp16-RNE confirmed best store mode.
(Implemented SR + EF in the real kernel, env-selected via `SGLANG_REPLAYSSM_STORE_MODE`; both compile
+ run correctly under CUDA graph. SR made graph-safe via `b_cache_base` per-step variation, no host sync.)

### Perf (bench_serving, fixed 512-in/512-out, range-ratio 1.0)
bf16 ≈ fp16 ≈ fp32 within ~±8% noise (C32: bf16 3991 / fp16 4303 / fp32 3963 tok/s). Precision tier is
**perf-neutral** on this MoE-heavy model; fp32's 2× checkpoint bandwidth doesn't bite.

---

## 3. SOTA landscape (gh + flashinfer search)
SR + Philox is the **universal** production technique for low-precision SSM state:
- **TRT-LLM** `modules/mamba/replay_selective_state_update.py` — a "replay SSU" (direct cousin of
  ReplaySSM), SR for fp16 / fp8-e4m3 / int8 / int16, Philox `randint4x` amortized.
- **flashinfer** `mamba/kernel_checkpointing_ssu.cuh` + `conversion.cuh` — "checkpointing SSU",
  `cvt.rs.f16x2`, "Triton bit-equality", per-lane Philox refresh.
- **vLLM** `mamba/ops/mamba_ssm.py` — SR in the SSU store.
- **PR 26929** (SGLang) — FP16 Mamba SSM cache + SR for the **SSU/Mamba2** path (Nemotron).

**Crucial distinction:** SR helps the **SSU/Mamba2** path (state rewritten *every* step → bias
accumulates → unbiased SR pays off). It does **not** transfer to GDN ReplaySSM (checkpoint written
only on flush; fp16-RNE bias already negligible; `(I+A)^-1` fold + dots dominate). **Nobody uses
error-feedback** — SR is the universal choice. **Philox rounds** = RNG quality/cost knob (default 10);
moot for us since SR doesn't help GDN, and ideal-RNG SR already lost in the repro.

---

## 4. Algorithmic levers (Codex gpt-5.5 analysis of 6 SSM papers + the kernel)
Codex ranked changes by quality × throughput ÷ cost: #1 exact-fallback (6.7), #2 cond-flush (6.0),
#3 tree-verify topk>1 (3.0), #4 residual-refinement (2.0), #5 IO cache-len tuning (2.7), #6 OSDN
preconditioning (1.6), #7 GDN-2 (1.7). Traps: Kaczmarz (keys already normed → no-op), wholesale MXR
(T_mat tiny), Neumann truncation (unstable), blind cache_len↑, tree-as-mask-only.

### Lever #2 — conditioning-triggered early flush — IMPLEMENTED + TESTED → ✗ HURTS
Verify kernel computes `row_l1(A)` and `atomic_or`s a per-slot `force_flush` bit when > threshold;
cursor kernel ORs it into the flush decision. Env: `SGLANG_REPLAYSSM_COND_FLUSH=1` +
`SGLANG_REPLAYSSM_COND_THRESH`. 180-sample A/B:
| mode | acc | loop≥50 | loop≥200 |
|---|---|---|---|
| base (fp16-RNE) | 0.617 | 6/180 | **0/180** |
| cf2.5 (moderate) | 0.633 | 12/180 | 9/180 |
| cf1.5 (aggressive) | **0.567** | **21/180** | **15/180** |
→ **Monotone worse** with flush aggressiveness. Mechanism: the flush *fold* is the lossy `(I+A)^-1`
operation; forcing early flushes *commits* the ill-conditioned reconstruction into `h0` sooner/more
often. #2 relocates the lossy fold, it doesn't avoid it.

### Lever #1 — conditioning-adaptive exact fallback — NOT BUILT (premise broken)
Validated the shared proxy first (`proxy_validate.py`) before the expensive scheduler-routing build:
| window BS=4 | proxy row_l1(A) | max\|(I+A)^-1\| | cond(I+A) |
|---|---|---|---|
| benign | 0.02 | **1.00** | 1.0 |
| nodecay | 2.80 | **1.00** | 4.9 |
| patho | 2.94 | **1.00** | 5.2 |
- The proxy **discriminates** regimes (P(proxy(patho)>proxy(benign)) = 1.000), BUT
- **`max|(I+A)^-1| = 1.00` everywhere → the inverse is WELL-CONDITIONED.** The "catastrophic
  cancellation / amplification" premise (ours + the CONTEXT brief) is **wrong**. Real driver = error
  accumulation in the non-contracting weak-decay regime, not inverse blowup.
- **Decisive:** #1's destination (exact recurrent verify) is **NOT more accurate than fp16-RNE**
  (recurrent 0.628/0.639 vs rne 0.628/0.617 across both 180-sample A/Bs; recurrent even had more
  loops). Routing bad chunks to something no-better can't help → **#1 abandoned, build correctly avoided.**

---

## 5. Capability / config findings
- topk≤1 (linear chain) only; topk>1 (EAGLE tree) falls back to recurrent (no ReplaySSM benefit).
  TP2/TP4 boot; topk>1 tree boots WITH cuda graph (refutes "EAGER-only") but replayssm falls back.
- ReplaySSM requires `no_buffer` mamba radix strategy; `page_size>1` needs `extra_buffer` (28695 rejects it).
- **Bug (28451, still upstream):** the ReplaySSM guard error f-string references the nonexistent
  `self.mamba_scheduler_strategy` → AttributeError crash on the documented default command. Fixed locally.
- Unit tests pass: `test/registered/attention/unittests/gdn/test_linear_replayssm_decode.py`.

## 6. Architecture context (tangents)
- **LFM2.5 / LFM3:** "conv + attn, NO SSM recurrence" — gated short conv (LFM3 K=7) + GQA/SWA. Uses
  the MambaPool **conv-state** cache only (`state_size=0`, no `temporal`). None of SSU/GDN/ReplaySSM/
  SR/26929 apply. LFM3 added a CuTe-DSL width-generic gated conv kernel (commit d168180 in the Liquid fork).
- **"Mamba cache" ≠ "has recurrence":** the MambaPool bundles conv-state + recurrent-state; conv-only
  models (LFM) use the pool for the conv window with a zero-sized temporal state.

---

## Artifacts (this branch, `scratch/`)
- `repro_cause.py` — mechanistic numeric repro (bf16/fp16/SR/EF/tf32 state error + token flips).
- `proxy_validate.py` — the well-conditioned-inverse finding (kills #1).
- `rigorous_suite.py` + `rigorous_all.json` — 240-sample fp16/bf16/fp32 A/B.
- `rigorous_storemode.py` + `rigorous_storemode_all.json` — RNE/SR/EF A/B.
- `rigorous_condflush.py` + `rigorous_condflush_all.json` — conditioning-flush A/B.
- `launch_*.sh` / `bench_maxprec.sh` — server launch + perf drivers.
- `ssm_papers/` — the 6 SSM papers + Codex CONTEXT brief.
- Kernel: `python/sglang/srt/layers/attention/fla/gdn_replayssm_spec_decode.py` (fp16 fix +
  env-gated SR/EF store modes + cond-flush lever — all default-off except the fp16 dtype honoring).
