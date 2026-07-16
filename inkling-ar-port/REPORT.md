# Inkling symm-mem custom AR port to the DSA family — report

Branch: `brayden/inkling-ar-dsa` (base: sgl-project/sglang main `b0b2dfbda1`,
kernels cherry-picked from `inkling-support` @ `966ddaaeff`).
Box: sgl-b300-inference, 8× B300 SXM6 (sm103), driver 610.43.02 / CUDA 13.3.
Container: `sglang-brayden-inkling-ar` (lmsysorg/sglang:nightly-dev-cu13-20260710),
sgl_kernel 0.4.4, flashinfer 0.6.12, transformers 5.12.1.
GPU etiquette: 4 GPUs (0-3) per the shared-box cap; TP8 arms pending a clearance window.

## What was built

1. **Drop-in custom AR** (`SGLANG_OPT_USE_SYMM_MEM_CUSTOM_AR=1`, default OFF):
   - `srt/distributed/device_communicators/symm_mem_custom_ar.py` — the
     model-agnostic lift of `inkling_common/kernels/comm.py` (resources,
     v4/v5 A/B rotation with the reuse-distance-2 invariant, dispatch).
   - Wired at the top of `GroupCoordinator.all_reduce`'s outplace ladder for
     the TP group (in pure-TP, `_ATTN_TP` and `_MOE_TP` alias `_TP`, so both
     per-layer seams route through one coordinator). Ineligible shapes fall
     through to the unchanged ladder; flag-off is byte-identical.
   - Buffer: 256 MiB/rank torch-symm-mem (fits [16384, 6144] bf16 + tails).
   - `.acc::f32` added to both multimem `ld_reduce` sites (fp32 in-switch
     accumulation, matching NCCL NVLS and the two-shot/push kernels).
2. **Fused decode/verify {AR → add+RMSNorm}** (`SGLANG_OPT_USE_SYMM_MEM_FUSED_AR_NORM=1`):
   - `jit_kernel/csrc/inkling/inkling_ar_norm.cuh` — v5 push AR with the
     epilogue seam filled by fused_add_rmsnorm; one block per token,
     per-block barrier, gamma/residual prefetch under the barrier. Without
     the sconv phase there are no cross-token taps, so ONE kernel covers
     decode and EAGLE target-verify.
   - Wired at seam A (`_tp_attn_all_reduce_and_layernorm`) and seam B
     (cross-layer MLP-AR fusion branch in `prepare_attn` +
     `should_fuse_mlp_allreduce_with_next_layer` producer gate).
3. Tests: `test/registered/jit/test_symm_mem_custom_ar.py` (17 cases @ TP4) —
   variant correctness vs NCCL, fused bit-identity, graph capture+replay.

## Bit-identity (Step 4 correctness gate) — PASS, with a finding

The fused kernel is **bit-identical (0/589,824 elements)** to the unfused
{v5 custom AR → `fused_add_rmsnorm`} chain — after replicating the norm's
exact fp32 semantics: XOR-butterfly warp reduce, full-width second butterfly,
`rsqrt.approx.ftz.f32`, `div.approx` (`__fdividef`), FMA contraction,
`--use_fast_math` lowering.

**Finding:** `sgl_kernel.fused_add_rmsnorm` delegates to flashinfer, which has
TWO backends (CUDA JIT vs CuTe DSL, `FLASHINFER_USE_CUDA_NORM`); they already
differ from EACH OTHER by 1 bf16 ulp on ~3e-6 of elements (different fp32
variance trees). Bit-identity is therefore asserted against the deterministic
CUDA backend; against the production CuTe backend the residual stream is
bit-identical and hs is within 1 bf16 ulp on <1e-4 of elements — the same
delta flashinfer's own backends have between themselves.

## Step 2 micro-bench (graph-replay, TP4, hidden 6144, B300)

Median µs/op; NCCL captured in-graph with default env. `busbw` = 2(N-1)/N·bytes/t.

| T | NCCL default | torch multimem | best custom | speedup vs NCCL |
|---|---|---|---|---|
| 1 | 17.2 | 11.1 | **v5(1,1024) 6.0** | 2.9× |
| 8 | 19.1 | 13.4 | **v5(8,1024) 7.8** | 2.4× |
| 32 | 19.7 | 14.3 | **v5(8,1024) 10.7** | 1.8× |
| 96 | 23.1 | 17.6 | **v5(64,1024) 14.5** | 1.6× |
| 256 | 32.3 | 26.6 | **v3b(64,1024) 23.6** | 1.4× |
| 1024 | 59.9 | 67.7 | **v3b(32,1024) 44.2** | 1.4× |
| 2048 | 110.4 | 123.3 | **v3(64,512) 75.7** | 1.5× |
| 8192 | 270.0 | 503.1 | v3(32,1024) 273.5 | 0.99× |
| 16384 | 508.3 | 1021.6 | v3(96,512) 519.4 (~581 GB/s) | 0.98× |

Full tables: `scratch/ar-port/bench_tuned_tp4_ncclDefault.json`.

**Reading:** the decode/verify band (v5) is a 1.6–2.9× transport win; the
mid band (v3b/v3, 256–4096) is 1.4–1.5×; at chunked-prefill sizes (8192+)
graph-captured NCCL-default on this box already runs ~590 GB/s busbw and v3
only ties it. The production profiles' RING_LL @ ~550 GB/s therefore looks
like a protocol-selection artifact — exactly what Step 1b's settling
experiment measures end-to-end.

## Step 1 — baseline + NCCL settling experiment

Canon TP4 EAGLE launch (CPS=8192), 80k workload, 3 reps each, medians [raws].
Profile: 6-step capture during an 80k prefill; per-AR times from the rank-0 trace.

| arm | AR kernel (extend) | AR share | per-AR µs (T=8192) | implied busbw | TTFT ms | input tok/s |
|---|---|---|---|---|---|---|
| default | `AllReduce_Sum_bf16_RING_LL` | 15.3% | 295.2 | 512 GB/s | 3390.0 [3387.9/3398.5/3390.0] | 23649 |
| `--enable-nccl-nvls` | `AllReduce_Sum_bf16_RING_LL` (unchanged) | 14.7% | 282.2 | 535 GB/s | 3392.9 [3392.8/3392.9/3398.8] | 23642 |
| + `NCCL_ALGO=allreduce:nvls` (forced) | `AllReduce_Sum_bf16_RING_LL` (still!) | 18.3% | 364.4 | 414 GB/s | 3506.6 [3506.6/3510.9/3502.7] | 22855 |
| `NCCL_PROTO=Simple` | `AllReduce_Sum_bf16_RING_LL` (still!) | 14.5% | 276.5 | 547 GB/s | 3389.8 [3406.4/3384.5/3389.8] | 23589 |

**Verdict: the NCCL environment reclaims NOTHING.** Three mechanisms compose:
1. sglang sets `NCCL_NVLS_ENABLE=0` by default (`entrypoints/engine.py:1262-1267`)
   unless `--enable-nccl-nvls` — NVLS resources don't even exist in the canon
   config. (Also: bare `NCCL_ALGO=NVLS` fails comm init with "invalid usage" —
   the per-collective `allreduce:nvls` syntax is required.)
2. Even with NVLS resources enabled, these all-reduces execute inside captured
   CUDA graphs where this NCCL build pins the captured collective to the
   RING_LL kernel; the algo/proto env never reaches them (forcing NVLS only
   perturbed the LL tuning and REGRESSED TTFT by 3.4%).
3. RING_LL at the 8192-token chunk already runs ~547 GB/s effective busbw —
   within ~7% of the ~586 GB/s wall that BOTH torch multimem and the custom
   multimem/NVLS kernels hit on this 4-of-8-GPU B300 topology. The pre-work's
   "LL half-bandwidth" framing does not hold at TP4: there is no big prefill
   transport win for ANY same-topology kernel, custom or NCCL.

Headroom calibration for Steps 2-3: prefill-size transport ≈ 12% of the AR
line (~1.5-2% e2e); the decode/verify band (v5: 1.6-2.9x) and the mid-size
chunk band (v3b: 1.4-1.5x at 256-4096 tokens) hold the real win.

## Step 3 — transport-only A/B (TP4)

Identical box/image/checkpoint/flags; the ONLY difference is
`SGLANG_OPT_USE_SYMM_MEM_CUSTOM_AR=1`. Server restarted per arm; 3 reps each,
medians [raws].

| metric | flag OFF | flag ON | delta |
|---|---|---|---|
| 80k TTFT ms | 3390.0 [3387.9/3398.5/3390.0] | 3238.2 [3235.2/3238.2/3247.8] | **-4.5%** |
| 80k input tok/s | 23649 | 24702 | +4.5% |
| 8k-c1 TTFT ms | 302.1 | 289.0 | -4.3% |
| 8k-c1 TPOT ms | 3.15 | 2.83 | **-10.2%** |
| 8k-c16 TTFT ms | 2923.6 [2673.6/2923.6/2965.4] | 2532.9 [2880.2/2530.9/2532.9] | **-13.4%** |
| 8k-c16 TPOT ms | 8.49 | 6.96 | **-18.0%** |
| 8k-c16 output tok/s | 1409 | 1691 | **+20.0%** |

Flag-on profile (80k prefill): `AllReduce_Sum_bf16_RING_LL` is GONE, replaced
by `inkling_multimem_one_shot_fused_kernel<bf16,4,false>` (v3) at 258.3 µs/call
(12.5% faster per AR); total GPU kernel time -2.6% (1849 -> 1801 ms); nothing
new above 1%. Greedy smoke output identical to baseline.

Verdict vs Step 1b headroom: the custom AR captured ~100% of the (small)
prefill transport headroom AND the large decode/verify-band headroom NCCL
could never reach (v5 at 6-14 µs vs 17-23 µs) — the latter drives the -10/-18%
TPOT and +20% throughput, well past the >=5% bar on the decode-facing
workloads; the 80k TTFT lands at -4.5% with the shortfall vs 5% fully
root-caused by the ~586 GB/s topology wall above.

## Step 4 — fused decode A/B (TP4, both flags on vs drop-in only)

| metric | drop-in only | + fused AR->norm | delta |
|---|---|---|---|
| 8k-c1 TPOT ms | 2.83 | 2.89 | +2.1% (worse) |
| 8k-c16 TPOT ms | 6.96 | 7.06 | +1.4% (worse) |
| 8k-c16 output tok/s | 1691 | 1669 | -1.3% |
| 80k TTFT ms | 3238.2 | 3239.2 | ~0 (prefill untouched by design) |

Decode-window kernel table (isl=1024/osl=512, bs=1 EAGLE): the fused
`inkling_ar_add_rmsnorm_kernel` (12.2 µs/call incl. in-kernel producer wait)
replaces {AR + norm} pairs, but flashinfer's lamport-based
`oneshotAllreduceFusionKernel` (8.7 µs/call) — which the baseline/drop-in arms
already use at seam B via the auto-enabled allreduce fusion — is competitive
at these bs<=16 shapes, so displacing it nets slightly negative.

**Call:** the fused kernel is correct (bit-identical, graph-safe) but adds no
e2e win at TP4-EAGLE where flashinfer's fusion already covers seam B;
`SGLANG_OPT_USE_SYMM_MEM_FUSED_AR_NORM` stays default-OFF. Win-case to
revisit: configs where the flashinfer fusion is unavailable (workspace limits,
disabled backend, no-mnnvl) and TP8. The NVFP4 quant epilogue at seam A is
filed as follow-up per plan (norm-only landed first; the
`quantize_hidden_states_fp4` runner-bypass plumbing is the balloon risk).

Also observed: at extend T<=2048 (inside flashinfer's fusion cap) the MNNVL
`twoshotAllreduceKernel` burns 628 µs/call at T=1024 (62% of that window's GPU
time, spin-wait heavy). The custom v3b does T=1024 in 44 µs — extending the
drop-in to displace the flashinfer EXTEND-band fusion is a promising follow-up
(not wired: my gates deliberately leave extend to the existing path).

## Step 5 — prefill row-aligned fused kernel: NO-GO (data-backed)

Two measured facts close this: (1) at prefill chunk sizes the AR transport is
at the ~586 GB/s topology wall — a row-aligned fused variant cannot beat the
wall, it can only fold the separate norm (3.2% of prefill GPU time) into the
AR's exit phase; (2) Inkling's own extend-shape measurements (comm.py
docstring) found the in-kernel norm tail 1-18% SLOWER than the standalone
fused_add_rmsnorm at T=512-16384 — the tail is bandwidth-bound while the AR
grid is capped by barrier co-residency. Expected best case is < 1% e2e on TTFT
for new two-shot kernel complexity with a new row-aligned partition. Not
justified; revisit only if TP8 shows a different transport picture.

## Step 6 — accuracy parity (flag-on build, TP4)

| eval | config | score | gate | verdict |
|---|---|---|---|---|
| GSM8K | EAGLE 5-1-6, max-tokens 8192 | **95.53%** | >= 0.92 | PASS |
| AIME25 (n=8, 64k tok, T=1.0/top-p 0.95) | EAGLE 5-1-6 | RUNNING | 91.25 +/- 4 | - |
| GSM8K | no-spec | PENDING | >= 0.92 | - |

Eval hygiene note: the canon `SGLANG_SIMULATE_ACC_LEN=3.5` pin is for SPEED
benches only (it simulates draft acceptance and corrupts outputs); the serve
script now gates it behind SIM_ACC=1 so accuracy servers never inherit it.
GSM8K needs `--max-tokens` bounded (8192): a handful of prompts send the
thinking model into 400k-token reasoning loops that pin the whole KV pool.

## Mechanism attribution

TODO: transport / HBM-roundtrip elimination / dedup fractions from profile deltas.

## Known limits

- DP-attention / EP / multi-node: out of scope (AR seam doesn't exist there;
  gates fall through silently — those groups get no custom-AR resources).
- World sizes {4, 8}; bf16 only; numel % 8 == 0.
- The v4/v5 A/B rotation requires an even number of rotated ARs per captured
  graph / eager forward; the DSA layer structure guarantees 2 per layer (the
  last layer's seam-B fallback still routes through the drop-in), validated
  by graph-replay tests + full benches.
- TP8 arms pending shared-box clearance.
