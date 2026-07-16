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

TODO(table): {default, NVLS, Simple} × {AR kernel name, AR ms & share,
per-AR ms, implied busbw, TTFT, input tok/s} + verdict.

## Step 3 — transport-only A/B

TODO(table): flag-off vs flag-on, TP4 workloads 80k/8k-c1/8k-c16, 3 reps,
medians + raws; flag-on profile shows RING_LL gone.

## Step 4 — fused decode A/B

TODO(table): decode TPOT c1/c16 with SGLANG_SIMULATE_ACC_LEN pinned; kernel
table showing QuantType-0 + NVFP4Quantize lines collapsed.
NVFP4 quant epilogue at seam A: deferred as follow-up (norm-only landed at
both seams first — call made explicitly per plan; the runner-bypass plumbing
through `quantize_hidden_states_fp4` is the balloon risk item).

## Step 5 — prefill row-aligned fused kernel: go/no-go

TODO: decision from Step 3 numbers.

## Step 6 — accuracy parity

TODO(table): GSM8K ≥ 0.92, AIME25 within ±σ of 91.25, spec + no-spec.

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
