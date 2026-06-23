# ReplaySSM spec-verify: fp16-checkpoint investigation

Base: PR #28695 (`gdn_replayssm_spec_decode`, Yuan Luo) — ReplaySSM GDN spec-verify ring.
Model: `Qwen/Qwen3.6-35B-A3B` (GDN-hybrid), B300, TP1 unless noted. Greedy unless noted.

## Problem
PR 28695's body documents that its chunked `(I+A)^-1` verify is **not long-output lossless**
(AIME degenerates into repetition loops). Root-caused here to the **bf16 round-trip of the
ReplaySSM checkpoint** every flush: bf16 (7-bit mantissa) dots set a ~1e-2 base error and the
bf16 store accumulates it; fp16 (10-bit) cuts both ~10x at the **same 2 bytes**.

## Patch (this branch)
`python/sglang/srt/layers/attention/fla/gdn_replayssm_spec_decode.py`: the reconstruction/store
casts were hard-wired to `q.dtype` (bf16); changed to the **checkpoint dtype `h0.dtype`** (no-op
for bf16, fp16 when `--mamba-ssm-dtype float16`). One `replace_all` of `q.dtype.element_ty` ->
`h0.dtype.element_ty`. Enable with `--mamba-ssm-dtype float16`.

Launch (SM100): `--linear-attn-decode-backend triton --attention-backend triton --page-size 1
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4 --enable-gdn-replayssm-spec --gdn-replayssm-spec-cache-len 16
--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --mamba-ssm-dtype float16`
(env: `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`).

## Result — rigorous suite (60 AIME-24+25 problems x 4 repeats = 240 samples/config, greedy, max 32k)
loop = max single-line repeat in (reasoning+content), lines>8 chars; loop>=50 pathological, >=200 severe.

| config                  | acc mean±std | loop>=50    | loop>=200  | trunc      | avg_tok |
|-------------------------|--------------|-------------|------------|------------|---------|
| recurrent (exact)       | 0.583±0.012  | 6/240 2.5%  | 6/240      | 91/240     | 26196   |
| bf16 (current PR)       | 0.588±0.018  | 19/240 7.9% | 13/240 5.4%| 95/240     | 26255   |
| **fp16 (this patch)**   | **0.650±0.012** | **4/240 1.7%** | 4/240 1.7% | 81/240 | 26157   |
| fp32 (storage+TF32 dots)| 0.637±0.025  | 13/240 5.4% | 9/240 3.8% | 82/240     | 26222   |

Perf (bench_serving, fixed 512/512, range-ratio 1.0): bf16 ≈ fp16 ≈ fp32 within ~±8% noise
(C32: bf16 3991 / fp16 4303 / fp32 3963 tok/s) — precision tier is perf-neutral on this MoE model.

**Verdict: fp16 checkpoint dominates** — ~4.6x fewer loops than bf16 + ~+6pt acc (~5σ), at 2B (free
mem/bandwidth) and perf-equivalent. fp32 is worse+noisier (slot-capacity confound: 534 vs 1246 slots).

## Stochastic rounding (PR #26929) — evaluated, does NOT help here
Ported the idea into the numeric repro (`fp16sr` mode = SR-rounded fp16 store, as PR #26929 does for
the Mamba2/SSU path on Nemotron). Result: SR is **consistently worse** than plain fp16 RNE for the
ReplaySSM checkpoint, even at 1000 folds (~16k tokens) — nodecay 3.3e-3 vs RNE 2.5e-3, patho 3.7e-3
vs 2.9e-3; flips 4 vs 2 / 3 vs 1. Reason: SR removes rounding *bias* but adds *variance*, and fp16's
10-bit RNE bias is already negligible here (the `(I+A)^-1` fold + fp16 dots dominate the error, not
the store bias). SR helps Nemotron's SSU path because it rewrites the state EVERY decode step (bias
accumulates step-by-step); ReplaySSM only writes the checkpoint on flush, where fp16-RNE already nails
it. **Conclusion: the PR's fp16 fix (commit 8c1ddbdb9a) is the right & sufficient solution; SR is not
worth porting to ReplaySSM.** (tf32 is marginally flatter but costs 2x checkpoint bandwidth.)

## Artifacts in this dir (minimal reproducible set; each file has a `VALUE:` header)
- `repro_cause.py` — deterministic, model-free numeric repro (per-fold state error + token-flip
  rate across bf16/fp16/tf32/fp32). The *mechanistic* evidence; runs in seconds on one GPU.
- `launch_maxprec.sh` — brings up the 4 servers (recurrent / bf16 / fp16 / fp32) on ports 31010-13.
- `rigorous_suite.py` — the 240-sample/config e2e accuracy suite (the headline result generator).
- `rigorous_all.json` — its raw output: `{config: [{i, rep_i, gold, ans, maxrep, trunc, tok}, ...]}`
  for all 4 configs (240 samples each). Canonical data behind the table above.
- `bench_maxprec.sh` — fixed-512/512 `bench_serving` perf across the 4 servers (the perf half).

Repro: `bash launch_maxprec.sh` → wait until all 4 ports answer `/get_model_info` →
`python scratch/rigorous_suite.py` (accuracy) and `bash scratch/bench_maxprec.sh` (perf).
(Superseded single-run head-to-head scripts and the decode-PR launchers were removed in the
cleanup commit; recover from history if needed.)
