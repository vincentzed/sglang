# A/B profile traces (torch profiler, rank 0, TP4, 80k prefill, 6 steps)

- `before_baseline_ring_ll_TP-0.trace.json.gz` — canon GLM-5.2-NVFP4 TP4 EAGLE,
  flag off: `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` = 15.3% of GPU time,
  295.2 us/call, plus separate FusedAddRMSNorm at 3.2%.
- `after_custom_ar_TP-0.trace.json.gz` — same launch with
  `SGLANG_OPT_USE_SYMM_MEM_CUSTOM_AR=1`: RING_LL eliminated, replaced by
  `inkling_multimem_one_shot_fused_kernel` (v3) at 258.3 us/call (13.8%);
  total GPU kernel time -2.6%.

Load in chrome://tracing or Perfetto. Full 4-rank captures for every arm
(base / nvls / nvlsforce / simple / dropin / fused-decode) live on the box
under `scratch/ar-port/profiles/` in the working tree.
