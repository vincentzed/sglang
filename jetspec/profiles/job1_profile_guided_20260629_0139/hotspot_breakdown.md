# Job 1 DFlash tree decode hotspot breakdown

Date: 2026-06-29 UTC

Mode:
- Linear mapping server: width=1 DFlash, `--attention-backend fa4 --page-size 16`, port 31910.
- Tree formal server: width=7, budget=64 DFlash, `--attention-backend fa4 --page-size 16`, port 31911.
- Both servers used `CUDA_VISIBLE_DEVICES=7`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`, `PYTHONPATH=python`, `--max-running-requests 1`, `--cuda-graph-max-bs-decode 1`, `--cuda-graph-backend-decode full`.
- `--mem-fraction-static 0.35` was used only so the two profile servers could co-reside on one B300. The paper benchmark rows in `jetspec/runs/final_*` used the normal single-server setup.

Profile command:

```bash
python3 /root/.claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --framework sglang \
  --mapping-url http://127.0.0.1:31910 \
  --formal-url http://127.0.0.1:31911 \
  --mapping-output-dir jetspec/profiles/job1_profile_guided_20260629_0139/linear_w1_decode \
  --formal-output-dir jetspec/profiles/job1_profile_guided_20260629_0139/tree_w7_b64_decode \
  --mapping-profile-prefix linear_w1_decode \
  --formal-profile-prefix tree_w7_b64_decode \
  --profile-workload decode \
  --warmup-steps 10 --num-steps 5 \
  --kernel-table-limit 0 --overlap-table-limit 0
```

Artifacts:
- Two-trace tables: `jetspec/profiles/job1_profile_guided_20260629_0139/two_trace_decode_analysis.txt`
- Linear single-trace tables: `jetspec/profiles/job1_profile_guided_20260629_0139/linear_w1_single_trace_analysis.txt`
- Tree single-trace tables: `jetspec/profiles/job1_profile_guided_20260629_0139/tree_w7_b64_single_trace_analysis.txt`
- Linear trace: `jetspec/profiles/job1_profile_guided_20260629_0139/linear_w1_decode/decode/1782697483.114899/linear_w1_decode-decode-1782697483.1781414-TP-0.trace.json.gz`
- Tree trace: `jetspec/profiles/job1_profile_guided_20260629_0139/tree_w7_b64_decode/decode/1782697503.2490833/tree_w7_b64_decode-decode-1782697503.3644614-TP-0.trace.json.gz`

## Summary

The profile confirms the current tree bottleneck is not just FA4 attention math. The tree path introduces host-side metadata construction, GPU tensor construction from Python lists, compact K/V gather/index setup, and synchronization before and around the compact target verify.

GPU-visible decode-window total from the single-trace kernel shares:
- Linear width=1: about `31.6 ms` over the captured active decode steps.
- Tree w7/b64: about `48.6 ms` over the captured active decode steps.
- Tree adds about `17.0 ms` GPU-visible work in the profiler window, or roughly `3.4 ms` per active profiled decode step.

CPU/Python trace signal:
- Linear `forward_batch_generation`: `28.309 ms / 5 calls` (`5.66 ms/call`).
- Tree `forward_batch_generation`: `109.933 ms / 5 calls`.
- Tree `_forward_batch_generation_tree`: `92.063 ms / 4 calls` (`23.02 ms/tree call`).

The paper-harness end-to-end reference remains: final linear rows are about `5.07 ms/step`; final tree w7/b64 rows are `14.73 ms/step` on GSM8K and `15.94 ms/step` on MATH-500.

## Top tree-only sinks

| Hotspot | Captured cost | Evidence | Actionability |
|---|---:|---|---|
| GPU tensor construction from host lists / pageable H2D copies in `_forward_batch_generation_tree` | `aten::copy_ 42.843 ms`, `aten::to 34.835 ms`, `aten::_to_copy 34.276 ms`; `cudaStreamSynchronize 36.524 ms / 972 calls`; top four syncs are about `26.6 ms` total | Trace stacks point at `_forward_batch_generation_tree` and DFlash metadata construction. Kernel table has `Memcpy HtoD (Pageable -> Device) 1.28 ms / 925 launches`, with `dflash_tree_utils.py:460` and `dflash_worker_v2.py:1905` as the main DFlash sites. | First target. Replace per-step list-to-CUDA tensor construction and pageable copies with preallocated/static buffers or a GPU-side/vectorized metadata path. |
| Tree mask / ancestor / retrieve metadata construction | `build_tree_custom_mask 7.452 ms / 4`, `build_ancestor_matrix_from_parents 7.008 ms / 4`, `build_retrieve_links_from_parents 4.348 ms / 4` | These utilities call `.tolist()` and build small CUDA tensors on the hot path. | High. Avoid building unused dense masks on compact FA4 paths, and move required metadata into fixed buffers. |
| Compact K/V movement and gather/index setup | `move_kv_cache_overlap_safe 7.025 ms / 4`; kernels: `index_copy_ 1.17 ms`, `index_select 1.14 ms`, D2D memcpy `1.07 ms` | Two-trace table and tree single-trace table show serialized gather/index kernels. | Medium-high. The compact verifier needs prefix+ancestor K/V, but the index setup should be made static/graph-friendly where possible. |
| Ragged compact verifier attention shape | Tree compact FA4 attention `5.38 ms / 144 launches`; linear FA4 attention `2.59 ms / 200 launches` | Tree uses one compact query row per node with prefix+ancestor gather. | Medium. This is probably the irreducible compute floor unless the verifier is restructured without changing FA4 semantics. |
| Scalar syncs / dynamic shape checks | `aten::item 2.142 ms`, `_local_scalar_dense 2.100 ms`, `.item()` Python sites `2.178 ms / 46 calls` | Smaller than the long tensor-construction syncs, but still on the decode critical path. | Medium. Remove once the larger metadata construction is cleaned up. |
| Draft top-k and CPU tree build | `_topk_from_vocab_parallel_head 1.705 ms / 4`, `build_tree_from_topk_cpu 1.025 ms / 4`, `_build_tree_cpu 0.749 ms / 4`, `topk 0.379 ms / 4` | Present but below metadata/sync and compact K/V costs. | Lower priority for the first pass. |

## First optimization target

The first optimization pass should attack the measured metadata/sync sink, not the FA4 kernel. The likely highest-return checks are:

1. In `_forward_batch_generation_tree`, stop constructing tiny CUDA tensors from Python lists every step (`torch.tensor(..., device=device)`) and replace them with preallocated buffers or one controlled copy path.
2. In the compact FA4 tree path, avoid building dense custom masks if they are not consumed by the compact verifier.
3. Keep all correctness gates token-exact against a fresh flushed FA4 page16 width=1 oracle before committing any optimization.
