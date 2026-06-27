# Old attempt (realtree) — handoff artifacts

The previous Codex run removed the branch-reverify band-aid and implemented the REAL target
tree verify, but stopped (interrupted) before it was fully lossless. It got VERY close and
isolated the bug. Use this to continue — do NOT start over.

## State of the code (already in the working tree, compiles cleanly)
WIP edits (also backed up as `jetspec/_oldattempt/realtree_wip.patch`) touch:
- `python/sglang/srt/speculative/dflash_worker_v2.py` (325-line rework: real tree verify, band-aid removed)
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` (route DFlash tree through the topk>1 verify path)
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`, `base_runner.py`, `eager_runner.py`, `model_runner.py` (cuda-graph wiring for tree decode)
- `python/sglang/srt/speculative/{dflash_info.py,dflash_tree_utils.py,spec_info.py}`, test file

## THE KEY FINDINGS (save you hours)
1. **`jetspec/baseline_linear.json` is STALE** — it was captured under a different launch
   mode/config (flashinfer attn / graph state). The CURRENT width=1 linear DFlash (no-graph)
   does NOT match it. So DO NOT gate losslessness against `baseline_linear.json`.
   **Use `jetspec/runs/8b_linear_current_nograph_31216.json` as the current-code 8B linear
   oracle** (recaptured from the exact current code/launch). Several earlier "mismatches"
   (incl. all of prompt 0) were FALSE ALARMS from the stale baseline.
2. **Against the fresh oracle, the real remaining bug is OVER-ACCEPTANCE in the tree-greedy
   accept path**, reproducing on 8B w7/b64 at **prompts 3, 5, 9 only** (first diffs ~@51/@36/@9).
   "Over-acceptance" = the tree accept commits a branch token the target's own greedy argmax
   would NOT have produced. Prompt 5's first diff is a simple duplicate token — easiest to trace
   the bad accept chunk. This is an accept-selection / node-gating bug, NOT a draft-tree, KV, or
   mask-construction bug (cap=0 root-only verification was internally consistent).

## What was tried (don't repeat)
- cachefix / tritonattn / nograph variants — all still failed because they were compared vs the
  STALE baseline; the real comparator is the fresh oracle above.
- cap=0 (root-only tree) — confirmed the divergence is in accept logic, not deeper tree rows.

## Failing-run artifacts
`jetspec/runs/8b_tree_w7_b64_realtree*.json`, `..._debug_prompt0_*.json` — contain per-prompt
`losslessness.mismatches` with first_diff positions.

## The fix to make
Match the accept semantics of the reference tree verify in
`jetspec/_ref/JetSpec/jetspec/core/tree_attention_kernel.py` and the verify stack in
`jetspec/_ref/JetSpec/jetspec/inference_engine/compiled_verify_stack.py` /
`paged_tree_attn.py`. The over-acceptance is almost certainly in how the accepted path is
selected from the target's per-node logits (greedy: accept child c at node n iff
target_argmax(logits[n]) == draft_token(c); stop at first node whose best child != target argmax).
