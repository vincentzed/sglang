# DFlash Crossproduct Tree Construction

Step 2 verdict: the SGLang port uses DFlash-specific crossproduct tree
construction, not EAGLE autoregressive `propose_tree`.

Budget rule:

- `tree_width <= 1` returns `block_size`. In DFlash this preserves the linear
  verify window: one root/current token plus `block_size - 1` draft tokens.
- `tree_width > 1` first computes the full geometric tree budget
  `(tree_width ** block_size - 1) // (tree_width - 1)`.
- If `tree_budget` is set and positive, the runtime budget is
  `min(full_tree_budget, tree_budget)`.

Canonical node order:

- The v1 SGLang port uses breadth-first heap expansion with `accum_logp`
  scoring.
- The root is node `0` with parent `-1` and depth `0`.
- Each popped parent expands its own per-depth top-k row. Children are appended
  in rank order, and equal heap scores are broken by insertion counter, so ties
  are deterministic.
- The verifier consumes this root-inclusive BFS order directly as flattened
  verify-token order.

Crossproduct semantics:

- DFlash runs one parallel block forward and reads per-depth top-k marginals
  from that single pass.
- Every expandable parent at depth `d` expands from the same DFlash top-k row
  for depth `d`; cumulative logprob decides which prefix nodes receive the
  limited budget.
- This differs from EAGLE tree drafting, which performs autoregressive draft
  forwards level by level and conditions each next level on generated draft
  hidden states.

Evidence:

- `test/registered/unit/spec/test_dflash_tree_construction.py` imports the
  cloned reference `jetspec/_ref/vllm-jetspec/vllm/v1/spec_decode/dflash_tree.py`
  and asserts exact `(token_ids, parent_indices, depths)` equality on toy cases,
  including ties and the 1-wide degenerate chain.
