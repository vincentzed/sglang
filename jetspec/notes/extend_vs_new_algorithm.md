# Extend DFLASH vs Register JETSPEC

Verdict: extend the existing `DFLASH` speculative algorithm with
`tree_width` / `tree_budget` knobs. Do not register a new `JETSPEC`
`SpeculativeAlgorithm` for v1.

Reasons:

- JetSpec checkpoints load as `DFlashDraftModel`; there is no new draft model
  family or retrained draft head to select.
- The existing DFlash worker already owns the draft runner, mask-token
  resolution, target hidden-state capture, draft KV materialization, overlap
  scheduling, and DFlash request validation.
- `tree_width=1` must remain the exact current linear DFlash path. A separate
  algorithm name would duplicate the same setup code and make the compatibility
  guarantee harder to audit.
- The new behavior is a verifier/proposal topology change after the same single
  DFlash block forward: width 1 means linear candidates, width greater than 1
  means crossproduct tree candidates.

Implementation implication:

- Keep `SpeculativeAlgorithm.DFLASH`.
- Add DFlash-specific config fields:
  `speculative_dflash_tree_width`, `speculative_dflash_tree_budget`,
  `speculative_dflash_tree_draft`, and `speculative_dflash_head_type`.
- Route only inside DFlash worker code. EAGLE, FrozenKV MTP, STANDALONE, and
  NGRAM remain untouched.
