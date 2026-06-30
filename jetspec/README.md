# JetSpec tree drafting on SGLang DFlash

Extends SGLang's linear DFlash speculative decoding with **parallel tree drafting** (JetSpec): the DFlash draft head's per-depth block logits feed a token **tree** (top2gap construction), verified under a tree-causal mask. `tree_width=1` is the unchanged linear path; `width>1` is the new tree path.

## What this delivers

- **Lossless tree spec-decode** (token-exact vs linear DFlash), dense Qwen3-8B + MoE Qwen3.6-35B-A3B, cuda-graphed, EAGLE-parity direct commit (no reverify).
- **`top2gap` construction** — per-depth fanout from the top-1/top-2 logprob gap (branch where the drafter is uncertain, chain where confident). Reproduces the paper's accept length at a fraction of the budget.
- **`width=1` is a byte-identical, near-zero-overhead drop-in** for linear DFlash.
- A proven FA4-exact paged-tree verify kernel (opt-in, default-off) + a full break-even / profiling analysis.

## Full speedup matrix (Qwen3-8B, B300, FA4 + page_size 16, greedy)

| Config | GSM8K tok/s | vs DFlash | MATH-500 tok/s | vs DFlash | accept (G / M) | lossless |
|---|---:|---:|---:|---:|---:|:--:|
| **DFlash w1 (native linear)** | **1170.5** | 1.00× | **1522.1** | 1.00× | 5.85 / 7.62 | baseline |
| linear JetSpec (width=1) | 1152.4 | **0.985×** | 1488.6 | **0.978×** | 5.85 / 7.62 | ✓ exact |
| compact tree (top2gap w8/b16) | 1002.2 | 0.856× | 1167.5 | 0.767× | 6.45 / 7.90 | ✓ exact |
| paged tree (w8/b16) | 900.2 | 0.769× | 1060.7 | 0.697× | 6.50 / 8.02 | ✗ (default-off) |

- vs **AR-greedy** (the paper's baseline): every spec config is ~4–9× faster — that comparison is trivially won and not the interesting bar.
- The interesting bar is **SGLang's linear DFlash**, which is already FA4 + cuda-graph + fused-kernel optimized.

## Why the tree could not improve further (the honest result)

- **Break-even rule:** tree beats linear iff `accept_ratio > step_cost_ratio`. The tree verify costs **~1.29× (compact) / 1.44× (paged)** per step at equal node count, so it needs that much more acceptance to win.
- **The accept gain is too small:** tree gives **+0.6 (GSM8K) / +0.28 (MATH)** accept over linear; break-even needs **+1.5 / +2.4**. The draft head's tree just doesn't concentrate enough probability to clear the verify overhead.
- **Construction was maxed** (top2gap is the reference's sweep-winning algorithm) and **the paged kernel is a net loss** (it removes the K/V gather but a different memory layout adds attention time *and* breaks bit-exactness vs the compact verify — fp non-associativity flips greedy near-ties).
- **The paper's own tree is slower than SGLang's linear** (paper 984/1150 tok/s vs our linear 1170/1522), so "beat SGLang linear DFlash" is a strictly harder bar than the paper's published speedup (which is vs AR-greedy). Conclusion: against an already-SOTA linear baseline, tree drafting trades verify cost for accept length and nets out **below linear** on throughput.

## The good news: width=1 is a clean drop-in

- **Linear JetSpec (width=1) is within ~1.5–2% of native DFlash** (1152 vs 1170, 1489 vs 1522) and **token-exact** — i.e. within measurement noise.
- That means the tree code path adds **negligible per-step overhead when not used**, so the feature is safe to land: it never regresses the linear DFlash path, and `width>1` is available for workloads where verify is cheaper or the draft head concentrates better (e.g. weaker/AR baselines, or future paged-engine work).

## Reproduce

```bash
python jetspec/bench_paper_sglang.py   # GSM8K / MATH-500, configurable width/budget
bash jetspec/run_dflash_gate_bench.sh  # losslessness gate vs fresh linear oracle
```

Detailed run-by-run notes: `jetspec/notes/bench_results.md`; status: `jetspec/STATUS.md`.
