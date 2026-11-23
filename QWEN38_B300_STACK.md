# Qwen3.8-Flash-Next NVFP4 on one B300: integration stack

This branch is upstream `main` (1918da6349, 2026-11-22) plus open pull requests that matter for
`nvidia/Qwen3.8-Flash-Next-NVFP4` on a single B300 (TP1, AgentX trace), plus our own changes.
It is a working stack for measurement, not a merge candidate. Every change of ours is also a separate,
review-ready branch on this fork (`pr/*`, table below).

## Results

AgentX trace, 1200 s runs, one B300, total tokens/s (prompt tokens including cache reads + generated tokens).
Speculative acceptance is pinned at 2.32 in both engines (benchmark rule), so these are speed numbers only.

| Engine | Concurrency | Runs | Mean |
|---|---|---|---|
| TokenSpeed 0.1.1.post20261117 (its recipe) | 108 | 318,463 / 317,617 | 318,040 |
| TokenSpeed | 128 | 278,755 | |
| SGLang, this branch | 124 | 395,685 / 395,141 / 389,827 | 393,551 (123.7% of TokenSpeed) |
| SGLang main, published recipe (600 s) | 108 | 170,086 | |

124 users is the knee for this stack (136 was lower on earlier trees); TokenSpeed's best is 108.
Accuracy on this exact tree, real speculative verification (no simulated acceptance): GSM8K 97.8 +- 0.4
(strict and flexible), mean accept length 2.892. The MXFP8 projections were also checked on GPQA-diamond
(81.3 vs 80.3 BF16).

## How to run

1. Start from the SGLang nightly image `lmsysorg/sglang:nightly-dev-cu13-20261122-e96e9675` and check out this branch.
2. FlashInfer 0.7.5 Python package as an overlay (the image ships 0.7.4):
   `uv pip install --target /opt/fi075 --no-deps flashinfer-python==0.7.5.post1`. The first start JIT-compiles for
   about 14 minutes; run the server once before measuring.
3. `SGLANG_RUST_BUILD_MODE=auto` builds the radix-tree extension from this checkout (45 s once).
4. Host memory: `--hicache-size 690` pins about 690 GB of host RAM plus ~200 GB of other pinned buffers.
5. Launch:

```bash
env \
  SGLANG_TIMEOUT_KEEP_ALIVE=1800 \
  SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=0 \
  SGLANG_OPT_SPEC_COMPACT_VERIFY_SAMPLING=1 \
  FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN=0 \
  PYTHONPATH=/opt/fi075:<checkout>/python \
  SGLANG_ENABLE_QSA_SPARSE_DECODE_CUDA=1 \
  SGLANG_OPT_GDN_REPLAYSSM_LAZY_FOLD=1 \
  SGLANG_HICACHE_LOAD_BLOCK_QUOTA=8 \
  SGLANG_HICACHE_MAMBA_LOAD_BLOCK_QUOTA=8 \
  SGLANG_MAMBA_PATH_CAP_COUNTS_HOST=1 \
  SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT=1 \
  SGLANG_RUST_BUILD_MODE=auto \
  SGLANG_ENABLE_QSA_INDEX_SCORE_DECODE_CUDA=1 \
  SGLANG_ENABLE_QSA_SPARSE_DECODE_SM100=1 \
  SGLANG_ENABLE_FLASHINFER_HC_SUM=1 \
  SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1 \
  SGLANG_ENABLE_QSA_PREFILL_SELECT_CUDA=1 \
  SGLANG_ENABLE_GDN_REPLAYSSM_VERIFY_CUDA=1 \
  SGLANG_ENABLE_QSA_SPARSE_PREFILL_SM100=1 \
  SGLANG_OPT_GATED_NORM_WIDE_ROWS=1 \
  SGLANG_OPT_GDN_PREFILL_SPLIT_CONV=1 \
  SGLANG_MAMBA_PATH_CAP_GRID=16384 \
  SGLANG_ENABLE_QSA_PREFILL_TOPK_CUDA=1 \
  SGLANG_OPT_GDN_VERIFY_PROJ_VIEWS=1 \
  SGLANG_OPT_QWEN4_VERIFY_CONV_ONE_PASS=1 \
  SGLANG_OPT_DRAFT_TOP1_BF16_LOGITS=1 \
  SGLANG_OPT_CONV_WINDOW_SCATTER_WIDE_BLOCKS=1 \
  SGLANG_OPT_MOE_FINALIZE_GATE_CUDA=1 \
  SGLANG_MAMBA_PATH_CAP_GRID_MIN_FREE=0.05 \
  SGLANG_ENABLE_HC_SUM_WS_MIX=1 \
  SGLANG_ENABLE_QSA_DECODE_TOPK_CUDA=1 \
  SGLANG_OPT_QSA_QK_NORM_ROWS=1 \
  SGLANG_OPT_GDN_PREFILL_ALPHA_GATES=1 \
  SGLANG_OPT_GDN_PREFILL_CONV_L2NORM=1 \
  SGLANG_OPT_GDN_PREFILL_SPARSE_CHECKPOINTS=1 \
  SGLANG_ENABLE_QSA_INDEX_SCORE_DECODE_SM100=1 \
  SGLANG_OPT_SPEC_FUSED_TOP_P_MASS=1 \
  SGLANG_QSA_SPARSE_DECODE_SM100_FP8_PV=1 \
  SGLANG_ONLINE_MXFP8_GDN_PROJ=1 \
  SGLANG_ONLINE_MXFP8_QSA_PROJ=1 \
  SGLANG_ONLINE_MXFP8_LM_HEAD=1 \
  python -m sglang.launch_server \
  --model-path nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 \
  --port 30000 \
  --attention-backend trtllm_mha \
  --context-length 262144 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 88 92 96 100 104 108 112 116 120 124 128 129 131 133 135 137 139 141 143 145 147 149 151 153 155 157 159 160 \
  --data-parallel-size 1 \
  --decode-log-interval 10 \
  --disable-shared-experts-fusion \
  --enable-cache-report \
  --enable-hierarchical-cache \
  --enable-linear-replayssm-spec \
  --enable-metrics \
  --expert-parallel-size 1 \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_back \
  --hybrid-cache-checkpoints-level 3 \
  --kv-cache-dtype fp8_e4m3 \
  --linear-attn-decode-backend flashinfer \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-verify-backend triton \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype bfloat16 \
  --mamba-track-interval 262144 \
  --moe-a2a-backend none \
  --moe-runner-backend flashinfer_trtllm \
  --page-size 64 \
  --pipeline-parallel-size 1 \
  --ple-offload-backend pinned \
  --ple-offload-embedding \
  --prefill-max-requests 6 \
  --qsa-indexer-dtype fp8_e4m3 \
  --quantization modelopt_mixed \
  --random-seed 35 \
  --reasoning-parser auto \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --scheduler-recv-interval 1 \
  --speculative-algorithm NEXTN \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --speculative-draft-model-revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --speculative-eagle-topk 1 \
  --speculative-moe-runner-backend flashinfer_trtllm \
  --speculative-num-draft-tokens 4 \
  --speculative-num-steps 3 \
  --stream-interval 20 \
  --strip-thinking-cache \
  --tensor-parallel-size 1 \
  --tokenizer-worker-num 1 \
  --trust-remote-code \
  --watchdog-timeout 1800 \
  --enable-prefill-delayer \
  --prefill-delayer-max-delay-ms 300 \
  --prefill-delayer-queue-min-ratio 0.1 \
  --mamba-max-states-per-path 2 \
  --max-running-requests 160 \
  --cuda-graph-max-bs-decode 160 \
  --hicache-mamba-size 220 \
  --hicache-size 690 \
  --max-mamba-cache-size 224 \
  --cuda-graph-max-bs-prefill 4096 \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768
```

For AgentX-style speed runs add `SGLANG_SIMULATE_ACC_LEN=2.32 SGLANG_SIMULATE_ACC_METHOD=match-expected
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`. Leave them out for accuracy runs and production.

## Our changes (review branches on this fork)

| Branch | Switch | What it does |
|---|---|---|
| `pr/qwen4-ple-prefetch-abort` | `SGLANG_FLASHINFER_AUTOTUNE_EXTEND` (existing) | fix a dirty PLE prefetch after an aborted forward |
| `pr/qsa-index-score-fp8` | `SGLANG_ENABLE_QSA_INDEX_SCORE_DECODE_CUDA` | CUDA decode index scoring for the FP8 indexer cache |
| `pr/qsa-decode-scoring` | `SGLANG_ENABLE_QSA_INDEX_SCORE_DECODE_SM100`, `SGLANG_ENABLE_QSA_DECODE_TOPK_CUDA` | exact tcgen05 decode scorer + exact top-512 |
| `pr/qsa-sparse-decode-sm100` | `SGLANG_ENABLE_QSA_SPARSE_DECODE_SM100` | tcgen05 sparse decode/verify attention |
| `pr/qsa-sparse-decode-fp8-pv` | `SGLANG_QSA_SPARSE_DECODE_SM100_FP8_PV` | block-scaled FP8 PV, split long lists |
| `pr/qsa-sparse-prefill-sm100` | `SGLANG_ENABLE_QSA_SPARSE_PREFILL_SM100` | tcgen05 sparse prefill attention |
| `pr/qsa-prefill-topk` | `SGLANG_ENABLE_QSA_PREFILL_TOPK_CUDA` | prefill scoring + exact top-512 |
| `pr/gdn-replayssm-verify-cuda` | `SGLANG_ENABLE_GDN_REPLAYSSM_VERIFY_CUDA` | tcgen05 GDN speculative verify |
| `pr/gdn-prefill-glue` | `SGLANG_OPT_GDN_FLASHINFER_PREFILL_FUSION` | GDN prefill conv/l2norm/gates/checkpoints |
| `pr/prefill-norm-retile` | `SGLANG_OPT_NORM_ROW_TILES` | GDN gated norm + QSA q/k norm tiles |
| `pr/verify-path-glue` | `SGLANG_OPT_SPEC_VERIFY_FUSION` | four bit-identical decode fusions |
| `pr/mamba-path-cap-grid` | `SGLANG_MAMBA_PATH_CAP_GRID` | one state per 16k span under the path cap |
| `pr/hc-sum-ws-mix` | `SGLANG_ENABLE_HC_SUM_WS_MIX` | warp-specialized hyper-connection up + mix |
| `pr/moe-finalize-gate-cuda` | `SGLANG_OPT_MOE_FINALIZE_GATE_CUDA` | shared-expert gate folded into gate_up |
| `pr/online-mxfp8-dense` | `SGLANG_ENABLE_ONLINE_MXFP8_DENSE` | online MXFP8 GDN/QSA projections + LM heads |

The integration tree runs the pre-bundle switch names (`SGLANG_OPT_GATED_NORM_WIDE_ROWS`, `SGLANG_OPT_GDN_PREFILL_*`,
`SGLANG_OPT_QSA_QK_NORM_ROWS`, the four verify-path switches, `SGLANG_ONLINE_MXFP8_*`); the launch command above is exact.

## Configuration findings (existing switches)

| Setting | Effect on AgentX |
|---|---|
| `--mamba-max-states-per-path 2` + `SGLANG_MAMBA_PATH_CAP_COUNTS_HOST=1` | +8.9%: bounded state checkpoints per conversation |
| `--max-mamba-cache-size 224 --hicache-mamba-size 220 --hicache-size 690` | GPU state slots sized to use; host caches hold the agents' history |
| `--mem-fraction-static 0.92 --cuda-graph-max-bs-prefill 4096`, 124 users | +3.3%: bigger KV pool, bigger decode batches |
| `--chunked-prefill-size 32768` | +0.7%: half the fixed cost per prefill chunk |
| post-verify CUDA graphs off | lazy capture during serving stalls the server (-1.2% when on) |
| FlashInfer 0.7.5 + `SGLANG_ENABLE_FLASHINFER_HC_SUM=1` | +2-3% |

## Open pull requests merged here

46631, 48296, 46632, 50538, 50622, 47867, 50954, 51051-51062, 51066-51068 (see the commit history for each merge);
sglang#51063 is not included because upstream `main` now fences write-back copies on the forward stream.

Nothing in this branch is taken from TokenSpeed; it is used only as a benchmark.
