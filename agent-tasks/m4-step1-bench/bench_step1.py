"""M4 Step-1 settling experiment.

Region under test = one DSA layer's overlap window at eager prefill
(GLM-5.2 shapes, TP4 per-rank), bs=1:

  side A (main-attention Q prep, per rank):
      q_b_proj GEMM [M,2048]x[2048,4096] bf16
      -> split q_nope/q_pe -> absorbed bmm [16,M,192]x[16,192,512] bf16
  side B (whole indexer):
      wq_b GEMM [M,2048]x[2048,4096] bf16
      wk_weights_proj GEMM [M,6144]x[6144,160] bf16
      fused_q_indexer_rope_first_quant + fused_k_indexer_norm_rope + act_quant
      deep_gemm.fp8_mqa_logits  [M rows x N kv]  (row-chunked like the server)
      fast_topk_transform_ragged_fused (topk=2048)

Arms: serial | two_stream (one SM pool) | green_ctx A-slice sweep.
"""

import argparse
import json

import deep_gemm
import torch
import torch.nn.functional as F
from sgl_kernel import fast_topk_transform_ragged_fused

from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_first_quant
from sglang.jit_kernel.dsv32 import fused_k_indexer_norm_rope
from sglang.srt.layers.attention.dsa.triton_kernel import act_quant
from sglang.srt.multiplex.green_ctx import split_device_green_ctx_by_sm_count

PRESETS = {
    # per-rank shapes
    "glm52": dict(
        hidden=6144,
        q_lora=2048,
        heads_rank=16,
        qk_head=256,
        nope=192,
        kv_lora=512,
        idx_heads=32,
    ),  # TP4
    "dsv32": dict(
        hidden=7168,
        q_lora=1536,
        heads_rank=16,
        qk_head=192,
        nope=128,
        kv_lora=512,
        idx_heads=64,
    ),  # TP8
}
HIDDEN = 6144
Q_LORA = 2048
HEADS_RANK = 16  # 64 heads / TP4
QK_HEAD = 256  # 192 nope + 64 rope
NOPE = 192
KV_LORA = 512
IDX_HEADS = 32
IDX_DIM = 128
IDX_ROPE = 64
TOPK = 2048
MAX_POS = 131072


def apply_preset(name):
    g = globals()
    for k, v in PRESETS[name].items():
        g[k.upper()] = v


class Workload:
    def __init__(self, m: int, n: int, sub_chunk_rows: int):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(1234)

        def rand(*shape, dtype=torch.bfloat16, scale=1.0):
            return (
                torch.randn(*shape, generator=g, device=dev, dtype=torch.float32).to(
                    dtype
                )
                * scale
            )

        self.m, self.n = m, n
        self.sub_chunk_rows = sub_chunk_rows if sub_chunk_rows > 0 else m
        self.q_lora = rand(m, Q_LORA)
        self.hidden = rand(m, HIDDEN)
        self.w_qb = rand(HEADS_RANK * QK_HEAD, Q_LORA, scale=0.02)
        self.w_kc = rand(HEADS_RANK, NOPE, KV_LORA, scale=0.02)
        self.w_wqb = rand(IDX_HEADS * IDX_DIM, Q_LORA, scale=0.02)
        self.w_wkw = rand(IDX_DIM + IDX_HEADS, HIDDEN, scale=0.02)
        self.ln_w = torch.ones(IDX_DIM, device=dev, dtype=torch.float32)
        self.ln_b = torch.zeros(IDX_DIM, device=dev, dtype=torch.float32)

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, IDX_ROPE, 2, dtype=torch.float32) / IDX_ROPE)
        )
        freqs = torch.outer(torch.arange(MAX_POS, dtype=torch.float32), inv_freq)
        self.cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dev)

        self.positions = torch.arange(n - m, n, device=dev, dtype=torch.int64)
        self.ks = torch.zeros(m, device=dev, dtype=torch.int32)
        self.ke = torch.arange(n - m + 1, n + 1, device=dev, dtype=torch.int32)
        self.lengths = self.ke.clone()
        self.topk_offset = torch.zeros(m, device=dev, dtype=torch.int32)

        hist = rand(n, IDX_DIM)
        self.k_fp8_buf, k_scale = act_quant(hist, 128, None)
        self.k_scale_buf = k_scale.view(torch.float32).squeeze(-1).contiguous()

        self.q_scale_gate = IDX_DIM**-0.5 * IDX_HEADS**-0.5
        self.topk_out = torch.full((m, TOPK), -1, device=dev, dtype=torch.int32)

    def side_a(self):
        q = F.linear(self.q_lora, self.w_qb).view(self.m, HEADS_RANK, QK_HEAD)
        q_nope, q_pe = q.split([NOPE, QK_HEAD - NOPE], dim=-1)
        q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        return q_nope_out, q_pe

    def side_b_proj(self):
        qi = F.linear(self.q_lora, self.w_wqb).view(self.m, IDX_HEADS, IDX_DIM)
        kw = F.linear(self.hidden, self.w_wkw)
        key_raw, weights_raw = kw.split([IDX_DIM, IDX_HEADS], dim=-1)
        q_fp8, weights = fused_q_indexer_rope_first_quant(
            qi.contiguous(),
            weights_raw,
            self.q_scale_gate,
            self.cos_sin,
            self.positions,
        )
        key = fused_k_indexer_norm_rope(
            key_raw.contiguous(),
            self.ln_w,
            self.ln_b,
            1e-6,
            self.cos_sin,
            self.positions,
        )
        k_fp8, k_scale = act_quant(key, 128, None)
        self.k_fp8_buf[self.n - self.m :] = k_fp8
        self.k_scale_buf[self.n - self.m :] = k_scale.view(torch.float32).squeeze(-1)
        return q_fp8, weights.squeeze(-1)

    def side_b_logits_topk(self, q_fp8, weights):
        start = 0
        while start < self.m:
            end = min(start + self.sub_chunk_rows, self.m)
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[start:end],
                (self.k_fp8_buf, self.k_scale_buf),
                weights[start:end],
                self.ks[start:end],
                self.ke[start:end],
                clean_logits=False,
            )
            self.topk_out[start:end] = fast_topk_transform_ragged_fused(
                score=logits,
                lengths=self.lengths[start:end],
                topk_indices_offset=self.topk_offset[start:end],
                topk=TOPK,
                row_starts=self.ks[start:end],
            )
            start = end
        return self.topk_out

    def side_b(self):
        q_fp8, weights = self.side_b_proj()
        return self.side_b_logits_topk(q_fp8, weights)


def time_arm(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))
    times.sort()
    return times[len(times) // 2], times[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--sub-chunk-rows", type=int, default=0)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--a-slices", type=int, nargs="*", default=[16, 24, 32, 48])
    p.add_argument(
        "--deepgemm-slice-sms",
        action="store_true",
        help="configure deep_gemm num_sms to the indexer slice size in green arms",
    )
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--preset", choices=list(PRESETS), default="glm52")
    p.add_argument("--workqueue-scope", type=str, default=None)
    p.add_argument("--decompose", action="store_true")
    args = p.parse_args()
    apply_preset(args.preset)

    torch.cuda.set_device(0)
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    wl = Workload(args.m, args.n, args.sub_chunk_rows)
    default = torch.cuda.current_stream()

    results = {}

    def record(name, fn, check_against=None):
        med, best = time_arm(fn, args.iters, args.warmup)
        ok = ""
        if check_against is not None:
            # The fused topk kernel returns the selected set in nondeterministic
            # order run-to-run (pre-existing behavior); compare as sorted sets.
            ok = bool(torch.equal(wl.topk_out.sort(dim=1).values, check_against))
        results[name] = {"median_ms": med, "best_ms": best, "topk_identical": ok}
        print(f"{name:>28}: median {med:8.3f} ms   best {best:8.3f} ms   {ok}")

    record("side_a_only", wl.side_a)
    record("side_b_only", wl.side_b)
    ref_topk = wl.topk_out.sort(dim=1).values.clone()

    if args.decompose:
        record("side_b_proj_only", wl.side_b_proj)
        pq, pw = wl.side_b_proj()
        torch.cuda.synchronize()
        record("side_b_logits_topk_only", lambda: wl.side_b_logits_topk(pq, pw))

    def serial():
        wl.side_a()
        wl.side_b()

    record("serial", serial, ref_topk)

    plain = torch.cuda.Stream()

    def two_stream():
        plain.wait_stream(default)
        with torch.cuda.stream(plain):
            wl.side_a()
        wl.side_b()
        default.wait_stream(plain)

    record("two_stream", two_stream, ref_topk)

    import time

    t0 = time.perf_counter()
    split_device_green_ctx_by_sm_count(0, [8, total_sms - 8])
    print(f"green ctx split creation: {(time.perf_counter() - t0) * 1e3:.1f} ms")

    for a_sms in args.a_slices:
        b_sms = total_sms - a_sms
        (ga, gb), res = split_device_green_ctx_by_sm_count(
            0, [a_sms, b_sms], workqueue_scope=args.workqueue_scope
        )
        granted = [r.sm.smCount for r in res]

        def green():
            ga.wait_stream(default)
            gb.wait_stream(default)
            with torch.cuda.stream(ga):
                wl.side_a()
            with torch.cuda.stream(gb):
                if args.deepgemm_slice_sms:
                    old = deep_gemm.get_num_sms()
                    deep_gemm.set_num_sms(granted[1])
                    wl.side_b()
                    deep_gemm.set_num_sms(old)
                else:
                    wl.side_b()
            default.wait_stream(ga)
            default.wait_stream(gb)

        record(f"green_a{granted[0]}_b{granted[1]}", green, ref_topk)

    for a_sms in args.a_slices:
        b_sms = total_sms - a_sms
        (ga, gb), res = split_device_green_ctx_by_sm_count(0, [a_sms, b_sms])
        granted = [r.sm.smCount for r in res]

        def green_narrow():
            ga.wait_stream(default)
            gb.wait_stream(default)
            with torch.cuda.stream(ga):
                wl.side_a()
            with torch.cuda.stream(gb):
                out = wl.side_b_proj()
            default.wait_stream(gb)
            wl.side_b_logits_topk(*out)
            default.wait_stream(ga)

        record(f"green_narrow_a{granted[0]}", green_narrow, ref_topk)

    serial_ms = results["serial"]["median_ms"]
    print(
        f"\nM={args.m} N={args.n} sub_chunk={wl.sub_chunk_rows} total_sms={total_sms}"
    )
    for k, v in results.items():
        if k in ("side_a_only", "side_b_only"):
            continue
        print(
            f"{k:>28}: {v['median_ms']:8.3f} ms  speedup vs serial: {serial_ms / v['median_ms']:6.3f}x"
        )

    if args.json_out:
        meta = dict(m=args.m, n=args.n, sub_chunk=wl.sub_chunk_rows, results=results)
        with open(args.json_out, "w") as f:
            json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
