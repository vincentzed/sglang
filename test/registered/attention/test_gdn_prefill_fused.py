import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import fused_qk_rmsnorm_rope_gate as qk_norm
from sglang.kernels.ops.attention.linear.fused_gdn_gating import (
    fused_gdn_alpha_beta,
    fused_gdn_gating,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


@unittest.skipIf(not is_cuda(), "CUDA-only kernels")
class TestGdnPrefillFused(unittest.TestCase):
    def test_alpha_beta_match_gating_and_exp(self):
        torch.manual_seed(0)
        heads = 48
        A_log = torch.randn(heads, device="cuda")
        dt_bias = torch.randn(heads, device="cuda", dtype=torch.bfloat16)
        for tokens in (1, 63, 1000):
            ba = torch.randn(tokens, 2 * heads, device="cuda", dtype=torch.bfloat16)
            b, a = ba[:, :heads], ba[:, heads:]
            g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
            alpha, beta_out = fused_gdn_alpha_beta(A_log, a, b, dt_bias)
            torch.testing.assert_close(alpha, torch.exp(g[0]), rtol=0, atol=0)
            torch.testing.assert_close(beta_out, beta[0], rtol=0, atol=0)

    def test_qk_norm_rows_match_per_head_programs(self):
        torch.manual_seed(0)
        num_q, num_kv, head_dim, rotary_dim = 24, 2, 256, 64
        cache = torch.rand(4096, rotary_dim, device="cuda") * 2 - 1
        q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
        k_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
        axis_map = (torch.arange(rotary_dim // 2, device="cuda") % 3).int()
        for tokens in (5, 300):
            qkv = torch.randn(
                tokens,
                num_q * 2 * head_dim + 2 * num_kv * head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            q_gate = qkv[:, : num_q * 2 * head_dim]
            k = qkv[:, num_q * 2 * head_dim : (num_q * 2 + num_kv) * head_dim]
            pos = torch.randint(0, 4096, (tokens,), device="cuda")
            for positions, amap in ((pos, None), (pos.repeat(3, 1), axis_map)):
                args = (q_gate, k, q_weight, k_weight, cache, positions, 1e-6)
                kwargs = dict(has_gate=True, mrope_axis_map=amap)
                dims = (num_q, num_kv, head_dim, rotary_dim)
                with patch.object(qk_norm, "_qk_norm_rows", False):
                    expected = qk_norm.fused_qk_gemma_rmsnorm_rope_gate(
                        *args, *dims, **kwargs
                    )
                with patch.object(qk_norm, "_qk_norm_rows", True):
                    actual = qk_norm.fused_qk_gemma_rmsnorm_rope_gate(
                        *args, *dims, **kwargs
                    )
                # Row sums reduce in another order: at most one bf16 ulp apart.
                for out, ref in zip(actual, expected):
                    ulp = ref.float().abs().clamp_min(1e-30) * 2**-7
                    self.assertTrue(
                        bool(((out.float() - ref.float()).abs() <= ulp).all())
                    )
                torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
