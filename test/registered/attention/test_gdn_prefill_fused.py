import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention import fused_qk_rmsnorm_rope_gate as qk_norm
from sglang.kernels.ops.attention.fla.l2norm import gdn_prefill_qkv_prepare_fwd
from sglang.kernels.ops.attention.linear.fused_gdn_gating import (
    fused_gdn_alpha_beta,
    fused_gdn_gating,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.kernels.ops.mamba.gdn_prefill_conv import (
    GDN_PREFILL_CONV_BLOCK_M,
    gdn_prefill_conv_qkv,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

HEAD_DIM = 128


def _conv_reference(x, weight, states, starts, cache_indices, has_initial_state):
    """FP32 causal depthwise conv + SiLU of each sequence of token-major x."""
    width = weight.shape[1]
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i + 1]
        if has_initial_state[i]:
            prior = states[cache_indices[i]].transpose(0, 1).float()
        else:
            prior = x.new_zeros(width - 1, x.shape[1], dtype=torch.float32)
        seq = torch.cat([prior, x[a:b].float()])
        acc = sum(weight[:, j].float() * seq[j : j + b - a] for j in range(width))
        out[a:b] = acc * torch.sigmoid(acc)
    return out


@unittest.skipIf(not is_cuda(), "CUDA-only kernels")
class TestGdnPrefillFused(unittest.TestCase):
    def _conv_case(self, seq_lens, use_block_table, token_major_states):
        qk_dim, v_dim, width = 2 * HEAD_DIM, 4 * HEAD_DIM, 4
        dim, batch_size = 2 * qk_dim + v_dim, len(seq_lens)
        lengths = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
        starts = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)]).int()
        total = sum(seq_lens)
        qkv = torch.randn(total, dim + v_dim, device="cuda", dtype=torch.bfloat16)[
            :, :dim
        ]
        weight = torch.randn(dim, width, device="cuda", dtype=qkv.dtype)
        cache_indices = torch.randperm(batch_size + 2, device="cuda")[:batch_size]
        cache_indices = cache_indices.int()
        has_initial_state = torch.arange(batch_size, device="cuda") % 2 == 0
        states = torch.randn(
            batch_size + 2, width - 1, dim, device="cuda", dtype=qkv.dtype
        ).transpose(1, 2)
        if token_major_states:
            states = states.contiguous()
        initial_states = states.clone()
        expected_states = states.clone(memory_format=torch.preserve_format)
        q, k, v = causal_conv1d_fn(
            qkv.transpose(0, 1),
            weight,
            None,
            expected_states,
            starts,
            seq_lens,
            cache_indices,
            has_initial_state,
            activation="silu",
            output_split_size=qk_dim,
            output_split_tail=v_dim,
        )
        expected = gdn_prefill_qkv_prepare_fwd(
            q.view(total, -1, HEAD_DIM),
            k.view(total, -1, HEAD_DIM),
            v.view(total, -1, HEAD_DIM),
            joint_qk=True,
        )
        block_table = None
        if use_block_table:
            pairs = [
                (seq, block)
                for seq, n in enumerate(seq_lens)
                for block in range(
                    (n + GDN_PREFILL_CONV_BLOCK_M - 1) // GDN_PREFILL_CONV_BLOCK_M
                )
            ]
            block_table = torch.tensor(pairs, device="cuda", dtype=torch.int32)
        planes = gdn_prefill_conv_qkv(
            qkv,
            weight,
            None,
            states,
            starts,
            cache_indices,
            has_initial_state,
            qk_dim,
            seq_lens_cpu=seq_lens,
            block_table=block_table,
        )
        exact = _conv_reference(
            qkv,
            weight,
            initial_states,
            starts.tolist(),
            cache_indices,
            has_initial_state,
        )
        exact = (
            exact[..., :qk_dim],
            exact[..., qk_dim : 2 * qk_dim],
            exact[..., 2 * qk_dim :],
        )
        for i, (plane, reference, fp32) in enumerate(zip(planes, expected, exact)):
            self.assertTrue(plane.is_contiguous())
            plane = plane.view_as(reference).float()
            fp32 = fp32.reshape(reference.shape)
            if i < 2:
                fp32 = F.normalize(fp32, dim=-1)
            else:
                # FP32 taps: v is the bf16 rounding of the FP32 conv.
                bound = fp32.abs() * 2**-8 + 1e-5
                self.assertTrue(bool(((plane - fp32).abs() <= bound).all()))
            # About as close to the FP32 conv as the stock bf16-product planes
            # (q and k round twice, around the l2norm, in both).
            self.assertLessEqual(
                (plane - fp32).abs().max().item(),
                1.25 * (reference.float() - fp32).abs().max().item() + 1e-4,
            )
        torch.testing.assert_close(states, expected_states, rtol=0, atol=0)

    def test_conv_l2norm_matches_split_conv_and_joint_l2norm(self):
        torch.manual_seed(0)
        for seq_lens in ([65, 1, 7, 300, 2], [1] * 33 + [129], [2049]):
            for use_block_table in (False, True):
                for token_major_states in (False, True):
                    with self.subTest(
                        seq_lens=seq_lens[:4],
                        block_table=use_block_table,
                        token_major_states=token_major_states,
                    ):
                        self._conv_case(seq_lens, use_block_table, token_major_states)

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
