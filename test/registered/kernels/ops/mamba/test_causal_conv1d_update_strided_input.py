"""Verify-mode causal_conv1d_update on q|k|v read in place from the in_proj output."""

import unittest

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


@requires_cuda
class TestCausalConv1dUpdateStridedInput(CustomTestCase):
    def _run(self, batch, dim, extra):
        generator = torch.Generator(device="cuda").manual_seed(batch)
        draft, width, slots = 4, 4, 200

        def randn(*shape):
            return torch.randn(shape, device="cuda", generator=generator).to(
                torch.bfloat16
            )

        projected = randn(batch * draft, dim + extra)
        weight = randn(dim, width)
        bias = None
        state = randn(slots, dim, width - 1)
        indices = torch.randperm(slots, device="cuda", generator=generator)[
            :batch
        ].int()
        phys = torch.zeros(
            (slots + 1, dim, draft + width - 2), device="cuda", dtype=torch.bfloat16
        )
        inter_idx = torch.randperm(slots, device="cuda", generator=generator)[
            :batch
        ].int()

        def run(x2d, conv_state, win_phys):
            win = win_phys.as_strided(
                (slots + 1, draft, dim, width - 1),
                (win_phys.stride(0), 1, win_phys.stride(1), 1),
            )
            x = x2d.view(batch, draft, -1).transpose(1, 2)
            out = causal_conv1d_update(
                x,
                conv_state,
                weight,
                bias,
                "silu",
                conv_state_indices=indices,
                intermediate_conv_window=win,
                intermediate_state_indices=inter_idx,
            )
            return out.transpose(1, 2).view(batch * draft, -1)

        ref_state, ref_phys = state.clone(), phys.clone()
        expected = run(projected[:, :dim].contiguous(), ref_state, ref_phys)
        actual = run(projected[:, :dim], state, phys)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(state, ref_state))
        self.assertTrue(torch.equal(phys, ref_phys))

    def test_gdn_verify_shapes(self):
        for batch in (1, 64, 104):
            self._run(batch, 10240, 6240)

    def test_small_dim(self):
        self._run(3, 384, 40)


if __name__ == "__main__":
    unittest.main()
