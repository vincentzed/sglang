"""Accepted conv-window scatter: 8192-element programs."""

import unittest

import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


def _dedup_window(layers, spec_slots, draft, dim, win, generator):
    phys = torch.randn(
        (layers, spec_slots + 1, dim, draft + win - 1),
        device="cuda",
        generator=generator,
    ).to(torch.bfloat16)
    view = phys.as_strided(
        (layers, spec_slots + 1, draft, dim, win),
        (phys.stride(0), phys.stride(1), 1, phys.stride(2), 1),
    )
    return phys, view


@requires_cuda
class TestConvWindowScatterWideBlocks(CustomTestCase):
    def _run(self, requests, dim, invalid_fraction, with_src):
        generator = torch.Generator(device="cuda").manual_seed(requests + dim)
        layers, slots, draft, win = 6, 300, 4, 3
        _, src = _dedup_window(layers, requests, draft, dim, win, generator)
        dst = torch.randn(
            (layers, slots, dim, win), device="cuda", generator=generator
        ).to(torch.bfloat16)
        dst_idx = torch.randperm(slots, device="cuda", generator=generator)[:requests]
        steps = torch.randint(0, draft, (requests,), device="cuda", generator=generator)
        steps[
            torch.rand(requests, device="cuda", generator=generator) < invalid_fraction
        ] = -1
        src_idx = (
            torch.randperm(requests, device="cuda", generator=generator)
            if with_src
            else None
        )
        expected = dst.clone()
        fused_conv_window_scatter_with_mask(expected, src, dst_idx, steps, src_idx)
        actual = dst.clone()
        with envs.SGLANG_OPT_CONV_WINDOW_SCATTER_WIDE_BLOCKS.override(True):
            fused_conv_window_scatter_with_mask(actual, src, dst_idx, steps, src_idx)
        self.assertTrue(torch.equal(actual, expected))
        rows = steps >= 0
        source = torch.arange(requests, device="cuda") if src_idx is None else src_idx
        ref = src[:, source[rows], steps[rows]]
        self.assertTrue(torch.equal(actual[:, dst_idx[rows]], ref))

    def test_serving_dim(self):
        self._run(104, 10240, 0.0, True)
        self._run(64, 10240, 0.0, False)

    def test_track_set_mostly_invalid(self):
        self._run(104, 10240, 0.9, True)

    def test_ragged_dim(self):
        self._run(9, 1000, 0.3, True)


if __name__ == "__main__":
    unittest.main()
