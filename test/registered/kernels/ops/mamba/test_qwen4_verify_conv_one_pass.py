"""Qwen4 PLE verify short conv: the one-pass kernel against the three-kernel path."""

import unittest

import torch

from sglang.kernels.ops.mamba.qwen4_short_conv import (
    _qwen4_verify_conv_one_pass_kernel,
    fused_qwen4_verify_conv,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


def _inputs(
    requests, channels, slots, kernel, dilation, width, seed, state_dtype, cache_dtype
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state_len = (kernel - 1) * dilation
    tokens = requests * width

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", generator=generator).to(dtype)

    x = randn(tokens, channels)
    residual = randn(tokens, channels)
    weight = randn(channels, 1, kernel) * 0.5
    state = randn(slots, channels, state_len, dtype=state_dtype)
    indices = torch.randperm(slots, device="cuda", generator=generator)[:requests]
    valid = torch.rand(tokens, device="cuda", generator=generator) < 0.8
    intermediate = randn(slots, width, channels, state_len, dtype=cache_dtype)
    return x, residual, weight, state, indices, valid, width, dilation, intermediate


@requires_cuda
class TestQwen4VerifyConvOnePass(CustomTestCase):
    def _check(self, requests, channels, state_dtype, cache_dtype, with_cache=True):
        args = list(
            _inputs(
                requests, channels, 160, 4, 3, 4, requests, state_dtype, cache_dtype
            )
        )
        if not with_cache:
            args[-1] = None
        ref_args = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
        expected = fused_qwen4_verify_conv(*ref_args)
        with envs.SGLANG_OPT_SPEC_VERIFY_FUSION.override(True):
            actual = fused_qwen4_verify_conv(*args)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )
        if with_cache:
            self.assertTrue(torch.equal(args[-1], ref_args[-1]))
        self.assertTrue(torch.equal(args[3], ref_args[3]))

    def test_serving_shapes_bitwise(self):
        for requests in (1, 7, 64, 104):
            self._check(requests, 10240, torch.bfloat16, torch.bfloat16)

    def test_fp32_state_and_cache_bitwise(self):
        self._check(33, 10240, torch.float32, torch.float32)

    def test_ragged_channels_and_no_cache(self):
        self._check(5, 1000, torch.bfloat16, torch.bfloat16)
        self._check(5, 10240, torch.bfloat16, torch.bfloat16, with_cache=False)

    def test_matches_float64_reference(self):
        x, residual, weight, state, indices, valid, width, dilation, cache = _inputs(
            9, 512, 16, 4, 3, 4, 5, torch.bfloat16, torch.bfloat16
        )
        with envs.SGLANG_OPT_SPEC_VERIFY_FUSION.override(True):
            actual = fused_qwen4_verify_conv(
                x, residual, weight, state, indices, valid, width, dilation, cache
            )
        seq = torch.cat(
            [state[indices].double(), x.view(9, width, 512).transpose(1, 2).double()],
            -1,
        )
        conv = torch.nn.functional.conv1d(
            seq, weight.double(), dilation=dilation, groups=512
        )
        conv = conv.transpose(1, 2).reshape(-1, 512)
        ref = (conv * torch.sigmoid(conv) + residual.double()) * valid[:, None]
        self.assertLess((actual.double() - ref).abs().max().item(), 0.1)
        self.assertIsNotNone(_qwen4_verify_conv_one_pass_kernel)


if __name__ == "__main__":
    unittest.main()
