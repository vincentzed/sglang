"""Greedy top-1 draft steps read the LM head's BF16 logits instead of an FP32 copy."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


def _copy(logits, mode, keep):
    processor = SimpleNamespace(
        vocab_size=logits.shape[1], keep_decode_logits_dtype=keep
    )
    metadata = SimpleNamespace(next_token_logits_buffer=None, forward_mode=mode)
    return LogitsProcessor._copy_logits_to_buffer(processor, logits, metadata)


@requires_cuda
class TestDraftTop1Bf16Logits(CustomTestCase):
    def test_decode_keeps_bf16_only_when_asked(self):
        logits = torch.randn(4, 1000, device="cuda").to(torch.bfloat16)
        self.assertIs(_copy(logits, ForwardMode.DECODE, True), logits)
        self.assertEqual(_copy(logits, ForwardMode.DECODE, False).dtype, torch.float32)
        self.assertEqual(
            _copy(logits, ForwardMode.DRAFT_EXTEND_V2, True).dtype, torch.float32
        )

    def test_argmax_matches_fp32_copy(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        for bs in (1, 64, 104):
            logits = (
                torch.randn(bs, 248320, device="cuda", generator=generator) * 4
            ).to(torch.bfloat16)
            logits[:, 7] = logits.max()
            results = []
            for value in (logits, logits.float()):
                positions = torch.zeros(bs, device="cuda", dtype=torch.int64)
                tokens = torch.zeros(bs, 3, device="cuda", dtype=torch.int64)
                _, index = draft_topk1_postprocess(value, positions, tokens, 1)
                results.append((index, tokens, positions))
            for a, b in zip(*results):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(
                torch.equal(results[0][0].flatten(), logits.float().argmax(-1))
            )


if __name__ == "__main__":
    unittest.main()
