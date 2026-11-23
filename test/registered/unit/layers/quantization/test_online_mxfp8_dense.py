import unittest

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import mxfp8_lm_head_matmul
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_QUANTIZED = (
    "model.language_model.layers.3.linear_attn.in_proj_qkvz",
    "model.language_model.layers.7.self_attn.qkv_proj",
    "mtp.layers.0.self_attn.qkv_proj",
)
_KEPT_BF16 = (
    "model.language_model.layers.3.linear_attn.out_proj",
    "model.language_model.layers.3.linear_attn.in_proj_ba",
    "model.language_model.layers.7.self_attn.o_proj",
    "model.language_model.layers.7.self_attn.indexer.index_qk_proj",
    "model.language_model.layers.7.mlp.shared_expert.gate_up_proj",
)


def _is_sm100() -> bool:
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


@unittest.skipIf(not is_cuda(), "CUDA-only")
class TestOnlineMxfp8Dense(unittest.TestCase):
    def test_switch_quantizes_only_the_input_projections(self):
        for prefix in _QUANTIZED + _KEPT_BF16:
            self.assertIsNone(modelopt_quant._online_mxfp8_dense_method(prefix))
        with envs.SGLANG_ENABLE_ONLINE_MXFP8_DENSE.override(True):
            for prefix in _QUANTIZED:
                method = modelopt_quant._online_mxfp8_dense_method(prefix)
                self.assertIsInstance(method, Fp8LinearMethod)
                self.assertTrue(method.use_mxfp8)
            for prefix in _KEPT_BF16:
                self.assertIsNone(modelopt_quant._online_mxfp8_dense_method(prefix))

    @unittest.skipIf(not _is_sm100(), "FlashInfer MXFP8 GEMM needs SM100")
    def test_lm_head_matches_fp32_and_keeps_one_weight_copy(self):
        torch.manual_seed(0)
        lm_head = nn.Linear(2560, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
        nn.init.normal_(lm_head.weight, std=0.02)
        for rows in (1, 64, 256):
            hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            logits = mxfp8_lm_head_matmul(hidden, lm_head)
            cache = lm_head.weight._mxfp8_weight_cache
            ref = hidden.float() @ lm_head.weight.float().T
            self.assertEqual(logits.dtype, torch.bfloat16)
            self.assertLess(((logits.float() - ref).norm() / ref.norm()).item(), 0.06)
            again = mxfp8_lm_head_matmul(hidden, lm_head)
            self.assertIs(lm_head.weight._mxfp8_weight_cache, cache)
            torch.testing.assert_close(again, logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
