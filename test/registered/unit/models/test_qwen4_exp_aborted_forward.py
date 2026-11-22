"""A Qwen4-exp forward that raises leaves no PLE prefetch pending."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import Qwen4ExpModel, Qwen4ExpPLELayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FailingLayer:
    ple = None

    @contextlib.contextmanager
    def hc_norms(self, prepared_input, next_reader):
        yield SimpleNamespace(output=None)

    def __call__(self, **kwargs):
        raise RuntimeError("layer failed")


class TestQwen4ExpAbortedForward(CustomTestCase):
    def test_a_forward_that_raises_drops_the_next_layers_prefetch(self):
        ple = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
        torch.nn.Module.__init__(ple)
        ple._prefetch_state = None

        def start_prefetch(batch, forward_batch):
            ple._prefetch_state = "pending"

        ple.start_prefetch = start_prefetch
        model = Qwen4ExpModel.__new__(Qwen4ExpModel)
        torch.nn.Module.__init__(model)
        model.layers = [_FailingLayer(), SimpleNamespace(ple=ple)]
        model._start_layer, model._end_layer = 0, 2
        model.has_ple = True
        model.ple_ngram_size = model.ple_ngram_eos_token_id = None
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True)
        )
        with (
            mock.patch.object(qwen4_exp_module.residual_batch, "start"),
            mock.patch.object(
                qwen4_exp_module, "_shards_rows_over_cp", return_value=False
            ),
            mock.patch.object(
                qwen4_exp_module, "_prepare_ple_batch", return_value=None
            ),
            self.assertRaisesRegex(RuntimeError, "layer failed"),
        ):
            model.forward(
                input_ids=torch.zeros(1, dtype=torch.long),
                positions=None,
                forward_batch=forward_batch,
                inputs_embeds=torch.zeros(1, 1),
            )
        self.assertIsNone(ple._prefetch_state)


if __name__ == "__main__":
    unittest.main()
