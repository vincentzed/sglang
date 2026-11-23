import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSiluAndMulSigmoidGate(unittest.TestCase):
    def reference(self, gate_up, inter_dim):
        g = gate_up[:, :inter_dim].float()
        u = gate_up[:, inter_dim : 2 * inter_dim].float()
        z = gate_up[:, 2 * inter_dim : 2 * inter_dim + 1].float()
        a = (torch.nn.functional.silu(g) * u).bfloat16().float()
        return (a * torch.sigmoid(z)).bfloat16()

    def test_matches_reference_and_graph_replay(self):
        from sglang.kernels.ops.moe.moe_shared_gate_act import (
            silu_and_mul_sigmoid_gate,
        )

        inter_dim = 640
        for rows in (1, 7, 416, 4096, 16384):
            with self.subTest(rows=rows):
                torch.manual_seed(rows)
                gate_up = torch.randn(
                    rows, 2 * inter_dim + 16, device="cuda", dtype=torch.bfloat16
                )
                gate_up[:, 2 * inter_dim] *= 4
                expected = self.reference(gate_up, inter_dim).float()
                actual = silu_and_mul_sigmoid_gate(gate_up, inter_dim)
                self.assertEqual(actual.shape, (rows, inter_dim))
                diff = (actual.float() - expected).abs()
                ulp = expected.abs().clamp_min(2**-126) * 2**-7
                self.assertTrue(bool((diff <= ulp).all()))
                self.assertGreater((diff == 0).float().mean().item(), 0.99)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = silu_and_mul_sigmoid_gate(gate_up, inter_dim)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual, replayed))

    def test_empty(self):
        from sglang.kernels.ops.moe.moe_shared_gate_act import (
            silu_and_mul_sigmoid_gate,
        )

        gate_up = torch.empty(0, 1296, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(silu_and_mul_sigmoid_gate(gate_up, 640).shape, (0, 640))


if __name__ == "__main__":
    unittest.main()
