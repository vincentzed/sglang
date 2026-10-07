"""DeepSeek-V4.1 bounded replay state a prefill CUDA graph captures, padded to fixed row counts."""

import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _padded_tail(token_indices, *, graph_rows: int, num_tokens: int) -> LateLayerTail:
    padding = graph_rows - len(token_indices)
    return LateLayerTail(
        token_indices=torch.tensor(token_indices + [num_tokens] * padding),
        positions=torch.zeros(graph_rows, dtype=torch.int64),
        extend_seq_lens=torch.tensor([len(token_indices)], dtype=torch.int32),
        extend_seq_lens_cpu=[len(token_indices)],
        swa_out_cache_loc=torch.tensor(token_indices + [0] * padding),
        spare_row=num_tokens,
    )


class TestGraphPaddedLateLayerTail(CustomTestCase):
    def test_padding_rows_never_overwrite_a_tail_row(self):
        """The tail ends on the extend's last row, the row padding reads. Scattering
        the padded tail back must still return every tail row, not a padding row's."""
        num_tokens = 8
        tail = _padded_tail([5, 6, 7], graph_rows=6, num_tokens=num_tokens)
        full = torch.arange(num_tokens, dtype=torch.float32)[:, None].repeat(1, 2)

        rows = tail.rows(full)
        self.assertEqual(rows.shape[0], 6)
        # A late layer changes padding rows too; they must not reach the output.
        rows = rows.clone()
        rows[3:] = -1.0
        scattered = tail.scatter(rows, num_tokens)

        self.assertEqual(scattered.shape[0], num_tokens)
        self.assertEqual(scattered[5:, 0].tolist(), [5.0, 6.0, 7.0])

    def test_replay_refresh_keeps_the_captured_index_tensors(self):
        """The graph reads the tail's row indices, positions and write slots by
        address, so a replay must refill those tensors, not rebind them."""
        captured = _padded_tail([1, 2, 3], graph_rows=4, num_tokens=4)
        live = _padded_tail([2, 3], graph_rows=4, num_tokens=4)

        refreshed = captured.refresh_for_breakable_cuda_graph_replay_(live)

        self.assertIs(refreshed.token_indices, captured.token_indices)
        self.assertIs(refreshed.positions, captured.positions)
        self.assertEqual(refreshed.token_indices.tolist(), [2, 3, 4, 4])
        self.assertIs(refreshed.swa_out_cache_loc, captured.swa_out_cache_loc)
        self.assertEqual(refreshed.swa_out_cache_loc.tolist(), [2, 3, 0, 0])
        self.assertEqual(refreshed.extend_seq_lens_cpu, [2])


class TestGraphPaddedWindowLayout(CustomTestCase):
    def test_padding_rows_leave_every_request_window_unchanged(self):
        """Two requests padded to a graph bucket and to more window groups than
        requests: the live rows keep their unpadded layout shifted by the extra
        history rows, and padding rows read nothing and commit nowhere."""
        req = torch.tensor([3, 3, 3, 7, 7])
        pos = torch.tensor([200, 201, 202, 5, 6])
        window, live, groups, padded = 4, 5, 3, 8
        plain = window_layout(req, pos, window=window, capacity=8, num_groups=2)
        graph = window_layout(
            req,
            pos,
            window=window,
            capacity=8,
            num_groups=groups,
            padded_rows=padded,
        )

        extra_history = (groups - 2) * window
        self.assertEqual(graph.size, groups * window + padded)
        self.assertEqual(
            graph.write_loc[:live].tolist(), (plain.write_loc + extra_history).tolist()
        )
        # History slots keep their place; this batch's own rows move past the padding groups.
        shifted = torch.where(
            plain.indices >= 2 * window, plain.indices + extra_history, plain.indices
        )
        self.assertEqual(graph.indices[:live].tolist(), shifted.tolist())
        self.assertEqual(graph.commit_mask[:live].tolist(), plain.commit_mask.tolist())
        self.assertEqual(
            graph.history_valid[: 2 * window].tolist(), plain.history_valid.tolist()
        )
        self.assertFalse(graph.history_valid[2 * window :].any())

        self.assertFalse(graph.commit_mask[live:].any())
        self.assertEqual(graph.lengths[live:].tolist(), [0] * (padded - live))
        self.assertEqual(len(set(graph.write_loc.tolist())), padded)


if __name__ == "__main__":
    unittest.main()
