import importlib.util
import pathlib
import sys
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import conv_window_dedup_enabled
from sglang.srt.speculative.dflash_tree_utils import (
    build_ancestor_matrix_from_parents,
    build_batched_retrieve_links_from_parents,
    build_retrieve_links_from_parents,
    build_tree_custom_mask,
    build_tree_from_topk_cpu,
    compute_tree_budget,
    tree_accept_greedy,
    tree_accept_greedy_batched,
    top2gap_fanout_caps,
)
from sglang.srt.server_args import _merge_speculative_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_reference_dflash_tree():
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    ref_path = (
        repo_root
        / "jetspec"
        / "_ref"
        / "vllm-jetspec"
        / "vllm"
        / "v1"
        / "spec_decode"
        / "dflash_tree.py"
    )
    if not ref_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("ref_dflash_tree", ref_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDFlashTreeConstruction(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = _load_reference_dflash_tree()

    def _assert_matches_reference(
        self,
        *,
        root_token,
        topk_tokens,
        topk_logprobs,
        budget,
    ):
        tokens = torch.tensor(topk_tokens, dtype=torch.long)
        logprobs = torch.tensor(topk_logprobs, dtype=torch.float32)

        if self.ref is None:
            self.skipTest("JetSpec reference clone is not present under jetspec/_ref.")

        ours = build_tree_from_topk_cpu(
            root_token=root_token,
            topk_tokens=tokens,
            topk_logprobs=logprobs,
            budget=budget,
            depth_first=False,
        )
        ref = self.ref.build_tree_from_topk(
            root_token=root_token,
            topk_tokens=tokens,
            topk_logprobs=logprobs,
            budget=budget,
            device=torch.device("cpu"),
            depth_first=False,
        )

        self.assertEqual(ours.token_ids, ref.token_ids.tolist())
        self.assertEqual(ours.parent_indices, ref.parent_indices.tolist())
        self.assertEqual(ours.depths, ref.depth.tolist())
        self.assertEqual(ours.num_nodes, ref.num_nodes)

    def test_budget_rule(self):
        self.assertEqual(compute_tree_budget(16, 1), 16)
        self.assertEqual(compute_tree_budget(4, 2), 15)
        self.assertEqual(compute_tree_budget(16, 7, max_budget=128), 128)
        self.assertEqual(compute_tree_budget(4, 3, max_budget=0), 40)

    def test_one_wide_degenerate_chain(self):
        self._assert_matches_reference(
            root_token=7,
            topk_tokens=[[10], [11], [12]],
            topk_logprobs=[[-0.1], [-0.2], [-0.3]],
            budget=4,
        )

    def test_full_binary_tree_budget(self):
        self._assert_matches_reference(
            root_token=3,
            topk_tokens=[[10, 20], [11, 21]],
            topk_logprobs=[[-0.1, -1.0], [-0.2, -0.7]],
            budget=5,
        )

    def test_limited_budget_expands_best_cumulative_prefixes(self):
        self._assert_matches_reference(
            root_token=1,
            topk_tokens=[[10, 20, 30], [11, 21, 31], [12, 22, 32]],
            topk_logprobs=[
                [-0.10, -0.20, -0.30],
                [-0.05, -0.50, -0.70],
                [-0.01, -0.40, -1.00],
            ],
            budget=7,
        )

    def test_ties_are_deterministic_by_insertion_order(self):
        self._assert_matches_reference(
            root_token=5,
            topk_tokens=[[101, 102], [201, 202], [301, 302]],
            topk_logprobs=[
                [-0.5, -0.5],
                [-0.25, -0.25],
                [-0.125, -0.125],
            ],
            budget=6,
        )

    def test_budget_can_stop_before_full_depth(self):
        self._assert_matches_reference(
            root_token=9,
            topk_tokens=[[10, 20], [11, 21], [12, 22], [13, 23]],
            topk_logprobs=[
                [-0.1, -0.4],
                [-0.1, -0.4],
                [-0.1, -0.4],
                [-0.1, -0.4],
            ],
            budget=4,
        )

    def test_explicit_fanout_caps_are_forwarded(self):
        tree = build_tree_from_topk_cpu(
            root_token=1,
            topk_tokens=torch.tensor(
                [
                    [10, 20, 30, 40],
                    [11, 21, 31, 41],
                    [12, 22, 32, 42],
                ]
            ),
            topk_logprobs=torch.tensor(
                [
                    [-0.1, -0.2, -0.3, -0.4],
                    [-0.1, -0.2, -0.3, -0.4],
                    [-0.1, -0.2, -0.3, -0.4],
                ]
            ),
            budget=8,
            depth_first=False,
            fanout_caps=[1, 4, 4],
        )

        root_children = [
            idx for idx, parent in enumerate(tree.parent_indices) if parent == 0
        ]
        self.assertEqual(root_children, [1])

    def test_top2gap_fanout_caps_use_top2_gap_sigmoid(self):
        caps = top2gap_fanout_caps(
            [
                [-0.10, -0.10, -2.0, -3.0],
                [-0.10, -1.10, -2.0, -3.0],
                [-0.10, -5.10, -6.0, -7.0],
            ],
            beta=1.0,
            g_0=1.0,
        )

        self.assertEqual(caps, [3, 2, 1])

    def test_top2gap_tree_uses_accum_logp_with_adaptive_fanout_caps(self):
        tree = build_tree_from_topk_cpu(
            root_token=1,
            topk_tokens=torch.tensor(
                [
                    [10, 20, 30, 40],
                    [11, 21, 31, 41],
                    [12, 22, 32, 42],
                ]
            ),
            topk_logprobs=torch.tensor(
                [
                    [-0.10, -5.10, -6.0, -7.0],
                    [-0.10, -5.10, -6.0, -7.0],
                    [-0.10, -5.10, -6.0, -7.0],
                ]
            ),
            budget=8,
            depth_first=False,
            score_mode="top2gap",
            top2gap_beta=1.0,
            top2gap_g_0=1.0,
        )

        self.assertEqual(
            [idx for idx, parent in enumerate(tree.parent_indices) if parent == 0],
            [1],
        )
        self.assertEqual(tree.num_nodes, 4)

    def test_tree_mask_and_retrieve_links_from_irregular_bfs_tree(self):
        tree = build_tree_from_topk_cpu(
            root_token=1,
            topk_tokens=torch.tensor([[10, 20], [11, 21], [12, 22]]),
            topk_logprobs=torch.tensor([[-0.1, -0.2], [-0.1, -0.6], [-0.1, -0.6]]),
            budget=6,
            depth_first=False,
        )
        parents = torch.tensor(tree.parent_indices, dtype=torch.long)
        ancestor = build_ancestor_matrix_from_parents(parents)
        for idx, parent in enumerate(tree.parent_indices[1:], start=1):
            self.assertTrue(bool(ancestor[idx, idx]))
            self.assertTrue(bool(ancestor[idx, parent]))
            self.assertTrue(bool(ancestor[idx, 0]))

        retrieve_index, retrieve_next_token, retrieve_next_sibling = (
            build_retrieve_links_from_parents(parents)
        )
        self.assertEqual(retrieve_index.tolist(), list(range(tree.num_nodes)))
        children = [[] for _ in range(tree.num_nodes)]
        for idx, parent in enumerate(tree.parent_indices[1:], start=1):
            children[parent].append(idx)
        for parent, child_list in enumerate(children):
            expected_first = child_list[0] if child_list else -1
            self.assertEqual(int(retrieve_next_token[parent]), expected_first)
            for child, sibling in zip(child_list, child_list[1:]):
                self.assertEqual(int(retrieve_next_sibling[child]), sibling)

    def test_retrieve_links_ignore_padded_budget_slots(self):
        parents = torch.tensor([-1, 0, 0, 0, 0], dtype=torch.long)
        retrieve_index, retrieve_next_token, retrieve_next_sibling = (
            build_retrieve_links_from_parents(
                parents,
                num_verify_tokens=5,
                num_nodes=3,
            )
        )
        self.assertEqual(retrieve_index.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(retrieve_next_token.tolist(), [1, -1, -1, -1, -1])
        self.assertEqual(retrieve_next_sibling.tolist(), [-1, 2, -1, -1, -1])

    def test_retrieve_index_uses_global_flat_offsets(self):
        parents = torch.tensor([-1, 0, 1], dtype=torch.long)
        retrieve_index, retrieve_next_token, retrieve_next_sibling = (
            build_retrieve_links_from_parents(
                parents,
                num_verify_tokens=5,
                row_offset=64,
            )
        )
        self.assertEqual(retrieve_index.tolist(), [64, 65, 66, 67, 68])
        self.assertEqual(retrieve_next_token.tolist(), [1, 2, -1, -1, -1])
        self.assertEqual(retrieve_next_sibling.tolist(), [-1, -1, -1, -1, -1])

    def test_retrieve_links_preserve_duplicate_token_siblings_for_state(self):
        parents = torch.tensor([-1, 0, 0, 1, 2], dtype=torch.long)
        tokens = torch.tensor([4, 5, 5, 7, 8], dtype=torch.long)
        _, retrieve_next_token, retrieve_next_sibling = build_retrieve_links_from_parents(
            parents,
            token_ids=tokens,
        )
        self.assertEqual(retrieve_next_token.tolist(), [1, 3, 4, -1, -1])
        self.assertEqual(retrieve_next_sibling.tolist(), [-1, 2, -1, -1, -1])

    def test_batched_retrieve_links_match_single_row_helper(self):
        parent_rows = torch.tensor(
            [
                [-1, 0, 0, 1, 1, 2, 0],
                [-1, 0, 1, 2, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        num_real_nodes = torch.tensor([6, 4], dtype=torch.long)

        retrieve_index, retrieve_next_token, retrieve_next_sibling = (
            build_batched_retrieve_links_from_parents(
                parent_rows,
                num_verify_tokens=7,
                num_real_nodes=num_real_nodes,
            )
        )

        self.assertEqual(
            retrieve_index.tolist(),
            [
                [0, 1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12, 13],
            ],
        )
        for row in range(parent_rows.shape[0]):
            _, expected_next_token, expected_next_sibling = (
                build_retrieve_links_from_parents(
                    parent_rows[row],
                    num_verify_tokens=7,
                    num_nodes=int(num_real_nodes[row]),
                )
            )
            self.assertEqual(
                retrieve_next_token[row].tolist(), expected_next_token.tolist()
            )
            self.assertEqual(
                retrieve_next_sibling[row].tolist(), expected_next_sibling.tolist()
            )

    def test_batched_retrieve_links_can_prefer_later_siblings(self):
        parent_rows = torch.tensor([[-1, 0, 0, 0, 1, 2]], dtype=torch.long)

        _, retrieve_next_token, retrieve_next_sibling = (
            build_batched_retrieve_links_from_parents(
                parent_rows,
                num_verify_tokens=6,
                num_real_nodes=torch.tensor([6], dtype=torch.long),
                prefer_later_sibling=True,
            )
        )

        self.assertEqual(retrieve_next_token.tolist(), [[3, 4, 5, -1, -1, -1]])
        self.assertEqual(retrieve_next_sibling.tolist(), [[-1, -1, 1, 2, -1, -1]])

    def test_tree_custom_mask_prepends_prefix(self):
        parents = torch.tensor([-1, 0, 0, 1], dtype=torch.long)
        mask = build_tree_custom_mask(
            parent_indices_batch=[parents],
            num_real_nodes=[4],
            prefix_lens=torch.tensor([3], dtype=torch.int64),
            num_verify_tokens=4,
            device="cpu",
        ).view(4, 7)
        self.assertTrue(torch.all(mask[:, :3]))
        self.assertEqual(
            mask[:, 3:].to(torch.int32).tolist(),
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
            ],
        )

    def test_tree_accept_greedy_returns_path_and_bonus(self):
        # Tree: 0(root=4) -> 1(token=5) -> 3(token=7), and root -> 2(token=6).
        tree_tokens = torch.tensor([4, 5, 6, 7], dtype=torch.long)
        parents = torch.tensor([-1, 0, 0, 1], dtype=torch.long)
        target_tokens = torch.tensor([5, 7, 99, 42], dtype=torch.long)
        path, correct_drafts, bonus = tree_accept_greedy(
            tree_tokens, parents, target_tokens
        )
        self.assertEqual(path, [0, 1, 3])
        self.assertEqual(correct_drafts, 2)
        self.assertEqual(bonus, 42)

    def test_tree_accept_greedy_matches_reference_duplicate_child_rule(self):
        # Duplicate token 5 appears under the root. JetSpec's child map keeps
        # the later duplicate, so accepting token 5 must walk through node 2.
        tree_tokens = torch.tensor([4, 5, 5, 7, 8], dtype=torch.long)
        parents = torch.tensor([-1, 0, 0, 1, 2], dtype=torch.long)
        target_tokens = torch.tensor([5, 7, 8, 99, 42], dtype=torch.long)
        path, correct_drafts, bonus = tree_accept_greedy(
            tree_tokens, parents, target_tokens
        )
        self.assertEqual(path, [0, 2, 4])
        self.assertEqual(correct_drafts, 2)
        self.assertEqual(bonus, 42)

    def test_tree_accept_greedy_batched_packs_reference_path(self):
        tree_tokens = torch.tensor([[4, 5, 5, 7, 8, 0]], dtype=torch.long)
        parents = torch.tensor([[-1, 0, 0, 1, 2, 0]], dtype=torch.long)
        depths = torch.tensor([[0, 1, 1, 2, 2, 1]], dtype=torch.long)
        target_tokens = torch.tensor([[5, 7, 8, 99, 42, 0]], dtype=torch.long)

        predict, accept_index, correct_drafts = tree_accept_greedy_batched(
            tree_tokens=tree_tokens,
            parent_indices=parents,
            depths=depths,
            target_tokens=target_tokens,
            num_real_nodes=torch.tensor([5], dtype=torch.long),
            max_tree_depth=3,
        )

        self.assertEqual(predict.tolist(), [5, 7, 8, 99, 42, 0])
        self.assertEqual(accept_index.tolist(), [[0, 2, 4, -1]])
        self.assertEqual(correct_drafts.tolist(), [2])

    def test_tree_accept_greedy_batched_ignores_padded_duplicate(self):
        tree_tokens = torch.tensor([[4, 5, 5, 5]], dtype=torch.long)
        parents = torch.tensor([[-1, 0, 0, 0]], dtype=torch.long)
        depths = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
        target_tokens = torch.tensor([[5, 11, 22, 33]], dtype=torch.long)

        _, accept_index, correct_drafts = tree_accept_greedy_batched(
            tree_tokens=tree_tokens,
            parent_indices=parents,
            depths=depths,
            target_tokens=target_tokens,
            num_real_nodes=torch.tensor([3], dtype=torch.long),
            max_tree_depth=2,
        )

        self.assertEqual(accept_index.tolist(), [[0, 2, -1]])
        self.assertEqual(correct_drafts.tolist(), [1])

    def test_speculative_config_json_maps_dflash_tree_keys(self):
        kwargs = {
            "speculative_config": (
                '{"dflash": {"tree_width": 7, "tree_budget": 128, '
                '"tree_draft": "top2gap", "top2gap_beta": 2.0, '
                '"top2gap_g_0": 0.5, "head_type": "causal"}}'
            )
        }
        _merge_speculative_config(kwargs)
        self.assertEqual(kwargs["speculative_dflash_tree_width"], 7)
        self.assertEqual(kwargs["speculative_dflash_tree_budget"], 128)
        self.assertEqual(kwargs["speculative_dflash_tree_draft"], "top2gap")
        self.assertEqual(kwargs["speculative_dflash_top2gap_beta"], 2.0)
        self.assertEqual(kwargs["speculative_dflash_top2gap_g0"], 0.5)
        self.assertEqual(kwargs["speculative_dflash_head_type"], "causal")

    def test_dflash_tree_uses_dense_mamba_intermediate_conv_windows(self):
        self.assertTrue(
            conv_window_dedup_enabled(
                is_npu=False,
                is_cpu=False,
                speculative_eagle_topk=1,
                speculative_dflash_tree_width=1,
            )
        )
        self.assertFalse(
            conv_window_dedup_enabled(
                is_npu=False,
                is_cpu=False,
                speculative_eagle_topk=1,
                speculative_dflash_tree_width=4,
            )
        )

    def test_speculative_config_rejects_unknown_key(self):
        with self.assertRaisesRegex(ValueError, "Unsupported --speculative-config"):
            _merge_speculative_config(
                {"speculative_config": '{"dflash": {"unknown": 1}}'}
            )


if __name__ == "__main__":
    unittest.main()
