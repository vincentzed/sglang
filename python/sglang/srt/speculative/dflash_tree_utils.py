from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


def compute_per_depth_entropy(logits: torch.Tensor) -> list[float]:
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return ent.tolist()


def _node_expand_score(
    cum_logprob: float,
    depth: int,
    score_mode: str,
    per_depth_entropy: Optional[list[float]],
    hybrid_alpha: float,
    parent_expansion_entropy: Optional[float] = None,
) -> float:
    if score_mode == "entropy":
        if parent_expansion_entropy is not None:
            return parent_expansion_entropy
        if per_depth_entropy is not None and depth < len(per_depth_entropy):
            return per_depth_entropy[depth]
        return 0.0
    if score_mode == "hybrid":
        if parent_expansion_entropy is not None:
            ent = parent_expansion_entropy
        elif per_depth_entropy is not None and depth < len(per_depth_entropy):
            ent = per_depth_entropy[depth]
        else:
            ent = 0.0
        return cum_logprob + hybrid_alpha * ent
    return cum_logprob


@dataclass
class DraftTree:
    # Root-inclusive canonical node order.
    token_ids: torch.Tensor
    parent_indices: torch.Tensor
    depths: torch.Tensor
    num_nodes: int

    def paths(self) -> list[list[int]]:
        return _paths_from_parents(self.parent_indices.tolist(), self.num_nodes)

    def longest_path(self) -> list[int]:
        return _longest_path_from_parents(
            self.parent_indices.tolist(), self.depths.tolist(), self.num_nodes
        )

    def to_cpu(self) -> DraftTreeCPU:
        return DraftTreeCPU(
            token_ids=self.token_ids.tolist(),
            parent_indices=self.parent_indices.tolist(),
            depths=self.depths.tolist(),
            num_nodes=self.num_nodes,
        )


@dataclass
class DraftTreeCPU:
    # Root-inclusive canonical node order.
    token_ids: list[int]
    parent_indices: list[int]
    depths: list[int]
    num_nodes: int

    def __post_init__(self) -> None:
        if not (
            len(self.token_ids)
            == len(self.parent_indices)
            == len(self.depths)
            == self.num_nodes
        ):
            raise ValueError(
                "DraftTreeCPU fields must all have length num_nodes. "
                f"tokens={len(self.token_ids)}, parents={len(self.parent_indices)}, "
                f"depths={len(self.depths)}, num_nodes={self.num_nodes}."
            )
        if self.num_nodes <= 0:
            raise ValueError("DraftTreeCPU must contain at least the root node.")
        if self.parent_indices[0] != -1 or self.depths[0] != 0:
            raise ValueError("DraftTreeCPU root must have parent -1 and depth 0.")
        for idx in range(1, self.num_nodes):
            parent = self.parent_indices[idx]
            if parent < 0 or parent >= idx:
                raise ValueError(
                    "DraftTreeCPU parent indices must point to an earlier node. "
                    f"idx={idx}, parent={parent}."
                )
            if self.depths[idx] != self.depths[parent] + 1:
                raise ValueError(
                    "DraftTreeCPU depth must be parent depth + 1. "
                    f"idx={idx}, depth={self.depths[idx]}, parent={parent}, "
                    f"parent_depth={self.depths[parent]}."
                )

    def paths(self) -> list[list[int]]:
        return _paths_from_parents(self.parent_indices, self.num_nodes)

    def longest_path(self) -> list[int]:
        return _longest_path_from_parents(
            self.parent_indices, self.depths, self.num_nodes
        )

    def to_gpu(self, device: torch.device | str) -> DraftTree:
        return DraftTree(
            token_ids=torch.tensor(self.token_ids, dtype=torch.long, device=device),
            parent_indices=torch.tensor(
                self.parent_indices, dtype=torch.long, device=device
            ),
            depths=torch.tensor(self.depths, dtype=torch.long, device=device),
            num_nodes=self.num_nodes,
        )


def _paths_from_parents(parents: list[int], num_nodes: int) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    for idx in range(1, num_nodes):
        children[parents[idx]].append(idx)
    leaves = [idx for idx in range(num_nodes) if not children[idx]]
    paths: list[list[int]] = []
    for leaf in leaves:
        path: list[int] = []
        node = leaf
        while node >= 0:
            path.append(node)
            node = parents[node]
        paths.append(path[::-1])
    return paths


def _longest_path_from_parents(
    parents: list[int], depths: list[int], num_nodes: int
) -> list[int]:
    if num_nodes == 1:
        return [0]
    child_count = [0] * num_nodes
    for idx in range(1, num_nodes):
        child_count[parents[idx]] += 1

    best_leaf = 0
    best_depth = -1
    for idx in range(num_nodes):
        if child_count[idx] == 0 and depths[idx] > best_depth:
            best_leaf = idx
            best_depth = depths[idx]

    path: list[int] = []
    node = best_leaf
    while node >= 0:
        path.append(node)
        node = parents[node]
    return path[::-1]


def compute_tree_budget(
    block_size: int,
    tree_width: int,
    max_budget: Optional[int] = None,
) -> int:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if tree_width <= 1:
        return int(block_size)
    full_tree = (int(tree_width) ** int(block_size) - 1) // (int(tree_width) - 1)
    if max_budget is not None and int(max_budget) > 0:
        return min(full_tree, int(max_budget))
    return full_tree


def sample_topk_from_logits(
    logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.topk(log_probs, int(k), dim=-1)


def batch_topk_to_cpu(logits: torch.Tensor, k: int) -> tuple[list, list, list]:
    log_probs = torch.log_softmax(logits, dim=-1)
    topk_logprobs, topk_tokens = torch.topk(log_probs, int(k), dim=-1)
    return log_probs.tolist(), topk_logprobs.tolist(), topk_tokens.tolist()


def _build_tree_breadth_first(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    score_mode: str = "accum_logp",
    per_depth_entropy: Optional[list[float]] = None,
    hybrid_alpha: float = 1.0,
    fanout_caps: Optional[list[int]] = None,
) -> DraftTreeCPU:
    depth_count = len(topk_tokens_cpu)
    width = len(topk_tokens_cpu[0]) if depth_count else 0

    tokens = [int(root_token)]
    parents = [-1]
    depths = [0]
    cum_logprob_at = [0.0]
    num_nodes = 1

    root_score = _node_expand_score(0.0, 0, score_mode, per_depth_entropy, hybrid_alpha)
    counter = 0
    heap: list[tuple[float, int, int]] = [(-root_score, counter, 0)]
    while heap and num_nodes < int(budget):
        _, _, node_idx = heapq.heappop(heap)
        depth = depths[node_idx]
        if depth >= depth_count:
            continue

        fanout_cap = fanout_caps[depth] if fanout_caps is not None else width
        children_to_add = min(fanout_cap, width, int(budget) - num_nodes)
        row_tokens = topk_tokens_cpu[depth]
        row_logprobs = topk_logprobs_cpu[depth]
        expansion_entropy = (
            per_depth_entropy[depth]
            if per_depth_entropy is not None and depth < len(per_depth_entropy)
            else None
        )
        for child_idx in range(children_to_add):
            tokens.append(int(row_tokens[child_idx]))
            child_cum_logprob = cum_logprob_at[node_idx] + float(
                row_logprobs[child_idx]
            )
            parents.append(node_idx)
            child_depth = depth + 1
            depths.append(child_depth)
            cum_logprob_at.append(child_cum_logprob)
            score = _node_expand_score(
                child_cum_logprob,
                child_depth,
                score_mode,
                per_depth_entropy,
                hybrid_alpha,
                parent_expansion_entropy=expansion_entropy,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, num_nodes))
            num_nodes += 1

    return DraftTreeCPU(
        token_ids=tokens,
        parent_indices=parents,
        depths=depths,
        num_nodes=num_nodes,
    )


def _build_tree_depth_first(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    score_mode: str = "accum_logp",
    per_depth_entropy: Optional[list[float]] = None,
    hybrid_alpha: float = 1.0,
    fanout_caps: Optional[list[int]] = None,
) -> DraftTreeCPU:
    depth_count = len(topk_tokens_cpu)
    width = len(topk_tokens_cpu[0]) if depth_count else 0

    tokens = [int(root_token)]
    parents = [-1]
    depths = [0]
    num_nodes = 1

    spine_set: set[int] = {0}
    prev_idx = 0
    for depth in range(depth_count):
        if num_nodes >= int(budget):
            break
        tokens.append(int(topk_tokens_cpu[depth][0]))
        parents.append(prev_idx)
        depths.append(depth + 1)
        spine_set.add(num_nodes)
        prev_idx = num_nodes
        num_nodes += 1

    counter = 0
    heap: list[tuple[float, int, int]] = []
    cum_logprob_at = [0.0] * num_nodes
    for idx in range(num_nodes):
        depth = depths[idx]
        if depth > 0:
            cum_logprob_at[idx] = (
                cum_logprob_at[parents[idx]] + topk_logprobs_cpu[depth - 1][0]
            )
        if depth < depth_count:
            score = _node_expand_score(
                cum_logprob_at[idx],
                depth,
                score_mode,
                per_depth_entropy,
                hybrid_alpha,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, idx))

    while heap and num_nodes < int(budget):
        _, _, node_idx = heapq.heappop(heap)
        depth = depths[node_idx]
        if depth >= depth_count:
            continue
        start_child = 1 if node_idx in spine_set else 0
        fanout_cap = fanout_caps[depth] if fanout_caps is not None else width
        children_to_add = min(
            max(fanout_cap - start_child, 0),
            width - start_child,
            int(budget) - num_nodes,
        )
        if children_to_add <= 0:
            continue

        expansion_entropy = (
            per_depth_entropy[depth]
            if per_depth_entropy is not None and depth < len(per_depth_entropy)
            else None
        )
        for child_idx in range(start_child, start_child + children_to_add):
            tokens.append(int(topk_tokens_cpu[depth][child_idx]))
            child_cum_logprob = cum_logprob_at[node_idx] + float(
                topk_logprobs_cpu[depth][child_idx]
            )
            parents.append(node_idx)
            child_depth = depth + 1
            depths.append(child_depth)
            cum_logprob_at.append(child_cum_logprob)
            score = _node_expand_score(
                child_cum_logprob,
                child_depth,
                score_mode,
                per_depth_entropy,
                hybrid_alpha,
                parent_expansion_entropy=expansion_entropy,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, num_nodes))
            num_nodes += 1

    return DraftTreeCPU(
        token_ids=tokens,
        parent_indices=parents,
        depths=depths,
        num_nodes=num_nodes,
    )


def _build_tree_cpu(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    *,
    depth_first: bool = False,
    score_mode: str = "accum_logp",
    per_depth_entropy: Optional[list[float]] = None,
    hybrid_alpha: float = 1.0,
) -> DraftTreeCPU:
    if score_mode not in ("accum_logp", "entropy", "hybrid"):
        raise ValueError(
            "DFlash tree v1 supports score_mode in "
            f"('accum_logp', 'entropy', 'hybrid'), got {score_mode!r}."
        )
    if len(topk_tokens_cpu) != len(topk_logprobs_cpu):
        raise ValueError(
            "topk_tokens and topk_logprobs must have the same depth count."
        )
    if len(topk_tokens_cpu) == 0:
        return DraftTreeCPU(
            token_ids=[int(root_token)],
            parent_indices=[-1],
            depths=[0],
            num_nodes=1,
        )
    width = len(topk_tokens_cpu[0])
    if width <= 0:
        raise ValueError("topk rows must be non-empty.")
    for depth, (tokens, logprobs) in enumerate(zip(topk_tokens_cpu, topk_logprobs_cpu)):
        if len(tokens) != width or len(logprobs) != width:
            raise ValueError(
                "DFlash tree top-k rows must have a fixed width. "
                f"depth={depth}, expected={width}, got tokens={len(tokens)}, "
                f"logprobs={len(logprobs)}."
            )

    builder = _build_tree_depth_first if depth_first else _build_tree_breadth_first
    return builder(
        root_token,
        topk_tokens_cpu,
        topk_logprobs_cpu,
        int(budget),
        score_mode=score_mode,
        per_depth_entropy=per_depth_entropy,
        hybrid_alpha=hybrid_alpha,
    )


def build_tree_from_topk_cpu(
    root_token: int,
    topk_tokens: torch.Tensor | list[list[int]],
    topk_logprobs: torch.Tensor | list[list[float]],
    budget: int,
    *,
    depth_first: bool = False,
    score_mode: str = "accum_logp",
    per_depth_entropy: Optional[list[float]] = None,
    hybrid_alpha: float = 1.0,
) -> DraftTreeCPU:
    tokens_cpu = (
        topk_tokens.tolist() if isinstance(topk_tokens, torch.Tensor) else topk_tokens
    )
    logprobs_cpu = (
        topk_logprobs.tolist()
        if isinstance(topk_logprobs, torch.Tensor)
        else topk_logprobs
    )
    return _build_tree_cpu(
        int(root_token),
        tokens_cpu,
        logprobs_cpu,
        int(budget),
        depth_first=depth_first,
        score_mode=score_mode,
        per_depth_entropy=per_depth_entropy,
        hybrid_alpha=hybrid_alpha,
    )


def build_tree_from_topk(
    root_token: int,
    topk_tokens: torch.Tensor,
    topk_logprobs: torch.Tensor,
    budget: int,
    device: torch.device | str,
    *,
    depth_first: bool = False,
    score_mode: str = "accum_logp",
    per_depth_entropy: Optional[list[float]] = None,
    hybrid_alpha: float = 1.0,
) -> DraftTree:
    return build_tree_from_topk_cpu(
        root_token,
        topk_tokens,
        topk_logprobs,
        budget,
        depth_first=depth_first,
        score_mode=score_mode,
        per_depth_entropy=per_depth_entropy,
        hybrid_alpha=hybrid_alpha,
    ).to_gpu(device)


def build_ancestor_matrix_from_parents(
    parent_indices: torch.Tensor | Iterable[int],
    *,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    parents = (
        parent_indices.tolist()
        if isinstance(parent_indices, torch.Tensor)
        else list(parent_indices)
    )
    num_nodes = len(parents)
    if num_nodes <= 0:
        raise ValueError("parent_indices must contain at least the root node.")
    mask = torch.eye(num_nodes, dtype=torch.bool, device=device)
    mask[:, 0] = True
    for idx in range(1, num_nodes):
        parent = int(parents[idx])
        while parent > 0:
            mask[idx, parent] = True
            parent = int(parents[parent])
    return mask


def build_retrieve_links_from_parents(
    parent_indices: torch.Tensor,
    *,
    num_verify_tokens: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = parent_indices.device
    parents = parent_indices.to(dtype=torch.long).tolist()
    num_nodes = len(parents)
    width = num_nodes if num_verify_tokens is None else int(num_verify_tokens)
    if width < num_nodes:
        raise ValueError(
            f"num_verify_tokens={width} is smaller than num_nodes={num_nodes}."
        )

    retrieve_index = torch.arange(width, dtype=torch.long, device=device)
    retrieve_next_token = torch.full((width,), -1, dtype=torch.long, device=device)
    retrieve_next_sibling = torch.full((width,), -1, dtype=torch.long, device=device)

    children: list[list[int]] = [[] for _ in range(width)]
    for idx in range(1, num_nodes):
        children[parents[idx]].append(idx)
    for idx, node_children in enumerate(children[:num_nodes]):
        if not node_children:
            continue
        retrieve_next_token[idx] = node_children[0]
        for child, sibling in zip(node_children, node_children[1:]):
            retrieve_next_sibling[child] = sibling
    return retrieve_index, retrieve_next_token, retrieve_next_sibling


def pad_tree_to_budget(
    tree: DraftTreeCPU,
    *,
    num_verify_tokens: int,
    pad_token: int = 0,
) -> DraftTreeCPU:
    if tree.num_nodes > int(num_verify_tokens):
        raise ValueError(
            f"tree.num_nodes={tree.num_nodes} exceeds num_verify_tokens={num_verify_tokens}."
        )
    if tree.num_nodes == int(num_verify_tokens):
        return tree
    pad_count = int(num_verify_tokens) - tree.num_nodes
    return DraftTreeCPU(
        token_ids=tree.token_ids + [int(pad_token)] * pad_count,
        parent_indices=tree.parent_indices + [0] * pad_count,
        depths=tree.depths + [1] * pad_count,
        num_nodes=int(num_verify_tokens),
    )


def build_tree_custom_mask(
    *,
    parent_indices_batch: list[torch.Tensor],
    num_real_nodes: list[int],
    prefix_lens: torch.Tensor,
    num_verify_tokens: int,
    device: torch.device | str,
) -> torch.Tensor:
    if len(parent_indices_batch) != int(prefix_lens.numel()):
        raise ValueError(
            "parent_indices_batch length must match prefix_lens. "
            f"got {len(parent_indices_batch)} and {int(prefix_lens.numel())}."
        )
    prefix_lens_cpu = prefix_lens.to(device="cpu", dtype=torch.int64).tolist()
    pieces: list[torch.Tensor] = []
    for parents, real_nodes, prefix_len in zip(
        parent_indices_batch, num_real_nodes, prefix_lens_cpu
    ):
        prefix_mask = torch.ones(
            (int(num_verify_tokens), int(prefix_len)), dtype=torch.bool, device=device
        )
        tree_mask = torch.zeros(
            (int(num_verify_tokens), int(num_verify_tokens)),
            dtype=torch.bool,
            device=device,
        )
        real_tree_mask = build_ancestor_matrix_from_parents(
            parents[: int(real_nodes)], device=device
        )
        tree_mask[: int(real_nodes), : int(real_nodes)] = real_tree_mask
        if int(real_nodes) < int(num_verify_tokens):
            pad_slice = slice(int(real_nodes), int(num_verify_tokens))
            tree_mask[pad_slice, pad_slice] = torch.eye(
                int(num_verify_tokens) - int(real_nodes),
                dtype=torch.bool,
                device=device,
            )
        pieces.append(torch.cat((prefix_mask, tree_mask), dim=1).reshape(-1))
    if not pieces:
        return torch.empty((0,), dtype=torch.bool, device=device)
    return torch.cat(pieces, dim=0)


def tree_accept_greedy(
    tree_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    num_nodes: Optional[int] = None,
) -> tuple[list[int], int, int]:
    if tree_tokens.device != target_tokens.device:
        target_tokens = target_tokens.to(tree_tokens.device)
    n = int(num_nodes) if num_nodes is not None else int(tree_tokens.numel())
    tokens = tree_tokens[:n].tolist()
    parents = parent_indices[:n].to(dtype=torch.long).tolist()
    targets = target_tokens[:n].tolist()

    children: list[list[int]] = [[] for _ in range(n)]
    for idx in range(1, n):
        children[parents[idx]].append(idx)

    best_path = [0]
    best_len = 0
    stack: list[list[int]] = [[0]]
    while stack:
        path = stack.pop()
        node = path[-1]
        if children[node]:
            for child in reversed(children[node]):
                stack.append([*path, child])
            continue

        correct_drafts = 0
        for depth_idx in range(1, len(path)):
            parent = path[depth_idx - 1]
            child = path[depth_idx]
            if tokens[child] != targets[parent]:
                break
            correct_drafts += 1
        if correct_drafts > best_len:
            best_len = correct_drafts
            best_path = path[: correct_drafts + 1]

    bonus_token = int(targets[best_path[-1]])
    return best_path, best_len, bonus_token


def tree_signature(tree: DraftTreeCPU | DraftTree) -> str:
    if isinstance(tree, DraftTree):
        depths = tree.depths.tolist()
        parents = tree.parent_indices.tolist()
    else:
        depths = tree.depths
        parents = tree.parent_indices
    num_nodes = tree.num_nodes
    child_count = [0] * num_nodes
    for idx in range(1, num_nodes):
        child_count[parents[idx]] += 1
    depth_counts: dict[int, int] = {}
    num_leaves = 0
    for idx, depth in enumerate(depths):
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        if child_count[idx] == 0:
            num_leaves += 1
    max_depth = max(depths) if depths else 0
    dist = ",".join(f"{depth}:{count}" for depth, count in sorted(depth_counts.items()))
    return f"N={num_nodes} D={max_depth} L={num_leaves} depth=[{dist}]"


def find_closest_capture_size(num_nodes: int, capture_sizes: list[int]) -> int:
    import bisect

    if not capture_sizes:
        raise ValueError("capture_sizes must be non-empty.")
    idx = bisect.bisect_left(capture_sizes, int(num_nodes))
    if idx < len(capture_sizes):
        return capture_sizes[idx]
    return capture_sizes[-1]
