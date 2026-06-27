from __future__ import annotations

import heapq
from typing import Iterable, Optional

import msgspec
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


class DraftTree(msgspec.Struct):
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


class DraftTreeCPU(msgspec.Struct):
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
    num_nodes: Optional[int] = None,
    row_offset: int = 0,
    token_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = parent_indices.device
    total_nodes = int(parent_indices.numel()) if num_nodes is None else int(num_nodes)
    parents = parent_indices[:total_nodes].to(dtype=torch.long).tolist()
    width = total_nodes if num_verify_tokens is None else int(num_verify_tokens)
    if width < total_nodes:
        raise ValueError(
            f"num_verify_tokens={width} is smaller than num_nodes={total_nodes}."
        )

    retrieve_index = torch.arange(
        int(row_offset), int(row_offset) + width, dtype=torch.long, device=device
    )
    retrieve_next_token = torch.full((width,), -1, dtype=torch.long, device=device)
    retrieve_next_sibling = torch.full((width,), -1, dtype=torch.long, device=device)

    children: list[list[int]] = [[] for _ in range(width)]
    for idx in range(1, total_nodes):
        children[parents[idx]].append(idx)
    for idx, node_children in enumerate(children[:total_nodes]):
        if not node_children:
            continue
        # These links drive tree-state kernels (conv/GDN/Mamba), so they must
        # encode the physical parent/child topology for every node.  JetSpec's
        # duplicate-token overwrite rule belongs only to acceptance; pruning
        # duplicate-token siblings here would leave the pruned nodes with an
        # incorrect parent state during verify.
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

    child_maps: list[dict[int, int]] = [dict() for _ in range(n)]
    for idx in range(1, n):
        parent = parents[idx]
        if 0 <= parent < n:
            child_maps[parent][int(tokens[idx])] = idx

    path = [0]
    current = 0
    while True:
        child = child_maps[current].get(int(targets[current]))
        if child is None:
            break
        path.append(child)
        current = child

    bonus_token = int(targets[current])
    return path, len(path) - 1, bonus_token


def tree_accept_greedy_batched(
    *,
    tree_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    target_tokens: torch.Tensor,
    num_real_nodes: torch.Tensor,
    max_tree_depth: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched JetSpec greedy tree acceptance.

    This mirrors ``jetspec.tree._core.accept.gpu_tree_accept`` for fixed-width
    padded batches.  Acceptance is a child-map walk: at node ``n``, accept the
    unique child whose draft token equals ``target_tokens[n]``.  If siblings
    under the same parent have the same token, the later node overwrites the
    earlier one, matching the Python ``dict`` child map used by JetSpec.

    Returns:
        predicts: flat target argmax tokens, used by the spec-v2 publish path.
        accept_index: global flat node indices for the root-inclusive accepted
            path, padded with -1 to ``max_tree_depth + 1``.
        num_correct_drafts: accepted draft-token count, excluding the trailing
            correction token emitted from the last accepted node.
    """

    if tree_tokens.ndim != 2:
        raise ValueError(
            f"tree_tokens must be 2D [bs, num_nodes], got {tuple(tree_tokens.shape)}."
        )
    if parent_indices.shape != tree_tokens.shape:
        raise ValueError(
            "parent_indices shape must match tree_tokens shape, "
            f"got {tuple(parent_indices.shape)} vs {tuple(tree_tokens.shape)}."
        )
    if depths.shape != tree_tokens.shape:
        raise ValueError(
            f"depths shape must match tree_tokens shape, got {tuple(depths.shape)}."
        )
    if target_tokens.shape != tree_tokens.shape:
        raise ValueError(
            "target_tokens shape must match tree_tokens shape, "
            f"got {tuple(target_tokens.shape)} vs {tuple(tree_tokens.shape)}."
        )

    bs, num_nodes = tree_tokens.shape
    device = tree_tokens.device
    path_width = int(max_tree_depth) + 1
    if path_width <= 0:
        raise ValueError(f"max_tree_depth must be non-negative, got {max_tree_depth}.")

    if bs == 0:
        return (
            target_tokens.reshape(-1).to(torch.int32),
            torch.empty((0, path_width), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
        )
    if num_nodes <= 0:
        raise ValueError("tree_tokens must contain at least the root node.")

    real_nodes = num_real_nodes.to(device=device, dtype=torch.long)
    if real_nodes.ndim != 1 or int(real_nodes.shape[0]) != bs:
        raise ValueError(
            "num_real_nodes must be 1D with one entry per batch row, "
            f"got {tuple(num_real_nodes.shape)} for bs={bs}."
        )

    node_ids = torch.arange(num_nodes, dtype=torch.long, device=device)
    valid_node = node_ids.unsqueeze(0) < real_nodes.unsqueeze(1)
    valid_node[:, 0] = real_nodes > 0

    parents = parent_indices.to(torch.long)
    safe_parents = parents.clamp(min=0, max=num_nodes - 1)
    valid_parent = (parents >= 0) & (parents < real_nodes.unsqueeze(1))

    same_parent = parents[:, :, None] == parents[:, None, :]
    same_token = tree_tokens[:, :, None] == tree_tokens[:, None, :]
    later_node = node_ids[None, None, :] > node_ids[None, :, None]
    later_valid = valid_node[:, None, :]
    overwritten = (same_parent & same_token & later_node & later_valid).any(dim=2)

    parent_targets = torch.gather(target_tokens, 1, safe_parents)
    match = (
        valid_node
        & valid_parent
        & ~overwritten
        & (tree_tokens == parent_targets)
    )
    match[:, 0] = valid_node[:, 0]

    prefix_match = match.clone()
    jump = safe_parents.clone()
    for _ in range(max(1, int(max_tree_depth).bit_length())):
        prefix_match = prefix_match & torch.gather(prefix_match, 1, jump)
        jump = torch.gather(jump, 1, jump)

    score = torch.where(
        prefix_match,
        depths.to(torch.long),
        torch.full_like(depths.to(torch.long), -1),
    )
    best_node = torch.argmax(score, dim=1)
    accepted_depth = torch.gather(depths.to(torch.long), 1, best_node[:, None]).squeeze(
        1
    )
    accepted_depth = accepted_depth.clamp(min=0, max=int(max_tree_depth))

    path_buf = torch.empty((bs, path_width), dtype=torch.long, device=device)
    current = best_node
    for depth in range(path_width - 1, -1, -1):
        path_buf[:, depth] = current
        current = torch.gather(safe_parents, 1, current[:, None]).squeeze(1)

    path_pos = torch.arange(path_width, dtype=torch.long, device=device)
    valid_path_pos = path_pos.unsqueeze(0) <= accepted_depth.unsqueeze(1)
    valid_start = (path_width - 1) - accepted_depth
    gather_pos = valid_start.unsqueeze(1) + path_pos.unsqueeze(0)
    local_accept = torch.gather(path_buf, 1, gather_pos.clamp(max=path_width - 1))
    row_offsets = (
        torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1) * num_nodes
    )
    accept_index = torch.where(
        valid_path_pos,
        row_offsets + local_accept,
        torch.full_like(local_accept, -1),
    ).to(torch.int32)

    return (
        target_tokens.reshape(-1).to(torch.int32),
        accept_index,
        accepted_depth.to(torch.int32),
    )


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
