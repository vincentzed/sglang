from __future__ import annotations

from typing import Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api

try:
    import cutlass
    import cutlass.cute as cute
    import flash_attn.cute.utils as fa4_cute_utils
    from flash_attn.cute import flash_attn_varlen_func as _fa4_flash_attn_varlen_func
except Exception as _e:  # pragma: no cover
    cutlass = None
    cute = None
    fa4_cute_utils = None
    _fa4_flash_attn_varlen_func = None
    _fa4_import_error = _e
else:
    _fa4_import_error = None


if cute is not None:

    @cute.jit
    def _dflash_tree_mask_mod(
        batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        prefix_lens = aux_tensors[0]
        ancestor_mask = aux_tensors[1]
        tree_budget = aux_tensors[2][0]

        batch_idx = fa4_cute_utils.ssa_to_scalar(batch_idx)
        q_idx = fa4_cute_utils.ssa_to_scalar(q_idx)
        kv_idx = fa4_cute_utils.ssa_to_scalar(kv_idx)

        prefix_len = prefix_lens[batch_idx]
        tree_idx = kv_idx - prefix_len
        is_prefix = kv_idx < prefix_len
        is_tree = tree_idx >= 0 and tree_idx < tree_budget

        safe_tree_idx = 0 if tree_idx < 0 else tree_idx
        safe_tree_idx = tree_budget - 1 if safe_tree_idx >= tree_budget else safe_tree_idx
        mask_offset = (
            batch_idx * tree_budget * tree_budget + q_idx * tree_budget + safe_tree_idx
        )
        allowed_tree = ancestor_mask[mask_offset] != 0
        return fa4_cute_utils.scalar_to_ssa(
            is_prefix or (is_tree and allowed_tree), cutlass.Boolean
        )


def _require_cuda_contiguous(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@debug_kernel_api
def dflash_paged_tree_verify_attn(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    ancestor_mask: torch.Tensor,
    tree_budget_tensor: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
) -> torch.Tensor:
    """FA4-exact DFlash tree verify over a normal paged KV sequence.

    ``q`` is shaped ``[B * N, Hq, D]`` with one query row per tree node. The
    paged KV sequence for each request is the shared prefix followed by all
    ``N`` tree-node KV slots. ``ancestor_mask[b, q_node, kv_node]`` controls
    which tree suffix slots each query row may see; prefix slots are always
    visible. FA4 owns softcap, softmax, and P*V numerics.
    """

    if _fa4_flash_attn_varlen_func is None:  # pragma: no cover
        raise ImportError(
            "Vendored FlashAttention CUTE is not available (cannot import "
            "flash_attn.cute). Please check your source tree."
        ) from _fa4_import_error
    if cute is None:  # pragma: no cover
        raise ImportError("CUTLASS CUTE is required for DFlash FA4 tree mask")

    _require_cuda_contiguous(q, "q")
    _require_cuda_contiguous(k_cache, "k_cache")
    _require_cuda_contiguous(v_cache, "v_cache")
    _require_cuda_contiguous(page_table, "page_table")
    _require_cuda_contiguous(cache_seqlens, "cache_seqlens")
    _require_cuda_contiguous(prefix_lens, "prefix_lens")
    _require_cuda_contiguous(ancestor_mask, "ancestor_mask")

    if q.dim() != 3:
        raise ValueError(f"q must have shape [B * N, Hq, D], got {tuple(q.shape)}")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must have shape [pages, page, Hkv, D]")
    if page_table.dim() != 2:
        raise ValueError("page_table must have shape [B, max_pages]")
    if prefix_lens.dim() != 1 or cache_seqlens.dim() != 1:
        raise ValueError("prefix_lens and cache_seqlens must be 1D tensors")
    if prefix_lens.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise ValueError("prefix_lens and cache_seqlens must be torch.int32")
    if page_table.dtype != torch.int32:
        raise ValueError("page_table must be torch.int32")
    if ancestor_mask.dtype != torch.int32:
        raise ValueError("ancestor_mask must be torch.int32")

    bs = int(prefix_lens.shape[0])
    rows, num_q_heads, head_dim = q.shape
    if rows <= 0 or bs <= 0:
        raise ValueError("q rows and prefix_lens batch size must be positive")
    if rows % bs != 0:
        raise ValueError(f"q rows={rows} must be divisible by bs={bs}")

    tree_budget = rows // bs
    expected_mask_shape = (bs, tree_budget, tree_budget)
    if tuple(ancestor_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"ancestor_mask must have shape {expected_mask_shape}, got {tuple(ancestor_mask.shape)}"
        )

    q_batched = q.view(bs, tree_budget, num_q_heads, head_dim)
    if tree_budget_tensor is None:
        tree_budget_tensor = torch.tensor(
            [tree_budget], dtype=torch.int32, device=q.device
        )
    _require_cuda_contiguous(tree_budget_tensor, "tree_budget_tensor")
    if tree_budget_tensor.dtype != torch.int32 or tree_budget_tensor.numel() != 1:
        raise ValueError("tree_budget_tensor must be a single torch.int32 value")

    result = _fa4_flash_attn_varlen_func(
        q=q_batched,
        k=k_cache,
        v=v_cache,
        page_table=page_table,
        seqused_k=cache_seqlens,
        max_seqlen_q=tree_budget,
        softmax_scale=softmax_scale,
        causal=False,
        softcap=float(softcap or 0.0),
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        mask_mod=_dflash_tree_mask_mod,
        aux_tensors=[prefix_lens, ancestor_mask.reshape(-1), tree_budget_tensor],
    )
    if isinstance(result, tuple):
        result = result[0]
    return result.reshape(rows, num_q_heads, result.shape[-1])
