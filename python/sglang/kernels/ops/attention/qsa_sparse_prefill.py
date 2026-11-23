"""QSA (Qwen4-Exp sparse attention) chunked-prefill attention over an FP8 E4M3
KV pool on SM100 / SM103, for groups of consecutive rows of a sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HEAD_DIM = 256
_TILE = 128
ROWS_PER_GROUP = 4
MAX_HEADS_PER_KV_HEAD = 16
# a row mask byte per position of the longest sequence has to fit shared memory
MAX_KV_LEN = 230_000


@cache_once
def _jit_qsa_sparse_prefill_sm100_module(rows_per_group: int, heads: int) -> Module:
    args = make_cpp_args(rows_per_group, heads)
    return load_jit(
        "qsa_sparse_prefill_sm100",
        *args,
        cuda_files=["attention/qsa_sparse_prefill_sm100.cuh"],
        cuda_wrappers=[("run", f"QsaSparsePrefillSm100<{args}>::run")],
    )


def qsa_sparse_prefill_sm100(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    indices: torch.Tensor,
    cu_q: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    *,
    extend_lens: Sequence[int],
    max_seq_len: int,
    rows_per_group: int = ROWS_PER_GROUP,
) -> torch.Tensor:
    """Sparse attention of the new rows of each sequence over their selected
    KV positions.

    Args:
        q: ``[rows, q_heads, 256]`` bf16, the sequences' new rows back to back.
        k_cache, v_cache: ``[slots, kv_heads, 256]`` float8_e4m3fn KV pool
            buffers (unit scale) that already hold the new rows.
        req_to_token: ``[requests, context]`` int32 position -> slot table.
        req_pool_indices: ``[sequences]`` row of ``req_to_token`` per sequence.
        indices: ``[rows, topk]`` int32 logical positions, ``-1`` = hole.
        cu_q: ``[sequences + 1]`` int32 first new row of each sequence.
        seq_lens: ``[sequences]`` int32 length of each sequence with its new rows.
        sm_scale: softmax scale.
        extend_lens: new rows of each sequence, on the host.
        max_seq_len: the longest sequence, on the host.

    Returns:
        ``[rows, q_heads, 256]`` bf16.
    """
    rows, q_heads, head_dim = q.shape
    kv_heads = k_cache.shape[1]
    heads = q_heads // kv_heads
    if head_dim != _HEAD_DIM or k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "qsa_sparse_prefill_sm100 needs head_dim 256 and a float8_e4m3fn KV cache, "
            f"got head_dim {head_dim} and {k_cache.dtype}"
        )
    out = torch.empty_like(q)
    if rows == 0:
        return out
    device = q.device
    groups = sum(-(-n // rows_per_group) for n in extend_lens)
    capacity = -(-rows_per_group * indices.shape[1] // _TILE) * _TILE
    slots = torch.empty((groups, capacity), dtype=torch.int32, device=device)
    masks = torch.empty((groups, capacity), dtype=torch.uint8, device=device)
    meta = torch.empty((groups, 4), dtype=torch.int32, device=device)
    _jit_qsa_sparse_prefill_sm100_module(rows_per_group, heads).run(
        q,
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        req_to_token,
        req_pool_indices.to(torch.int32),
        indices,
        cu_q.to(torch.int32),
        seq_lens.to(torch.int32),
        slots,
        masks,
        meta,
        out,
        float(sm_scale),
        max_seq_len,
    )
    return out
