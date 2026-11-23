"""QSA (Qwen4-Exp sparse attention) chunked-prefill attention on SM100 / SM103
over the sequences' packed K (fp8 e4m3) and V (bf16), for groups of consecutive
rows of a sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HEAD_DIM = 256
_TILE = 128
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
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    *,
    extend_lens: Sequence[int],
    max_seq_len: int,
    rows_per_group: int | None = None,
) -> torch.Tensor:
    """Sparse attention of the new rows of each sequence over their selected
    KV positions.

    Args:
        q: ``[rows, q_heads, 256]`` bf16, the sequences' new rows back to back.
        k: ``[total_k, kv_heads, 256]`` float8_e4m3fn, the sequences' keys back
            to back (``pack_qsa_prefill_kv``), the new rows included.
        v: ``[total_k, kv_heads, 256]`` bfloat16 values, packed like ``k``.
        indices: ``[rows, topk]`` int32 logical positions, ``-1`` = hole.
        cu_q: ``[sequences + 1]`` first new row of each sequence.
        cu_k: ``[sequences + 1]`` first row of each sequence in ``k`` / ``v``.
        seq_lens: ``[sequences]`` length of each sequence with its new rows.
        sm_scale: softmax scale.
        extend_lens: new rows of each sequence, on the host.
        max_seq_len: the longest sequence, on the host.
        rows_per_group: rows merged into one group; by default 4 below 512
            rows, else 8.

    Returns:
        ``[rows, q_heads, 256]`` bf16.
    """
    rows, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    heads = q_heads // kv_heads
    if (
        head_dim != _HEAD_DIM
        or k.dtype != torch.float8_e4m3fn
        or v.dtype != torch.bfloat16
    ):
        raise ValueError(
            "qsa_sparse_prefill_sm100 needs head_dim 256, float8_e4m3fn keys and "
            f"bfloat16 values, got head_dim {head_dim}, {k.dtype} and {v.dtype}"
        )
    out = torch.empty_like(q)
    if rows == 0:
        return out
    if rows_per_group is None:
        rows_per_group = 4 if rows < 512 else 8
    device = q.device
    groups = sum(-(-n // rows_per_group) for n in extend_lens)
    capacity = -(-rows_per_group * indices.shape[1] // _TILE) * _TILE
    slots = torch.empty((groups, capacity), dtype=torch.int32, device=device)
    masks = torch.empty((groups, capacity), dtype=torch.uint8, device=device)
    meta = torch.empty((groups, 4), dtype=torch.int32, device=device)
    _jit_qsa_sparse_prefill_sm100_module(rows_per_group, heads).run(
        q,
        k.view(torch.uint8),
        v.view(torch.uint8),
        indices,
        cu_q.to(torch.int32),
        cu_k.to(torch.int32),
        seq_lens.to(torch.int32),
        slots,
        masks,
        meta,
        out,
        float(sm_scale),
        max_seq_len,
    )
    return out
