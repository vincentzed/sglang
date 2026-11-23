"""GDN ReplaySSM commit fold of the compact bf16 ring in a CUDA tcgen05 kernel:
the capacity rows of ``gdn_replayssm_compact_commit_tiled_kernel``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_sm100_supported

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_DIM = 128
_RING = 16


@cache_once
def _jit_gdn_replayssm_fold_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "gdn_replayssm_fold",
        *args,
        cuda_files=["attention/gdn_replayssm_fold.cuh"],
        cuda_wrappers=[("fold", f"GdnReplaySSMFold<{args}>::run")],
        cuda_arch_policy="specific",
    )


def gdn_replayssm_fold_cuda_supported(
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: torch.Tensor | None,
    k_residual_cache: torch.Tensor | None,
) -> bool:
    if d_residual_cache is None or k_residual_cache is None:
        return False
    half = (checkpoint_state, d_cache, k_cache, d_residual_cache, k_residual_cache)
    return (
        checkpoint_state.is_cuda
        and is_sm100_supported()
        and all(t.dtype == torch.bfloat16 for t in half)
        and g_cache.dtype == torch.float32
        and tuple(checkpoint_state.shape[3:]) == (_DIM, _DIM)
        and d_cache.shape[3:] == (_RING, _DIM)
        and k_cache.shape[3:] == (_RING, _DIM)
        and d_residual_cache.shape == d_cache.shape
        and k_residual_cache.shape == k_cache.shape
        and d_cache.shape[2] % k_cache.shape[2] == 0
        and all(t[:1, :1].is_contiguous() for t in half)
        and g_cache[:1, :1].is_contiguous()
    )


def gdn_replayssm_fold_cuda(
    checkpoint_state: torch.Tensor,  # [layers, slots, HV, 128, 128] bf16, folded in place
    d_cache: torch.Tensor,  # [layers, rings, HV, 16, 128] bf16
    d_residual_cache: torch.Tensor,  # low part of d_cache
    k_cache: torch.Tensor,  # [layers, rings, H, 16, 128] bf16
    k_residual_cache: torch.Tensor,  # low part of k_cache
    g_cache: torch.Tensor,  # [layers, rings, HV, 16] fp32
    state_batch_indices: torch.Tensor,  # [rows]
    replay_indices: torch.Tensor,  # [rows]
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    null_block_id: int = -1,
) -> None:
    """Fold the ``write_pos`` committed records from ``cache_base`` of every row
    flagged in ``is_flush`` into its checkpoint, in every layer; the cursors are
    left to the finish kernel."""
    if is_flush.dtype != torch.int8:
        is_flush = is_flush.to(torch.int8)
    _jit_gdn_replayssm_fold_module().fold(
        checkpoint_state,
        d_cache,
        d_residual_cache,
        k_cache,
        k_residual_cache,
        g_cache,
        state_batch_indices,
        replay_indices,
        write_pos,
        cache_base,
        is_flush,
        null_block_id,
    )
