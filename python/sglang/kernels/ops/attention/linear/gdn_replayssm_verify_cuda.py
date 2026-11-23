"""GDN ReplaySSM speculative verify for the compact bf16 ring in a CUDA tensor-core
kernel: the contract of ``gdn_replayssm_spec_decode(..., launch_mode="verify")``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_DIM = 128
_RING = 16
_MAX_SPEC_LEN = 4
_INDEX_DTYPES = (torch.int32, torch.int64)


@cache_once
def _jit_gdn_replayssm_verify_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "gdn_replayssm_verify",
        *args,
        cuda_files=["attention/gdn_replayssm_verify.cuh"],
        cuda_wrappers=[("verify", f"GdnReplaySSMVerify<{args}>::run")],
    )


def prewarm_gdn_replayssm_verify_cuda() -> None:
    _jit_gdn_replayssm_verify_module()


def _rows_16_byte_aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and t.stride(0) % 8 == 0


def _inner_contiguous(t: torch.Tensor) -> bool:
    return t[:1].is_contiguous()


def gdn_replayssm_verify_cuda_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: torch.Tensor | None,
    k_residual_cache: torch.Tensor | None,
    beta_cache: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    max_spec_len: int,
) -> bool:
    if d_residual_cache is None or k_residual_cache is None or beta_cache is not None:
        return False
    half = (
        q,
        k,
        v,
        a,
        b,
        checkpoint_state,
        d_cache,
        k_cache,
        d_residual_cache,
        k_residual_cache,
    )
    return (
        q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device)[0] >= 9
        and max_spec_len <= _MAX_SPEC_LEN
        and all(t.dtype == torch.bfloat16 for t in half)
        and g_cache.dtype == torch.float32
        and A_log.dtype == dt_bias.dtype
        and A_log.dtype in (torch.float32, torch.bfloat16)
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
        and tuple(checkpoint_state.shape[2:]) == (_DIM, _DIM)
        and q.shape[-1] == _DIM
        and v.shape[-1] == _DIM
        and d_cache.shape[2] == _RING
        and d_residual_cache.shape == d_cache.shape
        and k_residual_cache.shape == k_cache.shape
        and query_start_loc.dtype in _INDEX_DTYPES
        and ssm_state_indices.dtype in _INDEX_DTYPES
        and replay_indices.dtype in _INDEX_DTYPES
        and write_pos.dtype == torch.int32
        and cache_base.dtype == torch.int32
        and all(_inner_contiguous(t) for t in half)
        and _inner_contiguous(g_cache)
        and all(
            _rows_16_byte_aligned(t)
            for t in (q, k, checkpoint_state, k_cache, k_residual_cache)
        )
    )


def gdn_replayssm_verify_cuda(
    q: torch.Tensor,  # [tokens, H, 128] bf16
    k: torch.Tensor,  # [tokens, H, 128] bf16
    v: torch.Tensor,  # [tokens, HV, 128] bf16
    a: torch.Tensor,  # [tokens, HV] bf16
    b: torch.Tensor,  # [tokens, HV] bf16
    A_log: torch.Tensor,  # [HV]
    dt_bias: torch.Tensor,  # [HV]
    checkpoint_state: torch.Tensor,  # [slots, HV, 128, 128] bf16, read only
    d_cache: torch.Tensor,  # [rings, HV, 16, 128] bf16
    k_cache: torch.Tensor,  # [rings, H, 16, 128] bf16
    g_cache: torch.Tensor,  # [rings, HV, 16] fp32
    d_residual_cache: torch.Tensor,  # low part of d_cache
    k_residual_cache: torch.Tensor,  # low part of k_cache
    out: torch.Tensor,  # [tokens, HV, 128] bf16
    query_start_loc: torch.Tensor,  # [rows + 1]
    ssm_state_indices: torch.Tensor,  # [rows]
    replay_indices: torch.Tensor,  # [rows]
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    scale: float,
    null_block_id: int = -1,
) -> torch.Tensor:
    if is_flush.dtype != torch.int8:
        is_flush = is_flush.to(torch.int8)
    _jit_gdn_replayssm_verify_module().verify(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        out,
        checkpoint_state,
        d_cache,
        d_residual_cache,
        k_cache,
        k_residual_cache,
        g_cache,
        query_start_loc,
        ssm_state_indices,
        replay_indices,
        write_pos,
        cache_base,
        is_flush,
        float(scale),
        null_block_id,
    )
    return out
