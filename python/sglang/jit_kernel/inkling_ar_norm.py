"""Fused {v5 push all-reduce -> residual-add + RMSNorm} decode/verify kernel.

The norm-only member of the Inkling AR family (inkling_ar_norm.cuh): one block
per token, per-block cross-GPU barrier, fp32 reduce rounded to bf16 at exactly
the points the unfused {AR -> fused_add_rmsnorm} chain would round. Occupies a
v5 staging rotation slot (same reuse-distance-2 invariant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Rounding calibration: sgl_kernel.fused_add_rmsnorm's 16B-vector path rounds
# the residual sum to bf16 before squaring/scaling (vLLM _f16Vec semantics).
# Verified bit-identical in test_symm_mem_custom_ar.py; flip only if the
# installed sgl_kernel changes its accumulation.
ROUND_SUM_TO_BF16 = True


@cache_once
def _jit_inkling_ar_norm_module(
    dtype: torch.dtype, world_size: int, round_sum: bool
) -> Module:
    args = make_cpp_args(dtype, world_size, round_sum)
    return load_jit(
        "inkling_ar_norm",
        *args,
        cuda_files=["inkling/inkling_ar_norm.cuh"],
        cuda_wrappers=[
            ("ar_add_rmsnorm", f"ArAddRmsNormKernel<{args}>::run"),
        ],
    )


def compile_inkling_ar_norm(dtype: torch.dtype, world_size: int) -> None:
    """Warm the JIT module for (dtype, world_size) so the first call is cheap
    (it can land inside a CUDA-graph capture)."""
    _jit_inkling_ar_norm_module(dtype, world_size, ROUND_SUM_TO_BF16)


def inkling_ar_add_rmsnorm(
    inp: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    hs_out: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    mc_stage_ptr: int,
    local_stage_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    *,
    enable_pdl: bool = True,
    vecs_per_thread: int = 1,
    round_sum: bool = ROUND_SUM_TO_BF16,
) -> None:
    """Fused {AR -> add + RMSNorm} over ``inp`` ([T, D] UNREDUCED partials).

    Writes ``hs_out`` (normed) and ``residual_out`` (post-add residual),
    exactly like the unfused ``custom-AR -> fused_add_rmsnorm`` chain. The
    caller manages the v5 staging rotation (mc_stage_ptr/local_stage_ptr point
    at the CURRENT rotation slot) and must flip it after this call.
    """
    module = _jit_inkling_ar_norm_module(inp.dtype, world_size, round_sum)
    module.ar_add_rmsnorm(
        inp,
        residual,
        residual_out,
        hs_out,
        norm_weight,
        eps,
        mc_stage_ptr,
        local_stage_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        int(enable_pdl),
        vecs_per_thread,
    )
