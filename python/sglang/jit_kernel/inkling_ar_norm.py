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

# Rounding calibration: sgl_kernel.fused_add_rmsnorm keeps the residual sum in
# fp32 through the variance and the gamma scale (bit-identity calibrated on
# sgl_kernel 0.4.4 / B300 in test_symm_mem_custom_ar.py::
# test_fused_ar_norm_bit_identity -- the test's calibration probe reports
# which mode matches if this ever changes).
ROUND_SUM_TO_BF16 = False


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
        # sgl-kernel builds flashinfer's FusedAddRMSNorm with --use_fast_math;
        # the norm epilogue must see identical instruction lowering (ftz,
        # approx div, fmad contraction) for bit-identity with that kernel.
        extra_cuda_cflags=["--use_fast_math"],
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
