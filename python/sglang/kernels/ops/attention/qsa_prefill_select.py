"""QSA (Qwen4-Exp) prefill block selection over packed FP8 E4M3 compressed keys."""

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

_TOPK = 512
_CANDIDATE_CAPACITY = 1664


@cache_once
def _jit_qsa_prefill_select_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "qsa_prefill_select",
        *args,
        cuda_files=["attention/qsa_prefill_select.cuh"],
        cuda_wrappers=[("select", f"QsaPrefillSelect<{args}>::run")],
    )


def qsa_prefill_select(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
) -> torch.Tensor:
    """int32 [rows, 512]: per row, the indices relative to ``row_starts`` of the 512
    best-scoring keys of fp8 ``k[row_starts:row_ends]`` for fp8 ``q`` [rows, heads >= 4,
    128], in unspecified order; rows with at most 512 keys get 0..len-1 then -1."""
    rows = q.shape[0]
    out = torch.empty((rows, _TOPK), dtype=torch.int32, device=q.device)
    candidates = torch.empty(
        (rows, _CANDIDATE_CAPACITY), dtype=torch.int64, device=q.device
    )
    _jit_qsa_prefill_select_module().select(
        q.view(torch.uint8),
        k.view(torch.uint8).view(k.shape[0], -1),
        row_starts.to(device=q.device, dtype=torch.int32),
        row_ends.to(device=q.device, dtype=torch.int32),
        out,
        candidates,
    )
    return out
