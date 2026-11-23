"""QSA (Qwen4-Exp) prefill block selection over packed FP8 E4M3 compressed keys: sampled
threshold, fused score + collect, exact pick."""

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
_CAPACITY = 1664
_WORK_STRIDE = 258
_ROWS_PER_CTA = 64
_CTAS_PER_SM = 2
_MIN_SPLIT_KEYS = 256


@cache_once
def _jit_qsa_prefill_topk_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "qsa_prefill_topk",
        *args,
        cuda_files=["attention/qsa_prefill_topk.cuh"],
        cuda_wrappers=[("select", f"QsaPrefillTopk<{args}>::run")],
    )


@cache_once
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def qsa_prefill_topk_splits(rows: int, keys: int, device: torch.device) -> int:
    tiles = (rows + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA
    want = (_CTAS_PER_SM * _sm_count(device.index or 0) + tiles - 1) // max(tiles, 1)
    return max(1, min(want, (keys + _MIN_SPLIT_KEYS - 1) // _MIN_SPLIT_KEYS, 1024))


def qsa_prefill_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    splits: int = 0,
) -> torch.Tensor:
    """int32 [rows, 512]: per row, the indices relative to ``row_starts`` of the 512
    best-scoring keys of fp8 ``k[row_starts:row_ends]`` for fp8 ``q`` [rows, heads >= 4,
    128] (score: sum over 4 heads of relu(q . k)), ties at the cut to the lower index, in
    unspecified order; rows with at most 512 keys get 0..len-1 then -1."""
    rows = q.shape[0]
    out = torch.empty((rows, _TOPK), dtype=torch.int32, device=q.device)
    candidates = torch.empty((rows, _CAPACITY), dtype=torch.int64, device=q.device)
    work = torch.zeros((rows, _WORK_STRIDE), dtype=torch.int32, device=q.device)
    k2 = k.view(torch.uint8).view(k.shape[0], -1)
    _jit_qsa_prefill_topk_module().select(
        q.view(torch.uint8),
        k2,
        row_starts.to(device=q.device, dtype=torch.int32),
        row_ends.to(device=q.device, dtype=torch.int32),
        out,
        candidates,
        work,
        splits or qsa_prefill_topk_splits(rows, k2.shape[0], q.device),
    )
    return out
