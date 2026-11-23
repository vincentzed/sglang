"""QSA (Qwen4-Exp) index scoring over the FP8 E4M3 compressed-key cache."""

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

_WARPS_PER_CTA = 8
_WARPS_PER_SM = 32
_DECODE_TOPK = 512


@cache_once
def _jit_qsa_index_score_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "qsa_index_score",
        *args,
        cuda_files=["attention/qsa_index_score.cuh"],
        cuda_wrappers=[
            ("decode", f"QsaIndexScoreDecode<{args}>::run"),
            ("decode_topk", f"QsaDecodeTopk<{args}>::run"),
        ],
    )


@cache_once
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _decode_splits(groups: int, device: torch.device) -> int:
    ctas = _num_sms(device.index or 0) * _WARPS_PER_SM // _WARPS_PER_CTA
    return max(1, ctas // groups)


def qsa_index_score_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    lens: torch.Tensor,
    logits: torch.Tensor,
    scale: float,
    *,
    group_rows: int = 1,
    min_scored_len: int = 0,
) -> None:
    """Score fp8 ``q`` [rows, heads >= 4, 128] against the paged fp8 keys into
    ``logits[row, :lens[row]]`` for rows longer than ``min_scored_len``. The
    ``group_rows`` consecutive rows of a request read the last row's pages."""
    groups = -(-q.shape[0] // group_rows)
    _jit_qsa_index_score_module().decode(
        q.view(torch.uint8),
        k_cache.view(torch.uint8).view(k_cache.shape[0], -1),
        page_table,
        lens,
        logits,
        float(scale),
        group_rows,
        min_scored_len,
        _decode_splits(groups, q.device),
    )


def qsa_decode_topk(logits: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Per row, the indices of the 512 largest ``logits[row, :lens[row]]``, ties
    at the cut by the lower index, in unspecified order; rows with at most 512
    entries get ``0..len-1`` then -1. Int32 [rows, 512]."""
    out = torch.empty(
        (logits.shape[0], _DECODE_TOPK), dtype=torch.int32, device=logits.device
    )
    _jit_qsa_index_score_module().decode_topk(logits, lens, out)
    return out
