"""SiLU-and-mul of the shared expert with its sigmoid gate folded in."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_sigmoid_gate_kernel(
    gate_up_ptr,
    out_ptr,
    gate_up_stride,
    out_stride,
    inter_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    offsets = tl.program_id(axis=1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inter_dim
    base = gate_up_ptr + row * gate_up_stride
    g = tl.load(base + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + inter_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(base + 2 * inter_dim).to(tl.float32)
    a = (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(
        out_ptr + row * out_stride + offsets,
        (a * tl.sigmoid(z)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def silu_and_mul_sigmoid_gate(gate_up: torch.Tensor, inter_dim: int) -> torch.Tensor:
    """gate_up: [T, >= 2 * inter_dim + 1] rows [gate | up | gate logit | pad].
    Returns sigmoid(logit) * bf16(silu(gate) * up), [T, inter_dim]."""
    num_tokens = gate_up.shape[0]
    assert gate_up.stride(1) == 1 and gate_up.shape[1] > 2 * inter_dim
    out = torch.empty(num_tokens, inter_dim, dtype=gate_up.dtype, device=gate_up.device)
    if num_tokens == 0:
        return out
    block = triton.next_power_of_2(inter_dim)
    _silu_and_mul_sigmoid_gate_kernel[(num_tokens, 1)](
        gate_up,
        out,
        gate_up.stride(0),
        out.stride(0),
        inter_dim=inter_dim,
        BLOCK_SIZE=block,
        num_warps=4,
    )
    return out
