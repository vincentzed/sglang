from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

GDN_PREFILL_CONV_BLOCK_M = 32
GDN_PREFILL_CONV_BLOCK_N = 128
GDN_PREFILL_CONV_NUM_WARPS = 4


@triton.jit(do_not_specialize=["total_tokens"])
def _gdn_prefill_conv_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_states_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    o_ptr,
    total_tokens,
    stride_x_token,
    stride_w_dim,
    stride_w_width,
    stride_state_seq,
    stride_state_dim,
    stride_state_token,
    eps,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    USE_BLOCK_TABLE: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if USE_BLOCK_TABLE:
        idx_seq = tl.load(block_table_ptr + tl.program_id(1) * 2)
        chunk = tl.load(block_table_ptr + tl.program_id(1) * 2 + 1)
    else:
        idx_seq = tl.program_id(2)
        chunk = tl.program_id(1)
    if idx_seq == PAD_SLOT_ID:
        return
    seq_start = tl.load(query_start_loc_ptr + idx_seq)
    seqlen = tl.load(query_start_loc_ptr + idx_seq + 1) - seq_start
    tok0 = chunk * BLOCK_M
    if tok0 >= seqlen:
        return
    slot = tl.load(cache_indices_ptr + idx_seq).to(tl.int64)
    if slot == PAD_SLOT_ID:
        return
    state_len: tl.constexpr = KERNEL_WIDTH - 1
    load_init = False
    if HAS_INITIAL_STATES:
        load_init = tl.load(has_initial_state_ptr + idx_seq).to(tl.int1)

    rows = tok0 + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < seqlen
    x_base = x_ptr + seq_start.to(tl.int64) * stride_x_token
    state_base = conv_states_ptr + slot * stride_state_seq + cols * stride_state_dim

    if HAS_BIAS:
        acc = (
            tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            + tl.load(bias_ptr + cols).to(tl.float32)[None, :]
        )
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for j in tl.static_range(KERNEL_WIDTH):
        src = rows - state_len + j
        xv = tl.load(
            x_base + (src.to(tl.int64) * stride_x_token)[:, None] + cols[None, :],
            mask=(row_ok & (src >= 0))[:, None],
            other=0.0,
        )
        w = tl.load(w_ptr + cols * stride_w_dim + j * stride_w_width).to(tl.float32)
        acc = tl.fma(xv.to(tl.float32), w[None, :], acc)
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))

    out_rows = (seq_start + rows).to(tl.int64)
    block_start = tl.program_id(0) * BLOCK_N
    if block_start < 2 * QK_DIM:
        y = acc.to(o_ptr.dtype.element_ty).to(tl.float32)
        y = y * (1 / tl.sqrt(tl.sum(y * y, axis=1) + eps))[:, None]
        plane_start = (block_start // QK_DIM) * QK_DIM
        o_ptrs = (
            o_ptr
            + plane_start.to(tl.int64) * total_tokens
            + out_rows[:, None] * QK_DIM
            + (cols - plane_start)[None, :]
        )
        tl.store(o_ptrs, y.to(o_ptr.dtype.element_ty), mask=row_ok[:, None])
    else:
        o_ptrs = (
            o_ptr
            + 2 * QK_DIM * total_tokens.to(tl.int64)
            + out_rows[:, None] * V_DIM
            + (cols - 2 * QK_DIM)[None, :]
        )
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=row_ok[:, None])

    if chunk == 0:
        if load_init:
            tl.debug_barrier()
            head = tl.arange(0, NP2_STATELEN)
            head_ok = (head < state_len) & (head < seqlen)
            acc0 = tl.zeros((NP2_STATELEN, BLOCK_N), dtype=tl.float32)
            if HAS_BIAS:
                acc0 += tl.load(bias_ptr + cols).to(tl.float32)[None, :]
            for j in tl.static_range(KERNEL_WIDTH):
                hsrc = head - state_len + j
                hx = tl.load(
                    x_base
                    + (hsrc.to(tl.int64) * stride_x_token)[:, None]
                    + cols[None, :],
                    mask=(head_ok & (hsrc >= 0))[:, None],
                    other=0.0,
                ).to(tl.float32)
                hs = tl.load(
                    state_base[None, :]
                    + ((hsrc + state_len) * stride_state_token)[:, None],
                    mask=(head_ok & (hsrc < 0))[:, None],
                    other=0.0,
                ).to(tl.float32)
                hw = tl.load(w_ptr + cols * stride_w_dim + j * stride_w_width).to(
                    tl.float32
                )
                acc0 = tl.fma(tl.where((hsrc >= 0)[:, None], hx, hs), hw[None, :], acc0)
            if SILU_ACTIVATION:
                acc0 = acc0 / (1 + tl.exp(-acc0))
            head_rows = (seq_start + head).to(tl.int64)
            if block_start < 2 * QK_DIM:
                y0 = acc0.to(o_ptr.dtype.element_ty).to(tl.float32)
                y0 = y0 * (1 / tl.sqrt(tl.sum(y0 * y0, axis=1) + eps))[:, None]
                plane0 = (block_start // QK_DIM) * QK_DIM
                o0 = (
                    o_ptr
                    + plane0.to(tl.int64) * total_tokens
                    + head_rows[:, None] * QK_DIM
                    + (cols - plane0)[None, :]
                )
                tl.store(o0, y0.to(o_ptr.dtype.element_ty), mask=head_ok[:, None])
            else:
                o0 = (
                    o_ptr
                    + 2 * QK_DIM * total_tokens.to(tl.int64)
                    + head_rows[:, None] * V_DIM
                    + (cols - 2 * QK_DIM)[None, :]
                )
                tl.store(o0, acc0.to(o_ptr.dtype.element_ty), mask=head_ok[:, None])
    if chunk == 0:
        tl.debug_barrier()
        idx = tl.arange(0, NP2_STATELEN)
        keep = idx < state_len
        if seqlen >= state_len:
            new_state = tl.load(
                x_base
                + ((seqlen - state_len + idx).to(tl.int64) * stride_x_token)[:, None]
                + cols[None, :],
                mask=keep[:, None],
                other=0.0,
            )
        else:
            from_old = (idx + seqlen) < state_len
            old = tl.load(
                state_base[None, :] + ((idx + seqlen) * stride_state_token)[:, None],
                mask=(keep & from_old & load_init)[:, None],
                other=0.0,
            )
            src_tok = idx - (state_len - seqlen)
            new_x = tl.load(
                x_base
                + (src_tok.to(tl.int64) * stride_x_token)[:, None]
                + cols[None, :],
                mask=(keep & (src_tok >= 0))[:, None],
                other=0.0,
            )
            new_state = tl.where(from_old[:, None], old.to(new_x.dtype), new_x)
        tl.debug_barrier()
        tl.store(
            state_base[None, :] + (idx * stride_state_token)[:, None],
            new_state,
            mask=keep[:, None],
        )


def gdn_prefill_conv_qkv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: Optional[torch.Tensor],
    qk_dim: int,
    seq_lens_cpu: Optional[List[int]] = None,
    block_table: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    eps: float = 1e-6,
    pad_slot_id: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal depthwise conv of a GDN prefill's token-major ``x [T, 2 qk_dim +
    v_dim]`` into contiguous q, k, v planes, q and k l2-normalized per
    ``GDN_PREFILL_CONV_BLOCK_N``-channel head; updates ``conv_states`` in place.
    ``block_table`` rows are (sequence, GDN_PREFILL_CONV_BLOCK_M-token chunk)."""
    total_tokens, dim = x.shape
    v_dim = dim - 2 * qk_dim
    if (
        x.stride(1) != 1
        or qk_dim % GDN_PREFILL_CONV_BLOCK_N
        or v_dim % GDN_PREFILL_CONV_BLOCK_N
    ):
        raise ValueError(
            "gdn_prefill_conv_qkv needs unit channel stride and 128-channel heads"
        )
    width = weight.shape[1]
    out = torch.empty(dim * total_tokens, dtype=x.dtype, device=x.device)
    if block_table is not None:
        grid = (dim // GDN_PREFILL_CONV_BLOCK_N, block_table.shape[0], 1)
    else:
        grid = (
            dim // GDN_PREFILL_CONV_BLOCK_N,
            triton.cdiv(max(seq_lens_cpu), GDN_PREFILL_CONV_BLOCK_M),
            len(seq_lens_cpu),
        )
    _gdn_prefill_conv_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        block_table,
        out,
        total_tokens,
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        eps,
        QK_DIM=qk_dim,
        V_DIM=v_dim,
        KERNEL_WIDTH=width,
        NP2_STATELEN=triton.next_power_of_2(width - 1),
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish", True),
        HAS_INITIAL_STATES=has_initial_state is not None,
        USE_BLOCK_TABLE=block_table is not None,
        PAD_SLOT_ID=pad_slot_id,
        BLOCK_M=GDN_PREFILL_CONV_BLOCK_M,
        BLOCK_N=GDN_PREFILL_CONV_BLOCK_N,
        num_warps=GDN_PREFILL_CONV_NUM_WARPS,
    )
    q, k, v = out.split(
        [qk_dim * total_tokens, qk_dim * total_tokens, v_dim * total_tokens]
    )
    return (
        q.view(total_tokens, qk_dim),
        k.view(total_tokens, qk_dim),
        v.view(total_tokens, v_dim),
    )
