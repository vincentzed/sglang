# Model-agnostic custom all-reduce over torch symmetric memory, lifted from the
# Inkling integration (srt/models/inkling_common/kernels/comm.py on the
# inkling-support branch) with the sconv-specific epilogues dropped.
#
# The kernel family (python/sglang/jit_kernel/inkling_all_reduce.py):
#   * v5 push one-shot: multicast-push into a rotating staging slot, ONE
#     per-block barrier, local fp32 reduce -- owns the latency band (decode /
#     target-verify token counts).
#   * v4 full one-shot: every rank multimem-ld_reduces the whole range from a
#     rotating input region -- the <=2-token bucket.
#   * v3/v3b multimem one-shot: in-place multicast ld_reduce/st over the main
#     buffer with grid/per-block barriers -- the large (prefill) band.
#   * v2 two-shot + "mm" (torch multimem): remaining buckets.
#
# CUDA-graph safety: barrier epochs are device-resident and monotonic, so
# capture+replay never sees stale flags. The v4/v5 A/B staging rotation flips
# at Python trace time and bakes into captured graphs, which is only safe if
# every captured graph (and every eager forward) issues an EVEN number of
# rotated ARs -- see the SAFETY INVARIANT below.
#
# SAFETY INVARIANT (reuse distance 2): v4/v5 are out-of-place with no exit
# barrier on the staged input, so a staging region may only be reused two ARs
# later -- the intervening AR's barrier proves every peer consumed it. The A/B
# alternation guarantees that IF the rotated-AR count per forward/graph is
# even (transformer layers issue 2 TP all-reduces per layer: post-attention
# and post-MLP, so counts are even for standard decoder stacks). If a model
# could ever issue an odd number of rotated ARs per forward, a replay boundary
# would put the same region in consecutive ARs -- re-derive the argument
# before enabling this flag there.

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from torch.distributed._symmetric_memory import _SymmetricMemory

logger = logging.getLogger(__name__)

# The custom kernels reduce one 16B vector (8 bf16 elems) at a time and their
# validate() rejects a num_items that isn't a multiple of this.
AR_VEC = 8

# World sizes the JIT kernels support (power-of-two static_assert + the
# multimem instructions require NVLink multicast).
AR_WORLD_SIZES = (4, 8)

# Buffer size when the custom AR is enabled: the largest in-place (v3) payload
# is a full chunked-prefill seam tensor ([chunked_prefill_size, hidden], e.g.
# [16384, 6144] bf16 = 192 MiB) plus the rotating v4/v5 tail regions below.
CUSTOM_AR_BUFFER_BYTES = 256 * 1024 * 1024

# v4 (full one-shot) is out-of-place and drops the exit barrier, so it needs a
# double-buffered input: two rotating input regions (A/B) + one output at the
# tail of the symm buffer. Sized to a few rows at hidden<=6144; v4 only fires
# for the smallest token buckets.
_AR_V4_REGION = 16 * 6144  # elems; 16B-aligned (multiple of AR_VEC)

# v5 (push one-shot) needs a per-rank staging slot on every GPU: two rotating
# staging areas of world * _AR_V5_REGION elems (A/B, same reuse-distance-2
# argument as v4 -- v5's single barrier plays the entry barrier's role) plus
# one local output region. Sized to the tuned v5 band (<=96 rows at
# hidden=6144) plus EAGLE target-verify chains (bs * draft_token_num rows).
_AR_V5_REGION = 160 * 6144  # elems; 16B-aligned (multiple of AR_VEC)


def custom_ar_tail_elems(world_size: int) -> int:
    """Elements reserved at the buffer tail for the v4/v5 rotating regions."""
    return 3 * _AR_V4_REGION + (2 * world_size + 1) * _AR_V5_REGION


class SymmMemCustomArResources(msgspec.Struct):
    """Per-group custom-AR resources: barrier flags/state + symm buffer peer
    and multicast pointers + v4/v5 rotating region offsets."""

    rank: int
    world: int
    buffer_ptrs_dev: int
    multicast_ptr: int
    flag_ptrs_dev: int
    state_ptr: int
    v4_in: tuple[int, int]  # (A, B) input region starts (elems)
    v4_out: int  # v4 output region start (elems)
    v5_in: tuple[int, int]  # (A, B) push-staging region starts (elems)
    v5_out: int  # v5 output region start (elems)
    v4_cur: int = 0  # rotation index, flips per v4 AR
    v5_cur: int = 0  # rotation index, flips per v5 AR
    refs: tuple = ()  # keep-alive: (flags, state, flags_handle)


@functools.cache
def _ar_jit():
    """The inkling_all_reduce JIT wrapper module, imported once on first use."""
    if not is_cuda():
        return None
    from sglang.jit_kernel import inkling_all_reduce

    return inkling_all_reduce


def build_custom_ar_resources(
    *,
    buffer: torch.Tensor,
    handle: "_SymmetricMemory",
    group_name: str,
    world_size: int,
) -> SymmMemCustomArResources | None:
    """Build the barrier flags/state + region layout for ``buffer``.

    Must run eagerly BEFORE any CUDA-graph capture (it rendezvous a flags
    buffer and JIT-compiles the kernels). Returns None when the kernels can't
    run for this group (unsupported world size, no multicast, ROCm).
    """
    if not is_cuda():
        return None
    if world_size not in AR_WORLD_SIZES:
        return None
    if torch.cuda.is_current_stream_capturing():
        logger.warning(
            "symm-mem custom AR resources requested during CUDA-graph capture; "
            "falling back to the default all-reduce path."
        )
        return None
    if handle.multicast_ptr == 0:
        return None

    import torch.distributed._symmetric_memory as torch_symm_mem

    jit = _ar_jit()
    device = buffer.device
    flags = torch_symm_mem.empty(
        jit.flags_numel(world_size), device=device, dtype=torch.uint32
    )
    flags.zero_()
    flags_handle = torch_symm_mem.rendezvous(flags, group_name)
    # Device-side barrier so no peer's first fused-AR kernel can write an epoch
    # into our flags while our zero_ is still pending on the stream.
    flags_handle.barrier()
    state = torch.zeros(jit.STATE_SIZE, device=device, dtype=torch.uint32)
    jit.compile_inkling_all_reduce(buffer.dtype, world_size)

    total = buffer.numel()
    v4reg = _AR_V4_REGION
    v5stage = world_size * _AR_V5_REGION
    v5_base = total - 3 * v4reg - 2 * v5stage - _AR_V5_REGION
    return SymmMemCustomArResources(
        rank=handle.rank,
        world=world_size,
        buffer_ptrs_dev=handle.buffer_ptrs_dev,
        multicast_ptr=handle.multicast_ptr,
        flag_ptrs_dev=flags_handle.buffer_ptrs_dev,
        state_ptr=state.data_ptr(),
        v4_in=(total - 3 * v4reg, total - 2 * v4reg),
        v4_out=total - v4reg,
        v5_in=(v5_base, v5_base + v5stage),
        v5_out=v5_base + 2 * v5stage,
        refs=(flags, state, flags_handle),
    )


def custom_all_reduce(
    inp: torch.Tensor,
    *,
    buffer: torch.Tensor,
    group_name: str,
    res: SymmMemCustomArResources,
) -> torch.Tensor:
    """All-reduce ``inp`` with the autotuned custom kernel family.

    The caller must have passed ``should_custom_all_reduce``-style eligibility
    (bf16, numel % AR_VEC == 0, fits the buffer, resources built). Returns a
    VIEW of the symm buffer holding the reduced result -- the caller must
    consume it before issuing the next all-reduce on this group (true for
    transformer AR->norm->GEMM seams, where every consumer kernel launches
    before the next AR).
    """
    jit = _ar_jit()
    n = inp.numel()
    num_tokens = inp.shape[0] if inp.dim() >= 2 else n
    kernel, nb, bs = jit.select_ar_config(num_tokens, res.world)
    esz = buffer.element_size()

    if kernel == "v5" and n <= _AR_V5_REGION and inp.data_ptr() % 16 == 0:
        # Push one-shot: multicast-push input into the rotating staging area,
        # one per-block barrier, local fp32 reduce into the out region. The
        # input is read locally, so it needs NO stage-in copy.
        cur = res.v5_cur
        stage_off = res.v5_in[cur]
        out_view = buffer[res.v5_out : res.v5_out + n]
        jit.inkling_multimem_push_oneshot(
            inp.view(-1),
            out_view,
            res.multicast_ptr + stage_off * esz,
            buffer.data_ptr() + stage_off * esz,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            res.world,
            n,
            nb,
            bs,
            per_block_barrier=True,
        )
        res.v5_cur = 1 - cur
        return out_view.view(inp.shape)

    if kernel == "v4" and n <= _AR_V4_REGION:
        # Out-of-place full one-shot in the double-buffered tail regions.
        cur = res.v4_cur
        in_off = res.v4_in[cur]
        in_view = buffer[in_off : in_off + n]
        out_view = buffer[res.v4_out : res.v4_out + n]
        in_view.copy_(inp.view(-1))
        jit.inkling_multimem_full_oneshot(
            in_view,
            out_view,
            res.multicast_ptr + in_off * esz,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            res.world,
            n,
            nb,
            bs,
        )
        res.v4_cur = 1 - cur
        return out_view.view(inp.shape)

    buf = buffer[:n]
    buf.copy_(inp.view(-1))
    if kernel in ("v3", "v3b"):
        jit.inkling_multimem_one_shot_fused(
            buf,
            res.multicast_ptr,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            res.world,
            n,
            nb,
            bs,
            per_block_barrier=(kernel == "v3b"),
        )
        return buf.view(inp.shape)
    if kernel == "v2":
        jit.inkling_two_shot_all_reduce_fused(
            buf,
            res.buffer_ptrs_dev,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            res.world,
            n,
            nb,
            bs,
        )
        return buf.view(inp.shape)
    # "mm" bucket (and v5/v4 payloads that outgrew their regions): torch
    # multimem on the symm buffer.
    torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", group_name)
    return buf.view(inp.shape)
