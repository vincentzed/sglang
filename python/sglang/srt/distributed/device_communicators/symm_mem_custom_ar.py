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
# capture+replay never sees stale flags.
#
# Staging-reuse safety: this port does NOT use the Inkling A/B staging
# rotation. The rotation's reuse-distance-2 invariant requires an EVEN number
# of rotated ARs per captured graph, which a model-agnostic drop-in cannot
# guarantee (measured counterexample: EAGLE verify graphs with flashinfer
# seam-B fusion issue an ODD count -- one v5 per layer at seam A plus a single
# unfused last-layer seam-B AR -- corrupting activations at every graph-replay
# boundary; AIME25 dropped 92.9% -> 78.3% before this was fixed). Instead,
# every push-style kernel here takes an ENTRY barrier before its multicast
# push: a rank can only overwrite the staging slot after every peer finished
# its previous kernel, which by stream order includes its previous read of the
# slot. One slot, safe for any AR sequence, ~2-4 us extra per call.

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from torch.distributed._symmetric_memory import _SymmetricMemory

    from sglang.srt.distributed.parallel_state import GroupCoordinator

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

# v5 (push one-shot) needs a per-rank staging slot on every GPU (world *
# _AR_V5_REGION elems) plus one local output region. The layout still reserves
# the Inkling A/B pair, but only slot A is used -- the in-kernel ENTRY barrier
# fences reuse (see the header comment). Sized to the tuned v5 band (<=96 rows
# at hidden=6144) plus EAGLE target-verify chains (bs * draft_token_num rows).
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


@functools.cache
def _ar_norm_jit():
    """The fused {AR -> add + RMSNorm} JIT wrapper module."""
    if not is_cuda():
        return None
    from sglang.jit_kernel import inkling_ar_norm

    return inkling_ar_norm


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
    if envs.SGLANG_OPT_USE_SYMM_MEM_FUSED_AR_NORM.get():
        # Warm the fused {AR -> add + RMSNorm} module too: its first call can
        # land inside a CUDA-graph capture, which must not pay an nvcc compile.
        _ar_norm_jit().compile_inkling_ar_norm(buffer.dtype, world_size)

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

    if kernel in ("v5", "v4") and n <= _AR_V5_REGION and inp.data_ptr() % 16 == 0:
        # v4 (full one-shot) selections also route here: v4's rotating input
        # staging has the same replay-boundary hazard, and entry-barrier v5
        # covers its tiny-token band safely.
        # Push one-shot WITH entry barrier: multicast-push into a single
        # staging slot, per-block barriers before and after the push, local
        # fp32 reduce into the out region. The input is read locally, so it
        # needs NO stage-in copy. The entry barrier (not A/B rotation) fences
        # staging reuse: this drop-in serves arbitrary model AR sequences,
        # where the rotated-AR count per captured graph is NOT guaranteed even
        # (e.g. EAGLE verify graphs with flashinfer seam-B fusion issue an ODD
        # count, which corrupted the single-barrier rotating variant at every
        # replay boundary).
        stage_off = res.v5_in[0]
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
            entry_barrier=True,
        )
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


def fused_ar_rmsnorm_shape_eligible(
    group: "GroupCoordinator",
    num_tokens: int,
    hidden: int,
) -> bool:
    """Shape-only gate for the fused {AR -> add + RMSNorm} kernel -- what a
    PRODUCER (should_fuse_mlp_allreduce_with_next_layer) can evaluate before
    the tensor exists. A pure function of per-forward state: every rank must
    take the same branch (the in-kernel barrier deadlocks on divergence)."""
    if not envs.SGLANG_OPT_USE_SYMM_MEM_FUSED_AR_NORM.get():
        return False
    comm = group.torch_symm_mem_comm
    if comm is None or comm.disabled or comm.custom_ar_resources is None:
        return False
    if num_tokens < 1 or num_tokens > _ar_jit().MAX_BARRIER_BLOCKS:
        return False
    if hidden % AR_VEC != 0 or hidden // AR_VEC > 1024:
        return False
    return num_tokens * hidden <= _AR_V5_REGION


def fused_ar_rmsnorm_eligible(
    group: "GroupCoordinator",
    inp: torch.Tensor,
    residual: torch.Tensor | None,
) -> bool:
    """Full consumer-side gate for the fused {AR -> add + RMSNorm} kernel.

    A consumer-side False after a producer-side (shape-only) True is safe --
    the caller's unfused branch still performs the all-reduce.
    """
    if residual is None or inp.dim() != 2:
        return False
    comm = group.torch_symm_mem_comm
    if comm is None or comm.disabled:
        return False
    if inp.dtype != comm.dtype or inp.device != comm.device:
        return False
    if not fused_ar_rmsnorm_shape_eligible(group, inp.shape[0], inp.shape[1]):
        return False
    if not (inp.is_contiguous() and residual.is_contiguous()):
        return False
    return inp.data_ptr() % 16 == 0 and residual.data_ptr() % 16 == 0


def fused_ar_add_rmsnorm(
    inp: torch.Tensor,
    residual: torch.Tensor,
    layernorm: torch.nn.Module,
    group: "GroupCoordinator",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused decode/verify {all-reduce -> residual-add + RMSNorm}: one kernel
    replacing {custom AR + fused_add_rmsnorm}. ``inp`` holds the UNREDUCED
    partial sums; returns ``(hidden_states, residual)`` exactly like the
    unfused ``layernorm(all_reduce(inp), residual)`` chain. The caller must
    have checked ``fused_ar_rmsnorm_eligible``. This IS a v5 AR with the
    epilogue seam filled in; the in-kernel entry barrier fences staging
    reuse (no rotation)."""
    comm = group.torch_symm_mem_comm
    res = comm.custom_ar_resources
    hs_out = torch.empty_like(inp)
    residual_out = torch.empty_like(residual)
    stage_off = res.v5_in[0]
    esz = comm.buffer.element_size()
    _ar_norm_jit().inkling_ar_add_rmsnorm(
        inp,
        residual,
        residual_out,
        hs_out,
        layernorm.weight,
        layernorm.variance_epsilon,
        res.multicast_ptr + stage_off * esz,
        comm.buffer.data_ptr() + stage_off * esz,
        res.flag_ptrs_dev,
        res.state_ptr,
        res.rank,
        res.world,
    )
    return hs_out, residual_out
