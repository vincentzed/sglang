"""Tests for the symm-mem custom all-reduce family + fused {AR -> add+RMSNorm}.

Covers:
  * correctness of every AR variant (v2/v3/v3b/v4/v5/mm) against NCCL,
  * BIT-identity of the fused {AR -> add + RMSNorm} kernel against the
    unfused {v5 custom AR -> sgl_kernel.fused_add_rmsnorm} chain (including
    calibration of the ROUND_SUM_TO_BF16 rounding knob),
  * CUDA-graph capture + replay (barrier epochs must advance across replays;
    the v5 A/B staging rotation must stay aligned).

Usage::

    python test/registered/jit/test_symm_mem_custom_ar.py --num-gpu 4
    python test/registered/jit/test_symm_mem_custom_ar.py --num-gpu 4,8
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as torch_symm_mem

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.tests.utils import multigpu_pytest_main
from sglang.jit_kernel.utils import cache_once, get_ci_test_range
from sglang.srt.distributed.device_communicators.symm_mem_custom_ar import (
    CUSTOM_AR_BUFFER_BYTES,
    build_custom_ar_resources,
    custom_all_reduce,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="extra-b", runner_config="8-gpu-h200")

TEST_NUM_TOKENS = [1, 2, 3, 8, 64, 96, 160, 256, 2048, 16384]
TEST_NUM_TOKENS = get_ci_test_range(TEST_NUM_TOKENS, [2, 96, 2048])
HIDDEN = 6144
FUSED_TOKENS = [1, 2, 8, 96, 160]
FUSED_TOKENS = get_ci_test_range(FUSED_TOKENS, [2, 96])


@cache_once
def _setup_once():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)

    group_name = dist.group.WORLD.group_name
    buffer = torch_symm_mem.empty(
        CUSTOM_AR_BUFFER_BYTES // 2, device=device, dtype=torch.bfloat16
    )
    handle = torch_symm_mem.rendezvous(buffer, group_name)
    if handle.multicast_ptr == 0:
        return None
    res = build_custom_ar_resources(
        buffer=buffer, handle=handle, group_name=group_name, world_size=world_size
    )
    from sglang.jit_kernel.inkling_ar_norm import compile_inkling_ar_norm

    compile_inkling_ar_norm(torch.bfloat16, world_size)
    return buffer, handle, res, group_name, device, world_size, local_rank


def _env():
    setup = _setup_once()
    if setup is None:
        pytest.skip("multimem multicast unavailable on this topology")
    return setup


def _dropin_ar(inp, buffer, res, group_name):
    return custom_all_reduce(inp, buffer=buffer, group_name=group_name, res=res)


def _fused_ar_norm(inp, residual, weight, eps, buffer, res, round_sum):
    from sglang.jit_kernel.inkling_ar_norm import inkling_ar_add_rmsnorm

    hs_out = torch.empty_like(inp)
    residual_out = torch.empty_like(residual)
    cur = res.v5_cur
    stage_off = res.v5_in[cur]
    esz = buffer.element_size()
    inkling_ar_add_rmsnorm(
        inp,
        residual,
        residual_out,
        hs_out,
        weight,
        eps,
        res.multicast_ptr + stage_off * esz,
        buffer.data_ptr() + stage_off * esz,
        res.flag_ptrs_dev,
        res.state_ptr,
        res.rank,
        res.world,
        round_sum=round_sum,
    )
    res.v5_cur = 1 - cur
    return hs_out, residual_out


@pytest.mark.parametrize("num_tokens", TEST_NUM_TOKENS)
def test_dropin_correctness(num_tokens: int):
    """The dispatching drop-in AR matches NCCL within bf16 reduction slop."""
    buffer, _, res, group_name, device, world, rank = _env()
    torch.manual_seed(100 + num_tokens)
    base = torch.randn(world, num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    inp = base[rank].contiguous()
    ref = inp.clone()
    dist.all_reduce(ref)
    out = _dropin_ar(inp, buffer, res, group_name).clone()
    torch.cuda.synchronize()
    dist.barrier()
    # All custom paths accumulate in fp32 (in-switch via .acc::f32 or locally),
    # like NCCL; only the final bf16 rounding may differ by reduction order.
    torch.testing.assert_close(out.float(), ref.float(), atol=0.15, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", FUSED_TOKENS)
def test_fused_ar_norm_bit_identity(num_tokens: int):
    """Fused {AR -> add+RMSNorm} is BIT-identical to the unfused chain
    {v5 custom AR -> sgl_kernel.fused_add_rmsnorm}."""
    from sgl_kernel import fused_add_rmsnorm

    from sglang.jit_kernel.inkling_ar_norm import ROUND_SUM_TO_BF16

    buffer, _, res, group_name, device, world, rank = _env()
    torch.manual_seed(4200 + num_tokens)
    eps = 1e-6
    base = torch.randn(world, num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    inp = base[rank].contiguous()
    residual = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    dist.broadcast(residual, src=0)
    weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    dist.broadcast(weight, src=0)

    # Unfused reference: v5 drop-in AR (same transport as the fused kernel),
    # then the production fused_add_rmsnorm (in-place on x and residual).
    x_ref = _dropin_ar(inp, buffer, res, group_name).clone()
    torch.cuda.synchronize()
    dist.barrier()
    res_ref = residual.clone()
    fused_add_rmsnorm(x_ref, res_ref, weight, eps)

    hs, res_out = _fused_ar_norm(
        inp, residual, weight, eps, buffer, res, round_sum=ROUND_SUM_TO_BF16
    )
    torch.cuda.synchronize()
    dist.barrier()

    if not (torch.equal(hs, x_ref) and torch.equal(res_out, res_ref)):
        # Calibration probe: does the OTHER rounding mode match instead?
        hs2, res_out2 = _fused_ar_norm(
            inp, residual, weight, eps, buffer, res, round_sum=not ROUND_SUM_TO_BF16
        )
        torch.cuda.synchronize()
        dist.barrier()
        other = torch.equal(hs2, x_ref) and torch.equal(res_out2, res_ref)
        raise AssertionError(
            f"fused != unfused at ROUND_SUM_TO_BF16={ROUND_SUM_TO_BF16}; "
            f"opposite mode matches: {other} "
            f"(if True, flip ROUND_SUM_TO_BF16 in jit_kernel/inkling_ar_norm.py). "
            f"max |hs diff|={(hs.float() - x_ref.float()).abs().max().item():.3e}"
        )


@pytest.mark.parametrize("num_tokens", get_ci_test_range([8, 96], [8]))
def test_graph_capture_replay(num_tokens: int):
    """Fused + drop-in calls survive CUDA-graph capture and N replays
    (barrier epochs advance; A/B staging rotation stays aligned)."""
    from sgl_kernel import fused_add_rmsnorm

    from sglang.jit_kernel.inkling_ar_norm import ROUND_SUM_TO_BF16

    buffer, _, res, group_name, device, world, rank = _env()
    torch.manual_seed(7000 + num_tokens)
    eps = 1e-6
    inp = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    dist.broadcast(residual, src=0)
    weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    dist.broadcast(weight, src=0)
    hs_g = torch.empty_like(inp)
    res_g = torch.empty_like(residual)

    def step():
        # An EVEN number of rotated (v5) calls per iteration, like a
        # transformer layer's two TP seams.
        h, r = _fused_ar_norm(
            inp, residual, weight, eps, buffer, res, round_sum=ROUND_SUM_TO_BF16
        )
        h2 = _dropin_ar(h, buffer, res, group_name)
        hs_g.copy_(h2)
        res_g.copy_(r)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    torch.cuda.synchronize()
    dist.barrier()

    # Eager reference for the same inputs.
    x_ref = inp.clone()
    dist.all_reduce(x_ref)
    res_ref = residual.clone()
    fused_add_rmsnorm(x_ref, res_ref, weight, eps)
    hs_ref = x_ref.clone()
    dist.all_reduce(hs_ref)

    for i in range(10):
        g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(
            hs_g.float(), hs_ref.float(), atol=0.15, rtol=2e-2, msg=f"replay {i}"
        )
        torch.testing.assert_close(res_g.float(), res_ref.float(), atol=0.05, rtol=2e-2)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
